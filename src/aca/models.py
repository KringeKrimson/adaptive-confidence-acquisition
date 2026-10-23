"""Sequential batch-one inference with synchronized, incremental timings."""
import gc
import time
import torch
from .ic import aggregate_ic, ic_messages, project_p_yes, yes_no_ids

SMALL = 'microsoft/Phi-3-mini-4k-instruct'
SMALL_MODELS = (SMALL, 'Qwen/Qwen2.5-1.5B-Instruct')
STRONG = 'Qwen/Qwen2.5-7B-Instruct'
STRONG_MODELS = (STRONG, 'Qwen/Qwen2.5-14B-Instruct')


def qa_messages(question):
    return [dict(role='system', content='Answer the question with only the short entity answer. Do not explain.'),
            dict(role='user', content=f'Question: {question}')]


def synchronize():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def load_model(name, revision, quantization='4bit-fp32'):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    if not torch.cuda.is_available():
        raise RuntimeError('A CUDA GPU is required for the research run. Local tests use tiny random models only.')
    dtype = torch.float32 if quantization == '4bit-fp32' else torch.float16
    options = dict(revision=revision, token=False, torch_dtype=dtype,
                   device_map={'': 0}, attn_implementation='eager', trust_remote_code=False)
    if quantization in {'4bit', '4bit-fp32'}:
        options['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True)
    tokenizer = AutoTokenizer.from_pretrained(name, revision=revision, token=False, trust_remote_code=False)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(name, **options).eval()
    return model, tokenizer


def tokenize(model, tokenizer, messages, reserve=0):
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, add_special_tokens=False, return_tensors='pt').to(model.device)
    limit = min(int(model.config.max_position_embeddings), 4096)
    if inputs.input_ids.shape[1] + reserve > limit:
        raise ValueError('Prompt exceeds context limit; refusing silent truncation')
    return inputs


class FiniteLogits:
    """Reject numerical failures before argmax can silently turn NaN into token 0."""
    def __call__(self, input_ids, scores):
        if not torch.isfinite(scores).all().item():
            raise FloatingPointError('Non-finite generation logits; refuse invalid predictions')
        return scores


class FirstAnswerLine:
    """Batch-one stop after a nonempty answer line, including leading blank lines."""
    def __init__(self, tokenizer, prompt_length):
        self.tokenizer, self.prompt_length = tokenizer, prompt_length

    def __call__(self, input_ids, scores, **kwargs):
        text = self.tokenizer.decode(input_ids[0, self.prompt_length:], skip_special_tokens=True).lstrip()
        return '\n' in text or '\r' in text


def generate_answer(model, tokenizer, question, max_new_tokens=32, capture_features=False, answer_stop='eos'):
    from transformers import LogitsProcessorList, StoppingCriteriaList
    if answer_stop not in {'eos','first-line'}:
        raise ValueError('Unknown answer stopping rule')
    captured = []
    # The first norm invocation is the ordinary QA prefill. No extra model pass.
    def hook(_module, _args, output):
        if not captured:
            captured.append(output[0, -1, :].detach().clone())
    handle = model.model.norm.register_forward_hook(hook) if capture_features else None
    synchronize()
    start = time.perf_counter()
    try:
        with torch.inference_mode():
            inputs = tokenize(model, tokenizer, qa_messages(question), reserve=max_new_tokens)
            stopping = (StoppingCriteriaList([FirstAnswerLine(tokenizer, inputs.input_ids.shape[1])])
                        if answer_stop == 'first-line' else None)
            generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                       use_cache=True, pad_token_id=tokenizer.pad_token_id,
                                       stopping_criteria=stopping,
                                       logits_processor=LogitsProcessorList([FiniteLogits()]))
            answer_ids = generated[0, inputs.input_ids.shape[1]:]
            raw_prediction = tokenizer.decode(answer_ids, skip_special_tokens=True).strip()
            prediction = raw_prediction.splitlines()[0].strip() if answer_stop=='first-line' and raw_prediction else raw_prediction
            feature = captured[0].float().cpu().tolist() if capture_features else None
        synchronize()
        elapsed = time.perf_counter() - start
    finally:
        if handle is not None:
            handle.remove()
    result = dict(prediction=prediction, features=feature, elapsed_seconds=elapsed,
                input_tokens=int(inputs.input_ids.shape[1]), output_tokens=len(answer_ids),
                model_calls=1, forward_passes=len(answer_ids),
                hit_token_limit=bool(len(answer_ids) == max_new_tokens))
    if answer_stop != 'eos':
        result.update(raw_prediction=raw_prediction,answer_stop=answer_stop)
    return result


def estimate_ic(model, tokenizer, question, last_k=3, alpha=1.0):
    ids = yes_no_ids(tokenizer)
    synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        inputs = tokenize(model, tokenizer, ic_messages(question))
        if inputs.input_ids.shape[1] < last_k:
            raise ValueError('IC token window exceeds prompt')
        output = model.model(**inputs, output_hidden_states=True, use_cache=False)
        matrix = project_p_yes(output.hidden_states, model.lm_head, ids, last_k)
        if not torch.isfinite(torch.as_tensor(matrix)).all().item():
            raise FloatingPointError('Non-finite IC probabilities')
        score = aggregate_ic(matrix, alpha)
    synchronize()
    elapsed = time.perf_counter() - start
    return dict(ic=score, p_yes=matrix.tolist(), yes_no_ids=ids,
                last_k=last_k, alpha=alpha, elapsed_seconds=elapsed,
                input_tokens=int(inputs.input_ids.shape[1]), output_tokens=0,
                model_calls=1, forward_passes=1)


def verify_ic_projection(model, tokenizer, last_k):
    """One setup-only forward: selected-head optimization versus full vocabulary."""
    import numpy as np
    ids=yes_no_ids(tokenizer)
    synchronize();start=time.perf_counter()
    with torch.inference_mode():
        inputs=tokenize(model,tokenizer,ic_messages('What is the capital of France?'))
        states=model.model(**inputs,output_hidden_states=True,use_cache=False).hidden_states
        reduced=project_p_yes(states,model.lm_head,ids,last_k)
        full=[]
        for state in states:
            logits=model.lm_head(state[:,-last_k:,:]).float()
            assert torch.isfinite(logits).all().item()
            full.append(logits[0,:,ids].softmax(-1)[:,0].cpu().numpy())
        full=np.stack(full,axis=-1)
        error=float(np.max(np.abs(full-reduced)))
        if not np.allclose(full,reduced,atol=2e-5,rtol=2e-5):
            raise ValueError(f'Full versus reduced IC projection mismatch: {error}')
    synchronize()
    return dict(max_probability_error=error,shape=list(reduced.shape),yes_no_ids=ids,
                elapsed_seconds=time.perf_counter()-start,model_calls=1,scope='setup-only, not a per-query service cost')


def release_model(model):
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
