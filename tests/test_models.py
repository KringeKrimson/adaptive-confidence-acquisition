"""Real forward/generate integration tests on tiny random Phi-3 and Llama.

These verify mechanics only and must never be reported as research results.
"""
import numpy as np
import pytest
import torch
from aca.ic import project_p_yes
from aca.models import generate_answer


@pytest.mark.parametrize('architecture',['phi3','llama'])
def test_prefill_features_and_ic_projection(architecture):
    from transformers import Phi3Config, Phi3ForCausalLM, LlamaConfig, LlamaForCausalLM
    torch.manual_seed(1)
    options=dict(vocab_size=64,hidden_size=32,intermediate_size=64,num_hidden_layers=2,
                 num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=512,
                 bos_token_id=1,eos_token_id=2,pad_token_id=0)
    if architecture=='phi3':
        model=Phi3ForCausalLM(Phi3Config(**options,original_max_position_embeddings=512)).eval()
    else:
        model=LlamaForCausalLM(LlamaConfig(**options)).eval()
    inputs=torch.tensor([[1,4,5,6,7]])
    with torch.inference_mode():
        expected=model.model(inputs,output_hidden_states=True)
        p=project_p_yes(expected.hidden_states,model.lm_head,[8,9],3)
        full=torch.stack([model.lm_head(s[:,-3:,:])[0,:,[8,9]].float().softmax(-1)[:,0]
                          for s in expected.hidden_states],dim=-1).numpy()
    assert np.allclose(p,full,atol=1e-6)
    # A tiny tokenizer stub exercises the same generate + norm-hook path.
    class Tokenizer:
        pad_token_id=0
        def apply_chat_template(self,*a,**kw): return 'prompt'
        def __call__(self,*a,**kw):
            from transformers import BatchEncoding
            return BatchEncoding(dict(input_ids=inputs,attention_mask=torch.ones_like(inputs)))
        def decode(self,*a,**kw): return 'answer'
    record=generate_answer(model,Tokenizer(),'question',3,True)
    assert len(record['features'])==32
    assert np.allclose(record['features'],expected.last_hidden_state[0,-1].numpy(),atol=1e-5)
    assert record['elapsed_seconds']>0
    assert record['output_tokens']>=1
