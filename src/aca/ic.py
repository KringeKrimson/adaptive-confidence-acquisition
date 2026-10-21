"""Chen et al. IC, aligned to authors' code at 5c15b32 (see README)."""
import numpy as np

AUTHOR_COMMIT = '5c15b32a6f9f36688098d0a749cb68bad11e9f23'


def positional_weights(length, alpha=1.0):
    if length < 1 or not np.isfinite(alpha) or alpha <= 0:
        raise ValueError('Positive length and locality required')
    weights = np.exp(-alpha * (np.arange(length) - (length - 1)) ** 2)
    return weights / weights.sum()


def aggregate_ic(probabilities, alpha=1.0):
    p = np.asarray(probabilities, dtype=np.float64)
    if p.ndim != 2 or not p.size or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError('Expected finite token-by-layer probabilities in [0,1]')
    return float(positional_weights(p.shape[0], alpha) @ p @ positional_weights(p.shape[1], alpha))


def ic_messages(question):
    # Exact create_prompt default (no demonstrations) from authors' released API.
    return [dict(role='system', content=(
        'You are a helpful assistant that assesses whether you can '
        'provide an accurate response to a question. Respond only '
        "with 'Yes' or 'No' to indicate whether you are capable of answering "
        'the following question.')),
        dict(role='user', content=f'<Question>: {question} </Question>')]


def yes_no_ids(tokenizer):
    ids = [tokenizer.encode(word, add_special_tokens=False) for word in ['Yes', 'No']]
    if any(len(x) != 1 for x in ids) or ids[0] == ids[1]:
        raise ValueError(f'IC requires distinct single-token Yes and No: {ids}')
    return [x[0] for x in ids]


def project_p_yes(hidden_states, head, ids, last_k):
    """Select head rows instead of allocating vocabulary logits; same binary softmax.

    Match released code: include embedding state and final normalized state;
    do not add extra RMS normalization to intermediate states.
    """
    import torch
    import torch.nn.functional as F
    rows = head.weight[ids, :]
    bias = head.bias[ids] if getattr(head, 'bias', None) is not None else None
    result = []
    for state in hidden_states:
        logits = F.linear(state[:, -last_k:, :].to(rows.dtype), rows, bias)
        result.append(torch.softmax(logits.float(), dim=-1)[0, :, 0])
    return torch.stack(result, dim=-1).cpu().numpy()
