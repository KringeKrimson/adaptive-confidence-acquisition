"""Second prospective development screen; original PROTOCOL is never mutated."""
from copy import deepcopy
from pathlib import Path
import numpy as np
from .storage import atomic_json, bind_manifest, fingerprint, load_records, read_json
from .viability import PROTOCOL as ORIGINAL, freeze_protocol

PROTOCOL = deepcopy(ORIGINAL)
PROTOCOL.update(version=2, name='qwen15_to_qwen14_prospective',
    model_roles=dict(small='Qwen/Qwen2.5-1.5B-Instruct',strong='Qwen/Qwen2.5-14B-Instruct'),
    model_revisions={'Qwen/Qwen2.5-1.5B-Instruct':'989aa7980e4cf806f80c7fef2b1adb7bc71aa306',
                     'Qwen/Qwen2.5-14B-Instruct':'cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8'},
    sampling_seed=4902, max_new_tokens=64, answer_stop='first-line', ic_last_k=6,ic_alpha=1.,
    qa_prompt='Same zero-shot short entity QA prompt as original. Greedy, stop on first nonempty answer line or EOS, hard cap 64 tokens, strict alias EM of that line; no answer-aware extraction.',
    ic_transfer='Authors 5c15b32 Qwen-14B uses final 6 prompt tokens, alpha=1; verified identical Qwen chat suffix and single-token Yes/No [9454,2753]. Qwen-1.5B is a model-transfer experiment, not a published-model reproduction. No demonstrations, embedding + intermediate + final normalized states, no extra intermediate RMS normalization.',
    features='New 1536-dimensional final normalized hidden state at final ordinary QA prompt token during generation prefill. Newly fit train-only probe, no Phi weights/features reused.',
    data='Fresh official-train-only 1024/256 sample, seed4902; exclude original 1024/256 development cohort and reserved32 pilot by ID and normalized question. No official-validation/final-test inference or test queries in this manifest.',
    smoke='First eight new training queries, separate cache. Require finite 1536-d features, six-by-29 IC matrices, correct token IDs, positive times, numerical projection check, no OOM, >256MiB reserved-memory headroom. No accuracy-based stopping/precision selection. If smoke succeeds, automatically collect full1024/256 with unchanged settings.',
    precision='Both models NF4 double-quantized weights, FP32 compute and nonquantized modules, eager attention, batch1, sequential GPU0 on T4x2. No mixed-setting comparisons. Include actual stopping/finite-check costs; loads/warmups/smoke separate.',
    followup='If PASS, freeze adaptive and baseline candidate grids and validation-only selection before any test inference. This runner never invokes final-test collection.')


def prepare_screen(root):
    """Recreate prior development IDs for exclusion without reading model outcomes."""
    from .data import prepare, validate_splits, question_key
    root=Path(root)
    inputs=root.parent/'prospective_inputs'
    pilot=prepare(inputs/'pilot',test=32,revision=PROTOCOL['dataset_revision'])
    old=prepare(inputs/'old_development',train=1024,validation=256,test=0,
                revision=PROTOCOL['dataset_revision'],exclude_manifest=inputs/'pilot'/'queries.json')
    excluded=dict(revision=PROTOCOL['dataset_revision'],queries=pilot['queries']+old['queries'])
    bind_manifest(inputs,'excluded.json',excluded)
    manifest=prepare(root,train=1024,validation=256,test=0,seed=PROTOCOL['sampling_seed'],
                     revision=PROTOCOL['dataset_revision'],exclude_manifest=inputs/'excluded.json')
    validate_splits(manifest['queries'])
    assert not {q['id'] for q in manifest['queries']}&{q['id'] for q in excluded['queries']}
    assert not {question_key(q['question']) for q in manifest['queries']}&{question_key(q['question']) for q in excluded['queries']}
    assert set(q['source_split'] for q in manifest['queries'])=={'train'}
    frozen=freeze_protocol(root,PROTOCOL)
    print('Prospective protocol hash:',fingerprint(frozen),flush=True)
    return manifest


def tokenizer_preflight(root):
    from transformers import AutoTokenizer,AutoConfig
    from .ic import ic_messages,yes_no_ids
    from .models import qa_messages
    name=PROTOCOL['model_roles']['small'];rev=PROTOCOL['model_revisions'][name]
    tok=AutoTokenizer.from_pretrained(name,revision=rev,token=False)
    cfg=AutoConfig.from_pretrained(name,revision=rev,token=False)
    assert cfg.hidden_size==1536 and cfg.num_hidden_layers==28
    assert yes_no_ids(tok)==[9454,2753]
    max_lengths={'qa':0,'ic':0};suffix=None
    for q in read_json(Path(root)/'queries.json')['queries']:
        for key,messages in [('qa',qa_messages(q['question'])),('ic',ic_messages(q['question']))]:
            ids=tok.encode(tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True),add_special_tokens=False)
            max_lengths[key]=max(max_lengths[key],len(ids))
            assert len(ids)+(64 if key=='qa' else 0)<=4096
            if key=='ic':
                if suffix is None: suffix=ids[-6:]
                assert suffix==ids[-6:]
    expected=[29,151645,198,151644,77091,198]
    assert suffix==expected
    result=dict(model=name,revision=rev,hidden_size=1536,hidden_states=29,yes_no_ids=yes_no_ids(tok),
                last_k=6,suffix_ids=suffix,suffix_tokens=tok.convert_ids_to_tokens(suffix),max_lengths=max_lengths)
    bind_manifest(root,'tokenizer_preflight.json',result)
    return result


def smoke_check(root):
    """Smoke has no accuracy gate and cannot tune precision or screening rules."""
    from .ic import aggregate_ic
    root=Path(root);queries=read_json(root/'queries.json')['queries']
    result={}
    for stage in ['small','strong','ic']:
        rows=load_records(root,stage,queries)
        for row in rows:
            assert np.isfinite(row['elapsed_seconds']) and row['elapsed_seconds']>0
            if stage=='small':
                assert np.asarray(row['features']).shape==(1536,)
                assert np.isfinite(row['features']).all()
            if stage=='ic':
                assert np.asarray(row['p_yes']).shape==(6,29) and row['yes_no_ids']==[9454,2753]
                assert np.isclose(row['ic'],aggregate_ic(row['p_yes']),atol=1e-12)
        result[stage]=dict(n=len(rows),mean_seconds=float(np.mean([r['elapsed_seconds'] for r in rows])))
    memory=[read_json(p) for p in (root/'overhead').glob('*-memory-*.json')]
    assert {row['model'] for row in memory}==set(PROTOCOL['model_roles'].values())
    for row in memory:
        assert row['total_bytes']-row['peak_reserved_bytes']>256*1024**2
    result.update(passed=True,memory=memory,accuracy_gate=False)
    atomic_json(root/'smoke_check.json',result)
    print('Automatic smoke PASS:',result,flush=True)
    return result


def run(root):
    from .collect import collect
    from .analysis import viability
    root=Path(root);manifest=prepare_screen(root)
    tokenizer_preflight(root)
    smoke=root/'smoke'
    first=[q for q in manifest['queries'] if q['split']=='train'][:8]
    bind_manifest(smoke,'queries.json',dict(revision=manifest['revision'],sampling='First8 training queries; engineering smoke only',queries=first))
    bind_manifest(smoke,'model_revisions.json',PROTOCOL['model_revisions'])
    options=dict(small_model=PROTOCOL['model_roles']['small'],strong_model=PROTOCOL['model_roles']['strong'],
                 max_new_tokens=64,answer_stop='first-line',quantization='4bit-fp32')
    collect(smoke,stages=['small','ic','strong'],splits=['train'],**options)
    smoke_check(smoke)
    # Projection identity on actual GPU model is verified during IC warmup.
    collect(root,stages=['small'],splits=['train','validation'],**options)
    collect(root,stages=['ic','strong'],splits=['validation'],**options)
    report=viability(root,protocol=PROTOCOL)
    assert not any(q['split']=='test' for q in manifest['queries'])
    print('PROSPECTIVE SCREEN:',{k:v for k,v in report.items() if k not in ['train_ids','validation_ids','calibration_ids','assessment_ids']},flush=True)
    return report
