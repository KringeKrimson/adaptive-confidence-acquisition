"""Recreate paper tables/plots from saved aggregate results, without fitting or inference.

This is not per-query/bootstrap replication. Saved intervals are copied unchanged.
All outputs go to a separate directory; frozen results are read-only.
"""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
BLUE, RUST, GREEN = '#176b91', '#b95732', '#398455'


def read(name):
    return json.loads((REPO / 'results' / f'{name}.json').read_text())


def save(fig, directory, name):
    for suffix in ['pdf', 'png']:
        fig.savefig(directory / f'{name}.{suffix}', dpi=300)
    plt.close(fig)


def reproduce(out):
    out = Path(out).resolve()
    if out == REPO or out == REPO / 'results' or REPO / 'results' in out.parents:
        raise ValueError('Use a separate output directory, never results/')
    out.mkdir(parents=True, exist_ok=True)
    figures = out / 'figures'
    figures.mkdir(exist_ok=True)
    original = read('qwen15_confirmatory')
    mixture = read('qwen15_exploratory_mixture')
    benefit = read('benefit_development')
    audit = read('pregeneration_audit')
    diag = read('pregeneration_diagnostics')
    rows = {r['id']: r for r in original['results']}
    # Validate scope and accounting; do not recalculate any inferential statistic.
    assert original['n_test'] == 512 and not original['meaningful_support']
    assert original['three_point_gain_ruled_out_for_tested_eligible_points']
    assert len(original['comparisons']) == 8
    assert benefit['status'] == 'STOP' and len(benefit['points']) == 16
    assert not benefit['fresh_test_inference_authorized']
    assert audit['n_calibration'] == audit['n_assessment'] == 128
    assert audit['no_new_gpu_inference'] and not audit['reserved_test_records_read']
    fixed = diag['illustrative_unselected_threshold_05']['weighted']
    example = fixed['assessment']
    assert np.isclose(example['answer_first_seconds'] - example['mean_seconds'], example['saving_seconds'])
    assert np.isclose(example['saving_seconds'] / example['answer_first_seconds'], example['saving_fraction'])
    assert fixed['calibration']['random']['gain'] < 0 < example['random']['gain']
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False,
                         'axes.spines.right': False, 'pdf.fonttype': 42,
                         'legend.fontsize': 7.5})

    # Both panels retain every frozen point, including duplicates.
    fig, axes = plt.subplots(1, 2, figsize=(6.35, 2.65), layout='constrained',
                             gridspec_kw={'width_ratios': [1.05, 1]})
    styles = [('cheap-only', GREEN, 's', 'Cheap only'),
              ('always-ic', '#8c6b29', 'v', 'Always IC'),
              ('adaptive', BLUE, 'o', 'Adaptive IC'),
              ('random-ic-budget-matched', '#8e769f', '^', 'Random IC'),
              ('cheap-cost-matched', RUST, 'x', 'Matched cheap')]
    ax = axes[0]
    for method, color, marker, label in styles:
        points = [r for r in original['results'] if r['method'] == method]
        ax.scatter([r['mean_seconds'] for r in points], [100*r['accuracy'] for r in points],
                   c=color, marker=marker, s=18, label=label, alpha=.85)
    for id_, marker, label in [('small-only', '*', 'Small'), ('strong-only', '*', 'Strong')]:
        r = rows[id_]
        ax.scatter(r['mean_seconds'], 100*r['accuracy'], c='black', marker=marker, s=43)
        ax.annotate(label, (r['mean_seconds'], 100*r['accuracy']), xytext=(4, 7) if id_=='small-only' else (3, -12),
                    textcoords='offset points', fontsize=8)
    ax.set(xlabel='Warm seconds / query', ylabel='Exact match (%)', title='All frozen policies')
    ax.legend(loc='lower right', frameon=False)
    ax = axes[1]
    for i, c in enumerate(original['comparisons']):
        r = c['primary']; lo, hi = np.asarray(r['familywise_95'])*100
        ax.plot([lo, hi], [i, i], color=BLUE, lw=1.2)
        ax.scatter(100*r['gain'], i, color=BLUE, s=15)
    ax.axvline(0, color='#777777', lw=.7)
    ax.axvline(3, color=GREEN, ls='--', lw=1)
    ax.set(yticks=range(8), yticklabels=[f'{c["penalty"]:g}' for c in original['comparisons']],
           xlabel='Adaptive − matched cheap (pp)', ylabel='Frozen penalty λ',
           title='Familywise 95% intervals', xlim=(-4.8, 3.6))
    ax.invert_yaxis()
    save(fig, figures, 'paper_confirmatory')

    fig, ax = plt.subplots(figsize=(6.35, 2.7), layout='constrained')
    ax.plot([r['mean_seconds'] for r in mixture['curve']],
            [100*r['accuracy'] for r in mixture['curve']], c='black', lw=1.5,
            label='Independent model preselection (expected)')
    for method, color, marker, label in styles:
        points = [r for r in original['results'] if r['method'] == method]
        ax.scatter([r['mean_seconds'] for r in points], [100*r['accuracy'] for r in points],
                   c=color, marker=marker, s=22, label=label, alpha=.85)
    for id_, label, delta in [('small-only', 'Small only', (5, 7)),
                               ('strong-only', 'Strong only', (-53, 8))]:
        r = rows[id_]; ax.scatter(r['mean_seconds'], 100*r['accuracy'], c='black', marker='*', s=55)
        ax.annotate(label, (r['mean_seconds'], 100*r['accuracy']), xytext=delta,
                    textcoords='offset points', fontsize=8,
                    bbox={'facecolor':'white','edgecolor':'none','pad':.6})
    ax.set(xlabel='Warm component seconds / query', ylabel='Exact match (%)',
           title='Original test: post-test exploratory comparison', ylim=(23, 66))
    ax.legend(loc='lower right', frameon=False, ncols=2)
    save(fig, figures, 'paper_preselection')

    fig, axes = plt.subplots(1, 2, figsize=(6.35, 2.7), layout='constrained')
    for ax, part in zip(axes, ['calibration', 'assessment']):
        component = audit['components'][part]
        low = diag['grid'][part]['weighted'][0]['accuracy']
        # Endpoints come from saved selected direct-model policies, not fitted labels.
        selected = audit['families']['ic-first-weighted']['points']
        high = max(r[part]['accuracy'] for r in selected)
        p = np.linspace(0, 1, 101)
        ax.plot(component['small_seconds'] + p*(component['strong_seconds']-component['small_seconds']),
                100*(low+p*(high-low)), c='black', lw=1.2, label='Random preselection')
        for signal, color, label in [('weighted', BLUE, 'Weighted IC'),
                                     ('top_right', RUST, 'Top-right P(Yes)'),
                                     ('unweighted', GREEN, 'Unweighted mean')]:
            points = diag['grid'][part][signal]
            ax.plot([r['mean_seconds'] for r in points], [100*r['accuracy'] for r in points],
                    color=color, lw=1, label=label)
        ax.scatter([component['small_seconds'], component['strong_seconds']],
                   [100*low, 100*high], c='black', marker='*', s=43, zorder=6)
        ex = fixed[part]
        ax.scatter(ex['mean_seconds'], 100*ex['accuracy'], facecolor='white',
                   edgecolor=BLUE, s=29, zorder=7)
        ax.set(title=f'{part.title()} (n=128)', xlabel='Warm seconds / query',
               ylabel='Exact match (%)', ylim=(24, 65), xlim=(.25, 1.95))
        if part == 'assessment':
            ax.plot([ex['mean_seconds'], ex['answer_first_seconds']],
                    [100*ex['accuracy']]*2, c=BLUE, lw=2)
            ax.annotate('Same .5 cut:\n0.283 s saved',
                        (ex['answer_first_seconds'], 100*ex['accuracy']),
                        xytext=(1.0, 41), fontsize=7.5,
                        arrowprops={'arrowstyle': '-', 'color': '#777777'})
        ax.legend(loc='lower right', frameon=False, fontsize=7)
    save(fig, figures, 'paper_pregeneration')

    lines = ['# Paper tables reproduced from saved aggregates', '',
             'No fitting, inference, threshold search, bootstrap rerun or private cache access.', '',
             '## Original confirmatory comparison', '',
             '| λ | Adaptive EM (%) | Matched cheap EM (%) | s/query | Gain (pp) | Familywise 95% (pp) |',
             '|---|---:|---:|---:|---:|---|']
    for i, c in enumerate(original['comparisons']):
        a, b = rows[f'p{i}:adaptive'], rows[f'p{i}:cheap-cost-matched']
        lo, hi = np.array(c['primary']['familywise_95'])*100
        lines.append(f'| {c["penalty"]:g} | {100*a["accuracy"]:.2f} | {100*b["accuracy"]:.2f} | {a["mean_seconds"]:.4f} | {100*c["primary"]["gain"]:+.2f} | [{lo:+.2f}, {hi:+.2f}] |')
    lines += ['', '## Fixed benefit development grid', '',
              '| Method | λ | Cal. gain (pp) | EM (%) | s/query | Strong (%) | Exact match | Gain (pp), familywise 95% |',
              '|---|---:|---:|---:|---:|---:|---|---|']
    for pt in benefit['points']:
        a=pt['assessment'];m=a['random_preselection'];lo,hi=np.array(a['familywise_95'])*100
        lines.append(f'| {pt["kind"]} | {pt["penalty"]:g} | {100*pt["calibration"]["random_preselection"]["gain"]:+.3f} | {100*a["accuracy"]:.2f} | {a["mean_seconds"]:.4f} | {100*a["strong_rate"]:.2f} | {m["exact_cost_match"]} | {100*m["gain"]:+.3f} [{lo:+.3f}, {hi:+.3f}] |')
    lines += ['', '## IC-first retrospective .5 illustration (assessment)', '',
              f'EM: {100*example["accuracy"]:.2f}%; answer-first {example["answer_first_seconds"]:.4f} s; IC-first {example["mean_seconds"]:.4f} s.',
              f'Saved {example["saving_seconds"]:.4f} s ({100*example["saving_fraction"]:.1f}%); random EM {100*example["random"]["accuracy"]:.2f}%.',
              f'Gain {100*example["random"]["gain"]:.2f} pp; pointwise 95% {np.array(example["pointwise_95"])*100}.',
              f'Calibration gain {100*fixed["calibration"]["random"]["gain"]:.2f} pp; not selected or confirmatory.', '']
    (out/'tables.md').write_text('\n'.join(lines))
    inputs = {f'results/{name}.json': hashlib.sha256((REPO/'results'/f'{name}.json').read_bytes()).hexdigest()
              for name in ['qwen15_confirmatory','qwen15_exploratory_mixture','benefit_development',
                           'pregeneration_audit','pregeneration_diagnostics']}
    (out/'provenance.json').write_text(json.dumps(dict(inputs=inputs, new_model_calls=0,
        fitting=False, thresholds_modified=False, intervals='Copied from saved evaluations'), indent=2)+'\n')
    print(f'Wrote tables and three paper figure pairs to {out}; saved evaluations unchanged.')
    return out


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default='outputs/paper-reproduction')
    reproduce(parser.parse_args().out)
