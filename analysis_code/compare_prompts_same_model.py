#!/usr/bin/env python3
"""Compare prompt versions against each other, holding the model fixed.

Reads the pipeline output folders produced by run_pipeline.py, named
    v<N>_<provider>_<model>/          e.g. v3_openai_gpt4o/
each containing runs.json (+ stats.json, selected.json, mode_analysis.json).

For every model it builds one table with one row per prompt version, says which
prompt works best for that model, and quantifies how sensitive the model is to
the prompt wording. It then aggregates across models to show whether a prompt is
consistently strong or only wins for some of them.

Per-run metrics are recomputed from runs.json rather than read from stats.json,
so the ground truth can be changed with --gt without re-running the pipeline.

Usage:
    python3 analysis_code/compare_prompts_same_model.py
    python3 analysis_code/compare_prompts_same_model.py \
        --results results/output --out analysis/prompts --gt 391-419
"""

import argparse
import csv
import json
import re
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use('Agg')          # headless: write files, never open a window
import matplotlib.pyplot as plt

FOLDER_RE = re.compile(r'^v(?P<prompt>\d+)_(?P<provider>[a-z]+)_(?P<model>.+)$')

# ---------------------------------------------------------------------------
# Display names
# ---------------------------------------------------------------------------
# Folder-name suffixes that record how a batch was run rather than which model
# ran it. The model id cannot carry them, so they are appended to the label.
CONFIG_MARKERS = ('nores',)


def display_name(model_id, folder_token):
    """Short label for tables and plots.

    Fireworks reports a path ('accounts/fireworks/models/kimi-k3') while every
    other provider reports a bare name, so taking the last path segment
    shortens the former and leaves the latter untouched. A marker such as
    '_nores' (reasoning disabled) lives only in the folder name, so it is
    carried into the label; without it two batches of the same model run under
    different settings would be indistinguishable in a figure.
    """
    name = (model_id or folder_token).split('/')[-1]
    for marker in CONFIG_MARKERS:
        if folder_token.endswith('_' + marker):
            name = f'{name} ({marker})'
    return name


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def parse_gt(spec):
    """'391-419' -> set(range(391, 420)). Also accepts '391-419,500-505'."""
    ids = set()
    for part in spec.split(','):
        part = part.strip()
        if '-' in part:
            lo, hi = part.split('-')
            ids.update(range(int(lo), int(hi) + 1))
        elif part:
            ids.add(int(part))
    if not ids:
        raise ValueError(f'--gt parsed to an empty set: {spec!r}')
    return ids


def load_config(folder):
    """Load one result folder; return None (with a note) if unusable."""
    m = FOLDER_RE.match(folder.name)
    if not m:
        return None

    runs_path = folder / 'runs.json'
    if not runs_path.exists():
        print(f'  skip {folder.name}: no runs.json', file=sys.stderr)
        return None
    try:
        runs = json.loads(runs_path.read_text())
    except json.JSONDecodeError as e:
        print(f'  skip {folder.name}: runs.json is not valid JSON ({e})',
              file=sys.stderr)
        return None
    if not runs:
        print(f'  skip {folder.name}: runs.json is empty', file=sys.stderr)
        return None

    meta = runs[0].get('meta', {})
    return {
        'folder': folder,
        'prompt': 'v' + m.group('prompt'),
        'provider': m.group('provider'),
        'model': display_name(meta.get('model'), m.group('model')),
        'runs': runs,
        'meta': meta,
        # identifies the exact conversation; differs whenever the prompt does
        'prompt_sha': meta.get('prompt_sha256', '')[:12],
    }


def discover(results_dir, exclude_providers):
    configs = []
    for folder in sorted(Path(results_dir).iterdir()):
        if not folder.is_dir():
            continue
        cfg = load_config(folder)
        if cfg is None or cfg['provider'] in exclude_providers:
            continue
        configs.append(cfg)
    return configs


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def prf(selected, gt, universe_size):
    tp = len(selected & gt)
    fp = len(selected - gt)
    fn = len(gt - selected)
    tn = universe_size - tp - fp - fn
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return dict(tp=tp, fp=fp, fn=fn, tn=tn,
                precision=precision, recall=recall, f1=f1)


def jaccard(a, b):
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


def stdev(xs):
    xs = list(xs)
    if len(xs) < 2:
        return 0.0
    mu = mean(xs)
    return (sum((x - mu) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


def analyse(cfg, gt):
    """Per-(model, prompt) metrics, including the modal set used later for
    measuring how much the answer itself moves between prompts."""
    runs = cfg['runs']
    lo, hi = cfg['meta'].get('chunk_slice', [0, 0])
    universe_size = (hi - lo) if hi > lo else max(len(gt) * 6, 172)

    valid = [r for r in runs if r.get('meta', {}).get('parse') != 'failed']
    sets = [set(int(c) for c in r.get('transaction_chunks', [])) for r in valid]
    per_run = [prf(s, gt, universe_size) for s in sets]
    pair_j = [jaccard(a, b) for a, b in combinations(sets, 2)]

    freq = Counter()
    for s in sets:
        freq.update(s)
    n = len(sets) or 1
    rates = {c: k / n for c, k in freq.items()}

    patterns = Counter(frozenset(s) for s in sets)
    modal = set(patterns.most_common(1)[0][0]) if patterns else set()

    return {
        'model': cfg['model'],
        'provider': cfg['provider'],
        'prompt': cfg['prompt'],
        'prompt_sha': cfg['prompt_sha'],
        'n_runs': len(runs),
        'n_valid': len(valid),
        'n_parse_failed': len(runs) - len(valid),

        'size_mean': mean(len(s) for s in sets),
        'size_std': stdev(len(s) for s in sets),
        'precision_mean': mean(p['precision'] for p in per_run),
        'recall_mean': mean(p['recall'] for p in per_run),
        'f1_mean': mean(p['f1'] for p in per_run),
        'f1_std': stdev(p['f1'] for p in per_run),
        'tp_mean': mean(p['tp'] for p in per_run),
        'fp_mean': mean(p['fp'] for p in per_run),
        'fn_mean': mean(p['fn'] for p in per_run),

        'p_correct': (sum(p['tp'] for p in per_run) / (len(per_run) * len(gt))
                      if per_run and gt else 0.0),
        'p_noise': (sum(p['fp'] for p in per_run)
                    / (len(per_run) * max(universe_size - len(gt), 1))
                    if per_run else 0.0),

        'jaccard_mean': mean(pair_j),
        'n_unique_sets': len(patterns),
        'modal_set_freq': (patterns.most_common(1)[0][1] / len(sets)
                           if sets else 0.0),
        'chunk_variability': mean(p * (1 - p) for p in rates.values()) if rates else 0.0,

        '_modal': modal,
        '_sizes': [len(s) for s in sets],
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def write_csv(path, rows, fields=None):
    if not rows:
        return
    fields = fields or [k for k in rows[0] if not k.startswith('_')]
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v)
                        for k, v in r.items() if k in fields})


def plot_f1_grouped(by_model, prompts, out):
    """F1 per model, one bar per prompt — the headline prompt-effect figure."""
    models = sorted(by_model, key=lambda m: -mean(
        r['f1_mean'] for r in by_model[m].values()))
    width = 0.8 / len(prompts)
    fig, ax = plt.subplots(figsize=(max(8, len(models) * 1.1), 5))
    for i, p in enumerate(prompts):
        xs = [j + i * width for j in range(len(models))]
        ys = [by_model[m].get(p, {}).get('f1_mean', 0) for m in models]
        es = [by_model[m].get(p, {}).get('f1_std', 0) for m in models]
        ax.bar(xs, ys, width, yerr=es, capsize=3, label=p)
    ax.set_xticks([j + 0.4 - width / 2 for j in range(len(models))])
    ax.set_xticklabels(models, rotation=40, ha='right', fontsize=8)
    ax.set_ylabel('F1 vs ground truth (mean of runs)')
    ax.set_title('Prompt effect per model')
    ax.set_ylim(0, 1.05)
    ax.legend(title='prompt')
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / 'f1_by_prompt_grouped.png', dpi=150)
    plt.close(fig)


def plot_sensitivity(sens, out):
    """How much each model's F1 moves between the best and worst prompt."""
    sens = sorted(sens, key=lambda r: -r['f1_range'])
    fig, ax = plt.subplots(figsize=(max(7, len(sens) * 0.85), 4.5))
    ax.bar(range(len(sens)), [r['f1_range'] for r in sens], color='#8172B3')
    ax.set_xticks(range(len(sens)))
    ax.set_xticklabels([r['model'] for r in sens], rotation=40, ha='right',
                       fontsize=8)
    ax.set_ylabel('F1 range across prompts (max - min)')
    ax.set_title('Prompt sensitivity by model\n'
                 '(taller = the wording matters more for this model)')
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / 'prompt_sensitivity.png', dpi=150)
    plt.close(fig)


def plot_size_by_prompt(by_model, prompts, out):
    """Selection size per prompt: shows whether a prompt changes how much the
    model returns, which is the mechanism behind most of the F1 differences."""
    models = sorted(by_model)
    width = 0.8 / len(prompts)
    fig, ax = plt.subplots(figsize=(max(8, len(models) * 1.1), 4.5))
    for i, p in enumerate(prompts):
        xs = [j + i * width for j in range(len(models))]
        ys = [by_model[m].get(p, {}).get('size_mean', 0) for m in models]
        es = [by_model[m].get(p, {}).get('size_std', 0) for m in models]
        ax.bar(xs, ys, width, yerr=es, capsize=3, label=p)
    ax.set_xticks([j + 0.4 - width / 2 for j in range(len(models))])
    ax.set_xticklabels(models, rotation=40, ha='right', fontsize=8)
    ax.set_ylabel('chunks selected per run (mean)')
    ax.set_title('Selection size by prompt')
    ax.legend(title='prompt')
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / 'selection_size_by_prompt.png', dpi=150)
    plt.close(fig)


def plot_stability_by_prompt(by_model, prompts, out):
    """Run-to-run agreement per prompt: does a clearer prompt also make the
    model more self-consistent?"""
    models = sorted(by_model)
    width = 0.8 / len(prompts)
    fig, ax = plt.subplots(figsize=(max(8, len(models) * 1.1), 4.5))
    for i, p in enumerate(prompts):
        xs = [j + i * width for j in range(len(models))]
        ys = [by_model[m].get(p, {}).get('jaccard_mean', 0) for m in models]
        ax.bar(xs, ys, width, label=p)
    ax.set_xticks([j + 0.4 - width / 2 for j in range(len(models))])
    ax.set_xticklabels(models, rotation=40, ha='right', fontsize=8)
    ax.set_ylabel('mean pairwise Jaccard across runs')
    ax.set_title('Run-to-run stability by prompt')
    ax.set_ylim(0, 1.05)
    ax.legend(title='prompt')
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / 'stability_by_prompt.png', dpi=150)
    plt.close(fig)


def plot_cross_prompt_overlap(by_model, prompts, out):
    """Mean Jaccard between the modal answers produced under each pair of
    prompts, averaged over models: how much the answer itself moves."""
    grid = []
    for a in prompts:
        row = []
        for b in prompts:
            vals = [jaccard(by_model[m][a]['_modal'], by_model[m][b]['_modal'])
                    for m in by_model
                    if a in by_model[m] and b in by_model[m]]
            row.append(mean(vals))
        grid.append(row)

    fig, ax = plt.subplots(figsize=(1.4 * len(prompts) + 2.5,
                                    1.2 * len(prompts) + 2.2))
    im = ax.imshow(grid, cmap='magma', vmin=0, vmax=1)
    ax.set_xticks(range(len(prompts)), prompts)
    ax.set_yticks(range(len(prompts)), prompts)
    for i in range(len(prompts)):
        for j in range(len(prompts)):
            ax.text(j, i, f'{grid[i][j]:.2f}', ha='center', va='center',
                    color='white' if grid[i][j] < 0.6 else 'black', fontsize=9)
    ax.set_title('Agreement between prompts\n'
                 '(mean Jaccard of modal answers, averaged over models)')
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(out / 'cross_prompt_agreement.png', dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description='Compare prompt versions within each model.')
    ap.add_argument('--results', default='results/output',
                    help='Folder containing the v<N>_<provider>_<model> dirs')
    ap.add_argument('--out', default='results/analysis/prompts',
                    help='Where to write CSVs and plots')
    ap.add_argument('--gt', default='391-419',
                    help='Ground-truth chunk ids, e.g. 391-419 (default)')
    ap.add_argument('--exclude-provider', action='append', default=[],
                    help='Provider to exclude; repeatable (default: fireworks)')
    args = ap.parse_args()

    results_dir = Path(args.results)
    if not results_dir.is_dir():
        sys.exit(f'ERROR: results folder not found: {results_dir}')

    gt = parse_gt(args.gt)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f'Loading from {results_dir} (excluding: {", ".join(args.exclude_provider)})')
    configs = discover(results_dir, set(args.exclude_provider))
    if not configs:
        sys.exit('ERROR: no usable result folders found. Expected directories '
                 'named like v3_openai_gpt4o containing runs.json.')

    rows = [analyse(c, gt) for c in configs]

    # ---- group by model ---------------------------------------------------
    by_model = {}
    for r in rows:
        by_model.setdefault(r['model'], {})[r['prompt']] = r
    prompts = sorted({r['prompt'] for r in rows})

    # Models missing some prompt version are kept but reported, since the
    # sensitivity numbers below are only comparable across a full set.
    incomplete = {m: sorted(set(prompts) - set(ps))
                  for m, ps in by_model.items() if len(ps) < len(prompts)}

    all_rows = []
    sensitivity = []

    for model in sorted(by_model):
        group = [by_model[model][p] for p in prompts if p in by_model[model]]
        group.sort(key=lambda r: -r['f1_mean'])
        all_rows.extend(group)
        write_csv(out / f'prompt_comparison_{model.replace("/", "_")}.csv', group)

        f1s = [r['f1_mean'] for r in group]
        sizes = [r['size_mean'] for r in group]
        # Cross-prompt agreement: how different the modal answers are
        cross = [jaccard(a['_modal'], b['_modal'])
                 for a, b in combinations(group, 2)]
        sensitivity.append({
            'model': model,
            'provider': group[0]['provider'],
            'n_prompts': len(group),
            'best_prompt': group[0]['prompt'],
            'best_f1': group[0]['f1_mean'],
            'worst_prompt': group[-1]['prompt'],
            'worst_f1': group[-1]['f1_mean'],
            'f1_range': max(f1s) - min(f1s),
            'size_range': max(sizes) - min(sizes),
            'cross_prompt_jaccard': mean(cross),
            **{f'f1_{r["prompt"]}': r['f1_mean'] for r in group},
            **{f'size_{r["prompt"]}': r['size_mean'] for r in group},
        })

    write_csv(out / 'prompt_comparison_all.csv', all_rows)
    write_csv(out / 'prompt_sensitivity.csv',
              sorted(sensitivity, key=lambda r: -r['f1_range']))

    # ---- per-prompt summary across models ---------------------------------
    prompt_summary = []
    for p in prompts:
        rs = [r for r in rows if r['prompt'] == p]
        wins = sum(1 for s in sensitivity if s['best_prompt'] == p)
        prompt_summary.append({
            'prompt': p,
            'prompt_sha': rs[0]['prompt_sha'],
            'n_models': len(rs),
            'f1_mean': mean(r['f1_mean'] for r in rs),
            'f1_min': min(r['f1_mean'] for r in rs),
            'f1_max': max(r['f1_mean'] for r in rs),
            'f1_spread_across_models': stdev(r['f1_mean'] for r in rs),
            'precision_mean': mean(r['precision_mean'] for r in rs),
            'recall_mean': mean(r['recall_mean'] for r in rs),
            'size_mean': mean(r['size_mean'] for r in rs),
            'jaccard_mean': mean(r['jaccard_mean'] for r in rs),
            'models_where_best': wins,
        })
    write_csv(out / 'prompt_summary.csv', prompt_summary)

    # ---- plots ------------------------------------------------------------
    plot_f1_grouped(by_model, prompts, out)
    plot_sensitivity(sensitivity, out)
    plot_size_by_prompt(by_model, prompts, out)
    plot_stability_by_prompt(by_model, prompts, out)
    if len(prompts) > 1:
        plot_cross_prompt_overlap(by_model, prompts, out)

    # ---- terminal summary -------------------------------------------------
    print(f'\nGround truth: {len(gt)} chunks ({args.gt})')
    print(f'{len(by_model)} models x {len(prompts)} prompts '
          f'({", ".join(prompts)})')
    if incomplete:
        print('  incomplete (missing prompt versions):')
        for m, miss in incomplete.items():
            print(f'    {m}: missing {", ".join(miss)}')

    print('\n=== prompt quality across models ===')
    print(f'{"prompt":<8}{"F1 mean":>9}{"min":>7}{"max":>7}{"size":>7}'
          f'{"Jacc":>7}{"best for":>10}')
    for s in prompt_summary:
        print(f'{s["prompt"]:<8}{s["f1_mean"]:>9.3f}{s["f1_min"]:>7.3f}'
              f'{s["f1_max"]:>7.3f}{s["size_mean"]:>7.1f}'
              f'{s["jaccard_mean"]:>7.3f}{s["models_where_best"]:>7}/{len(by_model)}')

    print('\n=== per-model best prompt and sensitivity ===')
    print(f'{"model":<26}{"best":>6}{"F1":>7}{"worst":>7}{"F1":>7}'
          f'{"range":>7}{"xJacc":>7}')
    for s in sorted(sensitivity, key=lambda r: -r['f1_range']):
        print(f'{s["model"]:<26}{s["best_prompt"]:>6}{s["best_f1"]:>7.3f}'
              f'{s["worst_prompt"]:>7}{s["worst_f1"]:>7.3f}'
              f'{s["f1_range"]:>7.3f}{s["cross_prompt_jaccard"]:>7.3f}')

    most = max(sensitivity, key=lambda r: r['f1_range'])
    least = min(sensitivity, key=lambda r: r['f1_range'])
    best_overall = max(prompt_summary, key=lambda r: r['f1_mean'])
    print(f'\nMost prompt-sensitive : {most["model"]} '
          f'(F1 range {most["f1_range"]:.3f})')
    print(f'Least prompt-sensitive: {least["model"]} '
          f'(F1 range {least["f1_range"]:.3f})')
    print(f'Strongest prompt overall: {best_overall["prompt"]} '
          f'(mean F1 {best_overall["f1_mean"]:.3f}, '
          f'best for {best_overall["models_where_best"]}/{len(by_model)} models)')

    print(f'\nWrote {len(by_model) + 3} CSVs and 5 plots to {out}/')


if __name__ == '__main__':
    main()
