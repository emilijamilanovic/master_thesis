#!/usr/bin/env python3
"""Compare models against each other, holding the prompt fixed.

Reads the pipeline output folders produced by run_pipeline.py, which are named
    v<N>_<provider>_<model>/          e.g. v3_openai_gpt4o/
and each contain runs.json (+ stats.json, selected.json, mode_analysis.json).

For every prompt version it builds one table with one row per model, ranks the
models, and flags which are consistently good and which are unstable across the
10 repeated runs.

All per-run metrics are recomputed here from runs.json rather than read from
stats.json, so the ground truth can be changed with --gt without re-running the
pipeline. stats.json is only used as a cross-check of the pooled estimates.

Usage:
    python3 analysis_code/compare_models_same_prompt.py
    python3 analysis_code/compare_models_same_prompt.py \
        --results results/output --out analysis/models --gt 391-419
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

# Folder naming convention written by run_pipeline.py -o
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
    """Load one result folder. Returns a dict, or None if it is unusable.

    Missing optional artifacts are tolerated; a missing or empty runs.json is
    not, since every metric depends on it.
    """
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

    def maybe(name):
        p = folder / name
        try:
            return json.loads(p.read_text()) if p.exists() else None
        except json.JSONDecodeError:
            return None

    meta = runs[0].get('meta', {})
    return {
        'folder': folder,
        'prompt': 'v' + m.group('prompt'),
        'provider': m.group('provider'),
        # Prefer the real model id recorded per run; fall back to the folder token
        'model': display_name(meta.get('model'), m.group('model')),
        'model_short': m.group('model'),
        'runs': runs,
        'stats': maybe('stats.json'),
        'meta': meta,
    }


def discover(results_dir, exclude_providers):
    """Find and load every v*_provider_model folder under results_dir."""
    configs = []
    for folder in sorted(Path(results_dir).iterdir()):
        if not folder.is_dir():
            continue
        cfg = load_config(folder)
        if cfg is None:
            continue
        if cfg['provider'] in exclude_providers:
            continue
        configs.append(cfg)
    return configs


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def prf(selected, gt, universe_size):
    """Precision / recall / F1 plus the raw confusion counts for one run."""
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
    """Set overlap; two empty selections count as identical."""
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
    """Compute every per-config metric used by the tables and plots."""
    runs = cfg['runs']
    meta = cfg['meta']

    # Universe = the chunk slice actually shown to the model, e.g. [313, 485)
    lo, hi = meta.get('chunk_slice', [0, 0])
    universe_size = (hi - lo) if hi > lo else max(len(gt) * 6, 172)

    # A run whose reply could not be parsed recorded an empty selection; keeping
    # it would silently depress recall, so it is counted and excluded.
    valid = [r for r in runs if r.get('meta', {}).get('parse') != 'failed']
    n_failed = len(runs) - len(valid)

    sets = [set(int(c) for c in r.get('transaction_chunks', [])) for r in valid]
    per_run = [prf(s, gt, universe_size) for s in sets]

    # --- stability across the repeated runs ---
    pair_j = [jaccard(a, b) for a, b in combinations(sets, 2)]
    pattern_counts = Counter(frozenset(s) for s in sets)
    modal_pattern, modal_count = (pattern_counts.most_common(1)[0]
                                  if pattern_counts else (frozenset(), 0))

    # --- chunk-level variability ---
    freq = Counter()
    for s in sets:
        freq.update(s)
    n = len(sets) or 1
    rates = {c: k / n for c, k in freq.items()}
    always = [c for c, p in rates.items() if p == 1.0]
    variable = [c for c, p in rates.items() if 0 < p < 1.0]
    # Mean Bernoulli variance over chunks that were ever selected: 0 when every
    # chunk is either always or never chosen, max 0.25 at a 50/50 coin flip.
    chunk_var = mean(p * (1 - p) for p in rates.values()) if rates else 0.0

    # --- aggregation rules over the 10 runs, scored against GT ---
    union = set().union(*sets) if sets else set()
    inter = set.intersection(*sets) if sets else set()
    majority = {c for c, p in rates.items() if p >= 0.5}

    row = {
        'prompt': cfg['prompt'],
        'provider': cfg['provider'],
        'model': cfg['model'],
        'n_runs': len(runs),
        'n_valid': len(valid),
        'n_parse_failed': n_failed,
        'n_parse_salvaged': sum(1 for r in valid
                                if r.get('meta', {}).get('parse') == 'regex'),

        'size_mean': mean(len(s) for s in sets),
        'size_std': stdev(len(s) for s in sets),
        'size_min': min((len(s) for s in sets), default=0),
        'size_max': max((len(s) for s in sets), default=0),

        'precision_mean': mean(p['precision'] for p in per_run),
        'recall_mean': mean(p['recall'] for p in per_run),
        'f1_mean': mean(p['f1'] for p in per_run),
        'f1_std': stdev(p['f1'] for p in per_run),
        'tp_mean': mean(p['tp'] for p in per_run),
        'fp_mean': mean(p['fp'] for p in per_run),
        'fn_mean': mean(p['fn'] for p in per_run),

        # pooled inclusion probabilities (same definition as the pipeline)
        'p_correct': (sum(p['tp'] for p in per_run) / (len(per_run) * len(gt))
                      if per_run and gt else 0.0),
        'p_noise': (sum(p['fp'] for p in per_run)
                    / (len(per_run) * max(universe_size - len(gt), 1))
                    if per_run else 0.0),

        'jaccard_mean': mean(pair_j),
        'jaccard_min': min(pair_j) if pair_j else 1.0,
        'n_unique_sets': len(pattern_counts),
        'modal_set_freq': modal_count / len(sets) if sets else 0.0,

        'chunks_ever': len(rates),
        'chunks_always': len(always),
        'chunks_variable': len(variable),
        'chunk_variability': chunk_var,

        'union_f1': prf(union, gt, universe_size)['f1'],
        'union_recall': prf(union, gt, universe_size)['recall'],
        'majority_f1': prf(majority, gt, universe_size)['f1'],
        'intersection_precision': prf(inter, gt, universe_size)['precision'],
    }

    # Cross-check the pooled p_correct against what the pipeline recorded.
    st = cfg.get('stats')
    if st and 'p_correct_hat' in st and st.get('gt_size') == len(gt):
        row['p_correct_pipeline'] = st['p_correct_hat']
    else:
        row['p_correct_pipeline'] = ''

    row['_rates'] = rates          # kept for the heatmap, stripped before CSV
    row['_sizes'] = [len(s) for s in sets]
    row['_f1s'] = [p['f1'] for p in per_run]
    return row


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def write_csv(path, rows, fields=None):
    """Write rows to CSV, dropping the private _-prefixed helper columns."""
    if not rows:
        return
    fields = fields or [k for k in rows[0] if not k.startswith('_')]
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v)
                        for k, v in r.items() if k in fields})


def plot_f1_by_model(rows, prompt, out):
    """Mean F1 per model with the run-to-run standard deviation as error bars."""
    rows = sorted(rows, key=lambda r: -r['f1_mean'])
    labels = [r['model'] for r in rows]
    fig, ax = plt.subplots(figsize=(max(7, len(rows) * 0.85), 4.5))
    ax.bar(range(len(rows)), [r['f1_mean'] for r in rows],
           yerr=[r['f1_std'] for r in rows], capsize=4, color='#4C72B0')
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, rotation=40, ha='right', fontsize=8)
    ax.set_ylabel('F1 vs ground truth (mean of runs)')
    ax.set_title(f'Per-run selection quality by model — prompt {prompt}')
    ax.set_ylim(0, 1)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / f'f1_by_model_{prompt}.png', dpi=150)
    plt.close(fig)


def plot_precision_recall(rows, prompt, out):
    """Where each model sits in the precision/recall plane."""
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for r in rows:
        ax.scatter(r['recall_mean'], r['precision_mean'], s=60,
                   color='#DD8452', edgecolor='black', linewidth=0.5, zorder=3)
        ax.annotate(r['model'], (r['recall_mean'], r['precision_mean']),
                    textcoords='offset points', xytext=(5, 4), fontsize=7)
    ax.set_xlabel('recall (mean over runs)')
    ax.set_ylabel('precision (mean over runs)')
    ax.set_title(f'Precision vs recall by model — prompt {prompt}')
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.05)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / f'precision_recall_{prompt}.png', dpi=150)
    plt.close(fig)


def plot_stability(rows, prompt, out):
    """Run-to-run agreement (mean pairwise Jaccard) and distinct answer count."""
    rows = sorted(rows, key=lambda r: -r['jaccard_mean'])
    labels = [r['model'] for r in rows]
    x = range(len(rows))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(7, len(rows) * 0.85), 7),
                                   sharex=True)
    ax1.bar(x, [r['jaccard_mean'] for r in rows], color='#55A868')
    ax1.set_ylabel('mean pairwise Jaccard')
    ax1.set_ylim(0, 1.02)
    ax1.set_title(f'Run-to-run stability — prompt {prompt}\n'
                  '(1.0 = every run returned the same set)')
    ax1.grid(axis='y', alpha=0.3)

    ax2.bar(x, [r['n_unique_sets'] for r in rows], color='#C44E52')
    ax2.set_ylabel('distinct selections\nin the run set')
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(labels, rotation=40, ha='right', fontsize=8)
    ax2.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / f'stability_{prompt}.png', dpi=150)
    plt.close(fig)


def plot_size_distribution(rows, prompt, gt, out):
    """Box plot of how many chunks each model selects per run."""
    rows = sorted(rows, key=lambda r: r['size_mean'])
    fig, ax = plt.subplots(figsize=(max(7, len(rows) * 0.85), 4.5))
    # Set tick labels separately: boxplot's own label kwarg was renamed
    # between matplotlib versions (labels -> tick_labels).
    ax.boxplot([r['_sizes'] for r in rows], showmeans=True)
    ax.set_xticks(range(1, len(rows) + 1))
    ax.set_xticklabels([r['model'] for r in rows])
    ax.axhline(len(gt), color='red', linestyle='--', linewidth=1,
               label=f'ground-truth size ({len(gt)})')
    ax.set_ylabel('chunks selected per run')
    ax.set_title(f'Selection size by model — prompt {prompt}')
    ax.legend(fontsize=8)
    plt.setp(ax.get_xticklabels(), rotation=40, ha='right', fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / f'selection_size_{prompt}.png', dpi=150)
    plt.close(fig)


def plot_chunk_heatmap(rows, prompt, gt, out):
    """Per-chunk selection frequency, models x chunks.

    Restricted to chunks selected at least once by some model, otherwise the
    172-column grid is unreadable. Ground-truth chunks are marked on the axis.
    """
    chunks = sorted({c for r in rows for c in r['_rates']})
    if not chunks:
        return
    rows = sorted(rows, key=lambda r: -r['f1_mean'])
    grid = [[r['_rates'].get(c, 0.0) for c in chunks] for r in rows]

    fig, ax = plt.subplots(figsize=(max(8, len(chunks) * 0.22),
                                    max(3.5, len(rows) * 0.42)))
    im = ax.imshow(grid, aspect='auto', cmap='viridis', vmin=0, vmax=1)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r['model'] for r in rows], fontsize=8)
    ax.set_xticks(range(len(chunks)))
    ax.set_xticklabels([f'{c}*' if c in gt else str(c) for c in chunks],
                       rotation=90, fontsize=6)
    ax.set_title(f'Selection frequency per chunk — prompt {prompt}\n'
                 '(* = in ground truth; colour = fraction of runs selecting it)')
    fig.colorbar(im, ax=ax, label='fraction of runs', shrink=0.8)
    fig.tight_layout()
    fig.savefig(out / f'chunk_frequency_{prompt}.png', dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description='Compare models within each prompt version.')
    ap.add_argument('--results', default='results/output',
                    help='Folder containing the v<N>_<provider>_<model> dirs')
    ap.add_argument('--out', default='results/analysis/models',
                    help='Where to write CSVs and plots')
    ap.add_argument('--gt', default='391-419',
                    help='Ground-truth chunk ids, e.g. 391-419 (default)')
    ap.add_argument('--exclude-provider', action='append', default=[],
                    help='Provider to exclude; repeatable (default: fireworks)')
    ap.add_argument('--prompt', action='append',
                    help='Restrict to these prompt versions, e.g. --prompt v3')
    args = ap.parse_args()

    results_dir = Path(args.results)
    if not results_dir.is_dir():
        sys.exit(f'ERROR: results folder not found: {results_dir}')

    gt = parse_gt(args.gt)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f'Loading from {results_dir} (excluding: {", ".join(args.exclude_provider)})')
    configs = discover(results_dir, set(args.exclude_provider))
    if args.prompt:
        configs = [c for c in configs if c['prompt'] in set(args.prompt)]
    if not configs:
        sys.exit('ERROR: no usable result folders found. Expected directories '
                 'named like v3_openai_gpt4o containing runs.json.')

    rows = [analyse(c, gt) for c in configs]

    # ---- group by prompt version -----------------------------------------
    by_prompt = {}
    for r in rows:
        by_prompt.setdefault(r['prompt'], []).append(r)

    all_rows = []
    print(f'\nGround truth: {len(gt)} chunks ({args.gt})')

    for prompt in sorted(by_prompt):
        group = sorted(by_prompt[prompt], key=lambda r: -r['f1_mean'])
        for i, r in enumerate(group, 1):
            r['rank_f1'] = i
        all_rows.extend(group)

        write_csv(out / f'model_comparison_{prompt}.csv', group)
        plot_f1_by_model(group, prompt, out)
        plot_precision_recall(group, prompt, out)
        plot_stability(group, prompt, out)
        plot_size_distribution(group, prompt, gt, out)
        plot_chunk_heatmap(group, prompt, gt, out)

        # ---- terminal summary for this prompt ----------------------------
        print(f'\n=== prompt {prompt} — {len(group)} models '
              f'({group[0]["n_runs"]} runs each) ===')
        print(f'{"model":<26}{"F1":>7}{"±":>7}{"prec":>7}{"rec":>7}'
              f'{"size":>7}{"Jacc":>7}{"uniq":>6}')
        for r in group:
            print(f'{r["model"]:<26}{r["f1_mean"]:>7.3f}{r["f1_std"]:>7.3f}'
                  f'{r["precision_mean"]:>7.3f}{r["recall_mean"]:>7.3f}'
                  f'{r["size_mean"]:>7.1f}{r["jaccard_mean"]:>7.3f}'
                  f'{r["n_unique_sets"]:>6}')

        # Consistently good: top-half F1 and above-median stability.
        med_j = sorted(r['jaccard_mean'] for r in group)[len(group) // 2]
        solid = [r['model'] for r in group[:max(1, len(group) // 2)]
                 if r['jaccard_mean'] >= med_j]
        # Unstable: rarely repeats itself, or F1 swings a lot between runs.
        unstable = [r['model'] for r in group
                    if r['jaccard_mean'] < 0.5 or r['f1_std'] > 0.15]
        print(f'  consistently strong : {", ".join(solid) or "-"}')
        print(f'  unstable            : {", ".join(unstable) or "-"}')
        failed = [(r['model'], r['n_parse_failed']) for r in group
                  if r['n_parse_failed']]
        if failed:
            print('  parse failures      : '
                  + ', '.join(f'{m} ({n})' for m, n in failed))

    write_csv(out / 'model_comparison_all.csv', all_rows)

    print(f'\nWrote {len(by_prompt) * 5 + 1} files to {out}/')
    print('  model_comparison_<prompt>.csv, model_comparison_all.csv')
    print('  f1_by_model / precision_recall / stability / selection_size / '
          'chunk_frequency  (one .png per prompt)')


if __name__ == '__main__':
    main()
