#!/usr/bin/env python3
"""How many runs are needed before the aggregated selection settles?

Subsamples n of the N available runs, applies a decision rule to each
subsample, and measures how the decision behaves as n grows:

  * agreement with the full-budget decision   -- convergence to the model's
    own settled answer (reliability);
  * F1 against ground truth                   -- convergence to the right
    answer (validity);
  * fraction of distinct answers discovered   -- how much of the model's
    answer menu an n-run budget has seen;
  * spread of the p_correct estimate          -- how stable the parameter
    feeding the voter is at budget n.

Works on any result folder produced by run_pipeline.py.

Usage:
    # one experiment folder, or a directory of them
    python3 analysis_code/budget_curves.py \
        --results results/output/q1_tables --out analysis/budget

    python3 analysis_code/budget_curves.py \
        --results results/output/q1_tables_v1/v1_openai_gpt4o --rule voter
"""

import argparse
import csv
import json
import math
import random
import re
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

FOLDER_RE = re.compile(r'^v(?P<prompt>\d+)_(?P<provider>[a-z]+)_(?P<model>.+)$')

# ---------------------------------------------------------------------------
# Display names
# ---------------------------------------------------------------------------
CONFIG_MARKERS = ('nores',)

# The thesis tables write some API identifiers differently; the figures follow
# them so that a model is named the same way everywhere.
NAME_OVERRIDES = {'claude-haiku-4-5-20251001': 'claude-haiku-4.5'}

# Prefix for every figure written by this script; set from --tag in main().
TAG = 'q0'


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
    name = NAME_OVERRIDES.get(name, name)
    for marker in CONFIG_MARKERS:
        if folder_token.endswith('_' + marker):
            name = f'{name} ({marker})'
    return name


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def parse_gt(spec):
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


def load_folder(folder, key):
    """Load one result folder into (label, runs, universe_size), or None."""
    runs_path = folder / 'runs.json'
    if not runs_path.exists():
        return None
    try:
        raw = json.loads(runs_path.read_text())
    except json.JSONDecodeError as e:
        print(f'  skip {folder.name}: runs.json is not valid JSON ({e})',
              file=sys.stderr)
        return None
    if not raw:
        print(f'  skip {folder.name}: runs.json is empty', file=sys.stderr)
        return None

    # Unparseable replies were recorded as empty selections; including them
    # would understate every metric here.
    valid = [r for r in raw if r.get('meta', {}).get('parse') != 'failed']
    if not valid:
        print(f'  skip {folder.name}: no parseable runs', file=sys.stderr)
        return None

    meta = valid[0].get('meta', {})
    lo, hi = meta.get('chunk_slice', [0, 0])
    universe = (hi - lo) if hi > lo else 172

    m = FOLDER_RE.match(folder.name)
    model = display_name(meta.get('model'),
                         m.group('model') if m else folder.name)
    prompt = ('v' + m.group('prompt')) if m else '?'

    sets = [frozenset(int(c) for c in r.get('transaction_chunks', []))
            for r in valid]
    return {
        'label': f'{model} ({prompt})',
        'model': model,
        'prompt': prompt,
        'folder': folder.name,
        'sets': sets,
        'n_total': len(sets),
        'n_dropped': len(raw) - len(valid),
        'universe': universe,
    }


def discover(results_dir, key):
    """Accept either a single result folder or a directory containing many."""
    results_dir = Path(results_dir)
    single = load_folder(results_dir, key)
    if single:
        return [single]

    configs = []
    for folder in sorted(results_dir.iterdir()):
        if folder.is_dir():
            cfg = load_folder(folder, key)
            if cfg:
                configs.append(cfg)
    return configs


# ---------------------------------------------------------------------------
# Decision rules and metrics
# ---------------------------------------------------------------------------
def decide(sets, rule, p_correct, p_noise, prior, threshold):
    """Aggregate a collection of run selections into one chunk set."""
    if not sets:
        return frozenset()
    n = len(sets)
    counts = Counter()
    for s in sets:
        counts.update(s)

    if rule == 'majority':
        return frozenset(c for c, k in counts.items() if k * 2 >= n)
    if rule == 'union':
        return frozenset(counts)
    if rule == 'intersection':
        return frozenset(c for c, k in counts.items() if k == n)
    if rule == 'voter':
        from pipeline.voter_bayesian import posterior_relevance
        return frozenset(
            c for c, k in counts.items()
            if posterior_relevance(k, n, p_correct, p_noise, prior) >= threshold)
    raise ValueError(f'unknown rule: {rule}')


def jaccard(a, b):
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def f1(selected, gt):
    tp = len(selected & gt)
    if not tp:
        return 0.0
    prec = tp / len(selected)
    rec = tp / len(gt)
    return 2 * prec * rec / (prec + rec)


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


def stdev(xs):
    xs = list(xs)
    if len(xs) < 2:
        return 0.0
    mu = mean(xs)
    return (sum((x - mu) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


def budget_grid(n_total):
    """Sample sizes to evaluate: dense at the low end where the curve moves."""
    if n_total <= 20:
        return list(range(1, n_total + 1))
    grid = [1, 2, 3, 5, 7, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100,
            125, 150, 200, 250, 300, 400, 500]
    grid = [n for n in grid if n <= n_total]
    if grid[-1] != n_total:
        grid.append(n_total)
    return grid


def curve_for(cfg, gt, rule, draws, args, seed=0):
    """One row per budget n, averaged over `draws` random subsamples."""
    sets = cfg['sets']
    N = cfg['n_total']
    rng = random.Random(seed)

    # Reference decision and the true menu, both at the full budget
    full_decision = decide(sets, rule, args.p_correct, args.p_noise,
                           args.prior, args.threshold)
    all_patterns = set(sets)
    n_patterns_total = len(all_patterns)

    rows = []
    for n in budget_grid(N):
        agree, truth, discovered, p_hats, sizes = [], [], [], [], []
        # n == N has only one possible subsample, so one draw suffices
        reps = 1 if n >= N else draws
        for _ in range(reps):
            sub = sets if n >= N else rng.sample(sets, n)
            d = decide(sub, rule, args.p_correct, args.p_noise,
                       args.prior, args.threshold)
            agree.append(jaccard(d, full_decision))
            truth.append(f1(d, gt))
            sizes.append(len(d))
            discovered.append(len(set(sub)) / n_patterns_total
                              if n_patterns_total else 1.0)
            # p_correct as the estimation step would compute it from n runs
            p_hats.append(mean(len(s & gt) for s in sub) / len(gt))

        rows.append({
            'model': cfg['model'], 'prompt': cfg['prompt'],
            'n_runs': n, 'n_draws': reps,
            'agreement_with_full_mean': mean(agree),
            'agreement_with_full_std': stdev(agree),
            'f1_vs_gt_mean': mean(truth),
            'f1_vs_gt_std': stdev(truth),
            'decision_size_mean': mean(sizes),
            'patterns_discovered_frac': mean(discovered),
            'p_correct_mean': mean(p_hats),
            'p_correct_std': stdev(p_hats),
        })
    return rows, full_decision, n_patterns_total


def first_n_reaching(rows, field, target):
    """Smallest budget whose mean `field` reaches `target` (None if never)."""
    for r in rows:
        if r[field] >= target:
            return r['n_runs']
    return None


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def write_csv(path, rows):
    if not rows:
        return
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v)
                        for k, v in r.items()})


def plot_two_curve(rows, cfg, out, rule):
    """The headline figure: reliability and validity on the same axes.

    The vertical gap that remains once both curves flatten is the part of the
    error that no amount of extra sampling can remove.
    """
    ns = [r['n_runs'] for r in rows]
    self_c = [r['agreement_with_full_mean'] for r in rows]
    truth_c = [r['f1_vs_gt_mean'] for r in rows]

    fig, ax = plt.subplots(figsize=(7, 4.8))
    ax.fill_between(ns, truth_c, self_c, color='grey', alpha=0.15,
                    label='validity ceiling (gap)')
    ax.plot(ns, self_c, 'o-', color='#4C72B0', markersize=4,
            label="agreement with full-budget answer (reliability)")
    ax.plot(ns, truth_c, 's-', color='#C44E52', markersize=4,
            label='F1 vs ground truth (validity)')
    ax.set_xlabel('number of runs')
    ax.set_ylabel('score')
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / f'{TAG}_budget_{cfg["folder"].split("_", 2)[-1]}.png',
                dpi=150)
    plt.close(fig)


def plot_discovery(rows, cfg, out):
    """How much of the model's answer menu an n-run budget has seen, and how
    stable the p_correct estimate is at that budget."""
    ns = [r['n_runs'] for r in rows]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 6), sharex=True)

    ax1.plot(ns, [r['patterns_discovered_frac'] for r in rows], 'o-',
             color='#55A868', markersize=4)
    ax1.axhline(0.95, color='grey', linestyle='--', linewidth=1,
                label='95% of distinct answers')
    ax1.set_ylabel('fraction of distinct\nanswers discovered')
    ax1.set_ylim(0, 1.05)
    ax1.set_title(f'Answer discovery and parameter stability — {cfg["label"]}')
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    means = [r['p_correct_mean'] for r in rows]
    stds = [r['p_correct_std'] for r in rows]
    ax2.plot(ns, means, 'o-', color='#8172B3', markersize=4)
    ax2.fill_between(ns, [m - s for m, s in zip(means, stds)],
                     [m + s for m, s in zip(means, stds)],
                     color='#8172B3', alpha=0.2, label='+/- 1 sd over subsamples')
    ax2.set_xlabel('number of runs')
    ax2.set_ylabel('estimated p_correct')
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / f'discovery_{cfg["folder"]}.png', dpi=150)
    plt.close(fig)


def plot_combined(all_rows, out, rule):
    """All configurations on one axis, for comparing budgets across models."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))
    by_label = {}
    for r in all_rows:
        by_label.setdefault(r['model'], []).append(r)

    for label, rows in sorted(by_label.items()):
        ns = [r['n_runs'] for r in rows]
        ax1.plot(ns, [r['agreement_with_full_mean'] for r in rows], 'o-',
                 markersize=3, label=label)
        ax2.plot(ns, [r['f1_vs_gt_mean'] for r in rows], 'o-',
                 markersize=3, label=label)

    for ax, title, ylab in (
            (ax1, 'Reliability: agreement with full-budget answer', 'Jaccard'),
            (ax2, 'Validity: F1 vs ground truth', 'F1')):
        ax.set_xlabel('number of runs')
        ax.set_ylabel(ylab)
        ax.set_ylim(0, 1.05)
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)
    ax2.legend(fontsize=7, loc='lower right')
    fig.tight_layout()
    fig.savefig(out / f'{TAG}_budget_combined.png', dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description='Run-budget curves by subsampling existing runs.')
    ap.add_argument('--results', default='results/output/q1_tables',
                    help='A result folder, or a directory containing several')
    ap.add_argument('--out', default='results/analysis/q1_budget',
                    help='Where to write CSVs and plots')
    ap.add_argument('--gt', default='391-419', help='Ground-truth chunk ids')
    ap.add_argument('--key', default='transaction_chunks',
                    help='Key holding the selected ids in runs.json')
    ap.add_argument('--rule', default='majority',
                    choices=['majority', 'union', 'intersection', 'voter'],
                    help='Aggregation rule whose convergence is measured')
    ap.add_argument('--draws', type=int, default=200,
                    help='Random subsamples per budget (default 200)')
    ap.add_argument('--target', type=float, default=0.95,
                    help='Agreement level used for the "runs needed" report')
    # only used by --rule voter
    ap.add_argument('--p-correct', type=float, default=0.6)
    ap.add_argument('--p-noise', type=float, default=0.01)
    ap.add_argument('--prior', type=float, default=0.5)
    ap.add_argument('--threshold', type=float, default=0.9)
    ap.add_argument('--only', action='append', default=[],
                    help='Draw per-configuration curves only for folders '
                         'containing these tokens; repeatable')
    ap.add_argument('--tag', required=True,
                    help='Prefix for the figure names, e.g. q1 or q2')
    args = ap.parse_args()
    globals()['TAG'] = args.tag

    if not Path(args.results).exists():
        sys.exit(f'ERROR: results path not found: {args.results}')

    gt = parse_gt(args.gt)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.rule == 'voter':
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  

    configs = discover(args.results, args.key)
    if not configs:
        sys.exit(f'ERROR: no usable runs.json found under {args.results}')

    print(f'Loaded {len(configs)} configuration(s) from {args.results}')
    print(f'Ground truth: {len(gt)} chunks ({args.gt}) | rule: {args.rule} | '
          f'{args.draws} subsamples per budget')

    all_rows, summary = [], []
    for cfg in configs:
        rows, full_decision, n_patterns = curve_for(cfg, gt, args.rule,
                                                    args.draws, args)
        all_rows.extend(rows)
        if not args.only or any(tok in cfg['folder'] for tok in args.only):
            plot_two_curve(rows, cfg, out, args.rule)

        n_reach = first_n_reaching(rows, 'agreement_with_full_mean', args.target)
        n_disc = first_n_reaching(rows, 'patterns_discovered_frac', 0.95)
        final_f1 = rows[-1]['f1_vs_gt_mean']
        summary.append({
            'model': cfg['model'], 'prompt': cfg['prompt'],
            'n_available': cfg['n_total'],
            'n_dropped_parse_failed': cfg['n_dropped'],
            'distinct_answers': n_patterns,
            f'runs_for_{args.target}_agreement': n_reach,
            'runs_for_95pct_answer_discovery': n_disc,
            'final_f1_vs_gt': final_f1,
            'validity_ceiling_gap': 1.0 - final_f1,
            'decision_size_at_full': rows[-1]['decision_size_mean'],
        })

    if len(configs) > 1:
        plot_combined(all_rows, out, args.rule)

    # ---- terminal summary -------------------------------------------------
    print(f'\n{"model":<26}{"N":>5}{"answers":>9}{"n@" + str(args.target):>7}'
          f'{"n@95%disc":>11}{"final F1":>10}{"gap":>7}')
    for s in summary:
        n_reach = s[f'runs_for_{args.target}_agreement']
        n_disc = s['runs_for_95pct_answer_discovery']
        print(f'{s["model"]:<26}{s["n_available"]:>5}{s["distinct_answers"]:>9}'
              f'{(n_reach if n_reach else "-"):>7}'
              f'{(n_disc if n_disc else "-"):>11}'
              f'{s["final_f1_vs_gt"]:>10.3f}{s["validity_ceiling_gap"]:>7.3f}')

    flat = [s for s in summary if s['distinct_answers'] <= 1]
    if flat:
        print(f'\n  {len(flat)} configuration(s) returned a single answer in '
              f'every run, so their curves are flat at 1.0 by construction: '
              f'{", ".join(s["model"] for s in flat)}')
    if any(s['n_available'] < 30 for s in summary):
        print('  NOTE: budgets above ~N/3 are estimated from few distinct '
              'subsamples; curves from short run sets are indicative only.')

    print(f'\nWrote {TAG}_budget_combined.png and the selected per-configuration curves to {out}/')


if __name__ == '__main__':
    main()
