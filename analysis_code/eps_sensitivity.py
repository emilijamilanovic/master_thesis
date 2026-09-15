#!/usr/bin/env python3
"""Is the voter's `eps` a numerical safeguard or a hidden model parameter?

voter_bayesian.py clamps the estimated rates away from 0 and 1:

    eps = 1e-12
    p_correct = min(max(p_correct, eps), 1 - eps)
    p_noise   = min(max(p_noise,   eps), 1 - eps)

If the selections are stable across several orders of magnitude, eps is a
safeguard. If they move, eps is a model parameter in disguise and has to be
justified rather than chosen for convenience.

Reads existing results only; writes nothing except its own CSV/report.

Usage:
    python3 analysis_code/eps_sensitivity.py
    python3 analysis_code/eps_sensitivity.py \
        --results results/output/q2_fonts --gt "200-201,214,224-229,236,239,250-252"
"""

import argparse
import csv
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent)) 
from pipeline.voter_bayesian import log_binom_pmf 

FOLDER_RE = re.compile(r'^v(?P<prompt>\d+)_(?P<provider>[a-z]+)_(?P<model>.+)$')
EPS_GRID = [1e-3, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12]
DEFAULT_EPS = 1e-12          # what voter_bayesian.py currently uses


# ---------------------------------------------------------------------------
# The voter, with eps exposed as a parameter
# ---------------------------------------------------------------------------
def posterior(k, n, p_correct, p_noise, prior, eps):
    """Identical to voter_bayesian.posterior_relevance but with eps injected,
    and using the numerically stable logistic."""
    p_correct = min(max(p_correct, eps), 1 - eps)
    p_noise = min(max(p_noise, eps), 1 - eps)
    prior = min(max(prior, eps), 1 - eps)

    log_lr = log_binom_pmf(k, n, p_correct) - log_binom_pmf(k, n, p_noise)
    logit = math.log(prior) - math.log(1 - prior) + log_lr
    if logit >= 0:
        return 1.0 / (1.0 + math.exp(-logit))
    z = math.exp(logit)
    return z / (1.0 + z)


def min_count(n, p_correct, p_noise, prior, threshold, eps):
    """Smallest k whose posterior clears the threshold: the decision rule."""
    for k in range(n + 1):
        if posterior(k, n, p_correct, p_noise, prior, eps) >= threshold:
            return k
    return None


def select(counts, n, p_correct, p_noise, prior, threshold, eps):
    return frozenset(c for c, k in counts.items()
                     if posterior(k, n, p_correct, p_noise, prior, eps) >= threshold)


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
    return frozenset(ids)


def load(folder, key):
    runs_p, stats_p = folder / 'runs.json', folder / 'stats.json'
    if not runs_p.exists() or not stats_p.exists():
        return None
    try:
        runs = json.loads(runs_p.read_text())
        stats = json.loads(stats_p.read_text())
    except json.JSONDecodeError:
        return None
    valid = [r for r in runs if r.get('meta', {}).get('parse') != 'failed']
    if not valid:
        return None

    counts = Counter()
    for r in valid:
        counts.update(int(c) for c in r.get(key, []))
    m = FOLDER_RE.match(folder.name)
    return {
        'label': (json.loads(runs_p.read_text())[0].get('meta', {}).get('model')
                  or folder.name),
        'prompt': ('v' + m.group('prompt')) if m else '?',
        'folder': folder.name,
        'n': len(valid),
        'counts': counts,
        'p_correct': stats.get('p_correct_hat'),
        'p_noise': stats.get('p_noise_hat'),
    }


def discover(path, key):
    path = Path(path)
    one = load(path, key)
    if one:
        return [one]
    out = []
    for d in sorted(path.iterdir()):
        if d.is_dir():
            cfg = load(d, key)
            if cfg:
                out.append(cfg)
            else:
                for sub in sorted(d.iterdir()) if d.is_dir() else []:
                    if sub.is_dir():
                        c2 = load(sub, key)
                        if c2:
                            out.append(c2)
    return out


def f1(sel, gt):
    tp = len(sel & gt)
    if not tp:
        return 0.0
    p, r = tp / len(sel), tp / len(gt)
    return 2 * p * r / (p + r)


def main():
    ap = argparse.ArgumentParser(description="Sensitivity of the voter to eps.")
    ap.add_argument('--results', default='results/output',
                    help='A result folder, or a directory containing several')
    ap.add_argument('--gt', default='391-419')
    ap.add_argument('--key', default='transaction_chunks')
    ap.add_argument('--prior', type=float, default=0.5)
    ap.add_argument('--threshold', type=float, default=0.9)
    ap.add_argument('-o', '--out', default='results/analysis/eps_sensitivity.csv')
    args = ap.parse_args()

    gt = parse_gt(args.gt)
    configs = discover(args.results, args.key)
    if not configs:
        sys.exit(f'ERROR: no usable results under {args.results}')

    # ---- View 1: the decision rule, analytically -------------------------
    print('=' * 78)
    print('VIEW 1 — decision rule: minimum runs (of 100) a chunk must appear in')
    print('=' * 78)
    print('Only the cases where the clamp actually binds are informative.\n')
    cases = [
        ('p_correct=0.6,  p_noise=0.01   (neither clamped)', 0.6, 0.01),
        ('p_correct=0.6,  p_noise=0      (noise clamped)', 0.6, 0.0),
        ('p_correct=1.0,  p_noise=0.01   (correct clamped)', 1.0, 0.01),
        ('p_correct=1.0,  p_noise=0      (both clamped)', 1.0, 0.0),
    ]
    print(f'{"case":<50}' + ''.join(f'{e:>9.0e}' for e in EPS_GRID))
    for label, p1, p0 in cases:
        row = [min_count(100, p1, p0, args.prior, args.threshold, e) for e in EPS_GRID]
        print(f'{label:<50}' + ''.join(f'{str(v):>9}' for v in row))

    # ---- View 2: the actual recorded experiments -------------------------
    print('\n' + '=' * 78)
    print('VIEW 2 — selected sets from the recorded runs, recomputed at each eps')
    print('=' * 78)
    rows, unstable = [], []
    for cfg in configs:
        p1, p0 = cfg['p_correct'], cfg['p_noise']
        if p1 is None or p0 is None:
            continue
        clamped = []
        if p0 <= 0.0:
            clamped.append('p_noise=0')
        if p1 >= 1.0:
            clamped.append('p_correct=1')

        sels = {e: select(cfg['counts'], cfg['n'], p1, p0, args.prior,
                          args.threshold, e) for e in EPS_GRID}
        ref = sels[DEFAULT_EPS]
        sizes = [len(sels[e]) for e in EPS_GRID]
        changed = len({frozenset(s) for s in sels.values()}) > 1

        rows.append({
            'folder': cfg['folder'], 'model': cfg['label'], 'prompt': cfg['prompt'],
            'n_runs': cfg['n'],
            'p_correct': round(p1, 4), 'p_noise': round(p0, 5),
            'clamp_binds': ';'.join(clamped) or 'none',
            **{f'size_eps_{e:.0e}': len(sels[e]) for e in EPS_GRID},
            **{f'f1_eps_{e:.0e}': round(f1(sels[e], gt), 3) for e in EPS_GRID},
            'selection_changes_with_eps': changed,
            'max_size_diff': max(sizes) - min(sizes),
        })
        if changed:
            unstable.append((cfg, sels))

    hdr = (f'{"model (prompt)":<30}{"p_corr":>7}{"p_noise":>8}{"clamp":>22}'
           + ''.join(f'{e:>8.0e}' for e in EPS_GRID) + '  changes?')
    print(hdr)
    for r in rows:
        sizes = ''.join(f'{r[f"size_eps_{e:.0e}"]:>8}' for e in EPS_GRID)
        print(f'{r["model"] + " (" + r["prompt"] + ")":<30}'
              f'{r["p_correct"]:>7.3f}{r["p_noise"]:>8.4f}{r["clamp_binds"]:>22}'
              f'{sizes}   {"YES" if r["selection_changes_with_eps"] else "no"}')

    # ---- verdict ----------------------------------------------------------
    print('\n' + '=' * 78)
    n_clamped = sum(1 for r in rows if r['clamp_binds'] != 'none')
    print(f'{len(rows)} configurations; clamp binds in {n_clamped}; '
          f'selection changes with eps in {len(unstable)}.')
    if unstable:
        print('\nConfigurations whose selection depends on eps:')
        for cfg, sels in unstable:
            print(f'  {cfg["label"]} ({cfg["prompt"]}), '
                  f'p_correct={cfg["p_correct"]:.3f} p_noise={cfg["p_noise"]:.4f}')
            for e in EPS_GRID:
                extra = sorted(sels[e] - sels[DEFAULT_EPS])
                missing = sorted(sels[DEFAULT_EPS] - sels[e])
                delta = ''
                if extra:
                    delta += f'  +{extra}'
                if missing:
                    delta += f'  -{missing}'
                print(f'      eps={e:.0e}: {len(sels[e]):>3} chunks{delta}')

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with open(args.out, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f'\nWrote {args.out}')


if __name__ == '__main__':
    main()
