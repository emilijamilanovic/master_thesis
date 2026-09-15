#!/usr/bin/env python3
"""Delivered set of majority vote and of the Bayesian voter, side by side.

For each configuration reports the size of the delivered set, how many of its
chunks are in the ground truth, and F1. The voter's set is read from
selected.json (the pipeline's own output); the majority set is rebuilt from
runs.json by keeping every chunk selected in at least half of the runs, which
is the rule budget_curves.py uses.

Nothing here re-runs the voter or changes any stored result.

Usage:
    python3 analysis_code/rule_sets.py \
        --results results/output/q2_fonts \
        --gt "200-201,214,224-229,236,239,250-252"
"""
import argparse
import collections
import json
from pathlib import Path


def parse_gt(spec):
    ids = set()
    for part in spec.split(','):
        part = part.strip()
        if '-' in part:
            lo, hi = part.split('-')
            ids.update(range(int(lo), int(hi) + 1))
        elif part:
            ids.add(int(part))
    return ids


def f1(sel, gt):
    tp = len(sel & gt)
    if not tp:
        return 0.0
    p, r = tp / len(sel), tp / len(gt)
    return 2 * p * r / (p + r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', required=True)
    ap.add_argument('--gt', required=True)
    ap.add_argument('--key', default='transaction_chunks')
    args = ap.parse_args()
    gt = parse_gt(args.gt)

    print(f'{"model":<26}{"maj size":>9}{"correct":>8}{"F1":>7}'
          f'{"voter size":>12}{"correct":>8}{"F1":>7}')
    for d in sorted(Path(args.results).iterdir()):
        runs_p, sel_p = d / 'runs.json', d / 'selected.json'
        if not (runs_p.exists() and sel_p.exists()):
            continue
        runs = [r for r in json.loads(runs_p.read_text())
                if r.get('meta', {}).get('parse') != 'failed']
        n = len(runs)
        counts = collections.Counter()
        for r in runs:
            counts.update(int(c) for c in r.get(args.key, []))

        maj = {c for c, k in counts.items() if k * 2 >= n}
        voter = set(json.loads(sel_p.read_text())['selected_chunks'])

        print(f'{d.name.replace("v3_", ""):<26}'
              f'{len(maj):>9}{len(maj & gt):>8}{f1(maj, gt):>7.3f}'
              f'{len(voter):>12}{len(voter & gt):>8}{f1(voter, gt):>7.3f}')


if __name__ == '__main__':
    main()
