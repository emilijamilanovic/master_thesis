#!/usr/bin/env python3
"""Predicted versus realised error of the delivered set, per configuration.

Reproduces the calibration tables of the thesis (Table 5.4 for the font
question, and its counterpart in the appendix for the table question).
For every configuration under --results it reads

    selected.json    the delivered set and the per-chunk posteriors
    mode_analysis.json (if present)  only to check the ground truth

and reports, for the delivered set S:

    size       |S|
    wrong      |S \\ G|, selected chunks that are not relevant
    missed     |G \\ S|, relevant chunks that were not delivered
    realised   |S \\ G| / |S|, the false discovery rate of Sec. 3.3
    predicted  mean over S of 1 - P(relevant | count), the error the
               voter predicts for its own selection

The two error columns come from different places: realised compares the
set with the labels, predicted uses only the counts and the two rates,
so the gap between them is what Sec. 5.4 discusses.

Realised error can be cross-checked without this script: it equals
1 - decision_rules.per_chunk_bayes_voter.vs_gt.precision in
mode_analysis.json, provided that file was produced with the same rates.

  python analysis_code/calibration_table.py \\
      --results results/output/q2_fonts \\
      --gt 200-201,214,224-229,236,239,250-252

  python analysis_code/calibration_table.py \\
      --results results/output/q1_tables --gt 391-419

--gt is checked against the ground truth recorded in mode_analysis.json,
so pointing the script at the wrong folder fails instead of printing
plausible numbers.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

PREFERRED = [
    'v3_openai_gpt41', 'v3_openai_gpt41mini', 'v3_openai_gpt4o',
    'v3_anthropic_haiku', 'v3_gemini_flash35', 'v3_gemini_flash36',
    'v3_fireworks_glm52_nores', 'v3_fireworks_kimi3_nores', 'v3_fireworks_qwen',
]


def parse_ids(spec):
    """'391-419' or '200-201,214,224-229' -> set of ints."""
    out = set()
    for part in str(spec).split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            lo, hi = part.split('-')
            out.update(range(int(lo), int(hi) + 1))
        else:
            out.add(int(part))
    return out


def order_key(name):
    return (PREFERRED.index(name), '') if name in PREFERRED else (len(PREFERRED), name)


def analyse(folder, gt):
    """One row, or None if the folder holds no selection."""
    selected_path = folder / 'selected.json'
    if not selected_path.exists():
        return None

    with open(selected_path) as fh:
        sel_json = json.load(fh)
    selected = set(sel_json['selected_chunks'])
    if not selected:
        print(f'  skip {folder.name}: the delivered set is empty', file=sys.stderr)
        return None

    mode_path = folder / 'mode_analysis.json'
    if mode_path.exists():
        with open(mode_path) as fh:
            recorded = parse_ids(json.load(fh)['ground_truth'])
        if recorded != gt:
            sys.exit(f'ERROR: {folder.name} was scored against a different ground truth '
                     f'({len(recorded)} chunks) than --gt ({len(gt)} chunks). '
                     f'Check that --results points at the intended question.')

    posterior = {s['chunk']: s['posterior'] for s in sel_json['scores']}
    missing = selected - set(posterior)
    if missing:
        sys.exit(f'ERROR: {folder.name}: no posterior recorded for chunk(s) {sorted(missing)}')

    wrong = sorted(selected - gt)
    missed = sorted(gt - selected)
    realised = len(wrong) / len(selected)
    predicted = sum(1 - posterior[c] for c in selected) / len(selected)
    budget = 1 - sel_json['threshold'] if sel_json.get('threshold') else None

    return {
        'config': folder.name,
        'n_runs': sel_json['n_runs'],
        'size': len(selected),
        'wrong': len(wrong),
        'missed': len(missed),
        'realised': round(realised, 4),
        'predicted': predicted,
        'over_budget': budget is not None and realised > budget,
        'budget': budget,
        'wrong_chunks': wrong,
        'missed_chunks': missed,
        'scored': len(posterior),
        'undecided': sum(1 for p in posterior.values() if 0.01 < p < 0.99),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--results', required=True,
                    help='Folder holding one subfolder per configuration')
    ap.add_argument('--gt', required=True,
                    help="Ground-truth chunk ids, e.g. '391-419'")
    ap.add_argument('-o', '--out', default=None,
                    help='Optional CSV to write the table to')
    args = ap.parse_args()

    gt = parse_ids(args.gt)
    results_dir = Path(args.results)
    if not results_dir.is_dir():
        sys.exit(f'ERROR: {results_dir} is not a directory')

    rows = [r for r in (analyse(d, gt) for d in sorted(results_dir.iterdir()) if d.is_dir())
            if r is not None]
    if not rows:
        sys.exit(f'ERROR: no selected.json found under {results_dir}')
    rows.sort(key=lambda r: order_key(r['config']))

    print(f'{results_dir}   ground truth: {len(gt)} chunks')
    print(f"{'config':<26}{'N':>5}{'size':>6}{'wrong':>7}{'missed':>8}"
          f"{'realised':>11}{'predicted':>12}  wrong chunks")
    for r in rows:
        flag = ' *' if r['over_budget'] else ''
        print(f"{r['config']:<26}{r['n_runs']:>5}{r['size']:>6}{r['wrong']:>7}{r['missed']:>8}"
              f"{r['realised']:>11.3f}{r['predicted']:>12.5f}  {r['wrong_chunks']}{flag}")

    scored = sum(r['scored'] for r in rows)
    undecided = sum(r['undecided'] for r in rows)
    over = sum(r['over_budget'] for r in rows)
    budgets = {r['budget'] for r in rows if r['budget'] is not None}
    budget = f'{budgets.pop():.2f}' if len(budgets) == 1 else 'the threshold'
    print(f'\n{scored} chunk-model pairs scored, {undecided} with posterior in (0.01, 0.99)')
    print(f'{over} of {len(rows)} configurations have a realised error above {budget} '
          f'(marked *)')

    if args.out:
        fields = ['config', 'n_runs', 'size', 'wrong', 'missed', 'realised', 'predicted',
                  'over_budget', 'wrong_chunks', 'missed_chunks']
        with open(args.out, 'w', newline='') as fh:
            writer = csv.DictWriter(fh, fieldnames=fields, extrasaction='ignore')
            writer.writeheader()
            for r in rows:
                writer.writerow({**r,
                                 'wrong_chunks': ' '.join(map(str, r['wrong_chunks'])),
                                 'missed_chunks': ' '.join(map(str, r['missed_chunks']))})
        print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
