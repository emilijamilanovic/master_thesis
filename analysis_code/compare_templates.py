#!/usr/bin/env python3
"""Statistics over the answer templates found by mode_analysis.py.

The two comparison scripts (compare_models_same_prompt.py,
compare_prompts_same_model.py) work from runs.json and describe per-chunk
behaviour. This one works from the mode_analysis.json produced in each result
folder and describes the *menu structure*: how many distinct whole answers a
configuration produces, how concentrated they are, how strongly chunks move
together, and how the alternative decision rules (modal / core / envelope /
Bayesian voter) compare.

It reads templates rather than recomputing them, so the clustering is exactly
the one the pipeline recorded (merge distance included). The ground truth is
whatever mode_analysis.py used at generation time; it is read back from each
file and a warning is printed if configurations disagree.

Usage:
    python3 analysis_code/compare_templates.py
    python3 analysis_code/compare_templates.py \
        --results results/output --out analysis/templates
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
def parse_chunk_spec(spec):
    """Inverse of mode_analysis.fmt_ids: '392, 394-396, 404' -> {392,394,...}.

    mode_analysis.py stores template contents as compact range strings, so they
    have to be expanded before templates can be compared across configurations.
    """
    ids = set()
    if not spec or spec.strip() == '-':
        return ids
    for part in spec.split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            lo, hi = part.split('-')
            ids.update(range(int(lo), int(hi) + 1))
        else:
            ids.add(int(part))
    return ids


def load_config(folder):
    """Load one folder's mode_analysis.json; None (with a note) if unusable."""
    m = FOLDER_RE.match(folder.name)
    if not m:
        return None

    path = folder / 'mode_analysis.json'
    if not path.exists():
        print(f'  skip {folder.name}: no mode_analysis.json '
              f'(was the pipeline run with --pipeline classic?)', file=sys.stderr)
        return None
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        print(f'  skip {folder.name}: mode_analysis.json is not valid JSON ({e})',
              file=sys.stderr)
        return None
    if not data.get('templates'):
        print(f'  skip {folder.name}: no templates recorded', file=sys.stderr)
        return None

    # The real model id lives in runs.json; fall back to the folder token.
    model = m.group('model')
    runs_path = folder / 'runs.json'
    if runs_path.exists():
        try:
            runs = json.loads(runs_path.read_text())
            model = runs[0].get('meta', {}).get('model') or model
        except (json.JSONDecodeError, IndexError, KeyError):
            pass

    return {
        'prompt': 'v' + m.group('prompt'),
        'provider': m.group('provider'),
        'model': display_name(model, m.group('model')),
        'mode': data,
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
# Per-configuration statistics
# ---------------------------------------------------------------------------
def jaccard(a, b):
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


def summarise(cfg):
    """One row per configuration: menu shape, dependence, decision rules."""
    d = cfg['mode']
    templates = d['templates']
    ic = d.get('independence_check', {})
    rules = d.get('decision_rules', {})

    modal = templates[0]
    second = templates[1] if len(templates) > 1 else None

    # Dependence: how much wider the per-run selection size is than an
    # independent-chunk model with the same marginals would predict.
    # Ratio is only meaningful when there is a varying block at all.
    std = ic.get('selection_size_std', {})
    obs, ind = std.get('observed', 0.0), std.get('independent_model', 0.0)
    ratio = (obs / ind) if ind > 0 else None

    row = {
        'prompt': cfg['prompt'],
        'provider': cfg['provider'],
        'model': cfg['model'],
        'n_runs': d.get('n_runs', 0),
        'ground_truth': d.get('ground_truth', ''),

        # --- menu shape ---
        'n_distinct_patterns': d.get('n_distinct_patterns', 0),
        'top_pattern_share': d.get('top_pattern_share', 0.0),
        'n_templates': d.get('n_templates', 0),
        'effective_n_modes': d.get('effective_n_modes', 0.0),
        'template_entropy_nats': d.get('template_entropy_nats', 0.0),
        'residual_flip_noise': d.get('residual_flip_noise_per_run', 0.0),
        'single_template': d.get('n_templates', 0) <= 1,
        'runs_to_discover_95pct': next(
            iter(d.get('runs_needed_to_discover_mass', {}).values()), None),

        # --- dominant answers ---
        'modal_weight': modal.get('weight', 0.0),
        'modal_ci_lo': (modal.get('weight_ci95') or [None, None])[0],
        'modal_ci_hi': (modal.get('weight_ci95') or [None, None])[1],
        'modal_size': modal.get('size', 0),
        'second_weight': second.get('weight') if second else 0.0,
        'second_size': second.get('size') if second else 0,

        # --- dependence between chunks ---
        'block_size': ic.get('block_size', 0),
        'block_mean_rate': ic.get('block_mean_inclusion_rate', 0.0),
        'size_std_observed': obs,
        'size_std_independent': ind,
        'size_std_ratio': ratio if ratio is not None else '',
        'P_all_block_observed': ic.get('P_all_block_together', {}).get('observed', ''),
        'P_all_block_independent': ic.get('P_all_block_together', {}).get('independent_model', ''),
    }

    # --- decision rules vs ground truth ---
    for name, key in (('modal', 'modal_template'),
                      ('core', 'core_intersection'),
                      ('envelope', 'envelope_union'),
                      ('voter', 'per_chunk_bayes_voter')):
        r = rules.get(key, {})
        g = r.get('vs_gt', {})
        row[f'{name}_size'] = r.get('size', 0)
        row[f'{name}_precision'] = g.get('precision', 0.0)
        row[f'{name}_recall'] = g.get('recall', 0.0)
        row[f'{name}_f1'] = g.get('f1', 0.0)

    # Template contents, expanded, for cross-configuration comparison
    row['_templates'] = [(t.get('weight', 0.0), parse_chunk_spec(t.get('chunks', '')))
                         for t in templates]
    row['_modal_set'] = row['_templates'][0][1] if row['_templates'] else set()
    return row


# ---------------------------------------------------------------------------
# Output helpers
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


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_menu_size(rows, prompt, out):
    """Effective number of modes per model, for one prompt version.

    1.0 means the configuration always returned (essentially) the same answer,
    so a flat row of bars at 1.0 is itself the result: no menu structure.
    """
    rows = sorted(rows, key=lambda r: -r['effective_n_modes'])
    models = [r['model'] for r in rows]
    fig, ax = plt.subplots(figsize=(max(7, len(rows) * 0.85), 4.5))
    ax.bar(range(len(rows)), [r['effective_n_modes'] for r in rows],
           color='#4C72B0')
    ax.axhline(1.0, color='grey', linestyle='--', linewidth=1,
               label='single answer')
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(models, rotation=40, ha='right', fontsize=8)
    ax.set_ylabel('effective number of answer modes')
    ax.set_title(f'Menu size by model — prompt {prompt}\n'
                 '(exp of template-weight entropy; 1.0 = one recurring answer)')
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / f'menu_size_{prompt}.png', dpi=150)
    plt.close(fig)


def plot_dependence(rows, prompt, out):
    """Observed vs independence-predicted spread of the selection size.

    Only configurations with a varying block are informative; a prompt where
    every model returned one fixed answer produces no plot at all.
    """
    live = [r for r in rows if r['block_size'] > 0]
    if not live:
        return False
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    lim = max(max(r['size_std_observed'] for r in live),
              max(r['size_std_independent'] for r in live)) * 1.15
    ax.plot([0, lim], [0, lim], color='grey', linestyle='--', linewidth=1,
            label='independence (y = x)')
    for r in live:
        ax.scatter(r['size_std_independent'], r['size_std_observed'], s=70,
                   color='#C44E52', edgecolor='black', linewidth=0.5, zorder=3)
        ax.annotate(r['model'],
                    (r['size_std_independent'], r['size_std_observed']),
                    textcoords='offset points', xytext=(6, 3), fontsize=7)
    ax.set_xlabel('selection-size std predicted by independent chunks')
    ax.set_ylabel('selection-size std observed')
    ax.set_title(f'Chunk dependence — prompt {prompt}\n'
                 'points above the line: chunks move together, not independently')
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / f'dependence_{prompt}.png', dpi=150)
    plt.close(fig)
    return True


def plot_rule_comparison(rows, prompt, out):
    """Precision / recall / F1 of each decision rule for one prompt version,
    averaged over models. Shows the precision-recall trade-off between the
    core (intersection) and envelope (union) rules."""
    rules = ['modal', 'core', 'envelope', 'voter']
    metrics = [('precision', '#4C72B0'), ('recall', '#DD8452'), ('f1', '#55A868')]
    width = 0.8 / len(metrics)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, (metric, colour) in enumerate(metrics):
        xs = [j + i * width for j in range(len(rules))]
        ys = [mean(r[f'{rule}_{metric}'] for r in rows) for rule in rules]
        ax.bar(xs, ys, width, label=metric, color=colour)
    ax.set_xticks([j + 0.4 - width / 2 for j in range(len(rules))])
    ax.set_xticklabels(rules)
    ax.set_ylabel('score vs ground truth (mean over models)')
    ax.set_title(f'Aggregation rules — prompt {prompt}')
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / f'rule_comparison_{prompt}.png', dpi=150)
    plt.close(fig)


def plot_modal_agreement(rows, prompt, out):
    """Do different models converge on the same answer? Pairwise Jaccard
    between the models' modal templates, for one prompt version."""
    group = sorted(rows, key=lambda r: r['model'])
    if len(group) < 2:
        return
    labels = [r['model'] for r in group]
    grid = [[jaccard(a['_modal_set'], b['_modal_set']) for b in group]
            for a in group]

    fig, ax = plt.subplots(figsize=(1.0 + 0.55 * len(group),
                                    0.9 + 0.5 * len(group)))
    im = ax.imshow(grid, cmap='viridis', vmin=0, vmax=1)
    ax.set_xticks(range(len(group)), labels, rotation=90, fontsize=6)
    ax.set_yticks(range(len(group)), labels, fontsize=6)
    ax.set_title(f'Agreement between models\' modal answers — {prompt}',
                 fontsize=9)
    fig.colorbar(im, ax=ax, label='Jaccard', shrink=0.8)
    fig.tight_layout()
    fig.savefig(out / f'modal_agreement_{prompt}.png', dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description='Statistics over the answer templates from mode_analysis.py.')
    ap.add_argument('--results', default='results/output',
                    help='Folder containing the v<N>_<provider>_<model> dirs')
    ap.add_argument('--out', default='results/analysis/templates',
                    help='Where to write CSVs and plots')
    ap.add_argument('--exclude-provider', action='append', default=[],
                    help='Provider to exclude; repeatable (default: fireworks)')
    args = ap.parse_args()

    results_dir = Path(args.results)
    if not results_dir.is_dir():
        sys.exit(f'ERROR: results folder not found: {results_dir}')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f'Loading from {results_dir} (excluding: {", ".join(args.exclude_provider)})')
    configs = discover(results_dir, set(args.exclude_provider))
    if not configs:
        sys.exit('ERROR: no usable mode_analysis.json files found. Expected '
                 'directories named like v3_openai_gpt4o.')

    rows = [summarise(c) for c in configs]
    prompts = sorted({r['prompt'] for r in rows})

    # The templates were scored against whatever GT the pipeline used; mixing
    # configurations scored against different keys would be misleading.
    gts = {r['ground_truth'] for r in rows if r['ground_truth']}
    if len(gts) > 1:
        print(f'WARNING: configurations were scored against different ground '
              f'truths: {", ".join(sorted(gts))}. Decision-rule columns are '
              f'not comparable across them.', file=sys.stderr)

    print(f'\n{len(rows)} configurations, ground truth '
          f'{", ".join(sorted(gts)) or "?"}')

    n_files = 0
    all_rows, all_catalog, all_rules = [], [], []

    # Everything below is produced once per prompt version, so the prompts can
    # be inspected independently; the *_all.csv files keep the pooled view.
    for prompt in prompts:
        group = sorted([r for r in rows if r['prompt'] == prompt],
                       key=lambda r: -r['effective_n_modes'])
        all_rows.extend(group)

        # --- CSV 1: one row per configuration -----------------------------
        write_csv(out / f'template_summary_{prompt}.csv', group)

        # --- CSV 2: the full menu, one row per template -------------------
        catalog = []
        for r in group:
            for rank, (w, chunks) in enumerate(r['_templates'], 1):
                catalog.append({
                    'prompt': r['prompt'], 'model': r['model'], 'rank': rank,
                    'weight': w, 'size': len(chunks),
                    'chunks': ','.join(str(c) for c in sorted(chunks)),
                })
        write_csv(out / f'template_catalog_{prompt}.csv', catalog)
        all_catalog.extend(catalog)

        # --- CSV 3: cross-model agreement of modal answers ----------------
        agreement = [{
            'prompt': prompt, 'model_a': a['model'], 'model_b': b['model'],
            'jaccard': jaccard(a['_modal_set'], b['_modal_set']),
            'identical': a['_modal_set'] == b['_modal_set'],
        } for a, b in combinations(group, 2)]
        write_csv(out / f'modal_agreement_{prompt}.csv', agreement)

        # --- CSV 4: decision rules averaged over models -------------------
        rule_rows = [{
            'prompt': prompt, 'rule': rule, 'n_configs': len(group),
            'size_mean': mean(r[f'{rule}_size'] for r in group),
            'precision_mean': mean(r[f'{rule}_precision'] for r in group),
            'recall_mean': mean(r[f'{rule}_recall'] for r in group),
            'f1_mean': mean(r[f'{rule}_f1'] for r in group),
        } for rule in ('modal', 'core', 'envelope', 'voter')]
        write_csv(out / f'rule_comparison_{prompt}.csv', rule_rows)
        all_rules.extend(rule_rows)

        # --- plots --------------------------------------------------------
        plot_menu_size(group, prompt, out)
        plot_rule_comparison(group, prompt, out)
        plot_modal_agreement(group, prompt, out)
        has_dep = plot_dependence(group, prompt, out)
        n_files += 4 + 3 + (1 if has_dep else 0)

        # --- terminal section for this prompt -----------------------------
        n_single = sum(1 for r in group if r['single_template'])
        print(f'\n=== prompt {prompt} — {len(group)} models '
              f'({group[0]["n_runs"]} runs each) ===')
        print(f'  {n_single}/{len(group)} produced a single answer template, '
              f'{len(group) - n_single} a menu of two or more')
        print(f'  mean effective modes {mean(r["effective_n_modes"] for r in group):.2f}'
              f' | mean distinct raw patterns {mean(r["n_distinct_patterns"] for r in group):.1f}'
              f' | mean flip noise {mean(r["residual_flip_noise"] for r in group):.2f}')

        multi = [r for r in group if not r['single_template']]
        if multi:
            print(f'  {"model":<26}{"tmpl":>5}{"eff":>6}{"modal w":>9}'
                  f'{"block":>7}{"std obs":>9}{"std ind":>9}{"ratio":>7}')
            for r in sorted(multi, key=lambda r: -(r['size_std_ratio'] or 0)):
                ratio = r['size_std_ratio']
                print(f'  {r["model"]:<26}{r["n_templates"]:>5}'
                      f'{r["effective_n_modes"]:>6.2f}{r["modal_weight"]:>9.2f}'
                      f'{r["block_size"]:>7}{r["size_std_observed"]:>9.2f}'
                      f'{r["size_std_independent"]:>9.2f}'
                      f'{(ratio if isinstance(ratio, float) else 0):>7.2f}')
            print('  (ratio > 1: chunks move in blocks rather than independently)')
        else:
            print('  no configuration produced a menu, so there is no '
                  'dependence structure to measure')

        print(f'  aggregation rules (mean F1): '
              f'modal {mean(r["modal_f1"] for r in group):.3f}, '
              f'core {mean(r["core_f1"] for r in group):.3f}, '
              f'envelope {mean(r["envelope_f1"] for r in group):.3f}, '
              f'voter {mean(r["voter_f1"] for r in group):.3f}')

        if agreement:
            ident = sum(1 for a in agreement if a['identical'])
            distinct = len({frozenset(r['_modal_set']) for r in group})
            print(f'  cross-model convergence: {ident}/{len(agreement)} model '
                  f'pairs identical, {distinct} distinct modal answers '
                  f'(mean Jaccard {mean(a["jaccard"] for a in agreement):.3f})')

    # ---- pooled views across prompts --------------------------------------
    write_csv(out / 'template_summary_all.csv', all_rows)
    write_csv(out / 'template_catalog_all.csv', all_catalog)
    write_csv(out / 'rule_comparison_all.csv', all_rules)

    print(f'\nWrote {n_files + 3} files to {out}/')
    print('  per prompt: template_summary / template_catalog / modal_agreement '
          '/ rule_comparison (.csv)')
    print('  per prompt: menu_size / rule_comparison / modal_agreement '
          '(+ dependence where a menu exists) (.png)')
    print('  pooled    : template_summary_all / template_catalog_all / '
          'rule_comparison_all (.csv)')


if __name__ == '__main__':
    main()
