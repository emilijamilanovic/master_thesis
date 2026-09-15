"""Answer-mode ("menu") analysis of repeated LLM chunk-selection runs.

Motivation: the per-chunk binomial voter treats every chunk as an independent
coin flip across runs. Inspection of the 500-run Pandoc data shows this is
false: runs concentrate on a handful of recurring whole answers (one pattern
covers ~51% of runs) and blocks of chunks move in and out together. This
script makes that structure explicit and quantifies what the independence
assumption gets wrong. See mode_analysis.md for the full write-up.

Pipeline (steps 1-4 of the prototype):
  1. Group runs by identical selection set -> raw pattern menu.
  2. Merge near-duplicate patterns (small Hamming distance) into templates;
     the template centroid is the per-chunk majority over its member runs.
  3. Report template weights pi (with bootstrap CIs), residual flip noise,
     entropy, and how many runs are needed to discover the major templates.
  4. Compare decision rules (modal / core / envelope / per-chunk Bayesian
     voter) against ground truth, and test the independence assumption by
     comparing observed co-inclusion behaviour with what independent
     per-chunk sampling would predict.

Usage:
  python pipeline/mode_analysis.py results/output/output_chunks_pandoc_500.json \
      -o results/output/mode_analysis.json
"""

import argparse
import json
import math
import random
from collections import Counter

from voter_bayesian import posterior_relevance


# ****************************************************************************
# Loading and formatting
# ****************************************************************************
def load_runs(path, key):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    runs = [frozenset(int(x) for x in r.get(key, [])) for r in data]
    return [r for r in runs if r]  # drop empty selections (parse failures)


def parse_gt(spec):
    """Parse a ground-truth spec into a set of chunk ids.

    Accepts a range ('391-419'), a list ('200,201,214'), or a mix
    ('200-201,214,224-229'), so that questions whose answer is not a single
    contiguous block can be scored.
    """
    ids = set()
    for part in spec.split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            lo, hi = part.split('-')
            ids.update(range(int(lo), int(hi) + 1))
        else:
            ids.add(int(part))
    if not ids:
        raise ValueError(f'--gt parsed to an empty set: {spec!r}')
    return frozenset(ids)


def fmt_ids(ids):
    """Compact range formatting: {392,394,395,396,404} -> '392, 394-396, 404'."""
    ids = sorted(ids)
    if not ids:
        return "-"
    parts, lo, hi = [], ids[0], ids[0]
    for x in ids[1:]:
        if x == hi + 1:
            hi = x
        else:
            parts.append(f"{lo}-{hi}" if hi > lo else str(lo))
            lo = hi = x
    parts.append(f"{lo}-{hi}" if hi > lo else str(lo))
    return ", ".join(parts)


# ****************************************************************************
# Step 1-2: pattern menu and template merging
# ****************************************************************************
def greedy_templates(runs, merge_dist):
    """Cluster identical patterns, then absorb each pattern into the most
    frequent template within Hamming distance merge_dist. Centroids are
    recomputed as the per-chunk majority over member runs."""
    counts = Counter(runs)
    patterns = sorted(counts.items(), key=lambda kv: -kv[1])

    clusters = []  # list of [seed_pattern, {pattern: count}]
    for pat, c in patterns:
        best, best_d = None, merge_dist + 1
        for cl in clusters:
            d = len(pat ^ cl[0])
            if d < best_d:
                best, best_d = cl, d
        if best is None:
            clusters.append([pat, {pat: c}])
        else:
            best[1][pat] = c

    templates = []
    for seed, members in clusters:
        n = sum(members.values())
        votes = Counter()
        for pat, c in members.items():
            for ch in pat:
                votes[ch] += c
        centroid = frozenset(ch for ch, v in votes.items() if v * 2 >= n)
        templates.append({"centroid": centroid, "members": members, "count": n})
    templates.sort(key=lambda t: -t["count"])
    return templates


def assign_and_noise(runs, templates):
    """Assign each run to its nearest centroid; return per-template counts and
    the residual flip noise (mean Hamming distance between run and template)."""
    cents = [t["centroid"] for t in templates]
    counts = [0] * len(cents)
    flips = 0
    for r in runs:
        j = min(range(len(cents)), key=lambda k: len(r ^ cents[k]))
        counts[j] += 1
        flips += len(r ^ cents[j])
    return counts, flips / len(runs)


# ****************************************************************************
# Step 3: statistics on template weights
# ****************************************************************************
def bootstrap_pi(runs, templates, boots, seed=0):
    cents = [t["centroid"] for t in templates]
    rng = random.Random(seed)
    n = len(runs)
    samples = [[] for _ in cents]
    for _ in range(boots):
        counts = [0] * len(cents)
        for _ in range(n):
            r = runs[rng.randrange(n)]
            j = min(range(len(cents)), key=lambda k: len(r ^ cents[k]))
            counts[j] += 1
        for j, c in enumerate(counts):
            samples[j].append(c / n)
    ci = []
    for s in samples:
        s.sort()
        ci.append((s[int(0.025 * boots)], s[int(0.975 * boots)]))
    return ci


def discovery_runs(pi, mass_target):
    """Smallest n such that the expected discovered template mass
    sum_t pi_t * (1 - (1-pi_t)^n) reaches mass_target."""
    for n in range(1, 10001):
        if sum(p * (1 - (1 - p) ** n) for p in pi) >= mass_target:
            return n
    return None


# ****************************************************************************
# Step 4a: decision rules and ground-truth evaluation
# ****************************************************************************
def evaluate(selected, gt):
    tp = len(selected & gt)
    fp = len(selected - gt)
    fn = len(gt - selected)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": round(prec, 3), "recall": round(rec, 3), "f1": round(f1, 3)}


def bayesian_voter_selection(runs, p_correct, p_noise, prior, threshold):
    n = len(runs)
    counts = Counter()
    for r in runs:
        counts.update(r)
    return frozenset(c for c, k in counts.items()
                     if posterior_relevance(k, n, p_correct, p_noise, prior) >= threshold)


# ****************************************************************************
# Step 4b: independence check (calibration of the binomial view)
# ****************************************************************************
def independence_check(runs, block, sims, seed=1):
    """Observed distribution of |run ∩ block| vs Binomial(|block|, mean rate),
    plus observed vs independence-simulated std of the run's selection size."""
    n = len(runs)
    rates = {}
    universe = set().union(*runs)
    for ch in universe:
        rates[ch] = sum(1 for r in runs if ch in r) / n

    b = len(block)
    mean_rate = sum(rates[ch] for ch in block) / b if b else 0.0
    obs = Counter(len(r & block) for r in runs)
    obs_all = obs.get(b, 0) / n
    obs_none = obs.get(0, 0) / n
    pred_all = mean_rate ** b
    pred_none = (1 - mean_rate) ** b

    sizes = [len(r) for r in runs]
    mu = sum(sizes) / n
    obs_std = math.sqrt(sum((s - mu) ** 2 for s in sizes) / n)

    rng = random.Random(seed)
    sim_sizes = []
    rate_items = list(rates.items())
    for _ in range(sims):
        sim_sizes.append(sum(1 for _, p in rate_items if rng.random() < p))
    smu = sum(sim_sizes) / sims
    sim_std = math.sqrt(sum((s - smu) ** 2 for s in sim_sizes) / sims)

    return {
        "block": fmt_ids(block), "block_size": b,
        "block_mean_inclusion_rate": round(mean_rate, 3),
        "P_all_block_together": {"observed": round(obs_all, 3), "independent_model": round(pred_all, 6)},
        "P_no_block_at_all": {"observed": round(obs_none, 3), "independent_model": round(pred_none, 6)},
        "selection_size_std": {"observed": round(obs_std, 2), "independent_model": round(sim_std, 2)},
    }


# ****************************************************************************
# Main
# ****************************************************************************
def main():
    ap = argparse.ArgumentParser(description="Answer-mode (menu) analysis of repeated selection runs.")
    ap.add_argument("runs_json", help="Path to runs JSON (list of dicts).")
    ap.add_argument("--key", default="transaction_chunks", help="Key with predicted chunk list.")
    ap.add_argument("--gt", default="391-419",
                    help="Ground-truth ids: a range ('391-419'), a list "
                         "('200,201,214'), or a mix ('200-201,214,224-229'). "
                         "Default: 391-419")
    ap.add_argument("--merge-dist", type=int, default=3,
                    help="Max Hamming distance for absorbing a pattern into a template. Default: 3")
    ap.add_argument("--mass", type=float, default=0.95,
                    help="Cumulative template mass defining the 'major' templates. Default: 0.95")
    ap.add_argument("--boots", type=int, default=1000, help="Bootstrap resamples for pi CIs.")
    ap.add_argument("--sims", type=int, default=10000, help="Simulations for the independence check.")
    ap.add_argument("--p-correct", type=float, default=0.614)
    ap.add_argument("--p-noise", type=float, default=0.0124)
    ap.add_argument("--prior", type=float, default=0.5)
    ap.add_argument("--threshold", type=float, default=0.9)
    ap.add_argument("-o", "--out", help="Path to write JSON result.")
    args = ap.parse_args()

    gt = parse_gt(args.gt)

    runs = load_runs(args.runs_json, args.key)
    n = len(runs)
    raw_patterns = Counter(runs)

    # Steps 1-2: menu and templates
    templates = greedy_templates(runs, args.merge_dist)
    counts, flip_noise = assign_and_noise(runs, templates)
    pi = [c / n for c in counts]
    order = sorted(range(len(templates)), key=lambda j: -counts[j])
    templates = [templates[j] for j in order]
    counts = [counts[j] for j in order]
    pi = [pi[j] for j in order]

    # Step 3: statistics
    ci = bootstrap_pi(runs, templates, args.boots)
    entropy = -sum(p * math.log(p) for p in pi if p > 0)
    major, mass = [], 0.0
    for j, p in enumerate(pi):
        major.append(j)
        mass += p
        if mass >= args.mass:
            break
    n_discover = discovery_runs(pi, args.mass)

    # Step 4a: decision rules
    modal = templates[0]["centroid"]
    core = frozenset.intersection(*(templates[j]["centroid"] for j in major))
    envelope = frozenset.union(*(templates[j]["centroid"] for j in major))
    voter = bayesian_voter_selection(runs, args.p_correct, args.p_noise, args.prior, args.threshold)

    rules = {
        "modal_template": modal, "core_intersection": core,
        "envelope_union": envelope, "per_chunk_bayes_voter": voter,
    }

    # Step 4b: independence check on the largest coherent block (envelope - core)
    block = envelope - core
    indep = independence_check(runs, block, args.sims)

    result = {
        "n_runs": n,
        "n_distinct_patterns": len(raw_patterns),
        "top_pattern_share": round(raw_patterns.most_common(1)[0][1] / n, 3),
        "merge_dist": args.merge_dist,
        "n_templates": len(templates),
        "residual_flip_noise_per_run": round(flip_noise, 2),
        "template_entropy_nats": round(entropy, 3),
        "effective_n_modes": round(math.exp(entropy), 2),
        "runs_needed_to_discover_mass": {str(args.mass): n_discover},
        "templates": [
            {"rank": j + 1, "weight": round(pi[j], 3),
             "weight_ci95": [round(ci[j][0], 3), round(ci[j][1], 3)],
             "n_runs": counts[j], "size": len(t["centroid"]),
             "chunks": fmt_ids(t["centroid"])}
            for j, t in enumerate(templates) if counts[j] > 0
        ],
        "decision_rules": {
            name: {"size": len(sel), "chunks": fmt_ids(sel), "vs_gt": evaluate(sel, gt)}
            for name, sel in rules.items()
        },
        "independence_check": indep,
        "ground_truth": fmt_ids(gt),
    }

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"Wrote mode analysis to {args.out}")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
