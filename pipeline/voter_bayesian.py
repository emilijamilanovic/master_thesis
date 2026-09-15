import json
import argparse
import math
from collections import Counter, defaultdict

def log_binom_pmf(k: int, n: int, p: float) -> float:
    """log[ C(n,k) * p^k * (1-p)^(n-k) ]"""
    if p <= 0.0:
        return 0.0 if k == 0 else -math.inf
    if p >= 1.0:
        return 0.0 if k == n else -math.inf
    # log C(n,k) = lgamma(n+1) - lgamma(k+1) - lgamma(n-k+1)
    return (math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
            + k * math.log(p) + (n - k) * math.log(1.0 - p))

def posterior_relevance(k: int, n: int, p_correct: float, p_noise: float, prior: float) -> float:
    """
    Compute P(H1 | k) with:
      H1: Binomial(n, p_correct)
      H0: Binomial(n, p_noise)
    Using Bayes: logit posterior = logit(prior) + log LR
    """
    eps = 1e-12
    p_correct = min(max(p_correct, eps), 1 - eps)
    p_noise   = min(max(p_noise,   eps), 1 - eps)
    prior     = min(max(prior,     eps), 1 - eps)

    ll1 = log_binom_pmf(k, n, p_correct)
    ll0 = log_binom_pmf(k, n, p_noise)
    log_lr = ll1 - ll0

    logit_prior = math.log(prior) - math.log(1 - prior)
    logit_post  = logit_prior + log_lr

    # Numerically stable logistic. The two forms
    #     1/(1+exp(-x))   and   exp(x)/(1+exp(x))
    # are algebraically identical (multiply the first by exp(x)/exp(x)), but
    # only one is computable at each extreme: exp() of a large positive number
    # raises OverflowError, while exp() of a large negative number underflows
    # harmlessly to 0. Branching on the sign always exponentiates a
    # non-positive value. Needed when p_correct is near 1, where a chunk
    # missing from even a few runs gives a hugely negative logit.
    if logit_post >= 0:
        return 1.0 / (1.0 + math.exp(-logit_post))
    z = math.exp(logit_post)
    return z / (1.0 + z)

def aggregate_counts(runs, key: str):

    counts = Counter()
    n_runs = len(runs)
    for r in runs:
        lst = r.get(key, [])
        if not isinstance(lst, list):
            raise TypeError(f"Run {r.get('run')}: '{key}' must be a list.")
        try:
            picked = set(int(x) for x in lst)
        except Exception as e:
            raise TypeError(f"Run {r.get('run')}: non-integer in '{key}': {e}")
        counts.update(picked)
    return n_runs, counts

def vote_chunks(runs_json_path: str,
                key: str,
                p_correct: float,
                p_noise: float,
                prior: float = 0.5,
                threshold: float = 0.9,
                top_k: int | None = None,
                return_scores: bool = True):
    """
    - Compute posterior P(relevant | count) for every chunk id.
    - Select those with posterior >= threshold.
    """
    with open(runs_json_path, "r", encoding="utf-8") as f:
        runs = json.load(f)

    if not isinstance(runs, list) or not runs:
        raise ValueError("Runs JSON must be a non-empty list.")

    n_runs, counts = aggregate_counts(runs, key)
    results = []
    for cid, c in counts.items():
        post = posterior_relevance(c, n_runs, p_correct, p_noise, prior)
        results.append({"chunk": cid, "count": c, "posterior": post})

    # sort by posterior desc, then by count desc, then by chunk id asc (stable)
    results.sort(key=lambda d: (-d["posterior"], -d["count"], d["chunk"]))

    # apply threshold
    if threshold is not None:
        filtered = [d for d in results if d["posterior"] >= threshold]

    else:
        filtered = results

    # top_k cap
    # if top_k is not None and top_k < len(filtered):
    #     filtered = filtered[:top_k]

    selected = [d["chunk"] for d in filtered]

    out = {
        "n_runs": n_runs,
        "key": key,
        "p_correct": p_correct,
        "p_noise": p_noise,
        "prior": prior,
        "threshold": threshold,
        "selected_chunks": selected,
    }
    if return_scores:
        out["scores"] = results 
    return out

def main():
    ap = argparse.ArgumentParser(description="Bayesian voter for chunk selection from multiple runs.")
    ap.add_argument("runs_json", help="Path to runs JSON (list of dicts).")
    ap.add_argument("--key", default="transaction_chunks", help="Key with predicted chunk list. Default: transaction_chunks")
    ap.add_argument("--p-correct", type=float, required=True, help="Per-run inclusion prob for a truly relevant chunk.")
    ap.add_argument("--p-noise",   type=float, required=True, help="Per-run inclusion prob for an irrelevant chunk.")
    # 0.5 (uninformative) matches run_pipeline.py and mode_analysis.py, so a
    # direct invocation reproduces what the pipeline did. At the run counts
    # used here the likelihood ratio dominates the prior anyway: the count
    # decision boundary moves ~0.2 counts per 100 runs between prior 0.3 and
    # 0.7. The base rate (|GT|/|slice|, about 0.17) is the other defensible
    # choice. See the note in the root README.
    ap.add_argument("--prior",     type=float, default=0.5,   help="Prior probability a random chunk is relevant. Default: 0.5")
    ap.add_argument("--threshold", type=float, default=0.90,  help="Posterior threshold to accept a chunk. Default: 0.90")
    ap.add_argument("--top-k",     type=int, default=None,    help="Optional cap on number of chunks to keep (after threshold).")
    ap.add_argument("-o", "--out", help="Path to write JSON result. If omitted, prints to stdout.")
    args = ap.parse_args()

    result = vote_chunks(
        runs_json_path=args.runs_json,
        key=args.key,
        p_correct=args.p_correct,
        p_noise=args.p_noise,
        prior=args.prior,
        threshold=args.threshold,
        top_k=args.top_k,
        return_scores=True,
    )

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"Wrote voted selection to {args.out}")
    else:
        print(json.dumps(result, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    import json
    main()
