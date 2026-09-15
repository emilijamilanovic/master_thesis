import json
import argparse
from statistics import mean

# IDs follow the pipeline numbering (Source.fulltext() -> Chunker, 754 chunks),
# the same numbering the LLM sees in the prompt.
# Range covers the whole "Tables" section: 391 is the section heading, 392 the
# sentence introducing the four table kinds, 393-394 the table_captions
# extension, and 395-419 the four format subsections (simple, multiline, grid,
# pipe). The task is to cite EVERY chunk describing a table format, so the
# section intro and caption material are included.

GROUND_TRUTH_IDS = set(range(391, 420))

# GROUND_TRUTH_IDS = set([200, 201, 214, 224, 225, 226, 227, 228, 229, 236, 239, 250, 251, 252])

def evaluate_runs(runs, gt: set[int], total_chunks: int, key: str):

    G = len(gt)
    U = total_chunks - G

    per_run = []
    total_tp = 0
    total_fp = 0

    for r in runs:
        preds = r.get(key, [])
        P = set(int(x) for x in preds)

        tp = len(P & gt)
        fp = len(P - gt)
        fn = G - tp
        tn = max(0, U - fp)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        noise_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        per_run.append({
            "run": r.get("run"),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision,
            "recall": recall,
            "noise_rate": noise_rate,
            "f1": f1,
        })

        total_tp += tp
        total_fp += fp

    n_runs = len(runs)

    p_correct_hat = total_tp / (n_runs * G) if G > 0 else 0.0
    p_noise_hat   = total_fp / (n_runs * U) if U > 0 else 0.0

    mean_recall     = mean([pr["recall"] for pr in per_run]) if per_run else 0.0
    mean_noise_rate = mean([pr["noise_rate"] for pr in per_run]) if per_run else 0.0

    summary = {
        "n_runs": n_runs,
        "total_chunks": total_chunks,
        "gt_size": G,
        "noise_universe_size": U,

        "total_tp": total_tp,
        "total_fp": total_fp,

        "p_correct_hat": p_correct_hat, 
        "p_noise_hat": p_noise_hat,

        "mean_recall": mean_recall,
        "mean_noise_rate": mean_noise_rate, 

        "per_run": per_run, 
    }
    return summary

def main():
    ap = argparse.ArgumentParser(description="Per-run metrics and pooled estimates of p_correct and p_noise.")
    ap.add_argument("runs_json", help="Path to runs JSON (list of dicts).")
    ap.add_argument("--key", default="transaction_chunks", help="Key containing predicted chunk ids (default: transaction_chunks).")
    ap.add_argument("--total_chunks", type=int, required=True, help="Total number of chunks in the document.")
    ap.add_argument("-o", "--out", help="Path to write JSON summary. If omitted, prints to stdout.")
    args = ap.parse_args()

    with open(args.runs_json, "r", encoding="utf-8") as f:
        runs = json.load(f)

    gt = set(GROUND_TRUTH_IDS)

    summary = evaluate_runs(runs, gt, args.total_chunks, args.key)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"Wrote summary to {args.out}")
    else:
        print(json.dumps(summary, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    import json
    main()
