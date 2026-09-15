#!/bin/bash
# Re-run mode_analysis.py with each configuration's own estimated rates.
# Without --p-correct/--p-noise the script falls back to generic defaults
# (0.614 / 0.0124), which is what produced the gemini-3.5-flash discrepancy.
set -e

run_dir () {                      # $1 = results dir, $2 = ground truth spec
  for d in "$1"/v3_*/; do
    P1=$(python3 -c "import json;print(json.load(open('$d/stats.json'))['p_correct_hat'])")
    P0=$(python3 -c "import json;print(json.load(open('$d/stats.json'))['p_noise_hat'])")
    echo "=== $d  p_correct=$P1  p_noise=$P0"
    python3 pipeline/mode_analysis.py "$d/runs.json" --gt "$2" --p-correct "$P1" --p-noise "$P0" --prior 0.5 --threshold 0.9 -o "$d/mode_analysis.json"
  done
}

run_dir results/output/q2_fonts "200-201,214,224-229,236,239,250-252"
run_dir results/output/q1_tables "391-419"
