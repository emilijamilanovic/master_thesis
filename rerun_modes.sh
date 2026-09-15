#!/bin/bash
# Re-run mode_analysis.py with each configuration's own estimated rates.
# Without --p-correct/--p-noise the script falls back to generic defaults
# (0.614 / 0.0124), which is what produced the gemini-3.5-flash discrepancy.
set -e

run_dir () {                      # $1 = results dir, $2 = ground truth spec
  for d in "$1"/v3_*/; do
    P1=$(.venv/bin/python -c "import json;print(json.load(open('$d/stats.json'))['p_correct_hat'])")
    P0=$(.venv/bin/python -c "import json;print(json.load(open('$d/stats.json'))['p_noise_hat'])")
    echo "=== $d  p_correct=$P1  p_noise=$P0"
    .venv/bin/python chunking_tests/mode_analysis.py "$d/runs.json" --gt "$2" --p-correct "$P1" --p-noise "$P0" --prior 0.5 --threshold 0.9 -o "$d/mode_analysis.json"
  done
}

run_dir chunking_tests/output/q2_fonts "200-201,214,224-229,236,239,250-252"
run_dir chunking_tests/output/budget_n100 "391-419"
