# Answer-mode ("menu") analysis — `mode_analysis.py`

```bash
python pipeline/mode_analysis.py \
    results/output/q1_tables/v3_openai_gpt41/runs.json \
    -o results/output/q1_tables/v3_openai_gpt41/mode_analysis.json
```

## The model

A run samples template `T` with probability `π_T` from a small menu
`{T_1 … T_M}` (each template = a fixed chunk set), then outputs `T` with a few
random chunk flips (noise rate `ε`). Everything about the menu — templates,
weights, noise — is estimable **without ground truth**; labels are only needed
to judge *which templates are right*. This cleanly separates sampling noise
(`π`, `ε`) from systematic model error (template contents vs truth).

## What the script does

1. **Menu (step 1).** Canonicalize each run to a chunk set; count identical
   patterns.
2. **Templates (step 2).** Greedy absorption: walk patterns in descending
   frequency; merge a pattern into the most frequent existing template within
   Hamming distance `--merge-dist` (default 3), else open a new template.
   Template centroid = per-chunk majority over member runs. (A proper
   Bernoulli-mixture EM with BIC model selection is the planned upgrade; the
   greedy version is deterministic and transparent.)
3. **Statistics (step 3).** Template weights `π` with bootstrap 95% CIs;
   residual flip noise per run; entropy of `π` (effective number of modes);
   analytic estimate of how many runs are needed to *discover* templates
   covering `--mass` (default 95%) of probability.
4. **Decisions + independence check (step 4).** Four selection rules evaluated
   against ground truth (`--gt`, default 391-419): modal template, **core**
   (intersection of major templates), **envelope** (union of major templates),
   and the per-chunk Bayesian voter (imported from `voter_bayesian.py`, same
   defaults as the pipeline). Then the independence check: observed
   co-inclusion of the envelope-minus-core block, and observed vs
   independence-simulated standard deviation of the per-run selection size.

