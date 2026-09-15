# Answer-mode ("menu") analysis — `mode_analysis.py`

Prototype for modeling repeated LLM chunk-selection runs as draws from a small
**menu of whole answers** instead of 172 independent per-chunk coin flips.
Runs on existing run files — no new API calls.

```bash
python chunking_tests/mode_analysis.py \
    chunking_tests/output/output_chunks_pandoc_500.json \
    -o chunking_tests/output/mode_analysis.json
```

## Why this exists (motivation)

**Empirical.** Inspecting the 500-run Pandoc experiment
(`output_chunks_pandoc_500.json`) showed the per-chunk independence picture is
wrong: the 500 runs contain only **48 distinct selection sets**; one set covers
**51%** of runs; twelve chunks with ~0.62 marginal inclusion co-occur almost
perfectly (P(both) ≈ 0.62 vs 0.41 under independence). The model does not flip
172 coins — it samples one of a few coherent answers, plus tiny noise.

**Consequences for the pipeline.** The Bayesian voter
(`voter_bayesian.py`) is correct *per chunk* (across runs, draws for a fixed
chunk really are independent), but any **joint** statement — expected false
positives in the selected set, FDR, confidence in the whole selection —
implicitly multiplies per-chunk probabilities and is overconfident when chunks
move in blocks. The independence check in this script quantifies that.

**Literature.** (keys refer to `literature/references.bib`)

- `balasubramanian2026ising` prove that conditional-independence aggregators
  (majority vote, Dawid–Skene) become miscalibrated and even Bayes-inconsistent
  under correlated votes. Their setting couples *judges* per item; ours couples
  *chunks* per run — the transposed problem. Their theorems motivate dropping
  independence; their Ising fix does not transfer (≈14,700 couplings from ~48
  effective observations), so we exploit the observed low-dimensional structure
  instead.
- `dawid1979maximum`: the template model is a latent class model — the same
  family as Dawid–Skene — with the latent variable moved from the item's true
  label to the *run's answer choice*.
- `denisovblanch2026consensus`: correlated errors bound what aggregation can
  claim (reliability, not truth). Templates make the correlated part explicit.
- `quach2023conformallm`: adaptive stopping for sampling — the discovery-curve
  output here is the ingredient for such a stopping rule.

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

## Results on the 500-run Pandoc file (2026-07-05)

Menu: 48 raw patterns → **10 templates**, effective number of modes **3.2**,
residual noise **0.49 chunk flips per run** ("menu + typos" confirmed).

| rank | weight (95% CI) | size | chunks |
|---|---|---|---|
| 1 "long answer" | 0.548 (0.50–0.59) | 21 | 392, 394–401, 408–419 |
| 2 "short answer" | 0.312 (0.27–0.36) | 8 | 392, 395–396, 404, 408, 414–416 |
| 3 "whole section" | 0.062 (0.04–0.08) | 29 | 391–419 |
| 4 "medium" | 0.050 (0.03–0.07) | 12 | 392, 394–396, 404, 408–409, 414–418 |
| 5–10 (tail) | ≤ 0.006 each | | minor variants |

Decision rules vs ground truth 391–419 (recomputed 2026-07-27 after the GT
boundary was extended to 391; the earlier table used GT 395–419):

| rule | size | precision | recall | F1 |
|---|---|---|---|---|
| core (∩ major templates) | 7 | 1.000 | 0.241 | 0.389 |
| modal template | 21 | 1.000 | 0.724 | 0.840 |
| per-chunk Bayesian voter | 22 | 1.000 | 0.759 | 0.863 |
| **envelope (∪ major templates)** | 29 | 1.000 | **1.000** | **1.000** |

Every rule now has precision 1.0: with GT 391–419 the model never selects a
chunk outside the ground-truth range, so no rule can produce a false positive.
The envelope recovers the section exactly. This makes precision uninformative
on this question/model pair — the discriminating axis is recall — and it is
also why `p_noise` estimates to 0 (see the caveat in the root README).

Independence check (block = envelope − core, 22 chunks, mean rate 0.463):
all 22 appear together in **3.4%** of runs where the independent-chunk model
predicts ~0 (0.463²² ≈ 4·10⁻⁸); the per-run selection-size standard deviation
is **6.36 observed vs 1.97 predicted** by independent sampling — a 3.2×
violation. This is the quantitative form of the voter's overconfidence.

**Stability.** The independent 100-run file recovers the *identical* top
templates (same chunk sets for ranks 1–3) with the same envelope/voter
evaluations. Template weights differ (0.38 vs 0.55 for the long answer) —
within wide small-sample CIs, but worth monitoring; sampling settings of the
two historical batches were not recorded (fixed since run-metadata logging).

**Budget.** Discovering 95% of template mass needs **~26 runs** (17 on the
100-run file) versus the 500 actually used — the practical payoff of the menu
view, and the basis for a sequential stopping rule.

## Interpretation for the thesis

- The envelope rule beats the tuned Bayesian voter on F1 with *perfect recall*
  while requiring an order of magnitude fewer runs and no `p_correct/p_noise`
  estimates — evidence that modeling the joint structure matters more than
  refining marginal thresholds.
- Core vs envelope is an interpretable precision/recall dial for the
  downstream citation use case ("always cite the core, optionally the
  envelope").
- Chunks 402–403 and 405–407 appear **only** in the rank-3 "whole section"
  template (6% weight): the ground-truth chunks the voter misses are not
  diffuse noise — they belong to a coherent minority reading. Aggregation-level
  choices (mass threshold) decide their fate, not more runs.

## Limitations / next steps

- Greedy absorption is order-dependent and `--merge-dist`-sensitive; upgrade
  path: Bernoulli-mixture EM + BIC. The envelope's perfect recall here depends
  on including the 6%-weight whole-section template (`--mass 0.95`); with
  `--mass 0.90` recall drops — report the sensitivity, don't hide it.
- The `P(all block together)` check uses the block's mean rate (crude); the
  size-std comparison uses per-chunk marginals and is the rigorous one.
- The menu is per (question, corpus, model, temperature). Whether templates
  are stable across those factors is an open experiment — and a thesis
  chapter.
- Templates fix reliability accounting, not validity: a template can still be
  systematically wrong (`denisovblanch2026consensus`).
