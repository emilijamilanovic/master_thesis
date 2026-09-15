# Master's thesis — code and results

Code, recorded runs and analysis outputs for the thesis on selecting evidence
chunks from repeated, non-deterministic LLM relevance judgements. A document is
split into numbered chunks, a model is asked repeatedly which chunks answer a
question, and the repeated answers are combined into one final set whose error
is estimated.

The thesis document itself is not in this repository.

## Layout

    pandoc.md                 source document (the Pandoc user guide)
    pandoc.chunked.md         the same document split into 754 numbered chunks
    run_pipeline.py           runs the pipeline steps in order
    tools/                    provider clients and helpers
    rag/using_llm/            the chunker that splits the document into chunks
    chunking_tests/           pipeline and analysis scripts
      output/                 the recorded runs
      analysis/               CSVs and figures produced from the runs
      plots/                  the figures used in the thesis, and their scripts

### The recorded runs

    output/q2_fonts/          font question, nine models, 100 runs each
    output/q1_tables/         table question, nine models, 100 runs each
    output/q1_tables_v1/      table question, first prompt version, six models

Each configuration folder holds four files:

    runs.json                 one entry per run: the selected chunk numbers plus
                              the model, temperature, token limit, endpoint, a
                              hash of the prompt, the candidate window, how the
                              reply was parsed, and a timestamp
    stats.json                per-run precision, recall and F1, and the two
                              pooled rates p1 and p0
    selected.json             the delivered set, with each chunk's count and
                              posterior
    mode_analysis.json        answer templates, the core and envelope, and the
                              two independence checks

Ground truth: the table question is chunks 391-419 (29 of 172 candidates,
window 313-484); the font question is chunks 200-201, 214, 224-229, 236, 239
and 250-252 (14 of 180, window 130-309).

## Pipeline

    1  generation_chunks_test.py        N repeated selections  -> runs.json
    2  estimating_probs_gt_single.py    per-run metrics, rates -> stats.json
    3  voter_bayesian.py                Bayesian voter         -> selected.json
    4  mode_analysis.py                 templates, dependence  -> mode_analysis.json
    5  compare_*.py, budget_curves.py   cross-model analysis   -> analysis/

Step 4 takes the rates as arguments. Without them it falls back to generic
defaults and silently produces a different delivered set; `rerun_modes.sh`
re-runs it for every configuration with that configuration's own rates.

## Reproducing the reported results

Delivered sets, majority vote against the Bayesian voter:

    python chunking_tests/rule_sets.py --results chunking_tests/output/q1_tables --gt 391-419

Predicted against realised error, and the posterior saturation counts:

    python chunking_tests/calibration_table.py \
        --results chunking_tests/output/q2_fonts \
        --gt '200-201,214,224-229,236,239,250-252'

Sensitivity to the clamp, and to the prior via `--prior`:

    python chunking_tests/eps_sensitivity.py \
        --results chunking_tests/output/q1_tables --gt 391-419 \
        -o chunking_tests/analysis/eps_sensitivity_q1_all.csv

Run-budget curves:

    python chunking_tests/budget_curves.py \
        --results chunking_tests/output/q1_tables --gt 391-419 \
        --out chunking_tests/analysis/q1_budget

The figures, and the commands that draw them, are in `chunking_tests/plots/`.

## Randomness

The analyses are seeded, so re-running them on the recorded runs reproduces the
reported values exactly: the bootstrap intervals use 1000 resamples, the
independence check simulates 10,000 runs, and the budget curves average 200
subsamples at each budget.

Generation is deliberately not reproducible. Repeating it produces different
selections, because that variation is what the thesis studies. Every reported
result is computed from the recorded runs rather than from fresh generation.

## Environment

Python 3.12.

    python -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
    cp .env.example .env        # then fill in the provider keys

All providers are reached through OpenAI-compatible endpoints; keys are read
from the environment and `.env` is never committed.
