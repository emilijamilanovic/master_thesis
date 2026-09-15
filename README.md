# Master's thesis — code and results

Code, recorded runs and analysis outputs for the thesis on selecting evidence
chunks from repeated, non-deterministic LLM relevance judgements. A document is
split into numbered chunks, a model is asked repeatedly which chunks answer a
question, and the repeated answers are combined into one final set whose error
is estimated.

## Layout

    pandoc.md                 source document (the Pandoc user guide)
    pandoc.chunked.md         the same document split into 754 numbered chunks
    pipeline/                 the pipeline steps, the chunker, tools/ and
                              run_pipeline.py, which runs the steps in order
    analysis_code/            scripts that read the recorded runs and produce
                              the tables and figures
    results/
      output/                 the recorded runs
      analysis/               CSVs and figures produced from the runs
      plots/                  the ten figures used in the thesis

### The recorded runs

    results/output/q2_fonts/       font question, nine models, 100 runs each
    results/output/q1_tables/      table question, nine models, 100 runs each
    results/output/q1_tables_v1/   table question, first prompt version, six models

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

    1  pipeline/generate_runs.py               N repeated selections  -> runs.json
    2  pipeline/estimate_rates.py              per-run metrics, rates -> stats.json
    3  pipeline/voter_bayesian.py              Bayesian voter         -> selected.json
    4  pipeline/mode_analysis.py               templates, dependence  -> mode_analysis.json
    5  analysis_code/*.py                      cross-model analysis   -> results/analysis/

Step 1 splits the document with `pipeline/chunker.py`, the heading-aware
chunker described in Chapter 3.

Step 4 takes the rates as arguments. Without them it falls back to generic
defaults and silently produces a different delivered set; `pipeline/rerun_modes.sh`
re-runs it for every configuration with that configuration's own rates.

## Reproducing the reported results

Delivered sets, majority vote against the Bayesian voter:

    python analysis_code/rule_sets.py --results results/output/q1_tables --gt 391-419

Predicted against realised error, and the posterior saturation counts:

    python analysis_code/calibration_table.py \
        --results results/output/q2_fonts \
        --gt '200-201,214,224-229,236,239,250-252'

Sensitivity to the clamp, and to the prior via `--prior`:

    python analysis_code/eps_sensitivity.py \
        --results results/output/q1_tables --gt 391-419 \
        -o results/analysis/eps_sensitivity_q1_all.csv

Run-budget curves:

    python analysis_code/budget_curves.py \
        --results results/output/q1_tables --gt 391-419 \
        --out results/analysis/q1_budget

The ten figures are in `results/plots/`; the scripts that draw them are
`analysis_code/figures_*.py`, and `results/plots/README.md` lists the commands.

## Environment

Python 3.12.

    python -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
    cp .env.example .env        # then fill in the provider keys

All providers are reached through OpenAI-compatible endpoints; keys are read
from the environment and `.env` is never committed.
