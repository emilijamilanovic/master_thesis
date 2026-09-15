# Figures for the thesis

The ten figures used in Chapter 5 (font question, `q2_*`) and Appendix C
(table question, `q1_*`), together with the scripts that draw them.

The three scripts are copies of `compare_models_same_prompt.py`,
`compare_templates.py` and `budget_curves.py` in analysis_code/, changed
in four ways:

* no figure titles, because the captions in the thesis explain each figure —
  the subplot titles of the combined budget figure are kept;
* no prompt version in the legend, so a curve is labelled `gpt-4o` rather than
  `gpt-4o (v3)`;
* `claude-haiku-4-5-20251001` is written `claude-haiku-4.5`, as in the thesis
  tables (`NAME_OVERRIDES` at the top of each script);
* figures are named `<tag>_<figure>.png` from the required `--tag`, and only
  the figures the thesis uses are written, so no CSVs are produced here.

The originals in the parent folder are untouched and still write everything
they always did into `results/analysis/`.

## Regenerating

Run from the repository root.

Font question, Chapter 5:

    python3 analysis_code/figures_compare_models_same_prompt.py \
        --results results/output/q2_fonts \
        --gt '200-201,214,224-229,236,239,250-252' \
        --prompt v3 --tag q2 --out results/plots

    python3 analysis_code/figures_compare_templates.py \
        --results results/output/q2_fonts \
        --tag q2 --out results/plots

    python3 analysis_code/figures_budget_curves.py \
        --results results/output/q2_fonts \
        --gt '200-201,214,224-229,236,239,250-252' \
        --tag q2 --only openai_gpt41mini --out results/plots

Table question, Appendix C:

    python3 analysis_code/figures_compare_models_same_prompt.py \
        --results results/output/q1_tables \
        --gt 391-419 --prompt v3 --tag q1 --out results/plots

    python3 analysis_code/figures_compare_templates.py \
        --results results/output/q1_tables \
        --tag q1 --out results/plots

    python3 analysis_code/figures_budget_curves.py \
        --results results/output/q1_tables \
        --gt 391-419 --tag q1 --only openai_gpt4o --out results/plots

`--only` selects which configurations get their own two-curve figure; without
it every configuration gets one. The thesis shows gpt-4.1-mini for the font
question and gpt-4o for the table question.

## What each file is

| file | used in |
| --- | --- |
| `q2_chunk_frequency.png` | selection rate per chunk, font question |
| `q2_menu_size.png` | effective number of answers per model |
| `q2_dependence.png` | observed against independent selection-size spread |
| `q2_budget_combined.png` | reliability and validity curves, nine models |
| `q2_budget_gpt41mini.png` | the two curves for gpt-4.1-mini |
| `q1_chunk_frequency.png` | the same five figures for the table question |
| `q1_menu_size.png` | |
| `q1_dependence.png` | |
| `q1_budget_combined.png` | |
| `q1_budget_gpt4o.png` | the two curves for gpt-4o |

To use them in the thesis, copy them into `thesis/figures/`.
