#!/usr/bin/env python3
"""One-command runner for the chunk-selection pipeline.

Executes the existing scripts in the correct order, wiring each step's
output into the next:

  1. generation   pipeline/generation_chunks_test.py
                  N repeated LLM selection runs      -> <out>/runs.json
  2. estimation   pipeline/estimating_probs_gt_single.py
                  per-run metrics + pooled p-hats    -> <out>/stats.json
                  (only meaningful for the labeled Pandoc question)
  3. sizing       pipeline/voter_stat.py (imported, advisory)
                  majority-vote run budget for the estimated p-hats
  4. voter        pipeline/voter_bayesian.py
                  Bayesian selection                 -> <out>/selected.json
  5. modes        pipeline/mode_analysis.py
                  answer-template analysis           -> <out>/mode_analysis.json

Two variants (--pipeline):
  full     steps 1-5, including the answer-template analysis (default)
  classic  steps 1-4 only: the original pipeline as documented in the
           README, before templates were introduced. Use this to
           reproduce or compare against pre-template results.

Typical uses
  # full pipeline, 100 fresh runs (needs the provider key in .env):
  python3 run_pipeline.py --runs 100

  # original pipeline without the template analysis:
  python3 run_pipeline.py --runs 100 --pipeline classic

  # a specific model for the cross-model study:
  python3 run_pipeline.py --runs 100 --provider anthropic --model sonnet

  # analyze an existing runs file, no API calls:
  python3 run_pipeline.py \
      --runs-file results/output/q1_tables/v3_openai_gpt41/runs.json

  # show the plan without executing anything:
  python3 run_pipeline.py --runs 100 --dry-run

Unlabeled/deployment mode (--no-gt) skips steps 2-3 and requires
explicit --p-correct/--p-noise for the voter.

TODO (needs a decision, not code): the question prompt and the ground
truth are currently hardcoded for the Pandoc "table formats" question
(prompt in generation_chunks_test.py, GT in estimating_probs_gt_single.py
and mode_analysis.py --gt). Running new questions requires editing the
prompt; multi-question support would need a small refactor.
TODO: mode_analysis.py has no no-GT mode; under --no-gt its "vs_gt"
numbers are computed against the default range and must be ignored.
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PIPELINE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PIPELINE_DIR.parent
SCRIPTS = PIPELINE_DIR

STEP_GENERATION = SCRIPTS / "generation_chunks_test.py"
STEP_ESTIMATION = SCRIPTS / "estimating_probs_gt_single.py"
STEP_VOTER = SCRIPTS / "voter_bayesian.py"
STEP_MODES = SCRIPTS / "mode_analysis.py"


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def load_dotenv(path=REPO_ROOT / ".env"):
    """Minimal .env loader (KEY=VALUE lines). Real environment wins."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


def banner(step, text):
    print(f"\n{'=' * 66}\n[{datetime.now():%H:%M:%S}] STEP {step}: {text}\n{'=' * 66}",
          flush=True)


def run_step(cmd, dry_run):
    """Run one pipeline script as a subprocess, streaming its output."""
    printable = " ".join(str(c) for c in cmd)
    if dry_run:
        print(f"  [dry-run] would execute: {printable}")
        return
    print(f"  $ {printable}", flush=True)
    result = subprocess.run([str(c) for c in cmd], cwd=REPO_ROOT)
    if result.returncode != 0:
        sys.exit(f"ERROR: step failed with exit code {result.returncode}: {printable}")


def fail(msg):
    sys.exit(f"ERROR: {msg}")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Run the full chunk-selection pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    gen = ap.add_argument_group("generation")
    gen.add_argument("--source", default=".",
                     help="Directory containing the source corpus (pandoc.md)")
    gen.add_argument("--runs", type=int, default=100,
                     help="Number of LLM runs to generate")
    gen.add_argument("--temperature", type=float, default=0.7,
                     help="Sampling temperature for generation")
    gen.add_argument("--provider",
                     help="Provider: openai, anthropic, gemini, fireworks "
                          "(default: openai, per pipeline/tools/llm.py defaults)")
    gen.add_argument("--model",
                     help="Model shortcut (sonnet, glm, gpt4o, ...) or full model id")
    gen.add_argument("--skip-generation", action="store_true",
                     help="Skip the LLM calls and analyze an existing runs file")
    gen.add_argument("--runs-file",
                     help="Existing runs JSON to analyze (implies --skip-generation)")

    ev = ap.add_argument_group("evaluation and decision")
    ev.add_argument("--total-chunks", type=int, default=172,
                    help="Number of candidate chunks in the slice (noise universe + GT)")
    ev.add_argument("--gt", default="391-419",
                    help="Ground-truth chunk-id range 'lo-hi' (for mode analysis)")
    ev.add_argument("--no-gt", action="store_true",
                    help="No ground truth available: skip estimation and sizing; "
                         "requires --p-correct and --p-noise")
    ev.add_argument("--p-correct", type=float,
                    help="Override per-run inclusion prob. for relevant chunks "
                         "(default: estimated from ground truth)")
    ev.add_argument("--p-noise", type=float,
                    help="Override per-run inclusion prob. for irrelevant chunks")
    ev.add_argument("--prior", type=float, default=0.5, help="Voter prior")
    ev.add_argument("--threshold", type=float, default=0.9,
                    help="Voter posterior threshold")

    ap.add_argument("--pipeline", choices=["full", "classic"], default="full",
                    help="'full' = steps 1-5 (with answer-template analysis); "
                         "'classic' = steps 1-4, the original pre-template pipeline")
    ap.add_argument("-o", "--output-dir",
                    help="Output directory (default: results/output/pipeline_<timestamp>)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print what would be executed without running anything")
    args = ap.parse_args()

    load_dotenv()

    # ------------------------------------------------------------------
    # Resolve paths and modes
    # ------------------------------------------------------------------
    skip_generation = args.skip_generation or args.runs_file is not None
    out_dir = Path(args.output_dir) if args.output_dir else \
        REPO_ROOT / "results" / "output" / f"pipeline_{datetime.now():%Y%m%d_%H%M%S}"
    runs_json = Path(args.runs_file) if args.runs_file else out_dir / "runs.json"
    stats_json = out_dir / "stats.json"
    selected_json = out_dir / "selected.json"
    modes_json = out_dir / "mode_analysis.json"

    # ------------------------------------------------------------------
    # Pre-flight checks (also performed in --dry-run, as warnings)
    # ------------------------------------------------------------------
    problems = []
    for script in (STEP_GENERATION, STEP_ESTIMATION, STEP_VOTER, STEP_MODES):
        if not script.exists():
            problems.append(f"missing script: {script}")

    if not skip_generation:
        source = Path(args.source)
        if not source.is_dir():
            problems.append(f"source directory not found: {source}")
        elif not any(p.name.startswith("pandoc") and p.suffix == ".md"
                     for p in source.iterdir()):
            problems.append(f"no pandoc*.md corpus in source directory: {source}")
        # Which provider key is needed depends on --provider (or the
        # default configured in pipeline/tools/llm.py).
        sys.path.insert(0, str(PIPELINE_DIR))
        try:
            import tools
            provider = args.provider or tools.llm.configs['defaults']['include'][0]
            key_env = tools.llm.configs.get(provider, {}).get('api_key_env')
            if key_env is None:
                problems.append(f"unknown provider: {provider!r}; known: "
                                f"{', '.join(tools.llm.PROVIDERS)}")
            else:
                key = os.environ.get(key_env, "")
                if not key or key == "your_key_here":
                    problems.append(f"{key_env} is not set (paste it into .env); "
                                    f"required for generation with '{provider}'")
        except Exception as e:
            problems.append(f"could not check provider configuration: {e}")
        if runs_json.exists():
            problems.append(f"{runs_json} already exists; use --runs-file to analyze "
                            f"it, or choose a different --output-dir")
    else:
        if not runs_json.exists():
            problems.append(f"runs file not found: {runs_json}")

    if args.no_gt and (args.p_correct is None or args.p_noise is None):
        problems.append("--no-gt requires explicit --p-correct and --p-noise "
                        "(no ground truth to estimate them from)")
    # Accepts the same forms as mode_analysis.py --gt: a range ('391-419'),
    # a list ('200,214'), or a mix ('200-201,214,224-229'), so that questions
    # whose answer is not one contiguous block can be scored.
    try:
        for part in args.gt.split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                lo, hi = (int(x) for x in part.split("-"))
                if lo > hi:
                    raise ValueError
            else:
                int(part)
    except ValueError:
        problems.append("--gt must be a range ('391-419'), a list ('200,214') "
                        f"or a mix ('200-201,224-229'); got: {args.gt!r}")

    if problems:
        prefix = "WARNING (dry-run continues)" if args.dry_run else "ERROR"
        for p in problems:
            print(f"{prefix}: {p}")
        if not args.dry_run:
            sys.exit(1)

    # ------------------------------------------------------------------
    # Plan summary
    # ------------------------------------------------------------------
    print(f"\nPipeline plan  ({'DRY RUN' if args.dry_run else 'executing'})")
    print(f"  variant    : {args.pipeline}"
          + ("  (steps 1-5, with answer templates)" if args.pipeline == "full"
             else "  (steps 1-4, original pre-template pipeline)"))
    if not skip_generation:
        # Resolve what will actually be called, so the plan is unambiguous.
        try:
            sys.path.insert(0, str(PIPELINE_DIR))
            import tools
            prov = args.provider or tools.llm.configs['defaults']['include'][0]
            shortcuts = tools.llm.configs.get(prov, {}).get('model_shortcuts', {})
            mdl = shortcuts.get(args.model or 'medium', args.model or '?')
            print(f"  provider   : {prov}"
                  + ("" if args.provider else " (default)"))
            print(f"  model      : {mdl}"
                  + ("" if args.model else " (default 'medium' shortcut)"))
        except Exception:
            print(f"  provider   : {args.provider or 'default'}"
                  f"  model: {args.model or 'default'}")
    print(f"  output dir : {out_dir}")
    print(f"  runs file  : {runs_json}"
          + ("  (existing, generation skipped)" if skip_generation else
             f"  ({args.runs} fresh runs, temperature {args.temperature})"))
    print(f"  ground truth: {'NONE (--no-gt)' if args.no_gt else args.gt}")

    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1: generation (LLM calls)
    # ------------------------------------------------------------------
    if skip_generation:
        print("\nSTEP 1: generation — skipped (using existing runs file)")
    else:
        banner(1, f"generate {args.runs} selection runs (this calls the LLM API)")
        gen_cmd = [sys.executable, STEP_GENERATION, args.source,
                   "-o", runs_json, "-n", args.runs,
                   "--temperature", args.temperature]
        if args.provider:
            gen_cmd += ["--provider", args.provider]
        if args.model:
            gen_cmd += ["--model", args.model]
        run_step(gen_cmd, args.dry_run)

    # ------------------------------------------------------------------
    # Step 2: estimation against ground truth
    # ------------------------------------------------------------------
    p_correct, p_noise = args.p_correct, args.p_noise
    if args.no_gt:
        print("\nSTEP 2: estimation — skipped (--no-gt); "
              f"using provided p_correct={p_correct}, p_noise={p_noise}")
    else:
        banner(2, "estimate p_correct / p_noise against ground truth")
        run_step([sys.executable, STEP_ESTIMATION, runs_json,
                  "--total_chunks", args.total_chunks, "-o", stats_json],
                 args.dry_run)
        if not args.dry_run:
            stats = json.loads(stats_json.read_text())
            if p_correct is None:
                p_correct = stats["p_correct_hat"]
            if p_noise is None:
                p_noise = stats["p_noise_hat"]
            print(f"  estimated p_correct={p_correct:.4f}  p_noise={p_noise:.4f}"
                  f"  (mean per-run recall {stats['mean_recall']:.3f})")

    # ------------------------------------------------------------------
    # Step 3: advisory run-budget sizing (imported from voter_stat)
    # ------------------------------------------------------------------
    if args.no_gt:
        print("STEP 3: sizing — skipped (--no-gt)")
    elif args.dry_run:
        print("\nSTEP 3: sizing — [dry-run] would compute majority-vote run "
              "budget via pipeline/voter_stat.find_min_runs")
    else:
        banner(3, "majority-vote run-budget check (advisory)")
        sys.path.insert(0, str(SCRIPTS))
        try:
            from voter_stat import find_min_runs
            n, p_c, p_n = find_min_runs(p_correct=p_correct, p_noise=p_noise)
            if n:
                print(f"  majority voting would need ~{n} runs at these estimates "
                      f"(P[correct wins]={p_c:.3f}, P[noise wins]={p_n:.5f})")
            else:
                print("  no run count up to 1000 satisfies the majority-vote "
                      "targets at these estimates")
        except Exception as e:  # advisory only — never kill the pipeline
            print(f"  sizing skipped ({e})")

    # ------------------------------------------------------------------
    # Step 4: Bayesian voter selection
    # ------------------------------------------------------------------
    banner(4, "Bayesian voter selection")
    if args.dry_run and p_correct is None:
        print("  [dry-run] p-correct/p-noise will come from step 2's stats.json")
        p_show, n_show = "<estimated>", "<estimated>"
    else:
        p_show, n_show = p_correct, p_noise
    run_step([sys.executable, STEP_VOTER, runs_json,
              "--p-correct", p_show, "--p-noise", n_show,
              "--prior", args.prior, "--threshold", args.threshold,
              "-o", selected_json], args.dry_run)

    # ------------------------------------------------------------------
    # Step 5: answer-template (mode) analysis
    # ------------------------------------------------------------------
    if args.pipeline == "classic":
        print("\nSTEP 5: answer-template analysis — skipped "
              "(--pipeline classic: original pre-template pipeline)")
    else:
        banner(5, "answer-template analysis")
        if args.no_gt:
            print("  NOTE: mode_analysis has no no-GT mode; its 'vs_gt' numbers "
                  "below use the default GT range and must be ignored.")
        modes_cmd = [sys.executable, STEP_MODES, runs_json,
                     "--gt", args.gt, "-o", modes_json]
        if p_correct is not None and not args.dry_run:
            modes_cmd += ["--p-correct", p_correct, "--p-noise", p_noise,
                          "--prior", args.prior, "--threshold", args.threshold]
        run_step(modes_cmd, args.dry_run)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n{'=' * 66}\nDone ({args.pipeline} pipeline). Artifacts in {out_dir}:")
    artifacts = [("runs", runs_json), ("stats", stats_json),
                 ("selected", selected_json)]
    if args.pipeline == "full":
        artifacts.append(("modes", modes_json))
    for name, path in artifacts:
        status = "written" if (not args.dry_run and path.exists()) else \
                 ("planned" if args.dry_run else "skipped")
        print(f"  {name:9s} {path}  [{status}]")


if __name__ == "__main__":
    main()
