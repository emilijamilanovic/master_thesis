#!/usr/bin/env python3
"""Verify that configured providers and model ids actually work.

Sends a minimal completion request (a few tokens) to each provider's
model shortcuts and reports which succeed. Use after adding a provider
or changing a model id in pipeline/tools/llm.py.

  python3 pipeline/tools/check_providers.py                # default shortcuts
  python3 pipeline/tools/check_providers.py --all          # every shortcut
  python3 pipeline/tools/check_providers.py -p gemini      # one provider

API keys are read from the environment; run_pipeline.py-style .env
loading is done here too. Keys are never printed.
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import tools  # noqa: E402


def load_dotenv(path=Path(__file__).resolve().parent.parent.parent / '.env'):
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith('#') and '=' in line:
            key, _, value = line.partition('=')
            os.environ.setdefault(key.strip(), value.strip())


def check(provider, shortcut):
    """Return (ok, detail) for one provider/shortcut combination."""
    try:
        ai = tools.llm.LLM(purpose=provider)
        ai.config['model_shortcut'] = shortcut
        # Generous budget: reasoning models spend tokens before answering,
        # and would otherwise return empty content.
        ai.config['max_tokens'] = 512
        model = ai.config['model_shortcuts'][shortcut]
    except Exception as e:
        return False, f'config error: {e}'

    try:
        text = ai.query('Reply with the single word: ok').response()
        return True, f'{model} -> {text.strip()[:40]!r}'
    except Exception as e:
        msg = str(e).replace(os.environ.get(f'{provider.upper()}_API_KEY', '\0'),
                             '<key>')
        return False, f'{model} -> {type(e).__name__}: {msg[:160]}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-p', '--provider', action='append',
                    help='Provider(s) to check (default: all configured)')
    ap.add_argument('--all', action='store_true',
                    help='Check every model shortcut, not just the default one')
    ap.add_argument('--shortcut', default='medium',
                    help='Which shortcut to check when not using --all')
    args = ap.parse_args()

    load_dotenv()
    providers = args.provider or tools.llm.PROVIDERS

    failures = 0
    for provider in providers:
        cfg = tools.llm.configs.get(provider, {})
        if not (cfg.get('api_key') or os.environ.get(cfg.get('api_key_env', ''))):
            print(f'{provider:12s} SKIP  {cfg.get("api_key_env", "key")} not set')
            continue

        shortcuts = sorted(cfg.get('model_shortcuts', {})) if args.all \
            else [args.shortcut]
        for shortcut in shortcuts:
            if shortcut not in cfg.get('model_shortcuts', {}):
                continue
            ok, detail = check(provider, shortcut)
            print(f'{provider:12s} {shortcut:9s} {"OK  " if ok else "FAIL"}  {detail}')
            failures += 0 if ok else 1

    print('\nAll checked combinations succeeded.' if not failures
          else f'\n{failures} combination(s) failed — fix the model ids in pipeline/tools/llm.py.')
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
