import collections
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
import tools
import re
from chunker import Chunker
import json
import ast
import hashlib
from datetime import datetime, timezone

# chunks for Q1
CHUNK_SLICE = (313, 485)

# chunks for Q2
# CHUNK_SLICE = (130, 310)


def parse_chunk_list(r):
    """Extract the selected chunk ids from a model reply.

    Returns (ids, status). Reasoning models often wrap the answer in
    prose ("Looking at the chunks... [1, 2]"), so a bare literal_eval is
    not enough; we fall back to the last bracketed list in the reply.
    status is recorded in the run metadata so that parse failures are
    visible instead of silently becoming an empty selection.
    """
    try:
        return [int(x) for x in ast.literal_eval(r.strip())], 'literal'
    except Exception:
        pass

    matches = re.findall(r'\[[\s\d,]*\]', r)
    if matches:
        try:
            return [int(x) for x in ast.literal_eval(matches[-1])], 'regex'
        except Exception:
            pass

    return [], 'failed'


def get_chunks_from_string(text):

    chunker = Chunker()
    chunks = list(chunker.chunks(text))
    return chunks

def nonl(s):
    return s.replace('\n', ' ')

def doc(s):
    return lambda v : v.doc_id.startswith(s)

system_prompt_generic = nonl('''
You are a knowledgeable technical assistant specializing in software tools and documentation. 
Your task is to help analyze technical documentation. 

Respond directly to the requests without any introduction or comments.
''')

def gen_bkg_nota(src, temperature, provider=None, model=None):

    # Get numbered chunks from file
    nota = src.get('pandoc')
    text = nota.fulltext()
    chunks = get_chunks_from_string(text)

    chunks = chunks[CHUNK_SLICE[0]:CHUNK_SLICE[1]]

    ai = tools.llm.LLM(purpose='generate')
    if provider:
        ai.override({'include': [provider]})
    if model:
        # A model shortcut (e.g. 'sonnet') or a full model id
        shortcuts = ai.config.get('model_shortcuts', {})
        if model in shortcuts:
            ai.config['model_shortcut'] = model
        else:
            ai.config['model'] = model
    ai.config['temperature'] = temperature
    m = tools.llm.MessageBuilder()
    conv = []

    conv.append(m.system(system_prompt_generic))

    conv.append(m.user(nonl('''
The following document is the official Pandoc documentation, split into numbered chunks.
Each chunk is marked with <n> at the beginning.
This documentation explains the syntax, features, and supported formats of Pandoc. 
''')))

    # Create the full numbered text and insert it into the prompt
    numbered_text = "\n\n".join(chunk.annotated_string() for chunk in chunks)
    conv.append(m.user(numbered_text))
    conv.append(m.user(nonl('''
Identify and provide a list of all chunk numbers relevant for describing the different table formats supported by Pandoc in Markdown.

Respond only with a list of paragraph numbers in the format: [x, y, z].
''')))

    cfg = ai.config
    model = cfg.get('model') or cfg['model_shortcuts'][cfg['model_shortcut']]
    prompt_sha256 = hashlib.sha256(
        '\x00'.join(msg['content'] for msg in conv).encode('utf-8')).hexdigest()

    r = ai.query(conv).response(dump=sys.stdout)

    # out.chunk(r)
    table_chunks, parse_status = parse_chunk_list(r)
    if parse_status == 'failed':
        print(f'\nWARNING: could not parse a chunk list from the reply '
              f'(recorded as an empty selection with parse="failed")',
              file=sys.stderr)

    conv.append(m.assistant(r))

    dict_chunks = {
    "transaction_chunks": table_chunks,
    "meta": {
        "model": model,
        "temperature": cfg.get('temperature'),
        "max_tokens": cfg.get('max_tokens'),
        "base_url": cfg.get('base_url'),
        "prompt_sha256": prompt_sha256,
        "chunk_slice": list(CHUNK_SLICE),
        "parse": parse_status,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    },
}
    return dict_chunks


# def gen_bkg(src, out):
#     out.section('Transaction background e aggiornamento')

#     gen_bkg_nota(src, out)
#     # gen_bkg_rel(src, out)

def generate(src, temperature, provider=None, model=None):

    # out.chunk('NOTA DEL RICHIEDENTE\n')

    return gen_bkg_nota(src, temperature, provider, model)



# ****************************************************************************
# Main
# ****************************************************************************
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('source', help='Source directory')
    ap.add_argument('-o', '--output', help='Path to output .json file')
    ap.add_argument('-n', '--num_runs', type=int, default=1,
                    help='Number of generation runs to perform')
    ap.add_argument('--temperature', type=float, default=0.7,
                    help='Sampling temperature (default: 0.7)')
    ap.add_argument('--provider',
                    help='Provider config to use (openai, anthropic, gemini, '
                         'fireworks). Default: openai, per tools/llm.py defaults')
    ap.add_argument('--model',
                    help='Model shortcut (e.g. sonnet, glm, gpt4o) or a full '
                         'model id. Default: the provider\'s "medium" shortcut')
    args = ap.parse_args()

    src = tools.source.Source(args.source)
    all_runs = []

    def save():
        """Write the runs collected so far.

        Called after every run, not only at the end: a network error or a
        provider timeout part-way through a long batch would otherwise
        discard every completed (and already paid for) run.
        """
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(all_runs, f, indent=2, ensure_ascii=False)

    for i in range(args.num_runs):
        print(f"Generating run #{i+1}")

        try:
            run_result = generate(src, args.temperature, args.provider, args.model)
        except Exception as e:
            # Keep what has been collected, then stop. Re-running with
            # -n <remaining> into a new file and concatenating is safer than
            # silently continuing with a partly broken configuration.
            save()
            print(f"\nERROR on run #{i+1}: {type(e).__name__}: {e}", file=sys.stderr)
            print(f"Kept the {len(all_runs)} completed run(s) in {args.output}",
                  file=sys.stderr)
            raise

        run_result["run"] = i + 1
        all_runs.append(run_result)
        save()

    failed = sum(1 for r in all_runs if r['meta']['parse'] == 'failed')
    salvaged = sum(1 for r in all_runs if r['meta']['parse'] == 'regex')
    print(f"\nWrote {len(all_runs)} runs to {args.output}"
          f"  (parse: {len(all_runs) - failed - salvaged} clean, "
          f"{salvaged} salvaged from prose, {failed} failed)")


if __name__ == '__main__':
    main()

