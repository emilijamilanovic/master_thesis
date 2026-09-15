import argparse
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import tools
from rag.using_llm.test_chunker import Chunker



def get_chunks_from_string(text):

    chunker = Chunker()
    chunks = list(chunker.chunks(text))
    return chunks

def chunking_input(src):

    # Get numbered chunks from file
    nota = src.get('nota_crediti')
    text = nota.fulltext()
    chunks = get_chunks_from_string(text)
   
    # Create the full numbered text and insert it into the prompt
    # numbered_text = "\n\n".join(chunk.annotated_string() for chunk in chunks)
    return chunks

def extract_chunks(chunks, ids):
    selected = []
    for cid in ids:
        if 0 <= cid < len(chunks):
            selected.append(f"<{cid}>\n{chunks[cid].content.strip()}")
    return selected

def save_chunks_to_markdown(name, chunk_lines, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{name}.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(chunk_lines))
    print(f"Saved {name}.md with {len(chunk_lines)} chunks")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source", help="Source directory containing nota_crediti.md")
    ap.add_argument("chunk_json", help="Path to JSON file with selected chunk IDs")
    ap.add_argument("output_dir", help="Where to save the separate markdown files")

    args = ap.parse_args()

    src = tools.source.Source(args.source)
    out = tools.output.Output(formats=[tools.output.OutputBuffer()])

    chunks = chunking_input(src)

    with open(args.chunk_json, "r", encoding="utf-8") as f:
        selected = json.load(f)

    for section, ids in selected.items():
        lines = extract_chunks(chunks, ids)
        save_chunks_to_markdown(section, lines, args.output_dir)

if __name__ == "__main__":
    main()

### IMPORTANT ###
# I have changed source.fulltext and added some lines to test_chunker.py