# Selection prompts

Personal reference of the prompts tried for the chunk-selection task. The code
does **not** read this file — `generation_chunks_test.py` holds the prompt
inline. To switch variants, copy the text from here over the question block in
that script.

`sha256` below is the hash of the **question text as actually sent** (i.e. after
`nonl()` collapses the newlines), so it identifies a variant independently of
the run metadata's `prompt_sha256`, which hashes the whole conversation
(question + system + intro + all 172 numbered chunks) and therefore also changes
when the corpus or the chunk slice changes.

Recompute after editing:

```bash
python3 - <<'PY'
import hashlib, re
src = open('chunking_tests/generation_chunks_test.py', encoding='utf-8').read()
q = re.findall(r"conv\.append\(m\.user\(nonl\('''(.*?)'''\)\)\)", src, re.S)[-1]
print(hashlib.sha256(q.replace('\n',' ').encode()).hexdigest())
PY
```

| id | sha256 (question, as sent) | status | what it asks |
|---|---|---|---|
| v1 | `84e74821be01…` | retired 2026-07-27; kept as comparison condition | the *most relevant* chunks — a shortlist |
| v2 | `02a10d73b543…` | **rejected, not run** | all chunks, with the relevance criteria spelled out |
| v3 | `e726d1cbd1d4…` | **in use** since 2026-07-27 | *all* relevant chunks, no criteria given |

Shared across all variants (unchanged so far):

- system prompt — `1d213a5620fc…`
- document intro — `962319c6b08e…`

---

## v3 — exhaustive, generic (current)

`e726d1cbd1d4a254c76897da147a97f91b2eec4b5950b323465cb0b342a93ff6`

In the code since 2026-07-27. Runs: 10-run cross-model batches
(`output/v3_*`) and 100-run budget batches (`output/budget_n100/v3_*`,
conversation hash `bc2af9b4b87b`).

v1 with a single change: **"the most relevant chunk numbers that describe"** →
**"all chunk numbers relevant for describing"**. Everything else, including the
`nonl()` treatment and the "paragraph numbers" wording, is left untouched so
that v1 vs v3 is a clean one-variable comparison.

This is the variant that generalizes. The task is exhaustive citation, so the
question must ask for *all* relevant chunks rather than the best ones — but it
must ask for them the way it would have to ask on a corpus with no ground
truth, where the relevant material cannot be described in advance (see the note
under v2).

```text
Identify and provide a list of all chunk numbers relevant for describing the different table formats supported by Pandoc in Markdown.

Respond only with a list of paragraph numbers in the format: [x, y, z].
```

Note: "paragraph numbers" is inconsistent with the `<n>` chunk markers. Kept
deliberately, to hold everything but the one wording change constant. Fixing it
would be a separate variant.

---

## v2 — exhaustive + explicit criteria (rejected, never run)

`02a10d73b543d59fc51ce4d30392b085e2036406efe95c207bd9c2cc4a0d1ad8`

Kept only as a record of a design that was considered and rejected.

```text
List all chunk numbers that describe the table formats supported by Pandoc in Markdown.

A chunk is relevant if it belongs to the documentation of these table formats, including:
- headings that introduce the table formats or an individual format,
- introductory or overview text describing the available formats,
- descriptions of a specific format or of an extension belonging to it,
- examples that illustrate a format,
- text describing captions or other features of these tables.

Be exhaustive rather than selective: include every chunk a reader would need in order to understand the table formats completely, not only the most representative ones.

Respond only with a list of chunk numbers in the format: [x, y, z].
```

**Why it was rejected.** The bullet list describes the ground truth: it was
written after the answer key (391–419, the whole Tables section) had been fixed,
and it enumerates the kinds of chunk that key contains — headings, intro text,
captions, examples. That is only possible *because* the labels exist.

In the intended application there is no such key. On a corpus of financial
documents the relevant evidence for, say, a single transaction is scattered:
one sentence names the counterparty, a sentence in another document gives the
amount, a third gives the purpose. Nothing about that material can be described
in advance the way a documentation section can, so a prompt of this shape could
not be written at deployment time.

Using v2 would therefore measure the wrong thing: how well a model applies a
specification derived from the answer key, on the one corpus where such a
specification happens to be writable. v3 asks for exhaustiveness without
describing the target, which is what a real deployment can do.

(Also worth noting: this prompt's bullets contain real newlines, so it would
have had to skip `nonl()` — another way in which it is not a drop-in variant.)

---

## v1 — original (retired, kept as comparison condition)

`84e74821be01b8f7a975bcd4b7daf9617372309dcc368a7f421020c5a0c92137`

Used for the 500-run Llama experiment, the 10-run cross-model batches
(`output/v1_*`) and the 100-run budget batches
(`output/budget_n100/v1_*`, conversation hash `ba1554f7b904`). Asks for "the most relevant" chunks, which
invites a shortlist, while the answer key is exhaustive — a mismatch between
question and key that shows up as low recall (gpt-4o returned ~6 chunks per run
against a 29-chunk key).

Keep as the comparison condition for v3: the two differ in one phrase, so the
difference in selection size and variability is attributable to that phrase.

```text
Identify and provide a list of the most relevant chunk numbers 
that describe the different table formats supported by Pandoc in Markdown.

Respond only with a list of paragraph numbers in the format: [x, y, z].
```

(There is a trailing space after "chunk numbers" in the source; it is part of
the hashed text.)

---

## Shared pieces

**System prompt** — `1d213a5620fc5734038f4404c6ad57236825428e7110705388542ca118912c7f`:

```text
You are a knowledgeable technical assistant specializing in software tools and documentation.
Your task is to help analyze technical documentation. 
                              
Respond directly to the requests without any introduction or comments.
```

**Document intro** — `962319c6b08e480dffea11ad8cc8d1f20bb5b9224d925db1073d31bf20796dcf`:

```text
The following document is the official Pandoc documentation, split into numbered chunks.
Each chunk is marked with <n> at the beginning.
This documentation explains the syntax, features, and supported formats of Pandoc. 
```

---

## When adding a variant

1. Give it the next id and record the date and the reason for the change.
2. Prefer changes that would also be possible without ground truth — otherwise
   the result does not transfer to the intended application.
3. Change one thing at a time, so any difference in the runs is attributable.
4. Runs made with different questions are **not poolable** — keep them in
   separate output folders. `prompt_sha256` in the run metadata differs
   automatically, so mixed folders can still be untangled.
