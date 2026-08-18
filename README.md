# copydecode

## The Pitch (No BS)

It reads a document (EPUB, PDF, TXT, Markdown, HTML, DOCX, JSON/JSONL), finds the sentences that read like machine-translation sludge, and makes a local LLM rewrite *only those sentences*, splicing them back between the untouched ones. Every rewrite goes through a paranoid rejection gate; when the gate fails, you keep the original sentence, which means most of a typical file comes back unchanged — deliberately.

## Current State & Reality Check

- **Version 0.3.0, right after a hard refactor (Aug 2026).** A self-retraining "learned tagger" that silently mutated global state was deleted, the checkpoint format changed (old resume state is dead), server detection was fixed, and the public API was consolidated into `copydecode/__init__.py`. Anything written about this repo before that refactor is wrong now.
- **Single-maintainer hobby tool, bus factor 1.** It fell out of a Chinese web-novel downloader. It behaves like a tool the author runs on his own machine, because it is one.
- **The architecture is honest plumbing, not a framework.** Document model (`document.py`) → format readers/writers as plain dict registries (`io.py`) → sentence tagger and stitcher (`spans.py`) → token packer (`chunking.py`, `qwen_tokens.py`) → HTTP wrapper over llama.cpp/vLLM/Ollama/OpenAI-compatible (`engine.py`) → orchestration (`pipeline.py`). ~4k lines of source. You can hold the whole thing in your head in an afternoon, which is its best property.
- **The intelligence is not in the code.** The code is scaffolding around two hand-tuned artifacts: a "does this look like MTL" heuristic (`router.mtl_score`) and a rejection gate (`spans.replacement_ok`). Both are regex lists and magic thresholds calibrated by one person reading Chinese web-novel output. There is no eval corpus and no quality metric anywhere in this repo. "It works" means "the author liked the diffs."
- **Tests: 120, fast, fully offline, actually meaningful** (gate behavior, parsing, packing, checkpoint recovery, server classification — all with fake engines). CI runs ruff + unittest on Ubuntu, Python 3.10 and 3.13. Note the gap: no GPU in CI and no Windows in CI, while the primary dev platform is Windows with CUDA. The parts most likely to rot — real llama-server flags, GitHub asset names, actual model behavior — are exactly the parts CI cannot see.
- **Production-ready?** As a CLI you run on your own files: yes, within its stated limits. As a library inside your service: read the skeletons below first; the library entry point will happily download gigabytes on first call unless you tell it not to (`auto_serve=False`).
- **License is AGPL-3.0-or-later.** If you are about to embed this in proprietary software, stop and talk to a lawyer or the author, in that order.

## Real-Life Use Cases

**Use it when:**

- You have English output from Google/DeepL (EPUB from a novel pipeline, a DOCX a coworker machine-translated, a TXT dump) and you want the worst sentences cleaned while everything else is left byte-identical. This is the core case and the only one with real mileage on it.
- You need any-language → English translation that cannot leave the machine. `--mode translate` is the privacy option, not the quality option — it is slower and usually worse than Google.
- You already split text yourself and want `copydecode.polish_paragraphs([...])` or a JSONL pipe, with your own caching around it.
- You need one canonical rendering of names/terms across a file (`--glossary`, deterministic single-pass string substitution, applied before the LLM ever sees the text).
- You have a GPU. 8 GB VRAM runs a 7B, 12 GB runs a 14B. That is the intended hardware.

**Run away when:**

- Your MTL is not Chinese-origin. The dirt heuristics were tuned on Chinese→English habits ("could not help but", calques, leftover 。！？). French→English DeepL output will mostly be tagged clean and the run will be an expensive no-op.
- You need publishable text, legal/medical accuracy, or layout fidelity. PDF in is text-extraction guesswork; PDF out is a freshly typeset document that looks nothing like the input.
- The PDF is scanned images. It is refused outright. OCR first.
- You are CPU-only and expect throughput. A 3B on CPU is a weak editor at a slow crawl; the tool will run, and you will hate it.
- You expected an Ollama product. Ollama is the compatibility fallback (it re-prefills every request — slow), and `copydecode serve` refuses to start llama.cpp while Ollama squats on the GPU. You have to actually quit Ollama from the tray.

## Known Skeletons & Technical Debt

Read this before your first bug report. None of these are hidden; they are trade-offs somebody has to know about.

**Heuristics and gates**

- Every threshold is a magic number: length gate 0.35–2.6× for polish and 0.35–4.5× for translate, whole-passage-echo cutoff 2.4×, ~2.6 chars/token estimate, prompt packing at `num_ctx/2`, `speculative.n_max=5`, 600 s HTTP timeout, 2 retries. All hand-tuned, none configurable without editing source.
- The gate is conservative on purpose, so it also rejects *good* rewrites. A "polished" output that is nearly identical to the input is normal operation, not a bug. Run with `--changelog` and read the sidecar before concluding it did nothing.
- Translate mode is guarded by the length gate and marker scrubbing only — far weaker than the polish gate. The LLM can still omit or invent; nothing in this repo can detect a fluent wrong translation.
- Mode auto-detection is a script-ratio sample. Mixed-language documents get misrouted; pass `--mode` explicitly if it matters.

**Format round-trips**

- EPUB is the good path: zip round-trip, untouched members stay byte-identical.
- DOCX reads body paragraphs and table cells only — headers, footers, text boxes, and shapes are invisible. A rewritten paragraph keeps the first run's formatting and blanks the rest (run-level bold/italic inside that paragraph is flattened). The writer mutates the same `python-docx` object the reader loaded; a `Document` is not safely writable twice.
- PDF "paragraphs" are re-joined from extracted lines by punctuation heuristics; columns, footnotes, and page furniture can leak into the text stream and get "polished." Output PDFs use the first system TTF found, with a latin-1 replacement fallback — CJK glyphs can tofu.
- Markdown keeps headings and fenced code, but list formatting and fence language tags do not survive faithfully.
- Segments are tied back to their DOM/paragraph position by index. If a reader and writer ever disagree on enumeration, text lands in the wrong place. The tests cover the known cases; new formats need the same paranoia.

**LLM plumbing**

- Server detection is endpoint probing (`/api/tags` → Ollama, `/props` → llama.cpp, `/v1/models` → vLLM/OpenAI-ish) on ports 8000/8080/11434 only. A llama.cpp still loading its model answers 503 and is reported as "no server." Anything future that speaks `/v1/models` gets treated as OpenAI-compatible and may or may not enjoy that.
- The llama.cpp path sends raw completions (`system + "\n\n" + user`) with ChatML stop strings — it never applies the model's chat template. This is off-distribution for chat-tuned models and happens to work well with Qwen2.5-Instruct GGUFs, which is what the tool downloads for you. Bring your own exotic model, get your own exotic failures.
- Exact `n_keep`/prefix-cache accounting only exists on llama.cpp (via `/tokenize`). vLLM relies on its automatic prefix caching; Ollama gets neither and re-prefills every pack.

**Download machinery**

- `serve.py` hardcodes three Qwen2.5 GGUF URLs (3B/7B/14B Q4_K_M) and matches GitHub release assets by literal filename suffixes, including CUDA/ROCm version strings (`cuda-12.4`, `cuda-13.3`, `rocm-7.2`). These strings **will** rot when llama.cpp renames its artifacts, and only a human will notice.
- Downloads resume via Range requests. llama.cpp GitHub assets are sha256-verified and **refused** if the release has no digest. The three bundled GGUFs are pinned to known hashes; a truncated or swapped file fails closed. The 7B and 14B files come from bartowski because the official Qwen uploads of those sizes are sharded and this tool does not stitch shards.
- Hardware sizing (`hardware.py`) is a table of VRAM tiers and name-regex parameter guessing. A model whose size is not in its name gets conservative defaults and a shrug.

**Concurrency**

- Chapter-parallel workers exist but only activate in one narrow case (any GPU backend, ≤8B model, ≥10.5 GB VRAM → 2 workers). Under `workers > 1`, `ChangeLog` counter increments are unsynchronized — the edit list itself survives (CPython `list.append`), but the summary counts can drift. Stats-only bug, known, unfixed.
- The prefix prefill warms one server slot; parallel workers past slot 0 pay the first-prompt cost.

**Library ergonomics**

- `import copydecode` imports the world (bs4, ebooklib, httpx, rich). No lazy loading.
- `polish_paragraphs(cancelled=...)` returns partial results on cancellation without telling you which indexes were processed.
- Checkpoints are invalidated by any option change or by the input file's mtime/size changing (fingerprint). Conservative by design: it silently redoes work rather than ever reusing wrong state.
- Glossary matching compiles one big regex alternation — fine for the intended 15–60 terms, unproven at 10k. Sources ending in punctuation ("Mr.") never match because of `\b` boundaries.

## Setup & Local Running

Requirements: Python ≥ 3.10. A GPU if you want the LLM steps to finish this week. First run of `copydecode serve` downloads a llama.cpp build (tens–hundreds of MB) plus a Qwen2.5 GGUF (**2–9 GB** depending on your VRAM) unless you point it at a server you already run.

```powershell
pip install copydecode

# or, from a clone, for hacking on it:
git clone https://github.com/joelsnl/copydecode
cd copydecode
python -m pip install -e ".[dev]"
python -m unittest discover -s tests     # offline, no LLM
ruff check .
```

Pick **one** way to provide an LLM:

```powershell
# 1) Let it install and run llama.cpp itself (downloads happen here)
copydecode serve --dry-run     # shows exactly what it would fetch, fetches nothing
copydecode serve               # llama.cpp on http://127.0.0.1:8080

# 2) Use a server you already run (vLLM, llama-server, Ollama on 8000/8080/11434)
copydecode book.epub --engine vllm --host http://127.0.0.1:8000

# 3) Forbid all downloads and fail loudly if nothing is listening
copydecode book.epub --no-serve
```

Then:

```powershell
copydecode devices                          # what the hardware detection decided
copydecode book.epub --dry-run              # full work plan; guaranteed no side effects
copydecode book.epub                        # writes book.polished.epub (or book.en.epub for translate)
copydecode book.epub --changelog --checkpoint   # before/after sidecars + resumable state
copydecode paper.docx -o out.docx --glossary terms.json
copydecode scan.pdf --format txt            # don't pretend the output PDF will look like the input
```

Optional extras: `pip install -e ".[pack]"` for exact Qwen token packing (otherwise a character estimate is used), `".[nllb]"` plus `COPYDECODE_NLLB_PATH` for a CPU NLLB draft pass in translate mode.

Environment variables, all optional:

| Variable | Effect |
| --- | --- |
| `COPYDECODE_CACHE` | Cache root. Default `%LOCALAPPDATA%\copydecode` / `~/.cache/copydecode`. |
| `COPYDECODE_GGUF` | Serve this GGUF instead of downloading one. |
| `COPYDECODE_LLAMA_SERVER` | Use this `llama-server` binary instead of downloading one. |
| `COPYDECODE_TOKENIZER` | Path to a `tokenizer.json`, or `off` to force the char estimate. |
| `COPYDECODE_NLLB_PATH` / `COPYDECODE_NLLB_DEVICE` | CTranslate2 NLLB model dir and `cpu`/`cuda`. |

Where state lives: model/binary cache under `COPYDECODE_CACHE`; resume state in `.copydecode/<name>/checkpoint.jsonl` next to the input (only with `--checkpoint`); change logs next to the output (only with `--changelog`). Nothing phones home; the only network traffic is model downloads and localhost.

From Python:

```python
import copydecode

# auto_serve=True (the default) may download gigabytes on first call.
# In anything unattended, pass auto_serve=False and manage the server yourself.
texts, model = copydecode.polish_paragraphs(
    ["She go to school every morning."], auto_serve=False
)
```

Everything intentional raises a `copydecode.CopydecodeError` subclass (`EngineError` for server trouble, `DocumentError` for file trouble). A raw traceback from anything else is a bug — file it with the traceback, not a screenshot.

**Publishing to PyPI** is Trusted Publishing from `.github/workflows/publish.yml`, triggered by a GitHub Release (or `workflow_dispatch`). There is no API token in this repo. Before the first upload, a human has to:

1. Create a PyPI account and 2FA.
2. Register a pending publisher at https://pypi.org/manage/account/publishing/ — owner `joelsnl`, repo `copydecode`, workflow `publish.yml`, environment `pypi`.
3. Create a GitHub Environment named `pypi` on this repo (required reviewers are optional but worth turning on).
4. Push, then cut a GitHub Release whose tag matches `pyproject.toml` (`v0.3.0` is fine).

License: **AGPL-3.0-or-later**. Private use anywhere, including at work, is unconditionally fine. Distribution or network services built on modified versions owe source under the AGPL. Commercial exceptions: open a GitHub issue titled `Commercial license`.
