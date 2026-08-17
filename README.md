# copydecode

A local tool that **copy-edits awkward machine-translated English**, and can **translate non-English text into English**, by running a LLM on your own GPU.

It reads **EPUB, PDF, TXT, Markdown, HTML, DOCX, JSON, and JSONL**. KEEP/REPLACE is the same for every format. It is a research leftover from a novel downloader, not a translation product. Read the limits before you install a 14B model.

## Where it came from

I built this while working on [HuaEPUB](https://github.com/joelsnl/HuaEPUB), a desktop app that downloads Chinese web novels and runs them through Google Translate. The English that comes out is often readable and often wrong in the same cheap ways: calques, stiff dialogue tags, “the corners of her mouth,” leftover Chinese punctuation.

I wanted a **local** pass after Google, not another cloud API. Early attempts asked a local model to rewrite whole paragraphs. That was worse than doing nothing. The model invented scenes, dropped subjects, leaked prompt instructions, and still reported a full page of “edits.”

copydecode is the version of that experiment that stopped trying to be a writer. It tags spans as KEEP or REPLACE, copies the KEEP text, and only decodes the dirty spans. A quality gate throws away replacements that blow up in length, leak `[n]` markers, add extra sentences, or drop names. When the gate fails, you keep the original English. On a real book that often means **most paragraphs do not change**. That is the point.

copydecode is the generic version of that experiment: any of the supported file types, any source language into English, same KEEP/REPLACE pass. HuaEPUB keeps its own novel-specific polish and does not import this package.

## What it is actually useful for

- You already have **English MTL** (Google, DeepL, or similar) in a file you can actually open at work — PDF, DOCX, TXT, Markdown, HTML, or EPUB — and you want a local model to tidy the worst sentences, not rewrite the document.
- You are willing to accept that a lot of lines will be left alone, and that the lines it does touch can still be slightly wrong.
- You have a **GPU** (roughly 8 GB+ VRAM for a 7B, 12 GB for a 14B). On CPU it will crawl and the 3B model is a weak editor.
- You can live with a first-run download of llama.cpp plus a multi-GB GGUF.

If you need something you could publish, hire a translator. If you need fast, decent English from Chinese, **Google Translate in HuaEPUB is still the better default**. This is the optional extra pass for people who dislike specific MTL habits and have spare VRAM.

## Realistic use cases

These are the jobs it is actually shaped for. If your case is not on this list, it can still run; it just was not why the KEEP/REPLACE gate exists.

**Polish English that already went through Google/DeepL.** That is the core. A Chinese web-novel EPUB from HuaEPUB or Calibre, a DeepL-dumped Word file from a coworker, a vendor PDF of a manual that reads like “the corners of her mouth.” You want the worst calques cleaned and the rest left alone. `--mode polish`. EPUB and DOCX keep more structure than PDF.

**Read a machine-translated PDF without sending it to another cloud.** Specs, papers, slide-exports-as-PDF, internal memos. Extractable text only. If you need to quote or edit afterward, `--format txt` or `--format docx` — do not expect the output PDF to look like the input. Scanned scans will fail; OCR first.

**Tidy Markdown or HTML you already have.** Notes, a README, a CMS export, documentation that was translated paragraph-by-paragraph. Fenced code stays put. Headings are mostly copied. This is closer to “edit the prose in my repo” than “typeset a book.”

**Keep a small glossary consistent across a file.** Character names, product names, one canonical rendering of a term the MT engine flip-flops on. `--glossary` is your list, not an auto-built world bible. Optional `--extract-glossary` will guess; you will still edit the JSON.

**Pipe paragraphs from your own code.** `polish_paragraphs([...])` or JSON/JSONL in and out. Useful if you already split a document and do not want copydecode to parse PDF layout. You own caching and file I/O.

**Local translate when you cannot use Google.** `--mode translate` on a TXT/DOCX/EPUB/PDF that is mostly CJK, kana, Hangul, Arabic, Cyrillic, Thai, or Devanagari. This is slower and usually worse than Google. It is the privacy/offline option, not the quality option. Mixed-language files get guessed wrong sometimes.

What I would not use it for: anything you would print with your name on it, legal or medical text, layout-sensitive forms, image-only scans, or “make this novel good.” Those fail in boring, predictable ways.

## What it is not

- Not a replacement for a human editor or a professional translator.
- Not a general “make my novel good” button. Genre, voice, and plot are out of scope; the prompts tell the model not to recast the text, and the gate is there because the model still tries.
- Not strong multilingual MT. Translate mode is “ask Qwen to translate this file locally.” It is slower than Google, worse on many language pairs, and still an LLM: it can omit or invent. Script detection is a heuristic (CJK, kana, Hangul, Arabic, Cyrillic, Thai, Devanagari). Mixed-language files will be guessed wrong sometimes.
- Not Ollama-the-product. Ollama is an optional backend. The supported path is llama.cpp (auto-downloaded) or vLLM if you already run it. If Ollama is sitting on the GPU, llama.cpp will refuse to start.
- Not a service. There is no hosted API, no queue, no uptime, no support contract in the box.

The KEEP/REPLACE tagger and the MTL “dirt” list started on Chinese web-novel English. Other languages’ bad habits are only caught if they look like generic broken English or you train the tagger yourself.

## How to use it

```powershell
python -m pip install -e .
copydecode serve --dry-run
copydecode serve
copydecode document.pdf --mode polish
copydecode notes.txt
copydecode paper.docx -o paper.polished.docx
copydecode scan.pdf --format txt
```

`--mode auto` (default) translates when the file is mostly a non-Latin script, and polishes when it is already English. `--mode translate` forces the LLM to translate. `--mode polish` only copy-edits English. `--format` / `-f` sets the output type (`epub`, `pdf`, `txt`, `md`, `html`, `docx`, `json`, `jsonl`); default is the same type as the input.

```powershell
copydecode devices
copydecode document.pdf --dry-run
copydecode document.epub --from-chapter 1 --to-chapter 1
```

If nothing is listening, `copydecode FILE` will try to download llama.cpp and a Qwen2.5 GGUF that fits this machine. `--no-serve` skips that. Cache is `%LOCALAPPDATA%\copydecode` on Windows, `~/.cache/copydecode` elsewhere (an older `novelpolisher` cache is reused if present).

From Python:

```python
from copydecode.api import polish_paragraphs

texts, model = polish_paragraphs(["She go to school every morning."])
```

Optional: `.[pack]` for a real Qwen tokenizer; `.[nllb]` plus `COPYDECODE_NLLB_PATH` for a CPU NLLB draft before the LLM. Unset NLLB and non-English goes straight to the LLM.

`--changelog` / `--checkpoint` are off unless you pass them. Whole-paragraph rewrite is `--no-copydecode` (usually a mistake).

## How a run works

1. Size the GPU/RAM and clamp context, chunk size, and workers.
2. Keep the source register. Optional `--glossary` is your terms only.
3. Skip boilerplate and English that already looks clean.
4. Polish: KEEP/REPLACE tags, stitch KEEP, decode REPLACE only.
5. Pack REPLACE jobs by Qwen tokens to about `num_ctx / 2`.
6. Translate: numbered paragraph completions, also packed.
7. Write the output file (EPUB zip round-trip, or a new PDF/TXT/DOCX/…).

Ollama works if it is already up, but it re-prefills every pack. Quit it from the tray before `copydecode serve` if you want the fast llama.cpp path.

## Hardware caps

| Machine | Model ceiling | Workers | Notes |
| --- | --- | --- | --- |
| RTX 4070 12 GB | 14B Q4/AWQ | 2 if ≤8B, else 1 | ctx 4096 |
| MacBook Air M4 16 GB | 7B Q4 | 1 | Memory is shared with macOS |
| CPU / 8 GB | 3B | 1 | Aggressive skip; expect pain |

Override with `--model`, `--workers`, `--num-ctx`, `--skip off|aggressive`. Reasoning models (DeepSeek-R1, QwQ) are never auto-selected; they waste the GPU on chain-of-thought this task does not need.

`copydecode serve` picks a llama.cpp build for Windows CUDA/Vulkan, Linux CUDA/Vulkan, or macOS Metal. vLLM is optional on NVIDIA if you already use it.

## License

**AGPL-3.0-or-later** for everyone who can live with copyleft. You may use, study, change, and share this, including at work, as long as you follow the AGPL. Private use is fine: running it on your own machine (work laptop included) to process your own files does not require a paid license. If you **distribute** a modified version, or a program that includes this code, you must offer the corresponding source under the AGPL. If you run a modified version as a **network service**, you must offer that source to its users. The full text is in `LICENSE`.

That is free as in freedom, not “embed this in a closed product and keep your code shut.”

**Commercial license (optional, paid).** The realistic case where money changes hands is a company that wants to ship copydecode inside proprietary software, or offer it as a service, and cannot or will not comply with the AGPL. There is no storefront, no price list, no SLA, and no indemnity in this repo. I might sell a separate proprietary license. I might say no. Until there is a signed agreement, the AGPL is what you have. Ask by opening a GitHub issue titled `Commercial license` on [joelsnl/copydecode](https://github.com/joelsnl/copydecode), or message [@joelsnl](https://github.com/joelsnl).

Most “we use this at work” usage is just private AGPL use. Do not expect a license invoice because someone polished a PDF on a company PC.

## File types

KEEP/REPLACE runs on paragraphs. The file type only changes how those paragraphs are read and written.

| Input | What you get back |
| --- | --- |
| EPUB | Zip round-trip: only rewritten HTML documents change. Nav/NCX/images stay. |
| TXT / Markdown | Paragraphs (Markdown also keeps headings, lists, and fenced code). Code fences are copied, not sent to the GPU. |
| HTML | Block elements rewritten in place when possible. |
| DOCX | Paragraph and table-cell text. A rewritten paragraph **flattens** run-level bold/italic in that paragraph. Untouched paragraphs keep their runs. |
| JSON / JSONL | One object per paragraph (`{"text": "..."}`). Useful as a pipe, not as a book. |
| PDF | **Extracted text only.** Output PDF is a newly typeset document. |

PDF limits, because people will assume otherwise:

- Scanned / image-only PDFs have no text. This tool will refuse them. Run OCR yourself first.
- Columns, headers, footers, footnotes, and page numbers often land in the paragraph stream. Some get skipped as boilerplate; some get polished as if they were body text.
- Images, fonts, and original layout are discarded. A two-column paper does not come back as a two-column paper.
- If you care about layout, write `--format txt` or `--format docx` and put the text wherever it actually belongs.

`--from-chapter` / `--to-chapter` index extractable sections: EPUB documents, PDF pages that had text, Markdown `H1`s, DOCX Heading 1s. They are not “PDF page 12” if pages 1–11 were blank image covers.
