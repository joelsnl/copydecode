# copydecode

Local **KEEP/REPLACE** copy-edit for machine-translated English, and **any-language → English** translation, for EPUB files.

It does not need [Ollama](https://ollama.com). The usual path downloads [llama.cpp](https://github.com/ggml-org/llama.cpp) plus a Qwen2.5 GGUF that fits this GPU. vLLM and Ollama still work if they are already running.

Speed comes from not treating this as a chatbot. Clean sentences skip the GPU. Remaining paragraphs are tagged KEEP/REPLACE: clean spans are copied by stitching, and only dirty spans are autoregressed. On CUDA the default server is vLLM (`:8000`) or llama.cpp (`:8080`) so the system+glossary prefix is cached and n-gram speculation can draft REPLACE tokens. Packing is Qwen-token-aware (`num_ctx / 2`).

## Install

```powershell
python -m pip install -e .
```

Optional extras:

```powershell
python -m pip install -e ".[pack]"   # exact Qwen tokenizer.json for pack counts
python -m pip install -e ".[nllb]"   # optional CTranslate2 first pass
```

```powershell
copydecode serve --dry-run
copydecode serve
copydecode document.epub --mode polish
```

If no LLM server is running, `copydecode document.epub` starts llama.cpp itself. `--no-serve` skips that. `copydecode serve --stop` kills a detached server.

Files land in `%LOCALAPPDATA%\copydecode` on Windows, `~/.cache/copydecode` elsewhere. An older `novelpolisher` cache directory is reused if present.

## Usage

```powershell
copydecode devices
copydecode document.epub --dry-run
copydecode document.epub --mode polish
copydecode document.epub --mode translate
copydecode document.epub --from-chapter 1 --to-chapter 1
```

`--mode auto` (default) translates when the EPUB is mostly a non-Latin script (Chinese, Japanese, Korean, Arabic, Cyrillic, Thai, Devanagari, …) and polishes when it is already English MTL.

The tool probes `localhost:8000` (vLLM), `:8080` (llama.cpp), then `:11434` (Ollama). CUDA boxes prefer vLLM/llama.cpp when they are up.

Ollama still works, but it re-prefills every pack. Quit it from the tray icon before `copydecode serve` so the model is not loaded twice.

## How a run works

1. Detect GPU/RAM and clamp context, chunk size, and workers.
2. Keep the source register and terminology. Optional `--glossary` JSON is document-specific terms only.
3. Skip boilerplate and already-clean English (more aggressive on weak machines).
4. For polish: learned or heuristic KEEP/REPLACE tags (Seq2Edits-lite). Copy KEEP by stitching. Autoregress only REPLACE.
5. Pack REPLACE jobs by Qwen tokens until the prompt hits `num_ctx / 2`.
6. Translate sends remaining paragraphs as numbered completions (also token-packed).
7. Run EPUB spine items in parallel when VRAM allows.
8. Write a zip-round-trip EPUB. `--changelog` writes `.changes.md` / `.json`. `--checkpoint` writes resume state.

`--no-copydecode` restores whole-paragraph polish.

## Learned KEEP/REPLACE tagger

With `--changelog`, `.changes.json` is Seq2Edits training data. A CPU logistic tagger learns which sentences must stay REPLACE. It never sits on the GPU next to the 14B.

```powershell
copydecode train-tagger .\*.changes.json --install
copydecode eval-log .\document.changes.json
```

## Hardware caps

| Machine | Model ceiling | Workers | Notes |
| --- | --- | --- | --- |
| RTX 4070 12 GB | 14B Q4/AWQ | 2 if ≤8B, else 1 | ctx 4096 |
| MacBook Air M4 16 GB | 7B Q4 | 1 | Memory is shared with macOS |
| CPU / 8 GB | 3B | 1 | Aggressive skip |

Override with `--model`, `--workers`, `--num-ctx`, `--skip off|aggressive`. Reasoning models (DeepSeek-R1, QwQ) are never auto-selected.

## Optional NLLB first pass

For source-language EPUBs, a CTranslate2 NLLB model can draft English, then copydecode edits that draft. NLLB defaults to **CPU** so it does not fight the 14B already on the GPU. Source language is guessed from script (Chinese, Japanese, Korean, Arabic, Russian, Hindi, Thai).

```powershell
python -m pip install -e ".[nllb]"
set COPYDECODE_NLLB_PATH=C:\models\nllb-ct2
```

If that env var is unset, non-English text is sent straight to the LLM.

## llama.cpp / vLLM

`copydecode serve` picks:

| Machine | llama.cpp build | Model |
| --- | --- | --- |
| Windows + NVIDIA | `bin-win-cuda-12.4` (or 13.3) | Qwen2.5 14B/7B/3B Q4 by VRAM |
| Windows + AMD | `bin-win-vulkan` | same |
| Linux + NVIDIA | Ubuntu CUDA zip if present, else Vulkan | same |
| Linux + AMD | Ubuntu Vulkan (ROCm zip as fallback) | same |
| macOS Apple Silicon | `bin-macos-arm64` (Metal) | 7B on 16 GB, 14B on 24 GB+ |
| CPU only | CPU zip/tarball | 3B, or 7B if you have ≥32 GB RAM |

vLLM is optional on NVIDIA:

```powershell
vllm serve Qwen/Qwen2.5-14B-Instruct-AWQ --port 8000 --max-model-len 4096 --gpu-memory-utilization 0.90 --enable-prefix-caching --speculative-config "{\"method\":\"ngram\",\"num_speculative_tokens\":5}"
```

## Library API

```python
from copydecode.api import polish_paragraphs

texts, model = polish_paragraphs(["She go to school every morning."])
```

This is the same path HuaEPUB uses for **Polish English**.

## License

MIT
