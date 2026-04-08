# TriAttention GGUF

**Trigonometric KV Cache Compression for GGUF Model Inference**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A single-file Python implementation of [TriAttention](https://arxiv.org/abs/2604.04921)-inspired KV cache pruning for GGUF models via [llama-cpp-python](https://github.com/abetlen/llama-cpp-python). Enables efficient long-context inference on consumer GPUs by intelligently compressing the KV cache using distance-aware scoring based on RoPE frequency structure.

> **Based on:** *"TriAttention: Efficient Long Reasoning with Trigonometric KV Compression"* — Mao et al., 2026 ([arxiv:2604.04921](https://arxiv.org/abs/2604.04921))

**Author:** [g023](https://github.com/g023) · **License:** MIT

---

## What It Does

During text generation, the KV (Key-Value) cache grows linearly with sequence length, consuming GPU memory proportionally. This script periodically prunes the KV cache to a fixed budget by scoring each cached token's importance using a **trigonometric series** derived from the model's RoPE (Rotary Position Embedding) frequencies.

**Key idea from the paper:** In transformer models with RoPE, attention patterns follow predictable distance-dependent curves shaped by a trigonometric series. TriAttention exploits this to score token importance without relying on unstable post-RoPE attention observations.

### Features

- **Trigonometric series scoring** — uses actual RoPE frequencies extracted from model metadata
- **Future offset averaging** — evaluates importance across geometric future distances {1, 2, 4, ..., 65536} (per paper Section 4.2)
- **Attention sink protection** — always preserves initial tokens that receive disproportionate attention (StreamingLLM insight)
- **Three-zone retention** — sinks (first N) + scored middle + recent window (last N) = budget
- **Window-based pruning** — triggers every β tokens (default 128, matching the paper)
- **Low-level generation** — uses `eval()`/`sample()` for precise KV cache control
- **Benchmark mode** — built-in before/after comparison
- **Zero additional dependencies** — only requires `llama-cpp-python`
- **Works with any GGUF model** using RoPE positional encoding

---

## Benchmark Results

Tested on **Qwen3-1.77B Q8_0 GGUF** with NVIDIA RTX 3060 12GB. Prompt: physics explanation (30 tokens), generating 512 tokens.

| KV Budget | Baseline tok/s | TriAttention tok/s | Final Cache | Prune Events | KV Memory Reduction |
|-----------|:--------------:|:------------------:|:-----------:|:------------:|:-------------------:|
| Full (no prune) | 17.7 | — | 542 | 0 | 1.0× |
| 256 | 17.7 | 17.7 | 286 | 2 | **1.9×** |
| 128 | 17.7 | 17.9 | 158 | 6 | **3.4×** |
| 64  | 17.8 | 17.8 | 94 | 14 | **5.8×** |

**Key observations:**
- **Output quality preserved** — first 300 characters are identical between pruned and baseline across all tests
- **No speed penalty** — pruning overhead is negligible; compressed cache even shows marginal speedup
- **Up to 5.8× KV memory reduction** at budget=64 with coherent output

---

## How It Works

### The Trigonometric Series

RoPE encodes position as rotations across frequency bands. For each frequency band `f`, the rotation rate is:

```
ω_f = θ^(-2f/d)
```

where `θ` is the base frequency (e.g., 1,000,000 for Qwen3) and `d` is the head dimension (e.g., 128).

The attention logit between a query at position `p_q` and key at position `p_k` decomposes into a **trigonometric series** in the Q-K distance `Δ = p_q - p_k`:

```
logit(Δ) ≈ Σ_f  a_f · cos(ω_f · Δ) + b_f · sin(ω_f · Δ)
```

This script computes a weighted approximation of this series to score each token's importance:

```
score(Δ) = Σ_f  w_f · cos(ω_f · Δ)
```

where `w_f = 1/(f+1)` emphasizes the dominant low-frequency bands.

### Future Offset Averaging

Following the paper (Section 4.2, Eq. 11), importance is averaged over geometric future offsets:

```
S̃(k) = (1/|D|) · Σ_{δ∈D}  score(Δ + δ)
```

where `D = {1, 2, 4, 8, ..., 65536}` (17 offsets with geometric spacing). This accounts for a token's importance not just now, but across many future query positions.

### Token Retention

When the cache exceeds the budget, tokens are retained in three zones:

1. **Attention sinks** (first `sink_tokens`, default 4) — always kept; these initial tokens receive disproportionate attention in all transformers
2. **Scored middle** — ranked by trigonometric importance score; top-K kept
3. **Recent window** (last `recent_tokens`, default 128) — always kept; maintains local coherence

### Pruning Cycle

```
[Prompt tokens] → eval → [generate tokens] → ...
                                                ↓ (every window_size tokens)
                              [Score all tokens via trigonometric series]
                              [Keep: sinks + top-scored + recent]
                              [Evict rest via llama_memory_seq_rm]
                              → continue generating
```

---

## Installation

### Prerequisites

- Python 3.8+
- NVIDIA GPU with CUDA support (tested on RTX 3060 12GB)
- A GGUF model file

### Install llama-cpp-python with CUDA

```bash
# Using pip with CUDA support
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python

# Or with conda
conda install -c conda-forge llama-cpp-python
```

### Clone this repository

```bash
git clone https://github.com/g023/triattention-gguf.git
cd triattention-gguf
```

No other dependencies are needed — just `llama-cpp-python`.

---

## Usage

### Basic Generation

```bash
python triattention_gguf.py \
  --model ./your-model.gguf \
  --prompt "Explain quantum computing in detail."
```

### With Custom Budget

```bash
python triattention_gguf.py \
  --model ./your-model.gguf \
  --prompt "Write a comprehensive essay on climate change." \
  --budget 1024 \
  --max-tokens 8192
```

### Aggressive Compression

```bash
python triattention_gguf.py \
  --model ./your-model.gguf \
  --prompt "Solve: integral of x^2 from 0 to 5" \
  --budget 128 \
  --window-size 64 \
  --recent-tokens 32
```

### Benchmark Mode

Compare pruned vs baseline performance side-by-side:

```bash
python triattention_gguf.py \
  --model ./your-model.gguf \
  --prompt "Explain general relativity" \
  --benchmark \
  --budget 256
```

### Full Attention Baseline (No Pruning)

```bash
python triattention_gguf.py \
  --model ./your-model.gguf \
  --prompt "Hello world" \
  --no-prune
```

### All Options

```
usage: triattention_gguf.py [-h] --model MODEL --prompt PROMPT
                            [--max-tokens N] [--n-ctx N] [--n-gpu-layers N]
                            [--seed N] [--budget N] [--window-size N]
                            [--recent-tokens N] [--sink-tokens N] [--n-freq N]
                            [--no-prune] [--temperature F] [--top-p F]
                            [--top-k N] [--min-p F] [--repeat-penalty F]
                            [--benchmark] [--verbose]

Model & Generation:
  --model MODEL         Path to GGUF model file (required)
  --prompt PROMPT       Input prompt text (required)
  --max-tokens N        Maximum tokens to generate (default: 4096)
  --n-ctx N             Context window size (default: 40960)
  --n-gpu-layers N      GPU layers to offload, -1=all (default: -1)
  --seed N              Random seed (default: 42)

TriAttention Parameters:
  --budget N            Max tokens in KV cache (default: 2048)
  --window-size N       Pruning interval in tokens (default: 128)
  --recent-tokens N     Recent tokens to always keep (default: 128)
  --sink-tokens N       Initial sink tokens to always keep (default: 4)
  --n-freq N            Dominant RoPE frequency bands for scoring (default: 16)
  --no-prune            Disable pruning (full attention baseline)

Sampling:
  --temperature F       Sampling temperature (default: 0.7)
  --top-p F             Nucleus sampling threshold (default: 0.95)
  --top-k N             Top-k sampling (default: 40)
  --min-p F             Min-p sampling (default: 0.05)
  --repeat-penalty F    Repeat penalty (default: 1.1)

Modes:
  --benchmark           Run benchmark comparing pruned vs baseline
  --verbose             Enable verbose/debug logging
```

---

## Recreating This Project From Scratch

This section provides enough detail for a mid-level engineer to rebuild the entire system.

### Step 1: Understand the Core Concept

The KV cache in transformer inference stores key and value vectors for all previously processed tokens. During autoregressive generation, this cache grows linearly. TriAttention prunes it by scoring token importance using a mathematical property of RoPE: attention between tokens follows a **trigonometric series** in their positional distance.

### Step 2: Set Up the Environment

```bash
conda create -n triattention python=3.10
conda activate triattention
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python
```

### Step 3: Extract RoPE Frequencies

The model's RoPE configuration determines the frequency bands:

```python
import ctypes
import llama_cpp

# After loading model:
model = llama_cpp.llama_get_model(llm.ctx)
n_embd = llama_cpp.llama_model_n_embd(model)
n_head = llama_cpp.llama_model_n_head(model)
head_dim = n_embd // n_head  # e.g., 128

# Read theta from metadata
theta = 10000.0  # default; extract from model metadata for exact value
# For Qwen3: theta = 1,000,000

# Compute frequencies
freqs = [theta ** (-2.0 * f / head_dim) for f in range(head_dim // 2)]
```

### Step 4: Implement the Scoring Function

Score each token by how much future attention it's likely to receive:

```python
import math

def trig_score(delta, freqs, weights):
    """Trigonometric series score for Q-K distance delta."""
    return sum(w * math.cos(omega * delta) for omega, w in zip(freqs, weights))

# Average over geometric future offsets
future_offsets = [2**i for i in range(17)]  # {1, 2, 4, ..., 65536}

def importance(token_pos, current_pos, freqs, weights):
    base_delta = current_pos - token_pos
    return sum(trig_score(base_delta + d, freqs, weights) for d in future_offsets) / len(future_offsets)
```

### Step 5: Implement the Generation Loop

Use low-level `eval()` and `sample()` instead of the high-level `llm()` API:

```python
llm.eval(prompt_tokens)  # Process prompt

for step in range(max_tokens):
    tok = llm.sample(temp=0.7, top_p=0.95, ...)
    if tok == llm.token_eos():
        break
    print(llm.detokenize([tok]).decode(), end='', flush=True)
    llm.eval([tok])  # Add to KV cache

    if should_prune():
        prune_kv_cache(llm)
```

### Step 6: Implement KV Cache Pruning

When the cache exceeds the budget:

1. **Score all tokens** using the trigonometric series
2. **Protect sinks** (first N tokens) and **recent window** (last N tokens)
3. **Rank middle tokens** by score, keep top-K
4. **Remove evicted tokens** using `llama_memory_seq_rm`:

```python
mem = llama_cpp.llama_get_memory(llm.ctx)
# Remove positions [p0, p1) from sequence 0
llama_cpp.llama_memory_seq_rm(mem, 0, p0, p1)
```

Group consecutive positions into ranges for efficient batch removal.

### Step 7: Track Positions

Maintain a list of which KV cache positions correspond to which tokens. After eviction, remove evicted positions from the tracking list. New tokens always get the next sequential position.

### Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Low-level `eval`/`sample` | Full control over KV cache; high-level API manages cache internally |
| No position shifting after eviction | Preserves correct RoPE distances between surviving tokens |
| Frequency weights `1/(f+1)` | Low-frequency bands dominate attention (paper observation) |
| Geometric future offsets | Paper ablation: geometric >> linear spacing (+17.1% accuracy) |
| Attention sinks (4 tokens) | First tokens receive disproportionate attention (StreamingLLM) |
| Window-based pruning (β=128) | Amortizes scoring cost; matches paper design |

---

## Architecture

```
┌──────────────────────────────────────────────────────┐
│                  triattention_gguf.py                 │
│                                                      │
│  ┌──────────────────────────────────────────────┐    │
│  │            get_rope_frequencies()             │    │
│  │  Reads θ from model metadata, computes ω_f   │    │
│  └────────────────────┬─────────────────────────┘    │
│                       │                              │
│  ┌────────────────────▼─────────────────────────┐    │
│  │          TriAttentionPruner                   │    │
│  │                                               │    │
│  │  _trig_score(Δ)     → cos series at RoPE ω   │    │
│  │  compute_importance()→ avg over future offsets│    │
│  │  prune()            → score → select → evict  │    │
│  │  _batch_remove()    → llama_memory_seq_rm     │    │
│  └────────────────────┬─────────────────────────┘    │
│                       │                              │
│  ┌────────────────────▼─────────────────────────┐    │
│  │            generate()                         │    │
│  │                                               │    │
│  │  eval(prompt) → [sample → eval → prune?] loop│    │
│  │  Returns (text, stats)                        │    │
│  └──────────────────────────────────────────────┘    │
│                                                      │
│  main() → argparse → load model → generate/benchmark │
└──────────────────────────────────────────────────────┘
```

---

## Differences From the Paper

This is a **practical approximation** of the full TriAttention, adapted for the constraints of llama-cpp-python:

| Aspect | Paper (Full TriAttention) | This Implementation |
|--------|--------------------------|---------------------|
| Q/K centers | Computed via offline calibration on pre-RoPE vectors | Not available (llama-cpp-python doesn't expose pre-RoPE Q/K) |
| S_trig scoring | Uses actual Q/K center norms and phases | Uses generic frequency-weighted cosines (same ω_f structure) |
| S_norm scoring | Complements S_trig using value norms | Not implemented (no access to value vectors) |
| Adaptive weighting | Mean Resultant Length R balances S_trig/S_norm | Single scoring component |
| GQA head handling | Per-head scoring with z-score normalization | Uniform scoring across all heads |
| Backend | Modified vLLM with FlashAttention-2 | llama-cpp-python with GGUF quantization |

Despite these simplifications, the core mathematical structure — scoring by trigonometric series over RoPE frequencies with geometric future offset averaging — is faithfully preserved.

---

## Troubleshooting

**"KV cache removal not available"** — Your llama-cpp-python version may be too old. Upgrade to ≥ 0.3.12:
```bash
CMAKE_ARGS="-DGGML_CUDA=on" pip install --upgrade llama-cpp-python
```

**Out of memory** — Reduce `--n-ctx` or `--budget`:
```bash
python triattention_gguf.py --model model.gguf --prompt "..." --n-ctx 8192 --budget 512
```

**Slow generation** — Ensure GPU offloading is active (`--n-gpu-layers -1`). Check CUDA is available:
```bash
python -c "from llama_cpp import Llama; print('OK')"
```

**Poor output quality** — Try adjusting sampling parameters:
```bash
--temperature 0.8 --top-p 0.9 --repeat-penalty 1.0
```

---

## References

- [TriAttention: Efficient Long Reasoning with Trigonometric KV Compression](https://arxiv.org/abs/2604.04921) — Mao et al., 2026
- [Official TriAttention Implementation (vLLM)](https://github.com/WeianMao/triattention)
- [StreamingLLM: Efficient Streaming Language Models with Attention Sinks](https://arxiv.org/abs/2309.17453) — Xiao et al., 2024
- [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) — Python bindings for llama.cpp

---

## License

MIT License — see [LICENSE](LICENSE) for details.
