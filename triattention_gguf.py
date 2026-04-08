#!/usr/bin/env python3
"""
TriAttention-inspired KV cache pruning for llama-cpp-python.

Implements trigonometric series distance scoring based on RoPE frequencies
to approximate the TriAttention KV cache compression method (arxiv:2604.04921).

Key features:
  - Trigonometric series scoring using actual RoPE frequencies from model metadata
  - Future offset averaging with geometric spacing {1,2,4,...,2^16}
  - Attention sink protection (first N tokens always kept)
  - Window-based pruning every β tokens (default 128, per paper)
  - Low-level eval/sample loop for precise KV cache control

Author: g023 (github.com/g023)
License: MIT
"""

import argparse
import ctypes
import math
import sys
import time
import logging
from llama_cpp import Llama
import llama_cpp

# =============================================================================
# GLOBAL CONFIGURATION PARAMETERS
# =============================================================================
# Model loading
N_CTX = 40960          # Context window size
N_GPU_LAYERS = -1      # Layers to offload to GPU (-1 = all)
N_THREADS = None       # Number of CPU threads (None = auto)
USE_MLOCK = True       # Lock model in memory
VERBOSE = False        # Verbose output from llama.cpp

# Sampling parameters
TEMPERATURE = 0.7      # Sampling temperature
TOP_P = 0.95           # Nucleus sampling threshold
TOP_K = 40             # Top-k sampling (0 = disabled)
MIN_P = 0.05           # Min-p sampling threshold
REPEAT_PENALTY = 1.1   # Penalty for repeating tokens
PRESENCE_PENALTY = 0.0 # Presence penalty
FREQUENCY_PENALTY = 0.0# Frequency penalty
SEED = 42              # Random seed for reproducibility

# TriAttention pruning parameters
KV_BUDGET = 2048       # Maximum tokens to keep in KV cache
WINDOW_SIZE = 128      # Prune every N tokens (paper: β=128)
RECENT_TOKENS = 128    # Number of most recent tokens to always keep
SINK_TOKENS = 4        # Number of initial tokens to always keep (attention sinks)
N_DOMINANT_FREQ = 16   # Number of dominant RoPE frequency bands for scoring

# Future offset set D = {2^0, 2^1, ..., 2^16} for score averaging (paper Section 4.2)
FUTURE_OFFSETS = [2**i for i in range(17)]  # 17 offsets: 1, 2, 4, ..., 65536

# Generation
MAX_TOKENS = 4096      # Maximum tokens to generate

# =============================================================================

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def get_rope_frequencies(llm, n_dominant):
    """
    Extract RoPE frequencies from model metadata.

    RoPE frequency for band f: ω_f = θ^(-2f/d)
    where θ is the base frequency and d is head_dim.
    Returns the n_dominant lowest-index (highest-value) frequencies.
    """
    model = llama_cpp.llama_get_model(llm.ctx)
    n_embd = llama_cpp.llama_model_n_embd(model)
    n_head = llama_cpp.llama_model_n_head(model)
    head_dim = n_embd // n_head

    # Read θ from metadata
    theta = 10000.0  # default
    n_meta = llama_cpp.llama_model_meta_count(model)
    for i in range(n_meta):
        buf = ctypes.create_string_buffer(256)
        llama_cpp.llama_model_meta_key_by_index(model, i, buf, 256)
        key = buf.value.decode()
        if 'rope' in key and 'freq_base' in key:
            buf2 = ctypes.create_string_buffer(256)
            llama_cpp.llama_model_meta_val_str_by_index(model, i, buf2, 256)
            theta = float(buf2.value.decode())
            break

    # Compute frequencies: ω_f = θ^(-2f/d) for f = 0, ..., d/2 - 1
    n_freq = head_dim // 2
    n_use = min(n_dominant, n_freq)
    freqs = []
    for f in range(n_use):
        omega = theta ** (-2.0 * f / head_dim)
        freqs.append(omega)

    logger.debug(f"RoPE θ={theta}, head_dim={head_dim}, using {n_use} frequency bands")
    logger.debug(f"Frequency range: [{freqs[-1]:.6e}, {freqs[0]:.6e}]")
    return freqs


class TriAttentionPruner:
    """
    TriAttention-inspired KV cache pruner using trigonometric series scoring.

    Scores each token's importance based on a trigonometric series computed from
    RoPE frequencies, averaged over geometric future offsets. Tokens are protected
    in three zones:
      1. Attention sinks (first sink_tokens) — always kept
      2. Recent window (last recent_tokens) — always kept
      3. Scored middle — top-K by trigonometric importance score

    Based on: "TriAttention: Efficient Long Reasoning with Trigonometric KV
    Compression" (Mao et al., 2026, arxiv:2604.04921)
    """

    def __init__(self, rope_freqs, budget=KV_BUDGET, window_size=WINDOW_SIZE,
                 recent_tokens=RECENT_TOKENS, sink_tokens=SINK_TOKENS,
                 future_offsets=None):
        self.rope_freqs = rope_freqs
        self.budget = budget
        self.window_size = window_size
        self.recent_tokens = recent_tokens
        self.sink_tokens = sink_tokens
        self.future_offsets = future_offsets if future_offsets is not None else FUTURE_OFFSETS

        # Precompute frequency weights: lower bands contribute more
        # Weight decays as 1/(f+1) — emphasizes dominant low-index bands
        self.freq_weights = [1.0 / (f + 1) for f in range(len(rope_freqs))]

        # Normalize weights so scores are in a consistent range
        w_sum = sum(self.freq_weights)
        self.freq_weights = [w / w_sum for w in self.freq_weights]

        # State tracking
        self.positions = []      # actual KV cache positions for tracked tokens
        self.n_tokens = 0        # total tokens currently in KV cache
        self.gen_step = 0        # total generation steps (for window timing)
        self.total_pruned = 0    # cumulative tokens pruned
        self.prune_count = 0     # number of pruning events

    def add_tokens(self, n, start_pos):
        """Register n new tokens starting at position start_pos."""
        for i in range(n):
            self.positions.append(start_pos + i)
        self.n_tokens += n
        self.gen_step += n

    def should_prune(self):
        """Check if pruning should trigger (cache exceeds budget at window boundary)."""
        return (self.n_tokens > self.budget and
                self.gen_step % self.window_size == 0)

    def _trig_score(self, delta):
        """
        Compute trigonometric series score for a given Q-K distance.

        Approximates: S_trig(Δ) ≈ Σ_f w_f · cos(ω_f · Δ)

        This captures the distance preference structure of RoPE attention
        (paper Eq. 3). Without Q/K centers, we use uniform phase (φ=0)
        and frequency-weighted amplitudes.
        """
        score = 0.0
        for f, (omega, weight) in enumerate(zip(self.rope_freqs, self.freq_weights)):
            score += weight * math.cos(omega * delta)
        return score

    def compute_importance(self, current_pos):
        """
        Compute importance score for each tracked token.

        Uses future offset averaging (paper Section 4.2, Eq. 11):
          S̃(k) = (1/|D|) · Σ_{δ∈D} S_trig(Δ + δ)
        where D = {1, 2, 4, ..., 2^16} and Δ = current_pos - token_pos.

        Returns list of scores aligned with self.positions.
        """
        n_offsets = len(self.future_offsets)
        scores = []
        for pos in self.positions:
            base_delta = current_pos - pos
            total = 0.0
            for offset in self.future_offsets:
                total += self._trig_score(base_delta + offset)
            scores.append(total / n_offsets)
        return scores

    def prune(self, llm):
        """
        Score all tokens and evict the lowest-scoring ones to meet the budget.

        Protection zones:
          - First sink_tokens positions are always kept (attention sinks)
          - Last recent_tokens positions are always kept (local context)
          - Middle tokens are scored and top-K are kept

        KV cache entries are removed via llama_memory_seq_rm.
        """
        n_total = len(self.positions)
        if n_total <= self.budget:
            return

        current_pos = self.positions[-1] if self.positions else 0
        scores = self.compute_importance(current_pos)

        # Determine protected indices
        keep = set()

        # Protect attention sinks (first sink_tokens)
        n_sink = min(self.sink_tokens, n_total)
        for i in range(n_sink):
            keep.add(i)

        # Protect recent window (last recent_tokens)
        n_recent = min(self.recent_tokens, n_total)
        recent_start = n_total - n_recent
        for i in range(recent_start, n_total):
            keep.add(i)

        # Score remaining and select top-K
        middle_budget = max(0, self.budget - len(keep))
        candidates = [(i, scores[i]) for i in range(n_total) if i not in keep]
        candidates.sort(key=lambda x: x[1], reverse=True)
        for i in range(min(middle_budget, len(candidates))):
            keep.add(candidates[i][0])

        # Determine eviction set — use actual KV cache positions
        evict_positions = []
        for i in range(n_total):
            if i not in keep:
                evict_positions.append(self.positions[i])

        if not evict_positions:
            return

        # Remove evicted positions from KV cache
        # Group consecutive positions for efficient removal
        evict_positions.sort()
        self._batch_remove(llm, evict_positions)

        # Update internal state
        n_evicted = n_total - len(keep)
        self.total_pruned += n_evicted
        self.prune_count += 1

        new_positions = [self.positions[i] for i in sorted(keep)]
        self.positions = new_positions
        self.n_tokens = len(new_positions)

        logger.debug(f"Pruned {n_evicted} tokens: {n_total} → {self.n_tokens} "
                     f"(budget={self.budget})")

    def _batch_remove(self, llm, positions):
        """Remove a list of positions from KV cache, grouping consecutive ranges."""
        mem = llama_cpp.llama_get_memory(llm.ctx)
        if mem is None:
            logger.warning("Could not get memory handle — pruning skipped")
            return

        # Group consecutive positions into ranges for efficient removal
        ranges = []
        start = positions[0]
        end = start
        for pos in positions[1:]:
            if pos == end + 1:
                end = pos
            else:
                ranges.append((start, end + 1))  # [start, end+1) exclusive
                start = pos
                end = pos
        ranges.append((start, end + 1))

        # Remove each range
        for p0, p1 in ranges:
            llama_cpp.llama_memory_seq_rm(mem, 0, p0, p1)


def generate(llm, prompt_tokens, pruner, max_tokens, sample_params, quiet=False):
    """
    Token-by-token generation with TriAttention KV cache pruning.

    Uses low-level eval/sample for precise KV cache control.
    Returns (generated_text, stats_dict).
    """
    # Evaluate prompt
    llm.eval(prompt_tokens)
    n_prompt = len(prompt_tokens)
    pruner.add_tokens(n_prompt, start_pos=0)

    generated_tokens = []
    generated_text_parts = []
    start_time = time.perf_counter()

    eos = llm.token_eos()

    for step in range(max_tokens):
        # Sample next token
        tok = llm.sample(
            top_k=sample_params['top_k'],
            top_p=sample_params['top_p'],
            min_p=sample_params['min_p'],
            temp=sample_params['temperature'],
            repeat_penalty=sample_params['repeat_penalty'],
            frequency_penalty=sample_params['frequency_penalty'],
            presence_penalty=sample_params['presence_penalty'],
        )

        if tok == eos:
            break

        generated_tokens.append(tok)

        # Decode and print
        text = llm.detokenize([tok]).decode('utf-8', errors='replace')
        generated_text_parts.append(text)
        if not quiet:
            print(text, end='', flush=True)

        # Evaluate the new token (adds to KV cache at next position)
        current_pos = n_prompt + step
        llm.eval([tok])
        pruner.add_tokens(1, start_pos=current_pos)

        # Prune if window boundary reached and cache exceeds budget
        if pruner.should_prune():
            pruner.prune(llm)

    elapsed = time.perf_counter() - start_time
    n_gen = len(generated_tokens)

    stats = {
        'prompt_tokens': n_prompt,
        'generated_tokens': n_gen,
        'total_time': elapsed,
        'tokens_per_sec': n_gen / elapsed if elapsed > 0 else 0,
        'prune_events': pruner.prune_count,
        'total_pruned': pruner.total_pruned,
        'final_cache_size': pruner.n_tokens,
    }

    return ''.join(generated_text_parts), stats


def run_benchmark(llm, prompt_tokens, pruner, max_tokens, sample_params):
    """Run generation with pruning and collect benchmark stats."""
    text, stats = generate(llm, prompt_tokens, pruner, max_tokens, sample_params,
                           quiet=True)
    return text, stats


def main():
    parser = argparse.ArgumentParser(
        description="TriAttention-inspired KV cache pruning for GGUF inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic generation with TriAttention pruning
  python triattention_gguf.py --model model.gguf --prompt "Explain quantum computing"

  # Aggressive compression for long generation
  python triattention_gguf.py --model model.gguf --prompt "Write a story" \\
      --budget 1024 --max-tokens 8192

  # Benchmark mode (runs with and without pruning)
  python triattention_gguf.py --model model.gguf --prompt "Explain AI" --benchmark

  # Disable pruning (full attention baseline)
  python triattention_gguf.py --model model.gguf --prompt "Hello" --no-prune
""")

    # Model and generation
    parser.add_argument("--model", type=str, required=True,
                        help="Path to GGUF model file")
    parser.add_argument("--prompt", type=str, required=True,
                        help="Input prompt text")
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS,
                        help=f"Maximum tokens to generate (default: {MAX_TOKENS})")
    parser.add_argument("--n-ctx", type=int, default=N_CTX,
                        help=f"Context window size (default: {N_CTX})")
    parser.add_argument("--n-gpu-layers", type=int, default=N_GPU_LAYERS,
                        help=f"GPU layers to offload, -1=all (default: {N_GPU_LAYERS})")
    parser.add_argument("--seed", type=int, default=SEED,
                        help=f"Random seed (default: {SEED})")

    # TriAttention parameters
    parser.add_argument("--budget", type=int, default=KV_BUDGET,
                        help=f"Max tokens in KV cache (default: {KV_BUDGET})")
    parser.add_argument("--window-size", type=int, default=WINDOW_SIZE,
                        help=f"Pruning interval in tokens (default: {WINDOW_SIZE})")
    parser.add_argument("--recent-tokens", type=int, default=RECENT_TOKENS,
                        help=f"Recent tokens to always keep (default: {RECENT_TOKENS})")
    parser.add_argument("--sink-tokens", type=int, default=SINK_TOKENS,
                        help=f"Initial sink tokens to always keep (default: {SINK_TOKENS})")
    parser.add_argument("--n-freq", type=int, default=N_DOMINANT_FREQ,
                        help=f"Dominant RoPE frequency bands (default: {N_DOMINANT_FREQ})")
    parser.add_argument("--no-prune", action="store_true",
                        help="Disable pruning (full attention baseline)")

    # Sampling
    parser.add_argument("--temperature", type=float, default=TEMPERATURE,
                        help=f"Sampling temperature (default: {TEMPERATURE})")
    parser.add_argument("--top-p", type=float, default=TOP_P,
                        help=f"Nucleus sampling threshold (default: {TOP_P})")
    parser.add_argument("--top-k", type=int, default=TOP_K,
                        help=f"Top-k sampling (default: {TOP_K})")
    parser.add_argument("--min-p", type=float, default=MIN_P,
                        help=f"Min-p sampling (default: {MIN_P})")
    parser.add_argument("--repeat-penalty", type=float, default=REPEAT_PENALTY,
                        help=f"Repeat penalty (default: {REPEAT_PENALTY})")

    # Modes
    parser.add_argument("--benchmark", action="store_true",
                        help="Run benchmark comparing pruned vs baseline")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose/debug logging")

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    # Load model
    logger.info(f"Loading model: {args.model}")
    llm = Llama(
        model_path=args.model,
        n_ctx=args.n_ctx,
        n_gpu_layers=args.n_gpu_layers,
        n_threads=N_THREADS,
        use_mlock=USE_MLOCK,
        verbose=VERBOSE,
        seed=args.seed,
    )
    logger.info("Model loaded.")

    # Tokenize prompt
    prompt_tokens = llm.tokenize(args.prompt.encode('utf-8'))
    logger.info(f"Prompt: {len(prompt_tokens)} tokens")

    # Sampling parameters
    sample_params = {
        'temperature': args.temperature,
        'top_p': args.top_p,
        'top_k': args.top_k,
        'min_p': args.min_p,
        'repeat_penalty': args.repeat_penalty,
        'presence_penalty': PRESENCE_PENALTY,
        'frequency_penalty': FREQUENCY_PENALTY,
    }

    if args.benchmark:
        # ── Benchmark mode ──────────────────────────────────────────────
        bench_tokens = min(args.max_tokens, 512)  # shorter for benchmark
        logger.info(f"Benchmark mode: generating {bench_tokens} tokens each run")

        # Run 1: With pruning
        logger.info("--- Run 1: TriAttention pruning ---")
        llm.reset()
        rope_freqs = get_rope_frequencies(llm, args.n_freq)
        pruner = TriAttentionPruner(
            rope_freqs=rope_freqs,
            budget=args.budget,
            window_size=args.window_size,
            recent_tokens=args.recent_tokens,
            sink_tokens=args.sink_tokens,
        )
        pruned_text, pruned_stats = run_benchmark(
            llm, prompt_tokens, pruner, bench_tokens, sample_params)

        # Run 2: Without pruning (baseline)
        logger.info("--- Run 2: Full attention baseline ---")
        llm.reset()
        baseline_pruner = TriAttentionPruner(
            rope_freqs=rope_freqs,
            budget=999999,  # effectively no pruning
            window_size=args.window_size,
            recent_tokens=args.recent_tokens,
            sink_tokens=args.sink_tokens,
        )
        baseline_text, baseline_stats = run_benchmark(
            llm, prompt_tokens, baseline_pruner, bench_tokens, sample_params)

        # Print benchmark report
        print("\n")
        print("=" * 68)
        print("  BENCHMARK RESULTS")
        print("=" * 68)
        print(f"  Model:          {args.model}")
        print(f"  Prompt tokens:  {len(prompt_tokens)}")
        print(f"  Gen tokens:     {bench_tokens}")
        print(f"  KV budget:      {args.budget}")
        print(f"  Window size:    {args.window_size}")
        print("-" * 68)
        print(f"  {'Metric':<30} {'Baseline':>15} {'TriAttention':>15}")
        print("-" * 68)
        print(f"  {'Tokens generated':<30} "
              f"{baseline_stats['generated_tokens']:>15} "
              f"{pruned_stats['generated_tokens']:>15}")
        print(f"  {'Time (sec)':<30} "
              f"{baseline_stats['total_time']:>15.2f} "
              f"{pruned_stats['total_time']:>15.2f}")
        print(f"  {'Tokens/sec':<30} "
              f"{baseline_stats['tokens_per_sec']:>15.1f} "
              f"{pruned_stats['tokens_per_sec']:>15.1f}")
        print(f"  {'Final cache size':<30} "
              f"{baseline_stats['final_cache_size']:>15} "
              f"{pruned_stats['final_cache_size']:>15}")
        print(f"  {'Prune events':<30} "
              f"{baseline_stats['prune_events']:>15} "
              f"{pruned_stats['prune_events']:>15}")
        print(f"  {'Total tokens pruned':<30} "
              f"{baseline_stats['total_pruned']:>15} "
              f"{pruned_stats['total_pruned']:>15}")

        if pruned_stats['final_cache_size'] > 0 and baseline_stats['final_cache_size'] > 0:
            mem_ratio = baseline_stats['final_cache_size'] / pruned_stats['final_cache_size']
            print(f"  {'KV memory reduction':<30} {'1.0x':>15} {f'{mem_ratio:.1f}x':>15}")
        print("=" * 68)

        print("\n--- Baseline output (first 300 chars) ---")
        print(baseline_text[:300])
        print("\n--- TriAttention output (first 300 chars) ---")
        print(pruned_text[:300])

    else:
        # ── Normal generation mode ──────────────────────────────────────
        rope_freqs = get_rope_frequencies(llm, args.n_freq)

        if args.no_prune:
            effective_budget = 999999
            logger.info("Pruning disabled (full attention mode)")
        else:
            effective_budget = args.budget
            logger.info(f"TriAttention pruning: budget={args.budget}, "
                        f"window={args.window_size}, recent={args.recent_tokens}, "
                        f"sinks={args.sink_tokens}")

        pruner = TriAttentionPruner(
            rope_freqs=rope_freqs,
            budget=effective_budget,
            window_size=args.window_size,
            recent_tokens=args.recent_tokens,
            sink_tokens=args.sink_tokens,
        )

        try:
            text, stats = generate(
                llm, prompt_tokens, pruner, args.max_tokens, sample_params)
            print()  # newline after streaming output

            logger.info(f"Generated {stats['generated_tokens']} tokens in "
                        f"{stats['total_time']:.2f}s "
                        f"({stats['tokens_per_sec']:.1f} tok/s)")
            if stats['prune_events'] > 0:
                logger.info(f"Pruning: {stats['prune_events']} events, "
                            f"{stats['total_pruned']} tokens evicted, "
                            f"final cache: {stats['final_cache_size']} tokens")

        except KeyboardInterrupt:
            print("\n\nGeneration interrupted.")
            sys.exit(0)


if __name__ == "__main__":
    main()