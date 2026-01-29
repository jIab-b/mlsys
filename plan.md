# Automated Kernel Optimization Agent — Project Plan

## Target

Optimized CUDA/PTX kernels for NVIDIA B200 (sm_100), targeting 46+ kernel definitions from the FlashInfer Trace eval suite (GEMM, attention variants, RMSNorm, MoE, sampling, etc.). Solutions must beat reference PyTorch implementations and compete with existing LLM-generated CUDA/Triton solutions.

---

## Architecture Overview

```
Strategy Search (LLM + enumeration)
    |
    v
Strategy Config (structured, bounded, enumerable)
    |
    v
UOp Graph Construction (mechanical, given strategy)
    |
    v
Constraint Validator (Rust-level safety — graph-based)
    |
    v
PTX Emission
    |
    v
Compile (load_inline) + Correctness Check + Benchmark
    |
    v
Feedback to Strategy Search
```

---

## Layer 1: Strategy Config

A structured dataclass representing high-level kernel design choices. Small discrete search space, pruned by hard hardware constraints.

**Fields:**
- Tiling: tile_m, tile_n, tile_k (from finite set per kernel type)
- Pipeline: persistent vs non-persistent, num_stages (2..7)
- Warp roles: warp_specialized (bool), num_producer_warps, num_consumer_warps
- Memory: smem_bytes (derived, <= 228KB), tmem allocation, regs_per_thread (<= 255)
- Compute: MMA atom selection (e.g., tcgen05_mma_f16_64x256x32)

**Hard constraint pruning (is_valid):**
- smem budget: num_stages * tile_smem_cost <= 228KB
- register budget: regs_per_thread <= 255, occupancy check
- persistent requires grid_blocks >= SM count
- tile dimensions must be legal for chosen MMA atom
- bottleneck estimation (compute-bound vs memory-bound) for ranking

**Who drives it:** LLM reasons about workload shape + hardware limits to propose or rank configs. Alternatively, brute-force enumerate all valid configs and autotune.

---

## Layer 2: UOp Graph

A concurrent dataflow graph with synchronization edges. Instantiated mechanically from a strategy config. This is the core IR — needed for Rust-level correctness guarantees.

### Node types

**Atoms (leaf nodes) — PTX inline functions with constraint annotations:**
- `TMALoad(src, dst, barrier)` — issue: one_thread_per_CTA, produces: mbar_pending
- `MBarInit(bar)` — issue: one_thread_per_CTA, produces: mbar_initialized
- `MBarArrive(bar)` — issue: one_thread_per_CTA, requires: mbar_pending, produces: mbar_arrived
- `MBarWait(bar)` — issue: all_threads_in_warp, requires: mbar_arrived, produces: smem_valid
- `TcGen05MMA(A, B, C)` — issue: one_thread_in_warp, requires: smem_valid(A), smem_valid(B)
- `Store(src, dst)` — issue: all_threads_in_warp

**Composite nodes (subgraphs):**
- `Softmax(x)` — exp, sum, div
- `TopK(scores, k)` — sort/select
- `ReLU(x)`, `RMSNorm(x, w)`, etc.

### Edge types
- **Data edges:** standard producer-consumer (tensor flows)
- **Barrier edges:** cross-warp/cross-CTA synchronization (mbar arrive → wait)
- **Order edges:** sequencing constraints (e.g., epilogue after last MMA)

### Subgraph structure
Warp-specialized kernels have multiple disconnected subgraphs (producer vs consumer) linked only by barrier edges. Pipeline stages create repeated subgraph copies with rotating buffer indices.

### Why a graph is required
Global safety properties cannot be checked locally per-atom:
- **No data races:** two subgraphs never write same smem without intervening barrier
- **No deadlocks:** every Wait has a reachable matching Arrive
- **No use-after-free:** smem buffer not reused before all consumers finish
- **Pipeline completeness:** every stage is both produced and consumed

---

## Layer 3: Constraint Validator

A state machine over barrier and memory states, operating on the UOp graph. Enforces Rust-level safety — if it passes, no memory corruption or race conditions are possible.

### Per-atom annotations
Each PTX inline function carries:
- **Issue semantics:** which threads must execute it (one per warp, one per CTA, one per cluster, all threads in warp)
- **Preconditions:** required barrier/memory states before issuing
- **Postconditions:** barrier/memory states produced after issuing
- **Conflicts:** what concurrent operations are illegal (e.g., smem read during TMA write to same address)

### Global checks (graph-level)
- Barrier state machine: init → pending → arrived → complete → (reinit). No skipped states.
- Reachability: every Wait node is reachable from a corresponding Arrive
- Memory safety: no aliased smem access without synchronization edge between them
- Completeness: all produced data is consumed, no dangling barriers

### Design principle
Anything not explicitly proven safe by the validator is rejected. The LLM/compiler cannot emit code that bypasses these checks. Invalid sequences are caught at graph construction time, never at runtime.

---

## Layer 4: PTX Emission

Given a validated UOp graph, emit PTX mechanically.

### Layout algebra (reimplemented from CuTe in Python)
- Shape + Stride representation for all tensors
- Thread-to-data mapping computation
- Swizzle pattern generation for bank-conflict-free smem access
- Address arithmetic for TMA descriptors

This is pure math — same formulas as CuTe, but as runtime Python values rather than C++ template types. Computed once per strategy config, baked into emitted PTX as constants.

### Code generation
- Each atom emits its PTX snippet with concrete register names and addresses
- Control flow (loops, conditionals, warp role branching) generated from graph structure
- Pipeline stage rotation (circular buffer index math) generated mechanically from num_stages

### Output
Raw PTX loaded via `torch.utils.cpp_extension.load_inline` or `cuModuleLoad`. Fast compile-validate cycle — no CUTLASS headers, no template instantiation.

---

## Layer 5: LLM Role

The LLM operates at two levels:

### Strategy selection (primary role)
- Analyzes workload shape (context length, batch size, head dims)
- Reasons about compute-bound vs memory-bound
- Proposes strategy configs from the valid set
- Ranks candidates by estimated performance

### Fallback code generation (secondary role)
- For novel kernels or edge cases the compiler can't handle
- Generates glue code (index math, masking, epilogues) using atom library
- Output validated by constraint system before compilation
- Retry loop on validation or correctness failure

### Tooling
- PTX MCP server: LLM queries instruction latencies, register costs, scheduling rules for sm_100
- Eval suite access: LLM reads kernel definitions, existing solutions, benchmark traces
- On-site open-source model for low-latency iteration

---

## Kernel Coverage

### From eval suite (46 definitions across 8 categories)

| Category | Count | Key challenge |
|----------|-------|---------------|
| GEMM | 8 | Tile shape selection, persistent vs non-persistent |
| RMSNorm | 9 | Reduction pattern, fused add variants |
| GQA paged attention | 8 | Paged KV cache gather, online softmax |
| GQA ragged attention | 2 | Variable-length sequences |
| MLA paged attention | 4 | Compressed KV (512d), positional encoding split |
| DSA sparse attention | 3 | Two-phase: index selection (FP8) then sparse attn |
| GDN | 2 | Gated dense network |
| MoE | 1 | FP8 block scaling, expert routing |
| Sampling | 9 | Top-k, top-p, combined variants |

### Pipeline patterns (covers ~90% of kernels)
1. **Simple non-persistent** — RMSNorm, sampling (single pass over data)
2. **Persistent GEMM** — warp-specialized producer/consumer with TMA pipeline
3. **Paged attention** — gather from page table + MMA + online softmax
4. **Two-pass sparse** — index kernel (lightweight, O(seq_len)) then attention kernel (heavy, O(topk))
5. **Reduction** — tree reduction within warp/block (RMSNorm, softmax denominators)

---

## Eval Infrastructure (existing)

- **Definitions:** 46 JSON files with shapes, dtypes, constraints, reference PyTorch implementations
- **Workloads:** 35 JSONL files with ~7000+ test cases, real tensor data in safetensors (15GB)
- **Traces:** 5900+ benchmark records with correctness + latency data across H100/B200
- **Existing solutions:** 312 implementations (CUDA/Triton/Python) from Claude, GPT-o3, Gemini — baseline to beat
- **Validation:** correctness against reference (max relative/absolute error), performance (latency, speedup factor)

---

## Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Output format | Raw PTX (not CuTe C++) | Fast compile loop, full sm_100 control, constraint system maps 1:1 to instructions |
| Layout math | Python reimplementation of CuTe algebra | Same math, inspectable/searchable, no C++ template overhead |
| Correctness model | Graph-based constraint validator | Rust-level safety requires global properties (reachability, race freedom) that need graph edges |
| Strategy search | LLM-guided + enumeration | Small discrete space, LLM reasons about workload, brute-force as fallback |
| Control flow | Compiler-generated from graph | Mechanical parts (barriers, pipeline rotation, index math) are more reliable than LLM generation |
| LLM role | Strategy selection + fallback glue code | LLM is good at high-level reasoning, bad at register allocation and barrier ordering |
