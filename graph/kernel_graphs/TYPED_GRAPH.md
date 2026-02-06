# Typed `.graph` Format

Kernel graphs in this directory now use a typed IR format (`# typed_graph:v1`) instead of legacy `Raw path:start-end` manifests.

## Goals

- No `Raw` nodes in kernel graphs.
- Explicit instruction-level nodes for PTX protocol validation.
- Stable, constrained search space for autotuning / RL.

## Directives

- Global declarations:
  - `buffer name=... space=tmem|smem|gmem|rmem dtype=... shape=[...]`
  - `barrier name=... scope=cta|cluster`
  - `desc name=... buf=... ...`
  - `default_tmem name=...`
- Section headers:
  - `# section:device`
  - `# section:host`
  - `# section:python`
- Control flow / structure:
  - `kernel_start ...`
  - `kernel_end`
  - `if cond=...` / `else` / `endif`
  - `for var=i iters=N` / `endfor`
- Operations:
  - `op <name> key=value ...`
  - Examples: `tcgen05_cp`, `tcgen05_mma`, `mbarrier_wait`, `tma_store_out`, `cta_group_set`, `persistent_loop_begin`.

## Validation

Validation now runs in three layers:

1. `state_machine.validate_graph`: graph/state/lifetime checks.
2. `ptx_spec.validate_graph_ptx_spec`: per-op PTX form checks.
3. `protocol_validator.validate_graph_protocol`: sequence/protocol checks (strict for typed graphs).

Run:

```bash
python tests/test_graph_ir.py
python llm/interface.py compile --graph graph/kernel_graphs/gemm1.graph --validate-only
python llm/interface.py compile --graph graph/kernel_graphs/grouped_gemm.graph --validate-only
```
