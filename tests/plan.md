TMEM per stage:

Score MMA: M=64, N=16 → 16 cols. Double-buffered = 32 cols.
Value MMA: M=128, N=32 (padded) → 32 cols. Need to read out after each tile, so 32 cols.

If we overlap score stage N+1 with value stage N:

Score slot: 16 cols
Value slot: 32 cols
Total: 48 cols
Single stage (no overlap): max(16, 32) = 32 cols if we reuse.

Tons of TMEM headroom either way.

Final design:


Grid: num_tokens blocks, 1 CTA per token
Warps: 1 producer + 1 MMA + 3 score-epi warpgroups (12) + 3 value-epi warpgroups (12) = 26 warps = 832 threads
Stages: 32 (2048 tokens / 64 per tile), double-buffered SMEM

Per tile of 64 KV tokens:

1. Producer warp: TMA load Kc[64,512] + Kp[64,64] into SMEM (double-buffered)
       ↓ tma_mbar
2. Score MMA (lane 0): Kc[64,512] @ Q_nope[16,512]^T → [64,16] in TMEM
                      + Kp[64,64] @ Q_pe[16,64]^T (enable_input_d=1)
       ↓ score_mma_mbar
3. Score epilogue (3 warpgroups, 12 warps):
   - tcgen05.ld [64,16] from TMEM into registers
   - Warp-shuffle online softmax across 64 tokens per head
   - Write [64,16] bf16 unnormalized weights to SMEM
   - Update running m_state, l_state in SMEM
       ↓ score_epi_mbar
4. Value MMA (lane 1): 4 sequential tiles, each:
   - A = Kc_slice[128,64] from SMEM (reuse same Kc buffer)
   - B = weights[32,64] from SMEM (padded 16→32)
   - Output [128,32] in TMEM (32 cols)
   - Collectors (enable_input_d=1) accumulate across 4 A-tiles
       ↓ value_mma_mbar
5. Value epilogue (3 warpgroups, 12 warps):
   - tcgen05.ld [512,16] result from TMEM (reading across 4 tile offsets)
   - Rescale running o_accum in SMEM: o_accum *= exp2(m_old - m_new)
   - Add this tile: o_accum += exp2(m_tile - m_new) * o_tile
       ↓ value_epi_mbar (signals producer: SMEM slot free)

After all 32 tiles:
- Final output = o_accum / l_state → write [16,512] bf16 to global
- LSE = m_state + log2(l_state) → write [16] fp32 to global

SMEM: ~199KB (fits in 228KB, 1 CTA/SM)
TMEM: 48 cols (fits in 512)