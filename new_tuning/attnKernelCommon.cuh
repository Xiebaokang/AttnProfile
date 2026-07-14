#pragma once

#include "utils.cuh"
#include <type_traits>
#include <utility>

template <size_t I, size_t End, typename F>
__device__ __forceinline__ void static_for(F const& f) {
    if constexpr (I < End) {
        f(std::integral_constant<size_t, I>{});
        static_for<I + 1, End>(f);
    }
}

template <int N, int Tile>
inline constexpr int round_up_to_tile_v = ((N + Tile - 1) / Tile) * Tile;

// Register-source A layout used by Hopper GMMA m64k16. Each thread owns four
// packed half2 values covering two rows and the two k8 halves.
template <int DIM>
__device__ __forceinline__ void load_q_global_fragment(
    uint32_t (&q_regs)[4], const fp16 *Q, int H, int SQO,
    int bs, int hn, int q_row_base, int q_col_base
) {
    const int tid_in_wg = threadIdx.x & 127;
    const int t0 = tid_in_wg & 3;
    const int t1 = (tid_in_wg >> 2) & 7;
    const int t2 = tid_in_wg >> 5;
    const int row0 = q_row_base + t1 + 16 * t2;
    const int row1 = row0 + 8;
    const int col0 = q_col_base + 2 * t0;
    const int col1 = col0 + 8;
    const fp16 *q0 = Q + ((static_cast<size_t>(bs) * H + hn) * SQO + row0) * DIM;
    const fp16 *q1 = Q + ((static_cast<size_t>(bs) * H + hn) * SQO + row1) * DIM;
    q_regs[0] = *reinterpret_cast<const uint32_t*>(q0 + col0);
    q_regs[1] = *reinterpret_cast<const uint32_t*>(q1 + col0);
    q_regs[2] = *reinterpret_cast<const uint32_t*>(q0 + col1);
    q_regs[3] = *reinterpret_cast<const uint32_t*>(q1 + col1);
}


template <int BM, int BN, int DIM, int NUM_SMEM, int P_SMEM_K_TILES, int Q_REG_K_TILES>
union SMemWS {
    static constexpr int MMA_K = 16;
    static constexpr int P_SMEM_COLS = ((P_SMEM_K_TILES * MMA_K + 63) / 64) * 64;
    static constexpr int Q_SMEM_K_TILES = DIM / MMA_K - Q_REG_K_TILES;
    static constexpr int Q_SMEM_COLS = Q_SMEM_K_TILES * MMA_K;
    struct {
    alignas(128) fp16 Q[BM*Q_SMEM_COLS];
    alignas(128) fp16 K[BN*DIM*NUM_SMEM];
    alignas(128) fp16 V[BN*DIM*NUM_SMEM];
    alignas(128) fp16 P[BM*P_SMEM_COLS];
    alignas(8) uint64_t Qmbar;
    alignas(8) uint64_t Kempty[NUM_SMEM];
    alignas(8) uint64_t Vempty[NUM_SMEM];
    alignas(8) uint64_t Kfull[NUM_SMEM];
    alignas(8) uint64_t Vfull[NUM_SMEM];
    };
    alignas(128) fp16 O[BM*DIM];
};

