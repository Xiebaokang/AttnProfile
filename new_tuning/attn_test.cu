#include "attnBaselineKernel.cuh"
#include "attn2StageKernel.cuh"
#include "attnForNKernel.cuh"

template<
    int B, int H, int S, int D, int BM, int BN, int NUM_SMEM,
    uint32_t PRODUCER_REG_DEALLOC,
    uint32_t CONSUMER_REG_ALLOC,
    int P_SMEM_K_TILES,
    int Q_REG_K_TILES,
    int KERNEL_IDX,
    int NUM_THREADS=384
>
void runAttn(fp16 *Q, fp16 *K, fp16 *V, fp16 *O) {
    if constexpr (KERNEL_IDX == 1) {
        runAttnWSBaselineKernel<
            B, H, S, D, BM, BN, NUM_SMEM,
            PRODUCER_REG_DEALLOC, CONSUMER_REG_ALLOC, NUM_THREADS>(Q, K, V, O);
    }else if constexpr (KERNEL_IDX == 2) {
        runAttnWS2StageKernel<
            B, H, S, D, BM, BN, NUM_SMEM,
            PRODUCER_REG_DEALLOC, CONSUMER_REG_ALLOC, P_SMEM_K_TILES, Q_REG_K_TILES, NUM_THREADS>(Q, K, V, O);
    } else if constexpr (KERNEL_IDX == 3) {
        runAttnWSForNKernel<
            B, H, S, D, BM, BN, NUM_SMEM,
            PRODUCER_REG_DEALLOC, CONSUMER_REG_ALLOC, P_SMEM_K_TILES, Q_REG_K_TILES, NUM_THREADS>(Q, K, V, O);
    } else {
        static_assert(
            KERNEL_IDX >= 1 && KERNEL_IDX <= 3,
            "KERNEL_IDX must be 1 (baseline), 2 (smem-P), or 3 (ForN smem-P)");
    }
}

// export CUDA_VISIBLE_DEVICES=1
// nvcc -std=c++17 -arch=sm_90a -O3 attn_test.cu -o ./bin/attn_test -lcuda -Xptxas=-v
// ncu --set full --launch-skip 100 --launch-count 1 ./bin/attn_test
int main() {
    constexpr int BM = 128;
    constexpr int BN = 176;
    constexpr int B = 1;
    constexpr int H = 16;
    constexpr int S = 32768;
    constexpr int D = 128;

    constexpr int NUM_SMEM = 2;
    constexpr uint32_t PRODUCER_REG_DEALLOC = 24;
    constexpr uint32_t CONSUMER_REG_ALLOC = 240;
    constexpr int P_SMEM_K_TILES = 0;
    constexpr int Q_REG_K_TILES = 0;  // last 64 D columns use RS
    constexpr int NUM_THREADS = 384;

    // The first D/16-Q_REG_K_TILES tiles use SS; the remaining tiles use
    // direct global-to-register loads and QK RS.
    constexpr int KERNEL_IDX = 1;
    auto *kernel = runAttn<
        B, H, S, D, BM, BN, NUM_SMEM,
        PRODUCER_REG_DEALLOC, CONSUMER_REG_ALLOC,
        P_SMEM_K_TILES, Q_REG_K_TILES, KERNEL_IDX, NUM_THREADS>;
    // verify_attn<B, H, S, D, BM, BN>(kernel);
    benchmark_attn<B, H, S, D, BM, BN>(kernel);
    // benchmarkAttnRegSweep<B, H, S, D, BM, BN, true, 8>();
    // benchmark_attn_ncu<B, H, S, D, BM, BN>(kernel);
    return 0;
}
