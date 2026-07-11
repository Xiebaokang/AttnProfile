#ifndef ATTN_PROFILE_UTILS_CUH
#define ATTN_PROFILE_UTILS_CUH

#include "tools.cuh"

#include <random>


struct StaticChunkScheduler {
    int total_tiles;
    int chunk;

    struct WorkTileInfo {
        int tile_idx;
        int tile_end;

        __device__ bool is_valid() const {
            return tile_idx < tile_end;
        }
    };

    __device__ WorkTileInfo get_initial_work() const {
        int start = blockIdx.x * chunk;
        int end = min(start + chunk, total_tiles);
        return {start, end};
    }

    __device__ WorkTileInfo get_next_work(WorkTileInfo cur) const {
        return {cur.tile_idx + 1, cur.tile_end};
    }
};

template <int BlockMajorSize, int BlockMinorSize, bool swizzle=true, bool padding=false>
__host__ static inline CUtensorMap create_tensor_map(fp16* gmem_ptr, int batch1, int batch2, int global_height, int global_width) {
    CUtensorMap tma_map;
    void* gmem_address = (void*)gmem_ptr;
    static_assert(BlockMinorSize >= 64);
    assert(global_width % 64 == 0);
    uint64_t gmem_prob_shape[5] = {
        64,
        (uint64_t)global_height,
        (uint64_t)global_width / 64,
        (uint64_t)batch2,
        (uint64_t)batch1
    };  // x, y, z
    uint64_t gmem_prob_stride[5] = {
        sizeof(fp16) * global_width,
        sizeof(fp16) * 64,
        sizeof(fp16) * global_height * global_width,
        sizeof(fp16) * global_height * global_width * batch2,
        0
    };  //
    uint32_t smem_box_shape[5] = {
        padding ? 72 : 64,
        uint32_t(BlockMajorSize),
        uint32_t(BlockMinorSize / 64),
        1,
        1
    };
    uint32_t smem_box_stride[5] = {1, 1, 1, 1, 1};

    CUresult result = cuTensorMapEncodeTiled(
        &tma_map, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, 5, gmem_address, gmem_prob_shape,
        gmem_prob_stride, smem_box_shape, smem_box_stride, CU_TENSOR_MAP_INTERLEAVE_NONE,
        swizzle ? CU_TENSOR_MAP_SWIZZLE_128B : CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

    assert(result == CUDA_SUCCESS);
    return tma_map;
}

using AttnRunner = void (*)(fp16*, fp16*, fp16*, fp16*);

template<int B, int H, int S, int D, int BM, int BN_TILE=128>
void verify_attn(AttnRunner kernel) {

    int dev = 0;
    cudaDeviceProp prop{};
    cudaCheck(cudaGetDevice(&dev));
    cudaCheck(cudaGetDeviceProperties(&prop, dev));
    if (prop.major < 9) {
        printf("This kernel uses Hopper-only instructions (TMA/WGMMA). Need SM90+ GPU.\n");
        return;
    }

    constexpr int SQO = ((S + BM - 1) / BM) * BM;
    constexpr int SKV = ((S + BN_TILE - 1) / BN_TILE) * BN_TILE;
    const size_t qo_numel = static_cast<size_t>(B) * H * SQO * D;
    const size_t kv_numel = static_cast<size_t>(B) * H * SKV * D;
    const size_t qo_bytes = qo_numel * sizeof(fp16);
    const size_t kv_bytes = kv_numel * sizeof(fp16);
    std::vector<fp16> hQ(qo_numel, 0.0f), hK(kv_numel, 0.0f), hV(kv_numel, 0.0f);
    std::vector<fp16> hO(qo_numel), hORef(qo_numel, 0.0f);

    // std::random_device rd;
    const uint32_t seed = 42;
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> qk_dist(-0.5f, 0.5f);
    std::uniform_real_distribution<float> v_dist(-1.0f, 1.0f);
    printf("verify_attn seed=%u\n", seed);

    auto idxQO = [=](int b, int h, int s, int d) {
        return (((static_cast<size_t>(b) * H + h) * SQO + s) * D + d);
    };
    auto idxKV = [=](int b, int h, int s, int d) {
        return (((static_cast<size_t>(b) * H + h) * SKV + s) * D + d);
    };

    for (int b = 0; b < B; ++b) {
        for (int h = 0; h < H; ++h) {
            for (int s = 0; s < S; ++s) {
                for (int d = 0; d < D; ++d) {
                    hQ[idxQO(b, h, s, d)] = __float2half_rn(qk_dist(gen));
                    hK[idxKV(b, h, s, d)] = __float2half_rn(qk_dist(gen));
                    hV[idxKV(b, h, s, d)] = __float2half_rn(v_dist(gen));
                }
            }
        }
    }

    fp16 *dQ = nullptr;
    fp16 *dK = nullptr;
    fp16 *dV = nullptr;
    fp16 *dO = nullptr;
    cudaCheck(cudaMalloc(&dQ, qo_bytes));
    cudaCheck(cudaMalloc(&dK, kv_bytes));
    cudaCheck(cudaMalloc(&dV, kv_bytes));
    cudaCheck(cudaMalloc(&dO, qo_bytes));
    cudaCheck(cudaMemcpy(dQ, hQ.data(), qo_bytes, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(dK, hK.data(), kv_bytes, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(dV, hV.data(), kv_bytes, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemset(dO, 0, qo_bytes));

    kernel(dQ, dK, dV, dO);

    cudaCheck(cudaGetLastError());
    cudaCheck(cudaDeviceSynchronize());
    cudaCheck(cudaMemcpy(hO.data(), dO, qo_bytes, cudaMemcpyDeviceToHost));

    const float attn_scale = sqrtf(1.0f / D) * 1.44269504f;
    std::vector<fp32> scores(S, 0.0f);
    std::vector<fp32> row_acc(D, 0.0f);

    for (int b = 0; b < B; ++b) {
        for (int h = 0; h < H; ++h) {
            for (int s1 = 0; s1 < S; ++s1) {
                for (int d = 0; d < D; ++d) {
                    row_acc[d] = 0.0f;
                }

                float row_max = -FLT_MAX;
                float row_sum = 0.0f;

                for (int iw = 0; iw < S; iw += BN_TILE) {
                    const int tile_end = std::min(iw + BN_TILE, S);
                    float tile_max = -FLT_MAX;

                    for (int s2 = iw; s2 < tile_end; ++s2) {
                        float score = 0.0f;
                        for (int d = 0; d < D; ++d) {
                            score += __half2float(hQ[idxQO(b, h, s1, d)])
                                   * __half2float(hK[idxKV(b, h, s2, d)]);
                        }
                        scores[s2] = score;
                        tile_max = fmaxf(tile_max, score);
                    }

                    const float new_max = fmaxf(row_max, tile_max);
                    const float old_scale = (row_max == -FLT_MAX)
                                          ? 0.0f
                                          : exp2f(row_max * attn_scale - new_max * attn_scale);

                    for (int d = 0; d < D; ++d) {
                        row_acc[d] *= old_scale;
                    }

                    float tile_sum = 0.0f;
                    for (int s2 = iw; s2 < tile_end; ++s2) {
                        const float prob = exp2f(scores[s2] * attn_scale - new_max * attn_scale);
                        tile_sum += prob;

                        // The kernel casts the online softmax scores to fp16 before PV.
                        const float prob_for_pv = __half2float(__float2half_rn(prob));
                        for (int d = 0; d < D; ++d) {
                            row_acc[d] += prob_for_pv * __half2float(hV[idxKV(b, h, s2, d)]);
                        }
                    }

                    row_sum = row_sum * old_scale + tile_sum;
                    row_max = new_max;
                }

                const float inv_sum = 1.0f / row_sum;
                for (int d = 0; d < D; ++d) {
                    hORef[idxQO(b, h, s1, d)] = __float2half_rn(row_acc[d] * inv_sum);
                }
            }
        }
    }

    float max_o_diff = 0.0f;
    int max_o_b = 0, max_o_h = 0, max_o_s = 0, max_o_d = 0;
    int o_mismatch_count = 0;
    int left_mismatch_count = 0;
    int right_mismatch_count = 0;
    constexpr float o_abs_tol = 5e-2f;
    for (int b = 0; b < B; ++b) {
        for (int h = 0; h < H; ++h) {
            for (int s = 0; s < S; ++s) {
                for (int d = 0; d < D; ++d) {
                    const float got = __half2float(hO[idxQO(b, h, s, d)]);
                    const float ref = __half2float(hORef[idxQO(b, h, s, d)]);
                    const float diff = fabsf(got - ref);
                    if (diff > max_o_diff) {
                        max_o_diff = diff;
                        max_o_b = b;
                        max_o_h = h;
                        max_o_s = s;
                        max_o_d = d;
                    }
                    if (diff > o_abs_tol) {
                        ++o_mismatch_count;
                        if (d < D / 2) {
                            ++left_mismatch_count;
                        } else {
                            ++right_mismatch_count;
                        }
                    }
                }
            }
        }
    }

    printf("[D] hO attention max_abs_diff=%.6f at b=%d h=%d s=%d d=%d got=%.6f ref=%.6f mismatches(abs>%.4f)=%d/%zu left=%d right=%d\n",
           max_o_diff, max_o_b, max_o_h, max_o_s, max_o_d,
           __half2float(hO[idxQO(max_o_b, max_o_h, max_o_s, max_o_d)]),
           __half2float(hORef[idxQO(max_o_b, max_o_h, max_o_s, max_o_d)]),
           o_abs_tol, o_mismatch_count, static_cast<size_t>(B) * H * S * D,
           left_mismatch_count, right_mismatch_count);

    cudaCheck(cudaFree(dQ));
    cudaCheck(cudaFree(dK));
    cudaCheck(cudaFree(dV));
    cudaCheck(cudaFree(dO));
}

template<int B, int H, int S, int D, int BM, int BN>
void benchmark_attn(AttnRunner kernel) {
    constexpr int SQO = ((S + BM - 1) / BM) * BM;
    constexpr int SKV = ((S + BN - 1) / BN) * BN;
    const size_t qo_bytes = static_cast<size_t>(B) * H * SQO * D * sizeof(fp16);
    const size_t kv_bytes = static_cast<size_t>(B) * H * SKV * D * sizeof(fp16);

    fp16 *dQ = nullptr;
    fp16 *dK = nullptr;
    fp16 *dV = nullptr;
    fp16 *dO = nullptr;

    cudaCheck(cudaMalloc(&dQ, qo_bytes));
    cudaCheck(cudaMalloc(&dK, kv_bytes));
    cudaCheck(cudaMalloc(&dV, kv_bytes));
    cudaCheck(cudaMalloc(&dO, qo_bytes));

    cudaCheck(cudaMemset(dQ, 0, qo_bytes));
    cudaCheck(cudaMemset(dK, 0, kv_bytes));
    cudaCheck(cudaMemset(dV, 0, kv_bytes));
    cudaCheck(cudaMemset(dO, 0, qo_bytes));

    for (int i = 0; i < 60; ++i) {
        kernel(dQ, dK, dV, dO);
        cudaCheck(cudaGetLastError());
    }
    cudaCheck(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    cudaCheck(cudaEventCreate(&start));
    cudaCheck(cudaEventCreate(&stop));

    cudaCheck(cudaEventRecord(start));
    for (int i = 0; i < 250; ++i) {
        kernel(dQ, dK, dV, dO);
        cudaCheck(cudaGetLastError());
    }
    cudaCheck(cudaEventRecord(stop));
    cudaCheck(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    cudaCheck(cudaEventElapsedTime(&elapsed_ms, start, stop));

    cudaCheck(cudaEventDestroy(start));
    cudaCheck(cudaEventDestroy(stop));

    const double avg_ms = static_cast<double>(elapsed_ms) / 250;
    const double flops = 4.0 * static_cast<double>(B) * H * S * S * D;
    const double tflops = flops / (avg_ms * 1.0e-3) / 1.0e12;

    printf("%d, %d, %d, %d, %d, %d, %.6f, %.3f\n", B, H, S, D, BM, BN, avg_ms, tflops);

    cudaCheck(cudaFree(dQ));
    cudaCheck(cudaFree(dK));
    cudaCheck(cudaFree(dV));
    cudaCheck(cudaFree(dO));
}

template<int B, int H, int S, int D, int BM, int BN>
void benchmark_attn_ncu(AttnRunner kernel) {
    constexpr int SQO = ((S + BM - 1) / BM) * BM;
    constexpr int SKV = ((S + BN - 1) / BN) * BN;
    const size_t qo_bytes = static_cast<size_t>(B) * H * SQO * D * sizeof(fp16);
    const size_t kv_bytes = static_cast<size_t>(B) * H * SKV * D * sizeof(fp16);

    fp16 *dQ = nullptr;
    fp16 *dK = nullptr;
    fp16 *dV = nullptr;
    fp16 *dO = nullptr;

    cudaCheck(cudaMalloc(&dQ, qo_bytes));
    cudaCheck(cudaMalloc(&dK, kv_bytes));
    cudaCheck(cudaMalloc(&dV, kv_bytes));
    cudaCheck(cudaMalloc(&dO, qo_bytes));

    cudaCheck(cudaMemset(dQ, 0, qo_bytes));
    cudaCheck(cudaMemset(dK, 0, kv_bytes));
    cudaCheck(cudaMemset(dV, 0, kv_bytes));
    cudaCheck(cudaMemset(dO, 0, qo_bytes));

    for (int i = 0; i < 100; ++i) {
        kernel(dQ, dK, dV, dO);
        cudaCheck(cudaGetLastError());
    }
    cudaCheck(cudaDeviceSynchronize());

    kernel(dQ, dK, dV, dO);
    cudaCheck(cudaGetLastError());
    cudaCheck(cudaDeviceSynchronize());

    cudaCheck(cudaFree(dQ));
    cudaCheck(cudaFree(dK));
    cudaCheck(cudaFree(dV));
    cudaCheck(cudaFree(dO));
}

#endif  // ATTN_PROFILE_UTILS_CUH
