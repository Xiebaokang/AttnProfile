#include "tools.cuh"
#include <random>

template <int BM, int BN, int DIM, int NUM_SMEM=1>
struct SMemWS {
    alignas(128) fp16 Q[BM*DIM];
    alignas(128) fp16 K[BN*DIM*NUM_SMEM];
    alignas(128) fp16 V[BN*DIM*NUM_SMEM];
    alignas(128) fp16 O[BM*DIM];
    alignas(8) uint64_t Qmbar;
    alignas(8) uint64_t Kempty[NUM_SMEM];
    alignas(8) uint64_t Vempty[NUM_SMEM];
    alignas(8) uint64_t Kfull[NUM_SMEM];
    alignas(8) uint64_t Vfull[NUM_SMEM];
    
};


template <int BM, int BN, int DIM, int NUM_SMEM=1, int NUM_CONSUMER=2>
struct SMemWSCX {
    alignas(128) fp16 Q[BM*DIM];
    alignas(128) fp16 K[BN*DIM*NUM_SMEM*NUM_CONSUMER];
    alignas(128) fp16 V[BN*DIM*NUM_SMEM*NUM_CONSUMER];
    alignas(128) fp16 O[BM*DIM*NUM_CONSUMER];
    alignas(128) fp32 smem_sum[BM*NUM_CONSUMER];
    alignas(128) fp32 smem_max[BM*NUM_CONSUMER];
    alignas(8) uint64_t Ofull;
    alignas(8) uint64_t Oempty;
    alignas(8) uint64_t Qfull;
    alignas(8) uint64_t Qempty;
    alignas(8) uint64_t Kempty[NUM_SMEM];
    alignas(8) uint64_t Vempty[NUM_SMEM];
    alignas(8) uint64_t Kfull[NUM_SMEM];
    alignas(8) uint64_t Vfull[NUM_SMEM];
};


template <int BM, int BN, int DIM, int NUM_SMEM_Q, int NUM_SMEM_KV>
struct SMemWSDoubleQ {
    alignas(128) fp16 Q[BM/2*DIM*NUM_SMEM_Q];
    alignas(128) fp16 K[BN*DIM*NUM_SMEM_KV];
    alignas(128) fp16 V[BN*DIM*NUM_SMEM_KV];
    alignas(128) fp16 O[BM*DIM];
    alignas(8) uint64_t Qempty[NUM_SMEM_Q];
    alignas(8) uint64_t Kempty[NUM_SMEM_KV];
    alignas(8) uint64_t Vempty[NUM_SMEM_KV];
    alignas(8) uint64_t Qfull[NUM_SMEM_Q];
    alignas(8) uint64_t Kfull[NUM_SMEM_KV];
    alignas(8) uint64_t Vfull[NUM_SMEM_KV];
};

template <int BM, int BN, int DIM, int NUM_SMEM_Q, int NUM_SMEM_KV>
struct SMemWSDoubleQPVSS {
    static constexpr int MMA_M = 64;
    static constexpr int NUM_CONSUMER = 2;
    alignas(128) fp16 Q[BM/2*DIM*NUM_SMEM_Q];
    alignas(128) fp16 K[BN*DIM*NUM_SMEM_KV];
    alignas(128) fp16 V[BN*DIM*NUM_SMEM_KV];
    union {
        alignas(128) fp16 P[NUM_CONSUMER*MMA_M*BN];
        alignas(128) fp16 O[BM*DIM];
    };
    alignas(8) uint64_t Qempty[NUM_SMEM_Q];
    alignas(8) uint64_t Kempty[NUM_SMEM_KV];
    alignas(8) uint64_t Vempty[NUM_SMEM_KV];
    alignas(8) uint64_t Qfull[NUM_SMEM_Q];
    alignas(8) uint64_t Kfull[NUM_SMEM_KV];
    alignas(8) uint64_t Vfull[NUM_SMEM_KV];
};


template<int BM, int BN, int DIM, int NUM_THREADS, int NUM_SMEM, int NUM_CONSUMER, int N>
__global__  __launch_bounds__(NUM_THREADS) 
void attnWSCXKernel(
    int B, int H, int S, 
    const __grid_constant__ CUtensorMap tensorMapQ, 
    const __grid_constant__ CUtensorMap tensorMapK, 
    const __grid_constant__ CUtensorMap tensorMapV, 
    const __grid_constant__ CUtensorMap tensorMapO
) {
    static_assert(NUM_THREADS >= 512, "attnWSCXKernel requires four warpgroups: consumer0, consumer1, accum, producer.");
    static_assert(NUM_CONSUMER == 2, "attnWSCXKernel currently maps two consumer warpgroups to wg_idx 0 and 1.");
    // WS attention
    const int bs = blockIdx.z;
    const int hn = blockIdx.y;
    const int by = blockIdx.x;
    const int tid = threadIdx.x;
    uint32_t wg_idx = tid >> 7;   // wg
    uint32_t lane_id = tid & 31;
    uint32_t warp_id_in_wg = (tid >> 5) & 0x3;

    // mma size
    assert((DIM >= 256 || DIM == 16 || DIM == 32 || DIM == 64 || DIM == 128) && "DIM ERROR!");
    assert((BN >= 256 || BN == 16 || BN == 32 || BN == 64 || BN == 128) && "BN ERROR!");
    constexpr int MMA_M = 64;
    constexpr int QK_MMA_N = BN <= 256 ? BN : 256;
    constexpr int PV_MMA_N = DIM <= 128 ? DIM : 128;
    constexpr int MMA_K = 16;
    const fp32 scale = sqrt((1.0f / DIM)) * 1.44269504f;  // log2(e)

    // setting shared memory
    extern __shared__ __align__(128) uint8_t smem[];
    SMemWSCX<BM, BN, DIM, NUM_SMEM, NUM_CONSUMER> &s = *reinterpret_cast<SMemWSCX<BM, BN, DIM, NUM_SMEM, NUM_CONSUMER>*>(smem);
    fp16 *sQ = s.Q, *sK = s.K, *sV = s.V, *sO = s.O;
    fp32 *smem_max = s.smem_max, *smem_sum = s.smem_sum;
    Barrier *Qempty = &s.Qempty, *Qfull = &s.Qfull, *Oempty = &s.Oempty, *Ofull = &s.Ofull;
    Barrier *Kempty = s.Kempty, *Vempty = s.Vempty, *Kfull = s.Kfull, *Vfull = s.Vfull;

    // init mbarrier
    if (threadIdx.x == 0) {
        init_barrier(Qempty, 256);
        init_barrier(Qfull, 1);
        init_barrier(Oempty, 128);
        init_barrier(Ofull, 256);
        for (int i = 0; i < NUM_SMEM; ++i) {
            init_barrier(&Kfull[i], 1);  // 1 thread arrive
            init_barrier(&Vfull[i], 1);
            init_barrier(&Kempty[i], 256);  // 256 thread arrive
            init_barrier(&Vempty[i], 256);
        }
    }
    __syncthreads();
    fence_view_async_shared();

    // TMA load
    if (wg_idx == 3) {              // producer
        warpgroup_reg_dealloc<24>();
        if (tid == 384) {  // first warp in producer warpgroup
            size_t smem_i = 0, phase_kv = 0;
            for (size_t iw=0; iw<S/NUM_CONSUMER; iw+=BN, smem_i++) {
                if (smem_i >= NUM_SMEM) { smem_i = 0; phase_kv ^= 1; }

                // load K
                wait(&Kempty[smem_i], phase_kv);
                expect_bytes(&Kfull[smem_i], NUM_CONSUMER * BN * DIM * sizeof(fp16));
                #pragma unroll
                for (size_t i=0; i<NUM_CONSUMER; i++) {
                    fp16 *KAddr = sK + (i * NUM_SMEM + smem_i) * BN * DIM;
                    load_async(KAddr, &tensorMapK, &Kfull[smem_i], bs, hn, i * S / NUM_CONSUMER + iw, 0);
                }

                // load V
                wait(&Vempty[smem_i], phase_kv);
                expect_bytes(&Vfull[smem_i], NUM_CONSUMER * BN * DIM * sizeof(fp16));
                #pragma unroll
                for (size_t i=0; i<NUM_CONSUMER; i++) {
                    fp16 *VAddr = sV + (i * NUM_SMEM + smem_i) * BN * DIM;
                    load_async(VAddr, &tensorMapV, &Vfull[smem_i], bs, hn, i * S / NUM_CONSUMER + iw, 0);
                }
            }
        } else if (tid == 416) {  // second warp in producer warpgroup
            size_t q_phase = 0;
            for (size_t iw=0; iw<S/NUM_CONSUMER; iw+=BN) {
                for (size_t i=0; i<N; i++, q_phase ^= 1) {
                    // load Q
                    wait(Qempty, q_phase);
                    expect_bytes(Qfull, BM * DIM * sizeof(fp16));
                    load_async(sQ, &tensorMapQ, Qfull, bs, hn, (by * N + i) * BM, 0);
                }
            }
        }
    } else if (wg_idx == 2) {       // accum
        warpgroup_reg_alloc<160>();

        arrive(Oempty);
        
        const int lane_row = lane_id & 0xf;         // 0..15
        const int lane_col = (lane_id >> 4) * 8;    // 0 or 8
        // define
        fp32 acc_o[N][BM/MMA_M][DIM/PV_MMA_N][PV_MMA_N/16][8];
        fp32 logsum[N][BM/MMA_M][2];
        fp32 scores_max[N][BM/MMA_M][2];
        fp32 scores_max_prev[N][BM/MMA_M][2];

        // init acc_o
        #pragma unroll
        for (size_t n=0; n<N; n++) {
            #pragma unroll
            for (size_t i=0; i<BM/MMA_M; i++) {
                #pragma unroll
                for (size_t j=0; j<DIM/PV_MMA_N; j++) {
                    #pragma unroll
                    for (size_t k=0; k<PV_MMA_N/16; k++) {
                        #pragma unroll
                        for (size_t l=0; l<8; l++) {
                            acc_o[n][i][j][k][l] = 0.0f;
                        }
                    }
                }
            }
        }
        
        // init logsum, score_max
        #pragma unroll
        for (size_t n=0; n<N; n++) {
            #pragma unroll
            for (size_t i=0; i<BM/MMA_M; i++) {
                #pragma unroll
                for (size_t j=0; j<2; j++) {
                    logsum[n][i][j] = 0.0f;
                    scores_max[n][i][j] = -FLT_MAX;
                }
            }
        }

        // for k
        size_t o_phase = 0;
        for (size_t iw=0; iw<S/NUM_CONSUMER; iw+=BN) {

            // for N
            for (size_t n=0; n<N; n++, o_phase ^= 1) {

                wait(Ofull, o_phase);
                // for NUM_CONSUMER
                for (size_t nc=0; nc<NUM_CONSUMER; nc++) {
                    // prev_max = max
                    #pragma unroll
                    for (size_t i=0; i<BM/MMA_M; i++) {
                        #pragma unroll
                        for (size_t j=0; j<2; j++) {
                            scores_max_prev[n][i][j] = scores_max[n][i][j];
                        }
                    }
                    
                    #pragma unroll
                    for (size_t i=0; i<BM/MMA_M; i++) {
                        int offset = nc * BM + i * MMA_M + warp_id_in_wg * 16 + lane_row * 2;
                        float2 tmp_max = *reinterpret_cast<float2*>(&smem_max[offset]);
                        float2 tmp_sum = *reinterpret_cast<float2*>(&smem_sum[offset]);
                        // float2 tmp_max, tmp_sum;

                        // if (lane_id < 16) {
                        //     int offset = nc * BM
                        //             + i * MMA_M
                        //             + warp_id_in_wg * 16
                        //             + lane_row * 2;

                        //     tmp_max = *reinterpret_cast<float2*>(&smem_max[offset]);
                        //     tmp_sum = *reinterpret_cast<float2*>(&smem_sum[offset]);
                        // }

                        // // lane 0..15 的结果广播到 lane 16..31
                        // tmp_max.x = __shfl_sync(0xffffffff, tmp_max.x, lane_row);
                        // tmp_max.y = __shfl_sync(0xffffffff, tmp_max.y, lane_row);
                        // tmp_sum.x = __shfl_sync(0xffffffff, tmp_sum.x, lane_row);
                        // tmp_sum.y = __shfl_sync(0xffffffff, tmp_sum.y, lane_row);
                        // score_max = max(tmp_max, score_max)
                        scores_max[n][i][0] = max(scores_max[n][i][0], tmp_max.x);
                        scores_max[n][i][1] = max(scores_max[n][i][1], tmp_max.y);
                        // all scale
                        const float prev_scale_0 = exp2f(scores_max_prev[n][i][0] - scores_max[n][i][0]);
                        const float prev_scale_1 = exp2f(scores_max_prev[n][i][1] - scores_max[n][i][1]);
                        const float tile_scale_0 = exp2f(tmp_max.x - scores_max[n][i][0]);
                        const float tile_scale_1 = exp2f(tmp_max.y - scores_max[n][i][1]);
                        // logsum = logsum * exp(pm - m) + tmp_sum * exp(tmp_max - m)
                        logsum[n][i][0] = logsum[n][i][0] * prev_scale_0 + tmp_sum.x * tile_scale_0;
                        logsum[n][i][1] = logsum[n][i][1] * prev_scale_1 + tmp_sum.y * tile_scale_1;
                        // acc_o = acc_o * exp(pm - m) + tmp_acc_o * exp(tmp_max - m)
                        #pragma unroll
                        for (size_t j=0; j<DIM/PV_MMA_N; j++) {
                            #pragma unroll
                            for (size_t k=0; k<PV_MMA_N/16; k++) {
                                int o_row = i * MMA_M
                                            + warp_id_in_wg * 16
                                            + lane_row;
                                int o_col = j * PV_MMA_N
                                            + k * 16
                                            + lane_col;
                                fp16 *_sO = sO + nc * BM * DIM
                                            + tma_smem_swizzle_128b_offset_2d<BM>(o_row, o_col);
                                uint32_t r0, r1, r2, r3;
                                ldmatrix_x4_reg(_sO, r0, r1, r2, r3);

                                float2 f01 = __half22float2(u32_to_half2(r0));
                                float2 f23 = __half22float2(u32_to_half2(r1));
                                float2 f45 = __half22float2(u32_to_half2(r2));
                                float2 f67 = __half22float2(u32_to_half2(r3));

                                acc_o[n][i][j][k][0] = fmaf(f01.x, tile_scale_0, acc_o[n][i][j][k][0] * prev_scale_0);
                                acc_o[n][i][j][k][1] = fmaf(f01.y, tile_scale_0, acc_o[n][i][j][k][1] * prev_scale_0);
                                acc_o[n][i][j][k][4] = fmaf(f45.x, tile_scale_0, acc_o[n][i][j][k][4] * prev_scale_0);
                                acc_o[n][i][j][k][5] = fmaf(f45.y, tile_scale_0, acc_o[n][i][j][k][5] * prev_scale_0);

                                acc_o[n][i][j][k][2] = fmaf(f23.x, tile_scale_1, acc_o[n][i][j][k][2] * prev_scale_1);
                                acc_o[n][i][j][k][3] = fmaf(f23.y, tile_scale_1, acc_o[n][i][j][k][3] * prev_scale_1);
                                acc_o[n][i][j][k][6] = fmaf(f67.x, tile_scale_1, acc_o[n][i][j][k][6] * prev_scale_1);
                                acc_o[n][i][j][k][7] = fmaf(f67.y, tile_scale_1, acc_o[n][i][j][k][7] * prev_scale_1);
                            }
                        }
                    }
                }
                arrive(Oempty);
            }
        }
        // compute result and save result 
        for (size_t n=0; n<N; n++) {
            // acc_o = acc_o / logsum
            #pragma unroll
            for (size_t i=0; i<BM/MMA_M; i++) {
                float inv_sum0 = 1.0f / logsum[n][i][0];
                float inv_sum1 = 1.0f / logsum[n][i][1];
                #pragma unroll
                for (size_t j=0; j<DIM/PV_MMA_N; j++) {
                    #pragma unroll
                    for (size_t k=0; k<PV_MMA_N/16; k++) {
                        acc_o[n][i][j][k][0] *= inv_sum0;
                        acc_o[n][i][j][k][1] *= inv_sum0;
                        acc_o[n][i][j][k][4] *= inv_sum0;
                        acc_o[n][i][j][k][5] *= inv_sum0;

                        acc_o[n][i][j][k][2] *= inv_sum1;
                        acc_o[n][i][j][k][3] *= inv_sum1;
                        acc_o[n][i][j][k][6] *= inv_sum1;
                        acc_o[n][i][j][k][7] *= inv_sum1;
                    }
                }
            }
            // copy acc_o to sO (load matritx)
            if (tid == 256) {
                tma_store_wait();
            }
            // bar_sync(128, 2);
            #pragma unroll
            for (size_t i=0; i<BM/MMA_M; i++) {
                #pragma unroll
                for (size_t j=0; j<DIM/PV_MMA_N; j++) {
                    #pragma unroll
                    for (size_t k=0; k<PV_MMA_N/16; k++) {
                        int o_row = i * MMA_M
                                    + warp_id_in_wg * 16
                                    + lane_row;
                        int o_col = j * PV_MMA_N
                                    + k * 16
                                    + lane_col;
                        fp16 *_sO = sO + tma_smem_swizzle_128b_offset_2d<BM>(o_row, o_col);
                        uint32_t r0 = half2_to_u32(__floats2half2_rn(
                            acc_o[n][i][j][k][0],
                            acc_o[n][i][j][k][1]
                        ));
                        uint32_t r1 = half2_to_u32(__floats2half2_rn(
                            acc_o[n][i][j][k][2],
                            acc_o[n][i][j][k][3]
                        ));
                        uint32_t r2 = half2_to_u32(__floats2half2_rn(
                            acc_o[n][i][j][k][4],
                            acc_o[n][i][j][k][5]
                        ));
                        uint32_t r3 = half2_to_u32(__floats2half2_rn(
                            acc_o[n][i][j][k][6],
                            acc_o[n][i][j][k][7]
                        ));

                        stmatrix_x4_reg(_sO, r0, r1, r2, r3);
                    }
                }
            }
            // Wait the accumulator warpgroup before issuing TMA store.
            fence_view_async_shared();
            bar_sync(128, 2);
            // tma store
            if (tid == 256) {
                store_async(&tensorMapO, sO, bs, hn, (by * N + n) * BM, 0);
                tma_store_arrive();
            }
        }
        if (tid == 256) {
            tma_store_wait();
        }
        bar_sync(128, 2);
    } else {                        // consumer
        warpgroup_reg_alloc<144>();
        #pragma unroll
        for (int st = 0; st < NUM_SMEM; ++st) {
            arrive(&Kempty[st]);
            arrive(&Vempty[st]);
        }
        arrive(Qempty);

        // main for loop
        size_t smem_i = 0, phase_kv = 0;
        size_t q_phase = 0;
        for (size_t iw=0; iw<S/NUM_CONSUMER; iw+=BN, ++smem_i) {
            if (smem_i >= NUM_SMEM) { smem_i = 0; phase_kv ^= 1; }
            
            for (size_t n=0; n<N; n++, q_phase ^= 1) {

                // define
                const unsigned mask = __activemask();
                fp32 acc_s[BM/MMA_M][BN/QK_MMA_N][QK_MMA_N/16][8];
                fp16 acc_s_cast[BM/MMA_M][BN/QK_MMA_N][QK_MMA_N/16][8];
                fp32 acc_o[BM/MMA_M][DIM/PV_MMA_N][PV_MMA_N/16][8];
                fp32 scores_max[BM/MMA_M][2];
                fp32 scores_sum[BM/MMA_M][2];

                // fill acc_s
                #pragma unroll
                for (size_t i=0; i<BM/MMA_M; i++) {
                    #pragma unroll
                    for (size_t j=0; j<BN/QK_MMA_N; j++) {
                        #pragma unroll
                        for (size_t k=0; k<QK_MMA_N/16; k++) {
                            #pragma unroll
                            for (size_t l=0; l<8; l++) {
                                acc_s[i][j][k][l] = 0.0f;  // no mask
                            }
                        }
                    }
                }
                
                // wait qk
                wait(Qfull, q_phase);
                wait(&Kfull[smem_i], phase_kv);

                // gemm-qk
                warpgroup_arrive();
                #pragma unroll
                for (size_t i=0; i<BM/MMA_M; i++) {   // block关于wg的布局：[2, 1]
                    #pragma unroll
                    for (size_t j=0; j<BN/QK_MMA_N; j++) {
                        #pragma unroll
                        for (size_t k=0; k<DIM; k+=MMA_K) {
                            fp16 *_QAddr = sQ + tma_smem_offset_2d<BM>(i * MMA_M, k);
                            fp16 *_KAddr = sK + (wg_idx * NUM_SMEM + smem_i) * BN * DIM + tma_smem_offset_2d<BN>(j * QK_MMA_N, k);
                            wgmma_ss<QK_MMA_N, 1, 1, 1, 0, 0, 16, 1024, true>(acc_s[i][j], _QAddr, _KAddr);
                        }
                    }
                }
                warpgroup_commit_batch();
                warpgroup_wait();
                arrive(Qempty);

                // compute m, p, sum, o
                // m = reduce_max(acc_s)
                #pragma unroll
                for (size_t i=0; i<BM/MMA_M; i++) {
                    #pragma unroll
                    for (size_t j=0; j<2; j++) {
                        scores_max[i][j] = -FLT_MAX;
                    }
                }
                #pragma unroll
                for (size_t i=0; i<BM/MMA_M; i++) {
                    #pragma unroll
                    for (size_t j=0; j<BN/QK_MMA_N; j++) {
                        #pragma unroll
                        for (size_t k=0; k<QK_MMA_N/16; k++) {
                            #pragma unroll
                            for (size_t l=0; l<8; l++) {
                                int row = (l >> 1) & 1;
                                scores_max[i][row] = max(scores_max[i][row], acc_s[i][j][k][l]);
                            }
                        }
                    }
                }
                #pragma unroll
                for (size_t i=0; i<BM/MMA_M; i++) {
                    #pragma unroll
                    for (size_t j=0; j<2; j++) {
                        #pragma unroll
                        for (size_t k=1; k<4; k*=2) {
                            scores_max[i][j] = max(scores_max[i][j], __shfl_xor_sync(mask, scores_max[i][j], k, 4));
                        }
                    }
                }
            
                // acc_s = exp(acc_s - m)
                #pragma unroll
                for (size_t i=0; i<BM/MMA_M; i++) {
                    #pragma unroll
                    for (size_t j=0; j<BN/QK_MMA_N; j++) {
                        #pragma unroll
                        for (size_t k=0; k<QK_MMA_N/16; k++) {
                            #pragma unroll
                            for (size_t l=0; l<8; l++) {
                                int row = (l >> 1) & 1;
                                acc_s[i][j][k][l] = exp2f(acc_s[i][j][k][l] * scale - scores_max[i][row] * scale);
                            }
                        }
                    }
                }

                // sum = reduce_sum(acc_s)
                #pragma unroll
                for (size_t i=0; i<BM/MMA_M; i++) {
                    #pragma unroll
                    for (size_t j=0; j<2; j++) {
                        scores_sum[i][j] = 0.0f;
                    }
                }
                #pragma unroll
                for (size_t i=0; i<BM/MMA_M; i++) {
                    #pragma unroll
                    for (size_t j=0; j<BN/QK_MMA_N; j++) {
                        #pragma unroll
                        for (size_t k=0; k<QK_MMA_N/16; k++) {
                            #pragma unroll
                            for (size_t l=0; l<8; l++) {
                                int row = (l >> 1) & 1;
                                scores_sum[i][row] +=  acc_s[i][j][k][l];
                            }
                        }
                    }
                }
                #pragma unroll
                for (size_t i=0; i<BM/MMA_M; i++) {
                    #pragma unroll
                    for (size_t j=0; j<2; j++) {
                        #pragma unroll
                        for (size_t k=1; k<4; k*=2) {
                            scores_sum[i][j] += __shfl_xor_sync(mask, scores_sum[i][j], k, 4);
                        }
                    }
                }
            
                // cast acc_s to acc_s_cast
                #pragma unroll
                for (size_t i=0; i<BM/MMA_M; i++) {
                    #pragma unroll
                    for (size_t j=0; j<BN/QK_MMA_N; j++) {
                        #pragma unroll
                        for (size_t k=0; k<QK_MMA_N/16; k++) {
                            #pragma unroll
                            for (size_t l=0; l<8; l+=2) {
                                uint1 _t2;
                                float2 _t1 = *(float2*)(&acc_s[i][j][k][l]);
                                *(half2*)(&(_t2)) = __float22half2_rn(*(float2*)(&(_t1)));
                                *(uint1*)(&acc_s_cast[i][j][k][l]) = _t2;
                            }
                        }
                    }
                }

                // init acc_o
                #pragma unroll
                for (size_t i=0; i<BM/MMA_M; i++) {
                    #pragma unroll
                    for (size_t j=0; j<DIM/PV_MMA_N; j++) {
                        #pragma unroll
                        for (size_t k=0; k<PV_MMA_N/16; k++) {
                            #pragma unroll
                            for (size_t l=0; l<8; l++) {
                                acc_o[i][j][k][l] = 0.0f;
                            }
                        }
                    }
                }

                // gemm-pv
                wait(&Vfull[smem_i], phase_kv);
                warpgroup_arrive();
                #pragma unroll
                for (size_t i=0; i<BM/MMA_M; i++) {
                    #pragma unroll
                    for (size_t j=0; j<DIM/PV_MMA_N; j++) {
                        #pragma unroll
                        for (size_t k=0; k<BN; k+=MMA_K) {
                            const int p_tile_outer = k / QK_MMA_N;
                            const int p_tile_inner = (k % QK_MMA_N) / 16;
                            uint32_t *_PAddr = reinterpret_cast<uint32_t*>(acc_s_cast[i][p_tile_outer][p_tile_inner]);
                            fp16 *_VAddr = sV + (wg_idx * NUM_SMEM + smem_i) * BN * DIM + tma_smem_offset_2d<BN>(k, j * PV_MMA_N);
                            wgmma_rs<PV_MMA_N, 1, 1, 1, 0, 1, BN * 64 * sizeof(fp16), 1024, true>(acc_o[i][j], _PAddr, _VAddr);
                        }
                    }
                }
                warpgroup_commit_batch();
                warpgroup_wait();

                // write m, acc_o, sum to shared memory
                // store acc_o to smem_o
                wait(Oempty, q_phase);
                const int lane_row = lane_id & 0xf;         // 0..15
                const int lane_col = (lane_id >> 4) * 8;    // 0 or 8
                #pragma unroll
                for (size_t i=0; i<BM/MMA_M; i++) {
                    #pragma unroll
                    for (size_t j=0; j<DIM/PV_MMA_N; j++) {
                        #pragma unroll
                        for (size_t k=0; k<PV_MMA_N/16; k++) {
                            int o_row = i * MMA_M
                                        + warp_id_in_wg * 16
                                        + lane_row;
                            int o_col = j * PV_MMA_N
                                        + k * 16
                                        + lane_col;
                            fp16 *_sO = sO + wg_idx * BM * DIM
                                        + tma_smem_swizzle_128b_offset_2d<BM>(o_row, o_col);
                            uint32_t r0 = half2_to_u32(__floats2half2_rn(
                                acc_o[i][j][k][0],
                                acc_o[i][j][k][1]
                            ));
                            uint32_t r1 = half2_to_u32(__floats2half2_rn(
                                acc_o[i][j][k][2],
                                acc_o[i][j][k][3]
                            ));
                            uint32_t r2 = half2_to_u32(__floats2half2_rn(
                                acc_o[i][j][k][4],
                                acc_o[i][j][k][5]
                            ));
                            uint32_t r3 = half2_to_u32(__floats2half2_rn(
                                acc_o[i][j][k][6],
                                acc_o[i][j][k][7]
                            ));

                            stmatrix_x4_reg(_sO, r0, r1, r2, r3);
                        }
                    }
                }
                // store m, sum to smem
                #pragma unroll
                for (size_t i = 0; i < BM / MMA_M; i++) {
                    float2 sm = make_float2(scores_max[i][0] * scale, scores_max[i][1] * scale);
                    // float2 ss = make_float2(scores_sum[i][0] * scale, scores_sum[i][1] * scale);
                    float2 ss = make_float2(scores_sum[i][0], scores_sum[i][1]);

                    int offset = i * MMA_M 
                                + warp_id_in_wg * 16 
                                + lane_row * 2;
                    *reinterpret_cast<float2*>(&smem_max[wg_idx * BM + offset]) = sm;
                    *reinterpret_cast<float2*>(&smem_sum[wg_idx * BM + offset]) = ss;
                }
                arrive(Ofull);
            }
            arrive(&Kempty[smem_i]);
            arrive(&Vempty[smem_i]);
        }
    }
}


template<int BM, int BN, int DIM, int NUM_THREADS, int NUM_SMEM>
__global__  __launch_bounds__(NUM_THREADS) 
void attnWSKernel(
    int B, int H, int S, 
    const __grid_constant__ CUtensorMap tensorMapQ, 
    const __grid_constant__ CUtensorMap tensorMapK, 
    const __grid_constant__ CUtensorMap tensorMapV, 
    const __grid_constant__ CUtensorMap tensorMapO
) {
    // WS attention
    const int bs = blockIdx.z;
    const int hn = blockIdx.y;
    const int by = blockIdx.x;
    const int tid = threadIdx.x;
    uint32_t wg_idx = tid >> 7;

    // mma size
    static_assert((DIM >= 256 || DIM == 16 || DIM == 32 || DIM == 64 || DIM == 128) && "DIM ERROR!");
    static_assert((BN >= 256 || BN == 16 || BN == 32 || BN == 64 || BN == 128) && "BN ERROR!");
    constexpr int MMA_M = 64;
    constexpr int QK_MMA_N = BN <= 256 ? BN : 256;
    constexpr int PV_MMA_N = DIM <= 256 ? DIM : 256;
    constexpr int MMA_K = 16;

    // setting shared memory
    extern __shared__ __align__(128) uint8_t smem[];
    SMemWS<BM, BN, DIM, NUM_SMEM> &s = *reinterpret_cast<SMemWS<BM, BN, DIM, NUM_SMEM>*>(smem);
    fp16 *sQ = s.Q, *sK = s.K, *sV = s.V, *sO = s.O;
    Barrier *Qmbar = &s.Qmbar;
    Barrier *Kempty = s.Kempty, *Vempty = s.Vempty, *Kfull = s.Kfull, *Vfull = s.Vfull;

    // init mbarrier
    if (threadIdx.x == 0) {
        init_barrier(Qmbar, 1);
        for (int i = 0; i < NUM_SMEM; ++i) {
            init_barrier(&Kfull[i], 1);  // 1 thread arrive
            init_barrier(&Vfull[i], 1);
            init_barrier(&Kempty[i], 256);  // 256 thread arrive
            init_barrier(&Vempty[i], 256);
        }
    }
    __syncthreads();
    fence_view_async_shared();

    // TMA load
    if (wg_idx == 2) {  // producer
        warpgroup_reg_dealloc<24>();
        if (tid == 256) {
            int smem_i = 0, phase = 0;
            // load Q
            expect_bytes(Qmbar, BM * DIM * sizeof(fp16));
            load_async(sQ, &tensorMapQ, Qmbar, bs, hn, by * BM, 0);
            for (size_t iw=0; iw<S; iw+=BN, ++smem_i) {
                if (smem_i >= NUM_SMEM) { smem_i = 0; phase ^= 1; }
                fp16 *KAddr = sK + smem_i * BN * DIM;
                fp16 *VAddr = sV + smem_i * BN * DIM;

                // load K
                wait(&Kempty[smem_i], phase);
                expect_bytes(&Kfull[smem_i], BN * DIM * sizeof(fp16));
                load_async(KAddr, &tensorMapK, &Kfull[smem_i], bs, hn, iw, 0);

                // load V
                wait(&Vempty[smem_i], phase);
                expect_bytes(&Vfull[smem_i], BN * DIM * sizeof(fp16));
                load_async(VAddr, &tensorMapV, &Vfull[smem_i], bs, hn, iw, 0);
            }
        }
    } else {  // consumer
        warpgroup_reg_alloc<240>();
        // Bootstrap empty-smem_i barriers so producer can issue the first K/V loads.
        #pragma unroll
        for (int st = 0; st < NUM_SMEM; ++st) {
            arrive(&Kempty[st]);
            arrive(&Vempty[st]);
        }

        // need args
        uint32_t lane_id = tid & 31;
        uint32_t warp_id_in_wg = (tid >> 5) & 0x3;
        const unsigned mask = __activemask();
        const fp32 scale = sqrt((1.0f / DIM)) * 1.44269504f;  // log2(e)

        // registers define
        fp32 acc_o[BM/(MMA_M*2)][DIM/PV_MMA_N][PV_MMA_N/16][8];
        fp32 scores_max[BM/(MMA_M*2)][2];
        fp32 scores_max_prev[BM/(MMA_M*2)][2];
        fp32 logsum[BM/(MMA_M*2)][2];

        // init acc_o
        #pragma unroll
        for (size_t i=0; i<BM/(MMA_M*2); i++) {
            #pragma unroll
            for (size_t j=0; j<DIM/PV_MMA_N; j++) {
                #pragma unroll
                for (size_t k=0; k<PV_MMA_N/16; k++) {
                    #pragma unroll
                    for (size_t l=0; l<8; l++) {
                        acc_o[i][j][k][l] = 0.0f;
                    }
                }
            }
        }
        
        // init logsum and scores_max
        #pragma unroll
        for (size_t i=0; i<BM/(MMA_M*2); i++) {
            #pragma unroll
            for (size_t j=0; j<2; j++) {
                logsum[i][j] = 0.0f;
                scores_max[i][j] = -FLT_MAX;
            }
        }
        
        wait(Qmbar, 0);
        // main for loop
        int smem_i = 0, phase = 0;
        for (size_t iw=0; iw<S; iw+=BN, ++smem_i) {
            if (smem_i >= NUM_SMEM) { smem_i = 0; phase ^= 1; }
            fp16 *KAddr = sK + smem_i * BN * DIM;
            fp16 *VAddr = sV + smem_i * BN * DIM;

            // fill acc_s
            fp32 acc_s[BM/(MMA_M*2)][BN/QK_MMA_N][QK_MMA_N/16][8];
            #pragma unroll
            for (size_t i=0; i<BM/(MMA_M*2); i++) {
                #pragma unroll
                for (size_t j=0; j<BN/QK_MMA_N; j++) {
                    #pragma unroll
                    for (size_t k=0; k<QK_MMA_N/16; k++) {
                        #pragma unroll
                        for (size_t l=0; l<8; l++) {
                            acc_s[i][j][k][l] = 0.0f;
                        }
                    }
                }
            }

            // gemm-qk
            wait(&Kfull[smem_i], phase);
            warpgroup_arrive();
            #pragma unroll
            for (size_t i=0; i<BM/(MMA_M*2); i++) {   // block关于wg的布局：[2, 1]
                #pragma unroll
                for (size_t j=0; j<BN/QK_MMA_N; j++) {
                    #pragma unroll
                    for (size_t k=0; k<DIM; k+=MMA_K) {
                        int q_row = wg_idx * BM/(MMA_M*2) * MMA_M + i * MMA_M;
                        int q_col = k;
                        int k_row = j * QK_MMA_N;
                        int k_col = k;
                        fp16 *_QAddr = sQ + tma_smem_offset_2d<BM>(q_row, q_col);
                        fp16 *_KAddr = KAddr + tma_smem_offset_2d<BN>(k_row, k_col);
                        wgmma_ss<QK_MMA_N, 1, 1, 1, 0, 0, 16, 1024, true>(acc_s[i][j], _QAddr, _KAddr);
                    }
                }
            }
            warpgroup_commit_batch();
            warpgroup_wait();
            arrive(&Kempty[smem_i]);

            // max_prev = max
            #pragma unroll
            for (size_t i=0; i<BM/(MMA_M*2); i++) {
                #pragma unroll
                for (size_t j=0; j<2; j++) {
                    scores_max_prev[i][j] = scores_max[i][j];
                }
            }

            // reduce max
            #pragma unroll
            for (size_t i=0; i<BM/(MMA_M*2); i++) {
                #pragma unroll
                for (size_t j=0; j<2; j++) {
                    scores_max[i][j] = -FLT_MAX;
                }
            }
            #pragma unroll
            for (size_t i=0; i<BM/(MMA_M*2); i++) {
                #pragma unroll
                for (size_t j=0; j<BN/QK_MMA_N; j++) {
                    #pragma unroll
                    for (size_t k=0; k<QK_MMA_N/16; k++) {
                        scores_max[i][0] = max(acc_s[i][j][k][0], scores_max[i][0]);
                        scores_max[i][0] = max(acc_s[i][j][k][1], scores_max[i][0]);
                        scores_max[i][0] = max(acc_s[i][j][k][4], scores_max[i][0]);
                        scores_max[i][0] = max(acc_s[i][j][k][5], scores_max[i][0]);
                        scores_max[i][1] = max(acc_s[i][j][k][2], scores_max[i][1]);
                        scores_max[i][1] = max(acc_s[i][j][k][3], scores_max[i][1]);
                        scores_max[i][1] = max(acc_s[i][j][k][6], scores_max[i][1]);
                        scores_max[i][1] = max(acc_s[i][j][k][7], scores_max[i][1]);
                    }
                }
            }
            #pragma unroll
            for (size_t i=0; i<BM/(MMA_M*2); i++) {
                #pragma unroll
                for (size_t j=0; j<2; j++) {
                    #pragma unroll
                    for (size_t k=1; k<4; k*=2) {
                        scores_max[i][j] = max(scores_max[i][j], __shfl_xor_sync(mask, scores_max[i][j], k, 4));
                    }
                }
            }

            // m = max(pm, m)
            #pragma unroll
            for (size_t i=0; i<BM/(MMA_M*2); i++) {
                #pragma unroll
                for (size_t j=0; j<2; j++) {
                    scores_max[i][j] = max(scores_max_prev[i][j], scores_max[i][j]);
                }
            }

            // scores_scale = exp2(pm  - m)
            fp32 scores_scale[BM/(MMA_M*2)][2];
            #pragma unroll
            for (size_t i=0; i<BM/(MMA_M*2); i++) {
                #pragma unroll
                for (size_t j=0; j<2; j++) {
                    scores_scale[i][j] = exp2f(scores_max_prev[i][j] * scale - scores_max[i][j] * scale);
                }
            }

            // acc_s = exp2(acc_s - m)
            #pragma unroll
            for (size_t i=0; i<BM/(MMA_M*2); i++) {
                #pragma unroll
                for (size_t j=0; j<BN/QK_MMA_N; j++) {
                    #pragma unroll
                    for (size_t k=0; k<QK_MMA_N/16; k++) {
                        acc_s[i][j][k][0] = exp2f(acc_s[i][j][k][0] * scale - scores_max[i][0] * scale);
                        acc_s[i][j][k][1] = exp2f(acc_s[i][j][k][1] * scale - scores_max[i][0] * scale);
                        acc_s[i][j][k][4] = exp2f(acc_s[i][j][k][4] * scale - scores_max[i][0] * scale);
                        acc_s[i][j][k][5] = exp2f(acc_s[i][j][k][5] * scale - scores_max[i][0] * scale);
                        acc_s[i][j][k][2] = exp2f(acc_s[i][j][k][2] * scale - scores_max[i][1] * scale);
                        acc_s[i][j][k][3] = exp2f(acc_s[i][j][k][3] * scale - scores_max[i][1] * scale);
                        acc_s[i][j][k][6] = exp2f(acc_s[i][j][k][6] * scale - scores_max[i][1] * scale);
                        acc_s[i][j][k][7] = exp2f(acc_s[i][j][k][7] * scale - scores_max[i][1] * scale);
                    }
                }
            }

            // reduce sum
            fp32 scores_sum[BM/(MMA_M*2)][2];
            #pragma unroll
            for (size_t i=0; i<BM/(MMA_M*2); i++) {
                #pragma unroll
                for (size_t j=0; j<2; j++) {
                    scores_sum[i][j] = 0.0f;
                }
            }
            #pragma unroll
            for (size_t i=0; i<BM/(MMA_M*2); i++) {
                #pragma unroll
                for (size_t j=0; j<BN/QK_MMA_N; j++) {
                    #pragma unroll
                    for (size_t k=0; k<QK_MMA_N/16; k++) {
                        scores_sum[i][0] += (acc_s[i][j][k][0] + acc_s[i][j][k][1] + acc_s[i][j][k][4] + acc_s[i][j][k][5]);
                        scores_sum[i][1] += (acc_s[i][j][k][2] + acc_s[i][j][k][3] + acc_s[i][j][k][6] + acc_s[i][j][k][7]);
                    }
                }
            }
            #pragma unroll
            for (size_t i=0; i<BM/(MMA_M*2); i++) {
                #pragma unroll
                for (size_t j=0; j<2; j++) {
                    #pragma unroll
                    for (size_t k=1; k<4; k*=2) {
                        scores_sum[i][j] += __shfl_xor_sync(mask, scores_sum[i][j], k, 4);
                    }
                }
            }

            // logsum = logsum * scores_scale + sum;
            #pragma unroll
            for (size_t i=0; i<BM/(MMA_M*2); i++) {
                #pragma unroll
                for (size_t j=0; j<2; j++) {
                    logsum[i][j] = logsum[i][j] * scores_scale[i][j] + scores_sum[i][j];
                }
            }

            // cast acc_s
            fp16 acc_s_cast[BM/(MMA_M*2)][BN/QK_MMA_N][QK_MMA_N/16][8];
            #pragma unroll
            for (size_t i=0; i<BM/(MMA_M*2); i++) {
                #pragma unroll
                for (size_t j=0; j<BN/QK_MMA_N; j++) {
                    #pragma unroll
                    for (size_t k=0; k<QK_MMA_N/16; k++) {
                        #pragma unroll
                        for (size_t l=0; l<8; l+=2) {
                            uint1 _t2;
                            float2 _t1 = *(float2*)(&acc_s[i][j][k][l]);
                            *(half2*)(&(_t2)) = __float22half2_rn(*(float2*)(&(_t1)));
                            *(uint1*)(&acc_s_cast[i][j][k][l]) = _t2;
                        }
                    }
                }
            }

            // acc_o = acc_o * scores_scale
            #pragma unroll
            for (size_t i=0; i<BM/(MMA_M*2); i++) {
                #pragma unroll
                for (size_t j=0; j<DIM/PV_MMA_N; j++) {
                    #pragma unroll
                    for (size_t k=0; k<PV_MMA_N/16; k++) {
                        acc_o[i][j][k][0] *= scores_scale[i][0];
                        acc_o[i][j][k][1] *= scores_scale[i][0];
                        acc_o[i][j][k][4] *= scores_scale[i][0];
                        acc_o[i][j][k][5] *= scores_scale[i][0];
                        acc_o[i][j][k][2] *= scores_scale[i][1];
                        acc_o[i][j][k][3] *= scores_scale[i][1];
                        acc_o[i][j][k][6] *= scores_scale[i][1];
                        acc_o[i][j][k][7] *= scores_scale[i][1];
                    }
                }
            }

            // gemm-pv
            wait(&Vfull[smem_i], phase);
            warpgroup_arrive();
            #pragma unroll
            for (size_t i=0; i<BM/(MMA_M*2); i++) {
                #pragma unroll
                for (size_t j=0; j<DIM/PV_MMA_N; j++) {
                    #pragma unroll
                    for (size_t k=0; k<BN; k+=MMA_K) {
                        // V is stored in shared as [K=BN, N=DIM].
                        // Use TransB=1 so WGMMA consumes it logically as [N, K].
                        int v_row = k;
                        int v_col = j * PV_MMA_N;
                        const int p_tile_outer = k / QK_MMA_N;
                        const int p_tile_inner = (k % QK_MMA_N) / 16;
                        fp16 *_VAddr = VAddr + tma_smem_offset_2d<BN>(v_row, v_col);
                        wgmma_rs<PV_MMA_N, 1, 1, 1, 0, 1, BN * 64 * sizeof(fp16), 1024, true>(
                            acc_o[i][j],
                            reinterpret_cast<uint32_t*>(acc_s_cast[i][p_tile_outer][p_tile_inner]),
                            _VAddr);
                    }
                }
            }
            warpgroup_commit_batch();
            warpgroup_wait();
            arrive(&Vempty[smem_i]);
        }

        // acc_o = acc_o / logsum
        #pragma unroll
        for (size_t i=0; i<BM/(MMA_M*2); i++) {
            float val0 = 1.0f / logsum[i][0];
            float val1 = 1.0f / logsum[i][1];
            #pragma unroll
            for (size_t j=0; j<DIM/PV_MMA_N; j++) {
                #pragma unroll
                for (size_t k=0; k<PV_MMA_N/16; k++) {
                    acc_o[i][j][k][0] *= val0;
                    acc_o[i][j][k][1] *= val0;
                    acc_o[i][j][k][4] *= val0;
                    acc_o[i][j][k][5] *= val0;
                    acc_o[i][j][k][2] *= val1;
                    acc_o[i][j][k][3] *= val1;
                    acc_o[i][j][k][6] *= val1;
                    acc_o[i][j][k][7] *= val1;
                }
            }
        }

        // copy acc_o to sO (load matritx)
        if (tid == 0) {
            tma_store_wait();
        }

        const int lane_row = lane_id & 0xf;
        const int lane_col = (lane_id >> 4) * 8;
        #pragma unroll
        for (size_t i=0; i<BM/(MMA_M*2); i++) {
            #pragma unroll
            for (size_t j=0; j<DIM/PV_MMA_N; j++) {
                #pragma unroll
                for (size_t k=0; k<PV_MMA_N/16; k++) {
                    int o_row = wg_idx * BM/(MMA_M*2) * MMA_M
                                + i * MMA_M
                                + warp_id_in_wg * 16
                                + lane_row;
                    int o_col = j * PV_MMA_N
                                + k * 16
                                + lane_col;
                    // fp16 *_sO = sO + tma_smem_offset_2d<BM>(o_row, o_col);
                    fp16 *_sO = sO + tma_smem_swizzle_128b_offset_2d<BM>(o_row, o_col);
                        uint32_t r0 = half2_to_u32(__floats2half2_rn(
                            acc_o[i][j][k][0],
                            acc_o[i][j][k][1]
                        ));
                        uint32_t r1 = half2_to_u32(__floats2half2_rn(
                            acc_o[i][j][k][2],
                            acc_o[i][j][k][3]
                        ));
                        uint32_t r2 = half2_to_u32(__floats2half2_rn(
                            acc_o[i][j][k][4],
                            acc_o[i][j][k][5]
                        ));
                        uint32_t r3 = half2_to_u32(__floats2half2_rn(
                            acc_o[i][j][k][6],
                            acc_o[i][j][k][7]
                        ));

                        stmatrix_x4_reg(_sO, r0, r1, r2, r3);
                }
            }
        }
        // Wait all consumer threads (2 warpgroups) before issuing TMA store from tid==0.
        fence_view_async_shared();
        bar_sync(256, 2);
        // tma store
        if (tid == 0) {
            store_async(&tensorMapO, sO, bs, hn, by * BM, 0);
            tma_store_arrive();
        }
    }
}


template<int BM, int BN, int DIM, int NUM_THREADS, int NUM_SMEM>
__global__  __launch_bounds__(NUM_THREADS) 
void attnWS2StageKernel(
    int B, int H, int S, 
    const __grid_constant__ CUtensorMap tensorMapQ, 
    const __grid_constant__ CUtensorMap tensorMapK, 
    const __grid_constant__ CUtensorMap tensorMapV, 
    const __grid_constant__ CUtensorMap tensorMapO
) {
    // WS attention
    const int bs = blockIdx.z;
    const int hn = blockIdx.y;
    const int by = blockIdx.x;
    const int tid = threadIdx.x;
    uint32_t wg_idx = tid >> 7;

    // mma size
    static_assert((DIM >= 256 || DIM == 16 || DIM == 32 || DIM == 64 || DIM == 128) && "DIM ERROR!");
    static_assert((BN >= 256 || BN == 16 || BN == 32 || BN == 64 || BN == 128) && "BN ERROR!");
    constexpr int MMA_M = 64;
    constexpr int QK_MMA_N = BN <= 256 ? BN : 256;
    constexpr int PV_MMA_N = DIM <= 256 ? DIM : 256;
    constexpr int MMA_K = 16;

    // setting shared memory
    extern __shared__ __align__(128) uint8_t smem[];
    SMemWS<BM, BN, DIM, NUM_SMEM> &s = *reinterpret_cast<SMemWS<BM, BN, DIM, NUM_SMEM>*>(smem);
    fp16 *sQ = s.Q, *sK = s.K, *sV = s.V, *sO = s.O;
    Barrier *Qmbar = &s.Qmbar;
    Barrier *Kempty = s.Kempty, *Vempty = s.Vempty, *Kfull = s.Kfull, *Vfull = s.Vfull;

    // init mbarrier
    if (threadIdx.x == 0) {
        init_barrier(Qmbar, 1);
        for (int i = 0; i < NUM_SMEM; ++i) {
            init_barrier(&Kfull[i], 1);  // 1 thread arrive
            init_barrier(&Vfull[i], 1);
            init_barrier(&Kempty[i], 256);  // 256 thread arrive
            init_barrier(&Vempty[i], 256);
        }
    }
    __syncthreads();
    fence_view_async_shared();

    // TMA load
    if (wg_idx == 2) {  // producer
        warpgroup_reg_dealloc<24>();
        if (tid == 256) {
            int smem_i = 0, phase = 0;
            // load Q
            expect_bytes(Qmbar, BM * DIM * sizeof(fp16));
            load_async(sQ, &tensorMapQ, Qmbar, bs, hn, by * BM, 0);
            for (size_t iw=0; iw<S; iw+=BN, ++smem_i) {
                if (smem_i >= NUM_SMEM) { smem_i = 0; phase ^= 1; }
                fp16 *KAddr = sK + smem_i * BN * DIM;
                fp16 *VAddr = sV + smem_i * BN * DIM;

                // load K
                wait(&Kempty[smem_i], phase);
                expect_bytes(&Kfull[smem_i], BN * DIM * sizeof(fp16));
                load_async(KAddr, &tensorMapK, &Kfull[smem_i], bs, hn, iw, 0);

                // load V
                wait(&Vempty[smem_i], phase);
                expect_bytes(&Vfull[smem_i], BN * DIM * sizeof(fp16));
                load_async(VAddr, &tensorMapV, &Vfull[smem_i], bs, hn, iw, 0);
            }
        }
    } else {  // consumer
        warpgroup_reg_alloc<240>();
        // Bootstrap empty-smem_i barriers so producer can issue the first K/V loads.
        #pragma unroll
        for (int st = 0; st < NUM_SMEM; ++st) {
            arrive(&Kempty[st]);
            arrive(&Vempty[st]);
        }

        // need args
        uint32_t lane_id = tid & 31;
        uint32_t warp_id_in_wg = (tid >> 5) & 0x3;
        const unsigned mask = __activemask();
        const fp32 scale = sqrt((1.0f / DIM)) * 1.44269504f;  // log2(e)

        // registers define
        fp32 acc_o[BM/(MMA_M*2)][DIM/PV_MMA_N][PV_MMA_N/16][8];
        fp32 scores_max[BM/(MMA_M*2)][2];
        fp32 scores_max_prev[BM/(MMA_M*2)][2];
        fp32 logsum[BM/(MMA_M*2)][2];

        fp32 acc_s[BM/(MMA_M*2)][BN/QK_MMA_N][QK_MMA_N/16][8];
        fp16 acc_s_cast[BM/(MMA_M*2)][BN/QK_MMA_N][QK_MMA_N/16][8];

        // define softmax compute function
        auto __softmax = [&]() {
            // max_prev = max
            #pragma unroll
            for (size_t i=0; i<BM/(MMA_M*2); i++) {
                #pragma unroll
                for (size_t j=0; j<2; j++) {
                    scores_max_prev[i][j] = scores_max[i][j];
                }
            }

            // reduce max
            #pragma unroll
            for (size_t i=0; i<BM/(MMA_M*2); i++) {
                #pragma unroll
                for (size_t j=0; j<2; j++) {
                    scores_max[i][j] = -FLT_MAX;
                }
            }
            #pragma unroll
            for (size_t i=0; i<BM/(MMA_M*2); i++) {
                #pragma unroll
                for (size_t j=0; j<BN/QK_MMA_N; j++) {
                    #pragma unroll
                    for (size_t k=0; k<QK_MMA_N/16; k++) {
                        scores_max[i][0] = max(acc_s[i][j][k][0], scores_max[i][0]);
                        scores_max[i][0] = max(acc_s[i][j][k][1], scores_max[i][0]);
                        scores_max[i][0] = max(acc_s[i][j][k][4], scores_max[i][0]);
                        scores_max[i][0] = max(acc_s[i][j][k][5], scores_max[i][0]);
                        scores_max[i][1] = max(acc_s[i][j][k][2], scores_max[i][1]);
                        scores_max[i][1] = max(acc_s[i][j][k][3], scores_max[i][1]);
                        scores_max[i][1] = max(acc_s[i][j][k][6], scores_max[i][1]);
                        scores_max[i][1] = max(acc_s[i][j][k][7], scores_max[i][1]);
                    }
                }
            }
            #pragma unroll
            for (size_t i=0; i<BM/(MMA_M*2); i++) {
                #pragma unroll
                for (size_t j=0; j<2; j++) {
                    #pragma unroll
                    for (size_t k=1; k<4; k*=2) {
                        scores_max[i][j] = max(scores_max[i][j], __shfl_xor_sync(mask, scores_max[i][j], k, 4));
                    }
                }
            }

            // m = max(pm, m)
            #pragma unroll
            for (size_t i=0; i<BM/(MMA_M*2); i++) {
                #pragma unroll
                for (size_t j=0; j<2; j++) {
                    scores_max[i][j] = max(scores_max_prev[i][j], scores_max[i][j]);
                }
            }

            // scores_scale = exp2(pm  - m)
            fp32 scores_scale[BM/(MMA_M*2)][2];
            #pragma unroll
            for (size_t i=0; i<BM/(MMA_M*2); i++) {
                #pragma unroll
                for (size_t j=0; j<2; j++) {
                    scores_scale[i][j] = exp2f(scores_max_prev[i][j] * scale - scores_max[i][j] * scale);
                }
            }

            // acc_s = exp2(acc_s - m)
            #pragma unroll
            for (size_t i=0; i<BM/(MMA_M*2); i++) {
                #pragma unroll
                for (size_t j=0; j<BN/QK_MMA_N; j++) {
                    #pragma unroll
                    for (size_t k=0; k<QK_MMA_N/16; k++) {
                        acc_s[i][j][k][0] = exp2f(acc_s[i][j][k][0] * scale - scores_max[i][0] * scale);
                        acc_s[i][j][k][1] = exp2f(acc_s[i][j][k][1] * scale - scores_max[i][0] * scale);
                        acc_s[i][j][k][4] = exp2f(acc_s[i][j][k][4] * scale - scores_max[i][0] * scale);
                        acc_s[i][j][k][5] = exp2f(acc_s[i][j][k][5] * scale - scores_max[i][0] * scale);
                        acc_s[i][j][k][2] = exp2f(acc_s[i][j][k][2] * scale - scores_max[i][1] * scale);
                        acc_s[i][j][k][3] = exp2f(acc_s[i][j][k][3] * scale - scores_max[i][1] * scale);
                        acc_s[i][j][k][6] = exp2f(acc_s[i][j][k][6] * scale - scores_max[i][1] * scale);
                        acc_s[i][j][k][7] = exp2f(acc_s[i][j][k][7] * scale - scores_max[i][1] * scale);
                    }
                }
            }

            // reduce sum
            fp32 scores_sum[BM/(MMA_M*2)][2];
            #pragma unroll
            for (size_t i=0; i<BM/(MMA_M*2); i++) {
                #pragma unroll
                for (size_t j=0; j<2; j++) {
                    scores_sum[i][j] = 0.0f;
                }
            }
            #pragma unroll
            for (size_t i=0; i<BM/(MMA_M*2); i++) {
                #pragma unroll
                for (size_t j=0; j<BN/QK_MMA_N; j++) {
                    #pragma unroll
                    for (size_t k=0; k<QK_MMA_N/16; k++) {
                        scores_sum[i][0] += (acc_s[i][j][k][0] + acc_s[i][j][k][1] + acc_s[i][j][k][4] + acc_s[i][j][k][5]);
                        scores_sum[i][1] += (acc_s[i][j][k][2] + acc_s[i][j][k][3] + acc_s[i][j][k][6] + acc_s[i][j][k][7]);
                    }
                }
            }
            #pragma unroll
            for (size_t i=0; i<BM/(MMA_M*2); i++) {
                #pragma unroll
                for (size_t j=0; j<2; j++) {
                    #pragma unroll
                    for (size_t k=1; k<4; k*=2) {
                        scores_sum[i][j] += __shfl_xor_sync(mask, scores_sum[i][j], k, 4);
                    }
                }
            }

            // logsum = logsum * scores_scale + sum;
            #pragma unroll
            for (size_t i=0; i<BM/(MMA_M*2); i++) {
                #pragma unroll
                for (size_t j=0; j<2; j++) {
                    logsum[i][j] = logsum[i][j] * scores_scale[i][j] + scores_sum[i][j];
                }
            }

            // cast acc_s
            #pragma unroll
            for (size_t i=0; i<BM/(MMA_M*2); i++) {
                #pragma unroll
                for (size_t j=0; j<BN/QK_MMA_N; j++) {
                    #pragma unroll
                    for (size_t k=0; k<QK_MMA_N/16; k++) {
                        #pragma unroll
                        for (size_t l=0; l<8; l+=2) {
                            uint1 _t2;
                            float2 _t1 = *(float2*)(&acc_s[i][j][k][l]);
                            *(half2*)(&(_t2)) = __float22half2_rn(*(float2*)(&(_t1)));
                            *(uint1*)(&acc_s_cast[i][j][k][l]) = _t2;
                        }
                    }
                }
            }
        
            // acc_o = acc_o * scores_scale
            #pragma unroll
            for (size_t i=0; i<BM/(MMA_M*2); i++) {
                #pragma unroll
                for (size_t j=0; j<DIM/PV_MMA_N; j++) {
                    #pragma unroll
                    for (size_t k=0; k<PV_MMA_N/16; k++) {
                        acc_o[i][j][k][0] *= scores_scale[i][0];
                        acc_o[i][j][k][1] *= scores_scale[i][0];
                        acc_o[i][j][k][4] *= scores_scale[i][0];
                        acc_o[i][j][k][5] *= scores_scale[i][0];
                        acc_o[i][j][k][2] *= scores_scale[i][1];
                        acc_o[i][j][k][3] *= scores_scale[i][1];
                        acc_o[i][j][k][6] *= scores_scale[i][1];
                        acc_o[i][j][k][7] *= scores_scale[i][1];
                    }
                }
            }

        };

        // init acc_o
        #pragma unroll
        for (size_t i=0; i<BM/(MMA_M*2); i++) {
            #pragma unroll
            for (size_t j=0; j<DIM/PV_MMA_N; j++) {
                #pragma unroll
                for (size_t k=0; k<PV_MMA_N/16; k++) {
                    #pragma unroll
                    for (size_t l=0; l<8; l++) {
                        acc_o[i][j][k][l] = 0.0f;
                    }
                }
            }
        }
        
        // init logsum and scores_max
        #pragma unroll
        for (size_t i=0; i<BM/(MMA_M*2); i++) {
            #pragma unroll
            for (size_t j=0; j<2; j++) {
                logsum[i][j] = 0.0f;
                scores_max[i][j] = -FLT_MAX;
            }
        }
        
        wait(Qmbar, 0);
        int smem_i = 0, phase = 0;

        // ===== prologue =====
        // fill acc_s
        #pragma unroll
        for (size_t i=0; i<BM/(MMA_M*2); i++) {
            #pragma unroll
            for (size_t j=0; j<BN/QK_MMA_N; j++) {
                #pragma unroll
                for (size_t k=0; k<QK_MMA_N/16; k++) {
                    #pragma unroll
                    for (size_t l=0; l<8; l++) {
                        acc_s[i][j][k][l] = 0.0f;
                    }
                }
            }
        }

        // gemm-qk
        wait(&Kfull[smem_i], phase);
        warpgroup_arrive();
        #pragma unroll
        for (size_t i=0; i<BM/(MMA_M*2); i++) {   // block关于wg的布局：[2, 1]
            #pragma unroll
            for (size_t j=0; j<BN/QK_MMA_N; j++) {
                #pragma unroll
                for (size_t k=0; k<DIM; k+=MMA_K) {
                    int q_row = wg_idx * BM/(MMA_M*2) * MMA_M + i * MMA_M;
                    int q_col = k;
                    int k_row = j * QK_MMA_N;
                    int k_col = k;
                    fp16 *_QAddr = sQ + tma_smem_offset_2d<BM>(q_row, q_col);
                    fp16 *_KAddr = sK + tma_smem_offset_2d<BN>(k_row, k_col);
                    wgmma_ss<QK_MMA_N, 1, 1, 1, 0, 0, 16, 1024, true>(acc_s[i][j], _QAddr, _KAddr);
                }
            }
        }
        warpgroup_commit_batch();
        warpgroup_wait();
        arrive(&Kempty[smem_i]);

        // softmax
        __softmax();

        // ==== main loop ====
        smem_i++;
        for (size_t iw=BN; iw<S; iw+=BN, ++smem_i) {
            if (smem_i >= NUM_SMEM) { smem_i = 0; phase ^= 1; }
            fp16 *KAddr = sK + smem_i * BN * DIM;

            // fill acc_s
            #pragma unroll
            for (size_t i=0; i<BM/(MMA_M*2); i++) {
                #pragma unroll
                for (size_t j=0; j<BN/QK_MMA_N; j++) {
                    #pragma unroll
                    for (size_t k=0; k<QK_MMA_N/16; k++) {
                        #pragma unroll
                        for (size_t l=0; l<8; l++) {
                            acc_s[i][j][k][l] = 0.0f;
                        }
                    }
                }
            }

            // gemm-qk
            wait(&Kfull[smem_i], phase);
            warpgroup_arrive();
            #pragma unroll
            for (size_t i=0; i<BM/(MMA_M*2); i++) {
                #pragma unroll
                for (size_t j=0; j<BN/QK_MMA_N; j++) {
                    #pragma unroll
                    for (size_t k=0; k<DIM; k+=MMA_K) {
                        int q_row = wg_idx * BM/(MMA_M*2) * MMA_M + i * MMA_M;
                        int q_col = k;
                        int k_row = j * QK_MMA_N;
                        int k_col = k;
                        fp16 *_QAddr = sQ + tma_smem_offset_2d<BM>(q_row, q_col);
                        fp16 *_KAddr = KAddr + tma_smem_offset_2d<BN>(k_row, k_col);
                        wgmma_ss<QK_MMA_N, 1, 1, 1, 0, 0, 16, 1024, true>(acc_s[i][j], _QAddr, _KAddr);
                    }
                }
            }
            warpgroup_commit_batch();
            warpgroup_wait();
            arrive(&Kempty[smem_i]);

            // prev index and phase
            int prev_smem_i = (smem_i + NUM_SMEM -1) % NUM_SMEM;
            int prve_phase = phase;
            if (prev_smem_i == NUM_SMEM -1) { prve_phase ^= 1; }
            fp16 *VAddr = sV + prev_smem_i * BN * DIM;

            // prev gemm-pv
            wait(&Vfull[prev_smem_i], prve_phase);
            warpgroup_arrive();
            #pragma unroll
            for (size_t i=0; i<BM/(MMA_M*2); i++) {
                #pragma unroll
                for (size_t j=0; j<DIM/PV_MMA_N; j++) {
                    #pragma unroll
                    for (size_t k=0; k<BN; k+=MMA_K) {
                        // V is stored in shared as [K=BN, N=DIM].
                        // Use TransB=1 so WGMMA consumes it logically as [N, K].
                        int v_row = k;
                        int v_col = j * PV_MMA_N;
                        const int p_tile_outer = k / QK_MMA_N;
                        const int p_tile_inner = (k % QK_MMA_N) / 16;
                        fp16 *_VAddr = VAddr + tma_smem_offset_2d<BN>(v_row, v_col);
                        wgmma_rs<PV_MMA_N, 1, 1, 1, 0, 1, BN * 64 * sizeof(fp16), 1024, true>(
                            acc_o[i][j],
                            reinterpret_cast<uint32_t*>(acc_s_cast[i][p_tile_outer][p_tile_inner]),
                            _VAddr);
                    }
                }
            }
            warpgroup_commit_batch();
            warpgroup_wait();
            arrive(&Vempty[prev_smem_i]);

            // softmax
            __softmax();
        }

        // ==== epilogue ====
        // last index and phase
        int last_smem_i = (smem_i % NUM_SMEM + NUM_SMEM - 1) % NUM_SMEM;
        int last_phase = phase;
        fp16 *VAddr = sV + last_smem_i * BN * DIM;

        // last gemm-pv
        wait(&Vfull[last_smem_i], last_phase);
        warpgroup_arrive();
        #pragma unroll
        for (size_t i=0; i<BM/(MMA_M*2); i++) {
            #pragma unroll
            for (size_t j=0; j<DIM/PV_MMA_N; j++) {
                #pragma unroll
                for (size_t k=0; k<BN; k+=MMA_K) {
                    // V is stored in shared as [K=BN, N=DIM].
                    // Use TransB=1 so WGMMA consumes it logically as [N, K].
                    int v_row = k;
                    int v_col = j * PV_MMA_N;
                    const int p_tile_outer = k / QK_MMA_N;
                    const int p_tile_inner = (k % QK_MMA_N) / 16;
                    fp16 *_VAddr = VAddr + tma_smem_offset_2d<BN>(v_row, v_col);
                    wgmma_rs<PV_MMA_N, 1, 1, 1, 0, 1, BN * 64 * sizeof(fp16), 1024, true>(
                        acc_o[i][j],
                        reinterpret_cast<uint32_t*>(acc_s_cast[i][p_tile_outer][p_tile_inner]),
                        _VAddr);
                }
            }
        }
        warpgroup_commit_batch();
        warpgroup_wait();
        arrive(&Vempty[last_smem_i]);

        // acc_o = acc_o / logsum
        #pragma unroll
        for (size_t i=0; i<BM/(MMA_M*2); i++) {
            float val0 = 1.0f / logsum[i][0];
            float val1 = 1.0f / logsum[i][1];
            #pragma unroll
            for (size_t j=0; j<DIM/PV_MMA_N; j++) {
                #pragma unroll
                for (size_t k=0; k<PV_MMA_N/16; k++) {
                    acc_o[i][j][k][0] *= val0;
                    acc_o[i][j][k][1] *= val0;
                    acc_o[i][j][k][4] *= val0;
                    acc_o[i][j][k][5] *= val0;
                    acc_o[i][j][k][2] *= val1;
                    acc_o[i][j][k][3] *= val1;
                    acc_o[i][j][k][6] *= val1;
                    acc_o[i][j][k][7] *= val1;
                }
            }
        }

        // copy acc_o to sO (load matritx)
        if (tid == 0) {
            tma_store_wait();
        }

        const int lane_row = lane_id & 0xf;
        const int lane_col = (lane_id >> 4) * 8;
        #pragma unroll
        for (size_t i=0; i<BM/(MMA_M*2); i++) {
            #pragma unroll
            for (size_t j=0; j<DIM/PV_MMA_N; j++) {
                #pragma unroll
                for (size_t k=0; k<PV_MMA_N/16; k++) {
                    int o_row = wg_idx * BM/(MMA_M*2) * MMA_M
                                + i * MMA_M
                                + warp_id_in_wg * 16
                                + lane_row;
                    int o_col = j * PV_MMA_N
                                + k * 16
                                + lane_col;
                    // fp16 *_sO = sO + tma_smem_offset_2d<BM>(o_row, o_col);
                    fp16 *_sO = sO + tma_smem_swizzle_128b_offset_2d<BM>(o_row, o_col);
                        uint32_t r0 = half2_to_u32(__floats2half2_rn(
                            acc_o[i][j][k][0],
                            acc_o[i][j][k][1]
                        ));
                        uint32_t r1 = half2_to_u32(__floats2half2_rn(
                            acc_o[i][j][k][2],
                            acc_o[i][j][k][3]
                        ));
                        uint32_t r2 = half2_to_u32(__floats2half2_rn(
                            acc_o[i][j][k][4],
                            acc_o[i][j][k][5]
                        ));
                        uint32_t r3 = half2_to_u32(__floats2half2_rn(
                            acc_o[i][j][k][6],
                            acc_o[i][j][k][7]
                        ));

                        stmatrix_x4_reg(_sO, r0, r1, r2, r3);
                }
            }
        }
        // Wait all consumer threads (2 warpgroups) before issuing TMA store from tid==0.
        fence_view_async_shared();
        bar_sync(256, 2);
        // tma store
        if (tid == 0) {
            store_async(&tensorMapO, sO, bs, hn, by * BM, 0);
            tma_store_arrive();
        }
    }
}


template<int BM, int BN, int DIM, int NUM_THREADS, int NUM_SMEM>
__global__  __launch_bounds__(NUM_THREADS) 
void attnWSKCXForNKernel(
    int B, int H, int S, 
    const __grid_constant__ CUtensorMap tensorMapQ, 
    const __grid_constant__ CUtensorMap tensorMapK, 
    const __grid_constant__ CUtensorMap tensorMapV, 
    const __grid_constant__ CUtensorMap tensorMapO
) {
    // WS attention
    const int bs = blockIdx.z;
    const int hn = blockIdx.y;
    const int by = blockIdx.x;
    const int tid = threadIdx.x;
    uint32_t wg_idx = tid >> 7;

    // mma size
    assert((DIM >= 256 || DIM == 16 || DIM == 32 || DIM == 64 || DIM == 128) && "DIM ERROR!");
    assert((BN >= 256 || BN == 16 || BN == 32 || BN == 64 || BN == 128) && "BN ERROR!");
    constexpr int MMA_M = 64;
    constexpr int QK_MMA_N = BN <= 256 ? BN : 256;
    constexpr int PV_MMA_N = DIM <= 256 ? DIM : 256;
    constexpr int MMA_K = 16;
    constexpr int N = BM / 2 / MMA_M;

    // setting shared memory
    extern __shared__ __align__(128) uint8_t smem[];
    SMemWS<BM, BN, DIM, NUM_SMEM> &s = *reinterpret_cast<SMemWS<BM, BN, DIM, NUM_SMEM>*>(smem);
    fp16 *sQ = s.Q, *sK = s.K, *sV = s.V, *sO = s.O;
    Barrier *Qmbar = &s.Qmbar;
    Barrier *Kempty = s.Kempty, *Vempty = s.Vempty, *Kfull = s.Kfull, *Vfull = s.Vfull;

    // init mbarrier
    if (threadIdx.x == 0) {
        init_barrier(Qmbar, 1);
        for (int i = 0; i < NUM_SMEM; ++i) {
            init_barrier(&Kfull[i], 1);  // 1 thread arrive
            init_barrier(&Vfull[i], 1);
            init_barrier(&Kempty[i], 256);  // 256 thread arrive
            init_barrier(&Vempty[i], 256);
        }
    }
    __syncthreads();
    fence_view_async_shared();

    // TMA load
    if (wg_idx == 2) {  // producer
        warpgroup_reg_dealloc<24>();
        if (tid == 256) {
            int smem_i = 0, phase = 0;
            // load Q
            expect_bytes(Qmbar, BM * DIM * sizeof(fp16));
            load_async(sQ, &tensorMapQ, Qmbar, bs, hn, by * BM, 0);
            for (size_t iw=0; iw<S; iw+=BN, ++smem_i) {
                if (smem_i >= NUM_SMEM) { smem_i = 0; phase ^= 1; }
                fp16 *KAddr = sK + smem_i * BN * DIM;
                fp16 *VAddr = sV + smem_i * BN * DIM;
                
                // load K
                wait(&Kempty[smem_i], phase);
                expect_bytes(&Kfull[smem_i], BN * DIM * sizeof(fp16));
                load_async(KAddr, &tensorMapK, &Kfull[smem_i], bs, hn, iw, 0);

                // load V
                wait(&Vempty[smem_i], phase);
                expect_bytes(&Vfull[smem_i], BN * DIM * sizeof(fp16));
                load_async(VAddr, &tensorMapV, &Vfull[smem_i], bs, hn, iw, 0);
            }
        }
    } else {  // consumer
        warpgroup_reg_alloc<240>();
        // Bootstrap empty-smem_i barriers so producer can issue the first K/V loads.
        #pragma unroll
        for (int st = 0; st < NUM_SMEM; ++st) {
            arrive(&Kempty[st]);
            arrive(&Vempty[st]);
        }

        // need args
        uint32_t lane_id = tid & 31;
        uint32_t warp_id_in_wg = (tid >> 5) & 0x3;
        const unsigned mask = __activemask();
        const fp32 scale = sqrt((1.0f / DIM)) * 1.44269504f;  // log2(e)

        // registers define
        fp32 acc_o[N][DIM/PV_MMA_N][PV_MMA_N/16][8];
        fp32 scores_max_prev[N][2];
        fp32 scores_max[N][2];
        fp32 logsum[N][2];

        // init acc_o
        #pragma unroll
        for (size_t i=0; i<N; i++) {
            #pragma unroll
            for (size_t j=0; j<DIM/PV_MMA_N; j++) {
                #pragma unroll
                for (size_t k=0; k<PV_MMA_N/16; k++) {
                    #pragma unroll
                    for (size_t l=0; l<8; l++) {
                        acc_o[i][j][k][l] = 0.0f;
                    }
                }
            }
        }
        
        // init logsum and scores_max
        #pragma unroll
        for (size_t i=0; i<N; i++) {
            #pragma unroll
            for (size_t j=0; j<2; j++) {
                logsum[i][j] = 0.0f;
                scores_max[i][j] = -FLT_MAX;
            }
        }
        
        wait(Qmbar, 0);
        // main for loop
        int smem_i = 0, phase = 0;
        for (size_t iw=0; iw<S; iw+=BN, ++smem_i) {
            if (smem_i >= NUM_SMEM) { smem_i = 0; phase ^= 1; }
            fp16 *KAddr = sK + smem_i * BN * DIM;
            fp16 *VAddr = sV + smem_i * BN * DIM;
            
            #pragma unroll
            for (size_t n=0; n<N; n++) {
                // fill acc_s
                fp32 acc_s[BN/QK_MMA_N][QK_MMA_N/16][8];
                #pragma unroll
                for (size_t j=0; j<BN/QK_MMA_N; j++) {
                    #pragma unroll
                    for (size_t k=0; k<QK_MMA_N/16; k++) {
                        #pragma unroll
                        for (size_t l=0; l<8; l++) {
                            acc_s[j][k][l] = 0.0f;
                        }
                    }
                }

                // gemm-qk
                wait(&Kfull[smem_i], phase);
                warpgroup_arrive();
                #pragma unroll
                for (size_t j=0; j<BN/QK_MMA_N; j++) {
                    #pragma unroll
                    for (size_t k=0; k<DIM; k+=MMA_K) {
                        int q_row = wg_idx * N * MMA_M + n * MMA_M;
                        int q_col = k;
                        int k_row = j * QK_MMA_N;
                        int k_col = k;
                        fp16 *_QAddr = sQ + tma_smem_offset_2d<BM>(q_row, q_col);
                        fp16 *_KAddr = KAddr + tma_smem_offset_2d<BN>(k_row, k_col);
                        wgmma_ss<QK_MMA_N, 1, 1, 1, 0, 0, 16, 1024, true>(acc_s[j], _QAddr, _KAddr);
                    }
                }
                warpgroup_commit_batch();
                warpgroup_wait();
                if (n == N-1) { arrive(&Kempty[smem_i]); };
                

                // max_prev = max
                #pragma unroll
                for (size_t j=0; j<2; j++) {
                    scores_max_prev[n][j] = scores_max[n][j];
                }

                // reduce max
                #pragma unroll
                for (size_t j=0; j<2; j++) {
                    scores_max[n][j] = -FLT_MAX;
                }
                #pragma unroll
                for (size_t j=0; j<BN/QK_MMA_N; j++) {
                    #pragma unroll
                    for (size_t k=0; k<QK_MMA_N/16; k++) {
                        scores_max[n][0] = max(acc_s[j][k][0], scores_max[n][0]);
                        scores_max[n][0] = max(acc_s[j][k][1], scores_max[n][0]);
                        scores_max[n][0] = max(acc_s[j][k][4], scores_max[n][0]);
                        scores_max[n][0] = max(acc_s[j][k][5], scores_max[n][0]);
                        scores_max[n][1] = max(acc_s[j][k][2], scores_max[n][1]);
                        scores_max[n][1] = max(acc_s[j][k][3], scores_max[n][1]);
                        scores_max[n][1] = max(acc_s[j][k][6], scores_max[n][1]);
                        scores_max[n][1] = max(acc_s[j][k][7], scores_max[n][1]);
                    }
                }
                #pragma unroll
                for (size_t j=0; j<2; j++) {
                    #pragma unroll
                    for (size_t k=1; k<4; k*=2) {
                        scores_max[n][j] = max(scores_max[n][j], __shfl_xor_sync(mask, scores_max[n][j], k, 4));
                    }
                }

                // m = max(pm, m)
                #pragma unroll
                for (size_t j=0; j<2; j++) {
                    scores_max[n][j] = max(scores_max_prev[n][j], scores_max[n][j]);
                }

                // scores_scale = exp2(pm  - m)
                fp32 scores_scale[2];
                #pragma unroll
                for (size_t j=0; j<2; j++) {
                    scores_scale[j] = exp2f(scores_max_prev[n][j] * scale - scores_max[n][j] * scale);
                }

                // acc_s = exp2(acc_s - m)
                #pragma unroll
                for (size_t j=0; j<BN/QK_MMA_N; j++) {
                    #pragma unroll
                    for (size_t k=0; k<QK_MMA_N/16; k++) {
                        acc_s[j][k][0] = exp2f(acc_s[j][k][0] * scale - scores_max[n][0] * scale);
                        acc_s[j][k][1] = exp2f(acc_s[j][k][1] * scale - scores_max[n][0] * scale);
                        acc_s[j][k][4] = exp2f(acc_s[j][k][4] * scale - scores_max[n][0] * scale);
                        acc_s[j][k][5] = exp2f(acc_s[j][k][5] * scale - scores_max[n][0] * scale);
                        acc_s[j][k][2] = exp2f(acc_s[j][k][2] * scale - scores_max[n][1] * scale);
                        acc_s[j][k][3] = exp2f(acc_s[j][k][3] * scale - scores_max[n][1] * scale);
                        acc_s[j][k][6] = exp2f(acc_s[j][k][6] * scale - scores_max[n][1] * scale);
                        acc_s[j][k][7] = exp2f(acc_s[j][k][7] * scale - scores_max[n][1] * scale);
                    }
                }

                // reduce sum
                fp32 scores_sum[2];
                #pragma unroll
                for (size_t j=0; j<2; j++) {
                    scores_sum[j] = 0.0f;
                }
                #pragma unroll
                for (size_t j=0; j<BN/QK_MMA_N; j++) {
                    #pragma unroll
                    for (size_t k=0; k<QK_MMA_N/16; k++) {
                        scores_sum[0] += (acc_s[j][k][0] + acc_s[j][k][1] + acc_s[j][k][4] + acc_s[j][k][5]);
                        scores_sum[1] += (acc_s[j][k][2] + acc_s[j][k][3] + acc_s[j][k][6] + acc_s[j][k][7]);
                    }
                }
                #pragma unroll
                for (size_t j=0; j<2; j++) {
                    #pragma unroll
                    for (size_t k=1; k<4; k*=2) {
                        scores_sum[j] += __shfl_xor_sync(mask, scores_sum[j], k, 4);
                    }
                }

                // logsum = logsum * scores_scale + sum;
                #pragma unroll
                for (size_t j=0; j<2; j++) {
                    logsum[n][j] = logsum[n][j] * scores_scale[j] + scores_sum[j];
                }

                // cast acc_s
                fp16 acc_s_cast[BN/QK_MMA_N][QK_MMA_N/16][8];
                #pragma unroll
                for (size_t j=0; j<BN/QK_MMA_N; j++) {
                    #pragma unroll
                    for (size_t k=0; k<QK_MMA_N/16; k++) {
                        #pragma unroll
                        for (size_t l=0; l<8; l+=2) {
                            uint1 _t2;
                            float2 _t1 = *(float2*)(&acc_s[j][k][l]);
                            *(half2*)(&(_t2)) = __float22half2_rn(*(float2*)(&(_t1)));
                            *(uint1*)(&acc_s_cast[j][k][l]) = _t2;
                        }
                    }
                }

                // acc_o = acc_o * scores_scale
                #pragma unroll
                for (size_t j=0; j<DIM/PV_MMA_N; j++) {
                    #pragma unroll
                    for (size_t k=0; k<PV_MMA_N/16; k++) {
                        acc_o[n][j][k][0] *= scores_scale[0];
                        acc_o[n][j][k][1] *= scores_scale[0];
                        acc_o[n][j][k][4] *= scores_scale[0];
                        acc_o[n][j][k][5] *= scores_scale[0];
                        acc_o[n][j][k][2] *= scores_scale[1];
                        acc_o[n][j][k][3] *= scores_scale[1];
                        acc_o[n][j][k][6] *= scores_scale[1];
                        acc_o[n][j][k][7] *= scores_scale[1];
                    }
                }

                // gemm-pv
                wait(&Vfull[smem_i], phase);
                warpgroup_arrive();
                #pragma unroll
                for (size_t j=0; j<DIM/PV_MMA_N; j++) {
                    #pragma unroll
                    for (size_t k=0; k<BN; k+=MMA_K) {
                        // V is stored in shared as [K=BN, N=DIM].
                        // Use TransB=1 so WGMMA consumes it logically as [N, K].
                        int v_row = k;
                        int v_col = j * PV_MMA_N;
                        const int p_tile_outer = k / QK_MMA_N;
                        const int p_tile_inner = (k % QK_MMA_N) / 16;
                        fp16 *_VAddr = VAddr + tma_smem_offset_2d<BN>(v_row, v_col);
                        wgmma_rs<PV_MMA_N, 1, 1, 1, 0, 1, BN * 64 * sizeof(fp16), 1024, true>(
                            acc_o[n][j],
                            reinterpret_cast<uint32_t*>(acc_s_cast[p_tile_outer][p_tile_inner]),
                            _VAddr);
                    }
                }
                warpgroup_commit_batch();
                warpgroup_wait();
                if (n == N-1) { arrive(&Vempty[smem_i]); };
            }
        }

        // acc_o = acc_o / logsum
        #pragma unroll
        for (size_t i=0; i<BM/(MMA_M*2); i++) {
            float val0 = 1.0f / logsum[i][0];
            float val1 = 1.0f / logsum[i][1];
            #pragma unroll
            for (size_t j=0; j<DIM/PV_MMA_N; j++) {
                #pragma unroll
                for (size_t k=0; k<PV_MMA_N/16; k++) {
                    acc_o[i][j][k][0] *= val0;
                    acc_o[i][j][k][1] *= val0;
                    acc_o[i][j][k][4] *= val0;
                    acc_o[i][j][k][5] *= val0;
                    acc_o[i][j][k][2] *= val1;
                    acc_o[i][j][k][3] *= val1;
                    acc_o[i][j][k][6] *= val1;
                    acc_o[i][j][k][7] *= val1;
                }
            }
        }

        // copy acc_o to sO (load matritx)
        if (tid == 0) {
            tma_store_wait();
        }

        const int lane_row = lane_id & 0xf;
        const int lane_col = (lane_id >> 4) * 8;
        #pragma unroll
        for (size_t i=0; i<BM/(MMA_M*2); i++) {
            #pragma unroll
            for (size_t j=0; j<DIM/PV_MMA_N; j++) {
                #pragma unroll
                for (size_t k=0; k<PV_MMA_N/16; k++) {
                    int o_row = wg_idx * BM/(MMA_M*2) * MMA_M
                                + i * MMA_M
                                + warp_id_in_wg * 16
                                + lane_row;
                    int o_col = j * PV_MMA_N
                                + k * 16
                                + lane_col;
                    // fp16 *_sO = sO + tma_smem_offset_2d<BM>(o_row, o_col);
                    fp16 *_sO = sO + tma_smem_swizzle_128b_offset_2d<BM>(o_row, o_col);
                        uint32_t r0 = half2_to_u32(__floats2half2_rn(
                            acc_o[i][j][k][0],
                            acc_o[i][j][k][1]
                        ));
                        uint32_t r1 = half2_to_u32(__floats2half2_rn(
                            acc_o[i][j][k][2],
                            acc_o[i][j][k][3]
                        ));
                        uint32_t r2 = half2_to_u32(__floats2half2_rn(
                            acc_o[i][j][k][4],
                            acc_o[i][j][k][5]
                        ));
                        uint32_t r3 = half2_to_u32(__floats2half2_rn(
                            acc_o[i][j][k][6],
                            acc_o[i][j][k][7]
                        ));

                        stmatrix_x4_reg(_sO, r0, r1, r2, r3);
                }
            }
        }
        // Wait all consumer threads (2 warpgroups) before issuing TMA store from tid==0.
        fence_view_async_shared();
        bar_sync(256, 2);
        // tma store
        if (tid == 0) {
            store_async(&tensorMapO, sO, bs, hn, by * BM, 0);
            tma_store_arrive();
        }
    }
}


template<int BM, int BN, int DIM, int NUM_THREADS, int NUM_SMEM>
__global__  __launch_bounds__(NUM_THREADS) 
void attnWSKCXForNLambdaUnrollKernel(
    int B, int H, int S, 
    const __grid_constant__ CUtensorMap tensorMapQ, 
    const __grid_constant__ CUtensorMap tensorMapK, 
    const __grid_constant__ CUtensorMap tensorMapV, 
    const __grid_constant__ CUtensorMap tensorMapO
) {
    // WS attention
    const int bs = blockIdx.z;
    const int hn = blockIdx.y;
    const int by = blockIdx.x;
    const int tid = threadIdx.x;
    uint32_t wg_idx = tid >> 7;

    // mma size
    assert((DIM >= 256 || DIM == 16 || DIM == 32 || DIM == 64 || DIM == 128) && "DIM ERROR!");
    assert((BN >= 256 || BN == 16 || BN == 32 || BN == 64 || BN == 128) && "BN ERROR!");
    constexpr int MMA_M = 64;
    constexpr int QK_MMA_N = BN <= 256 ? BN : 256;
    constexpr int PV_MMA_N = DIM <= 256 ? DIM : 256;
    constexpr int MMA_K = 16;
    constexpr int N = BM / 2 / MMA_M;

    // setting shared memory
    extern __shared__ __align__(128) uint8_t smem[];
    SMemWS<BM, BN, DIM, NUM_SMEM> &s = *reinterpret_cast<SMemWS<BM, BN, DIM, NUM_SMEM>*>(smem);
    fp16 *sQ = s.Q, *sK = s.K, *sV = s.V, *sO = s.O;
    Barrier *Qmbar = &s.Qmbar;
    Barrier *Kempty = s.Kempty, *Vempty = s.Vempty, *Kfull = s.Kfull, *Vfull = s.Vfull;

    // init mbarrier
    if (threadIdx.x == 0) {
        init_barrier(Qmbar, 1);
        for (int i = 0; i < NUM_SMEM; ++i) {
            init_barrier(&Kfull[i], 1);  // 1 thread arrive
            init_barrier(&Vfull[i], 1);
            init_barrier(&Kempty[i], 256);  // 256 thread arrive
            init_barrier(&Vempty[i], 256);
        }
    }
    __syncthreads();
    fence_view_async_shared();

    // TMA load
    if (wg_idx == 2) {  // producer
        warpgroup_reg_dealloc<24>();
        if (tid == 256) {
            int smem_i = 0, phase = 0;
            // load Q
            expect_bytes(Qmbar, BM * DIM * sizeof(fp16));
            load_async(sQ, &tensorMapQ, Qmbar, bs, hn, by * BM, 0);
            for (size_t iw=0; iw<S; iw+=BN, ++smem_i) {
                if (smem_i >= NUM_SMEM) { smem_i = 0; phase ^= 1; }
                fp16 *KAddr = sK + smem_i * BN * DIM;
                fp16 *VAddr = sV + smem_i * BN * DIM;
                
                // load K
                wait(&Kempty[smem_i], phase);
                expect_bytes(&Kfull[smem_i], BN * DIM * sizeof(fp16));
                load_async(KAddr, &tensorMapK, &Kfull[smem_i], bs, hn, iw, 0);

                // load V
                wait(&Vempty[smem_i], phase);
                expect_bytes(&Vfull[smem_i], BN * DIM * sizeof(fp16));
                load_async(VAddr, &tensorMapV, &Vfull[smem_i], bs, hn, iw, 0);
            }
        }
    } else {  // consumer
        warpgroup_reg_alloc<240>();
        // Bootstrap empty-smem_i barriers so producer can issue the first K/V loads.
        #pragma unroll
        for (int st = 0; st < NUM_SMEM; ++st) {
            arrive(&Kempty[st]);
            arrive(&Vempty[st]);
        }

        // need args
        uint32_t lane_id = tid & 31;
        uint32_t warp_id_in_wg = (tid >> 5) & 0x3;
        const unsigned mask = __activemask();
        const fp32 scale = sqrt((1.0f / DIM)) * 1.44269504f;  // log2(e)

        // registers define
        fp32 acc_o[N][DIM/PV_MMA_N][PV_MMA_N/16][8];
        fp32 scores_max_prev[N][2];
        fp32 scores_max[N][2];
        fp32 logsum[N][2];

        // init acc_o
        #pragma unroll
        for (size_t i=0; i<N; i++) {
            #pragma unroll
            for (size_t j=0; j<DIM/PV_MMA_N; j++) {
                #pragma unroll
                for (size_t k=0; k<PV_MMA_N/16; k++) {
                    #pragma unroll
                    for (size_t l=0; l<8; l++) {
                        acc_o[i][j][k][l] = 0.0f;
                    }
                }
            }
        }
        
        // init logsum and scores_max
        #pragma unroll
        for (size_t i=0; i<N; i++) {
            #pragma unroll
            for (size_t j=0; j<2; j++) {
                logsum[i][j] = 0.0f;
                scores_max[i][j] = -FLT_MAX;
            }
        }
        
        wait(Qmbar, 0);
        // main for loop
        int smem_i = 0, phase = 0;
        for (size_t iw=0; iw<S; iw+=BN, ++smem_i) {
            if (smem_i >= NUM_SMEM) { smem_i = 0; phase ^= 1; }
            fp16 *KAddr = sK + smem_i * BN * DIM;
            fp16 *VAddr = sV + smem_i * BN * DIM;
            
            auto compute_n = [&](auto n_const) {
                constexpr int n = decltype(n_const)::value;
                static_assert(n < N);

                // fill acc_s
                fp32 acc_s[BN/QK_MMA_N][QK_MMA_N/16][8];
                #pragma unroll
                for (size_t j=0; j<BN/QK_MMA_N; j++) {
                    #pragma unroll
                    for (size_t k=0; k<QK_MMA_N/16; k++) {
                        #pragma unroll
                        for (size_t l=0; l<8; l++) {
                            acc_s[j][k][l] = 0.0f;
                        }
                    }
                }

                // gemm-qk
                if constexpr (n == 0) { wait(&Kfull[smem_i], phase); }
                warpgroup_arrive();
                #pragma unroll
                for (size_t j=0; j<BN/QK_MMA_N; j++) {
                    #pragma unroll
                    for (size_t k=0; k<DIM; k+=MMA_K) {
                        int q_row = wg_idx * N * MMA_M + n * MMA_M;
                        int q_col = k;
                        int k_row = j * QK_MMA_N;
                        int k_col = k;
                        fp16 *_QAddr = sQ + tma_smem_offset_2d<BM>(q_row, q_col);
                        fp16 *_KAddr = KAddr + tma_smem_offset_2d<BN>(k_row, k_col);
                        wgmma_ss<QK_MMA_N, 1, 1, 1, 0, 0, 16, 1024, true>(acc_s[j], _QAddr, _KAddr);
                    }
                }
                warpgroup_commit_batch();
                warpgroup_wait();
                if constexpr (n == N-1) { arrive(&Kempty[smem_i]); }
                

                // max_prev = max
                #pragma unroll
                for (size_t j=0; j<2; j++) {
                    scores_max_prev[n][j] = scores_max[n][j];
                }

                // reduce max
                #pragma unroll
                for (size_t j=0; j<2; j++) {
                    scores_max[n][j] = -FLT_MAX;
                }
                #pragma unroll
                for (size_t j=0; j<BN/QK_MMA_N; j++) {
                    #pragma unroll
                    for (size_t k=0; k<QK_MMA_N/16; k++) {
                        scores_max[n][0] = max(acc_s[j][k][0], scores_max[n][0]);
                        scores_max[n][0] = max(acc_s[j][k][1], scores_max[n][0]);
                        scores_max[n][0] = max(acc_s[j][k][4], scores_max[n][0]);
                        scores_max[n][0] = max(acc_s[j][k][5], scores_max[n][0]);
                        scores_max[n][1] = max(acc_s[j][k][2], scores_max[n][1]);
                        scores_max[n][1] = max(acc_s[j][k][3], scores_max[n][1]);
                        scores_max[n][1] = max(acc_s[j][k][6], scores_max[n][1]);
                        scores_max[n][1] = max(acc_s[j][k][7], scores_max[n][1]);
                    }
                }
                #pragma unroll
                for (size_t j=0; j<2; j++) {
                    #pragma unroll
                    for (size_t k=1; k<4; k*=2) {
                        scores_max[n][j] = max(scores_max[n][j], __shfl_xor_sync(mask, scores_max[n][j], k, 4));
                    }
                }

                // m = max(pm, m)
                #pragma unroll
                for (size_t j=0; j<2; j++) {
                    scores_max[n][j] = max(scores_max_prev[n][j], scores_max[n][j]);
                }

                // scores_scale = exp2(pm  - m)
                fp32 scores_scale[2];
                #pragma unroll
                for (size_t j=0; j<2; j++) {
                    scores_scale[j] = exp2f(scores_max_prev[n][j] * scale - scores_max[n][j] * scale);
                }

                // acc_s = exp2(acc_s - m)
                #pragma unroll
                for (size_t j=0; j<BN/QK_MMA_N; j++) {
                    #pragma unroll
                    for (size_t k=0; k<QK_MMA_N/16; k++) {
                        acc_s[j][k][0] = exp2f(acc_s[j][k][0] * scale - scores_max[n][0] * scale);
                        acc_s[j][k][1] = exp2f(acc_s[j][k][1] * scale - scores_max[n][0] * scale);
                        acc_s[j][k][4] = exp2f(acc_s[j][k][4] * scale - scores_max[n][0] * scale);
                        acc_s[j][k][5] = exp2f(acc_s[j][k][5] * scale - scores_max[n][0] * scale);
                        acc_s[j][k][2] = exp2f(acc_s[j][k][2] * scale - scores_max[n][1] * scale);
                        acc_s[j][k][3] = exp2f(acc_s[j][k][3] * scale - scores_max[n][1] * scale);
                        acc_s[j][k][6] = exp2f(acc_s[j][k][6] * scale - scores_max[n][1] * scale);
                        acc_s[j][k][7] = exp2f(acc_s[j][k][7] * scale - scores_max[n][1] * scale);
                    }
                }

                // reduce sum
                fp32 scores_sum[2];
                #pragma unroll
                for (size_t j=0; j<2; j++) {
                    scores_sum[j] = 0.0f;
                }
                #pragma unroll
                for (size_t j=0; j<BN/QK_MMA_N; j++) {
                    #pragma unroll
                    for (size_t k=0; k<QK_MMA_N/16; k++) {
                        scores_sum[0] += (acc_s[j][k][0] + acc_s[j][k][1] + acc_s[j][k][4] + acc_s[j][k][5]);
                        scores_sum[1] += (acc_s[j][k][2] + acc_s[j][k][3] + acc_s[j][k][6] + acc_s[j][k][7]);
                    }
                }
                #pragma unroll
                for (size_t j=0; j<2; j++) {
                    #pragma unroll
                    for (size_t k=1; k<4; k*=2) {
                        scores_sum[j] += __shfl_xor_sync(mask, scores_sum[j], k, 4);
                    }
                }

                // logsum = logsum * scores_scale + sum;
                #pragma unroll
                for (size_t j=0; j<2; j++) {
                    logsum[n][j] = logsum[n][j] * scores_scale[j] + scores_sum[j];
                }

                // cast acc_s
                fp16 acc_s_cast[BN/QK_MMA_N][QK_MMA_N/16][8];
                #pragma unroll
                for (size_t j=0; j<BN/QK_MMA_N; j++) {
                    #pragma unroll
                    for (size_t k=0; k<QK_MMA_N/16; k++) {
                        #pragma unroll
                        for (size_t l=0; l<8; l+=2) {
                            uint1 _t2;
                            float2 _t1 = *(float2*)(&acc_s[j][k][l]);
                            *(half2*)(&(_t2)) = __float22half2_rn(*(float2*)(&(_t1)));
                            *(uint1*)(&acc_s_cast[j][k][l]) = _t2;
                        }
                    }
                }

                // acc_o = acc_o * scores_scale
                #pragma unroll
                for (size_t j=0; j<DIM/PV_MMA_N; j++) {
                    #pragma unroll
                    for (size_t k=0; k<PV_MMA_N/16; k++) {
                        acc_o[n][j][k][0] *= scores_scale[0];
                        acc_o[n][j][k][1] *= scores_scale[0];
                        acc_o[n][j][k][4] *= scores_scale[0];
                        acc_o[n][j][k][5] *= scores_scale[0];
                        acc_o[n][j][k][2] *= scores_scale[1];
                        acc_o[n][j][k][3] *= scores_scale[1];
                        acc_o[n][j][k][6] *= scores_scale[1];
                        acc_o[n][j][k][7] *= scores_scale[1];
                    }
                }

                // gemm-pv
                if constexpr (n == 0) { wait(&Vfull[smem_i], phase); }
                warpgroup_arrive();
                #pragma unroll
                for (size_t j=0; j<DIM/PV_MMA_N; j++) {
                    #pragma unroll
                    for (size_t k=0; k<BN; k+=MMA_K) {
                        // V is stored in shared as [K=BN, N=DIM].
                        // Use TransB=1 so WGMMA consumes it logically as [N, K].
                        int v_row = k;
                        int v_col = j * PV_MMA_N;
                        const int p_tile_outer = k / QK_MMA_N;
                        const int p_tile_inner = (k % QK_MMA_N) / 16;
                        fp16 *_VAddr = VAddr + tma_smem_offset_2d<BN>(v_row, v_col);
                        wgmma_rs<PV_MMA_N, 1, 1, 1, 0, 1, BN * 64 * sizeof(fp16), 1024, true>(
                            acc_o[n][j],
                            reinterpret_cast<uint32_t*>(acc_s_cast[p_tile_outer][p_tile_inner]),
                            _VAddr);
                    }
                }
                warpgroup_commit_batch();
                warpgroup_wait();
                if constexpr (n == N-1) { arrive(&Vempty[smem_i]); }
            };

            compute_n(std::integral_constant<int, 0>{});
            if constexpr (N > 1) { compute_n(std::integral_constant<int, 1>{}); }
            if constexpr (N > 2) { compute_n(std::integral_constant<int, 2>{}); }
            if constexpr (N > 3) { compute_n(std::integral_constant<int, 3>{}); }
        }

        // acc_o = acc_o / logsum
        #pragma unroll
        for (size_t i=0; i<BM/(MMA_M*2); i++) {
            float val0 = 1.0f / logsum[i][0];
            float val1 = 1.0f / logsum[i][1];
            #pragma unroll
            for (size_t j=0; j<DIM/PV_MMA_N; j++) {
                #pragma unroll
                for (size_t k=0; k<PV_MMA_N/16; k++) {
                    acc_o[i][j][k][0] *= val0;
                    acc_o[i][j][k][1] *= val0;
                    acc_o[i][j][k][4] *= val0;
                    acc_o[i][j][k][5] *= val0;
                    acc_o[i][j][k][2] *= val1;
                    acc_o[i][j][k][3] *= val1;
                    acc_o[i][j][k][6] *= val1;
                    acc_o[i][j][k][7] *= val1;
                }
            }
        }

        // copy acc_o to sO (load matritx)
        if (tid == 0) {
            tma_store_wait();
        }

        const int lane_row = lane_id & 0xf;
        const int lane_col = (lane_id >> 4) * 8;
        #pragma unroll
        for (size_t i=0; i<BM/(MMA_M*2); i++) {
            #pragma unroll
            for (size_t j=0; j<DIM/PV_MMA_N; j++) {
                #pragma unroll
                for (size_t k=0; k<PV_MMA_N/16; k++) {
                    int o_row = wg_idx * BM/(MMA_M*2) * MMA_M
                                + i * MMA_M
                                + warp_id_in_wg * 16
                                + lane_row;
                    int o_col = j * PV_MMA_N
                                + k * 16
                                + lane_col;
                    // fp16 *_sO = sO + tma_smem_offset_2d<BM>(o_row, o_col);
                    fp16 *_sO = sO + tma_smem_swizzle_128b_offset_2d<BM>(o_row, o_col);
                        uint32_t r0 = half2_to_u32(__floats2half2_rn(
                            acc_o[i][j][k][0],
                            acc_o[i][j][k][1]
                        ));
                        uint32_t r1 = half2_to_u32(__floats2half2_rn(
                            acc_o[i][j][k][2],
                            acc_o[i][j][k][3]
                        ));
                        uint32_t r2 = half2_to_u32(__floats2half2_rn(
                            acc_o[i][j][k][4],
                            acc_o[i][j][k][5]
                        ));
                        uint32_t r3 = half2_to_u32(__floats2half2_rn(
                            acc_o[i][j][k][6],
                            acc_o[i][j][k][7]
                        ));

                        stmatrix_x4_reg(_sO, r0, r1, r2, r3);
                }
            }
        }
        // Wait all consumer threads (2 warpgroups) before issuing TMA store from tid==0.
        fence_view_async_shared();
        bar_sync(256, 2);
        // tma store
        if (tid == 0) {
            store_async(&tensorMapO, sO, bs, hn, by * BM, 0);
            tma_store_arrive();
        }
    }
}


template<int BM, int BN, int DIM, int NUM_THREADS, int NUM_SMEM>
__global__  __launch_bounds__(NUM_THREADS) 
void attnWSKCXForNManualUnrollKernel(
    int B, int H, int S, 
    const __grid_constant__ CUtensorMap tensorMapQ, 
    const __grid_constant__ CUtensorMap tensorMapK, 
    const __grid_constant__ CUtensorMap tensorMapV, 
    const __grid_constant__ CUtensorMap tensorMapO
) {
    // WS attention
    const int bs = blockIdx.z;
    const int hn = blockIdx.y;
    const int by = blockIdx.x;
    const int tid = threadIdx.x;
    uint32_t wg_idx = tid >> 7;

    // mma size
    assert((DIM >= 256 || DIM == 16 || DIM == 32 || DIM == 64 || DIM == 128) && "DIM ERROR!");
    assert((BN >= 256 || BN == 16 || BN == 32 || BN == 64 || BN == 128) && "BN ERROR!");
    constexpr int MMA_M = 64;
    constexpr int QK_MMA_N = BN <= 256 ? BN : 256;
    constexpr int PV_MMA_N = DIM <= 256 ? DIM : 256;
    constexpr int MMA_K = 16;
    constexpr int N = BM / 2 / MMA_M;

    // setting shared memory
    extern __shared__ __align__(128) uint8_t smem[];
    SMemWS<BM, BN, DIM, NUM_SMEM> &s = *reinterpret_cast<SMemWS<BM, BN, DIM, NUM_SMEM>*>(smem);
    fp16 *sQ = s.Q, *sK = s.K, *sV = s.V, *sO = s.O;
    Barrier *Qmbar = &s.Qmbar;
    Barrier *Kempty = s.Kempty, *Vempty = s.Vempty, *Kfull = s.Kfull, *Vfull = s.Vfull;

    // init mbarrier
    if (threadIdx.x == 0) {
        init_barrier(Qmbar, 1);
        for (int i = 0; i < NUM_SMEM; ++i) {
            init_barrier(&Kfull[i], 1);  // 1 thread arrive
            init_barrier(&Vfull[i], 1);
            init_barrier(&Kempty[i], 256);  // 256 thread arrive
            init_barrier(&Vempty[i], 256);
        }
    }
    __syncthreads();
    fence_view_async_shared();

    // TMA load
    if (wg_idx == 2) {  // producer
        warpgroup_reg_dealloc<24>();
        if (tid == 256) {
            int smem_i = 0, phase = 0;
            // load Q
            expect_bytes(Qmbar, BM * DIM * sizeof(fp16));
            load_async(sQ, &tensorMapQ, Qmbar, bs, hn, by * BM, 0);
            for (size_t iw=0; iw<S; iw+=BN, ++smem_i) {
                if (smem_i >= NUM_SMEM) { smem_i = 0; phase ^= 1; }
                fp16 *KAddr = sK + smem_i * BN * DIM;
                fp16 *VAddr = sV + smem_i * BN * DIM;
                
                // load K
                wait(&Kempty[smem_i], phase);
                expect_bytes(&Kfull[smem_i], BN * DIM * sizeof(fp16));
                load_async(KAddr, &tensorMapK, &Kfull[smem_i], bs, hn, iw, 0);

                // load V
                wait(&Vempty[smem_i], phase);
                expect_bytes(&Vfull[smem_i], BN * DIM * sizeof(fp16));
                load_async(VAddr, &tensorMapV, &Vfull[smem_i], bs, hn, iw, 0);
            }
        }
    } else {  // consumer
        warpgroup_reg_alloc<240>();
        // Bootstrap empty-smem_i barriers so producer can issue the first K/V loads.
        #pragma unroll
        for (int st = 0; st < NUM_SMEM; ++st) {
            arrive(&Kempty[st]);
            arrive(&Vempty[st]);
        }

        // need args
        uint32_t lane_id = tid & 31;
        uint32_t warp_id_in_wg = (tid >> 5) & 0x3;
        const unsigned mask = __activemask();
        const fp32 scale = sqrt((1.0f / DIM)) * 1.44269504f;  // log2(e)

        // registers define
        fp32 acc_o[N][DIM/PV_MMA_N][PV_MMA_N/16][8];
        fp32 scores_max_prev[N][2];
        fp32 scores_max[N][2];
        fp32 logsum[N][2];

        // init acc_o
        #pragma unroll
        for (size_t i=0; i<N; i++) {
            #pragma unroll
            for (size_t j=0; j<DIM/PV_MMA_N; j++) {
                #pragma unroll
                for (size_t k=0; k<PV_MMA_N/16; k++) {
                    #pragma unroll
                    for (size_t l=0; l<8; l++) {
                        acc_o[i][j][k][l] = 0.0f;
                    }
                }
            }
        }
        
        // init logsum and scores_max
        #pragma unroll
        for (size_t i=0; i<N; i++) {
            #pragma unroll
            for (size_t j=0; j<2; j++) {
                logsum[i][j] = 0.0f;
                scores_max[i][j] = -FLT_MAX;
            }
        }
        
        wait(Qmbar, 0);
        // main for loop
        int smem_i = 0, phase = 0;
        for (size_t iw=0; iw<S; iw+=BN, ++smem_i) {
            if (smem_i >= NUM_SMEM) { smem_i = 0; phase ^= 1; }
            fp16 *KAddr = sK + smem_i * BN * DIM;
            fp16 *VAddr = sV + smem_i * BN * DIM;

            // fill acc_s
            fp32 acc_s[BN/QK_MMA_N][QK_MMA_N/16][8];
            #pragma unroll
            for (size_t j=0; j<BN/QK_MMA_N; j++) {
                #pragma unroll
                for (size_t k=0; k<QK_MMA_N/16; k++) {
                    #pragma unroll
                    for (size_t l=0; l<8; l++) {
                        acc_s[j][k][l] = 0.0f;
                    }
                }
            }
            // gemm-qk
            wait(&Kfull[smem_i], phase);
            warpgroup_arrive();
            #pragma unroll
            for (size_t j=0; j<BN/QK_MMA_N; j++) {
                #pragma unroll
                for (size_t k=0; k<DIM; k+=MMA_K) {
                    int q_row = wg_idx * N * MMA_M + 0 * MMA_M;
                    int q_col = k;
                    int k_row = j * QK_MMA_N;
                    int k_col = k;
                    fp16 *_QAddr = sQ + tma_smem_offset_2d<BM>(q_row, q_col);
                    fp16 *_KAddr = KAddr + tma_smem_offset_2d<BN>(k_row, k_col);
                    wgmma_ss<QK_MMA_N, 1, 1, 1, 0, 0, 16, 1024, true>(acc_s[j], _QAddr, _KAddr);
                }
            }
            warpgroup_commit_batch();
            warpgroup_wait();

            // max_prev = max
            #pragma unroll
            for (size_t j=0; j<2; j++) {
                scores_max_prev[0][j] = scores_max[0][j];
            }
            // reduce max
            #pragma unroll
            for (size_t j=0; j<2; j++) {
                scores_max[0][j] = -FLT_MAX;
            }
            #pragma unroll
            for (size_t j=0; j<BN/QK_MMA_N; j++) {
                #pragma unroll
                for (size_t k=0; k<QK_MMA_N/16; k++) {
                    scores_max[0][0] = max(acc_s[j][k][0], scores_max[0][0]);
                    scores_max[0][0] = max(acc_s[j][k][1], scores_max[0][0]);
                    scores_max[0][0] = max(acc_s[j][k][4], scores_max[0][0]);
                    scores_max[0][0] = max(acc_s[j][k][5], scores_max[0][0]);
                    scores_max[0][1] = max(acc_s[j][k][2], scores_max[0][1]);
                    scores_max[0][1] = max(acc_s[j][k][3], scores_max[0][1]);
                    scores_max[0][1] = max(acc_s[j][k][6], scores_max[0][1]);
                    scores_max[0][1] = max(acc_s[j][k][7], scores_max[0][1]);
                }
            }
            #pragma unroll
            for (size_t j=0; j<2; j++) {
                #pragma unroll
                for (size_t k=1; k<4; k*=2) {
                    scores_max[0][j] = max(scores_max[0][j], __shfl_xor_sync(mask, scores_max[0][j], k, 4));
                }
            }
            // m = max(pm, m)
            #pragma unroll
            for (size_t j=0; j<2; j++) {
                scores_max[0][j] = max(scores_max_prev[0][j], scores_max[0][j]);
            }
            // scores_scale = exp2(pm  - m)
            fp32 scores_scale[2];
            #pragma unroll
            for (size_t j=0; j<2; j++) {
                scores_scale[j] = exp2f(scores_max_prev[0][j] * scale - scores_max[0][j] * scale);
            }
            // acc_s = exp2(acc_s - m)
            #pragma unroll
            for (size_t j=0; j<BN/QK_MMA_N; j++) {
                #pragma unroll
                for (size_t k=0; k<QK_MMA_N/16; k++) {
                    acc_s[j][k][0] = exp2f(acc_s[j][k][0] * scale - scores_max[0][0] * scale);
                    acc_s[j][k][1] = exp2f(acc_s[j][k][1] * scale - scores_max[0][0] * scale);
                    acc_s[j][k][4] = exp2f(acc_s[j][k][4] * scale - scores_max[0][0] * scale);
                    acc_s[j][k][5] = exp2f(acc_s[j][k][5] * scale - scores_max[0][0] * scale);
                    acc_s[j][k][2] = exp2f(acc_s[j][k][2] * scale - scores_max[0][1] * scale);
                    acc_s[j][k][3] = exp2f(acc_s[j][k][3] * scale - scores_max[0][1] * scale);
                    acc_s[j][k][6] = exp2f(acc_s[j][k][6] * scale - scores_max[0][1] * scale);
                    acc_s[j][k][7] = exp2f(acc_s[j][k][7] * scale - scores_max[0][1] * scale);
                }
            }
            // reduce sum
            fp32 scores_sum[2];
            #pragma unroll
            for (size_t j=0; j<2; j++) {
                scores_sum[j] = 0.0f;
            }
            #pragma unroll
            for (size_t j=0; j<BN/QK_MMA_N; j++) {
                #pragma unroll
                for (size_t k=0; k<QK_MMA_N/16; k++) {
                    scores_sum[0] += (acc_s[j][k][0] + acc_s[j][k][1] + acc_s[j][k][4] + acc_s[j][k][5]);
                    scores_sum[1] += (acc_s[j][k][2] + acc_s[j][k][3] + acc_s[j][k][6] + acc_s[j][k][7]);
                }
            }
            #pragma unroll
            for (size_t j=0; j<2; j++) {
                #pragma unroll
                for (size_t k=1; k<4; k*=2) {
                    scores_sum[j] += __shfl_xor_sync(mask, scores_sum[j], k, 4);
                }
            }
            // logsum = logsum * scores_scale + sum;
            #pragma unroll
            for (size_t j=0; j<2; j++) {
                logsum[0][j] = logsum[0][j] * scores_scale[j] + scores_sum[j];
            }
            // cast acc_s
            fp16 acc_s_cast[BN/QK_MMA_N][QK_MMA_N/16][8];
            #pragma unroll
            for (size_t j=0; j<BN/QK_MMA_N; j++) {
                #pragma unroll
                for (size_t k=0; k<QK_MMA_N/16; k++) {
                    #pragma unroll
                    for (size_t l=0; l<8; l+=2) {
                        uint1 _t2;
                        float2 _t1 = *(float2*)(&acc_s[j][k][l]);
                        *(half2*)(&(_t2)) = __float22half2_rn(*(float2*)(&(_t1)));
                        *(uint1*)(&acc_s_cast[j][k][l]) = _t2;
                    }
                }
            }
            // acc_o = acc_o * scores_scale
            #pragma unroll
            for (size_t j=0; j<DIM/PV_MMA_N; j++) {
                #pragma unroll
                for (size_t k=0; k<PV_MMA_N/16; k++) {
                    acc_o[0][j][k][0] *= scores_scale[0];
                    acc_o[0][j][k][1] *= scores_scale[0];
                    acc_o[0][j][k][4] *= scores_scale[0];
                    acc_o[0][j][k][5] *= scores_scale[0];
                    acc_o[0][j][k][2] *= scores_scale[1];
                    acc_o[0][j][k][3] *= scores_scale[1];
                    acc_o[0][j][k][6] *= scores_scale[1];
                    acc_o[0][j][k][7] *= scores_scale[1];
                }
            }

            // gemm-pv
            wait(&Vfull[smem_i], phase);
            warpgroup_arrive();
            #pragma unroll
            for (size_t j=0; j<DIM/PV_MMA_N; j++) {
                #pragma unroll
                for (size_t k=0; k<BN; k+=MMA_K) {
                    // V is stored in shared as [K=BN, N=DIM].
                    // Use TransB=1 so WGMMA consumes it logically as [N, K].
                    int v_row = k;
                    int v_col = j * PV_MMA_N;
                    const int p_tile_outer = k / QK_MMA_N;
                    const int p_tile_inner = (k % QK_MMA_N) / 16;
                    fp16 *_VAddr = VAddr + tma_smem_offset_2d<BN>(v_row, v_col);
                    wgmma_rs<PV_MMA_N, 1, 1, 1, 0, 1, BN * 64 * sizeof(fp16), 1024, true>(
                        acc_o[0][j],
                        reinterpret_cast<uint32_t*>(acc_s_cast[p_tile_outer][p_tile_inner]),
                        _VAddr);
                }
            }
            warpgroup_commit_batch();
            warpgroup_wait();


            // fill acc_s
            #pragma unroll
            for (size_t j=0; j<BN/QK_MMA_N; j++) {
                #pragma unroll
                for (size_t k=0; k<QK_MMA_N/16; k++) {
                    #pragma unroll
                    for (size_t l=0; l<8; l++) {
                        acc_s[j][k][l] = 0.0f;
                    }
                }
            }
            // gemm-qk
            warpgroup_arrive();
            #pragma unroll
            for (size_t j=0; j<BN/QK_MMA_N; j++) {
                #pragma unroll
                for (size_t k=0; k<DIM; k+=MMA_K) {
                    int q_row = wg_idx * N * MMA_M + 1 * MMA_M;
                    int q_col = k;
                    int k_row = j * QK_MMA_N;
                    int k_col = k;
                    fp16 *_QAddr = sQ + tma_smem_offset_2d<BM>(q_row, q_col);
                    fp16 *_KAddr = KAddr + tma_smem_offset_2d<BN>(k_row, k_col);
                    wgmma_ss<QK_MMA_N, 1, 1, 1, 0, 0, 16, 1024, true>(acc_s[j], _QAddr, _KAddr);
                }
            }
            warpgroup_commit_batch();
            warpgroup_wait();
            arrive(&Kempty[smem_i]);

            // max_prev = max
            #pragma unroll
            for (size_t j=0; j<2; j++) {
                scores_max_prev[1][j] = scores_max[1][j];
            }
            // reduce max
            #pragma unroll
            for (size_t j=0; j<2; j++) {
                scores_max[1][j] = -FLT_MAX;
            }
            #pragma unroll
            for (size_t j=0; j<BN/QK_MMA_N; j++) {
                #pragma unroll
                for (size_t k=0; k<QK_MMA_N/16; k++) {
                    scores_max[1][0] = max(acc_s[j][k][0], scores_max[1][0]);
                    scores_max[1][0] = max(acc_s[j][k][1], scores_max[1][0]);
                    scores_max[1][0] = max(acc_s[j][k][4], scores_max[1][0]);
                    scores_max[1][0] = max(acc_s[j][k][5], scores_max[1][0]);
                    scores_max[1][1] = max(acc_s[j][k][2], scores_max[1][1]);
                    scores_max[1][1] = max(acc_s[j][k][3], scores_max[1][1]);
                    scores_max[1][1] = max(acc_s[j][k][6], scores_max[1][1]);
                    scores_max[1][1] = max(acc_s[j][k][7], scores_max[1][1]);
                }
            }
            #pragma unroll
            for (size_t j=0; j<2; j++) {
                #pragma unroll
                for (size_t k=1; k<4; k*=2) {
                    scores_max[1][j] = max(scores_max[1][j], __shfl_xor_sync(mask, scores_max[1][j], k, 4));
                }
            }
            // m = max(pm, m)
            #pragma unroll
            for (size_t j=0; j<2; j++) {
                scores_max[1][j] = max(scores_max_prev[1][j], scores_max[1][j]);
            }
            // scores_scale = exp2(pm  - m)
            #pragma unroll
            for (size_t j=0; j<2; j++) {
                scores_scale[j] = exp2f(scores_max_prev[1][j] * scale - scores_max[1][j] * scale);
            }
            // acc_s = exp2(acc_s - m)
            #pragma unroll
            for (size_t j=0; j<BN/QK_MMA_N; j++) {
                #pragma unroll
                for (size_t k=0; k<QK_MMA_N/16; k++) {
                    acc_s[j][k][0] = exp2f(acc_s[j][k][0] * scale - scores_max[1][0] * scale);
                    acc_s[j][k][1] = exp2f(acc_s[j][k][1] * scale - scores_max[1][0] * scale);
                    acc_s[j][k][4] = exp2f(acc_s[j][k][4] * scale - scores_max[1][0] * scale);
                    acc_s[j][k][5] = exp2f(acc_s[j][k][5] * scale - scores_max[1][0] * scale);
                    acc_s[j][k][2] = exp2f(acc_s[j][k][2] * scale - scores_max[1][1] * scale);
                    acc_s[j][k][3] = exp2f(acc_s[j][k][3] * scale - scores_max[1][1] * scale);
                    acc_s[j][k][6] = exp2f(acc_s[j][k][6] * scale - scores_max[1][1] * scale);
                    acc_s[j][k][7] = exp2f(acc_s[j][k][7] * scale - scores_max[1][1] * scale);
                }
            }
            // reduce sum
            #pragma unroll
            for (size_t j=0; j<2; j++) {
                scores_sum[j] = 0.0f;
            }
            #pragma unroll
            for (size_t j=0; j<BN/QK_MMA_N; j++) {
                #pragma unroll
                for (size_t k=0; k<QK_MMA_N/16; k++) {
                    scores_sum[0] += (acc_s[j][k][0] + acc_s[j][k][1] + acc_s[j][k][4] + acc_s[j][k][5]);
                    scores_sum[1] += (acc_s[j][k][2] + acc_s[j][k][3] + acc_s[j][k][6] + acc_s[j][k][7]);
                }
            }
            #pragma unroll
            for (size_t j=0; j<2; j++) {
                #pragma unroll
                for (size_t k=1; k<4; k*=2) {
                    scores_sum[j] += __shfl_xor_sync(mask, scores_sum[j], k, 4);
                }
            }
            // logsum = logsum * scores_scale + sum;
            #pragma unroll
            for (size_t j=0; j<2; j++) {
                logsum[1][j] = logsum[1][j] * scores_scale[j] + scores_sum[j];
            }
            // cast acc_s
            #pragma unroll
            for (size_t j=0; j<BN/QK_MMA_N; j++) {
                #pragma unroll
                for (size_t k=0; k<QK_MMA_N/16; k++) {
                    #pragma unroll
                    for (size_t l=0; l<8; l+=2) {
                        uint1 _t2;
                        float2 _t1 = *(float2*)(&acc_s[j][k][l]);
                        *(half2*)(&(_t2)) = __float22half2_rn(*(float2*)(&(_t1)));
                        *(uint1*)(&acc_s_cast[j][k][l]) = _t2;
                    }
                }
            }
            // acc_o = acc_o * scores_scale
            #pragma unroll
            for (size_t j=0; j<DIM/PV_MMA_N; j++) {
                #pragma unroll
                for (size_t k=0; k<PV_MMA_N/16; k++) {
                    acc_o[1][j][k][0] *= scores_scale[0];
                    acc_o[1][j][k][1] *= scores_scale[0];
                    acc_o[1][j][k][4] *= scores_scale[0];
                    acc_o[1][j][k][5] *= scores_scale[0];
                    acc_o[1][j][k][2] *= scores_scale[1];
                    acc_o[1][j][k][3] *= scores_scale[1];
                    acc_o[1][j][k][6] *= scores_scale[1];
                    acc_o[1][j][k][7] *= scores_scale[1];
                }
            }

            // gemm-pv
            warpgroup_arrive();
            #pragma unroll
            for (size_t j=0; j<DIM/PV_MMA_N; j++) {
                #pragma unroll
                for (size_t k=0; k<BN; k+=MMA_K) {
                    // V is stored in shared as [K=BN, N=DIM].
                    // Use TransB=1 so WGMMA consumes it logically as [N, K].
                    int v_row = k;
                    int v_col = j * PV_MMA_N;
                    const int p_tile_outer = k / QK_MMA_N;
                    const int p_tile_inner = (k % QK_MMA_N) / 16;
                    fp16 *_VAddr = VAddr + tma_smem_offset_2d<BN>(v_row, v_col);
                    wgmma_rs<PV_MMA_N, 1, 1, 1, 0, 1, BN * 64 * sizeof(fp16), 1024, true>(
                        acc_o[1][j],
                        reinterpret_cast<uint32_t*>(acc_s_cast[p_tile_outer][p_tile_inner]),
                        _VAddr);
                }
            }
            warpgroup_commit_batch();
            warpgroup_wait();
            arrive(&Vempty[smem_i]);
        }

        // acc_o = acc_o / logsum
        #pragma unroll
        for (size_t i=0; i<BM/(MMA_M*2); i++) {
            float val0 = 1.0f / logsum[i][0];
            float val1 = 1.0f / logsum[i][1];
            #pragma unroll
            for (size_t j=0; j<DIM/PV_MMA_N; j++) {
                #pragma unroll
                for (size_t k=0; k<PV_MMA_N/16; k++) {
                    acc_o[i][j][k][0] *= val0;
                    acc_o[i][j][k][1] *= val0;
                    acc_o[i][j][k][4] *= val0;
                    acc_o[i][j][k][5] *= val0;
                    acc_o[i][j][k][2] *= val1;
                    acc_o[i][j][k][3] *= val1;
                    acc_o[i][j][k][6] *= val1;
                    acc_o[i][j][k][7] *= val1;
                }
            }
        }

        // copy acc_o to sO (load matritx)
        if (tid == 0) {
            tma_store_wait();
        }

        const int lane_row = lane_id & 0xf;
        const int lane_col = (lane_id >> 4) * 8;
        #pragma unroll
        for (size_t i=0; i<BM/(MMA_M*2); i++) {
            #pragma unroll
            for (size_t j=0; j<DIM/PV_MMA_N; j++) {
                #pragma unroll
                for (size_t k=0; k<PV_MMA_N/16; k++) {
                    int o_row = wg_idx * BM/(MMA_M*2) * MMA_M
                                + i * MMA_M
                                + warp_id_in_wg * 16
                                + lane_row;
                    int o_col = j * PV_MMA_N
                                + k * 16
                                + lane_col;
                    // fp16 *_sO = sO + tma_smem_offset_2d<BM>(o_row, o_col);
                    fp16 *_sO = sO + tma_smem_swizzle_128b_offset_2d<BM>(o_row, o_col);
                        uint32_t r0 = half2_to_u32(__floats2half2_rn(
                            acc_o[i][j][k][0],
                            acc_o[i][j][k][1]
                        ));
                        uint32_t r1 = half2_to_u32(__floats2half2_rn(
                            acc_o[i][j][k][2],
                            acc_o[i][j][k][3]
                        ));
                        uint32_t r2 = half2_to_u32(__floats2half2_rn(
                            acc_o[i][j][k][4],
                            acc_o[i][j][k][5]
                        ));
                        uint32_t r3 = half2_to_u32(__floats2half2_rn(
                            acc_o[i][j][k][6],
                            acc_o[i][j][k][7]
                        ));

                        stmatrix_x4_reg(_sO, r0, r1, r2, r3);
                }
            }
        }
        // Wait all consumer threads (2 warpgroups) before issuing TMA store from tid==0.
        fence_view_async_shared();
        bar_sync(256, 2);
        // tma store
        if (tid == 0) {
            store_async(&tensorMapO, sO, bs, hn, by * BM, 0);
            tma_store_arrive();
        }
    }
}


template<int BM, int BN, int DIM, int NUM_THREADS, int NUM_SMEM_Q, int NUM_SMEM_KV>
__global__  __launch_bounds__(NUM_THREADS) 
void attnWSKCXForNUnrollDoubleQKernel(
    int B, int H, int S, 
    const __grid_constant__ CUtensorMap tensorMapQ, 
    const __grid_constant__ CUtensorMap tensorMapK, 
    const __grid_constant__ CUtensorMap tensorMapV, 
    const __grid_constant__ CUtensorMap tensorMapO
) {
    // WS attention
    const int bs = blockIdx.z;
    const int hn = blockIdx.y;
    const int by = blockIdx.x;
    const int tid = threadIdx.x;
    uint32_t wg_idx = tid >> 7;

    // mma size
    static_assert((DIM >= 256 || DIM == 16 || DIM == 32 || DIM == 64 || DIM == 128) && "DIM ERROR!");
    static_assert((BN >= 256 || BN == 16 || BN == 32 || BN == 64 || BN == 128) && "BN ERROR!");
    constexpr int MMA_M = 64;
    constexpr int QK_MMA_N = BN <= 256 ? BN : 256;
    constexpr int PV_MMA_N = DIM <= 256 ? DIM : 256;
    constexpr int MMA_K = 16;
    constexpr int N = BM / 2 / MMA_M;
    static_assert(N >= NUM_SMEM_Q && "BN ERROR!");
    static_assert(N <= 4, "attnWSKCXForNDoubleQKernel unroll currently supports N <= 4.");

    // setting shared memory
    extern __shared__ __align__(128) uint8_t smem[];
    SMemWSDoubleQ<BM, BN, DIM, NUM_SMEM_Q, NUM_SMEM_KV> &s = 
        *reinterpret_cast<SMemWSDoubleQ<BM, BN, DIM, NUM_SMEM_Q, NUM_SMEM_KV>*>(smem);
    fp16 *sQ = s.Q, *sK = s.K, *sV = s.V, *sO = s.O;
    Barrier *Qempty = s.Qempty, *Kempty = s.Kempty, *Vempty = s.Vempty;
    Barrier *Qfull = s.Qfull, *Kfull = s.Kfull, *Vfull = s.Vfull;

    // init mbarrier
    if (threadIdx.x == 0) {
        for (int i=0; i<NUM_SMEM_Q; ++i) {
            init_barrier(&Qfull[i], 1);
            init_barrier(&Qempty[i], 256);
        }
        for (int i = 0; i < NUM_SMEM_KV; ++i) {
            init_barrier(&Kfull[i], 1);  // 1 thread arrive
            init_barrier(&Vfull[i], 1);
            init_barrier(&Kempty[i], 256);  // 256 thread arrive
            init_barrier(&Vempty[i], 256);
        }
    }
    __syncthreads();
    fence_view_async_shared();

    // TMA load
    if (wg_idx == 2) {  // producer
        warpgroup_reg_dealloc<24>();
        if (tid == 256) {
            int smem_kv_i = 0, phase_kv = 0;
            for (size_t iw=0; iw<S; iw+=BN, ++smem_kv_i) {
                if (smem_kv_i >= NUM_SMEM_KV) { smem_kv_i = 0; phase_kv ^= 1; }
                fp16 *KAddr = sK + smem_kv_i * BN * DIM;
                fp16 *VAddr = sV + smem_kv_i * BN * DIM;
                
                // load K
                wait(&Kempty[smem_kv_i], phase_kv);
                expect_bytes(&Kfull[smem_kv_i], BN * DIM * sizeof(fp16));
                load_async(KAddr, &tensorMapK, &Kfull[smem_kv_i], bs, hn, iw, 0);

                // load V
                wait(&Vempty[smem_kv_i], phase_kv);
                expect_bytes(&Vfull[smem_kv_i], BN * DIM * sizeof(fp16));
                load_async(VAddr, &tensorMapV, &Vfull[smem_kv_i], bs, hn, iw, 0);
            }
        } else if (tid == 288) {
            int smem_q_i = 0, phase_q = 0;
            for (size_t iw=0; iw<S; iw+=BN) {
                #pragma unroll
                for (int n=0; n<N; n++) {
                    if (smem_q_i >= NUM_SMEM_Q) { smem_q_i = 0; phase_q ^= 1; }
                    
                    // load Q
                    wait(&Qempty[smem_q_i], phase_q);
                    expect_bytes(&Qfull[smem_q_i], 2 * MMA_M * DIM * sizeof(fp16));
                    #pragma unroll
                    for (size_t i=0; i<2; i++) {
                        fp16 *QAddr = sQ + smem_q_i * 2 * MMA_M * DIM + i * MMA_M * DIM;
                        load_async(QAddr, &tensorMapQ, &Qfull[smem_q_i], bs, hn, by * BM + i * N * MMA_M + n * MMA_M, 0);
                    }
                    ++smem_q_i;
                }
            }
        }
    } else {  // consumer
        warpgroup_reg_alloc<240>();
        // Bootstrap empty-smem_i barriers so producer can issue the first K/V loads.
        #pragma unroll
        for (int i = 0; i < NUM_SMEM_Q; ++i) {
            arrive(&Qempty[i]);
        }
        #pragma unroll
        for (int i = 0; i < NUM_SMEM_KV; ++i) {
            arrive(&Kempty[i]);
            arrive(&Vempty[i]);
        }
        
        // need args
        uint32_t lane_id = tid & 31;
        uint32_t warp_id_in_wg = (tid >> 5) & 0x3;
        const unsigned mask = __activemask();
        const fp32 scale = sqrt((1.0f / DIM)) * 1.44269504f;  // log2(e)

        // registers define
        fp32 acc_o[N][DIM/PV_MMA_N][PV_MMA_N/16][8];
        fp32 scores_max_prev[N][2];
        fp32 scores_max[N][2];
        fp32 logsum[N][2];

        // init acc_o
        #pragma unroll
        for (size_t i=0; i<N; i++) {
            #pragma unroll
            for (size_t j=0; j<DIM/PV_MMA_N; j++) {
                #pragma unroll
                for (size_t k=0; k<PV_MMA_N/16; k++) {
                    #pragma unroll
                    for (size_t l=0; l<8; l++) {
                        acc_o[i][j][k][l] = 0.0f;
                    }
                }
            }
        }
        
        // init logsum and scores_max
        #pragma unroll
        for (size_t i=0; i<N; i++) {
            #pragma unroll
            for (size_t j=0; j<2; j++) {
                logsum[i][j] = 0.0f;
                scores_max[i][j] = -FLT_MAX;
            }
        }
        
        // main for loop
        int smem_kv_i = 0, phase_kv = 0;
        int smem_q_i = 0, phase_q = 0;
        for (size_t iw=0; iw<S; iw+=BN, ++smem_kv_i) {
            if (smem_kv_i >= NUM_SMEM_KV) { smem_kv_i = 0; phase_kv ^= 1; }
            fp16 *KAddr = sK + smem_kv_i * BN * DIM;
            fp16 *VAddr = sV + smem_kv_i * BN * DIM;

            #pragma unroll
            for (int n=0; n<N; n++, ++smem_q_i) {

            // auto compute_n = [&](auto n_const) {
            //     constexpr int n = decltype(n_const)::value;
            //     static_assert(n < N);

                if (smem_q_i >= NUM_SMEM_Q) { smem_q_i = 0; phase_q ^= 1; }
                fp16 *QAddr = sQ + smem_q_i * 2 * MMA_M * DIM;

                // fill acc_s
                fp32 acc_s[BN/QK_MMA_N][QK_MMA_N/16][8];
                #pragma unroll
                for (size_t j=0; j<BN/QK_MMA_N; j++) {
                    #pragma unroll
                    for (size_t k=0; k<QK_MMA_N/16; k++) {
                        #pragma unroll
                        for (size_t l=0; l<8; l++) {
                            acc_s[j][k][l] = 0.0f;
                        }
                    }
                }

                // gemm-qk
                wait(&Qfull[smem_q_i], phase_q);
                wait(&Kfull[smem_kv_i], phase_kv);
                warpgroup_arrive();
                #pragma unroll
                for (size_t j=0; j<BN/QK_MMA_N; j++) {
                    #pragma unroll
                    for (size_t k=0; k<DIM; k+=MMA_K) {
                        int q_row = 0;
                        int q_col = k;
                        int k_row = j * QK_MMA_N;
                        int k_col = k;
                        fp16 *_QAddr = QAddr
                            + wg_idx * MMA_M * DIM
                            + tma_smem_offset_2d<MMA_M>(q_row, q_col);
                        fp16 *_KAddr = KAddr + tma_smem_offset_2d<BN>(k_row, k_col);
                        wgmma_ss<QK_MMA_N, 1, 1, 1, 0, 0, 16, 1024, true>(acc_s[j], _QAddr, _KAddr);
                    }
                }
                warpgroup_commit_batch();
                warpgroup_wait();
                arrive(&Qempty[smem_q_i]);
                if (n == N-1) { arrive(&Kempty[smem_kv_i]); }
                

                // max_prev = max
                #pragma unroll
                for (size_t j=0; j<2; j++) {
                    scores_max_prev[n][j] = scores_max[n][j];
                }

                // reduce max
                #pragma unroll
                for (size_t j=0; j<2; j++) {
                    scores_max[n][j] = -FLT_MAX;
                }
                #pragma unroll
                for (size_t j=0; j<BN/QK_MMA_N; j++) {
                    #pragma unroll
                    for (size_t k=0; k<QK_MMA_N/16; k++) {
                        scores_max[n][0] = max(acc_s[j][k][0], scores_max[n][0]);
                        scores_max[n][0] = max(acc_s[j][k][1], scores_max[n][0]);
                        scores_max[n][0] = max(acc_s[j][k][4], scores_max[n][0]);
                        scores_max[n][0] = max(acc_s[j][k][5], scores_max[n][0]);
                        scores_max[n][1] = max(acc_s[j][k][2], scores_max[n][1]);
                        scores_max[n][1] = max(acc_s[j][k][3], scores_max[n][1]);
                        scores_max[n][1] = max(acc_s[j][k][6], scores_max[n][1]);
                        scores_max[n][1] = max(acc_s[j][k][7], scores_max[n][1]);
                    }
                }
                #pragma unroll
                for (size_t j=0; j<2; j++) {
                    #pragma unroll
                    for (size_t k=1; k<4; k*=2) {
                        scores_max[n][j] = max(scores_max[n][j], __shfl_xor_sync(mask, scores_max[n][j], k, 4));
                    }
                }

                // m = max(pm, m)
                #pragma unroll
                for (size_t j=0; j<2; j++) {
                    scores_max[n][j] = max(scores_max_prev[n][j], scores_max[n][j]);
                }

                // scores_scale = exp2(pm  - m)
                fp32 scores_scale[2];
                #pragma unroll
                for (size_t j=0; j<2; j++) {
                    scores_scale[j] = exp2f(scores_max_prev[n][j] * scale - scores_max[n][j] * scale);
                }

                // acc_s = exp2(acc_s - m)
                #pragma unroll
                for (size_t j=0; j<BN/QK_MMA_N; j++) {
                    #pragma unroll
                    for (size_t k=0; k<QK_MMA_N/16; k++) {
                        acc_s[j][k][0] = exp2f(acc_s[j][k][0] * scale - scores_max[n][0] * scale);
                        acc_s[j][k][1] = exp2f(acc_s[j][k][1] * scale - scores_max[n][0] * scale);
                        acc_s[j][k][4] = exp2f(acc_s[j][k][4] * scale - scores_max[n][0] * scale);
                        acc_s[j][k][5] = exp2f(acc_s[j][k][5] * scale - scores_max[n][0] * scale);
                        acc_s[j][k][2] = exp2f(acc_s[j][k][2] * scale - scores_max[n][1] * scale);
                        acc_s[j][k][3] = exp2f(acc_s[j][k][3] * scale - scores_max[n][1] * scale);
                        acc_s[j][k][6] = exp2f(acc_s[j][k][6] * scale - scores_max[n][1] * scale);
                        acc_s[j][k][7] = exp2f(acc_s[j][k][7] * scale - scores_max[n][1] * scale);
                    }
                }

                // reduce sum
                fp32 scores_sum[2];
                #pragma unroll
                for (size_t j=0; j<2; j++) {
                    scores_sum[j] = 0.0f;
                }
                #pragma unroll
                for (size_t j=0; j<BN/QK_MMA_N; j++) {
                    #pragma unroll
                    for (size_t k=0; k<QK_MMA_N/16; k++) {
                        scores_sum[0] += (acc_s[j][k][0] + acc_s[j][k][1] + acc_s[j][k][4] + acc_s[j][k][5]);
                        scores_sum[1] += (acc_s[j][k][2] + acc_s[j][k][3] + acc_s[j][k][6] + acc_s[j][k][7]);
                    }
                }
                #pragma unroll
                for (size_t j=0; j<2; j++) {
                    #pragma unroll
                    for (size_t k=1; k<4; k*=2) {
                        scores_sum[j] += __shfl_xor_sync(mask, scores_sum[j], k, 4);
                    }
                }

                // logsum = logsum * scores_scale + sum;
                #pragma unroll
                for (size_t j=0; j<2; j++) {
                    logsum[n][j] = logsum[n][j] * scores_scale[j] + scores_sum[j];
                }

                // cast acc_s
                fp16 acc_s_cast[BN/QK_MMA_N][QK_MMA_N/16][8];
                #pragma unroll
                for (size_t j=0; j<BN/QK_MMA_N; j++) {
                    #pragma unroll
                    for (size_t k=0; k<QK_MMA_N/16; k++) {
                        #pragma unroll
                        for (size_t l=0; l<8; l+=2) {
                            uint1 _t2;
                            float2 _t1 = *(float2*)(&acc_s[j][k][l]);
                            *(half2*)(&(_t2)) = __float22half2_rn(*(float2*)(&(_t1)));
                            *(uint1*)(&acc_s_cast[j][k][l]) = _t2;
                        }
                    }
                }

                // acc_o = acc_o * scores_scale
                #pragma unroll
                for (size_t j=0; j<DIM/PV_MMA_N; j++) {
                    #pragma unroll
                    for (size_t k=0; k<PV_MMA_N/16; k++) {
                        acc_o[n][j][k][0] *= scores_scale[0];
                        acc_o[n][j][k][1] *= scores_scale[0];
                        acc_o[n][j][k][4] *= scores_scale[0];
                        acc_o[n][j][k][5] *= scores_scale[0];
                        acc_o[n][j][k][2] *= scores_scale[1];
                        acc_o[n][j][k][3] *= scores_scale[1];
                        acc_o[n][j][k][6] *= scores_scale[1];
                        acc_o[n][j][k][7] *= scores_scale[1];
                    }
                }

                // gemm-pv
                wait(&Vfull[smem_kv_i], phase_kv);
                warpgroup_arrive();
                #pragma unroll
                for (size_t j=0; j<DIM/PV_MMA_N; j++) {
                    #pragma unroll
                    for (size_t k=0; k<BN; k+=MMA_K) {
                        // V is stored in shared as [K=BN, N=DIM].
                        // Use TransB=1 so WGMMA consumes it logically as [N, K].
                        int v_row = k;
                        int v_col = j * PV_MMA_N;
                        const int p_tile_outer = k / QK_MMA_N;
                        const int p_tile_inner = (k % QK_MMA_N) / 16;
                        fp16 *_VAddr = VAddr + tma_smem_offset_2d<BN>(v_row, v_col);
                        wgmma_rs<PV_MMA_N, 1, 1, 1, 0, 1, BN * 64 * sizeof(fp16), 1024, true>(
                            acc_o[n][j],
                            reinterpret_cast<uint32_t*>(acc_s_cast[p_tile_outer][p_tile_inner]),
                            _VAddr);
                    }
                }
                warpgroup_commit_batch();
                warpgroup_wait();
                if (n == N-1) { arrive(&Vempty[smem_kv_i]); }
            };

        }

        // acc_o = acc_o / logsum
        #pragma unroll
        for (size_t i=0; i<BM/(MMA_M*2); i++) {
            float val0 = 1.0f / logsum[i][0];
            float val1 = 1.0f / logsum[i][1];
            #pragma unroll
            for (size_t j=0; j<DIM/PV_MMA_N; j++) {
                #pragma unroll
                for (size_t k=0; k<PV_MMA_N/16; k++) {
                    acc_o[i][j][k][0] *= val0;
                    acc_o[i][j][k][1] *= val0;
                    acc_o[i][j][k][4] *= val0;
                    acc_o[i][j][k][5] *= val0;
                    acc_o[i][j][k][2] *= val1;
                    acc_o[i][j][k][3] *= val1;
                    acc_o[i][j][k][6] *= val1;
                    acc_o[i][j][k][7] *= val1;
                }
            }
        }

        // copy acc_o to sO (load matritx)
        if (tid == 0) {
            tma_store_wait();
        }

        const int lane_row = lane_id & 0xf;
        const int lane_col = (lane_id >> 4) * 8;
        #pragma unroll
        for (size_t i=0; i<BM/(MMA_M*2); i++) {
            #pragma unroll
            for (size_t j=0; j<DIM/PV_MMA_N; j++) {
                #pragma unroll
                for (size_t k=0; k<PV_MMA_N/16; k++) {
                    int o_row = wg_idx * BM/(MMA_M*2) * MMA_M
                                + i * MMA_M
                                + warp_id_in_wg * 16
                                + lane_row;
                    int o_col = j * PV_MMA_N
                                + k * 16
                                + lane_col;
                    fp16 *_sO = sO + tma_smem_swizzle_128b_offset_2d<BM>(o_row, o_col);
                        uint32_t r0 = half2_to_u32(__floats2half2_rn(
                            acc_o[i][j][k][0],
                            acc_o[i][j][k][1]
                        ));
                        uint32_t r1 = half2_to_u32(__floats2half2_rn(
                            acc_o[i][j][k][2],
                            acc_o[i][j][k][3]
                        ));
                        uint32_t r2 = half2_to_u32(__floats2half2_rn(
                            acc_o[i][j][k][4],
                            acc_o[i][j][k][5]
                        ));
                        uint32_t r3 = half2_to_u32(__floats2half2_rn(
                            acc_o[i][j][k][6],
                            acc_o[i][j][k][7]
                        ));

                        stmatrix_x4_reg(_sO, r0, r1, r2, r3);
                }
            }
        }
        // Wait all consumer threads (2 warpgroups) before issuing TMA store from tid==0.
        fence_view_async_shared();
        bar_sync(256, 2);
        // tma store
        if (tid == 0) {
            store_async(&tensorMapO, sO, bs, hn, by * BM, 0);
            tma_store_arrive();
        }
    }
}


template<int BM, int BN, int DIM, int NUM_THREADS, int NUM_SMEM_Q, int NUM_SMEM_KV>
__global__  __launch_bounds__(NUM_THREADS) 
void attnWSKCXForNUnrollDoubleQPingpongKernel(
    int B, int H, int S, 
    const __grid_constant__ CUtensorMap tensorMapQ, 
    const __grid_constant__ CUtensorMap tensorMapK, 
    const __grid_constant__ CUtensorMap tensorMapV, 
    const __grid_constant__ CUtensorMap tensorMapO
) {
    // WS attention
    const int bs = blockIdx.z;
    const int hn = blockIdx.y;
    const int by = blockIdx.x;
    const int tid = threadIdx.x;
    uint32_t wg_idx = tid >> 7;

    // mma size
    static_assert((DIM >= 256 || DIM == 16 || DIM == 32 || DIM == 64 || DIM == 128) && "DIM ERROR!");
    static_assert((BN >= 256 || BN == 16 || BN == 32 || BN == 64 || BN == 128) && "BN ERROR!");
    constexpr int MMA_M = 64;
    constexpr int QK_MMA_N = BN <= 256 ? BN : 256;
    constexpr int PV_MMA_N = DIM <= 256 ? DIM : 256;
    constexpr int MMA_K = 16;
    constexpr int N = BM / 2 / MMA_M;
    static_assert(N >= NUM_SMEM_Q && "BN ERROR!");
    static_assert(N <= 4, "attnWSKCXForNDoubleQKernel unroll currently supports N <= 4.");

    // setting shared memory
    extern __shared__ __align__(128) uint8_t smem[];
    SMemWSDoubleQ<BM, BN, DIM, NUM_SMEM_Q, NUM_SMEM_KV> &s = 
        *reinterpret_cast<SMemWSDoubleQ<BM, BN, DIM, NUM_SMEM_Q, NUM_SMEM_KV>*>(smem);
    fp16 *sQ = s.Q, *sK = s.K, *sV = s.V, *sO = s.O;
    Barrier *Qempty = s.Qempty, *Kempty = s.Kempty, *Vempty = s.Vempty;
    Barrier *Qfull = s.Qfull, *Kfull = s.Kfull, *Vfull = s.Vfull;

    // init mbarrier
    if (threadIdx.x == 0) {
        for (int i=0; i<NUM_SMEM_Q; ++i) {
            init_barrier(&Qfull[i], 1);
            init_barrier(&Qempty[i], 256);
        }
        for (int i = 0; i < NUM_SMEM_KV; ++i) {
            init_barrier(&Kfull[i], 1);  // 1 thread arrive
            init_barrier(&Vfull[i], 1);
            init_barrier(&Kempty[i], 256);  // 256 thread arrive
            init_barrier(&Vempty[i], 256);
        }
    }
    __syncthreads();
    fence_view_async_shared();

    // TMA load
    if (wg_idx == 2) {  // producer
        warpgroup_reg_dealloc<24>();
        if (tid == 256) {
            int smem_kv_i = 0, phase_kv = 0;
            for (size_t iw=0; iw<S; iw+=BN, ++smem_kv_i) {
                if (smem_kv_i >= NUM_SMEM_KV) { smem_kv_i = 0; phase_kv ^= 1; }
                fp16 *KAddr = sK + smem_kv_i * BN * DIM;
                fp16 *VAddr = sV + smem_kv_i * BN * DIM;
                
                // load K
                wait(&Kempty[smem_kv_i], phase_kv);
                expect_bytes(&Kfull[smem_kv_i], BN * DIM * sizeof(fp16));
                load_async(KAddr, &tensorMapK, &Kfull[smem_kv_i], bs, hn, iw, 0);

                // load V
                wait(&Vempty[smem_kv_i], phase_kv);
                expect_bytes(&Vfull[smem_kv_i], BN * DIM * sizeof(fp16));
                load_async(VAddr, &tensorMapV, &Vfull[smem_kv_i], bs, hn, iw, 0);
            }
        } else if (tid == 288) {
            int smem_q_i = 0, phase_q = 0;
            for (size_t iw=0; iw<S; iw+=BN) {
                auto load_q_n = [&](auto n_const) {
                    constexpr int n = decltype(n_const)::value;
                    static_assert(n < N);

                    if (smem_q_i >= NUM_SMEM_Q) { smem_q_i = 0; phase_q ^= 1; }
                    
                    // load Q
                    wait(&Qempty[smem_q_i], phase_q);
                    expect_bytes(&Qfull[smem_q_i], 2 * MMA_M * DIM * sizeof(fp16));
                    #pragma unroll
                    for (size_t i=0; i<2; i++) {
                        fp16 *QAddr = sQ + smem_q_i * 2 * MMA_M * DIM + i * MMA_M * DIM;
                        load_async(QAddr, &tensorMapQ, &Qfull[smem_q_i], bs, hn, by * BM + i * N * MMA_M + n * MMA_M, 0);
                    }
                    ++smem_q_i;
                };

                load_q_n(std::integral_constant<int, 0>{});
                if constexpr (N > 1) { load_q_n(std::integral_constant<int, 1>{}); }
                if constexpr (N > 2) { load_q_n(std::integral_constant<int, 2>{}); }
                if constexpr (N > 3) { load_q_n(std::integral_constant<int, 3>{}); }
                if constexpr (N > 3) { load_q_n(std::integral_constant<int, 4>{}); }
            }
        }
    } else {  // consumer
        warpgroup_reg_alloc<240>();
        // Bootstrap empty-smem_i barriers so producer can issue the first K/V loads.
        #pragma unroll
        for (int i = 0; i < NUM_SMEM_Q; ++i) {
            arrive(&Qempty[i]);
        }
        #pragma unroll
        for (int i = 0; i < NUM_SMEM_KV; ++i) {
            arrive(&Kempty[i]);
            arrive(&Vempty[i]);
        }
        
        // need args
        uint32_t lane_id = tid & 31;
        uint32_t warp_id_in_wg = (tid >> 5) & 0x3;
        const unsigned mask = __activemask();
        const fp32 scale = sqrt((1.0f / DIM)) * 1.44269504f;  // log2(e)

        // registers define
        fp32 acc_o[N][DIM/PV_MMA_N][PV_MMA_N/16][8];
        fp32 scores_max_prev[N][2];
        fp32 scores_max[N][2];
        fp32 logsum[N][2];

        // init acc_o
        #pragma unroll
        for (size_t i=0; i<N; i++) {
            #pragma unroll
            for (size_t j=0; j<DIM/PV_MMA_N; j++) {
                #pragma unroll
                for (size_t k=0; k<PV_MMA_N/16; k++) {
                    #pragma unroll
                    for (size_t l=0; l<8; l++) {
                        acc_o[i][j][k][l] = 0.0f;
                    }
                }
            }
        }
        
        // init logsum and scores_max
        #pragma unroll
        for (size_t i=0; i<N; i++) {
            #pragma unroll
            for (size_t j=0; j<2; j++) {
                logsum[i][j] = 0.0f;
                scores_max[i][j] = -FLT_MAX;
            }
        }
        
        // main for loop
        int smem_kv_i = 0, phase_kv = 0;
        int smem_q_i = 0, phase_q = 0;
        for (size_t iw=0; iw<S; iw+=BN, ++smem_kv_i) {
            if (smem_kv_i >= NUM_SMEM_KV) { smem_kv_i = 0; phase_kv ^= 1; }
            fp16 *KAddr = sK + smem_kv_i * BN * DIM;
            fp16 *VAddr = sV + smem_kv_i * BN * DIM;

            // define
            fp32 acc_s[BN/QK_MMA_N][QK_MMA_N/16][8];
            fp16 acc_s_cast[BN/QK_MMA_N][QK_MMA_N/16][8];
            fp32 scores_scale[2];
            fp32 scores_sum[2];
            
            // fill acc_s
            #pragma unroll
            for (size_t j=0; j<BN/QK_MMA_N; j++) {
                #pragma unroll
                for (size_t k=0; k<QK_MMA_N/16; k++) {
                    #pragma unroll
                    for (size_t l=0; l<8; l++) {
                        acc_s[j][k][l] = 0.0f;
                    }
                }
            }

            // gemm-qk - 0
            wait(&Qfull[0], 0);
            wait(&Kfull[smem_kv_i], phase_kv);
            warpgroup_arrive();
            #pragma unroll
            for (size_t j=0; j<BN/QK_MMA_N; j++) {
                #pragma unroll
                for (size_t k=0; k<DIM; k+=MMA_K) {
                    int q_row = 0;
                    int q_col = k;
                    int k_row = j * QK_MMA_N;
                    int k_col = k;
                    fp16 *_QAddr = sQ
                        + wg_idx * MMA_M * DIM
                        + tma_smem_offset_2d<MMA_M>(q_row, q_col);
                    fp16 *_KAddr = KAddr + tma_smem_offset_2d<BN>(k_row, k_col);
                    wgmma_ss<QK_MMA_N, 1, 1, 1, 0, 0, 16, 1024, true>(acc_s[j], _QAddr, _KAddr);
                }
            }
            warpgroup_commit_batch();
            warpgroup_wait();
            arrive(&Qempty[0]);

            // max_prev = max
            #pragma unroll
            for (size_t j=0; j<2; j++) {
                scores_max_prev[0][j] = scores_max[0][j];
            }
            // reduce max
            #pragma unroll
            for (size_t j=0; j<2; j++) {
                scores_max[0][j] = -FLT_MAX;
            }
            #pragma unroll
            for (size_t j=0; j<BN/QK_MMA_N; j++) {
                #pragma unroll
                for (size_t k=0; k<QK_MMA_N/16; k++) {
                    scores_max[0][0] = max(acc_s[j][k][0], scores_max[0][0]);
                    scores_max[0][0] = max(acc_s[j][k][1], scores_max[0][0]);
                    scores_max[0][0] = max(acc_s[j][k][4], scores_max[0][0]);
                    scores_max[0][0] = max(acc_s[j][k][5], scores_max[0][0]);
                    scores_max[0][1] = max(acc_s[j][k][2], scores_max[0][1]);
                    scores_max[0][1] = max(acc_s[j][k][3], scores_max[0][1]);
                    scores_max[0][1] = max(acc_s[j][k][6], scores_max[0][1]);
                    scores_max[0][1] = max(acc_s[j][k][7], scores_max[0][1]);
                }
            }
            #pragma unroll
            for (size_t j=0; j<2; j++) {
                #pragma unroll
                for (size_t k=1; k<4; k*=2) {
                    scores_max[0][j] = max(scores_max[0][j], __shfl_xor_sync(mask, scores_max[0][j], k, 4));
                }
            }
            // m = max(pm, m)
            #pragma unroll
            for (size_t j=0; j<2; j++) {
                scores_max[0][j] = max(scores_max_prev[0][j], scores_max[0][j]);
            }
            // scores_scale = exp2(pm  - m)
            #pragma unroll
            for (size_t j=0; j<2; j++) {
                scores_scale[j] = exp2f(scores_max_prev[0][j] * scale - scores_max[0][j] * scale);
            }
            // acc_s = exp2(acc_s - m)
            #pragma unroll
            for (size_t j=0; j<BN/QK_MMA_N; j++) {
                #pragma unroll
                for (size_t k=0; k<QK_MMA_N/16; k++) {
                    acc_s[j][k][0] = exp2f(acc_s[j][k][0] * scale - scores_max[0][0] * scale);
                    acc_s[j][k][1] = exp2f(acc_s[j][k][1] * scale - scores_max[0][0] * scale);
                    acc_s[j][k][4] = exp2f(acc_s[j][k][4] * scale - scores_max[0][0] * scale);
                    acc_s[j][k][5] = exp2f(acc_s[j][k][5] * scale - scores_max[0][0] * scale);
                    acc_s[j][k][2] = exp2f(acc_s[j][k][2] * scale - scores_max[0][1] * scale);
                    acc_s[j][k][3] = exp2f(acc_s[j][k][3] * scale - scores_max[0][1] * scale);
                    acc_s[j][k][6] = exp2f(acc_s[j][k][6] * scale - scores_max[0][1] * scale);
                    acc_s[j][k][7] = exp2f(acc_s[j][k][7] * scale - scores_max[0][1] * scale);
                }
            }
            // reduce sum
            #pragma unroll
            for (size_t j=0; j<2; j++) {
                scores_sum[j] = 0.0f;
            }
            #pragma unroll
            for (size_t j=0; j<BN/QK_MMA_N; j++) {
                #pragma unroll
                for (size_t k=0; k<QK_MMA_N/16; k++) {
                    scores_sum[0] += (acc_s[j][k][0] + acc_s[j][k][1] + acc_s[j][k][4] + acc_s[j][k][5]);
                    scores_sum[1] += (acc_s[j][k][2] + acc_s[j][k][3] + acc_s[j][k][6] + acc_s[j][k][7]);
                }
            }
            #pragma unroll
            for (size_t j=0; j<2; j++) {
                #pragma unroll
                for (size_t k=1; k<4; k*=2) {
                    scores_sum[j] += __shfl_xor_sync(mask, scores_sum[j], k, 4);
                }
            }
            // cast acc_s
            #pragma unroll
            for (size_t j=0; j<BN/QK_MMA_N; j++) {
                #pragma unroll
                for (size_t k=0; k<QK_MMA_N/16; k++) {
                    #pragma unroll
                    for (size_t l=0; l<8; l+=2) {
                        uint1 _t2;
                        float2 _t1 = *(float2*)(&acc_s[j][k][l]);
                        *(half2*)(&(_t2)) = __float22half2_rn(*(float2*)(&(_t1)));
                        *(uint1*)(&acc_s_cast[j][k][l]) = _t2;
                    }
                }
            }
            // logsum = logsum * scores_scale + sum;
            #pragma unroll
            for (size_t j=0; j<2; j++) {
                logsum[0][j] = logsum[0][j] * scores_scale[j] + scores_sum[j];
            }
            // acc_o = acc_o * scores_scale
            #pragma unroll
            for (size_t j=0; j<DIM/PV_MMA_N; j++) {
                #pragma unroll
                for (size_t k=0; k<PV_MMA_N/16; k++) {
                    acc_o[0][j][k][0] *= scores_scale[0];
                    acc_o[0][j][k][1] *= scores_scale[0];
                    acc_o[0][j][k][4] *= scores_scale[0];
                    acc_o[0][j][k][5] *= scores_scale[0];
                    acc_o[0][j][k][2] *= scores_scale[1];
                    acc_o[0][j][k][3] *= scores_scale[1];
                    acc_o[0][j][k][6] *= scores_scale[1];
                    acc_o[0][j][k][7] *= scores_scale[1];
                }
            }
           
            // gemm-pv - 0
            wait(&Vfull[smem_kv_i], phase_kv);
            warpgroup_arrive();
            #pragma unroll
            for (size_t j=0; j<DIM/PV_MMA_N; j++) {
                #pragma unroll
                for (size_t k=0; k<BN; k+=MMA_K) {
                    // V is stored in shared as [K=BN, N=DIM].
                    // Use TransB=1 so WGMMA consumes it logically as [N, K].
                    int v_row = k;
                    int v_col = j * PV_MMA_N;
                    const int p_tile_outer = k / QK_MMA_N;
                    const int p_tile_inner = (k % QK_MMA_N) / 16;
                    fp16 *_VAddr = VAddr + tma_smem_offset_2d<BN>(v_row, v_col);
                    wgmma_rs<PV_MMA_N, 1, 1, 1, 0, 1, BN * 64 * sizeof(fp16), 1024, true>(
                        acc_o[0][j],
                        reinterpret_cast<uint32_t*>(acc_s_cast[p_tile_outer][p_tile_inner]),
                        _VAddr);
                }
            }
            warpgroup_commit_batch();
            warpgroup_wait();

            // gemm-qk - 1
            int next_smem_q_i = 0, next_phase_q = 1;
            if (NUM_SMEM_Q == 2) { next_smem_q_i = 1; next_phase_q = 0; }
            fp16 *QAddr = sQ + next_smem_q_i * 2 * MMA_M * DIM;
            wait(&Qfull[next_smem_q_i], next_phase_q);
            warpgroup_arrive();
            #pragma unroll
            for (size_t j=0; j<BN/QK_MMA_N; j++) {
                #pragma unroll
                for (size_t k=0; k<DIM; k+=MMA_K) {
                    int q_row = 0;
                    int q_col = k;
                    int k_row = j * QK_MMA_N;
                    int k_col = k;
                    fp16 *_QAddr = QAddr
                        + wg_idx * MMA_M * DIM
                        + tma_smem_offset_2d<MMA_M>(q_row, q_col);
                    fp16 *_KAddr = KAddr + tma_smem_offset_2d<BN>(k_row, k_col);
                    wgmma_ss<QK_MMA_N, 1, 1, 1, 0, 0, 16, 1024, true>(acc_s[j], _QAddr, _KAddr);
                }
            }
            warpgroup_commit_batch();
            warpgroup_wait();
            arrive(&Qempty[next_smem_q_i]);
            arrive(&Kempty[smem_kv_i]);

            // max_prev = max
            #pragma unroll
            for (size_t j=0; j<2; j++) {
                scores_max_prev[1][j] = scores_max[1][j];
            }
            // reduce max
            #pragma unroll
            for (size_t j=0; j<2; j++) {
                scores_max[1][j] = -FLT_MAX;
            }
            #pragma unroll
            for (size_t j=0; j<BN/QK_MMA_N; j++) {
                #pragma unroll
                for (size_t k=0; k<QK_MMA_N/16; k++) {
                    scores_max[1][0] = max(acc_s[j][k][0], scores_max[1][0]);
                    scores_max[1][0] = max(acc_s[j][k][1], scores_max[1][0]);
                    scores_max[1][0] = max(acc_s[j][k][4], scores_max[1][0]);
                    scores_max[1][0] = max(acc_s[j][k][5], scores_max[1][0]);
                    scores_max[1][1] = max(acc_s[j][k][2], scores_max[1][1]);
                    scores_max[1][1] = max(acc_s[j][k][3], scores_max[1][1]);
                    scores_max[1][1] = max(acc_s[j][k][6], scores_max[1][1]);
                    scores_max[1][1] = max(acc_s[j][k][7], scores_max[1][1]);
                }
            }
            #pragma unroll
            for (size_t j=0; j<2; j++) {
                #pragma unroll
                for (size_t k=1; k<4; k*=2) {
                    scores_max[1][j] = max(scores_max[1][j], __shfl_xor_sync(mask, scores_max[1][j], k, 4));
                }
            }
            // m = max(pm, m)
            #pragma unroll
            for (size_t j=0; j<2; j++) {
                scores_max[1][j] = max(scores_max_prev[1][j], scores_max[1][j]);
            }
            // scores_scale = exp2(pm  - m)
            #pragma unroll
            for (size_t j=0; j<2; j++) {
                scores_scale[j] = exp2f(scores_max_prev[1][j] * scale - scores_max[1][j] * scale);
            }
            // acc_s = exp2(acc_s - m)
            #pragma unroll
            for (size_t j=0; j<BN/QK_MMA_N; j++) {
                #pragma unroll
                for (size_t k=0; k<QK_MMA_N/16; k++) {
                    acc_s[j][k][0] = exp2f(acc_s[j][k][0] * scale - scores_max[1][0] * scale);
                    acc_s[j][k][1] = exp2f(acc_s[j][k][1] * scale - scores_max[1][0] * scale);
                    acc_s[j][k][4] = exp2f(acc_s[j][k][4] * scale - scores_max[1][0] * scale);
                    acc_s[j][k][5] = exp2f(acc_s[j][k][5] * scale - scores_max[1][0] * scale);
                    acc_s[j][k][2] = exp2f(acc_s[j][k][2] * scale - scores_max[1][1] * scale);
                    acc_s[j][k][3] = exp2f(acc_s[j][k][3] * scale - scores_max[1][1] * scale);
                    acc_s[j][k][6] = exp2f(acc_s[j][k][6] * scale - scores_max[1][1] * scale);
                    acc_s[j][k][7] = exp2f(acc_s[j][k][7] * scale - scores_max[1][1] * scale);
                }
            }
            // reduce sum
            #pragma unroll
            for (size_t j=0; j<2; j++) {
                scores_sum[j] = 0.0f;
            }
            #pragma unroll
            for (size_t j=0; j<BN/QK_MMA_N; j++) {
                #pragma unroll
                for (size_t k=0; k<QK_MMA_N/16; k++) {
                    scores_sum[0] += (acc_s[j][k][0] + acc_s[j][k][1] + acc_s[j][k][4] + acc_s[j][k][5]);
                    scores_sum[1] += (acc_s[j][k][2] + acc_s[j][k][3] + acc_s[j][k][6] + acc_s[j][k][7]);
                }
            }
            #pragma unroll
            for (size_t j=0; j<2; j++) {
                #pragma unroll
                for (size_t k=1; k<4; k*=2) {
                    scores_sum[j] += __shfl_xor_sync(mask, scores_sum[j], k, 4);
                }
            }           
            // cast acc_s
            #pragma unroll
            for (size_t j=0; j<BN/QK_MMA_N; j++) {
                #pragma unroll
                for (size_t k=0; k<QK_MMA_N/16; k++) {
                    #pragma unroll
                    for (size_t l=0; l<8; l+=2) {
                        uint1 _t2;
                        float2 _t1 = *(float2*)(&acc_s[j][k][l]);
                        *(half2*)(&(_t2)) = __float22half2_rn(*(float2*)(&(_t1)));
                        *(uint1*)(&acc_s_cast[j][k][l]) = _t2;
                    }
                }
            }
            // logsum = logsum * scores_scale + sum;
            #pragma unroll
            for (size_t j=0; j<2; j++) {
                logsum[1][j] = logsum[1][j] * scores_scale[j] + scores_sum[j];
            }
            // acc_o = acc_o * scores_scale
            #pragma unroll
            for (size_t j=0; j<DIM/PV_MMA_N; j++) {
                #pragma unroll
                for (size_t k=0; k<PV_MMA_N/16; k++) {
                    acc_o[1][j][k][0] *= scores_scale[0];
                    acc_o[1][j][k][1] *= scores_scale[0];
                    acc_o[1][j][k][4] *= scores_scale[0];
                    acc_o[1][j][k][5] *= scores_scale[0];
                    acc_o[1][j][k][2] *= scores_scale[1];
                    acc_o[1][j][k][3] *= scores_scale[1];
                    acc_o[1][j][k][6] *= scores_scale[1];
                    acc_o[1][j][k][7] *= scores_scale[1];
                }
            }

            // gemm-pv - 1
            wait(&Vfull[smem_kv_i], phase_kv);
            warpgroup_arrive();
            #pragma unroll
            for (size_t j=0; j<DIM/PV_MMA_N; j++) {
                #pragma unroll
                for (size_t k=0; k<BN; k+=MMA_K) {
                    // V is stored in shared as [K=BN, N=DIM].
                    // Use TransB=1 so WGMMA consumes it logically as [N, K].
                    int v_row = k;
                    int v_col = j * PV_MMA_N;
                    const int p_tile_outer = k / QK_MMA_N;
                    const int p_tile_inner = (k % QK_MMA_N) / 16;
                    fp16 *_VAddr = VAddr + tma_smem_offset_2d<BN>(v_row, v_col);
                    wgmma_rs<PV_MMA_N, 1, 1, 1, 0, 1, BN * 64 * sizeof(fp16), 1024, true>(
                        acc_o[1][j],
                        reinterpret_cast<uint32_t*>(acc_s_cast[p_tile_outer][p_tile_inner]),
                        _VAddr);
                }
            }
            warpgroup_commit_batch();
            warpgroup_wait();
            arrive(&Vempty[smem_kv_i]);
        }

        // acc_o = acc_o / logsum
        #pragma unroll
        for (size_t i=0; i<BM/(MMA_M*2); i++) {
            float val0 = 1.0f / logsum[i][0];
            float val1 = 1.0f / logsum[i][1];
            #pragma unroll
            for (size_t j=0; j<DIM/PV_MMA_N; j++) {
                #pragma unroll
                for (size_t k=0; k<PV_MMA_N/16; k++) {
                    acc_o[i][j][k][0] *= val0;
                    acc_o[i][j][k][1] *= val0;
                    acc_o[i][j][k][4] *= val0;
                    acc_o[i][j][k][5] *= val0;
                    acc_o[i][j][k][2] *= val1;
                    acc_o[i][j][k][3] *= val1;
                    acc_o[i][j][k][6] *= val1;
                    acc_o[i][j][k][7] *= val1;
                }
            }
        }

        // copy acc_o to sO (load matritx)
        if (tid == 0) {
            tma_store_wait();
        }

        const int lane_row = lane_id & 0xf;
        const int lane_col = (lane_id >> 4) * 8;
        #pragma unroll
        for (size_t i=0; i<BM/(MMA_M*2); i++) {
            #pragma unroll
            for (size_t j=0; j<DIM/PV_MMA_N; j++) {
                #pragma unroll
                for (size_t k=0; k<PV_MMA_N/16; k++) {
                    int o_row = wg_idx * BM/(MMA_M*2) * MMA_M
                                + i * MMA_M
                                + warp_id_in_wg * 16
                                + lane_row;
                    int o_col = j * PV_MMA_N
                                + k * 16
                                + lane_col;
                    fp16 *_sO = sO + tma_smem_swizzle_128b_offset_2d<BM>(o_row, o_col);
                        uint32_t r0 = half2_to_u32(__floats2half2_rn(
                            acc_o[i][j][k][0],
                            acc_o[i][j][k][1]
                        ));
                        uint32_t r1 = half2_to_u32(__floats2half2_rn(
                            acc_o[i][j][k][2],
                            acc_o[i][j][k][3]
                        ));
                        uint32_t r2 = half2_to_u32(__floats2half2_rn(
                            acc_o[i][j][k][4],
                            acc_o[i][j][k][5]
                        ));
                        uint32_t r3 = half2_to_u32(__floats2half2_rn(
                            acc_o[i][j][k][6],
                            acc_o[i][j][k][7]
                        ));

                        stmatrix_x4_reg(_sO, r0, r1, r2, r3);
                }
            }
        }
        // Wait all consumer threads (2 warpgroups) before issuing TMA store from tid==0.
        fence_view_async_shared();
        bar_sync(256, 2);
        // tma store
        if (tid == 0) {
            store_async(&tensorMapO, sO, bs, hn, by * BM, 0);
            tma_store_arrive();
        }
    }
}


template<int BM, int BN, int DIM, int NUM_THREADS, int NUM_SMEM_Q, int NUM_SMEM_KV>
__global__  __launch_bounds__(NUM_THREADS) 
void attnWSKCXForNUnrollDoubleQ2StageKernel(
    int B, int H, int S, 
    const __grid_constant__ CUtensorMap tensorMapQ, 
    const __grid_constant__ CUtensorMap tensorMapK, 
    const __grid_constant__ CUtensorMap tensorMapV, 
    const __grid_constant__ CUtensorMap tensorMapO
) {
    // WS attention
    const int bs = blockIdx.z;
    const int hn = blockIdx.y;
    const int by = blockIdx.x;
    const int tid = threadIdx.x;
    uint32_t wg_idx = tid >> 7;

    // mma size
    static_assert((DIM >= 256 || DIM == 16 || DIM == 32 || DIM == 64 || DIM == 128) && "DIM ERROR!");
    static_assert((BN >= 256 || BN == 16 || BN == 32 || BN == 64 || BN == 128) && "BN ERROR!");
    constexpr int MMA_M = 64;
    constexpr int QK_MMA_N = BN <= 256 ? BN : 256;
    constexpr int PV_MMA_N = DIM <= 256 ? DIM : 256;
    constexpr int MMA_K = 16;
    constexpr int N = BM / 2 / MMA_M;
    static_assert(N >= NUM_SMEM_Q && "BN ERROR!");
    static_assert(N <= 4, "attnWSKCXForNDoubleQKernel unroll currently supports N <= 4.");

    // setting shared memory
    extern __shared__ __align__(128) uint8_t smem[];
    SMemWSDoubleQ<BM, BN, DIM, NUM_SMEM_Q, NUM_SMEM_KV> &s = 
        *reinterpret_cast<SMemWSDoubleQ<BM, BN, DIM, NUM_SMEM_Q, NUM_SMEM_KV>*>(smem);
    fp16 *sQ = s.Q, *sK = s.K, *sV = s.V, *sO = s.O;
    Barrier *Qempty = s.Qempty, *Kempty = s.Kempty, *Vempty = s.Vempty;
    Barrier *Qfull = s.Qfull, *Kfull = s.Kfull, *Vfull = s.Vfull;

    // init mbarrier
    if (threadIdx.x == 0) {
        for (int i=0; i<NUM_SMEM_Q; ++i) {
            init_barrier(&Qfull[i], 1);
            init_barrier(&Qempty[i], 256);
        }
        for (int i = 0; i < NUM_SMEM_KV; ++i) {
            init_barrier(&Kfull[i], 1);  // 1 thread arrive
            init_barrier(&Vfull[i], 1);
            init_barrier(&Kempty[i], 256);  // 256 thread arrive
            init_barrier(&Vempty[i], 256);
        }
    }
    __syncthreads();
    fence_view_async_shared();

    // TMA load
    if (wg_idx == 2) {  // producer
        warpgroup_reg_dealloc<24>();
        if (tid == 256) {
            int smem_kv_i = 0, phase_kv = 0;
            for (size_t iw=0; iw<S; iw+=BN, ++smem_kv_i) {
                if (smem_kv_i >= NUM_SMEM_KV) { smem_kv_i = 0; phase_kv ^= 1; }
                fp16 *KAddr = sK + smem_kv_i * BN * DIM;
                fp16 *VAddr = sV + smem_kv_i * BN * DIM;
                
                // load K
                wait(&Kempty[smem_kv_i], phase_kv);
                expect_bytes(&Kfull[smem_kv_i], BN * DIM * sizeof(fp16));
                load_async(KAddr, &tensorMapK, &Kfull[smem_kv_i], bs, hn, iw, 0);

                // load V
                wait(&Vempty[smem_kv_i], phase_kv);
                expect_bytes(&Vfull[smem_kv_i], BN * DIM * sizeof(fp16));
                load_async(VAddr, &tensorMapV, &Vfull[smem_kv_i], bs, hn, iw, 0);
            }
        } else if (tid == 288) {
            int smem_q_i = 0, phase_q = 0;
            for (size_t iw=0; iw<S; iw+=BN) {
                auto load_q_n = [&](auto n_const) {
                    constexpr int n = decltype(n_const)::value;
                    static_assert(n < N);

                    if (smem_q_i >= NUM_SMEM_Q) { smem_q_i = 0; phase_q ^= 1; }
                    
                    // load Q
                    wait(&Qempty[smem_q_i], phase_q);
                    expect_bytes(&Qfull[smem_q_i], 2 * MMA_M * DIM * sizeof(fp16));
                    #pragma unroll
                    for (size_t i=0; i<2; i++) {
                        fp16 *QAddr = sQ + smem_q_i * 2 * MMA_M * DIM + i * MMA_M * DIM;
                        load_async(QAddr, &tensorMapQ, &Qfull[smem_q_i], bs, hn, by * BM + i * N * MMA_M + n * MMA_M, 0);
                    }
                    ++smem_q_i;
                };

                load_q_n(std::integral_constant<int, 0>{});
                if constexpr (N > 1) { load_q_n(std::integral_constant<int, 1>{}); }
                if constexpr (N > 2) { load_q_n(std::integral_constant<int, 2>{}); }
                if constexpr (N > 3) { load_q_n(std::integral_constant<int, 3>{}); }
                if constexpr (N > 4) { load_q_n(std::integral_constant<int, 4>{}); }
            }
        }
    } else {  // consumer
        warpgroup_reg_alloc<240>();
        // Bootstrap empty-smem_i barriers so producer can issue the first K/V loads.
        #pragma unroll
        for (int i = 0; i < NUM_SMEM_Q; ++i) {
            arrive(&Qempty[i]);
        }
        #pragma unroll
        for (int i = 0; i < NUM_SMEM_KV; ++i) {
            arrive(&Kempty[i]);
            arrive(&Vempty[i]);
        }
        
        // need args
        uint32_t lane_id = tid & 31;
        uint32_t warp_id_in_wg = (tid >> 5) & 0x3;
        const unsigned mask = __activemask();
        const fp32 scale = sqrt((1.0f / DIM)) * 1.44269504f;  // log2(e)

        // registers define
        fp32 acc_o[N][DIM/PV_MMA_N][PV_MMA_N/16][8];
        fp32 scores_max_prev[N][2];
        fp32 scores_max[N][2];
        fp32 logsum[N][2];

        // init acc_o
        #pragma unroll
        for (size_t i=0; i<N; i++) {
            #pragma unroll
            for (size_t j=0; j<DIM/PV_MMA_N; j++) {
                #pragma unroll
                for (size_t k=0; k<PV_MMA_N/16; k++) {
                    #pragma unroll
                    for (size_t l=0; l<8; l++) {
                        acc_o[i][j][k][l] = 0.0f;
                    }
                }
            }
        }
        
        // init logsum and scores_max
        #pragma unroll
        for (size_t i=0; i<N; i++) {
            #pragma unroll
            for (size_t j=0; j<2; j++) {
                logsum[i][j] = 0.0f;
                scores_max[i][j] = -FLT_MAX;
            }
        }
        
        fp32 acc_s[BN/QK_MMA_N][QK_MMA_N/16][8];
        fp16 acc_s_cast[BN/QK_MMA_N][QK_MMA_N/16][8];
        fp32 scores_scale[2];
        fp32 scores_sum[2];

        auto compute_qk = [&](size_t qk_step) {
            const size_t kv_tile = qk_step / N;
            const int n = qk_step - kv_tile * N;
            const int smem_q_i = qk_step % NUM_SMEM_Q;
            const int phase_q = (qk_step / NUM_SMEM_Q) & 1;
            const int smem_k_i = kv_tile % NUM_SMEM_KV;
            const int phase_k = (kv_tile / NUM_SMEM_KV) & 1;
            fp16 *QAddr = sQ + smem_q_i * 2 * MMA_M * DIM;
            fp16 *KAddr = sK + smem_k_i * BN * DIM;

            #pragma unroll
            for (size_t j=0; j<BN/QK_MMA_N; j++) {
                #pragma unroll
                for (size_t k=0; k<QK_MMA_N/16; k++) {
                    #pragma unroll
                    for (size_t l=0; l<8; l++) {
                        acc_s[j][k][l] = 0.0f;
                    }
                }
            }

            wait(&Qfull[smem_q_i], phase_q);
            wait(&Kfull[smem_k_i], phase_k);
            warpgroup_arrive();
            #pragma unroll
            for (size_t j=0; j<BN/QK_MMA_N; j++) {
                #pragma unroll
                for (size_t k=0; k<DIM; k+=MMA_K) {
                    int q_row = 0;
                    int q_col = k;
                    int k_row = j * QK_MMA_N;
                    int k_col = k;
                    fp16 *_QAddr = QAddr
                        + wg_idx * MMA_M * DIM
                        + tma_smem_offset_2d<MMA_M>(q_row, q_col);
                    fp16 *_KAddr = KAddr + tma_smem_offset_2d<BN>(k_row, k_col);
                    wgmma_ss<QK_MMA_N, 1, 1, 1, 0, 0, 16, 1024, true>(acc_s[j], _QAddr, _KAddr);
                }
            }
            warpgroup_commit_batch();
            warpgroup_wait();
            arrive(&Qempty[smem_q_i]);
            if (n == N-1) {
                arrive(&Kempty[smem_k_i]);
            }
        };

        auto compute_pv = [&](size_t pv_step) {
            const size_t kv_tile = pv_step / N;
            const int n = pv_step - kv_tile * N;
            const int smem_v_i = kv_tile % NUM_SMEM_KV;
            const int phase_v = (kv_tile / NUM_SMEM_KV) & 1;
            fp16 *VAddr = sV + smem_v_i * BN * DIM;

            wait(&Vfull[smem_v_i], phase_v);
            warpgroup_arrive();
            #pragma unroll
            for (size_t j=0; j<DIM/PV_MMA_N; j++) {
                #pragma unroll
                for (size_t k=0; k<BN; k+=MMA_K) {
                    // V is stored in shared as [K=BN, N=DIM].
                    // Use TransB=1 so WGMMA consumes it logically as [N, K].
                    int v_row = k;
                    int v_col = j * PV_MMA_N;
                    const int p_tile_outer = k / QK_MMA_N;
                    const int p_tile_inner = (k % QK_MMA_N) / 16;
                    fp16 *_VAddr = VAddr + tma_smem_offset_2d<BN>(v_row, v_col);
                    wgmma_rs<PV_MMA_N, 1, 1, 1, 0, 1, BN * 64 * sizeof(fp16), 1024, true>(
                        acc_o[n][j],
                        reinterpret_cast<uint32_t*>(acc_s_cast[p_tile_outer][p_tile_inner]),
                        _VAddr);
                }
            }
            warpgroup_commit_batch();
            warpgroup_wait();
            if (n == N-1) {
                arrive(&Vempty[smem_v_i]);
            }
        };

        auto compute_softmax = [&](int n) {
            // max_prev = max
            #pragma unroll
            for (size_t j=0; j<2; j++) {
                scores_max_prev[n][j] = scores_max[n][j];
            }
            // reduce max
            #pragma unroll
            for (size_t j=0; j<2; j++) {
                scores_max[n][j] = -FLT_MAX;
            }
            #pragma unroll
            for (size_t j=0; j<BN/QK_MMA_N; j++) {
                #pragma unroll
                for (size_t k=0; k<QK_MMA_N/16; k++) {
                    scores_max[n][0] = max(acc_s[j][k][0], scores_max[n][0]);
                    scores_max[n][0] = max(acc_s[j][k][1], scores_max[n][0]);
                    scores_max[n][0] = max(acc_s[j][k][4], scores_max[n][0]);
                    scores_max[n][0] = max(acc_s[j][k][5], scores_max[n][0]);
                    scores_max[n][1] = max(acc_s[j][k][2], scores_max[n][1]);
                    scores_max[n][1] = max(acc_s[j][k][3], scores_max[n][1]);
                    scores_max[n][1] = max(acc_s[j][k][6], scores_max[n][1]);
                    scores_max[n][1] = max(acc_s[j][k][7], scores_max[n][1]);
                }
            }
            #pragma unroll
            for (size_t j=0; j<2; j++) {
                #pragma unroll
                for (size_t k=1; k<4; k*=2) {
                    scores_max[n][j] = max(scores_max[n][j], __shfl_xor_sync(mask, scores_max[n][j], k, 4));
                }
            }
            // m = max(pm, m)
            #pragma unroll
            for (size_t j=0; j<2; j++) {
                scores_max[n][j] = max(scores_max_prev[n][j], scores_max[n][j]);
            }
            // scores_scale = exp2(pm  - m)
            #pragma unroll
            for (size_t j=0; j<2; j++) {
                scores_scale[j] = exp2f(scores_max_prev[n][j] * scale - scores_max[n][j] * scale);
            }
            // acc_s = exp2(acc_s - m)
            #pragma unroll
            for (size_t j=0; j<BN/QK_MMA_N; j++) {
                #pragma unroll
                for (size_t k=0; k<QK_MMA_N/16; k++) {
                    acc_s[j][k][0] = exp2f(acc_s[j][k][0] * scale - scores_max[n][0] * scale);
                    acc_s[j][k][1] = exp2f(acc_s[j][k][1] * scale - scores_max[n][0] * scale);
                    acc_s[j][k][4] = exp2f(acc_s[j][k][4] * scale - scores_max[n][0] * scale);
                    acc_s[j][k][5] = exp2f(acc_s[j][k][5] * scale - scores_max[n][0] * scale);
                    acc_s[j][k][2] = exp2f(acc_s[j][k][2] * scale - scores_max[n][1] * scale);
                    acc_s[j][k][3] = exp2f(acc_s[j][k][3] * scale - scores_max[n][1] * scale);
                    acc_s[j][k][6] = exp2f(acc_s[j][k][6] * scale - scores_max[n][1] * scale);
                    acc_s[j][k][7] = exp2f(acc_s[j][k][7] * scale - scores_max[n][1] * scale);
                }
            }
            // reduce sum
            #pragma unroll
            for (size_t j=0; j<2; j++) {
                scores_sum[j] = 0.0f;
            }
            #pragma unroll
            for (size_t j=0; j<BN/QK_MMA_N; j++) {
                #pragma unroll
                for (size_t k=0; k<QK_MMA_N/16; k++) {
                    scores_sum[0] += (acc_s[j][k][0] + acc_s[j][k][1] + acc_s[j][k][4] + acc_s[j][k][5]);
                    scores_sum[1] += (acc_s[j][k][2] + acc_s[j][k][3] + acc_s[j][k][6] + acc_s[j][k][7]);
                }
            }
            #pragma unroll
            for (size_t j=0; j<2; j++) {
                #pragma unroll
                for (size_t k=1; k<4; k*=2) {
                    scores_sum[j] += __shfl_xor_sync(mask, scores_sum[j], k, 4);
                }
            }
            // logsum = logsum * scores_scale + sum;
            #pragma unroll
            for (size_t j=0; j<2; j++) {
                logsum[n][j] = logsum[n][j] * scores_scale[j] + scores_sum[j];
            }
            // cast acc_s
            #pragma unroll
            for (size_t j=0; j<BN/QK_MMA_N; j++) {
                #pragma unroll
                for (size_t k=0; k<QK_MMA_N/16; k++) {
                    #pragma unroll
                    for (size_t l=0; l<8; l+=2) {
                        uint1 _t2;
                        float2 _t1 = *(float2*)(&acc_s[j][k][l]);
                        *(half2*)(&(_t2)) = __float22half2_rn(*(float2*)(&(_t1)));
                        *(uint1*)(&acc_s_cast[j][k][l]) = _t2;
                    }
                }
            }
            // acc_o = acc_o * scores_scale
            #pragma unroll
            for (size_t j=0; j<DIM/PV_MMA_N; j++) {
                #pragma unroll
                for (size_t k=0; k<PV_MMA_N/16; k++) {
                    acc_o[n][j][k][0] *= scores_scale[0];
                    acc_o[n][j][k][1] *= scores_scale[0];
                    acc_o[n][j][k][4] *= scores_scale[0];
                    acc_o[n][j][k][5] *= scores_scale[0];
                    acc_o[n][j][k][2] *= scores_scale[1];
                    acc_o[n][j][k][3] *= scores_scale[1];
                    acc_o[n][j][k][6] *= scores_scale[1];
                    acc_o[n][j][k][7] *= scores_scale[1];
                }
            }
        };

        const size_t total_qk_steps = (S / BN) * N;
        for (size_t qk_step=0; qk_step<=total_qk_steps; ++qk_step) {
            if (qk_step < total_qk_steps) {
                compute_qk(qk_step);
            }
            if (qk_step > 0) {
                compute_pv(qk_step - 1);
            }
            if (qk_step < total_qk_steps) {
                const size_t kv_tile = qk_step / N;
                compute_softmax(qk_step - kv_tile * N);
            }
        }

#if 0
        size_t smem_k_i = 0, phase_k = 0;
        size_t smem_q_i = 0, phase_q = 0;
        size_t smem_v_i = 0, phase_v = 0;

        // ==== prologue ====
        // fill acc_s
        fp32 acc_s[BN/QK_MMA_N][QK_MMA_N/16][8];
        #pragma unroll
        for (size_t j=0; j<BN/QK_MMA_N; j++) {
            #pragma unroll
            for (size_t k=0; k<QK_MMA_N/16; k++) {
                #pragma unroll
                for (size_t l=0; l<8; l++) {
                    acc_s[j][k][l] = 0.0f;
                }
            }
        }
        // gemm-qk
        wait(&Qfull[smem_q_i], phase_q);
        wait(&Kfull[smem_k_i], phase_k);
        warpgroup_arrive();
        #pragma unroll
        for (size_t j=0; j<BN/QK_MMA_N; j++) {
            #pragma unroll
            for (size_t k=0; k<DIM; k+=MMA_K) {
                int q_row = 0;
                int q_col = k;
                int k_row = j * QK_MMA_N;
                int k_col = k;
                fp16 *_QAddr = sQ
                    + wg_idx * MMA_M * DIM
                    + tma_smem_offset_2d<MMA_M>(q_row, q_col);
                fp16 *_KAddr = sK + tma_smem_offset_2d<BN>(k_row, k_col);
                wgmma_ss<QK_MMA_N, 1, 1, 1, 0, 0, 16, 1024, true>(acc_s[j], _QAddr, _KAddr);
            }
        }
        warpgroup_commit_batch();
        warpgroup_wait();
        arrive(&Qempty[smem_q_i]);
        
        // max_prev = max
        #pragma unroll
        for (size_t j=0; j<2; j++) {
            scores_max_prev[0][j] = scores_max[0][j];
        }
        // reduce max
        #pragma unroll
        for (size_t j=0; j<2; j++) {
            scores_max[0][j] = -FLT_MAX;
        }
        #pragma unroll
        for (size_t j=0; j<BN/QK_MMA_N; j++) {
            #pragma unroll
            for (size_t k=0; k<QK_MMA_N/16; k++) {
                scores_max[0][0] = max(acc_s[j][k][0], scores_max[0][0]);
                scores_max[0][0] = max(acc_s[j][k][1], scores_max[0][0]);
                scores_max[0][0] = max(acc_s[j][k][4], scores_max[0][0]);
                scores_max[0][0] = max(acc_s[j][k][5], scores_max[0][0]);
                scores_max[0][1] = max(acc_s[j][k][2], scores_max[0][1]);
                scores_max[0][1] = max(acc_s[j][k][3], scores_max[0][1]);
                scores_max[0][1] = max(acc_s[j][k][6], scores_max[0][1]);
                scores_max[0][1] = max(acc_s[j][k][7], scores_max[0][1]);
            }
        }
        #pragma unroll
        for (size_t j=0; j<2; j++) {
            #pragma unroll
            for (size_t k=1; k<4; k*=2) {
                scores_max[0][j] = max(scores_max[0][j], __shfl_xor_sync(mask, scores_max[0][j], k, 4));
            }
        }
        // m = max(pm, m)
        #pragma unroll
        for (size_t j=0; j<2; j++) {
            scores_max[0][j] = max(scores_max_prev[0][j], scores_max[0][j]);
        }
        // scores_scale = exp2(pm  - m)
        fp32 scores_scale[2];
        #pragma unroll
        for (size_t j=0; j<2; j++) {
            scores_scale[j] = exp2f(scores_max_prev[0][j] * scale - scores_max[0][j] * scale);
        }
        // acc_s = exp2(acc_s - m)
        #pragma unroll
        for (size_t j=0; j<BN/QK_MMA_N; j++) {
            #pragma unroll
            for (size_t k=0; k<QK_MMA_N/16; k++) {
                acc_s[j][k][0] = exp2f(acc_s[j][k][0] * scale - scores_max[0][0] * scale);
                acc_s[j][k][1] = exp2f(acc_s[j][k][1] * scale - scores_max[0][0] * scale);
                acc_s[j][k][4] = exp2f(acc_s[j][k][4] * scale - scores_max[0][0] * scale);
                acc_s[j][k][5] = exp2f(acc_s[j][k][5] * scale - scores_max[0][0] * scale);
                acc_s[j][k][2] = exp2f(acc_s[j][k][2] * scale - scores_max[0][1] * scale);
                acc_s[j][k][3] = exp2f(acc_s[j][k][3] * scale - scores_max[0][1] * scale);
                acc_s[j][k][6] = exp2f(acc_s[j][k][6] * scale - scores_max[0][1] * scale);
                acc_s[j][k][7] = exp2f(acc_s[j][k][7] * scale - scores_max[0][1] * scale);
            }
        }
        // reduce sum
        fp32 scores_sum[2];
        #pragma unroll
        for (size_t j=0; j<2; j++) {
            scores_sum[j] = 0.0f;
        }
        #pragma unroll
        for (size_t j=0; j<BN/QK_MMA_N; j++) {
            #pragma unroll
            for (size_t k=0; k<QK_MMA_N/16; k++) {
                scores_sum[0] += (acc_s[j][k][0] + acc_s[j][k][1] + acc_s[j][k][4] + acc_s[j][k][5]);
                scores_sum[1] += (acc_s[j][k][2] + acc_s[j][k][3] + acc_s[j][k][6] + acc_s[j][k][7]);
            }
        }
        #pragma unroll
        for (size_t j=0; j<2; j++) {
            #pragma unroll
            for (size_t k=1; k<4; k*=2) {
                scores_sum[j] += __shfl_xor_sync(mask, scores_sum[j], k, 4);
            }
        }
        // logsum = logsum * scores_scale + sum;
        #pragma unroll
        for (size_t j=0; j<2; j++) {
            logsum[0][j] = logsum[0][j] * scores_scale[j] + scores_sum[j];
        }
        // cast acc_s
        fp16 acc_s_cast[BN/QK_MMA_N][QK_MMA_N/16][8];
        #pragma unroll
        for (size_t j=0; j<BN/QK_MMA_N; j++) {
            #pragma unroll
            for (size_t k=0; k<QK_MMA_N/16; k++) {
                #pragma unroll
                for (size_t l=0; l<8; l+=2) {
                    uint1 _t2;
                    float2 _t1 = *(float2*)(&acc_s[j][k][l]);
                    *(half2*)(&(_t2)) = __float22half2_rn(*(float2*)(&(_t1)));
                    *(uint1*)(&acc_s_cast[j][k][l]) = _t2;
                }
            }
        }
        // acc_o = acc_o * scores_scale
        #pragma unroll
        for (size_t j=0; j<DIM/PV_MMA_N; j++) {
            #pragma unroll
            for (size_t k=0; k<PV_MMA_N/16; k++) {
                acc_o[0][j][k][0] *= scores_scale[0];
                acc_o[0][j][k][1] *= scores_scale[0];
                acc_o[0][j][k][4] *= scores_scale[0];
                acc_o[0][j][k][5] *= scores_scale[0];
                acc_o[0][j][k][2] *= scores_scale[1];
                acc_o[0][j][k][3] *= scores_scale[1];
                acc_o[0][j][k][6] *= scores_scale[1];
                acc_o[0][j][k][7] *= scores_scale[1];
            }
        }
        
        smem_q_i++;
        
        // ==== main loop ====
        for (size_t iw=BN; iw<S; iw+=BN) {

            if (smem_k_i >= NUM_SMEM_KV) { smem_k_i = 0; phase_k ^= 1; }
            if (smem_v_i >= NUM_SMEM_KV) { smem_v_i = 0; phase_v ^= 1; }

            // #pragma unroll
            for (int n=N-1; n>=0; n--, smem_q_i++) {

                if (smem_q_i >= NUM_SMEM_Q) { smem_q_i = 0; phase_q ^= 1; }
                fp16 *QAddr = sQ + smem_q_i * 2 * MMA_M * DIM;

                // fill acc_s
                #pragma unroll
                for (size_t j=0; j<BN/QK_MMA_N; j++) {
                    #pragma unroll
                    for (size_t k=0; k<QK_MMA_N/16; k++) {
                        #pragma unroll
                        for (size_t l=0; l<8; l++) {
                            acc_s[j][k][l] = 0.0f;
                        }
                    }
                }

                // gemm-qk
                fp16 *KAddr = sK + smem_k_i * BN * DIM;
                wait(&Qfull[smem_q_i], phase_q);
                wait(&Kfull[smem_k_i], phase_k);
                warpgroup_arrive();
                #pragma unroll
                for (size_t j=0; j<BN/QK_MMA_N; j++) {
                    #pragma unroll
                    for (size_t k=0; k<DIM; k+=MMA_K) {
                        int q_row = 0;
                        int q_col = k;
                        int k_row = j * QK_MMA_N;
                        int k_col = k;
                        fp16 *_QAddr = QAddr
                            + wg_idx * MMA_M * DIM
                            + tma_smem_offset_2d<MMA_M>(q_row, q_col);
                        fp16 *_KAddr = KAddr + tma_smem_offset_2d<BN>(k_row, k_col);
                        wgmma_ss<QK_MMA_N, 1, 1, 1, 0, 0, 16, 1024, true>(acc_s[j], _QAddr, _KAddr);
                    }
                }
                warpgroup_commit_batch();
                warpgroup_wait();
                arrive(&Qempty[smem_q_i]);
                if (n == N-1) {  // 这里的设置必须要smem == stage数量
                    arrive(&Kempty[smem_k_i]);
                    smem_k_i++;
                    if (smem_k_i >= NUM_SMEM_KV) { smem_k_i = 0; phase_k ^= 1; }
                }
                
                // gemm-pv
                fp16 *VAddr = sV + smem_v_i * BN * DIM;
                wait(&Vfull[smem_v_i], phase_v);
                warpgroup_arrive();
                #pragma unroll
                for (size_t j=0; j<DIM/PV_MMA_N; j++) {
                    #pragma unroll
                    for (size_t k=0; k<BN; k+=MMA_K) {
                        // V is stored in shared as [K=BN, N=DIM].
                        // Use TransB=1 so WGMMA consumes it logically as [N, K].
                        int v_row = k;
                        int v_col = j * PV_MMA_N;
                        const int p_tile_outer = k / QK_MMA_N;
                        const int p_tile_inner = (k % QK_MMA_N) / 16;
                        fp16 *_VAddr = VAddr + tma_smem_offset_2d<BN>(v_row, v_col);
                        wgmma_rs<PV_MMA_N, 1, 1, 1, 0, 1, BN * 64 * sizeof(fp16), 1024, true>(
                            acc_o[(n+N-1)%N][j],
                            reinterpret_cast<uint32_t*>(acc_s_cast[p_tile_outer][p_tile_inner]),
                            _VAddr);
                    }
                }
                warpgroup_commit_batch();
                warpgroup_wait();
                if (n == 0) {  // 这里的设置必须要smem == stage数量
                    arrive(&Vempty[smem_v_i]);
                    smem_v_i++;
                }

                // max_prev = max
                #pragma unroll
                for (size_t j=0; j<2; j++) {
                    scores_max_prev[n][j] = scores_max[n][j];
                }
                // reduce max
                #pragma unroll
                for (size_t j=0; j<2; j++) {
                    scores_max[n][j] = -FLT_MAX;
                }
                #pragma unroll
                for (size_t j=0; j<BN/QK_MMA_N; j++) {
                    #pragma unroll
                    for (size_t k=0; k<QK_MMA_N/16; k++) {
                        scores_max[n][0] = max(acc_s[j][k][0], scores_max[n][0]);
                        scores_max[n][0] = max(acc_s[j][k][1], scores_max[n][0]);
                        scores_max[n][0] = max(acc_s[j][k][4], scores_max[n][0]);
                        scores_max[n][0] = max(acc_s[j][k][5], scores_max[n][0]);
                        scores_max[n][1] = max(acc_s[j][k][2], scores_max[n][1]);
                        scores_max[n][1] = max(acc_s[j][k][3], scores_max[n][1]);
                        scores_max[n][1] = max(acc_s[j][k][6], scores_max[n][1]);
                        scores_max[n][1] = max(acc_s[j][k][7], scores_max[n][1]);
                    }
                }
                #pragma unroll
                for (size_t j=0; j<2; j++) {
                    #pragma unroll
                    for (size_t k=1; k<4; k*=2) {
                        scores_max[n][j] = max(scores_max[n][j], __shfl_xor_sync(mask, scores_max[n][j], k, 4));
                    }
                }
                // m = max(pm, m)
                #pragma unroll
                for (size_t j=0; j<2; j++) {
                    scores_max[n][j] = max(scores_max_prev[n][j], scores_max[n][j]);
                }
                // scores_scale = exp2(pm  - m)
                #pragma unroll
                for (size_t j=0; j<2; j++) {
                    scores_scale[j] = exp2f(scores_max_prev[n][j] * scale - scores_max[n][j] * scale);
                }
                // acc_s = exp2(acc_s - m)
                #pragma unroll
                for (size_t j=0; j<BN/QK_MMA_N; j++) {
                    #pragma unroll
                    for (size_t k=0; k<QK_MMA_N/16; k++) {
                        acc_s[j][k][0] = exp2f(acc_s[j][k][0] * scale - scores_max[n][0] * scale);
                        acc_s[j][k][1] = exp2f(acc_s[j][k][1] * scale - scores_max[n][0] * scale);
                        acc_s[j][k][4] = exp2f(acc_s[j][k][4] * scale - scores_max[n][0] * scale);
                        acc_s[j][k][5] = exp2f(acc_s[j][k][5] * scale - scores_max[n][0] * scale);
                        acc_s[j][k][2] = exp2f(acc_s[j][k][2] * scale - scores_max[n][1] * scale);
                        acc_s[j][k][3] = exp2f(acc_s[j][k][3] * scale - scores_max[n][1] * scale);
                        acc_s[j][k][6] = exp2f(acc_s[j][k][6] * scale - scores_max[n][1] * scale);
                        acc_s[j][k][7] = exp2f(acc_s[j][k][7] * scale - scores_max[n][1] * scale);
                    }
                }
                // reduce sum
                #pragma unroll
                for (size_t j=0; j<2; j++) {
                    scores_sum[j] = 0.0f;
                }
                #pragma unroll
                for (size_t j=0; j<BN/QK_MMA_N; j++) {
                    #pragma unroll
                    for (size_t k=0; k<QK_MMA_N/16; k++) {
                        scores_sum[0] += (acc_s[j][k][0] + acc_s[j][k][1] + acc_s[j][k][4] + acc_s[j][k][5]);
                        scores_sum[1] += (acc_s[j][k][2] + acc_s[j][k][3] + acc_s[j][k][6] + acc_s[j][k][7]);
                    }
                }
                #pragma unroll
                for (size_t j=0; j<2; j++) {
                    #pragma unroll
                    for (size_t k=1; k<4; k*=2) {
                        scores_sum[j] += __shfl_xor_sync(mask, scores_sum[j], k, 4);
                    }
                }
                // logsum = logsum * scores_scale + sum;
                #pragma unroll
                for (size_t j=0; j<2; j++) {
                    logsum[n][j] = logsum[n][j] * scores_scale[j] + scores_sum[j];
                }
                // cast acc_s
                #pragma unroll
                for (size_t j=0; j<BN/QK_MMA_N; j++) {
                    #pragma unroll
                    for (size_t k=0; k<QK_MMA_N/16; k++) {
                        #pragma unroll
                        for (size_t l=0; l<8; l+=2) {
                            uint1 _t2;
                            float2 _t1 = *(float2*)(&acc_s[j][k][l]);
                            *(half2*)(&(_t2)) = __float22half2_rn(*(float2*)(&(_t1)));
                            *(uint1*)(&acc_s_cast[j][k][l]) = _t2;
                        }
                    }
                }
                // acc_o = acc_o * scores_scale
                #pragma unroll
                for (size_t j=0; j<DIM/PV_MMA_N; j++) {
                    #pragma unroll
                    for (size_t k=0; k<PV_MMA_N/16; k++) {
                        acc_o[n][j][k][0] *= scores_scale[0];
                        acc_o[n][j][k][1] *= scores_scale[0];
                        acc_o[n][j][k][4] *= scores_scale[0];
                        acc_o[n][j][k][5] *= scores_scale[0];
                        acc_o[n][j][k][2] *= scores_scale[1];
                        acc_o[n][j][k][3] *= scores_scale[1];
                        acc_o[n][j][k][6] *= scores_scale[1];
                        acc_o[n][j][k][7] *= scores_scale[1];
                    }
                }
            }
        }

        // ==== eiplogue ====
        if (smem_q_i >= NUM_SMEM_Q) { smem_q_i = 0; phase_q ^= 1; }
        fp16 *QAddr = sQ + smem_q_i * 2 * MMA_M * DIM;

        // fill acc_s
        #pragma unroll
        for (size_t j=0; j<BN/QK_MMA_N; j++) {
            #pragma unroll
            for (size_t k=0; k<QK_MMA_N/16; k++) {
                #pragma unroll
                for (size_t l=0; l<8; l++) {
                    acc_s[j][k][l] = 0.0f;
                }
            }
        }
        // gemm-qk
        fp16 *KAddr = sK + smem_k_i * BN * DIM;
        wait(&Qfull[smem_q_i], phase_q);
        wait(&Kfull[smem_k_i], phase_k);
        warpgroup_arrive();
        #pragma unroll
        for (size_t j=0; j<BN/QK_MMA_N; j++) {
            #pragma unroll
            for (size_t k=0; k<DIM; k+=MMA_K) {
                int q_row = 0;
                int q_col = k;
                int k_row = j * QK_MMA_N;
                int k_col = k;
                fp16 *_QAddr = QAddr
                    + wg_idx * MMA_M * DIM
                    + tma_smem_offset_2d<MMA_M>(q_row, q_col);
                fp16 *_KAddr = KAddr + tma_smem_offset_2d<BN>(k_row, k_col);
                wgmma_ss<QK_MMA_N, 1, 1, 1, 0, 0, 16, 1024, true>(acc_s[j], _QAddr, _KAddr);
            }
        }
        warpgroup_commit_batch();
        warpgroup_wait();

        // gemm-pv
        fp16 *VAddr = sV + smem_v_i * BN * DIM;
        wait(&Vfull[smem_v_i], phase_v);
        warpgroup_arrive();
        #pragma unroll
        for (size_t j=0; j<DIM/PV_MMA_N; j++) {
            #pragma unroll
            for (size_t k=0; k<BN; k+=MMA_K) {
                // V is stored in shared as [K=BN, N=DIM].
                // Use TransB=1 so WGMMA consumes it logically as [N, K].
                int v_row = k;
                int v_col = j * PV_MMA_N;
                const int p_tile_outer = k / QK_MMA_N;
                const int p_tile_inner = (k % QK_MMA_N) / 16;
                fp16 *_VAddr = VAddr + tma_smem_offset_2d<BN>(v_row, v_col);
                wgmma_rs<PV_MMA_N, 1, 1, 1, 0, 1, BN * 64 * sizeof(fp16), 1024, true>(
                    acc_o[0][j],
                    reinterpret_cast<uint32_t*>(acc_s_cast[p_tile_outer][p_tile_inner]),
                    _VAddr);
            }
        }
        warpgroup_commit_batch();
        warpgroup_wait();

        // max_prev = max
        #pragma unroll
        for (size_t j=0; j<2; j++) {
            scores_max_prev[1][j] = scores_max[1][j];
        }
        // reduce max
        #pragma unroll
        for (size_t j=0; j<2; j++) {
            scores_max[1][j] = -FLT_MAX;
        }
        #pragma unroll
        for (size_t j=0; j<BN/QK_MMA_N; j++) {
            #pragma unroll
            for (size_t k=0; k<QK_MMA_N/16; k++) {
                scores_max[1][0] = max(acc_s[j][k][0], scores_max[1][0]);
                scores_max[1][0] = max(acc_s[j][k][1], scores_max[1][0]);
                scores_max[1][0] = max(acc_s[j][k][4], scores_max[1][0]);
                scores_max[1][0] = max(acc_s[j][k][5], scores_max[1][0]);
                scores_max[1][1] = max(acc_s[j][k][2], scores_max[1][1]);
                scores_max[1][1] = max(acc_s[j][k][3], scores_max[1][1]);
                scores_max[1][1] = max(acc_s[j][k][6], scores_max[1][1]);
                scores_max[1][1] = max(acc_s[j][k][7], scores_max[1][1]);
            }
        }
        #pragma unroll
        for (size_t j=0; j<2; j++) {
            #pragma unroll
            for (size_t k=1; k<4; k*=2) {
                scores_max[1][j] = max(scores_max[1][j], __shfl_xor_sync(mask, scores_max[1][j], k, 4));
            }
        }
        // m = max(pm, m)
        #pragma unroll
        for (size_t j=0; j<2; j++) {
            scores_max[1][j] = max(scores_max_prev[1][j], scores_max[1][j]);
        }
        // scores_scale = exp2(pm  - m)
        #pragma unroll
        for (size_t j=0; j<2; j++) {
            scores_scale[j] = exp2f(scores_max_prev[1][j] * scale - scores_max[1][j] * scale);
        }
        // acc_s = exp2(acc_s - m)
        #pragma unroll
        for (size_t j=0; j<BN/QK_MMA_N; j++) {
            #pragma unroll
            for (size_t k=0; k<QK_MMA_N/16; k++) {
                acc_s[j][k][0] = exp2f(acc_s[j][k][0] * scale - scores_max[1][0] * scale);
                acc_s[j][k][1] = exp2f(acc_s[j][k][1] * scale - scores_max[1][0] * scale);
                acc_s[j][k][4] = exp2f(acc_s[j][k][4] * scale - scores_max[1][0] * scale);
                acc_s[j][k][5] = exp2f(acc_s[j][k][5] * scale - scores_max[1][0] * scale);
                acc_s[j][k][2] = exp2f(acc_s[j][k][2] * scale - scores_max[1][1] * scale);
                acc_s[j][k][3] = exp2f(acc_s[j][k][3] * scale - scores_max[1][1] * scale);
                acc_s[j][k][6] = exp2f(acc_s[j][k][6] * scale - scores_max[1][1] * scale);
                acc_s[j][k][7] = exp2f(acc_s[j][k][7] * scale - scores_max[1][1] * scale);
            }
        }
        // reduce sum
        #pragma unroll
        for (size_t j=0; j<2; j++) {
            scores_sum[j] = 0.0f;
        }
        #pragma unroll
        for (size_t j=0; j<BN/QK_MMA_N; j++) {
            #pragma unroll
            for (size_t k=0; k<QK_MMA_N/16; k++) {
                scores_sum[0] += (acc_s[j][k][0] + acc_s[j][k][1] + acc_s[j][k][4] + acc_s[j][k][5]);
                scores_sum[1] += (acc_s[j][k][2] + acc_s[j][k][3] + acc_s[j][k][6] + acc_s[j][k][7]);
            }
        }
        #pragma unroll
        for (size_t j=0; j<2; j++) {
            #pragma unroll
            for (size_t k=1; k<4; k*=2) {
                scores_sum[j] += __shfl_xor_sync(mask, scores_sum[j], k, 4);
            }
        }
        // logsum = logsum * scores_scale + sum;
        #pragma unroll
        for (size_t j=0; j<2; j++) {
            logsum[1][j] = logsum[1][j] * scores_scale[j] + scores_sum[j];
        }
        // cast acc_s
        #pragma unroll
        for (size_t j=0; j<BN/QK_MMA_N; j++) {
            #pragma unroll
            for (size_t k=0; k<QK_MMA_N/16; k++) {
                #pragma unroll
                for (size_t l=0; l<8; l+=2) {
                    uint1 _t2;
                    float2 _t1 = *(float2*)(&acc_s[j][k][l]);
                    *(half2*)(&(_t2)) = __float22half2_rn(*(float2*)(&(_t1)));
                    *(uint1*)(&acc_s_cast[j][k][l]) = _t2;
                }
            }
        }
        // acc_o = acc_o * scores_scale
        #pragma unroll
        for (size_t j=0; j<DIM/PV_MMA_N; j++) {
            #pragma unroll
            for (size_t k=0; k<PV_MMA_N/16; k++) {
                acc_o[1][j][k][0] *= scores_scale[0];
                acc_o[1][j][k][1] *= scores_scale[0];
                acc_o[1][j][k][4] *= scores_scale[0];
                acc_o[1][j][k][5] *= scores_scale[0];
                acc_o[1][j][k][2] *= scores_scale[1];
                acc_o[1][j][k][3] *= scores_scale[1];
                acc_o[1][j][k][6] *= scores_scale[1];
                acc_o[1][j][k][7] *= scores_scale[1];
            }
        }

        // gemm-pv
        wait(&Vfull[smem_v_i], phase_v);
        warpgroup_arrive();
        #pragma unroll
        for (size_t j=0; j<DIM/PV_MMA_N; j++) {
            #pragma unroll
            for (size_t k=0; k<BN; k+=MMA_K) {
                // V is stored in shared as [K=BN, N=DIM].
                // Use TransB=1 so WGMMA consumes it logically as [N, K].
                int v_row = k;
                int v_col = j * PV_MMA_N;
                const int p_tile_outer = k / QK_MMA_N;
                const int p_tile_inner = (k % QK_MMA_N) / 16;
                fp16 *_VAddr = VAddr + tma_smem_offset_2d<BN>(v_row, v_col);
                wgmma_rs<PV_MMA_N, 1, 1, 1, 0, 1, BN * 64 * sizeof(fp16), 1024, true>(
                    acc_o[1][j],
                    reinterpret_cast<uint32_t*>(acc_s_cast[p_tile_outer][p_tile_inner]),
                    _VAddr);
            }
        }
        warpgroup_commit_batch();
        warpgroup_wait();
#endif

        // acc_o = acc_o / logsum
        #pragma unroll
        for (size_t i=0; i<BM/(MMA_M*2); i++) {
            float val0 = 1.0f / logsum[i][0];
            float val1 = 1.0f / logsum[i][1];
            #pragma unroll
            for (size_t j=0; j<DIM/PV_MMA_N; j++) {
                #pragma unroll
                for (size_t k=0; k<PV_MMA_N/16; k++) {
                    acc_o[i][j][k][0] *= val0;
                    acc_o[i][j][k][1] *= val0;
                    acc_o[i][j][k][4] *= val0;
                    acc_o[i][j][k][5] *= val0;
                    acc_o[i][j][k][2] *= val1;
                    acc_o[i][j][k][3] *= val1;
                    acc_o[i][j][k][6] *= val1;
                    acc_o[i][j][k][7] *= val1;
                }
            }
        }

        // copy acc_o to sO (load matritx)
        if (tid == 0) {
            tma_store_wait();
        }

        const int lane_row = lane_id & 0xf;
        const int lane_col = (lane_id >> 4) * 8;
        #pragma unroll
        for (size_t i=0; i<BM/(MMA_M*2); i++) {
            #pragma unroll
            for (size_t j=0; j<DIM/PV_MMA_N; j++) {
                #pragma unroll
                for (size_t k=0; k<PV_MMA_N/16; k++) {
                    int o_row = wg_idx * BM/(MMA_M*2) * MMA_M
                                + i * MMA_M
                                + warp_id_in_wg * 16
                                + lane_row;
                    int o_col = j * PV_MMA_N
                                + k * 16
                                + lane_col;
                    fp16 *_sO = sO + tma_smem_swizzle_128b_offset_2d<BM>(o_row, o_col);
                        uint32_t r0 = half2_to_u32(__floats2half2_rn(
                            acc_o[i][j][k][0],
                            acc_o[i][j][k][1]
                        ));
                        uint32_t r1 = half2_to_u32(__floats2half2_rn(
                            acc_o[i][j][k][2],
                            acc_o[i][j][k][3]
                        ));
                        uint32_t r2 = half2_to_u32(__floats2half2_rn(
                            acc_o[i][j][k][4],
                            acc_o[i][j][k][5]
                        ));
                        uint32_t r3 = half2_to_u32(__floats2half2_rn(
                            acc_o[i][j][k][6],
                            acc_o[i][j][k][7]
                        ));

                        stmatrix_x4_reg(_sO, r0, r1, r2, r3);
                }
            }
        }
        // Wait all consumer threads (2 warpgroups) before issuing TMA store from tid==0.
        fence_view_async_shared();
        bar_sync(256, 2);
        // tma store
        if (tid == 0) {
            store_async(&tensorMapO, sO, bs, hn, by * BM, 0);
            tma_store_arrive();
        }
    }
}


template<int BM, int BN, int DIM, int NUM_THREADS, int NUM_SMEM_Q, int NUM_SMEM_KV>
__global__  __launch_bounds__(NUM_THREADS) 
void attnWSKCXForNUnrollDoubleQ2StagePVSSKernel(
    int B, int H, int S, 
    const __grid_constant__ CUtensorMap tensorMapQ, 
    const __grid_constant__ CUtensorMap tensorMapK, 
    const __grid_constant__ CUtensorMap tensorMapV, 
    const __grid_constant__ CUtensorMap tensorMapO
) {
    // WS attention
    const int bs = blockIdx.z;
    const int hn = blockIdx.y;
    const int by = blockIdx.x;
    const int tid = threadIdx.x;
    uint32_t wg_idx = tid >> 7;

    // mma size
    static_assert((DIM >= 256 || DIM == 16 || DIM == 32 || DIM == 64 || DIM == 128) && "DIM ERROR!");
    static_assert((BN >= 256 || BN == 16 || BN == 32 || BN == 64 || BN == 128) && "BN ERROR!");
    constexpr int MMA_M = 64;
    constexpr int QK_MMA_N = BN <= 256 ? BN : 256;
    constexpr int PV_MMA_N = DIM <= 256 ? DIM : 256;
    constexpr int MMA_K = 16;
    constexpr int N = BM / 2 / MMA_M;
    static_assert(N >= NUM_SMEM_Q && "BN ERROR!");
    static_assert(N <= 4, "attnWSKCXForNDoubleQKernel unroll currently supports N <= 4.");

    // setting shared memory
    extern __shared__ __align__(128) uint8_t smem[];
    SMemWSDoubleQPVSS<BM, BN, DIM, NUM_SMEM_Q, NUM_SMEM_KV> &s = 
        *reinterpret_cast<SMemWSDoubleQPVSS<BM, BN, DIM, NUM_SMEM_Q, NUM_SMEM_KV>*>(smem);
    fp16 *sQ = s.Q, *sK = s.K, *sV = s.V, *sP = s.P, *sO = s.O;
    Barrier *Qempty = s.Qempty, *Kempty = s.Kempty, *Vempty = s.Vempty;
    Barrier *Qfull = s.Qfull, *Kfull = s.Kfull, *Vfull = s.Vfull;

    // init mbarrier
    if (threadIdx.x == 0) {
        for (int i=0; i<NUM_SMEM_Q; ++i) {
            init_barrier(&Qfull[i], 1);
            init_barrier(&Qempty[i], 256);
        }
        for (int i = 0; i < NUM_SMEM_KV; ++i) {
            init_barrier(&Kfull[i], 1);  // 1 thread arrive
            init_barrier(&Vfull[i], 1);
            init_barrier(&Kempty[i], 256);  // 256 thread arrive
            init_barrier(&Vempty[i], 256);
        }
    }
    __syncthreads();
    fence_view_async_shared();

    // TMA load
    if (wg_idx == 2) {  // producer
        warpgroup_reg_dealloc<24>();
        if (tid == 256) {
            int smem_kv_i = 0, phase_kv = 0;
            for (size_t iw=0; iw<S; iw+=BN, ++smem_kv_i) {
                if (smem_kv_i >= NUM_SMEM_KV) { smem_kv_i = 0; phase_kv ^= 1; }
                fp16 *KAddr = sK + smem_kv_i * BN * DIM;
                fp16 *VAddr = sV + smem_kv_i * BN * DIM;
                
                // load K
                wait(&Kempty[smem_kv_i], phase_kv);
                expect_bytes(&Kfull[smem_kv_i], BN * DIM * sizeof(fp16));
                load_async(KAddr, &tensorMapK, &Kfull[smem_kv_i], bs, hn, iw, 0);

                // load V
                wait(&Vempty[smem_kv_i], phase_kv);
                expect_bytes(&Vfull[smem_kv_i], BN * DIM * sizeof(fp16));
                load_async(VAddr, &tensorMapV, &Vfull[smem_kv_i], bs, hn, iw, 0);
            }
        } else if (tid == 288) {
            int smem_q_i = 0, phase_q = 0;
            for (size_t iw=0; iw<S; iw+=BN) {
                auto load_q_n = [&](auto n_const) {
                    constexpr int n = decltype(n_const)::value;
                    static_assert(n < N);

                    if (smem_q_i >= NUM_SMEM_Q) { smem_q_i = 0; phase_q ^= 1; }
                    
                    // load Q
                    wait(&Qempty[smem_q_i], phase_q);
                    expect_bytes(&Qfull[smem_q_i], 2 * MMA_M * DIM * sizeof(fp16));
                    #pragma unroll
                    for (size_t i=0; i<2; i++) {
                        fp16 *QAddr = sQ + smem_q_i * 2 * MMA_M * DIM + i * MMA_M * DIM;
                        load_async(QAddr, &tensorMapQ, &Qfull[smem_q_i], bs, hn, by * BM + i * N * MMA_M + n * MMA_M, 0);
                    }
                    ++smem_q_i;
                };

                load_q_n(std::integral_constant<int, 0>{});
                if constexpr (N > 1) { load_q_n(std::integral_constant<int, 1>{}); }
                if constexpr (N > 2) { load_q_n(std::integral_constant<int, 2>{}); }
                if constexpr (N > 3) { load_q_n(std::integral_constant<int, 3>{}); }
                if constexpr (N > 4) { load_q_n(std::integral_constant<int, 4>{}); }
            }
        }
    } else {  // consumer
        warpgroup_reg_alloc<240>();
        // Bootstrap empty-smem_i barriers so producer can issue the first K/V loads.
        #pragma unroll
        for (int i = 0; i < NUM_SMEM_Q; ++i) {
            arrive(&Qempty[i]);
        }
        #pragma unroll
        for (int i = 0; i < NUM_SMEM_KV; ++i) {
            arrive(&Kempty[i]);
            arrive(&Vempty[i]);
        }
        
        // need args
        uint32_t lane_id = tid & 31;
        uint32_t warp_id_in_wg = (tid >> 5) & 0x3;
        const int p_lane_row = lane_id & 0xf;
        const int p_lane_col = (lane_id >> 4) * 8;
        const unsigned mask = __activemask();
        const fp32 scale = sqrt((1.0f / DIM)) * 1.44269504f;  // log2(e)

        // registers define
        fp32 acc_o[N][DIM/PV_MMA_N][PV_MMA_N/16][8];
        fp32 scores_max_prev[N][2];
        fp32 scores_max[N][2];
        fp32 logsum[N][2];

        // init acc_o
        #pragma unroll
        for (size_t i=0; i<N; i++) {
            #pragma unroll
            for (size_t j=0; j<DIM/PV_MMA_N; j++) {
                #pragma unroll
                for (size_t k=0; k<PV_MMA_N/16; k++) {
                    #pragma unroll
                    for (size_t l=0; l<8; l++) {
                        acc_o[i][j][k][l] = 0.0f;
                    }
                }
            }
        }
        
        // init logsum and scores_max
        #pragma unroll
        for (size_t i=0; i<N; i++) {
            #pragma unroll
            for (size_t j=0; j<2; j++) {
                logsum[i][j] = 0.0f;
                scores_max[i][j] = -FLT_MAX;
            }
        }
        

        size_t smem_k_i = 0, phase_k = 0;
        size_t smem_q_i = 0, phase_q = 0;
        size_t smem_v_i = 0, phase_v = 0;

        // ==== prologue ====
        // fill acc_s
        fp32 acc_s[BN/QK_MMA_N][QK_MMA_N/16][8];
        auto store_p_smem = [&]() {
            fp16 *PBase = sP + wg_idx * MMA_M * BN;
            #pragma unroll
            for (size_t j=0; j<BN/QK_MMA_N; j++) {
                #pragma unroll
                for (size_t k=0; k<QK_MMA_N/16; k++) {
                    const int p_row = warp_id_in_wg * 16 + p_lane_row;
                    const int p_col = j * QK_MMA_N + k * 16 + p_lane_col;
                    fp16 *_sP = PBase + tma_smem_swizzle_128b_offset_2d<MMA_M>(p_row, p_col);
                    uint32_t r0 = half2_to_u32(__floats2half2_rn(acc_s[j][k][0], acc_s[j][k][1]));
                    uint32_t r1 = half2_to_u32(__floats2half2_rn(acc_s[j][k][2], acc_s[j][k][3]));
                    uint32_t r2 = half2_to_u32(__floats2half2_rn(acc_s[j][k][4], acc_s[j][k][5]));
                    uint32_t r3 = half2_to_u32(__floats2half2_rn(acc_s[j][k][6], acc_s[j][k][7]));
                    stmatrix_x4_reg(_sP, r0, r1, r2, r3);
                }
            }
            bar_sync(256, 3);
        };
        #pragma unroll
        for (size_t j=0; j<BN/QK_MMA_N; j++) {
            #pragma unroll
            for (size_t k=0; k<QK_MMA_N/16; k++) {
                #pragma unroll
                for (size_t l=0; l<8; l++) {
                    acc_s[j][k][l] = 0.0f;
                }
            }
        }
        // gemm-qk
        wait(&Qfull[smem_q_i], phase_q);
        wait(&Kfull[smem_k_i], phase_k);
        warpgroup_arrive();
        #pragma unroll
        for (size_t j=0; j<BN/QK_MMA_N; j++) {
            #pragma unroll
            for (size_t k=0; k<DIM; k+=MMA_K) {
                int q_row = 0;
                int q_col = k;
                int k_row = j * QK_MMA_N;
                int k_col = k;
                fp16 *_QAddr = sQ
                    + wg_idx * MMA_M * DIM
                    + tma_smem_offset_2d<MMA_M>(q_row, q_col);
                fp16 *_KAddr = sK + tma_smem_offset_2d<BN>(k_row, k_col);
                wgmma_ss<QK_MMA_N, 1, 1, 1, 0, 0, 16, 1024, true>(acc_s[j], _QAddr, _KAddr);
            }
        }
        warpgroup_commit_batch();
        warpgroup_wait();
        arrive(&Qempty[smem_q_i]);
        
        // max_prev = max
        #pragma unroll
        for (size_t j=0; j<2; j++) {
            scores_max_prev[0][j] = scores_max[0][j];
        }
        // reduce max
        #pragma unroll
        for (size_t j=0; j<2; j++) {
            scores_max[0][j] = -FLT_MAX;
        }
        #pragma unroll
        for (size_t j=0; j<BN/QK_MMA_N; j++) {
            #pragma unroll
            for (size_t k=0; k<QK_MMA_N/16; k++) {
                scores_max[0][0] = max(acc_s[j][k][0], scores_max[0][0]);
                scores_max[0][0] = max(acc_s[j][k][1], scores_max[0][0]);
                scores_max[0][0] = max(acc_s[j][k][4], scores_max[0][0]);
                scores_max[0][0] = max(acc_s[j][k][5], scores_max[0][0]);
                scores_max[0][1] = max(acc_s[j][k][2], scores_max[0][1]);
                scores_max[0][1] = max(acc_s[j][k][3], scores_max[0][1]);
                scores_max[0][1] = max(acc_s[j][k][6], scores_max[0][1]);
                scores_max[0][1] = max(acc_s[j][k][7], scores_max[0][1]);
            }
        }
        #pragma unroll
        for (size_t j=0; j<2; j++) {
            #pragma unroll
            for (size_t k=1; k<4; k*=2) {
                scores_max[0][j] = max(scores_max[0][j], __shfl_xor_sync(mask, scores_max[0][j], k, 4));
            }
        }
        // m = max(pm, m)
        #pragma unroll
        for (size_t j=0; j<2; j++) {
            scores_max[0][j] = max(scores_max_prev[0][j], scores_max[0][j]);
        }
        // scores_scale = exp2(pm  - m)
        fp32 scores_scale[2];
        #pragma unroll
        for (size_t j=0; j<2; j++) {
            scores_scale[j] = exp2f(scores_max_prev[0][j] * scale - scores_max[0][j] * scale);
        }
        // acc_s = exp2(acc_s - m)
        #pragma unroll
        for (size_t j=0; j<BN/QK_MMA_N; j++) {
            #pragma unroll
            for (size_t k=0; k<QK_MMA_N/16; k++) {
                acc_s[j][k][0] = exp2f(acc_s[j][k][0] * scale - scores_max[0][0] * scale);
                acc_s[j][k][1] = exp2f(acc_s[j][k][1] * scale - scores_max[0][0] * scale);
                acc_s[j][k][4] = exp2f(acc_s[j][k][4] * scale - scores_max[0][0] * scale);
                acc_s[j][k][5] = exp2f(acc_s[j][k][5] * scale - scores_max[0][0] * scale);
                acc_s[j][k][2] = exp2f(acc_s[j][k][2] * scale - scores_max[0][1] * scale);
                acc_s[j][k][3] = exp2f(acc_s[j][k][3] * scale - scores_max[0][1] * scale);
                acc_s[j][k][6] = exp2f(acc_s[j][k][6] * scale - scores_max[0][1] * scale);
                acc_s[j][k][7] = exp2f(acc_s[j][k][7] * scale - scores_max[0][1] * scale);
            }
        }
        // reduce sum
        fp32 scores_sum[2];
        #pragma unroll
        for (size_t j=0; j<2; j++) {
            scores_sum[j] = 0.0f;
        }
        #pragma unroll
        for (size_t j=0; j<BN/QK_MMA_N; j++) {
            #pragma unroll
            for (size_t k=0; k<QK_MMA_N/16; k++) {
                scores_sum[0] += (acc_s[j][k][0] + acc_s[j][k][1] + acc_s[j][k][4] + acc_s[j][k][5]);
                scores_sum[1] += (acc_s[j][k][2] + acc_s[j][k][3] + acc_s[j][k][6] + acc_s[j][k][7]);
            }
        }
        #pragma unroll
        for (size_t j=0; j<2; j++) {
            #pragma unroll
            for (size_t k=1; k<4; k*=2) {
                scores_sum[j] += __shfl_xor_sync(mask, scores_sum[j], k, 4);
            }
        }
        // logsum = logsum * scores_scale + sum;
        #pragma unroll
        for (size_t j=0; j<2; j++) {
            logsum[0][j] = logsum[0][j] * scores_scale[j] + scores_sum[j];
        }
        // store P = exp2(QK - m) to smem for the following PV stage
        store_p_smem();
        // acc_o = acc_o * scores_scale
        #pragma unroll
        for (size_t j=0; j<DIM/PV_MMA_N; j++) {
            #pragma unroll
            for (size_t k=0; k<PV_MMA_N/16; k++) {
                acc_o[0][j][k][0] *= scores_scale[0];
                acc_o[0][j][k][1] *= scores_scale[0];
                acc_o[0][j][k][4] *= scores_scale[0];
                acc_o[0][j][k][5] *= scores_scale[0];
                acc_o[0][j][k][2] *= scores_scale[1];
                acc_o[0][j][k][3] *= scores_scale[1];
                acc_o[0][j][k][6] *= scores_scale[1];
                acc_o[0][j][k][7] *= scores_scale[1];
            }
        }
        
        smem_q_i++;
        
        // ==== main loop ====
        for (size_t iw=BN; iw<S; iw+=BN) {

            if (smem_k_i >= NUM_SMEM_KV) { smem_k_i = 0; phase_k ^= 1; }
            if (smem_v_i >= NUM_SMEM_KV) { smem_v_i = 0; phase_v ^= 1; }

            #pragma unroll
            for (int n=N-1; n>=0; n--, smem_q_i++) {

                if (smem_q_i >= NUM_SMEM_Q) { smem_q_i = 0; phase_q ^= 1; }
                fp16 *QAddr = sQ + smem_q_i * 2 * MMA_M * DIM;

                // fill acc_s
                // fp32 acc_s[BN/QK_MMA_N][QK_MMA_N/16][8];
                #pragma unroll
                for (size_t j=0; j<BN/QK_MMA_N; j++) {
                    #pragma unroll
                    for (size_t k=0; k<QK_MMA_N/16; k++) {
                        #pragma unroll
                        for (size_t l=0; l<8; l++) {
                            acc_s[j][k][l] = 0.0f;
                        }
                    }
                }

                // gemm-qk
                fp16 *KAddr = sK + smem_k_i * BN * DIM;
                wait(&Qfull[smem_q_i], phase_q);
                wait(&Kfull[smem_k_i], phase_k);
                warpgroup_arrive();
                #pragma unroll
                for (size_t j=0; j<BN/QK_MMA_N; j++) {
                    #pragma unroll
                    for (size_t k=0; k<DIM; k+=MMA_K) {
                        int q_row = 0;
                        int q_col = k;
                        int k_row = j * QK_MMA_N;
                        int k_col = k;
                        fp16 *_QAddr = QAddr
                            + wg_idx * MMA_M * DIM
                            + tma_smem_offset_2d<MMA_M>(q_row, q_col);
                        fp16 *_KAddr = KAddr + tma_smem_offset_2d<BN>(k_row, k_col);
                        wgmma_ss<QK_MMA_N, 1, 1, 1, 0, 0, 16, 1024, true>(acc_s[j], _QAddr, _KAddr);
                    }
                }
                warpgroup_commit_batch();
                warpgroup_wait();
                arrive(&Qempty[smem_q_i]);
                if (n == N-1) {  // 这里的设置必须要smem == stage数量
                    arrive(&Kempty[smem_k_i]);
                    smem_k_i++;
                    if (smem_k_i >= NUM_SMEM_KV) { smem_k_i = 0; phase_k ^= 1; }
                }
                
                // gemm-pv
                fp16 *VAddr = sV + smem_v_i * BN * DIM;
                wait(&Vfull[smem_v_i], phase_v);
                warpgroup_arrive();
                #pragma unroll
                for (size_t j=0; j<DIM/PV_MMA_N; j++) {
                    #pragma unroll
                    for (size_t k=0; k<BN; k+=MMA_K) {
                        // V is stored in shared as [K=BN, N=DIM].
                        // Use TransB=1 so WGMMA consumes it logically as [N, K].
                        int v_row = k;
                        int v_col = j * PV_MMA_N;
                        fp16 *_PAddr = sP
                            + wg_idx * MMA_M * BN
                            + tma_smem_offset_2d<MMA_M>(0, k);
                        fp16 *_VAddr = VAddr + tma_smem_offset_2d<BN>(v_row, v_col);
                        wgmma_ss_ab<PV_MMA_N, 1, 1, 1, 0, 1,
                            16, 1024, true, BN * 64 * sizeof(fp16), 1024, true>(
                            acc_o[(n+N-1)%N][j],
                            _PAddr,
                            _VAddr);
                    }
                }
                warpgroup_commit_batch();
                warpgroup_wait();
                if (n == 0) {  // 这里的设置必须要smem == stage数量
                    arrive(&Vempty[smem_v_i]);
                    smem_v_i++;
                }

                // max_prev = max
                #pragma unroll
                for (size_t j=0; j<2; j++) {
                    scores_max_prev[n][j] = scores_max[n][j];
                }

                // reduce max
                #pragma unroll
                for (size_t j=0; j<2; j++) {
                    scores_max[n][j] = -FLT_MAX;
                }
                #pragma unroll
                for (size_t j=0; j<BN/QK_MMA_N; j++) {
                    #pragma unroll
                    for (size_t k=0; k<QK_MMA_N/16; k++) {
                        scores_max[n][0] = max(acc_s[j][k][0], scores_max[n][0]);
                        scores_max[n][0] = max(acc_s[j][k][1], scores_max[n][0]);
                        scores_max[n][0] = max(acc_s[j][k][4], scores_max[n][0]);
                        scores_max[n][0] = max(acc_s[j][k][5], scores_max[n][0]);
                        scores_max[n][1] = max(acc_s[j][k][2], scores_max[n][1]);
                        scores_max[n][1] = max(acc_s[j][k][3], scores_max[n][1]);
                        scores_max[n][1] = max(acc_s[j][k][6], scores_max[n][1]);
                        scores_max[n][1] = max(acc_s[j][k][7], scores_max[n][1]);
                    }
                }
                #pragma unroll
                for (size_t j=0; j<2; j++) {
                    #pragma unroll
                    for (size_t k=1; k<4; k*=2) {
                        scores_max[n][j] = max(scores_max[n][j], __shfl_xor_sync(mask, scores_max[n][j], k, 4));
                    }
                }

                // m = max(pm, m)
                #pragma unroll
                for (size_t j=0; j<2; j++) {
                    scores_max[n][j] = max(scores_max_prev[n][j], scores_max[n][j]);
                }

                // scores_scale = exp2(pm  - m)
                #pragma unroll
                for (size_t j=0; j<2; j++) {
                    scores_scale[j] = exp2f(scores_max_prev[n][j] * scale - scores_max[n][j] * scale);
                }

                // acc_s = exp2(acc_s - m)
                #pragma unroll
                for (size_t j=0; j<BN/QK_MMA_N; j++) {
                    #pragma unroll
                    for (size_t k=0; k<QK_MMA_N/16; k++) {
                        acc_s[j][k][0] = exp2f(acc_s[j][k][0] * scale - scores_max[n][0] * scale);
                        acc_s[j][k][1] = exp2f(acc_s[j][k][1] * scale - scores_max[n][0] * scale);
                        acc_s[j][k][4] = exp2f(acc_s[j][k][4] * scale - scores_max[n][0] * scale);
                        acc_s[j][k][5] = exp2f(acc_s[j][k][5] * scale - scores_max[n][0] * scale);
                        acc_s[j][k][2] = exp2f(acc_s[j][k][2] * scale - scores_max[n][1] * scale);
                        acc_s[j][k][3] = exp2f(acc_s[j][k][3] * scale - scores_max[n][1] * scale);
                        acc_s[j][k][6] = exp2f(acc_s[j][k][6] * scale - scores_max[n][1] * scale);
                        acc_s[j][k][7] = exp2f(acc_s[j][k][7] * scale - scores_max[n][1] * scale);
                    }
                }

                // reduce sum
                #pragma unroll
                for (size_t j=0; j<2; j++) {
                    scores_sum[j] = 0.0f;
                }
                #pragma unroll
                for (size_t j=0; j<BN/QK_MMA_N; j++) {
                    #pragma unroll
                    for (size_t k=0; k<QK_MMA_N/16; k++) {
                        scores_sum[0] += (acc_s[j][k][0] + acc_s[j][k][1] + acc_s[j][k][4] + acc_s[j][k][5]);
                        scores_sum[1] += (acc_s[j][k][2] + acc_s[j][k][3] + acc_s[j][k][6] + acc_s[j][k][7]);
                    }
                }
                #pragma unroll
                for (size_t j=0; j<2; j++) {
                    #pragma unroll
                    for (size_t k=1; k<4; k*=2) {
                        scores_sum[j] += __shfl_xor_sync(mask, scores_sum[j], k, 4);
                    }
                }

                // logsum = logsum * scores_scale + sum;
                #pragma unroll
                for (size_t j=0; j<2; j++) {
                    logsum[n][j] = logsum[n][j] * scores_scale[j] + scores_sum[j];
                }

                // store P = exp2(QK - m) to smem for the following PV stage
                store_p_smem();

                // acc_o = acc_o * scores_scale
                #pragma unroll
                for (size_t j=0; j<DIM/PV_MMA_N; j++) {
                    #pragma unroll
                    for (size_t k=0; k<PV_MMA_N/16; k++) {
                        acc_o[n][j][k][0] *= scores_scale[0];
                        acc_o[n][j][k][1] *= scores_scale[0];
                        acc_o[n][j][k][4] *= scores_scale[0];
                        acc_o[n][j][k][5] *= scores_scale[0];
                        acc_o[n][j][k][2] *= scores_scale[1];
                        acc_o[n][j][k][3] *= scores_scale[1];
                        acc_o[n][j][k][6] *= scores_scale[1];
                        acc_o[n][j][k][7] *= scores_scale[1];
                    }
                }

            }
        }

        // ==== eiplogue ====
        if (smem_q_i >= NUM_SMEM_Q) { smem_q_i = 0; phase_q ^= 1; }
        fp16 *QAddr = sQ + smem_q_i * 2 * MMA_M * DIM;

        // fill acc_s
        #pragma unroll
        for (size_t j=0; j<BN/QK_MMA_N; j++) {
            #pragma unroll
            for (size_t k=0; k<QK_MMA_N/16; k++) {
                #pragma unroll
                for (size_t l=0; l<8; l++) {
                    acc_s[j][k][l] = 0.0f;
                }
            }
        }

        // gemm-qk
        fp16 *KAddr = sK + smem_k_i * BN * DIM;
        wait(&Qfull[smem_q_i], phase_q);
        wait(&Kfull[smem_k_i], phase_k);
        warpgroup_arrive();
        #pragma unroll
        for (size_t j=0; j<BN/QK_MMA_N; j++) {
            #pragma unroll
            for (size_t k=0; k<DIM; k+=MMA_K) {
                int q_row = 0;
                int q_col = k;
                int k_row = j * QK_MMA_N;
                int k_col = k;
                fp16 *_QAddr = QAddr
                    + wg_idx * MMA_M * DIM
                    + tma_smem_offset_2d<MMA_M>(q_row, q_col);
                fp16 *_KAddr = KAddr + tma_smem_offset_2d<BN>(k_row, k_col);
                wgmma_ss<QK_MMA_N, 1, 1, 1, 0, 0, 16, 1024, true>(acc_s[j], _QAddr, _KAddr);
            }
        }
        warpgroup_commit_batch();
        warpgroup_wait();

        // gemm-pv
        fp16 *VAddr = sV + smem_v_i * BN * DIM;
        wait(&Vfull[smem_v_i], phase_v);
        warpgroup_arrive();
        #pragma unroll
        for (size_t j=0; j<DIM/PV_MMA_N; j++) {
            #pragma unroll
            for (size_t k=0; k<BN; k+=MMA_K) {
                // V is stored in shared as [K=BN, N=DIM].
                // Use TransB=1 so WGMMA consumes it logically as [N, K].
                int v_row = k;
                int v_col = j * PV_MMA_N;
                fp16 *_PAddr = sP
                    + wg_idx * MMA_M * BN
                    + tma_smem_offset_2d<MMA_M>(0, k);
                fp16 *_VAddr = VAddr + tma_smem_offset_2d<BN>(v_row, v_col);
                wgmma_ss_ab<PV_MMA_N, 1, 1, 1, 0, 1,
                    16, 1024, true, BN * 64 * sizeof(fp16), 1024, true>(
                    acc_o[0][j],
                    _PAddr,
                    _VAddr);
            }
        }
        warpgroup_commit_batch();
        warpgroup_wait();

        // max_prev = max
        #pragma unroll
        for (size_t j=0; j<2; j++) {
            scores_max_prev[1][j] = scores_max[1][j];
        }
        // reduce max
        #pragma unroll
        for (size_t j=0; j<2; j++) {
            scores_max[1][j] = -FLT_MAX;
        }
        #pragma unroll
        for (size_t j=0; j<BN/QK_MMA_N; j++) {
            #pragma unroll
            for (size_t k=0; k<QK_MMA_N/16; k++) {
                scores_max[1][0] = max(acc_s[j][k][0], scores_max[1][0]);
                scores_max[1][0] = max(acc_s[j][k][1], scores_max[1][0]);
                scores_max[1][0] = max(acc_s[j][k][4], scores_max[1][0]);
                scores_max[1][0] = max(acc_s[j][k][5], scores_max[1][0]);
                scores_max[1][1] = max(acc_s[j][k][2], scores_max[1][1]);
                scores_max[1][1] = max(acc_s[j][k][3], scores_max[1][1]);
                scores_max[1][1] = max(acc_s[j][k][6], scores_max[1][1]);
                scores_max[1][1] = max(acc_s[j][k][7], scores_max[1][1]);
            }
        }
        #pragma unroll
        for (size_t j=0; j<2; j++) {
            #pragma unroll
            for (size_t k=1; k<4; k*=2) {
                scores_max[1][j] = max(scores_max[1][j], __shfl_xor_sync(mask, scores_max[1][j], k, 4));
            }
        }
        // m = max(pm, m)
        #pragma unroll
        for (size_t j=0; j<2; j++) {
            scores_max[1][j] = max(scores_max_prev[1][j], scores_max[1][j]);
        }
        // scores_scale = exp2(pm  - m)
        #pragma unroll
        for (size_t j=0; j<2; j++) {
            scores_scale[j] = exp2f(scores_max_prev[1][j] * scale - scores_max[1][j] * scale);
        }
        // acc_s = exp2(acc_s - m)
        #pragma unroll
        for (size_t j=0; j<BN/QK_MMA_N; j++) {
            #pragma unroll
            for (size_t k=0; k<QK_MMA_N/16; k++) {
                acc_s[j][k][0] = exp2f(acc_s[j][k][0] * scale - scores_max[1][0] * scale);
                acc_s[j][k][1] = exp2f(acc_s[j][k][1] * scale - scores_max[1][0] * scale);
                acc_s[j][k][4] = exp2f(acc_s[j][k][4] * scale - scores_max[1][0] * scale);
                acc_s[j][k][5] = exp2f(acc_s[j][k][5] * scale - scores_max[1][0] * scale);
                acc_s[j][k][2] = exp2f(acc_s[j][k][2] * scale - scores_max[1][1] * scale);
                acc_s[j][k][3] = exp2f(acc_s[j][k][3] * scale - scores_max[1][1] * scale);
                acc_s[j][k][6] = exp2f(acc_s[j][k][6] * scale - scores_max[1][1] * scale);
                acc_s[j][k][7] = exp2f(acc_s[j][k][7] * scale - scores_max[1][1] * scale);
            }
        }
        // reduce sum
        #pragma unroll
        for (size_t j=0; j<2; j++) {
            scores_sum[j] = 0.0f;
        }
        #pragma unroll
        for (size_t j=0; j<BN/QK_MMA_N; j++) {
            #pragma unroll
            for (size_t k=0; k<QK_MMA_N/16; k++) {
                scores_sum[0] += (acc_s[j][k][0] + acc_s[j][k][1] + acc_s[j][k][4] + acc_s[j][k][5]);
                scores_sum[1] += (acc_s[j][k][2] + acc_s[j][k][3] + acc_s[j][k][6] + acc_s[j][k][7]);
            }
        }
        #pragma unroll
        for (size_t j=0; j<2; j++) {
            #pragma unroll
            for (size_t k=1; k<4; k*=2) {
                scores_sum[j] += __shfl_xor_sync(mask, scores_sum[j], k, 4);
            }
        }
        // logsum = logsum * scores_scale + sum;
        #pragma unroll
        for (size_t j=0; j<2; j++) {
            logsum[1][j] = logsum[1][j] * scores_scale[j] + scores_sum[j];
        }
        // store P = exp2(QK - m) to smem for the following PV stage
        store_p_smem();
        // acc_o = acc_o * scores_scale
        #pragma unroll
        for (size_t j=0; j<DIM/PV_MMA_N; j++) {
            #pragma unroll
            for (size_t k=0; k<PV_MMA_N/16; k++) {
                acc_o[1][j][k][0] *= scores_scale[0];
                acc_o[1][j][k][1] *= scores_scale[0];
                acc_o[1][j][k][4] *= scores_scale[0];
                acc_o[1][j][k][5] *= scores_scale[0];
                acc_o[1][j][k][2] *= scores_scale[1];
                acc_o[1][j][k][3] *= scores_scale[1];
                acc_o[1][j][k][6] *= scores_scale[1];
                acc_o[1][j][k][7] *= scores_scale[1];
            }
        }

        // gemm-pv
        wait(&Vfull[smem_v_i], phase_v);
        warpgroup_arrive();
        #pragma unroll
        for (size_t j=0; j<DIM/PV_MMA_N; j++) {
            #pragma unroll
            for (size_t k=0; k<BN; k+=MMA_K) {
                // V is stored in shared as [K=BN, N=DIM].
                // Use TransB=1 so WGMMA consumes it logically as [N, K].
                int v_row = k;
                int v_col = j * PV_MMA_N;
                fp16 *_PAddr = sP
                    + wg_idx * MMA_M * BN
                    + tma_smem_offset_2d<MMA_M>(0, k);
                fp16 *_VAddr = VAddr + tma_smem_offset_2d<BN>(v_row, v_col);
                wgmma_ss_ab<PV_MMA_N, 1, 1, 1, 0, 1,
                    16, 1024, true, BN * 64 * sizeof(fp16), 1024, true>(
                    acc_o[1][j],
                    _PAddr,
                    _VAddr);
            }
        }
        warpgroup_commit_batch();
        warpgroup_wait();

        // acc_o = acc_o / logsum
        #pragma unroll
        for (size_t i=0; i<BM/(MMA_M*2); i++) {
            float val0 = 1.0f / logsum[i][0];
            float val1 = 1.0f / logsum[i][1];
            #pragma unroll
            for (size_t j=0; j<DIM/PV_MMA_N; j++) {
                #pragma unroll
                for (size_t k=0; k<PV_MMA_N/16; k++) {
                    acc_o[i][j][k][0] *= val0;
                    acc_o[i][j][k][1] *= val0;
                    acc_o[i][j][k][4] *= val0;
                    acc_o[i][j][k][5] *= val0;
                    acc_o[i][j][k][2] *= val1;
                    acc_o[i][j][k][3] *= val1;
                    acc_o[i][j][k][6] *= val1;
                    acc_o[i][j][k][7] *= val1;
                }
            }
        }

        // copy acc_o to sO (load matritx)
        if (tid == 0) {
            tma_store_wait();
        }

        const int lane_row = lane_id & 0xf;
        const int lane_col = (lane_id >> 4) * 8;
        #pragma unroll
        for (size_t i=0; i<BM/(MMA_M*2); i++) {
            #pragma unroll
            for (size_t j=0; j<DIM/PV_MMA_N; j++) {
                #pragma unroll
                for (size_t k=0; k<PV_MMA_N/16; k++) {
                    int o_row = wg_idx * BM/(MMA_M*2) * MMA_M
                                + i * MMA_M
                                + warp_id_in_wg * 16
                                + lane_row;
                    int o_col = j * PV_MMA_N
                                + k * 16
                                + lane_col;
                    fp16 *_sO = sO + tma_smem_swizzle_128b_offset_2d<BM>(o_row, o_col);
                        uint32_t r0 = half2_to_u32(__floats2half2_rn(
                            acc_o[i][j][k][0],
                            acc_o[i][j][k][1]
                        ));
                        uint32_t r1 = half2_to_u32(__floats2half2_rn(
                            acc_o[i][j][k][2],
                            acc_o[i][j][k][3]
                        ));
                        uint32_t r2 = half2_to_u32(__floats2half2_rn(
                            acc_o[i][j][k][4],
                            acc_o[i][j][k][5]
                        ));
                        uint32_t r3 = half2_to_u32(__floats2half2_rn(
                            acc_o[i][j][k][6],
                            acc_o[i][j][k][7]
                        ));

                        stmatrix_x4_reg(_sO, r0, r1, r2, r3);
                }
            }
        }
        // Wait all consumer threads (2 warpgroups) before issuing TMA store from tid==0.
        fence_view_async_shared();
        bar_sync(256, 2);
        // tma store
        if (tid == 0) {
            store_async(&tensorMapO, sO, bs, hn, by * BM, 0);
            tma_store_arrive();
        }
    }
}


template<int BM, int BN, int DIM, int NUM_THREADS, int NUM_SMEM>
__global__  __launch_bounds__(NUM_THREADS) 
void attnWSKCXMergeForNKernel(
    int B, int H, int S, 
    const __grid_constant__ CUtensorMap tensorMapQ, 
    const __grid_constant__ CUtensorMap tensorMapK, 
    const __grid_constant__ CUtensorMap tensorMapV, 
    const __grid_constant__ CUtensorMap tensorMapO
) {
    // WS attention
    const int bs = blockIdx.z;
    const int hn = blockIdx.y;
    const int by = blockIdx.x;
    const int tid = threadIdx.x;
    uint32_t wg_idx = tid >> 7;

    // mma size
    assert((DIM >= 256 || DIM == 16 || DIM == 32 || DIM == 64 || DIM == 128) && "DIM ERROR!");
    assert((BN >= 256 || BN == 16 || BN == 32 || BN == 64 || BN == 128) && "BN ERROR!");
    constexpr int MMA_M = 64;
    constexpr int QK_MMA_N = BN <= 256 ? BN : 256;
    constexpr int PV_MMA_N = DIM <= 256 ? DIM : 256;
    constexpr int MMA_K = 16;
    constexpr int N = BM / 2 / MMA_M;

    // setting shared memory
    extern __shared__ __align__(128) uint8_t smem[];
    SMemWS<BM, BN, DIM, NUM_SMEM> &s = *reinterpret_cast<SMemWS<BM, BN, DIM, NUM_SMEM>*>(smem);
    fp16 *sQ = s.Q, *sK = s.K, *sV = s.V, *sO = s.O;
    Barrier *Qmbar = &s.Qmbar;
    Barrier *Kempty = s.Kempty, *Vempty = s.Vempty, *Kfull = s.Kfull, *Vfull = s.Vfull;

    // init mbarrier
    if (threadIdx.x == 0) {
        init_barrier(Qmbar, 1);
        for (int i = 0; i < NUM_SMEM; ++i) {
            init_barrier(&Kfull[i], 1);  // 1 thread arrive
            init_barrier(&Vfull[i], 1);
            init_barrier(&Kempty[i], 256);  // 256 thread arrive
            init_barrier(&Vempty[i], 256);
        }
    }
    __syncthreads();
    fence_view_async_shared();

    // TMA load
    if (wg_idx == 2) {  // producer
        warpgroup_reg_dealloc<24>();
        if (tid == 256) {
            int smem_i = 0, phase = 0;
            // load Q
            expect_bytes(Qmbar, BM * DIM * sizeof(fp16));
            load_async(sQ, &tensorMapQ, Qmbar, bs, hn, by * BM, 0);
            for (size_t iw=0; iw<S; iw+=BN, ++smem_i) {
                if (smem_i >= NUM_SMEM) { smem_i = 0; phase ^= 1; }
                fp16 *KAddr = sK + smem_i * BN * DIM;
                fp16 *VAddr = sV + smem_i * BN * DIM;
                
                // load K
                wait(&Kempty[smem_i], phase);
                expect_bytes(&Kfull[smem_i], BN * DIM * sizeof(fp16));
                load_async(KAddr, &tensorMapK, &Kfull[smem_i], bs, hn, iw, 0);

                // load V
                wait(&Vempty[smem_i], phase);
                expect_bytes(&Vfull[smem_i], BN * DIM * sizeof(fp16));
                load_async(VAddr, &tensorMapV, &Vfull[smem_i], bs, hn, iw, 0);
            }
        }
    } else {  // consumer
        warpgroup_reg_alloc<240>();
        // Bootstrap empty-smem_i barriers so producer can issue the first K/V loads.
        #pragma unroll
        for (int st = 0; st < NUM_SMEM; ++st) {
            arrive(&Kempty[st]);
            arrive(&Vempty[st]);
        }

        // need args
        uint32_t lane_id = tid & 31;
        uint32_t warp_id_in_wg = (tid >> 5) & 0x3;
        const unsigned mask = __activemask();
        const fp32 scale = sqrt((1.0f / DIM)) * 1.44269504f;  // log2(e)

        // registers define
        fp32 acc_o[N][DIM/PV_MMA_N][PV_MMA_N/16][8];
        fp32 scores_max_prev[N][2];
        fp32 scores_max[N][2];
        fp32 logsum[N][2];

        // init acc_o
        #pragma unroll
        for (size_t i=0; i<N; i++) {
            #pragma unroll
            for (size_t j=0; j<DIM/PV_MMA_N; j++) {
                #pragma unroll
                for (size_t k=0; k<PV_MMA_N/16; k++) {
                    #pragma unroll
                    for (size_t l=0; l<8; l++) {
                        acc_o[i][j][k][l] = 0.0f;
                    }
                }
            }
        }
        
        // init logsum and scores_max
        #pragma unroll
        for (size_t i=0; i<N; i++) {
            #pragma unroll
            for (size_t j=0; j<2; j++) {
                logsum[i][j] = 0.0f;
                scores_max[i][j] = -FLT_MAX;
            }
        }
        
        wait(Qmbar, 0);
        // main for loop
        const size_t NUM_KV_TILES = S / BN;
        for (size_t iter=0; iter<NUM_KV_TILES * N; iter++) {
            const size_t kv_iter = iter / N;
            const size_t n = iter - kv_iter * N;
            const int smem_i = kv_iter % NUM_SMEM;
            const int phase = (kv_iter / NUM_SMEM) & 1;
            fp16 *KAddr = sK + smem_i * BN * DIM;
            fp16 *VAddr = sV + smem_i * BN * DIM;
            
                // fill acc_s
                fp32 acc_s[BN/QK_MMA_N][QK_MMA_N/16][8];
                #pragma unroll
                for (size_t j=0; j<BN/QK_MMA_N; j++) {
                    #pragma unroll
                    for (size_t k=0; k<QK_MMA_N/16; k++) {
                        #pragma unroll
                        for (size_t l=0; l<8; l++) {
                            acc_s[j][k][l] = 0.0f;
                        }
                    }
                }

                // gemm-qk
                wait(&Kfull[smem_i], phase);
                warpgroup_arrive();
                #pragma unroll
                for (size_t j=0; j<BN/QK_MMA_N; j++) {
                    #pragma unroll
                    for (size_t k=0; k<DIM; k+=MMA_K) {
                        int q_row = wg_idx * N * MMA_M + n * MMA_M;
                        int q_col = k;
                        int k_row = j * QK_MMA_N;
                        int k_col = k;
                        fp16 *_QAddr = sQ + tma_smem_offset_2d<BM>(q_row, q_col);
                        fp16 *_KAddr = KAddr + tma_smem_offset_2d<BN>(k_row, k_col);
                        wgmma_ss<QK_MMA_N, 1, 1, 1, 0, 0, 16, 1024, true>(acc_s[j], _QAddr, _KAddr);
                    }
                }
                warpgroup_commit_batch();
                warpgroup_wait();
                if (n == N-1) { arrive(&Kempty[smem_i]); };
                
                // max_prev = max
                #pragma unroll
                for (size_t j=0; j<2; j++) {
                    scores_max_prev[n][j] = scores_max[n][j];
                }

                // reduce max
                #pragma unroll
                for (size_t j=0; j<2; j++) {
                    scores_max[n][j] = -FLT_MAX;
                }
                #pragma unroll
                for (size_t j=0; j<BN/QK_MMA_N; j++) {
                    #pragma unroll
                    for (size_t k=0; k<QK_MMA_N/16; k++) {
                        scores_max[n][0] = max(acc_s[j][k][0], scores_max[n][0]);
                        scores_max[n][0] = max(acc_s[j][k][1], scores_max[n][0]);
                        scores_max[n][0] = max(acc_s[j][k][4], scores_max[n][0]);
                        scores_max[n][0] = max(acc_s[j][k][5], scores_max[n][0]);
                        scores_max[n][1] = max(acc_s[j][k][2], scores_max[n][1]);
                        scores_max[n][1] = max(acc_s[j][k][3], scores_max[n][1]);
                        scores_max[n][1] = max(acc_s[j][k][6], scores_max[n][1]);
                        scores_max[n][1] = max(acc_s[j][k][7], scores_max[n][1]);
                    }
                }
                #pragma unroll
                for (size_t j=0; j<2; j++) {
                    #pragma unroll
                    for (size_t k=1; k<4; k*=2) {
                        scores_max[n][j] = max(scores_max[n][j], __shfl_xor_sync(mask, scores_max[n][j], k, 4));
                    }
                }

                // m = max(pm, m)
                #pragma unroll
                for (size_t j=0; j<2; j++) {
                    scores_max[n][j] = max(scores_max_prev[n][j], scores_max[n][j]);
                }

                // scores_scale = exp2(pm  - m)
                fp32 scores_scale[2];
                #pragma unroll
                for (size_t j=0; j<2; j++) {
                    scores_scale[j] = exp2f(scores_max_prev[n][j] * scale - scores_max[n][j] * scale);
                }

                // acc_s = exp2(acc_s - m)
                #pragma unroll
                for (size_t j=0; j<BN/QK_MMA_N; j++) {
                    #pragma unroll
                    for (size_t k=0; k<QK_MMA_N/16; k++) {
                        acc_s[j][k][0] = exp2f(acc_s[j][k][0] * scale - scores_max[n][0] * scale);
                        acc_s[j][k][1] = exp2f(acc_s[j][k][1] * scale - scores_max[n][0] * scale);
                        acc_s[j][k][4] = exp2f(acc_s[j][k][4] * scale - scores_max[n][0] * scale);
                        acc_s[j][k][5] = exp2f(acc_s[j][k][5] * scale - scores_max[n][0] * scale);
                        acc_s[j][k][2] = exp2f(acc_s[j][k][2] * scale - scores_max[n][1] * scale);
                        acc_s[j][k][3] = exp2f(acc_s[j][k][3] * scale - scores_max[n][1] * scale);
                        acc_s[j][k][6] = exp2f(acc_s[j][k][6] * scale - scores_max[n][1] * scale);
                        acc_s[j][k][7] = exp2f(acc_s[j][k][7] * scale - scores_max[n][1] * scale);
                    }
                }

                // reduce sum
                fp32 scores_sum[2];
                #pragma unroll
                for (size_t j=0; j<2; j++) {
                    scores_sum[j] = 0.0f;
                }
                #pragma unroll
                for (size_t j=0; j<BN/QK_MMA_N; j++) {
                    #pragma unroll
                    for (size_t k=0; k<QK_MMA_N/16; k++) {
                        scores_sum[0] += (acc_s[j][k][0] + acc_s[j][k][1] + acc_s[j][k][4] + acc_s[j][k][5]);
                        scores_sum[1] += (acc_s[j][k][2] + acc_s[j][k][3] + acc_s[j][k][6] + acc_s[j][k][7]);
                    }
                }
                #pragma unroll
                for (size_t j=0; j<2; j++) {
                    #pragma unroll
                    for (size_t k=1; k<4; k*=2) {
                        scores_sum[j] += __shfl_xor_sync(mask, scores_sum[j], k, 4);
                    }
                }

                // logsum = logsum * scores_scale + sum;
                #pragma unroll
                for (size_t j=0; j<2; j++) {
                    logsum[n][j] = logsum[n][j] * scores_scale[j] + scores_sum[j];
                }

                // cast acc_s
                fp16 acc_s_cast[BN/QK_MMA_N][QK_MMA_N/16][8];
                #pragma unroll
                for (size_t j=0; j<BN/QK_MMA_N; j++) {
                    #pragma unroll
                    for (size_t k=0; k<QK_MMA_N/16; k++) {
                        #pragma unroll
                        for (size_t l=0; l<8; l+=2) {
                            uint1 _t2;
                            float2 _t1 = *(float2*)(&acc_s[j][k][l]);
                            *(half2*)(&(_t2)) = __float22half2_rn(*(float2*)(&(_t1)));
                            *(uint1*)(&acc_s_cast[j][k][l]) = _t2;
                        }
                    }
                }

                // acc_o = acc_o * scores_scale
                #pragma unroll
                for (size_t j=0; j<DIM/PV_MMA_N; j++) {
                    #pragma unroll
                    for (size_t k=0; k<PV_MMA_N/16; k++) {
                        acc_o[n][j][k][0] *= scores_scale[0];
                        acc_o[n][j][k][1] *= scores_scale[0];
                        acc_o[n][j][k][4] *= scores_scale[0];
                        acc_o[n][j][k][5] *= scores_scale[0];
                        acc_o[n][j][k][2] *= scores_scale[1];
                        acc_o[n][j][k][3] *= scores_scale[1];
                        acc_o[n][j][k][6] *= scores_scale[1];
                        acc_o[n][j][k][7] *= scores_scale[1];
                    }
                }

                // gemm-pv
                wait(&Vfull[smem_i], phase);
                warpgroup_arrive();
                #pragma unroll
                for (size_t j=0; j<DIM/PV_MMA_N; j++) {
                    #pragma unroll
                    for (size_t k=0; k<BN; k+=MMA_K) {
                        // V is stored in shared as [K=BN, N=DIM].
                        // Use TransB=1 so WGMMA consumes it logically as [N, K].
                        int v_row = k;
                        int v_col = j * PV_MMA_N;
                        const int p_tile_outer = k / QK_MMA_N;
                        const int p_tile_inner = (k % QK_MMA_N) / 16;
                        fp16 *_VAddr = VAddr + tma_smem_offset_2d<BN>(v_row, v_col);
                        wgmma_rs<PV_MMA_N, 1, 1, 1, 0, 1, BN * 64 * sizeof(fp16), 1024, true>(
                            acc_o[n][j],
                            reinterpret_cast<uint32_t*>(acc_s_cast[p_tile_outer][p_tile_inner]),
                            _VAddr);
                    }
                }
                warpgroup_commit_batch();
                warpgroup_wait();
                if (n == N-1) { arrive(&Vempty[smem_i]); };
        }

        // acc_o = acc_o / logsum
        #pragma unroll
        for (size_t i=0; i<BM/(MMA_M*2); i++) {
            float val0 = 1.0f / logsum[i][0];
            float val1 = 1.0f / logsum[i][1];
            #pragma unroll
            for (size_t j=0; j<DIM/PV_MMA_N; j++) {
                #pragma unroll
                for (size_t k=0; k<PV_MMA_N/16; k++) {
                    acc_o[i][j][k][0] *= val0;
                    acc_o[i][j][k][1] *= val0;
                    acc_o[i][j][k][4] *= val0;
                    acc_o[i][j][k][5] *= val0;
                    acc_o[i][j][k][2] *= val1;
                    acc_o[i][j][k][3] *= val1;
                    acc_o[i][j][k][6] *= val1;
                    acc_o[i][j][k][7] *= val1;
                }
            }
        }

        // copy acc_o to sO (load matritx)
        if (tid == 0) {
            tma_store_wait();
        }

        const int lane_row = lane_id & 0xf;
        const int lane_col = (lane_id >> 4) * 8;
        #pragma unroll
        for (size_t i=0; i<BM/(MMA_M*2); i++) {
            #pragma unroll
            for (size_t j=0; j<DIM/PV_MMA_N; j++) {
                #pragma unroll
                for (size_t k=0; k<PV_MMA_N/16; k++) {
                    int o_row = wg_idx * BM/(MMA_M*2) * MMA_M
                                + i * MMA_M
                                + warp_id_in_wg * 16
                                + lane_row;
                    int o_col = j * PV_MMA_N
                                + k * 16
                                + lane_col;
                    // fp16 *_sO = sO + tma_smem_offset_2d<BM>(o_row, o_col);
                    fp16 *_sO = sO + tma_smem_swizzle_128b_offset_2d<BM>(o_row, o_col);
                        uint32_t r0 = half2_to_u32(__floats2half2_rn(
                            acc_o[i][j][k][0],
                            acc_o[i][j][k][1]
                        ));
                        uint32_t r1 = half2_to_u32(__floats2half2_rn(
                            acc_o[i][j][k][2],
                            acc_o[i][j][k][3]
                        ));
                        uint32_t r2 = half2_to_u32(__floats2half2_rn(
                            acc_o[i][j][k][4],
                            acc_o[i][j][k][5]
                        ));
                        uint32_t r3 = half2_to_u32(__floats2half2_rn(
                            acc_o[i][j][k][6],
                            acc_o[i][j][k][7]
                        ));

                        stmatrix_x4_reg(_sO, r0, r1, r2, r3);
                }
            }
        }
        // Wait all consumer threads (2 warpgroups) before issuing TMA store from tid==0.
        fence_view_async_shared();
        bar_sync(256, 2);
        // tma store
        if (tid == 0) {
            store_async(&tensorMapO, sO, bs, hn, by * BM, 0);
            tma_store_arrive();
        }
    }
}



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


template<int B, int H, int S, int D=128, int BM=64, int BN=64, int NUM_THREADS=512, int NUM_SMEM=1, int NUM_CONSUMER=2, int N=2>
void runAttnWSCXKernel(fp16 *Q, fp16 *K, fp16 *V, fp16 *O) {
    CUtensorMap d_tma_map_Q = create_tensor_map<BM, D>(Q, B, H, S, D);
    CUtensorMap d_tma_map_K = create_tensor_map<BN, D>(K, B, H, S, D);
    CUtensorMap d_tma_map_V = create_tensor_map<BN, D>(V, B, H, S, D);
    CUtensorMap d_tma_map_O = create_tensor_map<BM, D>(O, B, H, S, D);

    auto* kernel = attnWSCXKernel<BM, BN, D, NUM_THREADS, NUM_SMEM, NUM_CONSUMER, N>;
    constexpr size_t sMemSize = sizeof(SMemWSCX<BM, BN, D, NUM_SMEM, NUM_CONSUMER>);
    static_assert(sMemSize < 256 * 1024);
    cudaCheck(cudaFuncSetAttribute(
        kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, sMemSize));

    dim3 grid = {static_cast<unsigned int>(S/N/BM), static_cast<unsigned int>(H), static_cast<unsigned int>(B)};
    kernel<<<grid, NUM_THREADS, sMemSize>>>(B, H, S, d_tma_map_Q, d_tma_map_K, d_tma_map_V, d_tma_map_O);
}

template<int B, int H, int S, int D=128, int BM=128, int BN=128, int NUM_THREADS=384, int NUM_SMEM=1>
void runAttnWSKernel(fp16 *Q, fp16 *K, fp16 *V, fp16 *O) {
    CUtensorMap d_tma_map_Q = create_tensor_map<BM, D>(Q, B, H, S, D);
    CUtensorMap d_tma_map_K = create_tensor_map<BN, D>(K, B, H, S, D);
    CUtensorMap d_tma_map_V = create_tensor_map<BN, D>(V, B, H, S, D);
    CUtensorMap d_tma_map_O = create_tensor_map<BM, D>(O, B, H, S, D);

    auto* kernel = attnWSKernel<BM, BN, D, NUM_THREADS, NUM_SMEM>;
    constexpr size_t sMemSize = sizeof(SMemWS<BM, BN, D, NUM_SMEM>);
    static_assert(sMemSize < 256 * 1024);
    cudaCheck(cudaFuncSetAttribute(
        kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, sMemSize));

    dim3 grid = {static_cast<unsigned int>(S/BM), static_cast<unsigned int>(H), static_cast<unsigned int>(B)};
    kernel<<<grid, NUM_THREADS, sMemSize>>>(B, H, S, d_tma_map_Q, d_tma_map_K, d_tma_map_V, d_tma_map_O);
}

template<int B, int H, int S, int D=128, int BM=128, int BN=128, int NUM_THREADS=384, int NUM_SMEM=2>
void runAttnWS2StageKernel(fp16 *Q, fp16 *K, fp16 *V, fp16 *O) {
    CUtensorMap d_tma_map_Q = create_tensor_map<BM, D>(Q, B, H, S, D);
    CUtensorMap d_tma_map_K = create_tensor_map<BN, D>(K, B, H, S, D);
    CUtensorMap d_tma_map_V = create_tensor_map<BN, D>(V, B, H, S, D);
    CUtensorMap d_tma_map_O = create_tensor_map<BM, D>(O, B, H, S, D);
    static_assert(NUM_SMEM >= 2);

    auto* kernel = attnWS2StageKernel<BM, BN, D, NUM_THREADS, NUM_SMEM>;
    constexpr size_t sMemSize = sizeof(SMemWS<BM, BN, D, NUM_SMEM>);
    static_assert(sMemSize < 256 * 1024);
    cudaCheck(cudaFuncSetAttribute(
        kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, sMemSize));

    dim3 grid = {static_cast<unsigned int>(S/BM), static_cast<unsigned int>(H), static_cast<unsigned int>(B)};
    kernel<<<grid, NUM_THREADS, sMemSize>>>(B, H, S, d_tma_map_Q, d_tma_map_K, d_tma_map_V, d_tma_map_O);
}

template<int B, int H, int S, int D=128, int BM=128, int BN=128, int NUM_THREADS=384, int NUM_SMEM=1>
void runAttnWSCXForNKernel(fp16 *Q, fp16 *K, fp16 *V, fp16 *O) {
    CUtensorMap d_tma_map_Q = create_tensor_map<BM, D>(Q, B, H, S, D);
    CUtensorMap d_tma_map_K = create_tensor_map<BN, D>(K, B, H, S, D);
    CUtensorMap d_tma_map_V = create_tensor_map<BN, D>(V, B, H, S, D);
    CUtensorMap d_tma_map_O = create_tensor_map<BM, D>(O, B, H, S, D);

    auto* kernel = attnWSKCXForNKernel<BM, BN, D, NUM_THREADS, NUM_SMEM>;
    constexpr size_t sMemSize = sizeof(SMemWS<BM, BN, D, NUM_SMEM>);
    static_assert(sMemSize < 256 * 1024);
    cudaCheck(cudaFuncSetAttribute(
        kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, sMemSize));

    dim3 grid = {static_cast<unsigned int>(S/BM), static_cast<unsigned int>(H), static_cast<unsigned int>(B)};
    kernel<<<grid, NUM_THREADS, sMemSize>>>(B, H, S, d_tma_map_Q, d_tma_map_K, d_tma_map_V, d_tma_map_O);
}

template<int B, int H, int S, int D=128, int BM=128, int BN=128, int NUM_THREADS=384, int NUM_SMEM=1>
void runWSKCXForNLambdaUnrollKernel(fp16 *Q, fp16 *K, fp16 *V, fp16 *O) {
    CUtensorMap d_tma_map_Q = create_tensor_map<BM, D>(Q, B, H, S, D);
    CUtensorMap d_tma_map_K = create_tensor_map<BN, D>(K, B, H, S, D);
    CUtensorMap d_tma_map_V = create_tensor_map<BN, D>(V, B, H, S, D);
    CUtensorMap d_tma_map_O = create_tensor_map<BM, D>(O, B, H, S, D);

    auto* kernel = attnWSKCXForNLambdaUnrollKernel<BM, BN, D, NUM_THREADS, NUM_SMEM>;
    constexpr size_t sMemSize = sizeof(SMemWS<BM, BN, D, NUM_SMEM>);
    static_assert(sMemSize < 256 * 1024);
    cudaCheck(cudaFuncSetAttribute(
        kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, sMemSize));

    dim3 grid = {static_cast<unsigned int>(S/BM), static_cast<unsigned int>(H), static_cast<unsigned int>(B)};
    kernel<<<grid, NUM_THREADS, sMemSize>>>(B, H, S, d_tma_map_Q, d_tma_map_K, d_tma_map_V, d_tma_map_O);
}

template<int B, int H, int S, int D=128, int BM=128, int BN=128, int NUM_THREADS=384, int NUM_SMEM=1>
void runWSKCXForNManualUnrollKernel(fp16 *Q, fp16 *K, fp16 *V, fp16 *O) {
    CUtensorMap d_tma_map_Q = create_tensor_map<BM, D>(Q, B, H, S, D);
    CUtensorMap d_tma_map_K = create_tensor_map<BN, D>(K, B, H, S, D);
    CUtensorMap d_tma_map_V = create_tensor_map<BN, D>(V, B, H, S, D);
    CUtensorMap d_tma_map_O = create_tensor_map<BM, D>(O, B, H, S, D);

    auto* kernel = attnWSKCXForNManualUnrollKernel<BM, BN, D, NUM_THREADS, NUM_SMEM>;
    constexpr size_t sMemSize = sizeof(SMemWS<BM, BN, D, NUM_SMEM>);
    static_assert(sMemSize < 256 * 1024);
    cudaCheck(cudaFuncSetAttribute(
        kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, sMemSize));

    dim3 grid = {static_cast<unsigned int>(S/BM), static_cast<unsigned int>(H), static_cast<unsigned int>(B)};
    kernel<<<grid, NUM_THREADS, sMemSize>>>(B, H, S, d_tma_map_Q, d_tma_map_K, d_tma_map_V, d_tma_map_O);
}

template<int B, int H, int S, int D=128, int BM=128, int BN=128, int NUM_THREADS=384, int NUM_SMEM_Q=1, int NUM_SMEM_KV=1>
void runWSKCXForNUnrollDoubleQKernel(fp16 *Q, fp16 *K, fp16 *V, fp16 *O) {
    constexpr int MMA_M = 64;
    CUtensorMap d_tma_map_Q = create_tensor_map<MMA_M, D>(Q, B, H, S, D);
    CUtensorMap d_tma_map_K = create_tensor_map<BN, D>(K, B, H, S, D);
    CUtensorMap d_tma_map_V = create_tensor_map<BN, D>(V, B, H, S, D);
    CUtensorMap d_tma_map_O = create_tensor_map<BM, D>(O, B, H, S, D);

    auto* kernel = attnWSKCXForNUnrollDoubleQKernel<BM, BN, D, NUM_THREADS, NUM_SMEM_Q, NUM_SMEM_KV>;
    constexpr size_t sMemSize = sizeof(SMemWSDoubleQ<BM, BN, D, NUM_SMEM_Q, NUM_SMEM_KV>);
    static_assert(sMemSize < 256 * 1024);
    cudaCheck(cudaFuncSetAttribute(
        kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, sMemSize));

    dim3 grid = {static_cast<unsigned int>(S/BM), static_cast<unsigned int>(H), static_cast<unsigned int>(B)};
    kernel<<<grid, NUM_THREADS, sMemSize>>>(B, H, S, d_tma_map_Q, d_tma_map_K, d_tma_map_V, d_tma_map_O);
}

template<int B, int H, int S, int D=128, int BM=128, int BN=128, int NUM_THREADS=384, int NUM_SMEM_Q=1, int NUM_SMEM_KV=1>
void runWSKCXForNUnrollDoubleQPingpongKernel(fp16 *Q, fp16 *K, fp16 *V, fp16 *O) {
    constexpr int MMA_M = 64;
    CUtensorMap d_tma_map_Q = create_tensor_map<MMA_M, D>(Q, B, H, S, D);
    CUtensorMap d_tma_map_K = create_tensor_map<BN, D>(K, B, H, S, D);
    CUtensorMap d_tma_map_V = create_tensor_map<BN, D>(V, B, H, S, D);
    CUtensorMap d_tma_map_O = create_tensor_map<BM, D>(O, B, H, S, D);

    auto* kernel = attnWSKCXForNUnrollDoubleQPingpongKernel<BM, BN, D, NUM_THREADS, NUM_SMEM_Q, NUM_SMEM_KV>;
    constexpr size_t sMemSize = sizeof(SMemWSDoubleQ<BM, BN, D, NUM_SMEM_Q, NUM_SMEM_KV>);
    static_assert(sMemSize < 256 * 1024);
    cudaCheck(cudaFuncSetAttribute(
        kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, sMemSize));

    dim3 grid = {static_cast<unsigned int>(S/BM), static_cast<unsigned int>(H), static_cast<unsigned int>(B)};
    kernel<<<grid, NUM_THREADS, sMemSize>>>(B, H, S, d_tma_map_Q, d_tma_map_K, d_tma_map_V, d_tma_map_O);
}

template<int B, int H, int S, int D=128, int BM=128, int BN=128, int NUM_THREADS=384, int NUM_SMEM_Q=1, int NUM_SMEM_KV=2>
void runWSKCXForNUnrollDoubleQ2StageKernel(fp16 *Q, fp16 *K, fp16 *V, fp16 *O) {
    constexpr int MMA_M = 64;
    CUtensorMap d_tma_map_Q = create_tensor_map<MMA_M, D>(Q, B, H, S, D);
    CUtensorMap d_tma_map_K = create_tensor_map<BN, D>(K, B, H, S, D);
    CUtensorMap d_tma_map_V = create_tensor_map<BN, D>(V, B, H, S, D);
    CUtensorMap d_tma_map_O = create_tensor_map<BM, D>(O, B, H, S, D);
    static_assert(NUM_SMEM_KV >= 2);

    auto* kernel = attnWSKCXForNUnrollDoubleQ2StageKernel<BM, BN, D, NUM_THREADS, NUM_SMEM_Q, NUM_SMEM_KV>;
    constexpr size_t sMemSize = sizeof(SMemWSDoubleQ<BM, BN, D, NUM_SMEM_Q, NUM_SMEM_KV>);
    static_assert(sMemSize < 256 * 1024);
    cudaCheck(cudaFuncSetAttribute(
        kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, sMemSize));

    dim3 grid = {static_cast<unsigned int>(S/BM), static_cast<unsigned int>(H), static_cast<unsigned int>(B)};
    kernel<<<grid, NUM_THREADS, sMemSize>>>(B, H, S, d_tma_map_Q, d_tma_map_K, d_tma_map_V, d_tma_map_O);
}

template<int B, int H, int S, int D=128, int BM=128, int BN=128, int NUM_THREADS=384, int NUM_SMEM_Q=1, int NUM_SMEM_KV=2>
void runWSKCXForNUnrollDoubleQ2StagePVSSKernel(fp16 *Q, fp16 *K, fp16 *V, fp16 *O) {
    constexpr int MMA_M = 64;
    CUtensorMap d_tma_map_Q = create_tensor_map<MMA_M, D>(Q, B, H, S, D);
    CUtensorMap d_tma_map_K = create_tensor_map<BN, D>(K, B, H, S, D);
    CUtensorMap d_tma_map_V = create_tensor_map<BN, D>(V, B, H, S, D);
    CUtensorMap d_tma_map_O = create_tensor_map<BM, D>(O, B, H, S, D);
    static_assert(NUM_SMEM_KV >= 2);

    auto* kernel = attnWSKCXForNUnrollDoubleQ2StagePVSSKernel<BM, BN, D, NUM_THREADS, NUM_SMEM_Q, NUM_SMEM_KV>;
    constexpr size_t sMemSize = sizeof(SMemWSDoubleQPVSS<BM, BN, D, NUM_SMEM_Q, NUM_SMEM_KV>);
    static_assert(sMemSize < 256 * 1024);
    cudaCheck(cudaFuncSetAttribute(
        kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, sMemSize));

    dim3 grid = {static_cast<unsigned int>(S/BM), static_cast<unsigned int>(H), static_cast<unsigned int>(B)};
    kernel<<<grid, NUM_THREADS, sMemSize>>>(B, H, S, d_tma_map_Q, d_tma_map_K, d_tma_map_V, d_tma_map_O);
}

template<int B, int H, int S, int D=128, int BM=128, int BN=128, int NUM_THREADS=384, int NUM_SMEM=1>
void runWSKCXMergeForNKernel(fp16 *Q, fp16 *K, fp16 *V, fp16 *O) {
    CUtensorMap d_tma_map_Q = create_tensor_map<BM, D>(Q, B, H, S, D);
    CUtensorMap d_tma_map_K = create_tensor_map<BN, D>(K, B, H, S, D);
    CUtensorMap d_tma_map_V = create_tensor_map<BN, D>(V, B, H, S, D);
    CUtensorMap d_tma_map_O = create_tensor_map<BM, D>(O, B, H, S, D);

    auto* kernel = attnWSKCXMergeForNKernel<BM, BN, D, NUM_THREADS, NUM_SMEM>;
    constexpr size_t sMemSize = sizeof(SMemWS<BM, BN, D, NUM_SMEM>);
    static_assert(sMemSize < 256 * 1024);
    cudaCheck(cudaFuncSetAttribute(
        kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, sMemSize));

    dim3 grid = {static_cast<unsigned int>(S/BM), static_cast<unsigned int>(H), static_cast<unsigned int>(B)};
    kernel<<<grid, NUM_THREADS, sMemSize>>>(B, H, S, d_tma_map_Q, d_tma_map_K, d_tma_map_V, d_tma_map_O);
}


using AttnRunner = void (*)(fp16*, fp16*, fp16*, fp16*);

template<int B, int H, int S, int D, int BN_TILE=128>
void verify_attn(AttnRunner kernel) {

    int dev = 0;
    cudaDeviceProp prop{};
    cudaCheck(cudaGetDevice(&dev));
    cudaCheck(cudaGetDeviceProperties(&prop, dev));
    if (prop.major < 9) {
        printf("This kernel uses Hopper-only instructions (TMA/WGMMA). Need SM90+ GPU.\n");
        return;
    }

    const size_t numel = static_cast<size_t>(B) * H * S * D;
    const size_t bytes = numel * sizeof(fp16);
    std::vector<fp16> hQ(numel), hK(numel), hV(numel), hO(numel), hORef(numel, 0.0f);
    std::vector<fp32> hSRef(static_cast<size_t>(B) * H * S * S,  0.0f);

    // std::random_device rd;
    const uint32_t seed = 42;
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> qk_dist(-0.5f, 0.5f);
    std::uniform_real_distribution<float> v_dist(-1.0f, 1.0f);
    printf("verify_attn seed=%u\n", seed);

    auto idx4 = [=](int b, int h, int s, int d) {
        return (((static_cast<size_t>(b) * H + h) * S + s) * D + d);
    };
    auto idxS = [=](int b, int h, int s1, int s2) {
        return (((static_cast<size_t>(b) * H + h) * S + s1) * S + s2);
    };

    for (int b = 0; b < B; ++b) {
        for (int h = 0; h < H; ++h) {
            for (int s = 0; s < S; ++s) {
                for (int d = 0; d < D; ++d) {
                    const size_t idx = idx4(b, h, s, d);
                    hQ[idx] = __float2half_rn(qk_dist(gen));
                    hK[idx] = __float2half_rn(qk_dist(gen));
                    hV[idx] = __float2half_rn(v_dist(gen));
                }
            }
        }
    }

    fp16 *dQ = nullptr;
    fp16 *dK = nullptr;
    fp16 *dV = nullptr;
    fp16 *dO = nullptr;
    cudaCheck(cudaMalloc(&dQ, bytes));
    cudaCheck(cudaMalloc(&dK, bytes));
    cudaCheck(cudaMalloc(&dV, bytes));
    cudaCheck(cudaMalloc(&dO, bytes));
    cudaCheck(cudaMemcpy(dQ, hQ.data(), bytes, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(dK, hK.data(), bytes, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(dV, hV.data(), bytes, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemset(dO, 0, bytes));

    kernel(dQ, dK, dV, dO);

    cudaCheck(cudaGetLastError());
    cudaCheck(cudaDeviceSynchronize());
    cudaCheck(cudaMemcpy(hO.data(), dO, bytes, cudaMemcpyDeviceToHost));

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
                    const int tile_end = (iw + BN_TILE < S) ? (iw + BN_TILE) : S;
                    float tile_max = -FLT_MAX;

                    for (int s2 = iw; s2 < tile_end; ++s2) {
                        float score = 0.0f;
                        for (int d = 0; d < D; ++d) {
                            score += __half2float(hQ[idx4(b, h, s1, d)])
                                   * __half2float(hK[idx4(b, h, s2, d)]);
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
                            row_acc[d] += prob_for_pv * __half2float(hV[idx4(b, h, s2, d)]);
                        }
                    }

                    row_sum = row_sum * old_scale + tile_sum;
                    row_max = new_max;
                }

                const float inv_sum = 1.0f / row_sum;
                for (int d = 0; d < D; ++d) {
                    hORef[idx4(b, h, s1, d)] = __float2half_rn(row_acc[d] * inv_sum);
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
                    const float got = __half2float(hO[idx4(b, h, s, d)]);
                    const float ref = __half2float(hORef[idx4(b, h, s, d)]);
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
           __half2float(hO[idx4(max_o_b, max_o_h, max_o_s, max_o_d)]),
           __half2float(hORef[idx4(max_o_b, max_o_h, max_o_s, max_o_d)]),
           o_abs_tol, o_mismatch_count, static_cast<size_t>(B) * H * S * D,
           left_mismatch_count, right_mismatch_count);

    cudaCheck(cudaFree(dQ));
    cudaCheck(cudaFree(dK));
    cudaCheck(cudaFree(dV));
    cudaCheck(cudaFree(dO));
}

template<int B, int H, int S, int D, int BM, int BN>
void benchmark_attn(AttnRunner kernel) {
    const size_t numel = static_cast<size_t>(B) * H * S * D;
    const size_t bytes = numel * sizeof(fp16);

    fp16 *dQ = nullptr;
    fp16 *dK = nullptr;
    fp16 *dV = nullptr;
    fp16 *dO = nullptr;

    cudaCheck(cudaMalloc(&dQ, bytes));
    cudaCheck(cudaMalloc(&dK, bytes));
    cudaCheck(cudaMalloc(&dV, bytes));
    cudaCheck(cudaMalloc(&dO, bytes));

    cudaCheck(cudaMemset(dQ, 0, bytes));
    cudaCheck(cudaMemset(dK, 0, bytes));
    cudaCheck(cudaMemset(dV, 0, bytes));
    cudaCheck(cudaMemset(dO, 0, bytes));

    for (int i = 0; i < 100; ++i) {
        kernel(dQ, dK, dV, dO);
        cudaCheck(cudaGetLastError());
    }
    cudaCheck(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    cudaCheck(cudaEventCreate(&start));
    cudaCheck(cudaEventCreate(&stop));

    cudaCheck(cudaEventRecord(start));
    for (int i = 0; i < 500; ++i) {
        kernel(dQ, dK, dV, dO);
        cudaCheck(cudaGetLastError());
    }
    cudaCheck(cudaEventRecord(stop));
    cudaCheck(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    cudaCheck(cudaEventElapsedTime(&elapsed_ms, start, stop));

    cudaCheck(cudaEventDestroy(start));
    cudaCheck(cudaEventDestroy(stop));

    const double avg_ms = static_cast<double>(elapsed_ms) / 500;
    const double flops = 4.0 * static_cast<double>(B) * H * S * S * D;
    const double tflops = flops / (avg_ms * 1.0e-3) / 1.0e12;

    printf("%d, %d, %d, %d, %d, %d, %.6f, %.3f, ok\n", B, H, S, D, BM, BN, avg_ms, tflops);

    cudaCheck(cudaFree(dQ));
    cudaCheck(cudaFree(dK));
    cudaCheck(cudaFree(dV));
    cudaCheck(cudaFree(dO));
}

template<int B, int H, int S, int D, int BM, int BN>
void benchmark_attn_ncu(AttnRunner kernel) {
    const size_t numel = static_cast<size_t>(B) * H * S * D;
    const size_t bytes = numel * sizeof(fp16);

    fp16 *dQ = nullptr;
    fp16 *dK = nullptr;
    fp16 *dV = nullptr;
    fp16 *dO = nullptr;

    cudaCheck(cudaMalloc(&dQ, bytes));
    cudaCheck(cudaMalloc(&dK, bytes));
    cudaCheck(cudaMalloc(&dV, bytes));
    cudaCheck(cudaMalloc(&dO, bytes));

    cudaCheck(cudaMemset(dQ, 0, bytes));
    cudaCheck(cudaMemset(dK, 0, bytes));
    cudaCheck(cudaMemset(dV, 0, bytes));
    cudaCheck(cudaMemset(dO, 0, bytes));

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


// export CUDA_VISIBLE_DEVICES=1
// nvcc -std=c++17 -arch=sm_90a -O3 attn_wscx.cu -o attn_wscx_test -lcuda -Xptxas=-v
// ncu --set full --launch-skip 100 --launch-count 1 ./attn_wscx_test
int main() {
    constexpr int B = 1;
    constexpr int H = 16;
    constexpr int S = 114*256;
    constexpr int D = 64;
    constexpr int BM = 128;
    constexpr int BN = 256;
    // auto *kernel1 = runAttnWSCXKernel<B, H, S, D, 64, BN, /*num_thread*/512, /*num_smem*/1, /*num_consumer*/2, /*N*/2>;
    // auto *kernel2 = runAttnWSKernel<B, H, S, D, BM, BN, 384, 1>;
    auto *kernel9 = runAttnWS2StageKernel<B, H, S, D, BM, BN, 384, 2>;

    // auto *kernel3 = runAttnWSCXForNKernel<B, H, S, D, BM, BN, 384, 1>;  // 368.472
    // auto *kernel4 = runWSKCXForNLambdaUnrollKernel<B, H, S, D, BM, BN, 384, 1>;  // 312.091
    // auto *kernel5 = runWSKCXForNManualUnrollKernel<B, H, S, D, BM, BN, 384, 1>;  // 310.572

    // auto *kernel6 = runWSKCXForNUnrollDoubleQKernel<B, H, S, D, BM, BN, 384, 1, 1>;  // unroll
    // auto *kernel7 = runWSKCXMergeForNKernel<B, H, S, D, BM, BN, 384, 1>;

    // auto *kernel8 = runWSKCXForNUnrollDoubleQPingpongKernel<B, H, S, D, BM, BN, 384, 1, 1>;
    // auto *kernel9 = runWSKCXForNUnrollDoubleQ2StageKernel<B, H, S, D, BM, BN, 384, 1, 2>;
    // auto *kernel10 = runWSKCXForNUnrollDoubleQ2StagePVSSKernel<B, H, S, D, BM, BN, 384, 1, 2>;

    // verify_attn<B, H, S, D, BN>(kernel5);

    // benchmark_attn<B, H, S, D, BM, BN>(kernel2);
    // benchmark_attn<B, H, S, D, BM, BN>(kernel3);
    // benchmark_attn<B, H, S, D, BM, BN>(kernel4);
    // benchmark_attn<B, H, S, D, BM, BN>(kernel5);
    // benchmark_attn<B, H, S, D, BM, BN>(kernel6);
    // benchmark_attn<B, H, S, D, BM, BN>(kernel7);
    // benchmark_attn<B, H, S, D, BM, BN>(kernel8);
    benchmark_attn<B, H, S, D, BM, BN>(kernel9);

    // benchmark_attn_ncu<B, H, S, D, BM, BN>(kernel3);
    // benchmark_attn_ncu<B, H, S, D, BM, BN>(kernel4);
    // benchmark_attn_ncu<B, H, S, D, BM, BN>(kernel5);
    // benchmark_attn_ncu<B, H, S, D, BM, BN>(kernel7);
    return 0;
}
