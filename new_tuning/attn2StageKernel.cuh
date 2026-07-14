#pragma once

#include "attnKernelCommon.cuh"

template<
    int BM, int BN, int DIM, int NUM_THREADS, int NUM_SMEM,
    uint32_t PRODUCER_REG_DEALLOC = 24,
    uint32_t CONSUMER_REG_ALLOC = 240,
    int P_SMEM_K_TILES = (BN / 16) / 2,
    int Q_REG_K_TILES = (DIM / 16) / 2
>
__global__  __launch_bounds__(NUM_THREADS) 
void attnWS2StageKernel(
    int B, int H, int S, const fp16 *Q,
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
    // static_assert((DIM >= 256 || DIM == 16 || DIM == 32 || DIM == 64 || DIM == 128) && "DIM ERROR!");
    // static_assert((BN >= 256 || BN == 16 || BN == 32 || BN == 64 || BN == 128 || BN == 96) && "BN ERROR!");
    constexpr int MMA_M = 64;
    constexpr int QK_MMA_N = BN <= 256 ? BN : 256;
    constexpr int PV_MMA_N = DIM <= 256 ? DIM : 256;
    constexpr int MMA_K = 16;

    constexpr int P_TOTAL_TILES = BN / MMA_K;
    constexpr int Q_TOTAL_TILES = DIM / MMA_K;
    constexpr int Q_SMEM_K_TILES = Q_TOTAL_TILES - Q_REG_K_TILES;
    constexpr int Q_SMEM_COLS = Q_SMEM_K_TILES * MMA_K;

    constexpr int P_REG_TILES = P_TOTAL_TILES - P_SMEM_K_TILES;
    constexpr int Q_REG_TILES = Q_REG_K_TILES;
    
    constexpr int P_REG_STORAGE_TILES = P_REG_TILES > 0 ? P_REG_TILES : 1;
    constexpr int Q_REG_STORAGE_TILES = Q_REG_TILES > 0 ? Q_REG_TILES : 1;

    static_assert(PRODUCER_REG_DEALLOC % 8 == 0);
    static_assert(CONSUMER_REG_ALLOC % 8 == 0);
    static_assert(BN % MMA_K == 0);
    static_assert(DIM % MMA_K == 0);
    static_assert(P_SMEM_K_TILES >= 0);
    static_assert(P_SMEM_K_TILES <= P_TOTAL_TILES);
    static_assert(Q_REG_K_TILES >= 0);
    static_assert(Q_REG_K_TILES <= Q_TOTAL_TILES);
    static_assert(Q_SMEM_K_TILES == 0 || Q_SMEM_K_TILES % 4 == 0,
                  "Q shared-memory TMA width must be a multiple of 64 elements");

    // setting shared memory
    extern __shared__ __align__(128) uint8_t smem[];
    SMemWS<BM, BN, DIM, NUM_SMEM, P_SMEM_K_TILES, Q_REG_K_TILES> &s =
        *reinterpret_cast<SMemWS<BM, BN, DIM, NUM_SMEM, P_SMEM_K_TILES, Q_REG_K_TILES>*>(smem);
    fp16 *sQ = s.Q, *sK = s.K, *sV = s.V, *sP = s.P, *sO = s.O;
    Barrier *Qmbar = &s.Qmbar;
    Barrier *Kempty = s.Kempty, *Vempty = s.Vempty, *Kfull = s.Kfull, *Vfull = s.Vfull;

    // init mbarrier
    if (threadIdx.x == 0) {
        init_barrier(Qmbar, 1);
        #pragma unroll
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
        warpgroup_reg_dealloc<PRODUCER_REG_DEALLOC>();
        if (tid == 256) {
            size_t smem_i = 0;
            int phase = 0;
            // load Q
            if constexpr (Q_SMEM_K_TILES > 0) {
                expect_bytes(Qmbar, BM * Q_SMEM_COLS * sizeof(fp16));
                load_async(sQ, &tensorMapQ, Qmbar, bs, hn, by * BM, 0);
            }

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
        warpgroup_reg_alloc<CONSUMER_REG_ALLOC>();
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
        fp16 acc_s_cast[BM/(MMA_M*2)][P_REG_STORAGE_TILES][8];
        uint32_t q_regs[BM/(MMA_M*2)][Q_REG_STORAGE_TILES][4];

        const int SQO = ((S + BM - 1) / BM) * BM;
        if constexpr (Q_REG_TILES > 0) {
            #pragma unroll
            for (size_t i=0; i<BM/(MMA_M*2); i++) {
                #pragma unroll
                for (size_t qt=0; qt<Q_REG_TILES; qt++) {
                    const int q_row = by * BM + wg_idx * BM/(MMA_M*2) * MMA_M + i * MMA_M;
                    const int q_col = Q_SMEM_COLS + qt * MMA_K;
                    load_q_global_fragment<DIM>(q_regs[i][qt], Q, H, SQO, bs, hn, q_row, q_col);
                }
            }
        }

        auto __ldsmemP = [&]() {
            // Keep the first P_SMEM_K_TILES k16 tiles of P in smem and the rest in registers.
            const int p_lane_row = lane_id & 0xf;
            const int p_lane_col = (lane_id >> 4) * 8;
            #pragma unroll
            for (size_t i=0; i<BM/(MMA_M*2); i++) {
                #pragma unroll
                for (size_t j=0; j<BN/QK_MMA_N; j++) {
                    #pragma unroll
                    for (size_t k=0; k<QK_MMA_N/16; k++) {
                        constexpr int k_tiles_per_j = QK_MMA_N / MMA_K;
                        const int p_tile = j * k_tiles_per_j + k;
                        if (p_tile < P_SMEM_K_TILES) {
                            int p_row = wg_idx * BM/(MMA_M*2) * MMA_M
                                      + i * MMA_M
                                      + warp_id_in_wg * 16
                                      + p_lane_row;
                            int p_col = p_tile * MMA_K + p_lane_col;
                            fp16 *_sP = sP + tma_smem_swizzle_128b_offset_2d<BM>(p_row, p_col);
                            uint32_t r0 = half2_to_u32(__floats2half2_rn(
                                acc_s[i][j][k][0],
                                acc_s[i][j][k][1]
                            ));
                            uint32_t r1 = half2_to_u32(__floats2half2_rn(
                                acc_s[i][j][k][2],
                                acc_s[i][j][k][3]
                            ));
                            uint32_t r2 = half2_to_u32(__floats2half2_rn(
                                acc_s[i][j][k][4],
                                acc_s[i][j][k][5]
                            ));
                            uint32_t r3 = half2_to_u32(__floats2half2_rn(
                                acc_s[i][j][k][6],
                                acc_s[i][j][k][7]
                            ));
                            stmatrix_x4_reg(_sP, r0, r1, r2, r3);
                        } else {
                            const int reg_tile = p_tile - P_SMEM_K_TILES;
                            #pragma unroll
                            for (size_t l=0; l<8; l+=2) {
                                uint1 _t2;
                                float2 _t1 = *(float2*)(&acc_s[i][j][k][l]);
                                *(half2*)(&(_t2)) = __float22half2_rn(*(float2*)(&(_t1)));
                                *(uint1*)(&acc_s_cast[i][reg_tile][l]) = _t2;
                            }
                        }
                    }
                }
            }
        };

        auto __softmax = [&](int n_block) {
            // Mask the logical KV tail after QK and before softmax. Zero-filled
            // K alone is insufficient because exp(0) would change the denominator.
            if ((n_block + 1) * BN > S) {
                #pragma unroll
                for (size_t i=0; i<BM/(MMA_M*2); i++) {
                    #pragma unroll
                    for (size_t j=0; j<BN/QK_MMA_N; j++) {
                        #pragma unroll
                        for (size_t k=0; k<QK_MMA_N/16; k++) {
                            #pragma unroll
                            for (size_t l=0; l<8; l++) {
                                const int col_in_mma = (lane_id & 3) * 2 + (l >= 4 ? 8 : 0) + (l & 1);
                                const int col = n_block * BN + j * QK_MMA_N + k * 16 + col_in_mma;
                                if (col >= S) { acc_s[i][j][k][l] = -FLT_MAX; }
                            }
                        }
                    }
                }
            }
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
            
            __ldsmemP();
            
            // scores_scale = exp2(pm  - m)
            fp32 scores_scale[BM/(MMA_M*2)][2];
            #pragma unroll
            for (size_t i=0; i<BM/(MMA_M*2); i++) {
                #pragma unroll
                for (size_t j=0; j<2; j++) {
                    scores_scale[i][j] = exp2f(scores_max_prev[i][j] * scale - scores_max[i][j] * scale);
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

        auto __gemm_qk = [&](int k_smem_i, int k_phase) {
            fp16 *KAddr = sK + k_smem_i * BN * DIM;

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

            wait(&Kfull[k_smem_i], k_phase);
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
                        fp16 *_KAddr = KAddr + tma_smem_offset_2d<BN>(k_row, k_col);
                        if constexpr (Q_REG_TILES == 0) {
                            fp16 *_QAddr = sQ + tma_smem_offset_2d<BM>(q_row, q_col);
                            wgmma_ss<QK_MMA_N, 1, 1, 1, 0, 0, 16, 1024, true>(acc_s[i][j], _QAddr, _KAddr);
                        } else if constexpr (Q_SMEM_K_TILES == 0) {
                            const int q_reg_tile = k / MMA_K - Q_SMEM_K_TILES;
                            wgmma_rs<QK_MMA_N, 1, 1, 1, 0, 0, 16, 1024, true>(
                                acc_s[i][j], q_regs[i][q_reg_tile], _KAddr);
                        } else if (k < Q_SMEM_COLS) {
                            fp16 *_QAddr = sQ + tma_smem_offset_2d<BM>(q_row, q_col);
                            wgmma_ss<QK_MMA_N, 1, 1, 1, 0, 0, 16, 1024, true>(acc_s[i][j], _QAddr, _KAddr);
                        } else {
                            const int q_reg_tile = k / MMA_K - Q_SMEM_K_TILES;
                            wgmma_rs<QK_MMA_N, 1, 1, 1, 0, 0, 16, 1024, true>(
                                acc_s[i][j], q_regs[i][q_reg_tile], _KAddr);
                        }
                    }
                }
            }
            warpgroup_commit_batch();
            warpgroup_wait();
            arrive(&Kempty[k_smem_i]);
        };

        auto __gemm_pv = [&](int v_smem_i, int v_phase) {
            fp16 *VAddr = sV + v_smem_i * BN * DIM;

            wait(&Vfull[v_smem_i], v_phase);
            __syncwarp();
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
                        fp16 *_VAddr = VAddr + tma_smem_offset_2d<BN>(v_row, v_col);
                        const int p_tile = k / MMA_K;
                        if (p_tile < P_SMEM_K_TILES) {
                            int p_row = wg_idx * BM/(MMA_M*2) * MMA_M + i * MMA_M;
                            fp16 *_PAddr = sP + tma_smem_offset_2d<BM>(p_row, k);
                            wgmma_ss_ab<PV_MMA_N, 1, 1, 1, 0, 1,
                                MMA_K * sizeof(fp16), 1024, true,
                                BN * 64 * sizeof(fp16), 1024, true>(
                                acc_o[i][j],
                                _PAddr,
                                _VAddr);
                        } else {
                            const int reg_tile = p_tile - P_SMEM_K_TILES;
                            wgmma_rs<PV_MMA_N, 1, 1, 1, 0, 1, BN * 64 * sizeof(fp16), 1024, true>(
                                acc_o[i][j],
                                reinterpret_cast<uint32_t*>(acc_s_cast[i][reg_tile]),
                                _VAddr);
                        }
                    }
                }
            }
            warpgroup_commit_batch();
            warpgroup_wait();
            arrive(&Vempty[v_smem_i]);
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
        
        if constexpr (Q_SMEM_K_TILES > 0) { wait(Qmbar, 0); }
        int smem_i = 0, phase = 0;

        // ===== prologue =====
        __gemm_qk(smem_i, phase);
        __softmax(0);

        // ==== main loop ====
        smem_i++;
        for (size_t iw=BN; iw<S; iw+=BN, ++smem_i) {
            if (smem_i >= NUM_SMEM) { smem_i = 0; phase ^= 1; }
            // gemm-qk
            __gemm_qk(smem_i, phase);

            // prev gemm-pv
            const int prev_smem_i = (smem_i + NUM_SMEM -1) % NUM_SMEM;
            int prve_phase = phase;
            if (prev_smem_i == NUM_SMEM -1) { prve_phase ^= 1; }
            __gemm_pv(prev_smem_i, prve_phase);

            // softmax
            __softmax(iw / BN);
        }

        // ==== epilogue ====
        const int last_smem_i = (smem_i % NUM_SMEM + NUM_SMEM - 1) % NUM_SMEM;
        const int last_phase = phase;
        // last gemm-pv
        __gemm_pv(last_smem_i, last_phase);

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



template<
    int B, int H, int S, int D=128, int BM=128, int BN=128, int NUM_SMEM=2,
    uint32_t PRODUCER_REG_DEALLOC = 24,
    uint32_t CONSUMER_REG_ALLOC = 240,
    int P_SMEM_K_TILES = (BN / 16) / 2,
    int Q_REG_K_TILES = (D / 16) / 2,
    int NUM_THREADS=384
>
void runAttnWS2StageKernel(fp16 *Q, fp16 *K, fp16 *V, fp16 *O) {
    constexpr int SQO = round_up_to_tile_v<S, BM>;
    constexpr int SKV = round_up_to_tile_v<S, BN>;
    constexpr int Q_SMEM_COLS = D - Q_REG_K_TILES * 16;
    CUtensorMap d_tma_map_Q;
    if constexpr (Q_SMEM_COLS > 0) {
        d_tma_map_Q = create_tensor_map<BM, Q_SMEM_COLS>(Q, B, H, SQO, D);
    } else {
        // Unused dummy map for the all-RS configuration.
        d_tma_map_Q = create_tensor_map<BM, 64>(Q, B, H, SQO, D);
    }
    CUtensorMap d_tma_map_K = create_tensor_map<BN, D>(K, B, H, SKV, D);
    CUtensorMap d_tma_map_V = create_tensor_map<BN, D>(V, B, H, SKV, D);
    CUtensorMap d_tma_map_O = create_tensor_map<BM, D>(O, B, H, SQO, D);

    auto* kernel = attnWS2StageKernel<BM, BN, D, NUM_THREADS, NUM_SMEM, PRODUCER_REG_DEALLOC, CONSUMER_REG_ALLOC, P_SMEM_K_TILES, Q_REG_K_TILES>;
    constexpr size_t sMemSize = sizeof(SMemWS<BM, BN, D, NUM_SMEM, P_SMEM_K_TILES, Q_REG_K_TILES>);

    // static_assert(sMemSize < 224 * 1024);
    cudaCheck(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, sMemSize));

    dim3 grid = {static_cast<unsigned int>(SQO/BM), static_cast<unsigned int>(H), static_cast<unsigned int>(B)};
    kernel<<<grid, NUM_THREADS, sMemSize>>>(B, H, S, Q, d_tma_map_Q, d_tma_map_K, d_tma_map_V, d_tma_map_O);
}

