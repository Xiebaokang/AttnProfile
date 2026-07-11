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

template <int BM, int BN, int DIM, int NUM_SMEM>
union SMemWSBaseline {
    struct {
    alignas(128) fp16 Q[BM*DIM];
    alignas(128) fp16 K[BN*DIM*NUM_SMEM];
    alignas(128) fp16 V[BN*DIM*NUM_SMEM];
    alignas(8) uint64_t Qmbar;
    alignas(8) uint64_t Kempty[NUM_SMEM];
    alignas(8) uint64_t Vempty[NUM_SMEM];
    alignas(8) uint64_t Kfull[NUM_SMEM];
    alignas(8) uint64_t Vfull[NUM_SMEM];
    };
    alignas(128) fp16 O[BM*DIM];
};

template <int BM, int BN, int DIM, int NUM_SMEM, int P_SMEM_K_TILES=(BN / 16) / 2>
union SMemWS {
    static constexpr int MMA_K = 16;
    static constexpr int P_SMEM_COLS = ((P_SMEM_K_TILES * MMA_K + 63) / 64) * 64;
    struct {
    alignas(128) fp16 Q[BM*DIM];
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


template<
    int BM, int BN, int DIM, int NUM_THREADS, int NUM_SMEM,
    uint32_t PRODUCER_REG_DEALLOC = 24,
    uint32_t CONSUMER_REG_ALLOC = 240
>
__global__  __launch_bounds__(NUM_THREADS) 
void attnWS2StageBaselineKernel(
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
    // static_assert((DIM >= 256 || DIM == 16 || DIM == 32 || DIM == 64 || DIM == 128) && "DIM ERROR!");
    // static_assert((BN >= 256 || BN == 16 || BN == 32 || BN == 64 || BN == 128 || BN == 96) && "BN ERROR!");
    constexpr int MMA_M = 64;
    constexpr int QK_MMA_N = BN <= 256 ? BN : 256;
    constexpr int PV_MMA_N = DIM <= 256 ? DIM : 256;
    constexpr int MMA_K = 16;
    static_assert(PRODUCER_REG_DEALLOC % 8 == 0);
    static_assert(CONSUMER_REG_ALLOC % 8 == 0);

    // setting shared memory
    extern __shared__ __align__(128) uint8_t smem[];
    SMemWSBaseline<BM, BN, DIM, NUM_SMEM> &s = *reinterpret_cast<SMemWSBaseline<BM, BN, DIM, NUM_SMEM>*>(smem);
    fp16 *sQ = s.Q, *sK = s.K, *sV = s.V, *sO = s.O;
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
        fp16 acc_s_cast[BM/(MMA_M*2)][BN/QK_MMA_N][QK_MMA_N/16][8];

        // define softmax compute function
        auto __softmax = [&](int n_block) {
            // Like FA3's Seqlenk_mask: TMA may zero-fill the physical tail, but
            // padded K columns must be -inf before softmax so they contribute 0.
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
                        fp16 *_QAddr = sQ + tma_smem_offset_2d<BM>(q_row, q_col);
                        fp16 *_KAddr = KAddr + tma_smem_offset_2d<BN>(k_row, k_col);
                        wgmma_ss<QK_MMA_N, 1, 1, 1, 0, 0, 16, 1024, true>(acc_s[i][j], _QAddr, _KAddr);
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
        
        wait(Qmbar, 0);
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
    int BM, int BN, int DIM, int NUM_THREADS, int NUM_SMEM,
    uint32_t PRODUCER_REG_DEALLOC = 24,
    uint32_t CONSUMER_REG_ALLOC = 240,
    int P_SMEM_K_TILES = (BN / 16) / 2
>
__global__  __launch_bounds__(NUM_THREADS) 
void attnWS2StageSmemPKernel(
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
    // static_assert((DIM >= 256 || DIM == 16 || DIM == 32 || DIM == 64 || DIM == 128) && "DIM ERROR!");
    // static_assert((BN >= 256 || BN == 16 || BN == 32 || BN == 64 || BN == 128 || BN == 96) && "BN ERROR!");
    constexpr int MMA_M = 64;
    constexpr int QK_MMA_N = BN <= 256 ? BN : 256;
    constexpr int PV_MMA_N = DIM <= 256 ? DIM : 256;
    constexpr int MMA_K = 16;
    constexpr int P_TOTAL_TILES = BN / MMA_K;
    constexpr int P_REG_TILES = P_TOTAL_TILES - P_SMEM_K_TILES;
    constexpr int P_REG_STORAGE_TILES = P_REG_TILES > 0 ? P_REG_TILES : 1;
    static_assert(PRODUCER_REG_DEALLOC % 8 == 0);
    static_assert(CONSUMER_REG_ALLOC % 8 == 0);
    static_assert(BN % MMA_K == 0);
    static_assert(P_SMEM_K_TILES > 0);
    static_assert(P_SMEM_K_TILES <= P_TOTAL_TILES);

    // setting shared memory
    extern __shared__ __align__(128) uint8_t smem[];
    SMemWS<BM, BN, DIM, NUM_SMEM, P_SMEM_K_TILES> &s =
        *reinterpret_cast<SMemWS<BM, BN, DIM, NUM_SMEM, P_SMEM_K_TILES>*>(smem);
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
                        fp16 *_QAddr = sQ + tma_smem_offset_2d<BM>(q_row, q_col);
                        fp16 *_KAddr = KAddr + tma_smem_offset_2d<BN>(k_row, k_col);
                        wgmma_ss<QK_MMA_N, 1, 1, 1, 0, 0, 16, 1024, true>(acc_s[i][j], _QAddr, _KAddr);
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
        
        wait(Qmbar, 0);
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
    int BM, int BN, int DIM, int NUM_THREADS, int NUM_SMEM,
    uint32_t PRODUCER_REG_DEALLOC = 24,
    uint32_t CONSUMER_REG_ALLOC = 240,
    int P_SMEM_K_TILES = (BN / 16) / 2
>
__global__  __launch_bounds__(NUM_THREADS)
void attnWSKForNSmemPKernel(
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
    // static_assert(DIM >= 256 || DIM == 16 || DIM == 32 || DIM == 64 || DIM == 128, "DIM ERROR!");
    // static_assert(BN >= 256 || BN == 16 || BN == 32 || BN == 64 || BN == 96 || BN == 112 || BN == 128, "BN ERROR!");
    constexpr int MMA_M = 64;
    constexpr int QK_MMA_N = BN <= 256 ? BN : 256;
    constexpr int PV_MMA_N = DIM <= 256 ? DIM : 256;
    constexpr int MMA_K = 16;
    constexpr int QO_TMA_M = 128;
    constexpr int QO_TMA_N = 64;
    constexpr int N = BM / 2 / MMA_M;
    constexpr int P_TOTAL_TILES = BN / MMA_K;
    constexpr int P_REG_TILES = P_TOTAL_TILES - P_SMEM_K_TILES;
    constexpr int P_REG_STORAGE_TILES = P_REG_TILES > 0 ? P_REG_TILES : 1;
    static_assert(NUM_SMEM >= 1);
    static_assert(N >= 1);
    static_assert(BM % (2 * MMA_M) == 0);
    static_assert(BM % QO_TMA_M == 0);
    static_assert(DIM % QO_TMA_N == 0);
    static_assert(PRODUCER_REG_DEALLOC % 8 == 0);
    static_assert(CONSUMER_REG_ALLOC % 8 == 0);
    static_assert(BN % MMA_K == 0);
    static_assert(P_SMEM_K_TILES > 0);
    static_assert(P_SMEM_K_TILES <= P_TOTAL_TILES);

    // setting shared memory
    extern __shared__ __align__(128) uint8_t smem[];
    SMemWS<BM, BN, DIM, NUM_SMEM, P_SMEM_K_TILES> &s =
        *reinterpret_cast<SMemWS<BM, BN, DIM, NUM_SMEM, P_SMEM_K_TILES>*>(smem);
    fp16 *sQ = s.Q, *sK = s.K, *sV = s.V, *sP = s.P, *sO = s.O;
    Barrier *Qmbar = &s.Qmbar;
    Barrier *Kempty = s.Kempty, *Vempty = s.Vempty, *Kfull = s.Kfull, *Vfull = s.Vfull;

    // init mbarrier
    if (threadIdx.x == 0) {
        init_barrier(Qmbar, 1);
        #pragma unroll
        for (int i = 0; i < NUM_SMEM; ++i) {
            init_barrier(&Kfull[i], 1);
            init_barrier(&Vfull[i], 1);
            init_barrier(&Kempty[i], 256);
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
            expect_bytes(Qmbar, BM * DIM * sizeof(fp16));
            #pragma unroll
            for (int q_row = 0; q_row < BM; q_row += QO_TMA_M) {
                #pragma unroll
                for (int q_col = 0; q_col < DIM; q_col += QO_TMA_N) {
                    fp16 *QAddr = sQ + tma_smem_offset_2d<BM>(q_row, q_col);
                    load_async(QAddr, &tensorMapQ, Qmbar, bs, hn, by * BM + q_row, q_col);
                }
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
        fp32 acc_o[N][DIM/PV_MMA_N][PV_MMA_N/16][8];
        fp32 scores_max[N][2];
        fp32 scores_max_prev[N][2];
        fp32 logsum[N][2];
        fp32 acc_s[BN/QK_MMA_N][QK_MMA_N/16][8];
        fp16 acc_s_cast[P_REG_STORAGE_TILES][8];

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

        auto gemm_qk = [&](auto n_const, int smem_i, int phase) {
            constexpr size_t n = decltype(n_const)::value;
            fp16 *KAddr = sK + smem_i * BN * DIM;
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
        };

        auto gemm_pv = [&](auto n_const, auto wait_vfull_const, auto arrive_vempty_const, int smem_i, int phase) {
            constexpr size_t n = decltype(n_const)::value;
            constexpr bool wait_vfull = decltype(wait_vfull_const)::value;
            constexpr bool arrive_vempty = decltype(arrive_vempty_const)::value;
            fp16 *VAddr = sV + smem_i * BN * DIM;
            if constexpr (wait_vfull) { wait(&Vfull[smem_i], phase); }
            __syncwarp();
            warpgroup_arrive();
            #pragma unroll
            for (size_t j=0; j<DIM/PV_MMA_N; j++) {
                #pragma unroll
                for (size_t k=0; k<BN; k+=MMA_K) {
                    int v_row = k;
                    int v_col = j * PV_MMA_N;
                    fp16 *_VAddr = VAddr + tma_smem_offset_2d<BN>(v_row, v_col);
                    const int p_tile = k / MMA_K;
                    if (p_tile < P_SMEM_K_TILES) {
                        int p_row = wg_idx * N * MMA_M + n * MMA_M;
                        fp16 *_PAddr = sP + tma_smem_offset_2d<BM>(p_row, k);
                        wgmma_ss_ab<PV_MMA_N, 1, 1, 1, 0, 1,
                            MMA_K * sizeof(fp16), 1024, true,
                            BN * 64 * sizeof(fp16), 1024, true>(
                            acc_o[n][j],
                            _PAddr,
                            _VAddr);
                    } else {
                        const int reg_tile = p_tile - P_SMEM_K_TILES;
                        wgmma_rs<PV_MMA_N, 1, 1, 1, 0, 1, BN * 64 * sizeof(fp16), 1024, true>(
                            acc_o[n][j],
                            reinterpret_cast<uint32_t*>(acc_s_cast[reg_tile]),
                            _VAddr);
                    }
                }
            }
            warpgroup_commit_batch();
            warpgroup_wait();
            if constexpr (arrive_vempty) { arrive(&Vempty[smem_i]); }
        };

        auto ldsmemP = [&](auto n_const) {
            constexpr size_t n = decltype(n_const)::value;
            const int p_lane_row = lane_id & 0xf;
            const int p_lane_col = (lane_id >> 4) * 8;
            #pragma unroll
            for (size_t j=0; j<BN/QK_MMA_N; j++) {
                #pragma unroll
                for (size_t k=0; k<QK_MMA_N/16; k++) {
                    constexpr int k_tiles_per_j = QK_MMA_N / MMA_K;
                    const int p_tile = j * k_tiles_per_j + k;
                    if (p_tile < P_SMEM_K_TILES) {
                        int p_row = wg_idx * N * MMA_M
                                  + n * MMA_M
                                  + warp_id_in_wg * 16
                                  + p_lane_row;
                        int p_col = p_tile * MMA_K + p_lane_col;
                        fp16 *_sP = sP + tma_smem_swizzle_128b_offset_2d<BM>(p_row, p_col);
                        uint32_t r0 = half2_to_u32(__floats2half2_rn(
                            acc_s[j][k][0],
                            acc_s[j][k][1]
                        ));
                        uint32_t r1 = half2_to_u32(__floats2half2_rn(
                            acc_s[j][k][2],
                            acc_s[j][k][3]
                        ));
                        uint32_t r2 = half2_to_u32(__floats2half2_rn(
                            acc_s[j][k][4],
                            acc_s[j][k][5]
                        ));
                        uint32_t r3 = half2_to_u32(__floats2half2_rn(
                            acc_s[j][k][6],
                            acc_s[j][k][7]
                        ));
                        stmatrix_x4_reg(_sP, r0, r1, r2, r3);
                    } else {
                        const int reg_tile = p_tile - P_SMEM_K_TILES;
                        #pragma unroll
                        for (size_t l=0; l<8; l+=2) {
                            uint1 _t2;
                            float2 _t1 = *(float2*)(&acc_s[j][k][l]);
                            *(half2*)(&(_t2)) = __float22half2_rn(*(float2*)(&(_t1)));
                            *(uint1*)(&acc_s_cast[reg_tile][l]) = _t2;
                        }
                    }
                }
            }
        };

        auto softmax = [&](auto n_const, int n_block) {
            constexpr size_t n = decltype(n_const)::value;
            if ((n_block + 1) * BN > S) {
                #pragma unroll
                for (size_t j=0; j<BN/QK_MMA_N; j++) {
                    #pragma unroll
                    for (size_t k=0; k<QK_MMA_N/16; k++) {
                        #pragma unroll
                        for (size_t l=0; l<8; l++) {
                            const int col_in_mma = (lane_id & 3) * 2 + (l >= 4 ? 8 : 0) + (l & 1);
                            const int col = n_block * BN + j * QK_MMA_N + k * 16 + col_in_mma;
                            if (col >= S) { acc_s[j][k][l] = -FLT_MAX; }
                        }
                    }
                }
            }
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
            // acc_s = exp(acc_s - m)
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
            
            ldsmemP(n_const);

            // scale = exp(pm - m)
            fp32 scores_scale[2];
            #pragma unroll
            for (size_t j=0; j<2; j++) {
                scores_scale[j] = exp2f(scores_max_prev[n][j] * scale - scores_max[n][j] * scale);
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
            // logsum
            #pragma unroll
            for (size_t j=0; j<2; j++) {
                logsum[n][j] = logsum[n][j] * scores_scale[j] + scores_sum[j];
            }
            // rescale acc_o
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

        wait(Qmbar, 0);
        int smem_i = 0, phase = 0;

        // Prologue
        gemm_qk(std::integral_constant<size_t, 0>{}, smem_i, phase);
        softmax(std::integral_constant<size_t, 0>{}, 0);

        // main loop
        const size_t num_n_blocks = (S + BN - 1) / BN;
        for (size_t q_tile=1; q_tile<N*num_n_blocks; q_tile++) {
            const size_t cur_n = q_tile % N;
            const size_t prev_n = (q_tile + N - 1) % N;
            int prev_smem_i = smem_i;
            int prev_phase = phase;
            if (cur_n == 0) {
                if constexpr (NUM_SMEM == 2) {
                    smem_i ^= 1;
                    if (smem_i == 0) { phase ^= 1; }
                } else {
                    smem_i++;
                    if (smem_i >= NUM_SMEM) { smem_i = 0; phase ^= 1; }
                }
            }

            static_for<0, N>([&](auto n_const) {
                constexpr size_t n = decltype(n_const)::value;
                if (cur_n == n) {
                    gemm_qk(n_const, smem_i, phase);
                }
            });
            static_for<0, N>([&](auto n_const) {
                constexpr size_t n = decltype(n_const)::value;
                if (prev_n == n) {
                    gemm_pv(
                        n_const,
                        std::bool_constant<n == 0>{},
                        std::bool_constant<n == N - 1>{},
                        prev_smem_i,
                        prev_phase);
                }
            });
            static_for<0, N>([&](auto n_const) {
                constexpr size_t n = decltype(n_const)::value;
                if (cur_n == n) {
                    softmax(n_const, q_tile / N);
                }
            });
        }

        // Epilogue
        gemm_pv(
            std::integral_constant<size_t, N - 1>{},
            std::bool_constant<N == 1>{},
            std::true_type{},
            smem_i,
            phase);
        
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
        fence_view_async_shared();
        bar_sync(256, 2);
        // tma store
        if (tid == 0) {
            #pragma unroll
            for (int o_row = 0; o_row < BM; o_row += QO_TMA_M) {
                #pragma unroll
                for (int o_col = 0; o_col < DIM; o_col += QO_TMA_N) {
                    fp16 *OAddr = sO + tma_smem_offset_2d<BM>(o_row, o_col);
                    store_async(&tensorMapO, OAddr, bs, hn, by * BM + o_row, o_col);
                }
            }
            tma_store_arrive();
        }
    }
}


template<
    int B, int H, int S, int D=128, int BM=128, int BN=128, int NUM_SMEM=2,
    uint32_t PRODUCER_REG_DEALLOC = 24,
    uint32_t CONSUMER_REG_ALLOC = 240,
    int NUM_THREADS=384
>
void runAttnWS2StageBaselineKernel(fp16 *Q, fp16 *K, fp16 *V, fp16 *O) {
    constexpr int SQO = round_up_to_tile_v<S, BM>;
    constexpr int SKV = round_up_to_tile_v<S, BN>;
    CUtensorMap d_tma_map_Q = create_tensor_map<BM, D>(Q, B, H, SQO, D);
    CUtensorMap d_tma_map_K = create_tensor_map<BN, D>(K, B, H, SKV, D);
    CUtensorMap d_tma_map_V = create_tensor_map<BN, D>(V, B, H, SKV, D);
    CUtensorMap d_tma_map_O = create_tensor_map<BM, D>(O, B, H, SQO, D);

    auto* kernel = attnWS2StageBaselineKernel<BM, BN, D, NUM_THREADS, NUM_SMEM, PRODUCER_REG_DEALLOC, CONSUMER_REG_ALLOC>;
    constexpr size_t sMemSize = sizeof(SMemWSBaseline<BM, BN, D, NUM_SMEM>);

    // static_assert(sMemSize < 224 * 1024);
    // printf("[D] SmemSize: %ld\n", sMemSize);
    cudaCheck(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, sMemSize));

    dim3 grid = {static_cast<unsigned int>(SQO/BM), static_cast<unsigned int>(H), static_cast<unsigned int>(B)};
    kernel<<<grid, NUM_THREADS, sMemSize>>>(B, H, S, d_tma_map_Q, d_tma_map_K, d_tma_map_V, d_tma_map_O);
}


template<
    int B, int H, int S, int D=128, int BM=128, int BN=128, int NUM_SMEM=2,
    uint32_t PRODUCER_REG_DEALLOC = 24,
    uint32_t CONSUMER_REG_ALLOC = 240,
    int P_SMEM_K_TILES = (BN / 16) / 2,
    int NUM_THREADS=384
>
void runAttnWS2StageSmemPKernel(fp16 *Q, fp16 *K, fp16 *V, fp16 *O) {
    constexpr int SQO = round_up_to_tile_v<S, BM>;
    constexpr int SKV = round_up_to_tile_v<S, BN>;
    CUtensorMap d_tma_map_Q = create_tensor_map<BM, D>(Q, B, H, SQO, D);
    CUtensorMap d_tma_map_K = create_tensor_map<BN, D>(K, B, H, SKV, D);
    CUtensorMap d_tma_map_V = create_tensor_map<BN, D>(V, B, H, SKV, D);
    CUtensorMap d_tma_map_O = create_tensor_map<BM, D>(O, B, H, SQO, D);

    auto* kernel = attnWS2StageSmemPKernel<BM, BN, D, NUM_THREADS, NUM_SMEM, PRODUCER_REG_DEALLOC, CONSUMER_REG_ALLOC, P_SMEM_K_TILES>;
    constexpr size_t sMemSize = sizeof(SMemWS<BM, BN, D, NUM_SMEM, P_SMEM_K_TILES>);

    // static_assert(sMemSize < 224 * 1024);
    cudaCheck(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, sMemSize));

    dim3 grid = {static_cast<unsigned int>(SQO/BM), static_cast<unsigned int>(H), static_cast<unsigned int>(B)};
    kernel<<<grid, NUM_THREADS, sMemSize>>>(B, H, S, d_tma_map_Q, d_tma_map_K, d_tma_map_V, d_tma_map_O);
}


template<
    int B, int H, int S, int D=128, int BM=128, int BN=128, int NUM_SMEM=2,
    uint32_t PRODUCER_REG_DEALLOC = 24,
    uint32_t CONSUMER_REG_ALLOC = 240,
    int P_SMEM_K_TILES = (BN / 16) / 2,
    int NUM_THREADS=384
>
void runAttnWSForNSmemPKernel(fp16 *Q, fp16 *K, fp16 *V, fp16 *O) {
    constexpr int QO_TMA_M = 128;
    constexpr int QO_TMA_N = 64;
    static_assert(BM % QO_TMA_M == 0);
    static_assert(D % QO_TMA_N == 0);

    constexpr int SQO = round_up_to_tile_v<S, BM>;
    constexpr int SKV = round_up_to_tile_v<S, BN>;
    CUtensorMap d_tma_map_Q = create_tensor_map<QO_TMA_M, QO_TMA_N>(Q, B, H, SQO, D);
    CUtensorMap d_tma_map_K = create_tensor_map<BN, D>(K, B, H, SKV, D);
    CUtensorMap d_tma_map_V = create_tensor_map<BN, D>(V, B, H, SKV, D);
    CUtensorMap d_tma_map_O = create_tensor_map<QO_TMA_M, QO_TMA_N>(O, B, H, SQO, D);

    auto* kernel = attnWSKForNSmemPKernel<BM, BN, D, NUM_THREADS, NUM_SMEM, PRODUCER_REG_DEALLOC, CONSUMER_REG_ALLOC, P_SMEM_K_TILES>;
    constexpr size_t sMemSize = sizeof(SMemWS<BM, BN, D, NUM_SMEM, P_SMEM_K_TILES>);

    // static_assert(sMemSize < 224 * 1024);
    cudaCheck(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, sMemSize));

    dim3 grid = {static_cast<unsigned int>(SQO/BM), static_cast<unsigned int>(H), static_cast<unsigned int>(B)};
    kernel<<<grid, NUM_THREADS, sMemSize>>>(B, H, S, d_tma_map_Q, d_tma_map_K, d_tma_map_V, d_tma_map_O);
}

template<
    int B, int H, int S, int D, int BM, int BN, int NUM_SMEM,
    uint32_t PRODUCER_REG_DEALLOC,
    uint32_t CONSUMER_REG_ALLOC,
    int P_SMEM_K_TILES,
    int KERNEL_IDX,
    int NUM_THREADS=384
>
void runAttn(fp16 *Q, fp16 *K, fp16 *V, fp16 *O) {
    if constexpr (KERNEL_IDX == 1) {
        runAttnWS2StageBaselineKernel<
            B, H, S, D, BM, BN, NUM_SMEM,
            PRODUCER_REG_DEALLOC, CONSUMER_REG_ALLOC, NUM_THREADS>(Q, K, V, O);
    } else if constexpr (KERNEL_IDX == 2) {
        runAttnWS2StageSmemPKernel<
            B, H, S, D, BM, BN, NUM_SMEM,
            PRODUCER_REG_DEALLOC, CONSUMER_REG_ALLOC, P_SMEM_K_TILES, NUM_THREADS>(Q, K, V, O);
    } else if constexpr (KERNEL_IDX == 3) {
        runAttnWSForNSmemPKernel<
            B, H, S, D, BM, BN, NUM_SMEM,
            PRODUCER_REG_DEALLOC, CONSUMER_REG_ALLOC, P_SMEM_K_TILES, NUM_THREADS>(Q, K, V, O);
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
    constexpr int BN = 160;
    constexpr int B = 1;
    constexpr int H = 16;
    constexpr int S = 32768;
    constexpr int D = 64;

    constexpr int NUM_SMEM = 1;
    constexpr uint32_t PRODUCER_REG_DEALLOC = 24;
    constexpr uint32_t CONSUMER_REG_ALLOC = 184;
    constexpr int P_SMEM_K_TILES = 0;
    constexpr int NUM_THREADS = 384;

    // Tune the last two template args as:
    // <..., NUM_THREADS, NUM_SMEM, PRODUCER_REG_DEALLOC, CONSUMER_REG_ALLOC>
    // auto *kernel = runAttnWS2StageBaselineKernel<B, H, S, D, BM, BN, NUM_SMEM, PRODUCER_REG_DEALLOC, CONSUMER_REG_ALLOC>;
    // auto *kernel = runAttnWS2StageSmemPKernel<B, H, S, D, BM, BN, NUM_SMEM, PRODUCER_REG_DEALLOC, CONSUMER_REG_ALLOC, P_SMEM_K_TILES>;
    constexpr int KERNEL_IDX = 1;
    auto *kernel = runAttn<
        B, H, S, D, BM, BN, NUM_SMEM,
        PRODUCER_REG_DEALLOC, CONSUMER_REG_ALLOC,
        P_SMEM_K_TILES, KERNEL_IDX, NUM_THREADS>;
    // verify_attn<B, H, S, D, BM, BN>(kernel);
    benchmark_attn<B, H, S, D, BM, BN>(kernel);
    // benchmarkAttnRegSweep<B, H, S, D, BM, BN, true, 8>();
    // benchmark_attn_ncu<B, H, S, D, BM, BN>(kernel);
    return 0;
}
