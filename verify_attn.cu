#include "tools.cuh"



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

template <int BM, int BN, int DIM, int NUM_SMEM=1>
struct SMem {
    alignas(128) fp16 Q[BM*DIM];
    alignas(128) fp16 K[BN*DIM*NUM_SMEM];
    alignas(128) fp16 V[BN*DIM*NUM_SMEM];
    alignas(128) fp16 O[BM*DIM];
    alignas(8) uint64_t Qmbar;
    alignas(8) uint64_t Kmbar[NUM_SMEM];
    alignas(8) uint64_t Vmbar[NUM_SMEM];
};

template <int BM, int BN, int DIM, int NUM_SMEM=1, int NUM_CONSUMER=2>
struct SMemWSPingpong {
    alignas(128) fp16 Q[BM*DIM];
    alignas(128) fp16 K[BN*DIM*NUM_SMEM];
    alignas(128) fp16 V[BN*DIM*NUM_SMEM];
    alignas(128) fp16 O[BM*DIM];
    alignas(8) uint64_t Qmbar;
    alignas(8) uint64_t Kempty[NUM_SMEM];
    alignas(8) uint64_t Vempty[NUM_SMEM];
    alignas(8) uint64_t Kfull[NUM_SMEM];
    alignas(8) uint64_t Vfull[NUM_SMEM];
    alignas(8) uint64_t Ppmbar[NUM_CONSUMER];
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


template<int BM, int BN, int DIM, int NUM_THREADS, int NUM_SMEM>
__global__  __launch_bounds__(NUM_THREADS) 
void attnKernel(
    int B, int H, int S, 
    const __grid_constant__ CUtensorMap tensorMapQ, 
    const __grid_constant__ CUtensorMap tensorMapK, 
    const __grid_constant__ CUtensorMap tensorMapV, 
    const __grid_constant__ CUtensorMap tensorMapO
) {
    // 该 kernel 为串行的attention，仅仅测量 TC 和 CC 计算的时间
    const int bs = blockIdx.z;
    const int hn = blockIdx.y;
    const int by = blockIdx.x;
    const int tid = threadIdx.x;
    // wgid, lane, warp
    uint32_t wg_idx = tid >> 7;
    uint32_t lane_id = tid & 31;
    uint32_t warp_id_in_wg = (tid >> 5) & 0x3;  // local warp id inside each 128-thread warpgroup
    // mma size
    assert((DIM >= 256 || DIM == 16 || DIM == 32 || DIM == 64 || DIM == 128) && "DIM ERROR!");
    assert((BN >= 256 || BN == 16 || BN == 32 || BN == 64 || BN == 128) && "BN ERROR!");
    constexpr int MMA_M = 64;
    constexpr int QK_MMA_N = BN <= 256 ? BN : 256;
    constexpr int PV_MMA_N = DIM <= 256 ? DIM : 256;
    constexpr int MMA_K = 16;
    // setting shared memory
    extern __shared__ __align__(128) uint8_t smem[];
    SMem<BM, BN, DIM, NUM_SMEM> &s = *reinterpret_cast<SMem<BM, BN, DIM, NUM_SMEM>*>(smem);
    fp16 *sQ = s.Q, *sK = s.K, *sV = s.V, *sO = s.O;
    Barrier *Qmbar = &s.Qmbar;
    Barrier *Kmbar = s.Kmbar, *Vmbar = s.Vmbar;
    // acc
    fp32 acc_s[BM/(MMA_M*2)][BN/QK_MMA_N][QK_MMA_N/16][8];       // define acc_s
    fp16 acc_s_cast[BM/(MMA_M*2)][BN/QK_MMA_N][QK_MMA_N/16][8];  // define acc_s_cast
    fp32 acc_o[BM/(MMA_M*2)][DIM/PV_MMA_N][PV_MMA_N/16][8];      // define acc_o
    // others
    fp32 scores_max[BM/(MMA_M*2)][2];
    fp32 scores_max_prev[BM/(MMA_M*2)][2];
    fp32 scores_scale[BM/(MMA_M*2)][2];
    fp32 scores_sum[BM/(MMA_M*2)][2];
    fp32 logsum[BM/(MMA_M*2)][2];
    // init mbarrier
    if (threadIdx.x == 0) {
        init_barrier(Qmbar, 1);
        for (int i = 0; i < NUM_SMEM; ++i) {
            init_barrier(&Kmbar[i], 1);
            init_barrier(&Vmbar[i], 1);
        }
    }
    __syncthreads();
    fence_view_async_shared();

    // init other args
    int smem_i = 0, phase = 0;
    fp32 scale = sqrt((1.0f / DIM)) * 1.44269504f;  // log2(e)

    // load Q
    if (tid == 0) {
        expect_bytes(Qmbar, BM * DIM * sizeof(fp16));
        load_async(sQ, &tensorMapQ, Qmbar, bs, hn, by * BM, 0);
    }
    wait(Qmbar, 0);

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
    // loop
    for (size_t iw=0; iw<S; iw+=BN, ++smem_i) {
        if (smem_i >= NUM_SMEM) { smem_i = 0; phase ^= 1; }
        fp16 *KAddr = sK + smem_i * BN * DIM;
        fp16 *VAddr = sV + smem_i * BN * DIM; 
        // load K
        if (tid == 0) {
            expect_bytes(&Kmbar[smem_i], BN * DIM * sizeof(fp16));
            load_async(KAddr, &tensorMapK, &Kmbar[smem_i], bs, hn, iw, 0);
        }
        wait(&Kmbar[smem_i], phase);
        // fill acc_s
        #pragma unroll
        for (size_t i=0; i<BM/(MMA_M*2); i++) {
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
        // gemm-qk
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
        // max_prev = max
        #pragma unroll
        for (size_t i=0; i<BM/(MMA_M*2); i++) {
            #pragma unroll
            for (size_t j=0; j<2; j++) {
                scores_max_prev[i][j] = scores_max[i][j];
            }
        }
        // max = -inf
        #pragma unroll
        for (size_t i=0; i<BM/(MMA_M*2); i++) {
            #pragma unroll
            for (size_t j=0; j<2; j++) {
                scores_max[i][j] = -FLT_MAX;
            }
        }
        // reduce max
        #pragma unroll
        for (size_t i=0; i<BM/(MMA_M*2); i++) {
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
        // __shfl_xor_sync
        const unsigned mask = __activemask();
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
        // max(prev, now)
        #pragma unroll
        for (size_t i=0; i<BM/(MMA_M*2); i++) {
            #pragma unroll
            for (size_t j=0; j<2; j++) {
                scores_max[i][j] = max(scores_max_prev[i][j], scores_max[i][j]);
            }
        }
        // scores_scale = exp2(max_prev * scale  - max * scale)
        #pragma unroll
        for (size_t i=0; i<BM/(MMA_M*2); i++) {
            #pragma unroll
            for (size_t j=0; j<2; j++) {
                scores_scale[i][j] = exp2f(scores_max_prev[i][j] * scale - scores_max[i][j] * scale);
            }
        }
        // acc_s = exp2(acc_s * scale - max)
        #pragma unroll
        for (size_t i=0; i<BM/(MMA_M*2); i++) {
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
        // reduce sum
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
                    #pragma unroll
                    for (size_t l=0; l<8; l++) {
                        int row = (l >> 1) & 1;
                        scores_sum[i][row] +=  acc_s[i][j][k][l];
                    }
                }
            }
        }
        // __shfl_xor_sync
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
                    #pragma unroll
                    for (size_t l=0; l<8; l++) {
                        int row = (l >> 1) & 1;
                        acc_o[i][j][k][l] *= scores_scale[i][row];
                    }
                }
            }
        }
        // load V
        if (tid == 0) {
            expect_bytes(&Vmbar[smem_i], BN * DIM * sizeof(fp16));
            load_async(VAddr, &tensorMapV, &Vmbar[smem_i], bs, hn, iw, 0);
        }
        wait(&Kmbar[smem_i], phase);
        // gemm-pv
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
                    const int p_tile_inner = (k % QK_MMA_N) / MMA_K;
                    fp16 *_VAddr = VAddr + tma_smem_offset_2d<BN>(v_row, v_col);
                    wgmma_rs<PV_MMA_N, 1, 1, 1, 0, 1, 16, 1024, true>(
                        acc_o[i][j],
                        reinterpret_cast<uint32_t*>(acc_s_cast[i][p_tile_outer][p_tile_inner]),
                        _VAddr);
                }
            }
        }
        warpgroup_commit_batch();
        warpgroup_wait();
    }
    // acc_o = acc_o * scores_scale
    #pragma unroll
    for (size_t i=0; i<BM/(MMA_M*2); i++) {
        #pragma unroll
        for (size_t j=0; j<DIM/PV_MMA_N; j++) {
            #pragma unroll
            for (size_t k=0; k<PV_MMA_N/16; k++) {
                #pragma unroll
                for (size_t l=0; l<8; l++) {
                    int row = (l >> 1) & 1;
                    acc_o[i][j][k][l] /= logsum[i][row];
                }
            }
        }
    }
    // copy acc_o to sO (load matritx)
    if (tid == 0) {
        tma_store_wait();
    }
    fp16 d_fp16[8];
    uint32_t* data_ptr = (uint32_t*)d_fp16;
    const int lane_row = lane_id & 0xf;         // 0..15
    const int lane_col = (lane_id >> 4) * 8;    // 0 or 8
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
                #pragma unroll
                for (size_t l=0; l<8; l++) {
                    d_fp16[l] = (fp16)(acc_o[i][j][k][l]);
                }
                stmatrix_x4<fp16>(_sO, data_ptr);
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
    assert((DIM >= 256 || DIM == 16 || DIM == 32 || DIM == 64 || DIM == 128) && "DIM ERROR!");
    assert((BN >= 256 || BN == 16 || BN == 32 || BN == 64 || BN == 128) && "BN ERROR!");
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
    // if (tid >= 256) {  // producer
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
        
        int smem_i = 0, phase = 0;
        fp32 scale = sqrt((1.0f / DIM)) * 1.44269504f;  // log2(e)

        fp32 acc_s[BM/(MMA_M*2)][BN/QK_MMA_N][QK_MMA_N/16][8];       // define acc_s
        fp16 acc_s_cast[BM/(MMA_M*2)][BN/QK_MMA_N][QK_MMA_N/16][8];  // define acc_s_cast
        fp32 acc_o[BM/(MMA_M*2)][DIM/PV_MMA_N][PV_MMA_N/16][8];      // define acc_o
        // others
        fp32 scores_max[BM/(MMA_M*2)][2];
        fp32 scores_max_prev[BM/(MMA_M*2)][2];
        fp32 scores_scale[BM/(MMA_M*2)][2];
        fp32 scores_sum[BM/(MMA_M*2)][2];
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
        // uint32_t wg_idx = tid >> 7;
        uint32_t lane_id = tid & 31;
        uint32_t warp_id_in_wg = (tid >> 5) & 0x3;  // local warp id inside each 128-thread warpgroup
        wait(Qmbar, 0);
        
        // main for loop
        for (size_t iw=0; iw<S; iw+=BN, ++smem_i) {
            if (smem_i >= NUM_SMEM) { smem_i = 0; phase ^= 1; }
            fp16 *KAddr = sK + smem_i * BN * DIM;
            fp16 *VAddr = sV + smem_i * BN * DIM;
            // fill acc_s
            #pragma unroll
            for (size_t i=0; i<BM/(MMA_M*2); i++) {
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
            arrive(&Kempty[smem_i]);  // 释放 tma K 前阻塞
            // softmax
            // max_prev = max
            #pragma unroll
            for (size_t i=0; i<BM/(MMA_M*2); i++) {
                #pragma unroll
                for (size_t j=0; j<2; j++) {
                    scores_max_prev[i][j] = scores_max[i][j];
                }
            }
            // max = -inf
            #pragma unroll
            for (size_t i=0; i<BM/(MMA_M*2); i++) {
                #pragma unroll
                for (size_t j=0; j<2; j++) {
                    scores_max[i][j] = -FLT_MAX;
                }
            }
            // reduce max
            #pragma unroll
            for (size_t i=0; i<BM/(MMA_M*2); i++) {
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
            // __shfl_xor_sync
            const unsigned mask = __activemask();
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
            // max(prev, now)
            #pragma unroll
            for (size_t i=0; i<BM/(MMA_M*2); i++) {
                #pragma unroll
                for (size_t j=0; j<2; j++) {
                    scores_max[i][j] = max(scores_max_prev[i][j], scores_max[i][j]);
                }
            }
            // scores_scale = exp2(max_prev * scale  - max * scale)
            #pragma unroll
            for (size_t i=0; i<BM/(MMA_M*2); i++) {
                #pragma unroll
                for (size_t j=0; j<2; j++) {
                    scores_scale[i][j] = exp2f(scores_max_prev[i][j] * scale - scores_max[i][j] * scale);
                }
            }
            // acc_s = exp2(acc_s * scale - max)
            #pragma unroll
            for (size_t i=0; i<BM/(MMA_M*2); i++) {
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
            // reduce sum
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
                        #pragma unroll
                        for (size_t l=0; l<8; l++) {
                            int row = (l >> 1) & 1;
                            scores_sum[i][row] +=  acc_s[i][j][k][l];
                        }
                    }
                }
            }
            // __shfl_xor_sync
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
                        #pragma unroll
                        for (size_t l=0; l<8; l++) {
                            int row = (l >> 1) & 1;
                            acc_o[i][j][k][l] *= scores_scale[i][row];
                        }
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
                        const int p_tile_inner = (k % QK_MMA_N) / MMA_K;
                        fp16 *_VAddr = VAddr + tma_smem_offset_2d<BN>(v_row, v_col);
                        wgmma_rs<PV_MMA_N, 1, 1, 1, 0, 1, 16, 1024, true>(
                            acc_o[i][j],
                            reinterpret_cast<uint32_t*>(acc_s_cast[i][p_tile_outer][p_tile_inner]),
                            _VAddr);
                    }
                }
            }
            warpgroup_commit_batch();
            warpgroup_wait();
            arrive(&Vempty[smem_i]);  // 释放 tma V 前阻塞
        }
        // acc_o = acc_o * scores_scale
        #pragma unroll
        for (size_t i=0; i<BM/(MMA_M*2); i++) {
            #pragma unroll
            for (size_t j=0; j<DIM/PV_MMA_N; j++) {
                #pragma unroll
                for (size_t k=0; k<PV_MMA_N/16; k++) {
                    #pragma unroll
                    for (size_t l=0; l<8; l++) {
                        int row = (l >> 1) & 1;
                        acc_o[i][j][k][l] /= logsum[i][row];
                    }
                }
            }
        }
        // copy acc_o to sO (load matritx)
        if (tid == 0) {
            tma_store_wait();
        }
        fp16 d_fp16[8];
        uint32_t* data_ptr = (uint32_t*)d_fp16;
        const int lane_row = lane_id & 0xf;         // 0..15
        const int lane_col = (lane_id >> 4) * 8;    // 0 or 8
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
                    #pragma unroll
                    for (size_t l=0; l<8; l++) {
                        d_fp16[l] = (fp16)(acc_o[i][j][k][l]);
                    }
                    stmatrix_x4<fp16>(_sO, data_ptr);
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
void attnWSStage2Kernel(
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
        
        int smem_i = 0, phase = 0;
        fp32 scale = sqrt((1.0f / DIM)) * 1.44269504f;  // log2(e)

        fp32 acc_s[BM/(MMA_M*2)][BN/QK_MMA_N][QK_MMA_N/16][8];       // define acc_s
        fp16 acc_s_cast[BM/(MMA_M*2)][BN/QK_MMA_N][QK_MMA_N/16][8];  // define acc_s_cast
        fp32 acc_o[BM/(MMA_M*2)][DIM/PV_MMA_N][PV_MMA_N/16][8];      // define acc_o
        // others
        fp32 scores_max[BM/(MMA_M*2)][2];
        fp32 scores_max_prev[BM/(MMA_M*2)][2];
        fp32 scores_scale[BM/(MMA_M*2)][2];
        fp32 scores_sum[BM/(MMA_M*2)][2];
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
        uint32_t wg_idx = tid >> 7;
        uint32_t lane_id = tid & 31;
        uint32_t warp_id_in_wg = (tid >> 5) & 0x3;  // local warp id inside each 128-thread warpgroup
        wait(Qmbar, 0);
        // =====  prologue  =====
        // fill acc_s
        #pragma unroll
        for (size_t i=0; i<BM/(MMA_M*2); i++) {
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
        arrive(&Kempty[smem_i]);  // 释放 tma K 前阻塞
        // softmax
        // max_prev = max
        #pragma unroll
        for (size_t i=0; i<BM/(MMA_M*2); i++) {
            #pragma unroll
            for (size_t j=0; j<2; j++) {
                scores_max_prev[i][j] = scores_max[i][j];
            }
        }
        // max = -inf
        #pragma unroll
        for (size_t i=0; i<BM/(MMA_M*2); i++) {
            #pragma unroll
            for (size_t j=0; j<2; j++) {
                scores_max[i][j] = -FLT_MAX;
            }
        }
        // reduce max
        #pragma unroll
        for (size_t i=0; i<BM/(MMA_M*2); i++) {
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
        // __shfl_xor_sync
        const unsigned mask = __activemask();
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
        // max(prev, now)
        #pragma unroll
        for (size_t i=0; i<BM/(MMA_M*2); i++) {
            #pragma unroll
            for (size_t j=0; j<2; j++) {
                scores_max[i][j] = max(scores_max_prev[i][j], scores_max[i][j]);
            }
        }
        // scores_scale = exp2(max_prev * scale  - max * scale)
        #pragma unroll
        for (size_t i=0; i<BM/(MMA_M*2); i++) {
            #pragma unroll
            for (size_t j=0; j<2; j++) {
                scores_scale[i][j] = exp2f(scores_max_prev[i][j] * scale - scores_max[i][j] * scale);
            }
        }
        // acc_s = exp2(acc_s * scale - max)
        #pragma unroll
        for (size_t i=0; i<BM/(MMA_M*2); i++) {
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
        // reduce sum
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
                    #pragma unroll
                    for (size_t l=0; l<8; l++) {
                        int row = (l >> 1) & 1;
                        scores_sum[i][row] +=  acc_s[i][j][k][l];
                    }
                }
            }
        }
        // __shfl_xor_sync
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
                    #pragma unroll
                    for (size_t l=0; l<8; l++) {
                        int row = (l >> 1) & 1;
                        acc_o[i][j][k][l] *= scores_scale[i][row];
                    }
                }
            }
        }
        smem_i++;

        // =====  main loop  =====
        for (size_t iw=BN; iw<S; iw+=BN, ++smem_i) {
            if (smem_i == NUM_SMEM) { smem_i = 0; phase ^= 1; }
            int prev_idx = (smem_i + NUM_SMEM -1) % NUM_SMEM;
            int prve_phase = phase;
            if (prev_idx == NUM_SMEM -1) { prve_phase ^= 1; }
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
                            acc_s[i][j][k][l] = 0.0f;  // no mask
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
            // +++++++++++++++ qk(j)
            warpgroup_wait();
            arrive(&Kempty[smem_i]);  // 释放 tma K 前阻塞

            // gemm-pv
            fp16 *VAddr = sV + prev_idx * BN * DIM;
            wait(&Vfull[prev_idx], prve_phase);
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
                        const int p_tile_inner = (k % QK_MMA_N) / MMA_K;
                        fp16 *_VAddr = VAddr + tma_smem_offset_2d<BN>(v_row, v_col);
                        wgmma_rs<PV_MMA_N, 1, 1, 1, 0, 1, 16, 1024, true>(
                            acc_o[i][j],
                            reinterpret_cast<uint32_t*>(acc_s_cast[i][p_tile_outer][p_tile_inner]),
                            _VAddr);
                    }
                }
            }
            warpgroup_commit_batch();
            // +++++++++++++++++++++ pv(j-1)
            warpgroup_wait();
            arrive(&Vempty[prev_idx]);  // 释放 tma V 前阻塞

            // softmax
            // max_prev = max
            #pragma unroll
            for (size_t i=0; i<BM/(MMA_M*2); i++) {
                #pragma unroll
                for (size_t j=0; j<2; j++) {
                    scores_max_prev[i][j] = scores_max[i][j];
                }
            }
            // max = -inf
            #pragma unroll
            for (size_t i=0; i<BM/(MMA_M*2); i++) {
                #pragma unroll
                for (size_t j=0; j<2; j++) {
                    scores_max[i][j] = -FLT_MAX;
                }
            }
            // reduce max
            #pragma unroll
            for (size_t i=0; i<BM/(MMA_M*2); i++) {
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
            // __shfl_xor_sync
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
            // max(prev, now)
            #pragma unroll
            for (size_t i=0; i<BM/(MMA_M*2); i++) {
                #pragma unroll
                for (size_t j=0; j<2; j++) {
                    scores_max[i][j] = max(scores_max_prev[i][j], scores_max[i][j]);
                }
            }
            // scores_scale = exp2(max_prev * scale  - max * scale)
            #pragma unroll
            for (size_t i=0; i<BM/(MMA_M*2); i++) {
                #pragma unroll
                for (size_t j=0; j<2; j++) {
                    scores_scale[i][j] = exp2f(scores_max_prev[i][j] * scale - scores_max[i][j] * scale);
                }
            }
            // acc_s = exp2(acc_s * scale - max)
            #pragma unroll
            for (size_t i=0; i<BM/(MMA_M*2); i++) {
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
            // reduce sum
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
                        #pragma unroll
                        for (size_t l=0; l<8; l++) {
                            int row = (l >> 1) & 1;
                            scores_sum[i][row] +=  acc_s[i][j][k][l];
                        }
                    }
                }
            }
            // __shfl_xor_sync
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
                        #pragma unroll
                        for (size_t l=0; l<8; l++) {
                            int row = (l >> 1) & 1;
                            acc_o[i][j][k][l] *= scores_scale[i][row];
                        }
                    }
                }
            }
        }
        
        // ======  epilogue  ======
        // gemm-pv
        int last_idx = (smem_i % NUM_SMEM + NUM_SMEM - 1) % NUM_SMEM;
        int last_phase = phase;
        fp16 *VAddr = sV + last_idx * BN * DIM;
        wait(&Vfull[last_idx], last_phase);
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
                    const int p_tile_inner = (k % QK_MMA_N) / MMA_K;
                    fp16 *_VAddr = VAddr + tma_smem_offset_2d<BN>(v_row, v_col);
                    wgmma_rs<PV_MMA_N, 1, 1, 1, 0, 1, 16, 1024, true>(
                        acc_o[i][j],
                        reinterpret_cast<uint32_t*>(acc_s_cast[i][p_tile_outer][p_tile_inner]),
                        _VAddr);
                }
            }
        }
        warpgroup_commit_batch();
        warpgroup_wait();
        arrive(&Vempty[last_idx]);  // 释放 tma V 前阻塞

        // acc_o = acc_o * scores_scale
        #pragma unroll
        for (size_t i=0; i<BM/(MMA_M*2); i++) {
            #pragma unroll
            for (size_t j=0; j<DIM/PV_MMA_N; j++) {
                #pragma unroll
                for (size_t k=0; k<PV_MMA_N/16; k++) {
                    #pragma unroll
                    for (size_t l=0; l<8; l++) {
                        int row = (l >> 1) & 1;
                        acc_o[i][j][k][l] /= logsum[i][row];
                    }
                }
            }
        }
        // copy acc_o to sO (load matritx)
        if (tid == 0) {
            tma_store_wait();
        }
        fp16 d_fp16[8];
        uint32_t* data_ptr = (uint32_t*)d_fp16;
        const int lane_row = lane_id & 0xf;         // 0..15
        const int lane_col = (lane_id >> 4) * 8;    // 0 or 8
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
                    #pragma unroll
                    for (size_t l=0; l<8; l++) {
                        d_fp16[l] = (fp16)(acc_o[i][j][k][l]);
                    }
                    stmatrix_x4<fp16>(_sO, data_ptr);
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
void attnWSStage3Kernel(
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
        
        int smem_i = 0, phase = 0;
        fp32 scale = sqrt((1.0f / DIM)) * 1.44269504f;  // log2(e)

        fp32 acc_s[2][BM/(MMA_M*2)][BN/QK_MMA_N][QK_MMA_N/16][8];       // define acc_s
        fp16 acc_s_cast[BM/(MMA_M*2)][BN/QK_MMA_N][QK_MMA_N/16][8];  // define acc_s_cast
        fp32 acc_o[BM/(MMA_M*2)][DIM/PV_MMA_N][PV_MMA_N/16][8];      // define acc_o
        // others
        fp32 scores_max[BM/(MMA_M*2)][2];
        fp32 scores_max_prev[BM/(MMA_M*2)][2];
        fp32 scores_scale[BM/(MMA_M*2)][2];
        fp32 scores_sum[BM/(MMA_M*2)][2];
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
        
        uint32_t lane_id = tid & 31;
        uint32_t warp_id_in_wg = (tid >> 5) & 0x3;  // local warp id inside each 128-thread warpgroup
        wait(Qmbar, 0);
        // ======== prologue ========
        // fill acc_s
        #pragma unroll
        for (size_t i=0; i<BM/(MMA_M*2); i++) {
            #pragma unroll
            for (size_t j=0; j<BN/QK_MMA_N; j++) {
                #pragma unroll
                for (size_t k=0; k<QK_MMA_N/16; k++) {
                    #pragma unroll
                    for (size_t l=0; l<8; l++) {
                        acc_s[0][i][j][k][l] = 0.0f;  // no mask
                        acc_s[1][i][j][k][l] = 0.0f;  // no mask
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
                    wgmma_ss<QK_MMA_N, 1, 1, 1, 0, 0, 16, 1024, true>(acc_s[0][i][j], _QAddr, _KAddr);
                }
            }
        }
        warpgroup_commit_batch();
        warpgroup_wait();
        // warpgroup_fence_operand
        #pragma unroll
        for (size_t i=0; i<BM/(MMA_M*2); i++) {
            #pragma unroll
            for (size_t j=0; j<BN/QK_MMA_N; j++) {
                #pragma unroll
                for (size_t k=0; k<QK_MMA_N/16; k++) {
                    #pragma unroll
                    for (size_t l=0; l<8; l++) {
                        warpgroup_fence_operand(acc_s[0][i][j][k][l]);
                    }
                }
            }
        }
        arrive(&Kempty[smem_i]);  // 释放 tma K 前阻塞
        
        
        // softmax
        // max_prev = max
        #pragma unroll
        for (size_t i=0; i<BM/(MMA_M*2); i++) {
            #pragma unroll
            for (size_t j=0; j<2; j++) {
                scores_max_prev[i][j] = scores_max[i][j];
            }
        }
        // max = -inf
        #pragma unroll
        for (size_t i=0; i<BM/(MMA_M*2); i++) {
            #pragma unroll
            for (size_t j=0; j<2; j++) {
                scores_max[i][j] = -FLT_MAX;
            }
        }
        // reduce max
        #pragma unroll
        for (size_t i=0; i<BM/(MMA_M*2); i++) {
            #pragma unroll
            for (size_t j=0; j<BN/QK_MMA_N; j++) {
                #pragma unroll
                for (size_t k=0; k<QK_MMA_N/16; k++) {
                    #pragma unroll
                    for (size_t l=0; l<8; l++) {
                        int row = (l >> 1) & 1;
                        scores_max[i][row] = max(scores_max[i][row], acc_s[0][i][j][k][l]);
                    }
                }
            }
        }
        // __shfl_xor_sync
        const unsigned mask = __activemask();
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
        // max(prev, now)
        #pragma unroll
        for (size_t i=0; i<BM/(MMA_M*2); i++) {
            #pragma unroll
            for (size_t j=0; j<2; j++) {
                scores_max[i][j] = max(scores_max_prev[i][j], scores_max[i][j]);
            }
        }
        // scores_scale = exp2(max_prev * scale  - max * scale)
        #pragma unroll
        for (size_t i=0; i<BM/(MMA_M*2); i++) {
            #pragma unroll
            for (size_t j=0; j<2; j++) {
                scores_scale[i][j] = exp2f(scores_max_prev[i][j] * scale - scores_max[i][j] * scale);
            }
        }
        // acc_s = exp2(acc_s * scale - max)
        #pragma unroll
        for (size_t i=0; i<BM/(MMA_M*2); i++) {
            #pragma unroll
            for (size_t j=0; j<BN/QK_MMA_N; j++) {
                #pragma unroll
                for (size_t k=0; k<QK_MMA_N/16; k++) {
                    #pragma unroll
                    for (size_t l=0; l<8; l++) {
                        int row = (l >> 1) & 1;
                        acc_s[0][i][j][k][l] = exp2f(acc_s[0][i][j][k][l] * scale - scores_max[i][row] * scale);
                    }
                }
            }
        }
        // reduce sum
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
                    #pragma unroll
                    for (size_t l=0; l<8; l++) {
                        int row = (l >> 1) & 1;
                        scores_sum[i][row] +=  acc_s[0][i][j][k][l];
                    }
                }
            }
        }
        // __shfl_xor_sync
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
                        float2 _t1 = *(float2*)(&acc_s[0][i][j][k][l]);
                        *(half2*)(&(_t2)) = __float22half2_rn(*(float2*)(&(_t1)));
                        *(uint1*)(&acc_s_cast[i][j][k][l]) = _t2;
                    }
                }
            }
        }
        smem_i++;  // 1

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
                    fp16 *_KAddr = sK + BN * DIM + tma_smem_offset_2d<BN>(k_row, k_col);
                    wgmma_ss<QK_MMA_N, 1, 1, 1, 0, 0, 16, 1024, true>(acc_s[1][i][j], _QAddr, _KAddr);
                }
            }
        }
        warpgroup_commit_batch();
        warpgroup_wait();
        // warpgroup_fence_operand
        #pragma unroll
        for (size_t i=0; i<BM/(MMA_M*2); i++) {
            #pragma unroll
            for (size_t j=0; j<BN/QK_MMA_N; j++) {
                #pragma unroll
                for (size_t k=0; k<QK_MMA_N/16; k++) {
                    #pragma unroll
                    for (size_t l=0; l<8; l++) {
                        warpgroup_fence_operand(acc_s[1][i][j][k][l]);
                    }
                }
            }
        }
        arrive(&Kempty[smem_i]);  // 释放 tma K 前阻塞
        smem_i++;  // 2

        // main for loop
        for (size_t iw=2*BN; iw<S; iw+=BN, ++smem_i) {
            const size_t tile_idx = iw / BN;
            if (smem_i == NUM_SMEM) { smem_i = 0; phase ^= 1; }
            const size_t prev_tile_idx = tile_idx - 2;
            const int prev_idx = static_cast<int>(prev_tile_idx % NUM_SMEM);  // prev 2
            const int prev_phase = static_cast<int>((prev_tile_idx / NUM_SMEM) & 1);
            const int acc_write_idx = static_cast<int>(tile_idx & 1);
            const int acc_reduce_idx = static_cast<int>((tile_idx - 1) & 1);
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
                            acc_s[acc_write_idx][i][j][k][l] = 0.0f;  // no mask
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
                        wgmma_ss<QK_MMA_N, 1, 1, 1, 0, 0, 16, 1024, true>(acc_s[acc_write_idx][i][j], _QAddr, _KAddr);
                    }
                }
            }
            warpgroup_commit_batch();
            warpgroup_wait();
            // warpgroup_fence_operand
            #pragma unroll
            for (size_t i=0; i<BM/(MMA_M*2); i++) {
                #pragma unroll
                for (size_t j=0; j<BN/QK_MMA_N; j++) {
                    #pragma unroll
                    for (size_t k=0; k<QK_MMA_N/16; k++) {
                        #pragma unroll
                        for (size_t l=0; l<8; l++) {
                            warpgroup_fence_operand(acc_s[acc_write_idx][i][j][k][l]);
                        }
                    }
                }
            }
            arrive(&Kempty[smem_i]);  // 释放 tma K 前阻塞

            // acc_o = acc_o * scores_scale (j-2)
            #pragma unroll
            for (size_t i=0; i<BM/(MMA_M*2); i++) {
                #pragma unroll
                for (size_t j=0; j<DIM/PV_MMA_N; j++) {
                    #pragma unroll
                    for (size_t k=0; k<PV_MMA_N/16; k++) {
                        #pragma unroll
                        for (size_t l=0; l<8; l++) {
                            int row = (l >> 1) & 1;
                            acc_o[i][j][k][l] *= scores_scale[i][row];
                        }
                    }
                }
            }
            // gemm-pv (j-2)
            fp16 *VAddr = sV + prev_idx * BN * DIM;
            wait(&Vfull[prev_idx], prev_phase);
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
                        const int p_tile_inner = (k % QK_MMA_N) / MMA_K;
                        fp16 *_VAddr = VAddr + tma_smem_offset_2d<BN>(v_row, v_col);
                        wgmma_rs<PV_MMA_N, 1, 1, 1, 0, 1, 16, 1024, true>(
                            acc_o[i][j],
                            reinterpret_cast<uint32_t*>(acc_s_cast[i][p_tile_outer][p_tile_inner]),
                            _VAddr);
                    }
                }
            }
            warpgroup_commit_batch();
            warpgroup_wait();
            // warpgroup_fence_operand
            #pragma unroll
            for (size_t i=0; i<BM/(MMA_M*2); i++) {
                #pragma unroll
                for (size_t j=0; j<DIM/PV_MMA_N; j++) {
                    #pragma unroll
                    for (size_t k=0; k<PV_MMA_N/16; k++) {
                        #pragma unroll
                        for (size_t l=0; l<8; l++) {
                            warpgroup_fence_operand(acc_o[i][j][k][l]);
                        }
                    }
                }
            }
            arrive(&Vempty[prev_idx]);  // 释放 tma V 前阻塞

            // softmax
            // max_prev = max
            #pragma unroll
            for (size_t i=0; i<BM/(MMA_M*2); i++) {
                #pragma unroll
                for (size_t j=0; j<2; j++) {
                    scores_max_prev[i][j] = scores_max[i][j];
                }
            }
            // max = -inf
            #pragma unroll
            for (size_t i=0; i<BM/(MMA_M*2); i++) {
                #pragma unroll
                for (size_t j=0; j<2; j++) {
                    scores_max[i][j] = -FLT_MAX;
                }
            }
            // reduce max
            #pragma unroll
            for (size_t i=0; i<BM/(MMA_M*2); i++) {
                #pragma unroll
                for (size_t j=0; j<BN/QK_MMA_N; j++) {
                    #pragma unroll
                    for (size_t k=0; k<QK_MMA_N/16; k++) {
                        #pragma unroll
                        for (size_t l=0; l<8; l++) {
                            int row = (l >> 1) & 1;
                            scores_max[i][row] = max(scores_max[i][row], acc_s[acc_reduce_idx][i][j][k][l]);
                        }
                    }
                }
            }
            // __shfl_xor_sync
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
            // max(prev, now)
            #pragma unroll
            for (size_t i=0; i<BM/(MMA_M*2); i++) {
                #pragma unroll
                for (size_t j=0; j<2; j++) {
                    scores_max[i][j] = max(scores_max_prev[i][j], scores_max[i][j]);
                }
            }
            // scores_scale = exp2(max_prev * scale  - max * scale)
            #pragma unroll
            for (size_t i=0; i<BM/(MMA_M*2); i++) {
                #pragma unroll
                for (size_t j=0; j<2; j++) {
                    scores_scale[i][j] = exp2f(scores_max_prev[i][j] * scale - scores_max[i][j] * scale);
                }
            }
            // acc_s = exp2(acc_s * scale - max)
            #pragma unroll
            for (size_t i=0; i<BM/(MMA_M*2); i++) {
                #pragma unroll
                for (size_t j=0; j<BN/QK_MMA_N; j++) {
                    #pragma unroll
                    for (size_t k=0; k<QK_MMA_N/16; k++) {
                        #pragma unroll
                        for (size_t l=0; l<8; l++) {
                            int row = (l >> 1) & 1;
                            acc_s[acc_reduce_idx][i][j][k][l] = exp2f(acc_s[acc_reduce_idx][i][j][k][l] * scale - scores_max[i][row] * scale);
                        }
                    }
                }
            }
            // reduce sum
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
                        #pragma unroll
                        for (size_t l=0; l<8; l++) {
                            int row = (l >> 1) & 1;
                            scores_sum[i][row] +=  acc_s[acc_reduce_idx][i][j][k][l];
                        }
                    }
                }
            }
            // __shfl_xor_sync
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
                            float2 _t1 = *(float2*)(&acc_s[acc_reduce_idx][i][j][k][l]);
                            *(half2*)(&(_t2)) = __float22half2_rn(*(float2*)(&(_t1)));
                            *(uint1*)(&acc_s_cast[i][j][k][l]) = _t2;
                        }
                    }
                }
            }
        }

        // ========= epiploger ==========
        const size_t num_tiles = S / BN;
        const size_t last_bo_tile_idx = num_tiles - 2;
        const int last_bo_idx = static_cast<int>(last_bo_tile_idx % NUM_SMEM);
        const int last_bo_phase = static_cast<int>((last_bo_tile_idx / NUM_SMEM) & 1);
        fp16 *VAddr = sV + last_bo_idx * BN * DIM;
        // acc_o = acc_o * scores_scale
        #pragma unroll
        for (size_t i=0; i<BM/(MMA_M*2); i++) {
            #pragma unroll
            for (size_t j=0; j<DIM/PV_MMA_N; j++) {
                #pragma unroll
                for (size_t k=0; k<PV_MMA_N/16; k++) {
                    #pragma unroll
                    for (size_t l=0; l<8; l++) {
                        int row = (l >> 1) & 1;
                        acc_o[i][j][k][l] *= scores_scale[i][row];
                    }
                }
            }
        }
        // gemm-pv
        wait(&Vfull[last_bo_idx], last_bo_phase);
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
                    const int p_tile_inner = (k % QK_MMA_N) / MMA_K;
                    fp16 *_VAddr = VAddr + tma_smem_offset_2d<BN>(v_row, v_col);
                    wgmma_rs<PV_MMA_N, 1, 1, 1, 0, 1, 16, 1024, true>(
                        acc_o[i][j],
                        reinterpret_cast<uint32_t*>(acc_s_cast[i][p_tile_outer][p_tile_inner]),
                        _VAddr);
                }
            }
        }
        warpgroup_commit_batch();
        warpgroup_wait();
        // warpgroup_fence_operand
        #pragma unroll
        for (size_t i=0; i<BM/(MMA_M*2); i++) {
            #pragma unroll
            for (size_t j=0; j<DIM/PV_MMA_N; j++) {
                #pragma unroll
                for (size_t k=0; k<PV_MMA_N/16; k++) {
                    #pragma unroll
                    for (size_t l=0; l<8; l++) {
                        warpgroup_fence_operand(acc_o[i][j][k][l]);
                    }
                }
            }
        }
        arrive(&Vempty[last_bo_idx]);  // 释放 tma V 前阻塞

        // softmax
        const size_t last_tile_idx = num_tiles - 1;
        const int last_acc_idx = static_cast<int>(last_tile_idx & 1);
        // max_prev = max
        #pragma unroll
        for (size_t i=0; i<BM/(MMA_M*2); i++) {
            #pragma unroll
            for (size_t j=0; j<2; j++) {
                scores_max_prev[i][j] = scores_max[i][j];
            }
        }
        // max = -inf
        #pragma unroll
        for (size_t i=0; i<BM/(MMA_M*2); i++) {
            #pragma unroll
            for (size_t j=0; j<2; j++) {
                scores_max[i][j] = -FLT_MAX;
            }
        }
        // reduce max
        #pragma unroll
        for (size_t i=0; i<BM/(MMA_M*2); i++) {
            #pragma unroll
            for (size_t j=0; j<BN/QK_MMA_N; j++) {
                #pragma unroll
                for (size_t k=0; k<QK_MMA_N/16; k++) {
                    #pragma unroll
                    for (size_t l=0; l<8; l++) {
                        int row = (l >> 1) & 1;
                        scores_max[i][row] = max(scores_max[i][row], acc_s[last_acc_idx][i][j][k][l]);
                    }
                }
            }
        }
        // __shfl_xor_sync
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
        // max(prev, now)
        #pragma unroll
        for (size_t i=0; i<BM/(MMA_M*2); i++) {
            #pragma unroll
            for (size_t j=0; j<2; j++) {
                scores_max[i][j] = max(scores_max_prev[i][j], scores_max[i][j]);
            }
        }
        // scores_scale = exp2(max_prev * scale  - max * scale)
        #pragma unroll
        for (size_t i=0; i<BM/(MMA_M*2); i++) {
            #pragma unroll
            for (size_t j=0; j<2; j++) {
                scores_scale[i][j] = exp2f(scores_max_prev[i][j] * scale - scores_max[i][j] * scale);
            }
        }
        // acc_s = exp2(acc_s * scale - max)
        #pragma unroll
        for (size_t i=0; i<BM/(MMA_M*2); i++) {
            #pragma unroll
            for (size_t j=0; j<BN/QK_MMA_N; j++) {
                #pragma unroll
                for (size_t k=0; k<QK_MMA_N/16; k++) {
                    #pragma unroll
                    for (size_t l=0; l<8; l++) {
                        int row = (l >> 1) & 1;
                        acc_s[last_acc_idx][i][j][k][l] = exp2f(acc_s[last_acc_idx][i][j][k][l] * scale - scores_max[i][row] * scale);
                    }
                }
            }
        }
        // reduce sum
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
                    #pragma unroll
                    for (size_t l=0; l<8; l++) {
                        int row = (l >> 1) & 1;
                        scores_sum[i][row] +=  acc_s[last_acc_idx][i][j][k][l];
                    }
                }
            }
        }
        // __shfl_xor_sync
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
                        float2 _t1 = *(float2*)(&acc_s[last_acc_idx][i][j][k][l]);
                        *(half2*)(&(_t2)) = __float22half2_rn(*(float2*)(&(_t1)));
                        *(uint1*)(&acc_s_cast[i][j][k][l]) = _t2;
                    }
                }
            }
        }

        const int last_idx = static_cast<int>(last_tile_idx % NUM_SMEM);
        const int last_phase = static_cast<int>((last_tile_idx / NUM_SMEM) & 1);
        VAddr = sV + last_idx * BN * DIM;
        // acc_o = acc_o * scores_scale 
        #pragma unroll
        for (size_t i=0; i<BM/(MMA_M*2); i++) {
            #pragma unroll
            for (size_t j=0; j<DIM/PV_MMA_N; j++) {
                #pragma unroll
                for (size_t k=0; k<PV_MMA_N/16; k++) {
                    #pragma unroll
                    for (size_t l=0; l<8; l++) {
                        int row = (l >> 1) & 1;
                        acc_o[i][j][k][l] *= scores_scale[i][row];
                    }
                }
            }
        }
        // gemm-pv (j-2)
        wait(&Vfull[last_idx], last_phase);
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
                    const int p_tile_inner = (k % QK_MMA_N) / MMA_K;
                    fp16 *_VAddr = VAddr + tma_smem_offset_2d<BN>(v_row, v_col);
                    wgmma_rs<PV_MMA_N, 1, 1, 1, 0, 1, 16, 1024, true>(
                        acc_o[i][j],
                        reinterpret_cast<uint32_t*>(acc_s_cast[i][p_tile_outer][p_tile_inner]),
                        _VAddr);
                }
            }
        }
        warpgroup_commit_batch();
        warpgroup_wait();
        // warpgroup_fence_operand
        #pragma unroll
        for (size_t i=0; i<BM/(MMA_M*2); i++) {
            #pragma unroll
            for (size_t j=0; j<DIM/PV_MMA_N; j++) {
                #pragma unroll
                for (size_t k=0; k<PV_MMA_N/16; k++) {
                    #pragma unroll
                    for (size_t l=0; l<8; l++) {
                        warpgroup_fence_operand(acc_o[i][j][k][l]);
                    }
                }
            }
        }
        arrive(&Vempty[last_idx]);  // 释放 tma V 前阻塞

        // acc_o = acc_o * scores_scale
        #pragma unroll
        for (size_t i=0; i<BM/(MMA_M*2); i++) {
            #pragma unroll
            for (size_t j=0; j<DIM/PV_MMA_N; j++) {
                #pragma unroll
                for (size_t k=0; k<PV_MMA_N/16; k++) {
                    #pragma unroll
                    for (size_t l=0; l<8; l++) {
                        int row = (l >> 1) & 1;
                        acc_o[i][j][k][l] /= logsum[i][row];
                    }
                }
            }
        }
        // copy acc_o to sO (load matritx)
        if (tid == 0) {
            tma_store_wait();
        }
        fp16 d_fp16[8];
        uint32_t* data_ptr = (uint32_t*)d_fp16;
        const int lane_row = lane_id & 0xf;         // 0..15
        const int lane_col = (lane_id >> 4) * 8;    // 0 or 8
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
                    #pragma unroll
                    for (size_t l=0; l<8; l++) {
                        d_fp16[l] = (fp16)(acc_o[i][j][k][l]);
                    }
                    stmatrix_x4<fp16>(_sO, data_ptr);
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
void attnWSPingpongKernel(
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
            init_barrier(&Kempty[i], 128);  // 128 thread arrive
            init_barrier(&Vempty[i], 128);
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
        if (wg_idx == 0) {
            #pragma unroll
            for (int st = 0; st < NUM_SMEM; ++st) {
                arrive(&Kempty[st]);
                arrive(&Vempty[st]);
            }
        }

        fp32 scale = sqrt((1.0f / DIM)) * 1.44269504f;  // log2(e)
        fp32 acc_s[BM/(MMA_M*2)][BN/QK_MMA_N][QK_MMA_N/16][8];       // define acc_s
        fp16 acc_s_cast[BM/(MMA_M*2)][BN/QK_MMA_N][QK_MMA_N/16][8];  // define acc_s_cast
        fp32 acc_o[BM/(MMA_M*2)][DIM/PV_MMA_N][PV_MMA_N/16][8];      // define acc_o
        // others
        fp32 scores_max[BM/(MMA_M*2)][2];
        fp32 scores_max_prev[BM/(MMA_M*2)][2];
        fp32 scores_scale[BM/(MMA_M*2)][2];
        fp32 scores_sum[BM/(MMA_M*2)][2];
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
        uint32_t lane_id = tid & 31;
        uint32_t warp_id_in_wg = (tid >> 5) & 0x3;  // local warp id inside each 128-thread warpgroup
        
        constexpr int BAR_1 = 3;
        constexpr int BAR_2 = 4;
        constexpr int BAR_3 = 5;
        constexpr int BAR_4 = 6;
        wait(Qmbar, 0);

        // step.1 prologue
        if (wg_idx == 0) {
            // fill acc_s
            #pragma unroll
            for (size_t i=0; i<BM/(MMA_M*2); i++) {
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
            // gemm-qk
            wait(&Kfull[0], 0);
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
                        fp16 *_KAddr = sK + tma_smem_offset_2d<BN>(k_row, k_col);
                        wgmma_ss<QK_MMA_N, 1, 1, 1, 0, 0, 16, 1024, true>(acc_s[i][j], _QAddr, _KAddr);
                    }
                }
            }
            warpgroup_commit_batch();
            warpgroup_wait();
        }
        bar_sync(256, BAR_1);

        const unsigned mask = __activemask();
        if (wg_idx == 0) {
            // softmax
            // max_prev = max
            #pragma unroll
            for (size_t i=0; i<BM/(MMA_M*2); i++) {
                #pragma unroll
                for (size_t j=0; j<2; j++) {
                    scores_max_prev[i][j] = scores_max[i][j];
                }
            }
            // max = -inf
            #pragma unroll
            for (size_t i=0; i<BM/(MMA_M*2); i++) {
                #pragma unroll
                for (size_t j=0; j<2; j++) {
                    scores_max[i][j] = -FLT_MAX;
                }
            }
            // reduce max
            #pragma unroll
            for (size_t i=0; i<BM/(MMA_M*2); i++) {
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
            // __shfl_xor_sync
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
            // max(prev, now)
            #pragma unroll
            for (size_t i=0; i<BM/(MMA_M*2); i++) {
                #pragma unroll
                for (size_t j=0; j<2; j++) {
                    scores_max[i][j] = max(scores_max_prev[i][j], scores_max[i][j]);
                }
            }
            // scores_scale = exp2(max_prev * scale  - max * scale)
            #pragma unroll
            for (size_t i=0; i<BM/(MMA_M*2); i++) {
                #pragma unroll
                for (size_t j=0; j<2; j++) {
                    scores_scale[i][j] = exp2f(scores_max_prev[i][j] * scale - scores_max[i][j] * scale);
                }
            }
            // acc_s = exp2(acc_s * scale - max)
            #pragma unroll
            for (size_t i=0; i<BM/(MMA_M*2); i++) {
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
            // reduce sum
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
                        #pragma unroll
                        for (size_t l=0; l<8; l++) {
                            int row = (l >> 1) & 1;
                            scores_sum[i][row] +=  acc_s[i][j][k][l];
                        }
                    }
                }
            }
            // __shfl_xor_sync
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
                        #pragma unroll
                        for (size_t l=0; l<8; l++) {
                            int row = (l >> 1) & 1;
                            acc_o[i][j][k][l] *= scores_scale[i][row];
                        }
                    }
                }
            }
        } else {
            // fill acc_s
            #pragma unroll
            for (size_t i=0; i<BM/(MMA_M*2); i++) {
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
            // gemm-qk
            wait(&Kfull[0], 0);
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
                        fp16 *_KAddr = sK + tma_smem_offset_2d<BN>(k_row, k_col);
                        wgmma_ss<QK_MMA_N, 1, 1, 1, 0, 0, 16, 1024, true>(acc_s[i][j], _QAddr, _KAddr);
                    }
                }
            }
            warpgroup_commit_batch();
            warpgroup_wait();
            arrive(&Kempty[0]);
        }
        bar_sync(256, BAR_2);
        
        // step.2 main for loop
        for (size_t i=0; i<(S/BN-1)*2; i++) {
            size_t wg0_pv__wg1_sm_pv = i / 2;
            size_t wg0_qk_sm__wg1_qk = i / 2 + 1;
            size_t wg0_pv__wg1_sm_pv_buf_idx = wg0_pv__wg1_sm_pv % NUM_SMEM;
            size_t wg0_qk_sm__wg1_qk_buf_idx = wg0_qk_sm__wg1_qk % NUM_SMEM;
            size_t wg0_pv__wg1_sm_pv_phase = (wg0_pv__wg1_sm_pv / NUM_SMEM) & 1;
            size_t wg0_qk_sm__wg1_qk_phase = (wg0_qk_sm__wg1_qk / NUM_SMEM) & 1;
            size_t solt_phase = i & 1;

            // addr
            fp16 *KAddr = sK + wg0_qk_sm__wg1_qk_buf_idx * BN * DIM;
            fp16 *VAddr = sV + wg0_pv__wg1_sm_pv_buf_idx * BN * DIM;

            // iteration
            if (wg_idx == solt_phase) {
                // gemm-pv
                wait(&Vfull[wg0_pv__wg1_sm_pv_buf_idx], wg0_pv__wg1_sm_pv_phase);
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
                            const int p_tile_inner = (k % QK_MMA_N) / MMA_K;
                            fp16 *_VAddr = VAddr + tma_smem_offset_2d<BN>(v_row, v_col);
                            wgmma_rs<PV_MMA_N, 1, 1, 1, 0, 1, 16, 1024, true>(
                                acc_o[i][j],
                                reinterpret_cast<uint32_t*>(acc_s_cast[i][p_tile_outer][p_tile_inner]),
                                _VAddr);
                        }
                    }
                }
                warpgroup_commit_batch();
                warpgroup_wait();
                if (solt_phase) arrive(&Vempty[wg0_pv__wg1_sm_pv_buf_idx]);
                // fill acc_s
                #pragma unroll
                for (size_t i=0; i<BM/(MMA_M*2); i++) {
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
                // gemm-qk
                wait(&Kfull[wg0_qk_sm__wg1_qk_buf_idx], wg0_qk_sm__wg1_qk_phase);
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
                if (solt_phase) arrive(&Kempty[wg0_qk_sm__wg1_qk_buf_idx]);
            } else {
                // softmax
                // max_prev = max
                #pragma unroll
                for (size_t i=0; i<BM/(MMA_M*2); i++) {
                    #pragma unroll
                    for (size_t j=0; j<2; j++) {
                        scores_max_prev[i][j] = scores_max[i][j];
                    }
                }
                // max = -inf
                #pragma unroll
                for (size_t i=0; i<BM/(MMA_M*2); i++) {
                    #pragma unroll
                    for (size_t j=0; j<2; j++) {
                        scores_max[i][j] = -FLT_MAX;
                    }
                }
                // reduce max
                #pragma unroll
                for (size_t i=0; i<BM/(MMA_M*2); i++) {
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
                // __shfl_xor_sync
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
                // max(prev, now)
                #pragma unroll
                for (size_t i=0; i<BM/(MMA_M*2); i++) {
                    #pragma unroll
                    for (size_t j=0; j<2; j++) {
                        scores_max[i][j] = max(scores_max_prev[i][j], scores_max[i][j]);
                    }
                }
                // scores_scale = exp2(max_prev * scale  - max * scale)
                #pragma unroll
                for (size_t i=0; i<BM/(MMA_M*2); i++) {
                    #pragma unroll
                    for (size_t j=0; j<2; j++) {
                        scores_scale[i][j] = exp2f(scores_max_prev[i][j] * scale - scores_max[i][j] * scale);
                    }
                }
                // acc_s = exp2(acc_s * scale - max)
                #pragma unroll
                for (size_t i=0; i<BM/(MMA_M*2); i++) {
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
                // reduce sum
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
                            #pragma unroll
                            for (size_t l=0; l<8; l++) {
                                int row = (l >> 1) & 1;
                                scores_sum[i][row] +=  acc_s[i][j][k][l];
                            }
                        }
                    }
                }
                // __shfl_xor_sync
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
                            #pragma unroll
                            for (size_t l=0; l<8; l++) {
                                int row = (l >> 1) & 1;
                                acc_o[i][j][k][l] *= scores_scale[i][row];
                            }
                        }
                    }
                }
            }
            bar_sync(256, BAR_3);
        }
        
        // step.3 epilogue
        size_t last_tile = (S / BN) - 1;
        size_t buf_idx = last_tile % NUM_SMEM;
        size_t phase = (last_tile / NUM_SMEM) & 1;
        fp16 *VAddr = sV + buf_idx * BN * DIM;
        if (wg_idx == 0) {
            // gemm-pv
            wait(&Vfull[buf_idx], phase);
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
                        const int p_tile_inner = (k % QK_MMA_N) / MMA_K;
                        fp16 *_VAddr = VAddr + tma_smem_offset_2d<BN>(v_row, v_col);
                        wgmma_rs<PV_MMA_N, 1, 1, 1, 0, 1, 16, 1024, true>(
                            acc_o[i][j],
                            reinterpret_cast<uint32_t*>(acc_s_cast[i][p_tile_outer][p_tile_inner]),
                            _VAddr);
                    }
                }
            }
            warpgroup_commit_batch();
            warpgroup_wait();
        } else {
            // softmax
            // max_prev = max
            #pragma unroll
            for (size_t i=0; i<BM/(MMA_M*2); i++) {
                #pragma unroll
                for (size_t j=0; j<2; j++) {
                    scores_max_prev[i][j] = scores_max[i][j];
                }
            }
            // max = -inf
            #pragma unroll
            for (size_t i=0; i<BM/(MMA_M*2); i++) {
                #pragma unroll
                for (size_t j=0; j<2; j++) {
                    scores_max[i][j] = -FLT_MAX;
                }
            }
            // reduce max
            #pragma unroll
            for (size_t i=0; i<BM/(MMA_M*2); i++) {
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
            // __shfl_xor_sync
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
            // max(prev, now)
            #pragma unroll
            for (size_t i=0; i<BM/(MMA_M*2); i++) {
                #pragma unroll
                for (size_t j=0; j<2; j++) {
                    scores_max[i][j] = max(scores_max_prev[i][j], scores_max[i][j]);
                }
            }
            // scores_scale = exp2(max_prev * scale  - max * scale)
            #pragma unroll
            for (size_t i=0; i<BM/(MMA_M*2); i++) {
                #pragma unroll
                for (size_t j=0; j<2; j++) {
                    scores_scale[i][j] = exp2f(scores_max_prev[i][j] * scale - scores_max[i][j] * scale);
                }
            }
            // acc_s = exp2(acc_s * scale - max)
            #pragma unroll
            for (size_t i=0; i<BM/(MMA_M*2); i++) {
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
            // reduce sum
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
                        #pragma unroll
                        for (size_t l=0; l<8; l++) {
                            int row = (l >> 1) & 1;
                            scores_sum[i][row] +=  acc_s[i][j][k][l];
                        }
                    }
                }
            }
            // __shfl_xor_sync
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
                        #pragma unroll
                        for (size_t l=0; l<8; l++) {
                            int row = (l >> 1) & 1;
                            acc_o[i][j][k][l] *= scores_scale[i][row];
                        }
                    }
                }
            }
            
            // gemm-pv
            wait(&Vfull[buf_idx], phase);
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
                        const int p_tile_inner = (k % QK_MMA_N) / MMA_K;
                        fp16 *_VAddr = VAddr + tma_smem_offset_2d<BN>(v_row, v_col);
                        wgmma_rs<PV_MMA_N, 1, 1, 1, 0, 1, 16, 1024, true>(
                            acc_o[i][j],
                            reinterpret_cast<uint32_t*>(acc_s_cast[i][p_tile_outer][p_tile_inner]),
                            _VAddr);
                    }
                }
            }
            warpgroup_commit_batch();
            warpgroup_wait();
            arrive(&Vempty[buf_idx]);
        }
        bar_sync(256, BAR_4);

        
        // acc_o = acc_o * scores_scale
        #pragma unroll
        for (size_t i=0; i<BM/(MMA_M*2); i++) {
            #pragma unroll
            for (size_t j=0; j<DIM/PV_MMA_N; j++) {
                #pragma unroll
                for (size_t k=0; k<PV_MMA_N/16; k++) {
                    #pragma unroll
                    for (size_t l=0; l<8; l++) {
                        int row = (l >> 1) & 1;
                        acc_o[i][j][k][l] /= logsum[i][row];
                    }
                }
            }
        }
        // copy acc_o to sO (load matritx)
        if (tid == 0) {
            tma_store_wait();
        }
        fp16 d_fp16[8];
        uint32_t* data_ptr = (uint32_t*)d_fp16;
        const int lane_row = lane_id & 0xf;         // 0..15
        const int lane_col = (lane_id >> 4) * 8;    // 0 or 8
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
                    #pragma unroll
                    for (size_t l=0; l<8; l++) {
                        d_fp16[l] = (fp16)(acc_o[i][j][k][l]);
                    }
                    stmatrix_x4<fp16>(_sO, data_ptr);
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
void attnWSPingpongMaxKernel(
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
    // setting shared memory
    extern __shared__ __align__(128) uint8_t smem[];
    SMemWSPingpong<BM, BN, DIM, NUM_SMEM> &s = *reinterpret_cast<SMemWSPingpong<BM, BN, DIM, NUM_SMEM>*>(smem);
    fp16 *sQ = s.Q, *sK = s.K, *sV = s.V, *sO = s.O;
    Barrier *Qmbar = &s.Qmbar;
    Barrier *Kempty = s.Kempty, *Vempty = s.Vempty, *Kfull = s.Kfull, *Vfull = s.Vfull;
    Barrier *Ppmbar = s.Ppmbar;
    // init mbarrier
    if (threadIdx.x == 0) {
        init_barrier(Qmbar, 1);
        for (int i = 0; i < NUM_SMEM; ++i) {
            init_barrier(&Kfull[i], 1);  // 1 thread arrive
            init_barrier(&Vfull[i], 1);
            init_barrier(&Kempty[i], 256);  // 256 thread arrive
            init_barrier(&Vempty[i], 256);
        }
        init_barrier(&Ppmbar[0], 1);
        init_barrier(&Ppmbar[1], 1);
        arrive(&Ppmbar[0]);
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
        
        int smem_i = 0, phase = 0;
        fp32 scale = sqrt((1.0f / DIM)) * 1.44269504f;  // log2(e)

        fp16 acc_s_cast[BM/(MMA_M*2)][BN/QK_MMA_N][QK_MMA_N/16][8];  // define acc_s_cast
        fp32 acc_o[BM/(MMA_M*2)][DIM/PV_MMA_N][PV_MMA_N/16][8];      // define acc_o
        // others
        fp32 scores_max[BM/(MMA_M*2)][2];
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
        uint32_t wg_idx = tid >> 7;
        uint32_t lane_id = tid & 31;
        uint32_t warp_id_in_wg = (tid >> 5) & 0x3;  // local warp id inside each 128-thread warpgroup
        wait(Qmbar, 0);

        // =====  prologue  =====
        fp32 acc_s[BM/(MMA_M*2)][BN/QK_MMA_N][QK_MMA_N/16][8];       // define acc_s
        // fill acc_s
        #pragma unroll
        for (size_t i=0; i<BM/(MMA_M*2); i++) {
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
        arrive(&Kempty[smem_i]);  // 释放 tma K 前阻塞

        // softmax
        fp32 scores_max_prev[BM/(MMA_M*2)][2];
        fp32 scores_scale[BM/(MMA_M*2)][2];
        fp32 scores_sum[BM/(MMA_M*2)][2];
        // max_prev = max
        #pragma unroll
        for (size_t i=0; i<BM/(MMA_M*2); i++) {
            #pragma unroll
            for (size_t j=0; j<2; j++) {
                scores_max_prev[i][j] = scores_max[i][j];
            }
        }
        // max = -inf
        #pragma unroll
        for (size_t i=0; i<BM/(MMA_M*2); i++) {
            #pragma unroll
            for (size_t j=0; j<2; j++) {
                scores_max[i][j] = -FLT_MAX;
            }
        }
        // reduce max
        #pragma unroll
        for (size_t i=0; i<BM/(MMA_M*2); i++) {
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
        // __shfl_xor_sync
        const unsigned mask = __activemask();
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
        // max(prev, now)
        #pragma unroll
        for (size_t i=0; i<BM/(MMA_M*2); i++) {
            #pragma unroll
            for (size_t j=0; j<2; j++) {
                scores_max[i][j] = max(scores_max_prev[i][j], scores_max[i][j]);
            }
        }
        // scores_scale = exp2(max_prev * scale  - max * scale)
        #pragma unroll
        for (size_t i=0; i<BM/(MMA_M*2); i++) {
            #pragma unroll
            for (size_t j=0; j<2; j++) {
                scores_scale[i][j] = exp2f(scores_max_prev[i][j] * scale - scores_max[i][j] * scale);
            }
        }
        // acc_s = exp2(acc_s * scale - max)
        #pragma unroll
        for (size_t i=0; i<BM/(MMA_M*2); i++) {
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
        // reduce sum
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
                    #pragma unroll
                    for (size_t l=0; l<8; l++) {
                        int row = (l >> 1) & 1;
                        scores_sum[i][row] +=  acc_s[i][j][k][l];
                    }
                }
            }
        }
        // __shfl_xor_sync
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
                    #pragma unroll
                    for (size_t l=0; l<8; l++) {
                        int row = (l >> 1) & 1;
                        acc_o[i][j][k][l] *= scores_scale[i][row];
                    }
                }
            }
        }
        smem_i++;

        // =====  main loop  =====
        constexpr int PP_SYNC_BAR_BASE = 5;
        int const consumer_id = static_cast<int>(wg_idx);
        int const next_consumer_id = consumer_id ^ 1;
        for (size_t iw=BN, pp_iter=0; iw<S; iw+=BN, ++smem_i, ++pp_iter) {
            if (smem_i == NUM_SMEM) { smem_i = 0; phase ^= 1; }
            int prev_idx = (smem_i + NUM_SMEM -1) % NUM_SMEM;
            int prve_phase = phase;
            if (prev_idx == NUM_SMEM -1) { prve_phase ^= 1; }
            fp16 *KAddr = sK + smem_i * BN * DIM;
            
            int p = pp_iter & 1;
            wait(&Ppmbar[consumer_id], p);
            fp32 acc_s[BM/(MMA_M*2)][BN/QK_MMA_N][QK_MMA_N/16][8];       // define acc_s
            // fill acc_s
            #pragma unroll
            for (size_t i=0; i<BM/(MMA_M*2); i++) {
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
            // +++++++++++++++ qk(j)

            // gemm-pv
            fp16 *VAddr = sV + prev_idx * BN * DIM;
            wait(&Vfull[prev_idx], prve_phase);
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
                        const int p_tile_inner = (k % QK_MMA_N) / MMA_K;
                        fp16 *_VAddr = VAddr + tma_smem_offset_2d<BN>(v_row, v_col);
                        wgmma_rs<PV_MMA_N, 1, 1, 1, 0, 1, 16, 1024, true>(
                            acc_o[i][j],
                            reinterpret_cast<uint32_t*>(acc_s_cast[i][p_tile_outer][p_tile_inner]),
                            _VAddr);
                    }
                }
            }
            warpgroup_commit_batch();
            // +++++++++++++++++++++ pv(j-1)

            // Ppmbar is a per-WG single-arrival token. After this WG has
            // issued QK and PV, release the other consumer WG and continue
            // with waits / softmax locally.
            bar_sync(128, PP_SYNC_BAR_BASE + consumer_id);
            if ((tid & 127) == 0) {
                arrive(&Ppmbar[next_consumer_id]);
            }

            // Two WGMMA groups are outstanding: QK then PV. Wait for QK so
            // acc_s is ready for softmax, while PV may still be running.
            warpgroup_wait<1>();
            arrive(&Kempty[smem_i]);  // 释放 tma K 前阻塞

            // softmax
            fp32 scores_max_prev[BM/(MMA_M*2)][2];
            fp32 scores_scale[BM/(MMA_M*2)][2];
            fp32 scores_sum[BM/(MMA_M*2)][2];
            // max_prev = max
            #pragma unroll
            for (size_t i=0; i<BM/(MMA_M*2); i++) {
                #pragma unroll
                for (size_t j=0; j<2; j++) {
                    scores_max_prev[i][j] = scores_max[i][j];
                }
            }
            // max = -inf
            #pragma unroll
            for (size_t i=0; i<BM/(MMA_M*2); i++) {
                #pragma unroll
                for (size_t j=0; j<2; j++) {
                    scores_max[i][j] = -FLT_MAX;
                }
            }
            // reduce max
            #pragma unroll
            for (size_t i=0; i<BM/(MMA_M*2); i++) {
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
            // __shfl_xor_sync
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
            // max(prev, now)
            #pragma unroll
            for (size_t i=0; i<BM/(MMA_M*2); i++) {
                #pragma unroll
                for (size_t j=0; j<2; j++) {
                    scores_max[i][j] = max(scores_max_prev[i][j], scores_max[i][j]);
                }
            }
            // scores_scale = exp2(max_prev * scale  - max * scale)
            #pragma unroll
            for (size_t i=0; i<BM/(MMA_M*2); i++) {
                #pragma unroll
                for (size_t j=0; j<2; j++) {
                    scores_scale[i][j] = exp2f(scores_max_prev[i][j] * scale - scores_max[i][j] * scale);
                }
            }
            // acc_s = exp2(acc_s * scale - max)
            #pragma unroll
            for (size_t i=0; i<BM/(MMA_M*2); i++) {
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
            // reduce sum
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
                        #pragma unroll
                        for (size_t l=0; l<8; l++) {
                            int row = (l >> 1) & 1;
                            scores_sum[i][row] +=  acc_s[i][j][k][l];
                        }
                    }
                }
            }
            // __shfl_xor_sync
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
            warpgroup_wait<0>();
            arrive(&Vempty[prev_idx]);  // 释放 tma V 前阻塞
            #pragma unroll
            for (size_t i=0; i<BM/(MMA_M*2); i++) {
                #pragma unroll
                for (size_t j=0; j<DIM/PV_MMA_N; j++) {
                    #pragma unroll
                    for (size_t k=0; k<PV_MMA_N/16; k++) {
                        #pragma unroll
                        for (size_t l=0; l<8; l++) {
                            int row = (l >> 1) & 1;
                            acc_o[i][j][k][l] *= scores_scale[i][row];
                        }
                    }
                }
            }
        }
        
        // ======  epilogue  ======
        // gemm-pv
        int last_idx = (smem_i % NUM_SMEM + NUM_SMEM - 1) % NUM_SMEM;
        int last_phase = phase;
        fp16 *VAddr = sV + last_idx * BN * DIM;
        wait(&Vfull[last_idx], last_phase);
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
                    const int p_tile_inner = (k % QK_MMA_N) / MMA_K;
                    fp16 *_VAddr = VAddr + tma_smem_offset_2d<BN>(v_row, v_col);
                    wgmma_rs<PV_MMA_N, 1, 1, 1, 0, 1, 16, 1024, true>(
                        acc_o[i][j],
                        reinterpret_cast<uint32_t*>(acc_s_cast[i][p_tile_outer][p_tile_inner]),
                        _VAddr);
                }
            }
        }
        warpgroup_commit_batch();
        warpgroup_wait();
        arrive(&Vempty[last_idx]);  // 释放 tma V 前阻塞

        // acc_o = acc_o * scores_scale
        #pragma unroll
        for (size_t i=0; i<BM/(MMA_M*2); i++) {
            #pragma unroll
            for (size_t j=0; j<DIM/PV_MMA_N; j++) {
                #pragma unroll
                for (size_t k=0; k<PV_MMA_N/16; k++) {
                    #pragma unroll
                    for (size_t l=0; l<8; l++) {
                        int row = (l >> 1) & 1;
                        acc_o[i][j][k][l] /= logsum[i][row];
                    }
                }
            }
        }
        // copy acc_o to sO (load matritx)
        if (tid == 0) {
            tma_store_wait();
        }
        const int lane_row = lane_id & 0xf;         // 0..15
        const int lane_col = (lane_id >> 4) * 8;    // 0 or 8
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
                    fp16 d_fp16[8];
                    uint32_t* data_ptr = (uint32_t*)d_fp16;
                    #pragma unroll
                    for (size_t l=0; l<8; l++) {
                        d_fp16[l] = (fp16)(acc_o[i][j][k][l]);
                    }
                    stmatrix_x4<fp16>(_sO, data_ptr);
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
    constexpr int PV_MMA_N = DIM <= 256 ? DIM : 256;
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
        warpgroup_reg_alloc<128>();

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
                        // int offset = nc * BM + i * MMA_M + warp_id_in_wg * 16 + lane_row * 2;
                        // float2 tmp_max = *reinterpret_cast<float2*>(&smem_max[offset]);
                        // float2 tmp_sum = *reinterpret_cast<float2*>(&smem_sum[offset]);
                        float2 tmp_max, tmp_sum;

                        if (lane_id < 16) {
                            int offset = nc * BM
                                    + i * MMA_M
                                    + warp_id_in_wg * 16
                                    + lane_row * 2;

                            tmp_max = *reinterpret_cast<float2*>(&smem_max[offset]);
                            tmp_sum = *reinterpret_cast<float2*>(&smem_sum[offset]);
                        }

                        // lane 0..15 的结果广播到 lane 16..31
                        tmp_max.x = __shfl_sync(0xffffffff, tmp_max.x, lane_row);
                        tmp_max.y = __shfl_sync(0xffffffff, tmp_max.y, lane_row);
                        tmp_sum.x = __shfl_sync(0xffffffff, tmp_sum.x, lane_row);
                        tmp_sum.y = __shfl_sync(0xffffffff, tmp_sum.y, lane_row);
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
        warpgroup_reg_alloc<128>();
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
                            const int p_tile_inner = (k % QK_MMA_N) / MMA_K;
                            uint32_t *_PAddr = reinterpret_cast<uint32_t*>(acc_s_cast[i][p_tile_outer][p_tile_inner]);
                            fp16 *_VAddr = sV + (wg_idx * NUM_SMEM + smem_i) * BN * DIM + tma_smem_offset_2d<BN>(k, j * PV_MMA_N);
                            wgmma_rs<PV_MMA_N, 1, 1, 1, 0, 1, 16, 1024, true>(acc_o[i][j], _PAddr, _VAddr);
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


template<int B, int H, int S, int D=128, int BM=128, int BN=128, int NUM_THREADS=256, int NUM_SMEM=1>
void runAttnKernel(fp16 *Q, fp16 *K, fp16 *V, fp16 *O) {
    CUtensorMap d_tma_map_Q = create_tensor_map<BM, D>(Q, B, H, S, D);
    CUtensorMap d_tma_map_K = create_tensor_map<BN, D>(K, B, H, S, D);
    CUtensorMap d_tma_map_V = create_tensor_map<BN, D>(V, B, H, S, D);
    CUtensorMap d_tma_map_O = create_tensor_map<BM, D, false>(O, B, H, S, D);

    auto* kernel = attnKernel<BM, BN, D, NUM_THREADS, NUM_SMEM>;
    constexpr size_t sMemSize = sizeof(SMem<BM, BN, D, NUM_SMEM>);
    static_assert(sMemSize < 256 * 1024);
    cudaCheck(cudaFuncSetAttribute(
        kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, sMemSize));

    dim3 grid = {static_cast<unsigned int>(S/BM), static_cast<unsigned int>(H), static_cast<unsigned int>(B)};
    kernel<<<grid, NUM_THREADS, sMemSize>>>(B, H, S, d_tma_map_Q, d_tma_map_K, d_tma_map_V, d_tma_map_O);
}


template<int B, int H, int S, int D=128, int BM=128, int BN=128, int NUM_THREADS=384, int NUM_SMEM=1>
void runAttnWSKernel(fp16 *Q, fp16 *K, fp16 *V, fp16 *O) {
    CUtensorMap d_tma_map_Q = create_tensor_map<BM, D>(Q, B, H, S, D);
    CUtensorMap d_tma_map_K = create_tensor_map<BN, D>(K, B, H, S, D);
    CUtensorMap d_tma_map_V = create_tensor_map<BN, D>(V, B, H, S, D);
    CUtensorMap d_tma_map_O = create_tensor_map<BM, D, false>(O, B, H, S, D);

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
void runAttnWSStage2Kernel(fp16 *Q, fp16 *K, fp16 *V, fp16 *O) {
    CUtensorMap d_tma_map_Q = create_tensor_map<BM, D>(Q, B, H, S, D);
    CUtensorMap d_tma_map_K = create_tensor_map<BN, D>(K, B, H, S, D);
    CUtensorMap d_tma_map_V = create_tensor_map<BN, D>(V, B, H, S, D);
    CUtensorMap d_tma_map_O = create_tensor_map<BM, D, false>(O, B, H, S, D);

    static_assert(NUM_SMEM >= 2);
    auto* kernel = attnWSStage2Kernel<BM, BN, D, NUM_THREADS, NUM_SMEM>;
    constexpr size_t sMemSize = sizeof(SMemWS<BM, BN, D, NUM_SMEM>);
    static_assert(sMemSize < 256 * 1024);
    cudaCheck(cudaFuncSetAttribute(
        kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, sMemSize));

    dim3 grid = {static_cast<unsigned int>(S/BM), static_cast<unsigned int>(H), static_cast<unsigned int>(B)};
    kernel<<<grid, NUM_THREADS, sMemSize>>>(B, H, S, d_tma_map_Q, d_tma_map_K, d_tma_map_V, d_tma_map_O);
}


template<int B, int H, int S, int D=128, int BM=128, int BN=128, int NUM_THREADS=384, int NUM_SMEM=3>
void runAttnWSStage3Kernel(fp16 *Q, fp16 *K, fp16 *V, fp16 *O) {
    CUtensorMap d_tma_map_Q = create_tensor_map<BM, D>(Q, B, H, S, D);
    CUtensorMap d_tma_map_K = create_tensor_map<BN, D>(K, B, H, S, D);
    CUtensorMap d_tma_map_V = create_tensor_map<BN, D>(V, B, H, S, D);
    CUtensorMap d_tma_map_O = create_tensor_map<BM, D, false>(O, B, H, S, D);

    static_assert(NUM_SMEM >= 3);
    auto* kernel = attnWSStage3Kernel<BM, BN, D, NUM_THREADS, NUM_SMEM>;
    constexpr size_t sMemSize = sizeof(SMemWS<BM, BN, D, NUM_SMEM>);
    static_assert(sMemSize < 256 * 1024);
    cudaCheck(cudaFuncSetAttribute(
        kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, sMemSize));

    dim3 grid = {static_cast<unsigned int>(S/BM), static_cast<unsigned int>(H), static_cast<unsigned int>(B)};
    kernel<<<grid, NUM_THREADS, sMemSize>>>(B, H, S, d_tma_map_Q, d_tma_map_K, d_tma_map_V, d_tma_map_O);
}

template<int B, int H, int S, int D=128, int BM=128, int BN=128, int NUM_THREADS=384, int NUM_SMEM=1>
void runAttnWSPingpongKernel(fp16 *Q, fp16 *K, fp16 *V, fp16 *O) {
    CUtensorMap d_tma_map_Q = create_tensor_map<BM, D>(Q, B, H, S, D);
    CUtensorMap d_tma_map_K = create_tensor_map<BN, D>(K, B, H, S, D);
    CUtensorMap d_tma_map_V = create_tensor_map<BN, D>(V, B, H, S, D);
    CUtensorMap d_tma_map_O = create_tensor_map<BM, D, false>(O, B, H, S, D);

    auto* kernel = attnWSPingpongKernel<BM, BN, D, NUM_THREADS, NUM_SMEM>;
    constexpr size_t sMemSize = sizeof(SMemWS<BM, BN, D, NUM_SMEM>);
    static_assert(sMemSize < 256 * 1024);
    cudaCheck(cudaFuncSetAttribute(
        kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, sMemSize));

    dim3 grid = {static_cast<unsigned int>(S/BM), static_cast<unsigned int>(H), static_cast<unsigned int>(B)};
    kernel<<<grid, NUM_THREADS, sMemSize>>>(B, H, S, d_tma_map_Q, d_tma_map_K, d_tma_map_V, d_tma_map_O);
}

template<int B, int H, int S, int D=128, int BM=128, int BN=128, int NUM_THREADS=384, int NUM_SMEM=2>
void runAttnWSPingpongMaxKernel(fp16 *Q, fp16 *K, fp16 *V, fp16 *O) {
    CUtensorMap d_tma_map_Q = create_tensor_map<BM, D>(Q, B, H, S, D);
    CUtensorMap d_tma_map_K = create_tensor_map<BN, D>(K, B, H, S, D);
    CUtensorMap d_tma_map_V = create_tensor_map<BN, D>(V, B, H, S, D);
    CUtensorMap d_tma_map_O = create_tensor_map<BM, D, false>(O, B, H, S, D);

    auto* kernel = attnWSPingpongMaxKernel<BM, BN, D, NUM_THREADS, NUM_SMEM>;
    constexpr size_t sMemSize = sizeof(SMemWSPingpong<BM, BN, D, NUM_SMEM>);
    static_assert(sMemSize < 256 * 1024);
    cudaCheck(cudaFuncSetAttribute(
        kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, sMemSize));

    dim3 grid = {static_cast<unsigned int>(S/BM), static_cast<unsigned int>(H), static_cast<unsigned int>(B)};
    kernel<<<grid, NUM_THREADS, sMemSize>>>(B, H, S, d_tma_map_Q, d_tma_map_K, d_tma_map_V, d_tma_map_O);
}

template<int B, int H, int S, int D=128, int BM=64, int BN=64, int NUM_THREADS=512, int NUM_SMEM=1, int NUM_CONSUMER=2, int N=2>
void runAttnWSCXKernel(fp16 *Q, fp16 *K, fp16 *V, fp16 *O) {
    CUtensorMap d_tma_map_Q = create_tensor_map<BM, D>(Q, B, H, S, D);
    CUtensorMap d_tma_map_K = create_tensor_map<BN, D>(K, B, H, S, D);
    CUtensorMap d_tma_map_V = create_tensor_map<BN, D>(V, B, H, S, D);
    CUtensorMap d_tma_map_O = create_tensor_map<BM, D, false>(O, B, H, S, D);

    auto* kernel = attnWSCXKernel<BM, BN, D, NUM_THREADS, NUM_SMEM, NUM_CONSUMER, N>;
    constexpr size_t sMemSize = sizeof(SMemWSCX<BM, BN, D, NUM_SMEM, NUM_CONSUMER>);
    static_assert(sMemSize < 256 * 1024);
    cudaCheck(cudaFuncSetAttribute(
        kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, sMemSize));

    dim3 grid = {static_cast<unsigned int>(S/N/BM), static_cast<unsigned int>(H), static_cast<unsigned int>(B)};
    kernel<<<grid, NUM_THREADS, sMemSize>>>(B, H, S, d_tma_map_Q, d_tma_map_K, d_tma_map_V, d_tma_map_O);
}


// nvcc -std=c++17 -arch=sm_90a -O3 verify_attn.cu -o verify_attn_test -lcuda
int main() {
    constexpr int B = 1;
    constexpr int H = 1;
    constexpr int S = 512;
    constexpr int D = 128;
    // constexpr int BM = 128;
    constexpr int BM = 64;
    constexpr int BN = 64;

    constexpr float kAbsTol = 8e-2f;
    constexpr float kRelTol = 8e-2f;

    int dev = 0;
    cudaDeviceProp prop{};
    cudaCheck(cudaGetDevice(&dev));
    cudaCheck(cudaGetDeviceProperties(&prop, dev));
    printf("GPU: %s, compute capability: %d.%d\n", prop.name, prop.major, prop.minor);
    if (prop.major < 9) {
        printf("This kernel uses Hopper-only instructions (TMA/WGMMA). Need SM90+ GPU.\n");
        return 1;
    }

    static_assert(S % 128 == 0, "S must be divisible by 128 for this kernel config.");
    static_assert(D % 16 == 0, "D must be divisible by 16 for MMA_K=16.");

    const size_t numel = static_cast<size_t>(B) * H * S * D;
    const size_t bytes = numel * sizeof(fp16);

    std::vector<fp16> hQ(numel), hK(numel), hV(numel), hO(numel);
    std::vector<float> hRef(numel, 0.0f);

    auto idx4 = [=](int b, int h, int s, int d) {
        return (((static_cast<size_t>(b) * H + h) * S + s) * D + d);
    };

    for (int b = 0; b < B; ++b) {
        for (int h = 0; h < H; ++h) {
            for (int s = 0; s < S; ++s) {
                for (int d = 0; d < D; ++d) {
                    const size_t idx = idx4(b, h, s, d);
                    const int q_i = static_cast<int>((idx * 13 + 7) % 31) - 15;
                    const int k_i = static_cast<int>((idx * 17 + 5) % 29) - 14;
                    const int v_i = static_cast<int>((idx * 19 + 3) % 23) - 11;
                    const float q = static_cast<float>(q_i) * (1.0f / 128.0f);
                    const float k = static_cast<float>(k_i) * (1.0f / 128.0f);
                    const float v = static_cast<float>(v_i) * (1.0f / 128.0f);
                    hQ[idx] = __float2half_rn(q);
                    hK[idx] = __float2half_rn(k);
                    hV[idx] = __float2half_rn(v);
                }
            }
        }
    }

    fp16 *dQ = nullptr, *dK = nullptr, *dV = nullptr, *dO = nullptr;
    cudaCheck(cudaMalloc(&dQ, bytes));
    cudaCheck(cudaMalloc(&dK, bytes));
    cudaCheck(cudaMalloc(&dV, bytes));
    cudaCheck(cudaMalloc(&dO, bytes));

    cudaCheck(cudaMemcpy(dQ, hQ.data(), bytes, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(dK, hK.data(), bytes, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(dV, hV.data(), bytes, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemset(dO, 0, bytes));

    // runAttnKernel<B, H, S, D, BM, BN, 256, 1>(dQ, dK, dV, dO);
    // runAttnWSKernel<B, H, S, D, BM, BN, 384, 2>(dQ, dK, dV, dO);
    // runAttnWSStage2Kernel<B, H, S, D, BM, BN, 384, 2>(dQ, dK, dV, dO);
    // runAttnWSStage3Kernel<B, H, S, D, BM, BN, 384, 3>(dQ, dK, dV, dO);
    // runAttnWSPingpongKernel<B, H, S, D, BM, BN, 384, 1>(dQ, dK, dV, dO);
    // runAttnWSPingpongMaxKernel<B, H, S, D, BM, BN, 384, 2>(dQ, dK, dV, dO);
    runAttnWSCXKernel<B, H, S, D, BM, BN, /*num_thread*/512, /*num_smem*/1, /*num_consumer*/2, /*N*/2>(dQ, dK, dV, dO);
    cudaCheck(cudaGetLastError());
    cudaCheck(cudaDeviceSynchronize());
    cudaCheck(cudaMemcpy(hO.data(), dO, bytes, cudaMemcpyDeviceToHost));
    printf("Kernel Run Success!\n");

    // CPU reference: O = softmax(Q * K^T / sqrt(D)) * V (online softmax, tiled by BN).
    // Match kernel quantization path: cast softmax probs to fp16 before PV accumulation,
    // and cast final O to fp16 for comparison.
    const float softmax_scale = sqrtf(1.0f / static_cast<float>(D)) * 1.44269504f;  // log2(e)
    for (int b = 0; b < B; ++b) {
        for (int h = 0; h < H; ++h) {
            for (int qm = 0; qm < S; ++qm) {
                std::vector<float> out_row(D, 0.0f);
                float m_prev = -FLT_MAX;
                float l_prev = 0.0f;
                for (int dn = 0; dn < D; ++dn) {
                    out_row[dn] = 0.0f;
                }
                for (int ks0 = 0; ks0 < S; ks0 += BN) {
                    const int tile_n = (ks0 + BN <= S) ? BN : (S - ks0);
                    std::vector<float> scores(tile_n, 0.0f);
                    float m_tile = -FLT_MAX;
                    for (int t = 0; t < tile_n; ++t) {
                        const int ks = ks0 + t;
                        float score = 0.0f;
                        for (int kd = 0; kd < D; ++kd) {
                            const float q = __half2float(hQ[idx4(b, h, qm, kd)]);
                            const float k = __half2float(hK[idx4(b, h, ks, kd)]);
                            score += q * k;
                        }
                        scores[t] = score;
                        m_tile = fmaxf(m_tile, score);
                    }

                    const float m_new = fmaxf(m_prev, m_tile);
                    const float alpha = exp2f((m_prev - m_new) * softmax_scale);

                    for (int dn = 0; dn < D; ++dn) {
                        out_row[dn] *= alpha;
                    }

                    float l_new = l_prev * alpha;
                    for (int t = 0; t < tile_n; ++t) {
                        const int ks = ks0 + t;
                        const float p = exp2f((scores[t] - m_new) * softmax_scale);
                        const float p_half = __half2float(__float2half_rn(p));
                        l_new += p;
                        for (int dn = 0; dn < D; ++dn) {
                            const float v = __half2float(hV[idx4(b, h, ks, dn)]);
                            out_row[dn] += p_half * v;
                        }
                    }

                    m_prev = m_new;
                    l_prev = l_new;
                }

                for (int dn = 0; dn < D; ++dn) {
                    const float out = out_row[dn] / l_prev;
                    hRef[idx4(b, h, qm, dn)] = __half2float(__float2half_rn(out));
                }
            }
        }
    }

    int mismatch = 0;
    float max_abs_err = 0.0f;
    size_t max_abs_idx = 0;
    for (size_t i = 0; i < numel; ++i) {
        const float got = __half2float(hO[i]);
        const float ref = hRef[i];
        const bool bad_value = !std::isfinite(got) || !std::isfinite(ref);
        const float abs_err = bad_value ? INFINITY : fabsf(got - ref);
        const float tol = kAbsTol + kRelTol * fabsf(ref);
        if (abs_err > max_abs_err) {
            max_abs_err = abs_err;
            max_abs_idx = i;
        }
        if (bad_value || abs_err > tol) {
            if (mismatch < 10) {
                printf("Mismatch[%d] idx=%zu got=%f ref=%f abs_err=%f tol=%f bad_value=%d\n",
                       mismatch, i, got, ref, abs_err, tol, static_cast<int>(bad_value));
            }
            ++mismatch;
        }
    }

    if (mismatch == 0) {
        printf("PASS: O matches reference. max_abs_err=%f at idx=%zu\n", max_abs_err, max_abs_idx);
    } else {
        printf("FAIL: mismatch=%d / %zu, max_abs_err=%f at idx=%zu\n",
               mismatch, numel, max_abs_err, max_abs_idx);
    }

    cudaCheck(cudaFree(dQ));
    cudaCheck(cudaFree(dK));
    cudaCheck(cudaFree(dV));
    cudaCheck(cudaFree(dO));

    return mismatch == 0 ? 0 : 1;
}
