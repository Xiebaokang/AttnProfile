#include "tools.cuh"

constexpr int kProfileSlots = 32;

struct ProfileResult {
    int kv_iters;
    unsigned long long load_q_start;
    unsigned long long load_q_end;
    unsigned long long load_k_starts[kProfileSlots];
    unsigned long long load_k_ends[kProfileSlots];
    unsigned long long load_v_starts[kProfileSlots];
    unsigned long long load_v_ends[kProfileSlots];

    unsigned long long qk_starts[kProfileSlots];
    unsigned long long qk_ends[kProfileSlots];

    unsigned long long softmax_starts[kProfileSlots];
    unsigned long long softmax_ends[kProfileSlots];

    unsigned long long pv_starts[kProfileSlots];
    unsigned long long pv_ends[kProfileSlots];
};

struct ProfileSM {
    int kv_iters;
    // unsigned long long load_k_starts[kProfileSlots];
    // unsigned long long load_k_ends[kProfileSlots];
    // unsigned long long load_v_starts[kProfileSlots];
    // unsigned long long load_v_ends[kProfileSlots];
    unsigned long long starts[kProfileSlots];
    unsigned long long ends[kProfileSlots];
};

struct Profile {
    int kv_iters;
    unsigned long long qk_starts[kProfileSlots];
    unsigned long long pv_ends[kProfileSlots];
};

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


template <int BM, int BN, int DIM, int NUM_STAGE=1>
struct SMem {
    alignas(128) fp16 Q[BM*DIM];
    alignas(128) fp16 K[BN*DIM*NUM_STAGE];
    alignas(128) fp16 V[BN*DIM*NUM_STAGE];
    alignas(128) fp16 O[BM*DIM];
    alignas(8) uint64_t Qmbar;
    alignas(8) uint64_t Kmbar[NUM_STAGE];
    alignas(8) uint64_t Vmbar[NUM_STAGE];
};


template<int BM, int BN, int DIM, int NUM_THREADS, int NUM_STAGE>
__global__  __launch_bounds__(NUM_THREADS) 
void attnKernel(
    int B, int H, int S, 
    const __grid_constant__ CUtensorMap tensorMapQ, 
    const __grid_constant__ CUtensorMap tensorMapK, 
    const __grid_constant__ CUtensorMap tensorMapV, 
    const __grid_constant__ CUtensorMap tensorMapO,
    ProfileResult *__restrict__ profile
) {
    // 该 kernel 为串行的attention，仅仅测量 TC 和 CC 计算的时间
    const int bs = blockIdx.z;
    const int hn = blockIdx.y;
    const int by = blockIdx.x;
    const int tid = threadIdx.x;
    const bool do_profile = (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && tid == 0);
    const int kv_iters = S / BN;
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
    // constexpr int QK_MMA_N = BN <= 128 ? BN : 128;
    // constexpr int PV_MMA_N = DIM <= 128 ? DIM : 128;
    constexpr int MMA_K = 16;
    // setting shared memory
    extern __shared__ __align__(128) uint8_t smem[];
    SMem<BM, BN, DIM, NUM_STAGE> &s = *reinterpret_cast<SMem<BM, BN, DIM, NUM_STAGE>*>(smem);
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
        for (int i = 0; i < NUM_STAGE; ++i) {
            init_barrier(&Kmbar[i], 1);
            init_barrier(&Vmbar[i], 1);
        }
    }
    __syncthreads();
    fence_view_async_shared();
    if (do_profile) {
        profile->kv_iters = (kv_iters < kProfileSlots) ? kv_iters : kProfileSlots;
    }

    // init other args
    int stage = 0, phase = 0;
    fp32 scale = sqrt((1.0f / DIM)) * 1.44269504f;  // log2(e)
    unsigned long long load_q_start = 0;
    unsigned long long load_q_end = 0;

    // load Q
    if (do_profile) {
        load_q_start = clock64();
    }
    if (tid == 0) {
        expect_bytes(Qmbar, BM * DIM * sizeof(fp16));
        load_async(sQ, &tensorMapQ, Qmbar, bs, hn, by * BM, 0);
    }
    wait(Qmbar, 0);
    if (do_profile) {
        load_q_end = clock64();
        profile->load_q_start = load_q_start;
        profile->load_q_end = load_q_end;
    }

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
    for (size_t iw=0; iw<S; iw+=BN, ++stage) {
        if (stage >= NUM_STAGE) { stage = 0; phase ^= 1; }
        fp16 *KAddr = sK + stage * BN * DIM;
        fp16 *VAddr = sV + stage * BN * DIM; 
        const int slot = static_cast<int>(iw / BN);
        unsigned long long load_k_start = 0;
        unsigned long long qk_start = 0;
        unsigned long long qk_end = 0;
        unsigned long long softmax_start = 0;
        unsigned long long softmax_end = 0;
        unsigned long long load_v_start = 0;
        unsigned long long pv_start = 0;
        unsigned long long pv_end = 0;

        // load K
        if (do_profile && slot < kProfileSlots) {
            load_k_start = clock64();
        }
        if (tid == 0) {
            expect_bytes(&Kmbar[stage], BN * DIM * sizeof(fp16));
            load_async(KAddr, &tensorMapK, &Kmbar[stage], bs, hn, iw, 0);
        }
        wait(&Kmbar[stage], phase);
        if (do_profile && slot < kProfileSlots) {
            profile->load_k_starts[slot] = load_k_start;
            profile->load_k_ends[slot] = clock64();
        }

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

        if (do_profile && slot < kProfileSlots) {
            qk_start = clock64();
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
        if (do_profile && slot < kProfileSlots) {
            qk_end = clock64();
            profile->qk_starts[slot] = qk_start;
            profile->qk_ends[slot] = qk_end;
        }

        // max_prev = max
        if (do_profile && slot < kProfileSlots) {
            softmax_start = clock64();
        }
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
                        int row = (l / 2) % 2;
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
                        int row = (l / 2) % 2;
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
                        int row = (l / 2) % 2;
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
                        int row = (l / 2) % 2;
                        acc_o[i][j][k][l] *= scores_scale[i][row];
                    }
                }
            }
        }
        if (do_profile && slot < kProfileSlots) {
            softmax_end = clock64();
            profile->softmax_starts[slot] = softmax_start;
            profile->softmax_ends[slot] = softmax_end;
        }

        // load V
        if (do_profile && slot < kProfileSlots) {
            load_v_start = clock64();
        }
        if (tid == 0) {
            expect_bytes(&Vmbar[stage], BN * DIM * sizeof(fp16));
            load_async(VAddr, &tensorMapV, &Vmbar[stage], bs, hn, iw, 0);
        }
        wait(&Vmbar[stage], phase);
        if (do_profile && slot < kProfileSlots) {
            profile->load_v_starts[slot] = load_v_start;
            profile->load_v_ends[slot] = clock64();
        }

        // gemm-pv
        if (do_profile && slot < kProfileSlots) {
            pv_start = clock64();
        }
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
        if (do_profile && slot < kProfileSlots) {
            pv_end = clock64();
            profile->pv_starts[slot] = pv_start;
            profile->pv_ends[slot] = pv_end;
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
                    int row = (l / 2) % 2;
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
                fp16 *_sO = sO + tma_smem_offset_2d<BM>(o_row, o_col);
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


template<int BM, int BN, int DIM, int NUM_THREADS, int NUM_STAGE>
__global__  __launch_bounds__(NUM_THREADS) 
void attnNoSoftmaxKernel(
    int B, int H, int S, 
    const __grid_constant__ CUtensorMap tensorMapQ, 
    const __grid_constant__ CUtensorMap tensorMapK, 
    const __grid_constant__ CUtensorMap tensorMapV, 
    const __grid_constant__ CUtensorMap tensorMapO,
    ProfileSM *__restrict__ profile
) {
    const int bs = blockIdx.z;
    const int hn = blockIdx.y;
    const int by = blockIdx.x;
    const int tid = threadIdx.x;
    const bool do_profile = (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && tid == 0);
    const int kv_iters = S / BN;
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
    SMem<BM, BN, DIM, NUM_STAGE> &s = *reinterpret_cast<SMem<BM, BN, DIM, NUM_STAGE>*>(smem);
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
        for (int i = 0; i < NUM_STAGE; ++i) {
            init_barrier(&Kmbar[i], 1);
            init_barrier(&Vmbar[i], 1);
        }
    }
    __syncthreads();
    fence_view_async_shared();
    if (do_profile) {
        profile->kv_iters = (kv_iters < kProfileSlots) ? kv_iters : kProfileSlots;
    }

    // init other args
    int stage = 0, phase = 0;
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
    for (size_t iw=0; iw<S; iw+=BN, ++stage) {
        if (stage >= NUM_STAGE) { stage = 0; phase ^= 1; }
        fp16 *KAddr = sK + stage * BN * DIM;
        fp16 *VAddr = sV + stage * BN * DIM; 
        const int slot = static_cast<int>(iw / BN);
        // unsigned long long load_k_start = 0;
        // unsigned long long load_v_start = 0;
        unsigned long long start = 0;

        // load K
        // if (do_profile && slot < kProfileSlots) {
        //     load_k_start = clock64();
        // }
        if (tid == 0) {
            expect_bytes(&Kmbar[stage], BN * DIM * sizeof(fp16));
            load_async(KAddr, &tensorMapK, &Kmbar[stage], bs, hn, iw, 0);
        }
        wait(&Kmbar[stage], phase);
        // if (do_profile && slot < kProfileSlots) {
        //     profile->load_k_starts[slot] = load_k_start;
        //     profile->load_k_ends[slot] = clock64();
        // }

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

        // if (do_profile && slot < kProfileSlots) {
        //     start = clock64();
        // }
        // max_prev = max
        #pragma unroll
        for (size_t i=0; i<BM/(MMA_M*2); i++) {
            #pragma unroll
            for (size_t j=0; j<2; j++) {
                scores_max_prev[i][j] = scores_max[i][j];
            }
        }
        const unsigned mask = __activemask();
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
                        int row = (l / 2) % 2;
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
                        int row = (l / 2) % 2;
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
                        int row = (l / 2) % 2;
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
                        int row = (l / 2) % 2;
                        acc_o[i][j][k][l] *= scores_scale[i][row];
                    }
                }
            }
        }
        if (do_profile && slot < kProfileSlots) {
            profile->starts[slot] = start;
            profile->ends[slot] = clock64();
        }

        // load V
        // if (do_profile && slot < kProfileSlots) {
        //     load_v_start = clock64();
        // }
        if (tid == 0) {
            expect_bytes(&Vmbar[stage], BN * DIM * sizeof(fp16));
            load_async(VAddr, &tensorMapV, &Vmbar[stage], bs, hn, iw, 0);
        }
        wait(&Vmbar[stage], phase);
        // if (do_profile && slot < kProfileSlots) {
        //     profile->load_v_starts[slot] = load_v_start;
        //     profile->load_v_ends[slot] = clock64();
        // }
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
                    int row = (l / 2) % 2;
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
                fp16 *_sO = sO + tma_smem_offset_2d<BM>(o_row, o_col);
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
    const __grid_constant__ CUtensorMap tensorMapO,
    Profile *__restrict__ profile
) {
    // WS attention
    const int bs = blockIdx.z;
    const int hn = blockIdx.y;
    const int by = blockIdx.x;
    const int tid = threadIdx.x;
    uint32_t wg_idx = tid >> 7;
    const bool do_profile = (blockIdx.x == 4 && blockIdx.y == 5 && blockIdx.z == 0);
    const int kv_iters = S / BN;
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
    if (do_profile && tid == 0) {
        profile->kv_iters = (kv_iters < kProfileSlots) ? kv_iters : kProfileSlots;
    }

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
            const int slot = static_cast<int>(iw / BN);
            unsigned long long qk_start = 0;
            unsigned long long pv_end = 0;
            wait(&Kfull[smem_i], phase);
            if (do_profile && tid == 0 && slot < kProfileSlots) {
                qk_start = clock64();
                profile->qk_starts[slot] = qk_start;
            }
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
                            int row = (l / 2) % 2;
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
                            int row = (l / 2) % 2;
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
                            int row = (l / 2) % 2;
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
                            int row = (l / 2) % 2;
                            acc_o[i][j][k][l] *= scores_scale[i][row];
                        }
                    }
                }
            }
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
            if (do_profile && tid == 0 && slot < kProfileSlots) {
                pv_end = clock64();
                profile->pv_ends[slot] = pv_end;
            }
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
                        int row = (l / 2) % 2;
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
                    fp16 *_sO = sO + tma_smem_offset_2d<BM>(o_row, o_col);
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


template<int B, int H, int S, int D=128, int BM=128, int BN=128, int NUM_THREADS=256, int NUM_STAGE=1>
void runAttnKernel(fp16 *Q, fp16 *K, fp16 *V, fp16 *O, ProfileResult *profile, int profile_grid_x = -1) {
    CUtensorMap d_tma_map_Q = create_tensor_map<BM, D>(Q, B, H, S, D);
    CUtensorMap d_tma_map_K = create_tensor_map<BN, D>(K, B, H, S, D);
    CUtensorMap d_tma_map_V = create_tensor_map<BN, D>(V, B, H, S, D);
    CUtensorMap d_tma_map_O = create_tensor_map<BM, D, false>(O, B, H, S, D);

    auto* kernel = attnKernel<BM, BN, D, NUM_THREADS, NUM_STAGE>;
    constexpr size_t sMemSize = sizeof(SMem<BM, BN, D, NUM_STAGE>);
    static_assert(sMemSize < 256 * 1024);
    cudaCheck(cudaFuncSetAttribute(
        kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, sMemSize));

    const unsigned int grid_x = (profile_grid_x > 0) ? static_cast<unsigned int>(profile_grid_x)
                                                      : static_cast<unsigned int>(S / BM);
    dim3 grid = {grid_x, static_cast<unsigned int>(H), static_cast<unsigned int>(B)};
    kernel<<<grid, NUM_THREADS, sMemSize>>>(B, H, S, d_tma_map_Q, d_tma_map_K, d_tma_map_V, d_tma_map_O, profile);
}

template<int B, int H, int S, int D=128, int BM=128, int BN=128, int NUM_THREADS=256, int NUM_STAGE=1>
void runAttnNoSoftmaxKernel(fp16 *Q, fp16 *K, fp16 *V, fp16 *O, ProfileSM *profile) {
    CUtensorMap d_tma_map_Q = create_tensor_map<BM, D>(Q, B, H, S, D);
    CUtensorMap d_tma_map_K = create_tensor_map<BN, D>(K, B, H, S, D);
    CUtensorMap d_tma_map_V = create_tensor_map<BN, D>(V, B, H, S, D);
    CUtensorMap d_tma_map_O = create_tensor_map<BM, D, false>(O, B, H, S, D);

    auto* kernel = attnNoSoftmaxKernel<BM, BN, D, NUM_THREADS, NUM_STAGE>;
    constexpr size_t sMemSize = sizeof(SMem<BM, BN, D, NUM_STAGE>);
    static_assert(sMemSize < 256 * 1024);
    cudaCheck(cudaFuncSetAttribute(
        kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, sMemSize));
    dim3 grid = {static_cast<unsigned int>(S / BM), static_cast<unsigned int>(H), static_cast<unsigned int>(B)};
    // dim3 grid = {1, 14, 1};
    // dim3 grid = {1, 1, 1};
    kernel<<<grid, NUM_THREADS, sMemSize>>>(B, H, S, d_tma_map_Q, d_tma_map_K, d_tma_map_V, d_tma_map_O, profile);
}

template<int B, int H, int S, int D=128, int BM=128, int BN=128, int NUM_THREADS=384, int NUM_SMEM=1>
void runAttnWSKernel(fp16 *Q, fp16 *K, fp16 *V, fp16 *O, Profile *profile) {
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
    kernel<<<grid, NUM_THREADS, sMemSize>>>(B, H, S, d_tma_map_Q, d_tma_map_K, d_tma_map_V, d_tma_map_O, profile);
}


template<int B, int H, int S, int D, int BM, int BN, int NUM_THREADS=256, int NUM_STAGE=1>
void run_one_case(
    fp16 *dQ,
    fp16 *dK,
    fp16 *dV,
    fp16 *dO,
    ProfileResult *d_profile
) {
    ProfileResult h_profile{};
    ProfileResult zero_profile{};

    constexpr int kWarmupIters = 10;
    constexpr int kMeasureIters = 100;
    constexpr int kProfileGridX = 1;

    for (int i = 0; i < kWarmupIters; ++i) {
        cudaCheck(cudaMemcpy(
            d_profile,
            &zero_profile,
            sizeof(ProfileResult),
            cudaMemcpyHostToDevice));

        runAttnKernel<B, H, S, D, BM, BN, NUM_THREADS, NUM_STAGE>(
            dQ, dK, dV, dO, d_profile, kProfileGridX);

        cudaCheck(cudaGetLastError());
    }
    cudaCheck(cudaDeviceSynchronize());

    double sum_load_q = 0.0;
    double sum_load_k = 0.0;
    double sum_load_v = 0.0;
    double sum_gemm_qk = 0.0;
    double sum_softmax = 0.0;
    double sum_gemm_pv = 0.0;

    for (int i = 0; i < kMeasureIters; ++i) {
        cudaCheck(cudaMemcpy(
            d_profile,
            &zero_profile,
            sizeof(ProfileResult),
            cudaMemcpyHostToDevice));

        runAttnKernel<B, H, S, D, BM, BN, NUM_THREADS, NUM_STAGE>(
            dQ, dK, dV, dO, d_profile, kProfileGridX);

        cudaCheck(cudaGetLastError());

        cudaCheck(cudaMemcpy(
            &h_profile,
            d_profile,
            sizeof(ProfileResult),
            cudaMemcpyDeviceToHost));

        const int slots = (h_profile.kv_iters < kProfileSlots) ? h_profile.kv_iters : kProfileSlots;
        if (slots <= 0) {
            continue;
        }

        sum_load_q += static_cast<double>(h_profile.load_q_end - h_profile.load_q_start);

        double load_k = 0.0;
        double load_v = 0.0;
        double gemm_qk = 0.0;
        double softmax = 0.0;
        double gemm_pv = 0.0;

        for (int slot = 0; slot < slots; ++slot) {
            load_k += static_cast<double>(h_profile.load_k_ends[slot] - h_profile.load_k_starts[slot]);
            load_v += static_cast<double>(h_profile.load_v_ends[slot] - h_profile.load_v_starts[slot]);
            gemm_qk += static_cast<double>(h_profile.qk_ends[slot] - h_profile.qk_starts[slot]);
            softmax += static_cast<double>(h_profile.softmax_ends[slot] - h_profile.softmax_starts[slot]);
            gemm_pv += static_cast<double>(h_profile.pv_ends[slot] - h_profile.pv_starts[slot]);
        }

        const double inv_slots = 1.0 / static_cast<double>(slots);
        sum_load_k += load_k * inv_slots;
        sum_load_v += load_v * inv_slots;
        sum_gemm_qk += gemm_qk * inv_slots;
        sum_softmax += softmax * inv_slots;
        sum_gemm_pv += gemm_pv * inv_slots;
    }

    constexpr double inv_n = 1.0 / static_cast<double>(kMeasureIters);

    printf("| %d | %d | %.2f | %.2f | %.2f | %.2f | %.2f | %.2f |\n",
           BM,
           BN,
           sum_load_q * inv_n,
           sum_load_k * inv_n,
           sum_load_v * inv_n,
           sum_gemm_qk * inv_n,
           sum_softmax * inv_n,
           sum_gemm_pv * inv_n);
}

template<int B, int H, int S, int D, int BM, int BN, int NUM_THREADS=256, int NUM_STAGE=1>
void run_one_case_nosm(
    fp16 *dQ,
    fp16 *dK,
    fp16 *dV,
    fp16 *dO,
    ProfileSM *d_profile
) {
    ProfileSM h_profile{};
    ProfileSM zero_profile{};
    double cost[kProfileSlots];

    constexpr int kWarmupIters = 10;
    constexpr int kMeasureIters = 100;

    for (int i = 0; i < kWarmupIters; ++i) {
        cudaCheck(cudaMemcpy(
            d_profile,
            &zero_profile,
            sizeof(ProfileSM),
            cudaMemcpyHostToDevice));

        runAttnNoSoftmaxKernel<B, H, S, D, BM, BN, NUM_THREADS, NUM_STAGE>(
            dQ, dK, dV, dO, d_profile);

        cudaCheck(cudaGetLastError());
    }
    cudaCheck(cudaDeviceSynchronize());

    for (int i = 0; i < kMeasureIters; ++i) {
        cudaCheck(cudaMemcpy(
            d_profile,
            &zero_profile,
            sizeof(ProfileSM),
            cudaMemcpyHostToDevice));

        runAttnNoSoftmaxKernel<B, H, S, D, BM, BN, NUM_THREADS, NUM_STAGE>(
            dQ, dK, dV, dO, d_profile);

        cudaCheck(cudaGetLastError());

        cudaCheck(cudaMemcpy(
            &h_profile,
            d_profile,
            sizeof(ProfileSM),
            cudaMemcpyDeviceToHost));

        const int slots = (h_profile.kv_iters < kProfileSlots) ? h_profile.kv_iters : kProfileSlots;
        if (slots <= 0) {
            continue;
        }

        for (int slot = 0; slot < slots; ++slot) {
            cost[slot] = static_cast<double>(h_profile.ends[slot] - h_profile.starts[slot]);
        }
    }
    printf("| slot | BM | BN | cost(avg) |\n");
    printf("| ---: | ---: | ---: | ---: | ---: |\n");

    for (int slot = 0; slot < kProfileSlots; ++slot) {
        printf("| %d | %d | %d | %.2f |\n", slot, BM, BN, cost[slot]);
    }
    
}

template<int B, int H, int S, int D=128, int BM=128, int BN=128, int NUM_THREADS=384, int NUM_SMEM=1>
void run_one_case_ws(
    fp16 *dQ,
    fp16 *dK,
    fp16 *dV,
    fp16 *dO,
    Profile *d_profile
) {
    Profile zero_profile{};
    Profile h_profile{};

    constexpr int kWarmupIters = 10;

    for (int i = 0; i < kWarmupIters; ++i) {
        cudaCheck(cudaMemcpy(
            d_profile,
            &zero_profile,
            sizeof(Profile),
            cudaMemcpyHostToDevice));

        runAttnWSKernel<B, H, S, D, BM, BN, NUM_THREADS, NUM_SMEM>(
            dQ, dK, dV, dO, d_profile);

        cudaCheck(cudaGetLastError());
    }
    cudaCheck(cudaDeviceSynchronize());

    cudaCheck(cudaMemcpy(
        d_profile,
        &zero_profile,
        sizeof(Profile),
        cudaMemcpyHostToDevice));

    runAttnWSKernel<B, H, S, D, BM, BN, NUM_THREADS, NUM_SMEM>(
        dQ, dK, dV, dO, d_profile);

    cudaCheck(cudaGetLastError());
    cudaCheck(cudaDeviceSynchronize());

    cudaCheck(cudaMemcpy(
        &h_profile,
        d_profile,
        sizeof(Profile),
        cudaMemcpyDeviceToHost));

    const int slots = (h_profile.kv_iters < kProfileSlots) ? h_profile.kv_iters : kProfileSlots;
    if (slots <= 0) {
        printf("\n### attnWSKernel<BM=%d, BN=%d, smem=%d>\n\n", BM, BN, NUM_SMEM);
        printf("| slot | pv_end_abs | next_qk_start_abs | pv_end_rel | next_qk_start_rel | gap_cycles |\n");
        printf("| ---: | ---: | ---: | ---: | ---: | ---: |\n");
        return;
    }

    unsigned long long base = 0;
    auto update_base = [&](unsigned long long t) {
        if (t != 0 && (base == 0 || t < base)) {
            base = t;
        }
    };

    for (int i = 0; i < slots; ++i) {
        update_base(h_profile.qk_starts[i]);
        update_base(h_profile.pv_ends[i]);
    }

    printf("\n### attnWSKernel<BM=%d, BN=%d, smem=%d>\n\n", BM, BN, NUM_SMEM);
    printf("| slot | pv_end_abs | next_qk_start_abs | pv_end_rel | next_qk_start_rel | gap_cycles |\n");
    printf("| ---: | ---: | ---: | ---: | ---: | ---: |\n");

    for (int i = 0; i + 1 < slots; ++i) {
        const unsigned long long pv_end = h_profile.pv_ends[i];
        const unsigned long long next_qk_start = h_profile.qk_starts[i + 1];

        printf("| %d | %llu | %llu | %llu | %llu | %llu |\n",
               i,
               pv_end % 1000000,
               next_qk_start % 1000000,
               pv_end - base,
               next_qk_start - base,
               next_qk_start - pv_end);
    }
}

template<int B, int H, int S, int D=128, int BM=128, int BN=128, int NUM_THREADS=384, int NUM_SMEM=1>
void benchmark_attn_ws_kernel(
    fp16 *dQ,
    fp16 *dK,
    fp16 *dV,
    fp16 *dO,
    Profile *d_profile
) {
    Profile zero_profile{};

    constexpr int kWarmupIters = 20;
    constexpr int kMeasureIters = 200;

    cudaEvent_t start;
    cudaEvent_t stop;
    cudaCheck(cudaEventCreate(&start));
    cudaCheck(cudaEventCreate(&stop));

    for (int i = 0; i < kWarmupIters; ++i) {
        cudaCheck(cudaMemcpy(
            d_profile,
            &zero_profile,
            sizeof(Profile),
            cudaMemcpyHostToDevice));
        cudaCheck(cudaMemset(dO, 0, static_cast<size_t>(B) * H * S * D * sizeof(fp16)));

        runAttnWSKernel<B, H, S, D, BM, BN, NUM_THREADS, NUM_SMEM>(
            dQ, dK, dV, dO, d_profile);

        cudaCheck(cudaGetLastError());
    }
    cudaCheck(cudaDeviceSynchronize());

    float total_ms = 0.0f;
    for (int i = 0; i < kMeasureIters; ++i) {
        cudaCheck(cudaMemcpy(
            d_profile,
            &zero_profile,
            sizeof(Profile),
            cudaMemcpyHostToDevice));
        cudaCheck(cudaMemset(dO, 0, static_cast<size_t>(B) * H * S * D * sizeof(fp16)));

        cudaCheck(cudaEventRecord(start));
        runAttnWSKernel<B, H, S, D, BM, BN, NUM_THREADS, NUM_SMEM>(
            dQ, dK, dV, dO, d_profile);
        cudaCheck(cudaEventRecord(stop));

        cudaCheck(cudaGetLastError());
        cudaCheck(cudaEventSynchronize(stop));

        float elapsed_ms = 0.0f;
        cudaCheck(cudaEventElapsedTime(&elapsed_ms, start, stop));
        total_ms += elapsed_ms;
    }

    cudaCheck(cudaEventDestroy(start));
    cudaCheck(cudaEventDestroy(stop));

    printf("attnWSKernel<BM=%d, BN=%d, smem=%d> avg kernel time over %d runs: %.3f ms\n",
           BM,
           BN,
           NUM_SMEM,
           kMeasureIters,
           total_ms / static_cast<float>(kMeasureIters));
}

// nvcc -std=c++17 -arch=sm_90a -O3 attn.cu -o attn_test -lcuda
int main() {
    constexpr int B = 1;
    constexpr int H = 16;
    constexpr int S = 4096;
    constexpr int D = 128;

    const size_t numel = static_cast<size_t>(B) * H * S * D;
    const size_t bytes = numel * sizeof(fp16);

    std::vector<fp16> hQ(numel), hK(numel), hV(numel);

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

    ProfileResult *d_profile = nullptr;
    cudaCheck(cudaMalloc(&d_profile, sizeof(ProfileResult)));
    ProfileSM *d_sm_profile = nullptr;
    cudaCheck(cudaMalloc(&d_sm_profile, sizeof(ProfileSM)));
    Profile *d_ws_profile = nullptr;
    cudaCheck(cudaMalloc(&d_ws_profile, sizeof(Profile)));

    // printf("| BM | BN | LoadQ(avg) | LoadK(avg) | LoadV(avg) | GEMM-QK(avg) | SOFTMAX(avg) | GEMM-PV(avg) |\n");
    // printf("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n");

    // run_one_case<B, H, S, D, 128,  32, 256, 2>(dQ, dK, dV, dO, d_profile);
    // run_one_case<B, H, S, D, 128,  64, 256, 2>(dQ, dK, dV, dO, d_profile);
    // run_one_case<B, H, S, D, 128, 128, 256, 2>(dQ, dK, dV, dO, d_profile);

    // run_one_case<B, H, S, D, 128,  32>(dQ, dK, dV, dO, d_profile);
    // run_one_case<B, H, S, D, 128,  64>(dQ, dK, dV, dO, d_profile);
    // run_one_case<B, H, S, D, 128, 128>(dQ, dK, dV, dO, d_profile);
    // run_one_case<B, H, S, D, 128, 256>(dQ, dK, dV, dO, d_profile);

    // run_one_case<B, H, S, D, 256,  32>(dQ, dK, dV, dO, d_profile);
    // run_one_case<B, H, S, D, 256,  64>(dQ, dK, dV, dO, d_profile);
    // run_one_case<B, H, S, D, 256, 128>(dQ, dK, dV, dO, d_profile);

    // run_one_case_ws<B, H, S, D, 128,  128, 384, 1>(dQ, dK, dV, dO, d_ws_profile);
    // run_one_case_ws<B, H, S, D, 128,  128, 384, 2>(dQ, dK, dV, dO, d_ws_profile);
    // run_one_case_ws<B, H, S, D, 128,  128, 384, 3>(dQ, dK, dV, dO, d_ws_profile);

    run_one_case_nosm<B, H, S, D, 128,  128>(dQ, dK, dV, dO, d_sm_profile);

    // benchmark_attn_ws_kernel<B, H, S, D, 128, 128, 384, 1>(dQ, dK, dV, dO, d_ws_profile);
    // benchmark_attn_ws_kernel<B, H, S, D, 128, 128, 384, 2>(dQ, dK, dV, dO, d_ws_profile);

    cudaCheck(cudaGetLastError());
    cudaCheck(cudaDeviceSynchronize());

    cudaCheck(cudaFree(d_ws_profile));
    cudaCheck(cudaFree(d_sm_profile));
    cudaCheck(cudaFree(d_profile));
    cudaCheck(cudaFree(dQ));
    cudaCheck(cudaFree(dK));
    cudaCheck(cudaFree(dV));
    cudaCheck(cudaFree(dO));

    return 0;
}
