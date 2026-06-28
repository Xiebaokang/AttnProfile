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
    constexpr int PV_MMA_N = DIM <= 128 ? DIM : 128;
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

            // // gemm-pv
            wait(&Vfull[smem_i], phase);
            warpgroup_arrive();
            #pragma unroll
            for (size_t i=0; i<BM/(MMA_M*2); i++) {
                #pragma unroll
                for (size_t j=0; j<DIM/PV_MMA_N; j++) {
                    #pragma unroll
                    for (size_t k=0; k<BN; k+=MMA_K) {
                        // V is stored in shared as [K=BN, N=DIM].
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


// export CUDA_VISIBLE_DEVICES=1
// nvcc -std=c++17 -arch=sm_90a -O3 debug_attn.cu -o debug_attn_test -lcuda -Xptxas=-v
// ncu --set full --launch-skip 100 --launch-count 1 ./debug_attn_test
int main() {
    constexpr int B = 1;
    constexpr int H = 1;
    constexpr int S = 256;
    constexpr int D = 128;
    constexpr int BM = 128;
    constexpr int BN = 128;

    auto *kernel = runAttnWSKernel<B, H, S, D, BM, BN, 384, 1>;
    verify_attn<B, H, S, D, BN>(kernel);
    return 0;
}
