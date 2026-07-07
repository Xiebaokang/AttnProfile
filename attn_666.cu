#include "utils.cuh"
#include <type_traits>

template <int BM, int BN, int DIM, int NUM_SMEM>
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
    //alignas(8) uint8_t indexBit[8][2];
};

template<int BM, int BN, int DIM, int NUM_THREADS, int NUM_SMEM>
__global__  __launch_bounds__(NUM_THREADS, 1)
void attnWSKCXForNSegmentedKernel(
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
    // constexpr int N = 1;
    static_assert(NUM_SMEM >= 1);

    // setting shared memory
    extern __shared__ __align__(128) uint8_t smem[];
    SMemWS<BM, BN, DIM, NUM_SMEM> &s = *reinterpret_cast<SMemWS<BM, BN, DIM, NUM_SMEM>*>(smem);
    fp16 *sQ = s.Q, *sK = s.K, *sV = s.V, *sO = s.O;
    Barrier *Qmbar = &s.Qmbar;
    Barrier *Kempty = s.Kempty, *Vempty = s.Vempty, *Kfull = s.Kfull, *Vfull = s.Vfull;
    // uint8_t *indexBit = s.indexBit;

    // init mbarrier
    if (threadIdx.x == 0) {
        // // init 
        // indexBit[0][0] = 0; indexBit[0][1] = 0;
        // indexBit[1][0] = 0; indexBit[1][1] = 0;
        // indexBit[2][0] = 1; indexBit[2][1] = 0;
        // indexBit[3][0] = 1; indexBit[3][1] = 0;
        // indexBit[4][0] = 0; indexBit[4][1] = 1;
        // indexBit[5][0] = 0; indexBit[5][1] = 1;
        // indexBit[6][0] = 1; indexBit[6][1] = 1;
        // indexBit[7][0] = 1; indexBit[7][1] = 1;

        init_barrier(Qmbar, 1);
        #pragma unroll
        for (size_t i=0; i<NUM_SMEM; i++) {
            init_barrier(&Kempty[i], 512);
            init_barrier(&Vempty[i], 512);
            init_barrier(&Kfull[i], 1);
            init_barrier(&Vfull[i], 1);
        }
    }
    __syncthreads();
    fence_view_async_shared();

    // TMA load
    if (wg_idx == 2) {  // producer
        warpgroup_reg_dealloc<24>();
        if (tid == 256) {
            size_t smem_i = 0;
            int phase = 0;
            // load Q
            expect_bytes(Qmbar, BM * DIM * sizeof(fp16));
            load_async(sQ, &tensorMapQ, Qmbar, bs, hn, by * BM, 0);
            for (size_t iw=0; iw<S; iw+=2*BN, smem_i++) {
                if (smem_i == NUM_SMEM) { smem_i = 0; phase ^= 1; }
                fp16 *addrK = sK + smem_i * BN * DIM;
                fp16 *addrV = sV + smem_i * BN * DIM;

                // load K
                wait(&Kempty[smem_i], phase);
                expect_bytes(&Kfull[smem_i], BN * DIM * sizeof(fp16));
                load_async(addrK, &tensorMapK, &Kfull[smem_i], bs, hn, iw, 0);

                // load V
                wait(&Vempty[smem_i], phase);
                expect_bytes(&Vfull[smem_i], BN * DIM * sizeof(fp16));
                load_async(addrV, &tensorMapV, &Vfull[smem_i], bs, hn, iw, 0);
            }

            smem_i ++;
            if (smem_i == NUM_SMEM) { smem_i = 0; phase ^= 1; }
            wait(&Kempty[smem_i], phase);
            wait(&Vempty[smem_i], phase);

            smem_i ++;
            if (smem_i == NUM_SMEM) { smem_i = 0; phase ^= 1; }
            wait(&Kempty[smem_i], phase);
            wait(&Vempty[smem_i], phase);
        }
    } else {  // consumer
        warpgroup_reg_alloc<240>();
        #pragma unroll
        for (size_t i=0; i<NUM_SMEM; i++) {
            arrive(&Kempty[i]);
            arrive(&Kempty[i]);
            arrive(&Vempty[i]);
            arrive(&Vempty[i]);
        }
        //bar_sync(256, 1);
        //arrive(&Kempty[0]);
        //arrive(&Kempty[1]);


        // args
        uint32_t lane_id = tid & 31;
        uint32_t warp_id_in_wg = (tid >> 5) & 0x3;
        const unsigned mask = __activemask();
        const fp32 scale = sqrt((1.0f / DIM)) * 1.44269504f;  // log2(e)

        // registers define
        fp32 acc_o[N][N*DIM/PV_MMA_N][PV_MMA_N/16][8];
        fp32 scores_max_prev[N][2];
        fp32 scores_max[N][2];
        fp32 logsum[N][2];

        // init acc_o
        #pragma unroll
        for (size_t i=0; i<N; i++) {
            #pragma unroll
            for (size_t j=0; j<N*DIM/PV_MMA_N; j++) {
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
            
            //if (tid == 64 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0)
            //    { printf("full1\n"); }

            if (n == 0) { wait(&Kfull[smem_i], phase); }
            
            //if (tid == 64 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0)
            //    { printf("full2\n"); }

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
            //if (n == N-1) { arrive(&Kempty[smem_i]); }
        };

        auto gemm_pv = [&](size_t acc_o_i, bool wait_vfull, bool arrive_vempty, int smem_i, int phase, int kk) {
            fp16 *VAddr = sV + smem_i * BN * DIM;

            // if (tid == 64 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0)
            //     { printf("full1 smem_i=%d, phase = %d\n",smem_i,phase); }

            if (wait_vfull) { wait(&Vfull[smem_i], phase); }

            // if (tid == 64 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0)
            //     { printf("full2\n"); }

            warpgroup_arrive();
            
            int start = kk * DIM/PV_MMA_N;
            int end = start + DIM/PV_MMA_N;
            #pragma unroll
            for (size_t j = start; j< end; j++) {
                #pragma unroll
                for (size_t k=0; k<BN; k+=MMA_K) {
                    int v_row = k;
                    int v_col = j * PV_MMA_N;
                    const int p_tile_outer = k / QK_MMA_N;
                    const int p_tile_inner = (k % QK_MMA_N) / 16;
                    fp16 *_VAddr = VAddr + tma_smem_offset_2d<BN>(v_row, v_col);
                    wgmma_rs<PV_MMA_N, 1, 1, 1, 0, 1, BN * 64 * sizeof(fp16), 1024, true>(
                        acc_o[acc_o_i][j],
                        reinterpret_cast<uint32_t*>(acc_s_cast[p_tile_outer][p_tile_inner]),
                        _VAddr);
                }
            }
            warpgroup_commit_batch();
            warpgroup_wait();
            //if (arrive_vempty) { arrive(&Vempty[smem_i]); }
        };

        //auto softmax = [&](auto n_const) {
        auto softmax = [&](int n) {
            //constexpr size_t n = decltype(n_const)::value;
            #pragma unroll
            for (size_t j=0; j<2; j++) {
                scores_max_prev[n][j] = scores_max[n][j];
            }
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
            #pragma unroll
            for (size_t j=0; j<2; j++) {
                scores_max[n][j] = max(scores_max_prev[n][j], scores_max[n][j]);
            }
            fp32 scores_scale[2];
            #pragma unroll
            for (size_t j=0; j<2; j++) {
                scores_scale[j] = exp2f(scores_max_prev[n][j] * scale - scores_max[n][j] * scale);
            }
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
            #pragma unroll
            for (size_t j=0; j<2; j++) {
                logsum[n][j] = logsum[n][j] * scores_scale[j] + scores_sum[j];
            }
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
        arrive(&Kempty[smem_i]);

        //softmax(std::integral_constant<size_t, 0>{});
        softmax(0);

        //if (tid == 64 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0)
        //        { printf("Prologue is OK.\n"); }

        // main loop
        int cnt = 0;
        int cnt1 = -1;
        //smem_i++;
        //for (size_t iw=BN; iw<S; iw+=BN, ++smem_i) {
        for (size_t iw=BN; iw<S; iw+=BN) {
            
            //if ( (iw / BN) % 2 == 0 ) { smem_i += 1; }
            //if (smem_i >= NUM_SMEM) { smem_i = 0; phase ^= 1; }
            if ( (iw / BN) % 2 == 0 ) 
                { cnt = (cnt + 1) % 4;}  
            else
                {cnt1 = (cnt1 + 1) % 4;}

            phase = (cnt >> 1) & 1;
            smem_i = cnt & 1;


            //if (tid == 64 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0)
            //    { printf("iw = %d, smem_i =%d, phase =%d\n",iw,smem_i, phase); }
            
            gemm_qk(std::integral_constant<size_t, 0>{}, smem_i, phase);
            arrive(&Kempty[smem_i]);

            //if (tid == 64 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0)
            //    { printf("QK is OK\n"); }

            //const int prev_smem_i = (smem_i + NUM_SMEM - 1) % NUM_SMEM;
            //int prev_phase = phase;
            //if (prev_smem_i == NUM_SMEM - 1) { prev_phase ^= 1; }

            //int prev_smem_i = smem_i; 
            //int prev_phase = phase;
            // if ( (iw / BN) % 2 == 0 ) {
            //     prev_smem_i = (smem_i + 1) % 2;
            //     if (smem_i == 0) { prev_phase ^= 1; }
            // }

            int prev_smem_i = cnt1 & 1;
            int prev_phase = (cnt1 >> 1) & 1;
            
            int kk = (iw / BN -1) % 2;

            gemm_pv(0, true, true, prev_smem_i, prev_phase, kk);
            arrive(&Vempty[prev_smem_i]);

            //if (tid == 64 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0)
            //    { printf("PV is OK\n"); }
            
            kk = (iw / BN) % 2;
            softmax(kk);
            //softmax(std::integral_constant<size_t, 0>{});
            
        }
        
        // Epilogue
        const int last_smem_i = (smem_i % NUM_SMEM + NUM_SMEM - 1) % NUM_SMEM;
        const int last_phase = phase;
        gemm_pv(0, true, true, last_smem_i, last_phase, 1);
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
        fence_view_async_shared();
        bar_sync(256, 2);
        // tma store
        if (tid == 0) {
            store_async(&tensorMapO, sO, bs, hn, by * BM, 0);
            tma_store_arrive();
        }
    }
}


template<int B, int H, int S, int D=128, int BM=128, int BN=128, int NUM_THREADS=384, int NUM_SMEM=2>
void runAttnWSCXForNKernel(fp16 *Q, fp16 *K, fp16 *V, fp16 *O) {
    CUtensorMap d_tma_map_Q = create_tensor_map<BM, D>(Q, B, H, S, D);
    CUtensorMap d_tma_map_K = create_tensor_map<BN, D>(K, B, H, S, D);
    CUtensorMap d_tma_map_V = create_tensor_map<BN, D>(V, B, H, S, D);
    CUtensorMap d_tma_map_O = create_tensor_map<BM, D>(O, B, H, S, D);

    auto* kernel = attnWSKCXForNSegmentedKernel<BM, BN, D, NUM_THREADS, NUM_SMEM>;
    constexpr size_t sMemSize = sizeof(SMemWS<BM, BN, D, NUM_SMEM>);
    static_assert(sMemSize < 256 * 1024);
    cudaCheck(cudaFuncSetAttribute(
        kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, sMemSize));

    dim3 grid = {static_cast<unsigned int>(S/BM/4), static_cast<unsigned int>(H), static_cast<unsigned int>(B)};
    kernel<<<grid, NUM_THREADS, sMemSize>>>(B, H, S, d_tma_map_Q, d_tma_map_K, d_tma_map_V, d_tma_map_O);
}

// export CUDA_VISIBLE_DEVICES=1
// nvcc -std=c++17 -arch=sm_90a -O3 attn_333.cu -o attn_test -lcuda -Xptxas=-v
// ncu --set full --launch-skip 100 --launch-count 1 ./attn_test
int main() {
    constexpr int B = 1;
    constexpr int H = 16;
    constexpr int S = 114*256*2;
    constexpr int D = 64;
    constexpr int BM = 128;
    constexpr int BN = 128;

    // constexpr int B = 1;
    // constexpr int H = 1;
    // constexpr int S = 1024;
    // constexpr int D = 64;
    // constexpr int BM = 128;
    // constexpr int BN = 128;

    auto *kernel = runAttnWSCXForNKernel<B, H, S, D, BM, BN, 384, 2>;
    // verify_attn<B, H, S, D, BN>(kernel);
     benchmark_attn<B, H, S, D, BM, BN>(kernel);
    // benchmark_attn_ncu<B, H, S, D, BM, BN>(kernel);
    return 0;
}
