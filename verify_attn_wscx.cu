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
                        acc_o[i][j][k][0] = exp2f(acc_s[i][j][k][0] * scale - scores_max[i][0] * scale);
                        acc_o[i][j][k][1] = exp2f(acc_s[i][j][k][1] * scale - scores_max[i][0] * scale);
                        acc_o[i][j][k][4] = exp2f(acc_s[i][j][k][4] * scale - scores_max[i][0] * scale);
                        acc_o[i][j][k][5] = exp2f(acc_s[i][j][k][5] * scale - scores_max[i][0] * scale);
                        acc_o[i][j][k][2] = exp2f(acc_s[i][j][k][2] * scale - scores_max[i][1] * scale);
                        acc_o[i][j][k][3] = exp2f(acc_s[i][j][k][3] * scale - scores_max[i][1] * scale);
                        acc_o[i][j][k][6] = exp2f(acc_s[i][j][k][6] * scale - scores_max[i][1] * scale);
                        acc_o[i][j][k][7] = exp2f(acc_s[i][j][k][7] * scale - scores_max[i][1] * scale);
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
                        scores_sum[i][0] += (acc_o[i][j][k][0] + acc_o[i][j][k][1] + acc_o[i][j][k][4] + acc_o[i][j][k][5]);
                        scores_sum[i][1] += (acc_o[i][j][k][2] + acc_o[i][j][k][3] + acc_o[i][j][k][6] + acc_o[i][j][k][7]);
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
void attnWSKCX2ernel(
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
                arrive(&Kempty[smem_i]);

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
                        acc_o[n][j][k][0] = exp2f(acc_s[j][k][0] * scale - scores_max[n][0] * scale);
                        acc_o[n][j][k][1] = exp2f(acc_s[j][k][1] * scale - scores_max[n][0] * scale);
                        acc_o[n][j][k][4] = exp2f(acc_s[j][k][4] * scale - scores_max[n][0] * scale);
                        acc_o[n][j][k][5] = exp2f(acc_s[j][k][5] * scale - scores_max[n][0] * scale);
                        acc_o[n][j][k][2] = exp2f(acc_s[j][k][2] * scale - scores_max[n][1] * scale);
                        acc_o[n][j][k][3] = exp2f(acc_s[j][k][3] * scale - scores_max[n][1] * scale);
                        acc_o[n][j][k][6] = exp2f(acc_s[j][k][6] * scale - scores_max[n][1] * scale);
                        acc_o[n][j][k][7] = exp2f(acc_s[j][k][7] * scale - scores_max[n][1] * scale);
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
                        scores_sum[0] += (acc_o[n][j][k][0] + acc_o[n][j][k][1] + acc_o[n][j][k][4] + acc_o[n][j][k][5]);
                        scores_sum[1] += (acc_o[n][j][k][2] + acc_o[n][j][k][3] + acc_o[n][j][k][6] + acc_o[n][j][k][7]);
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
                            float2 _t1 = *(float2*)(&acc_s[i][j][k][l]);
                            *(half2*)(&(_t2)) = __float22half2_rn(*(float2*)(&(_t1)));
                            *(uint1*)(&acc_s_cast[i][j][k][l]) = _t2;
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
                        const int p_tile_inner = (k % QK_MMA_N) / MMA_K;
                        fp16 *_VAddr = VAddr + tma_smem_offset_2d<BN>(v_row, v_col);
                        wgmma_rs<PV_MMA_N, 1, 1, 1, 0, 1, 16, 1024, true>(
                            acc_o[n][j],
                            reinterpret_cast<uint32_t*>(acc_s_cast[p_tile_outer][p_tile_inner]),
                            _VAddr);
                    }
                }
                warpgroup_commit_batch();
                warpgroup_wait();
                arrive(&Vempty[smem_i]);
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
