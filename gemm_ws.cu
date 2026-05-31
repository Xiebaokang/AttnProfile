#include "tools.cuh"


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

__device__ __forceinline__ int square_func(int n) {
  int a = static_cast<int>(sqrtf(static_cast<float>(n)));
  while (a >= 1) {
    if (n % a == 0) {
      return a;
    }
    --a;
  }
  return 1;
}

template <int BM, int BN, int DIM, int NUM_STAGE=1>
struct SMemWS {
    alignas(128) fp16 Q[BM*DIM];
    alignas(128) fp16 K[BN*DIM*NUM_STAGE];
    alignas(128) fp16 P[BM*BN];
    alignas(8) uint64_t Qmbar;
    alignas(8) uint64_t Kempty[NUM_STAGE];
    alignas(8) uint64_t Kfull[NUM_STAGE];
};

template<int BM, int BN, int DIM, int NUM_THREADS, int NUM_STAGE>
__global__  __launch_bounds__(NUM_THREADS) 
void attnWSKernel(
    int B, int H, int S, 
    const __grid_constant__ CUtensorMap tensorMapQ, 
    const __grid_constant__ CUtensorMap tensorMapK, 
    fp16 *P
) {
  // WS attention
    const int bs = blockIdx.z;
    const int hn = blockIdx.y;
    const int by = blockIdx.x;
    const int tid = threadIdx.x;

    // mma size
    assert((BN >= 256 || BN == 16 || BN == 32 || BN == 64 || BN == 128) && "DIM ERROR!");
    constexpr int MMA_M = 64;
    constexpr int MMA_N = BN <= 256 ? BN : 256;
    constexpr int MMA_K = 16;
    
    // setting shared memory
    extern __shared__ __align__(128) uint8_t smem[];
    SMemWS<BM, BN, DIM, NUM_STAGE> &s = *reinterpret_cast<SMemWS<BM, BN, DIM, NUM_STAGE>*>(smem);
    fp16 *sQ = s.Q, *sK = s.K, *sP = s.P;
    Barrier *Qmbar = &s.Qmbar;
    Barrier *Kempty = s.Kempty, *Kfull = s.Kfull;

    // init mbarrier
    if (threadIdx.x == 0) {
        init_barrier(Qmbar, 1);
        for (int i = 0; i < NUM_STAGE; ++i) {
            init_barrier(&Kfull[i], 1);  // 1 thread arrive
            init_barrier(&Kempty[i], 256);  // 256 thread arrive
        }
    }
    __syncthreads();
    fence_view_async_shared();

    // TMA load
    if (tid >= 256) {  // producer
        warpgroup_reg_dealloc<24>();
        if (tid == 256) {
            int stage = 0, phase = 0;
            // load Q
            expect_bytes(Qmbar, BM * DIM * sizeof(fp16));
            load_async(sQ, &tensorMapQ, Qmbar, bs, hn, by * BM, 0);
            for (size_t iw=0; iw<S; iw+=BN, ++stage) {
                if (stage >= NUM_STAGE) { stage = 0; phase ^= 1; }
                fp16 *KAddr = sK + stage * (BN * DIM);
                // load K
                wait(&Kempty[stage], phase);
                expect_bytes(&Kfull[stage], BN * DIM * sizeof(fp16));
                load_async(KAddr, &tensorMapK, &Kfull[stage], bs, hn, iw, 0);
            }
        }
    } else {  // consumer
        warpgroup_reg_alloc<240>();
        wait(Qmbar, 0);
        int stage = 0, phase = 0;
        #pragma unroll
        for (int st = 0; st < NUM_STAGE; ++st) {
            arrive(&Kempty[st]);
        }
        fp32 acc_s[BM/(MMA_M*2)][BN/MMA_N][MMA_N/16][8];  // define acc_s

        uint32_t wg_idx = tid >> 7;
        uint32_t lane_id = tid & 31;
        uint32_t warp_id_in_wg = (tid >> 5) & 0x3;  // local warp id inside each 128-thread warpgroup

        // main for loop
        for (size_t iw=0; iw<S; iw+=BN, ++stage) {
            if (stage >= NUM_STAGE) { stage = 0; phase ^= 1; }
            fp16 *KAddr = sK + stage * BN * DIM;
            // fill acc_s
            #pragma unroll
            for (size_t i=0; i<BM/(MMA_M*2); i++) {
                #pragma unroll
                for (size_t j=0; j<BN/MMA_N; j++) {
                    #pragma unroll
                    for (size_t k=0; k<MMA_N/16; k++) {
                        #pragma unroll
                        for (size_t l=0; l<8; l++) {
                            acc_s[i][j][k][l] = 0.0f;  // no mask
                        }
                    }
                }
            }

            // gemm-qk
            wait(&Kfull[stage], phase);
            warpgroup_arrive();
            #pragma unroll
            for (size_t i=0; i<BM/(MMA_M*2); i++) {   // block关于wg的布局：[2, 1]
                #pragma unroll
                for (size_t j=0; j<BN/MMA_N; j++) {
                    #pragma unroll
                    for (size_t k=0; k<DIM; k+=MMA_K) {
                        int q_row = wg_idx * BM/(MMA_M*2) * MMA_M + i * MMA_M;
                        int q_col = k;
                        int k_row = j * MMA_N;
                        int k_col = k;
                        fp16 *_QAddr = sQ + tma_smem_offset_2d<BM>(q_row, q_col);
                        fp16 *_KAddr = KAddr + tma_smem_offset_2d<BN>(k_row, k_col);
                        wgmma_ss<MMA_N, 1, 1, 1, 0, 0, 16, 1024, true>(acc_s[i][j], _QAddr, _KAddr);
                    }
                }
            }
            warpgroup_commit_batch();
            warpgroup_wait();
            arrive(&Kempty[stage]);  // 释放 tma K 前阻塞
            // store
            fp16 d_fp16[8];
            uint32_t* data_ptr = (uint32_t*)d_fp16;
            const int lane_row = lane_id & 0xf;         // 0..15
            const int lane_col = (lane_id >> 4) * 8;    // 0 or 8
            #pragma unroll
            for (size_t i=0; i<BM/(MMA_M*2); i++) {  // block repeat y
                #pragma unroll
                for (size_t j=0; j<BN/MMA_N; j++) {  // block repeat x
                    #pragma unroll
                    for (size_t k=0; k<MMA_N/16; k++) {  // warp widths 有多少 8x8x4 的块
                        int p_row = wg_idx * BM/(MMA_M*2) * MMA_M
                                  + i * MMA_M
                                  + warp_id_in_wg * 16
                                  + lane_row;
                        int p_col = j * MMA_N
                                  + k * 16
                                  + lane_col;
                        fp16 *_sP = sP + tma_smem_offset_2d<BM>(p_row, p_col);
                        #pragma unroll
                        for (size_t l=0; l<8; l++) {
                            d_fp16[l] = (fp16)acc_s[i][j][k][l];
                        }
                        stmatrix_x4<fp16>(_sP, data_ptr);
                    }
                }
            }
            bar_sync(256, 2);
            fp16 *_P = P + bs * (H * S * S) + hn * (S * S) + by * BM * S + iw;
            int x_size = square_func(BM * BN / 256);
            int y_size = BM * BN / 256 / x_size;
            int x = tid % (BN / x_size);
            int y = tid / (BN / x_size);
            for (size_t i=0; i<y_size; i++) {
              for (size_t j=0; j<x_size; j++) {
                int row = y * y_size + i;
                int col = x * x_size + j;
                _P[row * S + col] = sP[tma_smem_offset_2d<BM>(row, col)];
              }
            }
            bar_sync(256, 3);
        }
    }
}

template<int BM, int BN, int DIM, int NUM_THREADS, int NUM_STAGE>
__global__  __launch_bounds__(NUM_THREADS) 
void attnWSKernel_(
    int B, int H, int S, 
    const __grid_constant__ CUtensorMap tensorMapQ, 
    const __grid_constant__ CUtensorMap tensorMapK, 
    const __grid_constant__ CUtensorMap tensorMapP
) {
  // WS attention
    const int bs = blockIdx.z;
    const int hn = blockIdx.y;
    const int by = blockIdx.x;
    const int tid = threadIdx.x;

    // mma size
    assert((BN >= 256 || BN == 16 || BN == 32 || BN == 64 || BN == 128) && "DIM ERROR!");
    constexpr int MMA_M = 64;
    constexpr int MMA_N = BN <= 256 ? BN : 256;
    constexpr int MMA_K = 16;
    
    // setting shared memory
    extern __shared__ __align__(128) uint8_t smem[];
    SMemWS<BM, BN, DIM, NUM_STAGE> &s = *reinterpret_cast<SMemWS<BM, BN, DIM, NUM_STAGE>*>(smem);
    fp16 *sQ = s.Q, *sK = s.K, *sP = s.P;
    Barrier *Qmbar = &s.Qmbar;
    Barrier *Kempty = s.Kempty, *Kfull = s.Kfull;

    // init mbarrier
    if (threadIdx.x == 0) {
        init_barrier(Qmbar, 1);
        for (int i = 0; i < NUM_STAGE; ++i) {
            init_barrier(&Kfull[i], 1);  // 1 thread arrive
            init_barrier(&Kempty[i], 256);  // 256 thread arrive
        }
    }
    __syncthreads();
    fence_view_async_shared();

    // TMA load
    if (tid >= 256) {  // producer
        warpgroup_reg_dealloc<24>();
        if (tid == 256) {
            int stage = 0, phase = 0;
            // load Q
            expect_bytes(Qmbar, BM * DIM * sizeof(fp16));
            load_async(sQ, &tensorMapQ, Qmbar, bs, hn, by * BM, 0);
            for (size_t iw=0; iw<S; iw+=BN, ++stage) {
                if (stage >= NUM_STAGE) { stage = 0; phase ^= 1; }
                fp16 *KAddr = sK + stage * (BN * DIM);
                // load K
                wait(&Kempty[stage], phase);
                expect_bytes(&Kfull[stage], BN * DIM * sizeof(fp16));
                load_async(KAddr, &tensorMapK, &Kfull[stage], bs, hn, iw, 0);
            }
        }
    } else {  // consumer
        warpgroup_reg_alloc<240>();
        wait(Qmbar, 0);
        int stage = 0, phase = 0;
        #pragma unroll
        for (int st = 0; st < NUM_STAGE; ++st) {
            arrive(&Kempty[st]);
        }
        fp32 acc_s[BM/(MMA_M*2)][BN/MMA_N][MMA_N/16][8];  // define acc_s

        uint32_t wg_idx = tid >> 7;
        uint32_t lane_id = tid & 31;
        uint32_t warp_id_in_wg = (tid >> 5) & 0x3;  // local warp id inside each 128-thread warpgroup

        // main for loop
        for (size_t iw=0; iw<S; iw+=BN, ++stage) {
            if (stage >= NUM_STAGE) { stage = 0; phase ^= 1; }
            fp16 *KAddr = sK + stage * BN * DIM;
            // fill acc_s
            #pragma unroll
            for (size_t i=0; i<BM/(MMA_M*2); i++) {
                #pragma unroll
                for (size_t j=0; j<BN/MMA_N; j++) {
                    #pragma unroll
                    for (size_t k=0; k<MMA_N/16; k++) {
                        #pragma unroll
                        for (size_t l=0; l<8; l++) {
                            acc_s[i][j][k][l] = 0.0f;  // no mask
                        }
                    }
                }
            }

            // gemm-qk
            wait(&Kfull[stage], phase);
            warpgroup_arrive();
            #pragma unroll
            for (size_t i=0; i<BM/(MMA_M*2); i++) {   // block关于wg的布局：[2, 1]
                #pragma unroll
                for (size_t j=0; j<BN/MMA_N; j++) {
                    #pragma unroll
                    for (size_t k=0; k<DIM; k+=MMA_K) {
                        int q_row = wg_idx * BM/(MMA_M*2) * MMA_M + i * MMA_M;
                        int q_col = k;
                        int k_row = j * MMA_N;
                        int k_col = k;
                        fp16 *_QAddr = sQ + tma_smem_offset_2d<BM>(q_row, q_col);
                        fp16 *_KAddr = KAddr + tma_smem_offset_2d<BN>(k_row, k_col);
                        wgmma_ss<MMA_N, 1, 1, 1, 0, 0, 16, 1024, true>(acc_s[i][j], _QAddr, _KAddr);
                    }
                }
            }
            warpgroup_commit_batch();
            warpgroup_wait();
            arrive(&Kempty[stage]);  // 释放 tma K 前阻塞
            // Wait previous TMA store completion before reusing sP.
            // Only thread 0 issues store_async, so only thread 0 has a valid wait group.
            if (tid == 0) {
                tma_store_wait();
            }
            bar_sync(256, 1);
            // store
            fp16 d_fp16[8];
            uint32_t* data_ptr = (uint32_t*)d_fp16;
            const int lane_row = lane_id & 0xf;         // 0..15
            const int lane_col = (lane_id >> 4) * 8;    // 0 or 8
            #pragma unroll
            for (size_t i=0; i<BM/(MMA_M*2); i++) {  // block repeat y
                #pragma unroll
                for (size_t j=0; j<BN/MMA_N; j++) {  // block repeat x
                    #pragma unroll
                    for (size_t k=0; k<MMA_N/16; k++) {  // warp widths 有多少 8x8x4 的块
                        int p_row = wg_idx * BM/(MMA_M*2) * MMA_M
                                  + i * MMA_M
                                  + warp_id_in_wg * 16
                                  + lane_row;
                        int p_col = j * MMA_N
                                  + k * 16
                                  + lane_col;
                        fp16 *_sP = sP + tma_smem_offset_2d<BM>(p_row, p_col);
                        #pragma unroll
                        for (size_t l=0; l<8; l++) {
                            d_fp16[l] = (fp16)acc_s[i][j][k][l];
                        }
                        stmatrix_x4<fp16>(_sP, data_ptr);
                    }
                }
            }
            // stmatrix writes use generic proxy; TMA store reads shared via async proxy.
            // Make shared writes visible to async proxy before issuing store_async.
            fence_view_async_shared();
            bar_sync(256, 2);
            if (tid == 0) {
                store_async(&tensorMapP, sP, bs, hn, by * BM, iw);
                tma_store_arrive();
            }
        }
    }
}

template<int B, int H, int S, int D=128, int BM=128, int BN=128, int NUM_THREADS=384, int NUM_STAGE=1>
void runAttnWSKernel(fp16 *Q, fp16 *K, fp16 *P) {
    CUtensorMap d_tma_map_Q = create_tensor_map<BM, D>(Q, B, H, S, D);
    CUtensorMap d_tma_map_K = create_tensor_map<BN, D>(K, B, H, S, D);

    auto* kernel = attnWSKernel<BM, BN, D, NUM_THREADS, NUM_STAGE>;
    constexpr size_t sMemSize = sizeof(SMemWS<BM, BN, D, NUM_STAGE>);
    static_assert(sMemSize < 256 * 1024);
    cudaCheck(cudaFuncSetAttribute(
        kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, sMemSize));

    dim3 grid = {static_cast<unsigned int>(S/BM), static_cast<unsigned int>(H), static_cast<unsigned int>(B)};
    kernel<<<grid, NUM_THREADS, sMemSize>>>(B, H, S, d_tma_map_Q, d_tma_map_K, P);
}

template<int B, int H, int S, int D=128, int BM=128, int BN=128, int NUM_THREADS=384, int NUM_STAGE=1>
void runAttnWSKernel_(fp16 *Q, fp16 *K, fp16 *P) {
    CUtensorMap d_tma_map_Q = create_tensor_map<BM, D>(Q, B, H, S, D);
    CUtensorMap d_tma_map_K = create_tensor_map<BN, D>(K, B, H, S, D);
    CUtensorMap d_tma_map_P = create_tensor_map<BM, BN, false>(P, B, H, S, S);

    auto* kernel = attnWSKernel_<BM, BN, D, NUM_THREADS, NUM_STAGE>;
    constexpr size_t sMemSize = sizeof(SMemWS<BM, BN, D, NUM_STAGE>);
    static_assert(sMemSize < 256 * 1024);
    cudaCheck(cudaFuncSetAttribute(
        kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, sMemSize));

    dim3 grid = {static_cast<unsigned int>(S/BM), static_cast<unsigned int>(H), static_cast<unsigned int>(B)};
    kernel<<<grid, NUM_THREADS, sMemSize>>>(B, H, S, d_tma_map_Q, d_tma_map_K, d_tma_map_P);
}


int main() {
    constexpr int B = 1;
    constexpr int H = 1;
    constexpr int S = 256;
    constexpr int D = 128;
    constexpr int BM = 128;
    constexpr int BN = 128;
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

    static_assert(S % BM == 0, "S must be divisible by BM.");
    static_assert(S % BN == 0, "S must be divisible by BN.");

    const size_t qk_numel = static_cast<size_t>(B) * H * S * D;
    const size_t p_numel = static_cast<size_t>(B) * H * S * S;
    const size_t qk_bytes = qk_numel * sizeof(fp16);
    const size_t p_bytes = p_numel * sizeof(fp16);

    std::vector<fp16> hQ(qk_numel), hK(qk_numel), hP(p_numel);
    std::vector<float> hRef(p_numel, 0.0f);

    auto idx_qk = [=](int b, int h, int s, int d) {
        return (((static_cast<size_t>(b) * H + h) * S + s) * D + d);
    };
    auto idx_p = [=](int b, int h, int m, int n) {
        return (((static_cast<size_t>(b) * H + h) * S + m) * S + n);
    };

    for (int b = 0; b < B; ++b) {
        for (int h = 0; h < H; ++h) {
            for (int s = 0; s < S; ++s) {
                for (int d = 0; d < D; ++d) {
                    const size_t idx = idx_qk(b, h, s, d);
                    const int q_i = static_cast<int>((idx * 13 + 7) % 31) - 15;
                    const int k_i = static_cast<int>((idx * 17 + 5) % 29) - 14;
                    const float q = static_cast<float>(q_i) * (1.0f / 128.0f);
                    const float k = static_cast<float>(k_i) * (1.0f / 128.0f);
                    hQ[idx] = __float2half_rn(q);
                    hK[idx] = __float2half_rn(k);
                }
            }
        }
    }

    fp16 *dQ = nullptr, *dK = nullptr, *dP = nullptr;
    cudaCheck(cudaMalloc(&dQ, qk_bytes));
    cudaCheck(cudaMalloc(&dK, qk_bytes));
    cudaCheck(cudaMalloc(&dP, p_bytes));

    cudaCheck(cudaMemcpy(dQ, hQ.data(), qk_bytes, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(dK, hK.data(), qk_bytes, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemset(dP, 0.0, p_bytes));

    // runAttnWSKernel<B, H, S, D, BM, BN>(dQ, dK, dP);
    runAttnWSKernel_<B, H, S, D, BM, BN>(dQ, dK, dP);
    cudaCheck(cudaGetLastError());
    cudaCheck(cudaDeviceSynchronize());
    cudaCheck(cudaMemcpy(hP.data(), dP, p_bytes, cudaMemcpyDeviceToHost));
    printf("Kernel Run Success!\n");

    // CPU reference: P = Q * K^T
    // Match kernel path: fp32 accumulation, then cast to fp16.
    for (int b = 0; b < B; ++b) {
        for (int h = 0; h < H; ++h) {
            for (int m = 0; m < S; ++m) {
                for (int n = 0; n < S; ++n) {
                    float score = 0.0f;
                    for (int d = 0; d < D; ++d) {
                        const float q = __half2float(hQ[idx_qk(b, h, m, d)]);
                        const float k = __half2float(hK[idx_qk(b, h, n, d)]);
                        score += q * k;
                    }
                    hRef[idx_p(b, h, m, n)] = __half2float(__float2half_rn(score));
                }
            }
        }
    }

    for (int i=0; i<S; i++) {
        for (int j=0; j<S; j++) {
            printf("%.5f ", __half2float(hP[i * S + j]));
        }
        printf("\n");
    }

    printf("\n\n");

    for (int i=0; i<S; i++) {
        for (int j=0; j<S; j++) {
            printf("%.5f ", hRef[i * S + j]);
        }
        printf("\n");
    }
    printf("\n\n");

    int mismatch = 0;
    float max_abs_err = 0.0f;
    size_t max_abs_idx = 0;
    for (size_t i = 0; i < p_numel; ++i) {
        const float got = __half2float(hP[i]);
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
        printf("PASS: P matches reference. max_abs_err=%f at idx=%zu\n", max_abs_err, max_abs_idx);
    } else {
        printf("FAIL: mismatch=%d / %zu, max_abs_err=%f at idx=%zu\n",
               mismatch, p_numel, max_abs_err, max_abs_idx);
    }

    cudaCheck(cudaFree(dQ));
    cudaCheck(cudaFree(dK));
    cudaCheck(cudaFree(dP));

    return mismatch == 0 ? 0 : 1;
}
