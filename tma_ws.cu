#include "tools.cuh"

constexpr int kProfileSlots = 16;

struct Profile {
    unsigned long long load_q_start;
    unsigned long long load_q_end;
    int kv_iters;
    unsigned long long load_k_starts[kProfileSlots];
    unsigned long long load_k_ends[kProfileSlots];
    unsigned long long load_v_starts[kProfileSlots];
    unsigned long long load_v_ends[kProfileSlots];
};

template <int BM, int BN, int DIM, int NUM_STAGE=1>
struct SMemTMA {
    alignas(128) fp16 Q[BM*DIM];
    alignas(128) fp16 K[BN*DIM*NUM_STAGE];
    alignas(128) fp16 V[BN*DIM*NUM_STAGE];
    alignas(8) uint64_t Qmbar;
    alignas(8) uint64_t Kempty[NUM_STAGE];
    alignas(8) uint64_t Vempty[NUM_STAGE];
    alignas(8) uint64_t Kfull[NUM_STAGE];
    alignas(8) uint64_t Vfull[NUM_STAGE];
};

template<int BM, int BN, int DIM, int NUM_THREADS, int NUM_STAGE>
__global__  __launch_bounds__(NUM_THREADS) 
void attnTMAKernel(
    int B, int H, int S, 
    const __grid_constant__ CUtensorMap tensorMapQ, 
    const __grid_constant__ CUtensorMap tensorMapK, 
    const __grid_constant__ CUtensorMap tensorMapV,
    Profile *__restrict__ profile
) {
  // WS attention
    const int bs = blockIdx.z;
    const int hn = blockIdx.y;
    const int by = blockIdx.x;
    const int tid = threadIdx.x;
    const bool do_profile = (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0);
    const int kv_iters = S / BN;

    // mma size
    assert((DIM >= 256 || DIM == 16 || DIM == 32 || DIM == 64 || DIM == 128) && "DIM ERROR!");
    constexpr int MMA_M = 64;
    constexpr int MMA_N = DIM <= 256 ? DIM : 256;
    constexpr int MMA_K = 16;
    
    // setting shared memory
    extern __shared__ __align__(128) uint8_t smem[];
    SMemTMA<BM, BN, DIM, NUM_STAGE> &s = *reinterpret_cast<SMemTMA<BM, BN, DIM, NUM_STAGE>*>(smem);
    fp16 *sQ = s.Q, *sK = s.K, *sV = s.V;
    Barrier *Qmbar = &s.Qmbar;
    Barrier *Kempty = s.Kempty, *Vempty = s.Vempty, *Kfull = s.Kfull, *Vfull = s.Vfull;

    // init mbarrier
    if (threadIdx.x == 0) {
        init_barrier(Qmbar, 1);
        for (int i = 0; i < NUM_STAGE; ++i) {
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
    if (tid >= 256) {  // producer
        warpgroup_reg_dealloc<24>();
        if (tid == 256) {
            int stage = 0, phase = 0;
            // load Q
            expect_bytes(Qmbar, BM * DIM * sizeof(fp16));
            if (do_profile) { profile->load_q_start = clock64(); }
            load_async(sQ, &tensorMapQ, Qmbar, bs, hn, by * BM, 0);
            for (size_t iw=0; iw<S; iw+=BN, ++stage) {
                if (stage >= NUM_STAGE) { stage = 0; phase ^= 1; }
                fp16 *KAddr = sK + stage * BN * DIM;
                fp16 *VAddr = sV + stage * BN * DIM;
                // load K
                wait(&Kempty[stage], phase);
                expect_bytes(&Kfull[stage], BN * DIM * sizeof(fp16));
                const int slot = static_cast<int>(iw / BN);
                if (do_profile && slot < kProfileSlots) { profile->load_k_starts[slot] = clock64(); }
                load_async(KAddr, &tensorMapK, &Kfull[stage], bs, hn, iw, 0);
                // load V
                wait(&Vempty[stage], phase);
                expect_bytes(&Vfull[stage], BN * DIM * sizeof(fp16));
                if (do_profile && slot < kProfileSlots) { profile->load_v_starts[slot] = clock64(); }
                load_async(VAddr, &tensorMapV, &Vfull[stage], bs, hn, iw, 0);
            }
        }
    } else {  // consumer
        warpgroup_reg_alloc<240>();
        // Bootstrap empty-stage barriers so producer can issue the first K/V loads.
        #pragma unroll
        for (int st = 0; st < NUM_STAGE; ++st) {
            arrive(&Kempty[st]);
            arrive(&Vempty[st]);
        }
        wait(Qmbar, 0);
        if (do_profile && tid == 0) {
            profile->load_q_end = clock64();
        }
        // store Q
        // store_async(&tensorMapQ, sQ, bs, hn, by * BM, 0);
        // tma_store_arrive();
        // tma_store_wait();

        int stage = 0, phase = 0;
        // main for loop
        for (size_t iw=0; iw<S; iw+=BN, ++stage) {
            if (stage >= NUM_STAGE) { stage = 0; phase ^= 1; }
            // fp16 *KAddr = sK + stage * BN * DIM;
            // fp16 *VAddr = sV + stage * BN * DIM;
            const int slot = static_cast<int>(iw / BN);
            wait(&Kfull[stage], phase);
            if (do_profile && tid == 0 && slot < kProfileSlots) {
                profile->load_k_ends[slot] = clock64();
            }
            
            // // store K
            // store_async(&tensorMapK, KAddr, bs, hn, iw, 0);
            // tma_store_arrive();
            // tma_store_wait();
            arrive(&Kempty[stage]);  // 释放 tma K 前阻塞

            wait(&Vfull[stage], phase);
            if (do_profile && tid == 0 && slot < kProfileSlots) {
                profile->load_v_ends[slot] = clock64();
            }
            // // store V
            // store_async(&tensorMapV, VAddr, bs, hn, iw, 0);
            // tma_store_arrive();
            // tma_store_wait();
            arrive(&Vempty[stage]);  // 释放 tma V 前阻塞
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


template<int B, int H, int S, int D=128, int BM=128, int BN=128, int NUM_THREADS=384, int NUM_STAGE=1>
void runAttnTMAKernel(fp16 *Q, fp16 *K, fp16 *V, Profile *profile) {
    CUtensorMap d_tma_map_Q = create_tensor_map<BM, D>(Q, B, H, S, D);
    CUtensorMap d_tma_map_K = create_tensor_map<BN, D>(K, B, H, S, D);
    CUtensorMap d_tma_map_V = create_tensor_map<BN, D>(V, B, H, S, D);

    auto* kernel = attnTMAKernel<BM, BN, D, NUM_THREADS, NUM_STAGE>;
    constexpr size_t sMemSize = sizeof(SMemTMA<BM, BN, D, NUM_STAGE>);
    static_assert(sMemSize < 256 * 1024);
    cudaCheck(cudaFuncSetAttribute(
        kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, sMemSize));

    dim3 grid = {static_cast<unsigned int>(S/BM), static_cast<unsigned int>(H), static_cast<unsigned int>(B)};
    kernel<<<grid, NUM_THREADS, sMemSize>>>(B, H, S, d_tma_map_Q, d_tma_map_K, d_tma_map_V, profile);
}

template<int B, int H, int S, int D, int BM, int BN, int NUM_THREADS=384, int NUM_STAGE=1>
void run_one_case(fp16 *dQ, fp16 *dK, fp16 *dV, Profile *d_profile) {
    Profile zero_profile{};

    constexpr int kWarmupIters = 10;
    constexpr int kCaptureIters = 100;

    for (int i = 0; i < kWarmupIters; ++i) {
        cudaCheck(cudaMemcpy(d_profile, &zero_profile, sizeof(Profile), cudaMemcpyHostToDevice));
        runAttnTMAKernel<B, H, S, D, BM, BN, NUM_THREADS, NUM_STAGE>(dQ, dK, dV, d_profile);
        cudaCheck(cudaGetLastError());
    }
    cudaCheck(cudaDeviceSynchronize());

    for (int i = 0; i < kCaptureIters; ++i) {
        cudaCheck(cudaMemcpy(d_profile, &zero_profile, sizeof(Profile), cudaMemcpyHostToDevice));
        runAttnTMAKernel<B, H, S, D, BM, BN, NUM_THREADS, NUM_STAGE>(dQ, dK, dV, d_profile);
        cudaCheck(cudaGetLastError());
    }
    cudaCheck(cudaDeviceSynchronize());

    Profile h_profile{};
    cudaCheck(cudaMemcpy(&h_profile, d_profile, sizeof(Profile), cudaMemcpyDeviceToHost));

    const int slots = (h_profile.kv_iters < kProfileSlots) ? h_profile.kv_iters : kProfileSlots;

    unsigned long long base = h_profile.load_q_start;
    for (int i = 0; i < slots; ++i) {
        if (h_profile.load_k_starts[i] < base) base = h_profile.load_k_starts[i];
        if (h_profile.load_v_starts[i] < base) base = h_profile.load_v_starts[i];
    }

    printf("\n### BM = %d, BN = %d\n\n", BM, BN);
    printf("| op | slot | start_abs | end_abs | start_rel | end_rel | cycles |\n");
    printf("| --- | ---: | ---: | ---: | ---: | ---: | ---: |\n");

    const unsigned long long q_start = h_profile.load_q_start;
    const unsigned long long q_end   = h_profile.load_q_end;

    printf("| Q | 0 | %llu | %llu | %llu | %llu | %llu |\n",
           q_start % 10000,
           q_end % 10000,
           q_start - base,
           q_end - base,
           q_end - q_start);

    for (int i = 0; i < slots; ++i) {
        const unsigned long long ks = h_profile.load_k_starts[i];
        const unsigned long long ke = h_profile.load_k_ends[i];
        const unsigned long long vs = h_profile.load_v_starts[i];
        const unsigned long long ve = h_profile.load_v_ends[i];

        printf("| K | %d | %llu | %llu | %llu | %llu | %llu |\n",
               i,
               ks % 10000,
               ke % 10000,
               ks - base,
               ke - base,
               ke - ks);

        printf("| V | %d | %llu | %llu | %llu | %llu | %llu |\n",
               i,
               vs % 10000,
               ve % 10000,
               vs - base,
               ve - base,
               ve - vs);
    }
}

// nvcc -std=c++17 -arch=sm_90a -O3  tma_ws.cu -o tma_ws_test -lcuda

int main() {
    constexpr int B = 1;
    constexpr int H = 1;
    constexpr int S = 512;
    constexpr int D = 128;

    const size_t numel = static_cast<size_t>(B) * H * S * D;
    const size_t bytes = numel * sizeof(fp16);

    std::vector<fp16> hQ_in(numel), hK_in(numel), hV_in(numel);

    for (size_t i = 0; i < numel; ++i) {
        const int q_i = static_cast<int>((i * 13 + 7) % 31) - 15;
        const int k_i = static_cast<int>((i * 17 + 5) % 29) - 14;
        const int v_i = static_cast<int>((i * 19 + 3) % 23) - 11;

        hQ_in[i] = __float2half_rn(static_cast<float>(q_i) * (1.0f / 128.0f));
        hK_in[i] = __float2half_rn(static_cast<float>(k_i) * (1.0f / 128.0f));
        hV_in[i] = __float2half_rn(static_cast<float>(v_i) * (1.0f / 128.0f));
    }

    fp16 *dQ = nullptr;
    fp16 *dK = nullptr;
    fp16 *dV = nullptr;

    cudaCheck(cudaMalloc(&dQ, bytes));
    cudaCheck(cudaMalloc(&dK, bytes));
    cudaCheck(cudaMalloc(&dV, bytes));

    cudaCheck(cudaMemcpy(dQ, hQ_in.data(), bytes, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(dK, hK_in.data(), bytes, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(dV, hV_in.data(), bytes, cudaMemcpyHostToDevice));

    Profile *d_profile = nullptr;
    cudaCheck(cudaMalloc(&d_profile, sizeof(Profile)));

    printf("# TMA Load Timeline\n\n");
    printf("B = %d, H = %d, S = %d, D = %d\n", B, H, S, D);

    // run_one_case<B, H, S, D, 128,  32, 384, 2>(dQ, dK, dV, d_profile);
    // run_one_case<B, H, S, D, 128,  64, 384, 2>(dQ, dK, dV, d_profile);
    // run_one_case<B, H, S, D, 128, 128, 384, 2>(dQ, dK, dV, d_profile);

    run_one_case<B, H, S, D, 128,  32>(dQ, dK, dV, d_profile);
    run_one_case<B, H, S, D, 128,  64>(dQ, dK, dV, d_profile);
    run_one_case<B, H, S, D, 128, 128>(dQ, dK, dV, d_profile);
    run_one_case<B, H, S, D, 128, 256>(dQ, dK, dV, d_profile);

    run_one_case<B, H, S, D, 256,  32>(dQ, dK, dV, d_profile);
    run_one_case<B, H, S, D, 256,  64>(dQ, dK, dV, d_profile);
    run_one_case<B, H, S, D, 256, 128>(dQ, dK, dV, d_profile);

    cudaCheck(cudaFree(dQ));
    cudaCheck(cudaFree(dK));
    cudaCheck(cudaFree(dV));
    cudaCheck(cudaFree(d_profile));

    return 0;
}