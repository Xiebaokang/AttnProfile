#ifndef TOOLS_CUH
#define TOOLS_CUH

#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cfloat>


void cudaCheck(cudaError_t error, const char *file, int line) {
  if (error != cudaSuccess) {
    printf("[CUDA ERROR] at file %s:%d:\n%s\n", file, line,
           cudaGetErrorString(error));
    exit(1);
  }
}
#define cudaCheck(err) (cudaCheck(err, __FILE__, __LINE__))

// rename
typedef half fp16;
typedef float fp32;
typedef uint64_t Barrier;

// ptx inline
__device__ static inline uint64_t matrix_descriptor_encode(uint64_t x) {
    return (((x) & 0x3FFFF) >> 0x4);
}

template<int LBO=16, int SBO=2048, bool Swizzle=true>
__device__ uint64_t make_smem_desc(fp16* ptr) {
    // Convert shared memory pointer to integer
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
    uint64_t desc = matrix_descriptor_encode(addr);
    desc |= matrix_descriptor_encode((uint64_t)LBO) << 16;
    desc |= matrix_descriptor_encode((uint64_t)SBO) << 32;
    desc |= (Swizzle ? 1llu : 0llu) << 62;
    return desc;
}

__device__ void warpgroup_arrive() {
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
}

__device__ void warpgroup_commit_batch() {
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}

template <int WaitGroup=0>
__device__ __forceinline__ void warpgroup_wait() {
    static_assert(WaitGroup >= 0 && WaitGroup <= 7,
                  "WGMMA wait group must be in range [0, 7]");
    asm volatile(
        "wgmma.wait_group.sync.aligned %0;\n"
        :
        : "n"(WaitGroup)
        : "memory"
    );
}

__forceinline__ __device__ void warpgroup_fence_operand(float& reg) {
    asm volatile("" : "+f"(reg) :: "memory");
}

// ss mode
template<int ScaleD, int ScaleA, int ScaleB, int TransA, int TransB, int LBO, int SBO, bool Swizzle>
__device__ void wgmma256ss(float d[16][8], fp16* sA, fp16* sB) {
    uint64_t desc_a = make_smem_desc<LBO, SBO, Swizzle>(&sA[0]);
    uint64_t desc_b = make_smem_desc<LBO, SBO, Swizzle>(&sB[0]);
    asm volatile(
        "{\n"
        "wgmma.mma_async.sync.aligned.m64n256k16.f32.f16.f16 "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
        " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
        " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
        " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  "
        " %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  "
        " %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47,  "
        " %48,  %49,  %50,  %51,  %52,  %53,  %54,  %55,  "
        " %56,  %57,  %58,  %59,  %60,  %61,  %62,  %63,  "
        " %64,  %65,  %66,  %67,  %68,  %69,  %70,  %71,  "
        " %72,  %73,  %74,  %75,  %76,  %77,  %78,  %79,  "
        " %80,  %81,  %82,  %83,  %84,  %85,  %86,  %87,  "
        " %88,  %89,  %90,  %91,  %92,  %93,  %94,  %95,  "
        " %96,  %97,  %98,  %99,  %100, %101, %102, %103,  "
        " %104, %105, %106, %107, %108, %109, %110, %111,  "
        " %112, %113, %114, %115, %116, %117, %118, %119,  "
        " %120, %121, %122, %123, %124, %125, %126, %127},"
        " %128,"
        " %129,"
        " %130,    %131,  %132,  %133,  %134;\n"
        "}\n"
        :   "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[0][4]), "+f"(d[0][5]), "+f"(d[0][6]), "+f"(d[0][7]),
            "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]), "+f"(d[1][4]), "+f"(d[1][5]), "+f"(d[1][6]), "+f"(d[1][7]),
            "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]), "+f"(d[2][4]), "+f"(d[2][5]), "+f"(d[2][6]), "+f"(d[2][7]),
            "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]), "+f"(d[3][4]), "+f"(d[3][5]), "+f"(d[3][6]), "+f"(d[3][7]),
            "+f"(d[4][0]), "+f"(d[4][1]), "+f"(d[4][2]), "+f"(d[4][3]), "+f"(d[4][4]), "+f"(d[4][5]), "+f"(d[4][6]), "+f"(d[4][7]),
            "+f"(d[5][0]), "+f"(d[5][1]), "+f"(d[5][2]), "+f"(d[5][3]), "+f"(d[5][4]), "+f"(d[5][5]), "+f"(d[5][6]), "+f"(d[5][7]),
            "+f"(d[6][0]), "+f"(d[6][1]), "+f"(d[6][2]), "+f"(d[6][3]), "+f"(d[6][4]), "+f"(d[6][5]), "+f"(d[6][6]), "+f"(d[6][7]),
            "+f"(d[7][0]), "+f"(d[7][1]), "+f"(d[7][2]), "+f"(d[7][3]), "+f"(d[7][4]), "+f"(d[7][5]), "+f"(d[7][6]), "+f"(d[7][7]),
            "+f"(d[8][0]), "+f"(d[8][1]), "+f"(d[8][2]), "+f"(d[8][3]), "+f"(d[8][4]), "+f"(d[8][5]), "+f"(d[8][6]), "+f"(d[8][7]),
            "+f"(d[9][0]), "+f"(d[9][1]), "+f"(d[9][2]), "+f"(d[9][3]), "+f"(d[9][4]), "+f"(d[9][5]), "+f"(d[9][6]), "+f"(d[9][7]),
            "+f"(d[10][0]), "+f"(d[10][1]), "+f"(d[10][2]), "+f"(d[10][3]), "+f"(d[10][4]), "+f"(d[10][5]), "+f"(d[10][6]), "+f"(d[10][7]),
            "+f"(d[11][0]), "+f"(d[11][1]), "+f"(d[11][2]), "+f"(d[11][3]), "+f"(d[11][4]), "+f"(d[11][5]), "+f"(d[11][6]), "+f"(d[11][7]),
            "+f"(d[12][0]), "+f"(d[12][1]), "+f"(d[12][2]), "+f"(d[12][3]), "+f"(d[12][4]), "+f"(d[12][5]), "+f"(d[12][6]), "+f"(d[12][7]),
            "+f"(d[13][0]), "+f"(d[13][1]), "+f"(d[13][2]), "+f"(d[13][3]), "+f"(d[13][4]), "+f"(d[13][5]), "+f"(d[13][6]), "+f"(d[13][7]),
            "+f"(d[14][0]), "+f"(d[14][1]), "+f"(d[14][2]), "+f"(d[14][3]), "+f"(d[14][4]), "+f"(d[14][5]), "+f"(d[14][6]), "+f"(d[14][7]),
            "+f"(d[15][0]), "+f"(d[15][1]), "+f"(d[15][2]), "+f"(d[15][3]), "+f"(d[15][4]), "+f"(d[15][5]), "+f"(d[15][6]), "+f"(d[15][7])
        : "l"(desc_a), "l"(desc_b), "n"(int32_t(ScaleD)), "n"(int32_t(ScaleA)),
            "n"(int32_t(ScaleB)), "n"(int32_t(TransA)), "n"(int32_t(TransB)));
}

template<int ScaleD, int ScaleA, int ScaleB, int TransA, int TransB, int LBO, int SBO, bool Swizzle>
__device__ void wgmma128ss(float d[8][8], fp16* sA, fp16* sB) {
    uint64_t desc_a = make_smem_desc<LBO, SBO, Swizzle>(&sA[0]);
    uint64_t desc_b = make_smem_desc<LBO, SBO, Swizzle>(&sB[0]);
    asm volatile(
        "{\n"
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16 "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
        " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
        " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
        " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  "
        " %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  "
        " %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47,  "
        " %48,  %49,  %50,  %51,  %52,  %53,  %54,  %55,  "
        " %56,  %57,  %58,  %59,  %60,  %61,  %62,  %63},"
        " %64,"
        " %65,"
        " %66,    %67,  %68,  %69,  %70;\n"
        "}\n"
        : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[0][4]), "+f"(d[0][5]),
            "+f"(d[0][6]), "+f"(d[0][7]), "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]),
            "+f"(d[1][4]), "+f"(d[1][5]), "+f"(d[1][6]), "+f"(d[1][7]), "+f"(d[2][0]), "+f"(d[2][1]),
            "+f"(d[2][2]), "+f"(d[2][3]), "+f"(d[2][4]), "+f"(d[2][5]), "+f"(d[2][6]), "+f"(d[2][7]),
            "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]), "+f"(d[3][4]), "+f"(d[3][5]),
            "+f"(d[3][6]), "+f"(d[3][7]), "+f"(d[4][0]), "+f"(d[4][1]), "+f"(d[4][2]), "+f"(d[4][3]),
            "+f"(d[4][4]), "+f"(d[4][5]), "+f"(d[4][6]), "+f"(d[4][7]), "+f"(d[5][0]), "+f"(d[5][1]),
            "+f"(d[5][2]), "+f"(d[5][3]), "+f"(d[5][4]), "+f"(d[5][5]), "+f"(d[5][6]), "+f"(d[5][7]),
            "+f"(d[6][0]), "+f"(d[6][1]), "+f"(d[6][2]), "+f"(d[6][3]), "+f"(d[6][4]), "+f"(d[6][5]),
            "+f"(d[6][6]), "+f"(d[6][7]), "+f"(d[7][0]), "+f"(d[7][1]), "+f"(d[7][2]), "+f"(d[7][3]),
            "+f"(d[7][4]), "+f"(d[7][5]), "+f"(d[7][6]), "+f"(d[7][7])
        : "l"(desc_a), "l"(desc_b), "n"(int32_t(ScaleD)), "n"(int32_t(ScaleA)),
            "n"(int32_t(ScaleB)), "n"(int32_t(TransA)), "n"(int32_t(TransB)));
}

template<int ScaleD, int ScaleA, int ScaleB, int TransA, int TransB, int LBO, int SBO, bool Swizzle>
__device__ void wgmma64ss(float d[4][8], fp16* sA, fp16* sB) {
    uint64_t desc_a = make_smem_desc<LBO, SBO, Swizzle>(&sA[0]);
    uint64_t desc_b = make_smem_desc<LBO, SBO, Swizzle>(&sB[0]);
    asm volatile(
        "{\n"
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16 "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
        " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
        " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
        " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31},"
        " %32,"
        " %33,"
        " %34, %35, %36, %37, %38;\n"
        "}\n"
        : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[0][4]), "+f"(d[0][5]),
            "+f"(d[0][6]), "+f"(d[0][7]), "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]),
            "+f"(d[1][4]), "+f"(d[1][5]), "+f"(d[1][6]), "+f"(d[1][7]), "+f"(d[2][0]), "+f"(d[2][1]),
            "+f"(d[2][2]), "+f"(d[2][3]), "+f"(d[2][4]), "+f"(d[2][5]), "+f"(d[2][6]), "+f"(d[2][7]),
            "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]), "+f"(d[3][4]), "+f"(d[3][5]),
            "+f"(d[3][6]), "+f"(d[3][7])
        : "l"(desc_a), "l"(desc_b), "n"(int32_t(ScaleD)), "n"(int32_t(ScaleA)),
            "n"(int32_t(ScaleB)), "n"(int32_t(TransA)), "n"(int32_t(TransB)));
}

template<int ScaleD, int ScaleA, int ScaleB, int TransA, int TransB, int LBO, int SBO, bool Swizzle>
__device__ void wgmma32ss(float d[2][8], fp16* sA, fp16* sB) {
    uint64_t desc_a = make_smem_desc<LBO, SBO, Swizzle>(&sA[0]);
    uint64_t desc_b = make_smem_desc<LBO, SBO, Swizzle>(&sB[0]);
    asm volatile(
        "{\n"
        "wgmma.mma_async.sync.aligned.m64n32k16.f32.f16.f16 "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
        " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15},  "
        " %16,"
        " %17,"
        " %18, %19, %20, %21, %22;\n"
        "}\n"
        : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[0][4]), "+f"(d[0][5]),
            "+f"(d[0][6]), "+f"(d[0][7]), "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]),
            "+f"(d[1][4]), "+f"(d[1][5]), "+f"(d[1][6]), "+f"(d[1][7])
        : "l"(desc_a), "l"(desc_b), "n"(int32_t(ScaleD)), "n"(int32_t(ScaleA)),
            "n"(int32_t(ScaleB)), "n"(int32_t(TransA)), "n"(int32_t(TransB)));
}

template<int ScaleD, int ScaleA, int ScaleB, int TransA, int TransB, int LBO, int SBO, bool Swizzle>
__device__ void wgmma16ss(float d[1][8], fp16* sA, fp16* sB) {
    uint64_t desc_a = make_smem_desc<LBO, SBO, Swizzle>(&sA[0]);
    uint64_t desc_b = make_smem_desc<LBO, SBO, Swizzle>(&sB[0]);
    asm volatile(
        "{\n"
        "wgmma.mma_async.sync.aligned.m64n16k16.f32.f16.f16 "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7},   "
        " %8,"
        " %9,"
        " %10, %11, %12, %13, %14;\n"
        "}\n"
        : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[0][4]), "+f"(d[0][5]),
            "+f"(d[0][6]), "+f"(d[0][7])
        : "l"(desc_a), "l"(desc_b), "n"(int32_t(ScaleD)), "n"(int32_t(ScaleA)),
            "n"(int32_t(ScaleB)), "n"(int32_t(TransA)), "n"(int32_t(TransB)));
}

template<int WGMMA_N, int ScaleD=1, int ScaleA=1, int ScaleB=1, int TransA=0, int TransB=0, int LBO=16, int SBO=2048, bool Swizzle=true>
__device__ inline void wgmma_ss(float d[WGMMA_N/16][8], fp16* sA, fp16* sB) {
    static_assert(WGMMA_N == 16 || WGMMA_N == 32 || WGMMA_N == 64 || WGMMA_N == 128 || WGMMA_N == 256);
    if constexpr (WGMMA_N == 256)
        wgmma256ss<ScaleD, ScaleA, ScaleB, TransA, TransB, LBO, SBO, Swizzle>(d, sA, sB);
    if constexpr (WGMMA_N == 128)
        wgmma128ss<ScaleD, ScaleA, ScaleB, TransA, TransB, LBO, SBO, Swizzle>(d, sA, sB);
    if constexpr (WGMMA_N == 64)
        wgmma64ss<ScaleD, ScaleA, ScaleB, TransA, TransB, LBO, SBO, Swizzle>(d, sA, sB);
    if constexpr (WGMMA_N == 32)
        wgmma32ss<ScaleD, ScaleA, ScaleB, TransA, TransB, LBO, SBO, Swizzle>(d, sA, sB);
    if constexpr (WGMMA_N == 16)
        wgmma16ss<ScaleD, ScaleA, ScaleB, TransA, TransB, LBO, SBO, Swizzle>(d, sA, sB);
}

// rs mode
template<int ScaleD, int ScaleA, int ScaleB, int TransB, int LBO, int SBO, bool Swizzle>
__device__ void wgmma256rs(float d[16][8], uint32_t a[4], fp16* sB) {
    uint64_t desc_b = make_smem_desc<LBO, SBO, Swizzle>(&sB[0]);

    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %133, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n256k16.f32.f16.f16 "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
        " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
        " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
        " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  "
        " %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  "
        " %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47,  "
        " %48,  %49,  %50,  %51,  %52,  %53,  %54,  %55,  "
        " %56,  %57,  %58,  %59,  %60,  %61,  %62,  %63,  "
        " %64,  %65,  %66,  %67,  %68,  %69,  %70,  %71,  "
        " %72,  %73,  %74,  %75,  %76,  %77,  %78,  %79,  "
        " %80,  %81,  %82,  %83,  %84,  %85,  %86,  %87,  "
        " %88,  %89,  %90,  %91,  %92,  %93,  %94,  %95,  "
        " %96,  %97,  %98,  %99,  %100, %101, %102, %103, "
        " %104, %105, %106, %107, %108, %109, %110, %111, "
        " %112, %113, %114, %115, %116, %117, %118, %119, "
        " %120, %121, %122, %123, %124, %125, %126, %127}, "
        "{%128, %129, %130, %131}, "
        "%132, "
        "p, %134, %135, %136;\n"
        "}\n"
        :   "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[0][4]), "+f"(d[0][5]), "+f"(d[0][6]), "+f"(d[0][7]),
            "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]), "+f"(d[1][4]), "+f"(d[1][5]), "+f"(d[1][6]), "+f"(d[1][7]),
            "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]), "+f"(d[2][4]), "+f"(d[2][5]), "+f"(d[2][6]), "+f"(d[2][7]),
            "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]), "+f"(d[3][4]), "+f"(d[3][5]), "+f"(d[3][6]), "+f"(d[3][7]),
            "+f"(d[4][0]), "+f"(d[4][1]), "+f"(d[4][2]), "+f"(d[4][3]), "+f"(d[4][4]), "+f"(d[4][5]), "+f"(d[4][6]), "+f"(d[4][7]),
            "+f"(d[5][0]), "+f"(d[5][1]), "+f"(d[5][2]), "+f"(d[5][3]), "+f"(d[5][4]), "+f"(d[5][5]), "+f"(d[5][6]), "+f"(d[5][7]),
            "+f"(d[6][0]), "+f"(d[6][1]), "+f"(d[6][2]), "+f"(d[6][3]), "+f"(d[6][4]), "+f"(d[6][5]), "+f"(d[6][6]), "+f"(d[6][7]),
            "+f"(d[7][0]), "+f"(d[7][1]), "+f"(d[7][2]), "+f"(d[7][3]), "+f"(d[7][4]), "+f"(d[7][5]), "+f"(d[7][6]), "+f"(d[7][7]),
            "+f"(d[8][0]), "+f"(d[8][1]), "+f"(d[8][2]), "+f"(d[8][3]), "+f"(d[8][4]), "+f"(d[8][5]), "+f"(d[8][6]), "+f"(d[8][7]),
            "+f"(d[9][0]), "+f"(d[9][1]), "+f"(d[9][2]), "+f"(d[9][3]), "+f"(d[9][4]), "+f"(d[9][5]), "+f"(d[9][6]), "+f"(d[9][7]),
            "+f"(d[10][0]), "+f"(d[10][1]), "+f"(d[10][2]), "+f"(d[10][3]), "+f"(d[10][4]), "+f"(d[10][5]), "+f"(d[10][6]), "+f"(d[10][7]),
            "+f"(d[11][0]), "+f"(d[11][1]), "+f"(d[11][2]), "+f"(d[11][3]), "+f"(d[11][4]), "+f"(d[11][5]), "+f"(d[11][6]), "+f"(d[11][7]),
            "+f"(d[12][0]), "+f"(d[12][1]), "+f"(d[12][2]), "+f"(d[12][3]), "+f"(d[12][4]), "+f"(d[12][5]), "+f"(d[12][6]), "+f"(d[12][7]),
            "+f"(d[13][0]), "+f"(d[13][1]), "+f"(d[13][2]), "+f"(d[13][3]), "+f"(d[13][4]), "+f"(d[13][5]), "+f"(d[13][6]), "+f"(d[13][7]),
            "+f"(d[14][0]), "+f"(d[14][1]), "+f"(d[14][2]), "+f"(d[14][3]), "+f"(d[14][4]), "+f"(d[14][5]), "+f"(d[14][6]), "+f"(d[14][7]),
            "+f"(d[15][0]), "+f"(d[15][1]), "+f"(d[15][2]), "+f"(d[15][3]), "+f"(d[15][4]), "+f"(d[15][5]), "+f"(d[15][6]), "+f"(d[15][7])
        :   "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
            "l"(desc_b),
            "r"(int32_t(ScaleD)), "n"(int32_t(ScaleA)),
            "n"(int32_t(ScaleB)), "n"(int32_t(TransB))
        : "memory");
}

template<int ScaleD, int ScaleA, int ScaleB, int TransB, int LBO, int SBO, bool Swizzle>
__device__ void wgmma128rs(float d[8][8], uint32_t a[4], fp16* sB) {
    uint64_t desc_b = make_smem_desc<LBO, SBO, Swizzle>(&sB[0]);

    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %69, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16 "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
        " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
        " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
        " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  "
        " %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  "
        " %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47,  "
        " %48,  %49,  %50,  %51,  %52,  %53,  %54,  %55,  "
        " %56,  %57,  %58,  %59,  %60,  %61,  %62,  %63}, "
        "{%64, %65, %66, %67}, "
        "%68, "
        "p, %70, %71, %72;\n"
        "}\n"
        : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[0][4]), "+f"(d[0][5]), "+f"(d[0][6]), "+f"(d[0][7]),
          "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]), "+f"(d[1][4]), "+f"(d[1][5]), "+f"(d[1][6]), "+f"(d[1][7]),
          "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]), "+f"(d[2][4]), "+f"(d[2][5]), "+f"(d[2][6]), "+f"(d[2][7]),
          "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]), "+f"(d[3][4]), "+f"(d[3][5]), "+f"(d[3][6]), "+f"(d[3][7]),
          "+f"(d[4][0]), "+f"(d[4][1]), "+f"(d[4][2]), "+f"(d[4][3]), "+f"(d[4][4]), "+f"(d[4][5]), "+f"(d[4][6]), "+f"(d[4][7]),
          "+f"(d[5][0]), "+f"(d[5][1]), "+f"(d[5][2]), "+f"(d[5][3]), "+f"(d[5][4]), "+f"(d[5][5]), "+f"(d[5][6]), "+f"(d[5][7]),
          "+f"(d[6][0]), "+f"(d[6][1]), "+f"(d[6][2]), "+f"(d[6][3]), "+f"(d[6][4]), "+f"(d[6][5]), "+f"(d[6][6]), "+f"(d[6][7]),
          "+f"(d[7][0]), "+f"(d[7][1]), "+f"(d[7][2]), "+f"(d[7][3]), "+f"(d[7][4]), "+f"(d[7][5]), "+f"(d[7][6]), "+f"(d[7][7])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "l"(desc_b),
          "r"(int32_t(ScaleD)), "n"(int32_t(ScaleA)),
          "n"(int32_t(ScaleB)), "n"(int32_t(TransB))
        : "memory");
}

template<int ScaleD, int ScaleA, int ScaleB, int TransB, int LBO, int SBO, bool Swizzle>
__device__ void wgmma64rs(float d[4][8], uint32_t a[4], fp16* sB) {
    uint64_t desc_b = make_smem_desc<LBO, SBO, Swizzle>(&sB[0]);

    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %37, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16 "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
        " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
        " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
        " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31}, "
        "{%32, %33, %34, %35}, "
        "%36, "
        "p, %38, %39, %40;\n"
        "}\n"
        : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[0][4]), "+f"(d[0][5]), "+f"(d[0][6]), "+f"(d[0][7]),
          "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]), "+f"(d[1][4]), "+f"(d[1][5]), "+f"(d[1][6]), "+f"(d[1][7]),
          "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]), "+f"(d[2][4]), "+f"(d[2][5]), "+f"(d[2][6]), "+f"(d[2][7]),
          "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]), "+f"(d[3][4]), "+f"(d[3][5]), "+f"(d[3][6]), "+f"(d[3][7])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "l"(desc_b),
          "r"(int32_t(ScaleD)), "n"(int32_t(ScaleA)),
          "n"(int32_t(ScaleB)), "n"(int32_t(TransB))
        : "memory");
}

template<int ScaleD, int ScaleA, int ScaleB, int TransB, int LBO, int SBO, bool Swizzle>
__device__ void wgmma32rs(float d[2][8], uint32_t a[4], fp16* sB) {
    uint64_t desc_b = make_smem_desc<LBO, SBO, Swizzle>(&sB[0]);

    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %21, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n32k16.f32.f16.f16 "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
        " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15}, "
        "{%16, %17, %18, %19}, "
        "%20, "
        "p, %22, %23, %24;\n"
        "}\n"
        : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[0][4]), "+f"(d[0][5]), "+f"(d[0][6]), "+f"(d[0][7]),
          "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]), "+f"(d[1][4]), "+f"(d[1][5]), "+f"(d[1][6]), "+f"(d[1][7])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "l"(desc_b),
          "r"(int32_t(ScaleD)), "n"(int32_t(ScaleA)),
          "n"(int32_t(ScaleB)), "n"(int32_t(TransB))
        : "memory");
}

template<int ScaleD, int ScaleA, int ScaleB, int TransB, int LBO, int SBO, bool Swizzle>
__device__ void wgmma16rs(float d[1][8], uint32_t a[4], fp16* sB) {
    uint64_t desc_b = make_smem_desc<LBO, SBO, Swizzle>(&sB[0]);

    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %13, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n16k16.f32.f16.f16 "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7}, "
        "{%8, %9, %10, %11}, "
        "%12, "
        "p, %14, %15, %16;\n"
        "}\n"
        : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]),
          "+f"(d[0][4]), "+f"(d[0][5]), "+f"(d[0][6]), "+f"(d[0][7])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "l"(desc_b),
          "r"(int32_t(ScaleD)), "n"(int32_t(ScaleA)),
          "n"(int32_t(ScaleB)), "n"(int32_t(TransB))
        : "memory");
}

template<int WGMMA_N, int ScaleD=1, int ScaleA=1, int ScaleB=1, int TransA=0, int TransB=0, int LBO=16, int SBO=2048, bool Swizzle=true>
__device__ inline void wgmma_rs(float d[WGMMA_N / 16][8], uint32_t a[4], fp16* sB) {
    static_assert(WGMMA_N == 16 || WGMMA_N == 32 || WGMMA_N == 64 || WGMMA_N == 128 || WGMMA_N == 256);
    if constexpr (WGMMA_N == 256) {
        wgmma256rs<ScaleD, ScaleA, ScaleB, TransB, LBO, SBO, Swizzle>(d, a, sB);
    } else if constexpr (WGMMA_N == 128) {
        wgmma128rs<ScaleD, ScaleA, ScaleB, TransB, LBO, SBO, Swizzle>(d, a, sB);
    } else if constexpr (WGMMA_N == 64) {
        wgmma64rs<ScaleD, ScaleA, ScaleB, TransB, LBO, SBO, Swizzle>(d, a, sB);
    } else if constexpr (WGMMA_N == 32) {
        wgmma32rs<ScaleD, ScaleA, ScaleB, TransB, LBO, SBO, Swizzle>(d, a, sB);
    } else if constexpr (WGMMA_N == 16) {
        wgmma16rs<ScaleD, ScaleA, ScaleB, TransB, LBO, SBO, Swizzle>(d, a, sB);
    }
}

__device__ static inline void load_async(
    fp16 *dst, 
    void const* src_tma_map, 
    uint64_t* bar, 
    int b1, int b2, int global_row_idx, int global_col_idx
) {
    uint64_t tma_ptr  = reinterpret_cast<uint64_t>(src_tma_map);
    uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    uint32_t dst_ptr  = static_cast<uint32_t>(__cvta_generic_to_shared(dst));

    asm volatile (
        "cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes "
        "[%0], [%1, {%3, %4, %5, %6, %7}], [%2];"
        :
        : "r"(dst_ptr), "l"(tma_ptr), "r"(mbar_ptr),
            "n"(0), "r"(global_row_idx), "r"(global_col_idx / 64), "r"(b2), "r"(b1)
        : "memory"
    );
}

__device__ static inline void store_async(
    void const* dst_tma_map, 
    fp16 *src, 
    int b1, int b2, int global_row_idx, int global_col_idx
) {
    uint64_t tma_ptr  = reinterpret_cast<uint64_t>(dst_tma_map);
    uint32_t src_ptr  = static_cast<uint32_t>(__cvta_generic_to_shared(src));
    asm volatile (
      "cp.async.bulk.tensor.5d.global.shared::cta.bulk_group" 
      "[%0, {%2, %3, %4, %5, %6}], [%1];"
      :
      : "l"(tma_ptr), "r"(src_ptr),
        "n"(0), "r"(global_row_idx), "r"(global_col_idx / 64), "r"(b2), "r"(b1)
      : "memory"
    );
}

__forceinline__ __device__ void tma_store_arrive() {
    asm volatile("cp.async.bulk.commit_group;");
}

template<int count=0>
__device__ __forceinline__ void tma_store_wait() {
    asm volatile(
      "cp.async.bulk.wait_group.read %0;"
      :
      : "n"(count)
      : "memory");
}

__forceinline__ __device__ void fence_view_async_shared() {
    asm volatile (
        "{\n\t"
        "fence.proxy.async.shared::cta;\n"
        "}"
        ::: "memory");
}

__device__ static __forceinline__ void init_barrier(uint64_t* bar, uint32_t arrive_count) {
    uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar)); 
    asm volatile (
        "mbarrier.init.shared::cta.b64 [%0], %1;\n"
        :: "r"(bar_ptr), "r"(arrive_count));
}

__device__ static __forceinline__ void wait(uint64_t* bar, int kPhaseBit) {
    uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    // Call mbarrier.try_wait in a while loop till it returns true.
    asm volatile (
        "{\n"
        ".reg .pred                P1;\n"
        "LAB_WAIT:\n"
        "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
        "@P1                       bra.uni DONE;\n"
        "bra.uni                   LAB_WAIT;\n"
        "DONE:\n"
        "}\n"
        :: "r"(mbar_ptr),
        "r"(kPhaseBit)
    );
}

__device__ static __forceinline__ void arrive(uint64_t* bar, uint32_t count=1) {
    uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar)); 
    asm volatile (
        "mbarrier.arrive.release.cta.shared::cta.b64 _, [%0], %1;\n"
        :
        : "r"(mbar_ptr), "r"(count)
        : "memory"
    );
}

__device__ static __forceinline__ void expect_bytes(uint64_t* bar, uint32_t bytes) {
    uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar)); 
    asm volatile ("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n"
        :: "r"(bar_ptr), "r"(bytes));
}

template <uint32_t RegCount>
__device__ __forceinline__ void warpgroup_reg_alloc() {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" : : "n"(RegCount));
}

template <uint32_t RegCount>
__device__ __forceinline__ void warpgroup_reg_dealloc() {
    asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" : : "n"(RegCount));
}

template <typename T>
__device__ __forceinline__ void stmatrix_x4(T *smem_addr, const uint32_t data_ptr[4]) {
    uint32_t smem_addr_ = static_cast<uint32_t>(__cvta_generic_to_shared(smem_addr));
    asm volatile(
        "stmatrix.sync.aligned.m8n8.x4.shared::cta.b16 "
        "[%0], {%1, %2, %3, %4};\n"
        :
        : "r"(smem_addr_), "r"(data_ptr[0]), "r"(data_ptr[1]), "r"(data_ptr[2]), "r"(data_ptr[3])
        : "memory"
    );
}

template <typename T>
__device__ __forceinline__ void ldmatrix_x4(T *smem_addr, uint32_t data_ptr[4]) {
    uint32_t smem_addr_ = static_cast<uint32_t>(__cvta_generic_to_shared(smem_addr));

    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared::cta.b16 "
        "{%0, %1, %2, %3}, [%4];\n"
        : "=r"(data_ptr[0]),
          "=r"(data_ptr[1]),
          "=r"(data_ptr[2]),
          "=r"(data_ptr[3])
        : "r"(smem_addr_)
        : "memory"
    );
}


__device__ __forceinline__ void ldmatrix_x4_reg(
    fp16 *smem_addr,
    uint32_t &r0,
    uint32_t &r1,
    uint32_t &r2,
    uint32_t &r3
) {
    uint32_t smem_addr_ = static_cast<uint32_t>(__cvta_generic_to_shared(smem_addr));
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
        "{%0, %1, %2, %3}, [%4];\n"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
        : "r"(smem_addr_)
    );
}

__device__ __forceinline__ void stmatrix_x4_reg(
    fp16 *smem_addr,
    uint32_t r0,
    uint32_t r1,
    uint32_t r2,
    uint32_t r3
) {
    uint32_t smem_addr_ = static_cast<uint32_t>(__cvta_generic_to_shared(smem_addr));
    asm volatile(
        "stmatrix.sync.aligned.m8n8.x4.shared::cta.b16 "
        "[%0], {%1, %2, %3, %4};\n"
        :
        : "r"(smem_addr_), "r"(r0), "r"(r1), "r"(r2), "r"(r3)
        : "memory"
    );
}

__device__ __forceinline__ uint32_t half2_to_u32(half2 h) {
    union {
        half2 h;
        uint32_t u;
    } cvt;
    cvt.h = h;
    return cvt.u;
}

__device__ __forceinline__ half2 u32_to_half2(uint32_t u) {
    union {
        uint32_t u;
        half2 h;
    } cvt;
    cvt.u = u;
    return cvt.h;
}

__device__ static __forceinline__ void bar_sync(uint32_t num_threads, uint32_t barrier_id=0) {
  // barrier_id 指定要使用哪个硬件屏障资源，可以使用不同的屏障阻挡不同thread
  asm volatile("bar.sync %0, %1;\n"
                : 
                : "r"(barrier_id), "r"(num_threads)
                : "memory");
}

template<int BlockMajorSize>
__device__ __forceinline__
int tma_smem_offset_2d(int row, int col) {
    return (col / 64) * BlockMajorSize * 64
         + row * 64
         + (col % 64);
}

template<int BlockMajorSize>
__device__ __forceinline__
int tma_smem_swizzle_128b_offset_2d(int row, int col) {
    // int col_swizzled = (col % 64) ^ ((row & 7) << 3);
    int base_addr = (col / 64) * BlockMajorSize * 64;
    int addr = base_addr + row * 64 + (col % 64);
    return ((addr >> 3) & 0x38) ^ addr;
}


#endif
