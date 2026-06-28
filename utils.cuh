#ifndef UTILS_CUH
#define UTILS_CUH

#include "tools.cuh"

template<
int BM, 
int BN, 
int BK, 
int MMA_M, 
int MMA_N, 
int MMA_K, 
int LAYOUT_M, 
int LAYOUT_N, 
int LBO,
int SBO,
bool TRAN_A, 
bool TRAN_B,
bool CLEAR,
bool SWIZZLE>
__device__ __forceinline__ void gemm_ss(fp16 *A, fp16 *B, fp16 *C, const int wg_idx) {
    const int wg_y = wg_idx / LAYOUT_N;
    const int wg_x = wg_idx % LAYOUT_N;

    // clear
    

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
                wgmma_ss<QK_MMA_N, 1, 1, 1, 0, 0, 16, 1024, SWIZZLE>(acc_s[i][j], _QAddr, _KAddr);
            }
        }
    }
    warpgroup_commit_batch();
    warpgroup_wait();
}


#endif