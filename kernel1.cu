#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <cuda_runtime.h>
#include "common.h"
#include "matrix.h"

#define TILE_SIZE 32
#define MAX_ELEMS_PER_COL 16
#define MAX_NNZ_OUTPUT 10000000

__device__ static void clear_marker_row(int *m, unsigned int colsB) {
    for (unsigned int j = 0; j < colsB; ++j) {
        m[j] = -1;
    }
}

__global__ static void spmspm_gpu1_kernel(
    unsigned int rowsA,
    const unsigned int *A_rowPtrs,
    const unsigned int *A_colIdxs,
    const float *A_vals,
    const unsigned int *B_rowPtrs,
    const unsigned int *B_colIdxs,
    const float *B_vals,
    unsigned int colsB,
    unsigned int *C_rowIdxs,
    unsigned int *C_colIdxs,
    float *C_vals,
    int *marker,
    unsigned int *d_outputcount
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rowsA) return;

    // shared memory declarations
    __shared__ unsigned int s_row_ind[TILE_SIZE][MAX_ELEMS_PER_COL];
    __shared__ float s_vals[TILE_SIZE][MAX_ELEMS_PER_COL];

    int *m = marker + i * colsB;
    clear_marker_row(m, colsB);

    for (unsigned int pa = A_rowPtrs[i]; pa < A_rowPtrs[i + 1]; ++pa) {
        unsigned int k = A_colIdxs[pa];
        float vA = A_vals[pa];

        unsigned int bStart = B_rowPtrs[k];
        unsigned int bEnd = B_rowPtrs[k + 1];
        unsigned int len = bEnd - bStart;

        for (unsigned int pb = threadIdx.y; pb < len; pb += blockDim.y) {
            if (pb < MAX_ELEMS_PER_COL) {
                s_row_ind[threadIdx.x][pb] = B_colIdxs[bStart + pb];
                s_vals[threadIdx.x][pb] = B_vals[bStart + pb];
            }
        }
        __syncthreads();

        for (unsigned int pb = 0; pb < len && pb < MAX_ELEMS_PER_COL; ++pb) {
            unsigned int j = s_row_ind[threadIdx.x][pb];
            float prod = vA * s_vals[threadIdx.x][pb];
            if (m[j] == -1) {
                unsigned int index = atomicAdd(d_outputcount, 1u);
                if (index < MAX_NNZ_OUTPUT) {
                    C_rowIdxs[index] = i;
                    C_colIdxs[index] = j;
                    C_vals[index] = prod;
                    m[j] = index;
                }
            } else {
                C_vals[m[j]] += prod;
            }
        }
        __syncthreads();
    }
}

void spmspm_gpu1(
    COOMatrix* cooMatrix1,
    CSRMatrix* csrMatrix1,
    CSCMatrix* cscMatrix1,
    COOMatrix* cooMatrix2,
    CSRMatrix* csrMatrix2,
    CSCMatrix* cscMatrix2,
    COOMatrix* cooMatrix3,
    unsigned int numRows1,
    unsigned int numRows2,
    unsigned int numCols2,
    unsigned int numNonzeros1,
    unsigned int numNonzeros2
) {
    CSRMatrix hA;
    CUDA_ERROR_CHECK(cudaMemcpy(&hA, csrMatrix1, sizeof(hA), cudaMemcpyDeviceToHost));
    const unsigned int *d_A_rowPtrs = hA.rowPtrs;
    const unsigned int *d_A_colIdxs = hA.colIdxs;
    const float *d_A_vals = hA.values;

    CSRMatrix hB;
    CUDA_ERROR_CHECK(cudaMemcpy(&hB, csrMatrix2, sizeof(hB), cudaMemcpyDeviceToHost));
    const unsigned int *d_B_rowPtrs = hB.rowPtrs;
    const unsigned int *d_B_colIdxs = hB.colIdxs;
    const float *d_B_vals = hB.values;

    COOMatrix hC;
    CUDA_ERROR_CHECK(cudaMemcpy(&hC, cooMatrix3, sizeof(hC), cudaMemcpyDeviceToHost));
    unsigned int *d_C_rowIdxs = hC.rowIdxs;
    unsigned int *d_C_colIdxs = hC.colIdxs;
    float *d_C_vals = hC.values;

    int *d_marker;
    unsigned int *d_outputcountt;
    CUDA_ERROR_CHECK(cudaMalloc(&d_marker, numRows1 * numCols2 * sizeof(int)));
    CUDA_ERROR_CHECK(cudaMalloc(&d_outputcountt, sizeof(unsigned int)));
    CUDA_ERROR_CHECK(cudaMemset(d_outputcountt, 0, sizeof(unsigned int)));

    const int threads = TILE_SIZE;
    int blocks = (numRows1 + threads - 1) / threads;

    spmspm_gpu1_kernel<<<blocks, threads>>>(
        numRows1,
        d_A_rowPtrs, d_A_colIdxs, d_A_vals,
        d_B_rowPtrs, d_B_colIdxs, d_B_vals,
        numCols2,
        d_C_rowIdxs, d_C_colIdxs, d_C_vals,
        d_marker,
        d_outputcountt
    );
    CUDA_ERROR_CHECK(cudaGetLastError());
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());

    unsigned int newCount;
    CUDA_ERROR_CHECK(cudaMemcpy(&newCount, d_outputcountt, sizeof(newCount), cudaMemcpyDeviceToHost));
    CUDA_ERROR_CHECK(cudaMemcpy(
        (char*)cooMatrix3 + offsetof(COOMatrix, numNonzeros),
        &newCount,
        sizeof(newCount),
        cudaMemcpyHostToDevice
    ));

    CUDA_ERROR_CHECK(cudaFree(d_marker));
    CUDA_ERROR_CHECK(cudaFree(d_outputcountt));
}
