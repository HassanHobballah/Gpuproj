#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <cuda_runtime.h>
#include "common.h"
#include "matrix.h"

#define TILE_SIZE 32
#define MAX_ELEMS_PER_COL 16
#define MAX_NNZ_OUTPUT 10000000u
#define COARSEN 2 

// Privatized SpMSpM, per-thread output buffer in registers
__global__ void spmspm_gpu_privatized(
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
    unsigned int *d_outputcount
) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int startRow = tid * COARSEN;
    if (startRow >= rowsA) return;

    // output buffer per-thread (in registers)
    unsigned int priv_cols[MAX_ELEMS_PER_COL];
    float priv_vals[MAX_ELEMS_PER_COL];
    unsigned int priv_count;

    for (int r = 0; r < COARSEN; ++r) {
        unsigned int i = startRow + r;
        if (i >= rowsA) break;

        // reset private buffer
        priv_count = 0;

        unsigned int rowStart = A_rowPtrs[i];
        unsigned int rowEnd   = A_rowPtrs[i+1];

        // accumulate products into private buffer
        for (unsigned int pa = rowStart; pa < rowEnd; ++pa) {
            float vA = A_vals[pa];
            unsigned int k = A_colIdxs[pa];

            unsigned int b0 = B_rowPtrs[k];
            unsigned int b1 = B_rowPtrs[k+1];
            unsigned int len = b1 - b0;
            unsigned int limit = (len < MAX_ELEMS_PER_COL ? len : MAX_ELEMS_PER_COL);

            for (unsigned int pb = 0; pb < limit; ++pb) {
                unsigned int j = B_colIdxs[b0 + pb];
                float prod = vA * B_vals[b0 + pb];

                // search in private buffer
                unsigned int index = priv_count;
                bool found = false;
                for (unsigned int t = 0; t < priv_count; ++t) {
                    if (priv_cols[t] == j) { index = t; found = true; break; }
                }
                if (!found) {
                    priv_cols[priv_count] = j;
                    priv_vals[priv_count] = prod;
                    priv_count++;
                } else {
                    priv_vals[index] += prod;
                }
            }
        }

        // flush private buffer to global arrays
        unsigned int base = atomicAdd(d_outputcount, priv_count);
        for (unsigned int t = 0; t < priv_count; ++t) {
            unsigned int dst = base + t;
            C_rowIdxs[dst] = i;
            C_colIdxs[dst] = priv_cols[t];
            C_vals   [dst] = priv_vals[t];
        }
    }
}

void spmspm_gpu3(
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
    CSRMatrix hA; CUDA_ERROR_CHECK(cudaMemcpy(&hA, csrMatrix1, sizeof(hA), cudaMemcpyDeviceToHost));
    const unsigned int *d_A_rowPtrs = hA.rowPtrs;
    const unsigned int *d_A_colIdxs = hA.colIdxs;
    const float        *d_A_vals    = hA.values;
    CSRMatrix hB; CUDA_ERROR_CHECK(cudaMemcpy(&hB, csrMatrix2, sizeof(hB), cudaMemcpyDeviceToHost));
    const unsigned int *d_B_rowPtrs = hB.rowPtrs;
    const unsigned int *d_B_colIdxs = hB.colIdxs;
    const float        *d_B_vals    = hB.values;
    COOMatrix hC; CUDA_ERROR_CHECK(cudaMemcpy(&hC, cooMatrix3, sizeof(hC), cudaMemcpyDeviceToHost));
    unsigned int *d_C_rowIdxs = hC.rowIdxs;
    unsigned int *d_C_colIdxs = hC.colIdxs;
    float        *d_C_vals    = hC.values;

    unsigned int *d_outputcountt;
    CUDA_ERROR_CHECK(cudaMalloc(&d_outputcountt, sizeof(unsigned int)));
    CUDA_ERROR_CHECK(cudaMemset(d_outputcountt, 0, sizeof(unsigned int)));

    int threads = TILE_SIZE;
    int blocks  = (numRows1 + threads * COARSEN - 1) / (threads * COARSEN);
    spmspm_gpu_privatized<<<blocks, threads>>>(
        numRows1,
        d_A_rowPtrs, d_A_colIdxs, d_A_vals,
        d_B_rowPtrs, d_B_colIdxs, d_B_vals,
        numCols2,
        d_C_rowIdxs, d_C_colIdxs, d_C_vals,
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

    CUDA_ERROR_CHECK(cudaFree(d_outputcountt));
}
