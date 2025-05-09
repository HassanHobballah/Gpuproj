#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <cuda_runtime.h>
#include "common.h"
#include "matrix.h"

// clear one row marker
__device__ static void clear_marker_row(int *marker, unsigned int colsB) {
    for (unsigned int j = 0; j < colsB; ++j) {
        marker[j] = -1;
    }
}

// multiply A’s row i by B and write into C
__device__ static void multiply_and_scatter(
    unsigned int    i,
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
    // Iterate over each nonzero in row of A
    for (unsigned int pa = A_rowPtrs[i]; pa < A_rowPtrs[i + 1]; ++pa) {
        unsigned int k = A_colIdxs[pa];
        float vA = A_vals[pa];

        // For each nonzero in row of B
        for (unsigned int pb = B_rowPtrs[k]; pb < B_rowPtrs[k + 1]; ++pb) {
            unsigned int j = B_colIdxs[pb];
            float prod = vA * B_vals[pb];

            if (marker[j] == -1) {
                // Reserve in C via atomic add
                unsigned int index = atomicAdd(d_outputcount, 1u);
                C_rowIdxs[index] = i;
                C_colIdxs[index] = j;
                C_vals[index]    = prod;
                marker[j] = index;
            } else {
                // put into existing COO slot
                C_vals[marker[j]] += prod;
            }
        }
    }
}

__global__ static void spmspm_gpu0_kernel(
    unsigned int rowsA,
    const unsigned int *A_rowPtrs,
    const unsigned int *A_colIdxs,
    const float        *A_vals,
    const unsigned int *B_rowPtrs,
    const unsigned int *B_colIdxs,
    const float        *B_vals,
    unsigned int colsB,
    unsigned int *C_rowIdxs,
    unsigned int *C_colIdxs,
    float        *C_vals,
    int          *marker,
    unsigned int *d_outputcount
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rowsA) return;

    // Each thread has a slice of marker array of length colsB
    int *marker_row = marker + i * colsB;

    // Reset markers for this row
    clear_marker_row(marker_row, colsB);

    // Scatter into C for this row
    multiply_and_scatter(
        i,
        A_rowPtrs, A_colIdxs, A_vals,
        B_rowPtrs, B_colIdxs, B_vals,
        colsB,
        C_rowIdxs, C_colIdxs, C_vals,
        marker_row,
        d_outputcount
    );
}

void spmspm_gpu0(
    COOMatrix *cooMatrix1,
    CSRMatrix *csrMatrix1,
    CSCMatrix *cscMatrix1,
    COOMatrix *cooMatrix2,
    CSRMatrix *csrMatrix2,
    CSCMatrix *cscMatrix2,
    COOMatrix *cooMatrix3,
    unsigned int numRows1,
    unsigned int numRows2,
    unsigned int numCols2,
    unsigned int numNonzeros1,
    unsigned int numNonzeros2
) {
    // Unpack device pointers for CSR(A)
    CSRMatrix hA;
    CUDA_ERROR_CHECK(cudaMemcpy(&hA, csrMatrix1, sizeof(hA), cudaMemcpyDeviceToHost));
    const unsigned int *d_A_rowPtrs = hA.rowPtrs;
    const unsigned int *d_A_colIdxs = hA.colIdxs;
    const float        *d_A_vals    = hA.values;

    // Unpack device pointers for CSR(B)
    CSRMatrix hB;
    CUDA_ERROR_CHECK(cudaMemcpy(&hB, csrMatrix2, sizeof(hB), cudaMemcpyDeviceToHost));
    const unsigned int *d_B_rowPtrs = hB.rowPtrs;
    const unsigned int *d_B_colIdxs = hB.colIdxs;
    const float        *d_B_vals    = hB.values;

    // Unpack device pointers for COO(C)
    COOMatrix hC;
    CUDA_ERROR_CHECK(cudaMemcpy(&hC, cooMatrix3, sizeof(hC), cudaMemcpyDeviceToHost));
    unsigned int *d_C_rowIdxs = hC.rowIdxs;
    unsigned int *d_C_colIdxs = hC.colIdxs;
    float        *d_C_vals    = hC.values;

    // Allocate marker array and output counter on device
    int           *d_marker;
    unsigned int  *d_outputcount;
    CUDA_ERROR_CHECK(cudaMalloc(&d_marker,        numRows1 * numCols2 * sizeof(int)));
    CUDA_ERROR_CHECK(cudaMalloc(&d_outputcount,  sizeof(unsigned int)));
    CUDA_ERROR_CHECK(cudaMemset(d_outputcount, 0, sizeof(unsigned int)));

    // one thread per row of A
    const int threads = 512;
    int blocks = (numRows1 + threads - 1) / threads;
    spmspm_gpu0_kernel<<<blocks, threads>>>(
        numRows1,
        d_A_rowPtrs, d_A_colIdxs, d_A_vals,
        d_B_rowPtrs, d_B_colIdxs, d_B_vals,
        numCols2,
        d_C_rowIdxs, d_C_colIdxs, d_C_vals,
        d_marker,
        d_outputcount
    );
    CUDA_ERROR_CHECK(cudaGetLastError());
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());

    // Retrieve final count of COO entries and update struct
    unsigned int newCount;
    CUDA_ERROR_CHECK(cudaMemcpy(&newCount, d_outputcount, sizeof(newCount), cudaMemcpyDeviceToHost));
    CUDA_ERROR_CHECK(cudaMemcpy(
        (char*)cooMatrix3 + offsetof(COOMatrix, numNonzeros),
        &newCount,
        sizeof(newCount),
        cudaMemcpyHostToDevice
    ));

    // Free
    CUDA_ERROR_CHECK(cudaFree(d_marker));
    CUDA_ERROR_CHECK(cudaFree(d_outputcount));
}