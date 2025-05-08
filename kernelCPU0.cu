#include "common.h"
#include <stdlib.h>
#include <stdio.h>

// basic sequential SpMM: C = A × B
void spmspm_cpu0(
    COOMatrix* cooMatrix1,
    CSRMatrix* csrMatrix1,
    CSCMatrix* cscMatrix1,
    COOMatrix* cooMatrix2,
    CSRMatrix* csrMatrix2,
    CSCMatrix* cscMatrix2,
    COOMatrix* cooMatrix3
) {
    unsigned int numRows = csrMatrix1->numRows;
    unsigned int numCols = csrMatrix2->numCols;

    // allocate temporary storage
    float*        rowAcc  = (float*)calloc(numCols, sizeof(float));
    char*         flag    = (char*)calloc(numCols, sizeof(char));
    unsigned int* touched = (unsigned int*)malloc(numCols * sizeof(unsigned int));
    if (!rowAcc || !flag || !touched) {
        fprintf(stderr, "Memory allocation failed\n");
        free(rowAcc); free(flag); free(touched);
        return;
    }

    for (unsigned int i = 0; i < numRows; ++i) {
        unsigned int count = 0;
        unsigned int aStart = csrMatrix1->rowPtrs[i];
        unsigned int aEnd   = csrMatrix1->rowPtrs[i + 1];

        for (unsigned int i_of_a = aStart; i_of_a < aEnd; ++i_of_a) {
            unsigned int j    = csrMatrix1->colIdxs[i_of_a];
            float        valA = csrMatrix1->values[i_of_a];

            unsigned int bStart = csrMatrix2->rowPtrs[j];
            unsigned int bEnd   = csrMatrix2->rowPtrs[j + 1];
            for (unsigned int i_of_b = bStart; i_of_b < bEnd; ++i_of_b) {
                unsigned int k    = csrMatrix2->colIdxs[i_of_b];
                float        valB = csrMatrix2->values[i_of_b];

                if (!flag[k]) {
                    flag[k] = 1;
                    touched[count++] = k;
                }
                rowAcc[k] += valA * valB;
            }
        }

        for (unsigned int t = 0; t < count; ++t) {
            unsigned int k = touched[t];
            unsigned int idxC = cooMatrix3->numNonzeros++;
            cooMatrix3->rowIdxs[idxC] = i;
            cooMatrix3->colIdxs[idxC] = k;
            cooMatrix3->values[idxC]  = rowAcc[k];

            flag[k]    = 0;
            rowAcc[k]  = 0.0f;
        }
    }

    free(rowAcc);
    free(flag);
    free(touched);
}
