
//@HEADER
// ***************************************************
//
// HPCG: High Performance Conjugate Gradient Benchmark
//
// Contact:
// Michael A. Heroux ( maherou@sandia.gov)
// Jack Dongarra     (dongarra@eecs.utk.edu)
// Piotr Luszczek    (luszczek@eecs.utk.edu)
//
// ***************************************************
//@HEADER

/*!
 @file ComputeSYMGS_ref.cpp

 HPCG routine
 */

#ifndef HPCG_NO_MPI
#include "ExchangeHalo.hpp"
#endif
#include "ComputeSYMGS_ref.hpp"
#include <cassert>

/*!
  Computes one step of symmetric Gauss-Seidel:

  Assumption about the structure of matrix A:
  - Each row 'i' of the matrix has nonzero diagonal value whose address is matrixDiagonal[i]
  - Entries in row 'i' are ordered such that:
       - lower triangular terms are stored before the diagonal element.
       - upper triangular terms are stored after the diagonal element.
       - No other assumptions are made about entry ordering.

  Symmetric Gauss-Seidel notes:
  - We use the input vector x as the RHS and start with an initial guess for y of all zeros.
  - We perform one forward sweep.  x should be initially zero on the first GS sweep, but we do not attempt to exploit this fact.
  - We then perform one back sweep.
  - For simplicity we include the diagonal contribution in the for-j loop, then correct the sum after

  @param[in] A the known system matrix
  @param[in] r the input vector
  @param[inout] x On entry, x should contain relevant values, on exit x contains the result of one symmetric GS sweep with r as the RHS.


  @warning Early versions of this kernel (Version 1.1 and earlier) had the r and x arguments in reverse order, and out of sync with other kernels.

  @return returns 0 upon success and non-zero otherwise

  @see ComputeSYMGS
*/
int ComputeSYMGS_ref( const SparseMatrix & A, const Vector & r, Vector & x) {

  assert(x.localLength==A.localNumberOfColumns); // Make sure x contain space for halo values

#ifndef HPCG_NO_MPI
  ExchangeHalo(A,x, 0);
#endif

  const local_int_t nrow = A.localNumberOfRows;
  double ** matrixDiagonal = A.matrixDiagonal;  // An array of pointers to the diagonal entries A.matrixValues
  const double * const rv = r.values;
  double * const xv = x.values;

  for (local_int_t i=0; i< nrow; i++) {
    const double * const currentValues = A.matrixValues[i];
    const local_int_t * const currentColIndices = A.mtxIndL[i];
    const int currentNumberOfNonzeros = A.nonzerosInRow[i];
    const double  currentDiagonal = matrixDiagonal[i][0]; // Current diagonal value
    double sum = rv[i]; // RHS value

    for (local_int_t j=0; j<currentNumberOfNonzeros; ) {

      size_t givenVectorLength = __riscv_vsetvl_e64m1(currentNumberOfNonzeros - j);

      vfloat64m1_t va = __riscv_vle64_v_f64m1( currentValues+ j, givenVectorLength);
      vfloat64m1_t v_x;

      vuint64m1_t v_idx_col =__riscv_vle64_v_u64m1((const uint64_t *)( currentColIndices + j), givenVectorLength);

      v_idx_col = __riscv_vsll_vx_u64m1(v_idx_col, 3, givenVectorLength);
      v_x = __riscv_vluxei64_v_f64m1(xv, v_idx_col, givenVectorLength);
      vfloat64m1_t vprod =__riscv_vfmul_vv_f64m1(va, v_x, givenVectorLength);
      vfloat64m1_t partial_res =__riscv_vfmv_v_f_f64m1(0.0, givenVectorLength);
      partial_res = __riscv_vfredosum_vs_f64m1_f64m1(vprod, partial_res, givenVectorLength);
      sum -= __riscv_vfmv_f_s_f64m1_f64(partial_res);
      j += givenVectorLength;
    }
    xv[i] = (sum+(xv[i]*currentDiagonal))/currentDiagonal;
  } 

  // Now the back sweep.
  for (local_int_t i = nrow; i-- > 0; ) {
    const double * const currentValues = A.matrixValues[i];
    const local_int_t * const currentColIndices = A.mtxIndL[i];
    const int currentNumberOfNonzeros = A.nonzerosInRow[i];
    const double  currentDiagonal = matrixDiagonal[i][0]; // Current diagonal value
    double sum = rv[i]; // RHS value
    
        for (local_int_t j=0; j<currentNumberOfNonzeros; ) {

      size_t givenVectorLength = __riscv_vsetvl_e64m1(currentNumberOfNonzeros - j);

      vfloat64m1_t va = __riscv_vle64_v_f64m1( currentValues+ j, givenVectorLength);
      vfloat64m1_t v_x;

      vuint64m1_t v_idx_col =__riscv_vle64_v_u64m1((const uint64_t *)( currentColIndices + j), givenVectorLength);

      v_idx_col = __riscv_vsll_vx_u64m1(v_idx_col, 3, givenVectorLength);
      v_x = __riscv_vluxei64_v_f64m1(xv, v_idx_col, givenVectorLength);
      vfloat64m1_t vprod =__riscv_vfmul_vv_f64m1(va, v_x, givenVectorLength);
      vfloat64m1_t partial_res =__riscv_vfmv_v_f_f64m1(0.0, givenVectorLength);
      partial_res = __riscv_vfredosum_vs_f64m1_f64m1(vprod, partial_res, givenVectorLength);
      sum -= __riscv_vfmv_f_s_f64m1_f64(partial_res);
      j += givenVectorLength;
    }
    xv[i] = (sum+(xv[i]*currentDiagonal))/currentDiagonal;
  }

  return 0;
}

