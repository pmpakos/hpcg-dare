
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
 @file ComputeSPMV.cpp

 HPCG routine
 */

#include "ComputeSPMV.hpp"
#include "ComputeSPMV_ref.hpp"
#include <stdio.h>
#include <riscv_vector.h>
#include <omp.h>
#include <assert.h>

/*!
  Routine to compute sparse matrix vector product y = Ax where:
  Precondition: First call exchange_externals to get off-processor values of x

  This routine calls the reference SpMV implementation by default, but
  can be replaced by a custom, optimized routine suited for
  the target system.

  @param[in]  A the known system matrix
  @param[in]  x the known vector
  @param[out] y the On exit contains the result: Ax.

  @return returns 0 upon success and non-zero otherwise

  @see ComputeSPMV_ref
*/

int ComputeSPMV(const SparseMatrix &A, Vector &x, Vector &y) {

  A.isSpmvOptimized = true;
  assert(x.localLength >= A.localNumberOfColumns);
  assert(y.localLength >= A.localNumberOfRows);

#ifndef HPCG_NO_MPI
  ExchangeHalo(A, x, 0);
#endif

  const double * const xv = x.values;
  double * const yv = y.values;
  const local_int_t nrow = A.localNumberOfRows;
  
  #ifndef HPCG_NO_OPENMP
  #pragma omp parallel for schedule(static)
  #endif
  
  for (local_int_t i=0; i<nrow; i++) {
    double sum = 0.0;
    const double *cur_vals = A.matrixValues[i];
    const local_int_t *cur_inds = A.mtxIndL[i];
    const int cur_nnz = A.nonzerosInRow[i];

    for (local_int_t colid=0; colid<cur_nnz;) {

      size_t givenVectorLength = __riscv_vsetvl_e64m1(cur_nnz - colid);

      vfloat64m1_t va = __riscv_vle64_v_f64m1(cur_vals + colid, givenVectorLength);
      vfloat64m1_t v_x;

      vuint64m1_t v_idx_col =__riscv_vle64_v_u64m1((const uint64_t *)(cur_inds + colid), givenVectorLength);

      v_idx_col = __riscv_vsll_vx_u64m1(v_idx_col, 3, givenVectorLength);
      v_x = __riscv_vluxei64_v_f64m1(xv, v_idx_col, givenVectorLength);
      vfloat64m1_t vprod =__riscv_vfmul_vv_f64m1(va, v_x, givenVectorLength);
      vfloat64m1_t partial_res =__riscv_vfmv_v_f_f64m1(0.0, givenVectorLength);
      partial_res = __riscv_vfredosum_vs_f64m1_f64m1(vprod, partial_res, givenVectorLength);
      sum += __riscv_vfmv_f_s_f64m1_f64(partial_res);
      colid += givenVectorLength;
    }

    yv[i] = sum;

  }

  return 0;
}
