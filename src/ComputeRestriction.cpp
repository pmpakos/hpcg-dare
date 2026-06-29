
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
 @file ComputeRestriction_ref.cpp

 HPCG routine
 */


#ifndef HPCG_NO_OPENMP
#include <omp.h>
#endif

#include "ComputeRestriction_ref.hpp"

/*!
  Routine to compute the coarse residual vector.

  @param[inout]  A - Sparse matrix object containing pointers to mgData->Axf, the fine grid matrix-vector product and mgData->rc the coarse residual vector.
  @param[in]    rf - Fine grid RHS.


  Note that the fine grid residual is never explicitly constructed.
  We only compute it for the fine grid points that will be injected into corresponding coarse grid points.

  @return Returns zero on success and a non-zero value otherwise.
*/
/*int ComputeRestriction_ref(const SparseMatrix & A, const Vector & rf) {

  double * Axfv = A.mgData->Axf->values;
  double * rfv = rf.values;
  double * rcv = A.mgData->rc->values;
  local_int_t * f2c = A.mgData->f2cOperator;
  local_int_t nc = A.mgData->rc->localLength;

#ifndef HPCG_NO_OPENMP
#pragma omp parallel for
#endif
  for (local_int_t i=0; i<nc; ++i) rcv[i] = rfv[f2c[i]] - Axfv[f2c[i]];

  return 0;
}*/
#include <cassert>
#include <riscv_vector.h>
#include <omp.h>
int ComputeRestriction_ref(const SparseMatrix & A, const Vector & rf) {

  //isOptimized = true;

  assert(A.mgData != 0);

  const double * const Axfv = A.mgData->Axf->values;
  const double * const rfv = rf.values;
  double * const rcv = A.mgData->rc->values;
  const local_int_t * const f2c = A.mgData->f2cOperator;
  const local_int_t nc = A.mgData->rc->localLength;

#ifndef HPCG_NO_OPENMP
#pragma omp parallel
  {
    int tid = omp_get_thread_num();
    int nthreads = omp_get_num_threads();

    local_int_t chunk = nc / nthreads;
    local_int_t start = tid * chunk;
    local_int_t end = (tid == nthreads - 1) ? nc : start + chunk;

    for (local_int_t i = start; i < end; ) {
      size_t vl = __riscv_vsetvl_e64m1(end - i);

      vuint64m1_t idx64;

#if defined(HPCG_INDEX_64)
      idx64 = __riscv_vle64_v_u64m1((const uint64_t *)(f2c + i), vl);
#else
      vuint32mf2_t idx32 = __riscv_vle32_v_u32mf2((const uint32_t *)(f2c + i), vl);
      idx64 = __riscv_vzext_vf2_u64m1(idx32, vl);
#endif

      idx64 = __riscv_vsll_vx_u64m1(idx64, 3, vl);

      vfloat64m1_t vrf = __riscv_vluxei64_v_f64m1(rfv, idx64, vl);
      vfloat64m1_t vaxf = __riscv_vluxei64_v_f64m1(Axfv, idx64, vl);
      vfloat64m1_t vrc = __riscv_vfsub_vv_f64m1(vrf, vaxf, vl);

      __riscv_vse64_v_f64m1(rcv + i, vrc, vl);

      i += vl;
    }
  }
#else
  for (local_int_t i = 0; i < nc; i++) {
    rcv[i] = rfv[f2c[i]] - Axfv[f2c[i]];
  }
#endif

  return 0;
}
