#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>

#include <riscv_vector.h> 

void f32_gemm(
  size_t M, size_t N, size_t K,
  const float* A, size_t a_row_stride,
  const float* B,
  float* C, size_t c_row_stride,
  size_t c_col_stride)
{
  const size_t kc_bytes = K * sizeof(float);
  const size_t a_stride_bytes = a_row_stride * sizeof(float);
  const size_t cm_stride_bytes = c_row_stride * sizeof(float);
  const size_t cn_stride_bytes = c_col_stride * sizeof(float);

  size_t row = 0;
  while (row < M) {
      size_t rows_left = M - row;

      if (rows_left >= 7) {
          xnn_f32_gemm_ukernel_7x4v__rvv(
              7,
              N,
              kc_bytes,
              A + row * a_row_stride,
              a_stride_bytes,
              B,
              C + row * c_row_stride,
              cm_stride_bytes,
              cn_stride_bytes
          );
          row += 7;
      } else {
          xnn_f32_gemm_ukernel_1x4v__rvv(
              1,
              N,
              kc_bytes,
              A + row * a_row_stride,
              a_stride_bytes,
              B,
              C + row * c_row_stride,
              cm_stride_bytes,
              cn_stride_bytes
          );
          row += 1;
      }
  }
}

void xnn_f32_gemm_ukernel_7x4v__rvv(
    size_t mr,
    size_t nc,
    size_t kc,
    const float* a,
    size_t a_stride,
    const float* w,
    float* c,
    size_t cm_stride,
    size_t cn_stride)
{

  const float* a0 = a;
  float* c0 = c;
  const float* a1 = (const float*) ((uintptr_t) a0 + a_stride);
  float* c1 = (float*) ((uintptr_t) c0 + cm_stride);

  const float* a2 = (const float*) ((uintptr_t) a1 + a_stride);
  float* c2 = (float*) ((uintptr_t) c1 + cm_stride);

  const float* a3 = (const float*) ((uintptr_t) a2 + a_stride);
  float* c3 = (float*) ((uintptr_t) c2 + cm_stride);

  const float* a4 = (const float*) ((uintptr_t) a3 + a_stride);
  float* c4 = (float*) ((uintptr_t) c3 + cm_stride);

  const float* a5 = (const float*) ((uintptr_t) a4 + a_stride);
  float* c5 = (float*) ((uintptr_t) c4 + cm_stride);

  const float* a6 = (const float*) ((uintptr_t) a5 + a_stride);
  float* c6 = (float*) ((uintptr_t) c5 + cm_stride);

  const size_t nr = nc;
  size_t vl = nr;
  const float* w_new = w;
  do {
    vl = __riscv_vsetvl_e32m4(nc);
    nc = nc - vl;
    w_new = w + vl;

    register vfloat32m4_t vacc0 =  __riscv_vle32_v_f32m4(w, vl);

    w = w + nr;
    register vfloat32m4_t vacc1 =  __riscv_vmv_v_v_f32m4(vacc0, vl);
    register vfloat32m4_t vacc2 =  __riscv_vmv_v_v_f32m4(vacc0, vl);
    register vfloat32m4_t vacc3 =  __riscv_vmv_v_v_f32m4(vacc0, vl);
    register vfloat32m4_t vacc4 =  __riscv_vmv_v_v_f32m4(vacc0, vl);
    register vfloat32m4_t vacc5 =  __riscv_vmv_v_v_f32m4(vacc0, vl);
    register vfloat32m4_t vacc6 =  __riscv_vmv_v_v_f32m4(vacc0, vl);

    size_t k = kc;
    do {
      const float va0 = *a0++;
      const float va1 = *a1++;
      const float va2 = *a2++;
      const float va3 = *a3++;
      const float va4 = *a4++;
      const float va5 = *a5++;
      const float va6 = *a6++;
      register vfloat32m4_t vb = __riscv_vle32_v_f32m4(w, vl);

      w = w + nr;
      vacc0 = __riscv_vfmacc_vf_f32m4(vacc0, va0, vb, vl);
      vacc1 = __riscv_vfmacc_vf_f32m4(vacc1, va1, vb, vl);
      vacc2 = __riscv_vfmacc_vf_f32m4(vacc2, va2, vb, vl);
      vacc3 = __riscv_vfmacc_vf_f32m4(vacc3, va3, vb, vl);
      vacc4 = __riscv_vfmacc_vf_f32m4(vacc4, va4, vb, vl);
      vacc5 = __riscv_vfmacc_vf_f32m4(vacc5, va5, vb, vl);
      vacc6 = __riscv_vfmacc_vf_f32m4(vacc6, va6, vb, vl);
      k -= sizeof(float);
    } while (k != 0);
    // store 7 x vl results to c
    __riscv_vse32_v_f32m4(c0, vacc0, vl);
    c0 = c0 + vl;
    __riscv_vse32_v_f32m4(c1, vacc1, vl);
    c1 = c1 + vl;
    __riscv_vse32_v_f32m4(c2, vacc2, vl);
    c2 = c2 + vl;
    __riscv_vse32_v_f32m4(c3, vacc3, vl);
    c3 = c3 + vl;
    __riscv_vse32_v_f32m4(c4, vacc4, vl);
    c4 = c4 + vl;
    __riscv_vse32_v_f32m4(c5, vacc5, vl);
    c5 = c5 + vl;
    __riscv_vse32_v_f32m4(c6, vacc6, vl);
    c6 = c6 + vl;
    a0 = (const float*) ((uintptr_t) a0 - kc);
    a1 = (const float*) ((uintptr_t) a1 - kc);
    a2 = (const float*) ((uintptr_t) a2 - kc);
    a3 = (const float*) ((uintptr_t) a3 - kc);
    a4 = (const float*) ((uintptr_t) a4 - kc);
    a5 = (const float*) ((uintptr_t) a5 - kc);
    a6 = (const float*) ((uintptr_t) a6 - kc);
    w = w_new;

  } while (nc != 0);
}

void xnn_f32_gemm_ukernel_1x4v__rvv(
  size_t mr,
  size_t nc,
  size_t kc,
  const float* a,
  size_t a_stride,
  const float* w,
  float* restrict c,
  size_t cm_stride,
  size_t cn_stride)
{

const float* a0 = a;
float* c0 = c;

const size_t nr = nc;
size_t vl = nr;
const float* w_new = w;
do {
  vl = __riscv_vsetvl_e32m4(nc);
  nc = nc - vl;
  w_new = w + vl;

  vfloat32m4_t vacc0 =  __riscv_vle32_v_f32m4(w, vl);

  w = w + nr;

  size_t k = kc;
  do {
    const float va0 = *a0++;
    vfloat32m4_t vb = __riscv_vle32_v_f32m4(w, vl);
    w = w + nr;
    vacc0 = __riscv_vfmacc_vf_f32m4(vacc0, va0, vb, vl);
    k -= sizeof(float);
  } while (k != 0);
  // store 1 x vl results to c
  __riscv_vse32_v_f32m4(c0, vacc0, vl);

  c0 = c0 + vl;
  a0 = (const float*) ((uintptr_t) a0 - kc);
  w = w_new;
} while (nc != 0);
}

void pack_weight_matrix(
  size_t K, 
  size_t N, 
  const float* B,
  float* B_packed)
{
  size_t nr = __riscv_vsetvlmax_e32m4();
  if (N < nr) nr = N;
  int i = 0;

  for (size_t nc = 0; nc < N; nc += nr){
    // printf("nr: %d \n", nr);
    if (N - nc < nr) {
      nr = N - nc;
    }
    
    for (size_t row = 0; row < K + 1; row++) {
      for (size_t c = 0; c < nr; c++) { 
        // if (i < 200) printf("row: %d, c: %d, nc+c:%d, index: %d, i: %d, value loaded: %d \n", row, c, nc+c, row * N + nc + c, i, (int) B[row * N + nc + c]);
        B_packed[i] = B[row * N + nc + c];
        i ++;
      }
    }
  }
}

void xnn_f32_gemm_ukernel_7x4v__rvv_packed(
  size_t mr,
  size_t nc,
  size_t kc,
  const float* a,
  size_t a_stride,
  const float* w,
  float* c,
  size_t cm_stride,
  size_t cn_stride)
{

const float* a0 = a;
float* c0 = c;
const float* a1 = (const float*) ((uintptr_t) a0 + a_stride);
float* c1 = (float*) ((uintptr_t) c0 + cm_stride);

const float* a2 = (const float*) ((uintptr_t) a1 + a_stride);
float* c2 = (float*) ((uintptr_t) c1 + cm_stride);

const float* a3 = (const float*) ((uintptr_t) a2 + a_stride);
float* c3 = (float*) ((uintptr_t) c2 + cm_stride);

const float* a4 = (const float*) ((uintptr_t) a3 + a_stride);
float* c4 = (float*) ((uintptr_t) c3 + cm_stride);

const float* a5 = (const float*) ((uintptr_t) a4 + a_stride);
float* c5 = (float*) ((uintptr_t) c4 + cm_stride);

const float* a6 = (const float*) ((uintptr_t) a5 + a_stride);
float* c6 = (float*) ((uintptr_t) c5 + cm_stride);


size_t nr = __riscv_vsetvlmax_e32m4();
nr = (nc > nr) ? nr : nc;
size_t vl = nr;
do {
  if (nc < nr) {
    vl = __riscv_vsetvl_e32m4(nc);
    nr = nc;
  }
  nc = nc - vl;

  vfloat32m4_t vacc0 =  __riscv_vle32_v_f32m4(w, vl);
  w = w + nr;
  vfloat32m4_t vacc1 =  __riscv_vmv_v_v_f32m4(vacc0, vl);
  vfloat32m4_t vacc2 =  __riscv_vmv_v_v_f32m4(vacc0, vl);
  vfloat32m4_t vacc3 =  __riscv_vmv_v_v_f32m4(vacc0, vl);
  vfloat32m4_t vacc4 =  __riscv_vmv_v_v_f32m4(vacc0, vl);
  vfloat32m4_t vacc5 =  __riscv_vmv_v_v_f32m4(vacc0, vl);
  vfloat32m4_t vacc6 =  __riscv_vmv_v_v_f32m4(vacc0, vl);

  size_t k = kc;
  do {
    const float va0 = *a0++;
    const float va1 = *a1++;
    const float va2 = *a2++;
    const float va3 = *a3++;
    const float va4 = *a4++;
    const float va5 = *a5++;
    const float va6 = *a6++;
    vfloat32m4_t vb = __riscv_vle32_v_f32m4(w, vl);
    w = w + nr;
    vacc0 = __riscv_vfmacc_vf_f32m4(vacc0, va0, vb, vl);
    vacc1 = __riscv_vfmacc_vf_f32m4(vacc1, va1, vb, vl);
    vacc2 = __riscv_vfmacc_vf_f32m4(vacc2, va2, vb, vl);
    vacc3 = __riscv_vfmacc_vf_f32m4(vacc3, va3, vb, vl);
    vacc4 = __riscv_vfmacc_vf_f32m4(vacc4, va4, vb, vl);
    vacc5 = __riscv_vfmacc_vf_f32m4(vacc5, va5, vb, vl);
    vacc6 = __riscv_vfmacc_vf_f32m4(vacc6, va6, vb, vl);
    k -= sizeof(float);
  } while (k != 0);
  // store 7 x vl results to c
  __riscv_vse32_v_f32m4(c0, vacc0, vl);
  c0 += vl;
  __riscv_vse32_v_f32m4(c1, vacc1, vl);
  c1 += vl;
  __riscv_vse32_v_f32m4(c2, vacc2, vl);
  c2 += vl;
  __riscv_vse32_v_f32m4(c3, vacc3, vl);
  c3 += vl;
  __riscv_vse32_v_f32m4(c4, vacc4, vl);
  c4 += vl;
  __riscv_vse32_v_f32m4(c5, vacc5, vl);
  c5 += vl;
  __riscv_vse32_v_f32m4(c6, vacc6, vl);
  c6 += vl;
  a0 = (const float*) ((uintptr_t) a0 - kc);
  a1 = (const float*) ((uintptr_t) a1 - kc);
  a2 = (const float*) ((uintptr_t) a2 - kc);
  a3 = (const float*) ((uintptr_t) a3 - kc);
  a4 = (const float*) ((uintptr_t) a4 - kc);
  a5 = (const float*) ((uintptr_t) a5 - kc);
  a6 = (const float*) ((uintptr_t) a6 - kc);
} while (nc != 0);
}

void xnn_f32_gemm_ukernel_1x4v__rvv_packed(
  size_t mr,
  size_t nc,
  size_t kc,
  const float* a,
  size_t a_stride,
  const float* w,
  float* c,
  size_t cm_stride,
  size_t cn_stride)
{

const float* a0 = a;
float* c0 = c;

size_t nr = __riscv_vsetvlmax_e32m4();
nr = (nc > nr) ? nr : nc;
size_t vl = nr;
do {
  if (nc < nr) {
    vl = __riscv_vsetvl_e32m4(nc);
    nr = nc;
  }
  nc = nc - vl;

  vfloat32m4_t vacc0 =  __riscv_vle32_v_f32m4(w, vl);
  w = w + nr;

  size_t k = kc;
  do {
    const float va0 = *a0++;
    vfloat32m4_t vb = __riscv_vle32_v_f32m4(w, vl);
    w = w + nr;
    vacc0 = __riscv_vfmacc_vf_f32m4(vacc0, va0, vb, vl);
    k -= sizeof(float);
  } while (k != 0);
  // store 1 x vl results to c
  __riscv_vse32_v_f32m4(c0, vacc0, vl);
  c0 += vl;
  a0 = (const float*) ((uintptr_t) a0 - kc);
} while (nc != 0);

}

void f32_gemm_packed(
  size_t M, size_t N, size_t K,
  const float* A, size_t a_row_stride,
  const float* B,
  float* C, size_t c_row_stride,
  size_t c_col_stride)
{
  const size_t kc_bytes = K * sizeof(float);
  const size_t a_stride_bytes = a_row_stride * sizeof(float);
  const size_t cm_stride_bytes = c_row_stride * sizeof(float);
  const size_t cn_stride_bytes = c_col_stride * sizeof(float);

  size_t row = 0;
  while (row < M) {
      size_t rows_left = M - row;

      if (rows_left >= 7) {
          xnn_f32_gemm_ukernel_7x4v__rvv_packed(
              7,
              N,
              kc_bytes,
              A + row * a_row_stride,
              a_stride_bytes,
              B,
              C + row * c_row_stride,
              cm_stride_bytes,
              cn_stride_bytes
          );
          row += 7;
      } else {
          xnn_f32_gemm_ukernel_1x4v__rvv_packed(
              1,
              N,
              kc_bytes,
              A + row * a_row_stride,
              a_stride_bytes,
              B,
              C + row * c_row_stride,
              cm_stride_bytes,
              cn_stride_bytes
          );
          row += 1;
      }
  }
}

void f32_gemm_naive(
  size_t M, size_t N, size_t K,
  const float* A, size_t a_row_stride,
  const float* B,
  float* C, size_t c_row_stride, size_t c_col_stride)
{
  for (size_t i = 0; i < M; i++) {
      for (size_t j = 0; j < N; j++) {
          float sum = B[j];
          for (size_t k = 0; k < K; k++) {
              float a_val = A[i * a_row_stride + k];
              float b_val = B[(k+1) * N + j];
              sum += a_val * b_val;
          }
          C[i * c_row_stride + j * c_col_stride] = sum;
      }
  }
}

int verify_matrix(
  size_t M, size_t N, size_t K,
  const float* A, size_t a_row_stride, 
  const float* B,
  size_t c_row_stride, size_t c_col_stride, 
  float* test, float* ref)
{
  for (size_t i = 0; i < M; i++) {
      for (size_t j = 0; j < N; j++) {
          float ref_val = ref[i * c_row_stride + j * c_col_stride];
          float test_val = test[i * c_row_stride + j * c_col_stride];
          if (ref_val != test_val) {
              printf("Mismatch at (%d, %d): ref = %d, test = %d\n", i, j, ref_val, test_val);
              return 0;
          }
      }
  }
  printf("Test PASSED \n");
  return 1;
}
