#include "hal_ope_prepack.h"
#include "rocc.h"

// ===== High memory but fast OPE Helper Functions =====

// Pack L columns, 8 rows of A into dst
static inline void pack_A_block_i8x8(int8_t *dst, const int8_t *A, int lda,
                                     int M, int i0, int k0, int L)
{
  for (int kk = 0; kk < L; ++kk) {
    for (int r = 0; r < 8; ++r) {
      int ir = i0 + r;
      dst[kk*8 + r] = (ir < M) ? A[ir*lda + (k0 + kk)] : 0;
    }
  }
}
// Pack L rows, 8 columns of B into dst
static inline void pack_B_block_Kx8(int8_t *dst, const int8_t *B, int ldb,
                                    int N, int j0, int k0, int L)
{
  for (int kk = 0; kk < L; ++kk) {
    for (int c = 0; c < 8; ++c) {
      int jc = j0 + c;
      dst[kk*8 + c] = (jc < N) ? B[(k0 + kk)*ldb + jc] : 0;
    }
  }
}

inline void ope_pack_all_A(const int8_t* A, int lda, const ope_pack_plan* p)
{
  for (int it = 0, i0 = 0; it < p->itiles; ++it, i0 += 8) {
    int8_t* dst = p->A_pack + (size_t)it * p->K * 8;
    pack_A_block_i8x8(dst, A, lda, p->M, i0, 0, p->K);
  }
}

inline void ope_pack_all_B(const int8_t* B, int ldb, const ope_pack_plan* p)
{
  for (int jt = 0, j0 = 0; jt < p->jtiles; ++jt, j0 += 8) {
    int8_t* dst = p->B_pack + (size_t)jt * p->K * 8;
    pack_B_block_Kx8(dst, B, ldb, p->N, j0, 0, p->K);
  }
}

static inline void extract_full_buffered_to_C(int32_t* C, int ldc)
{
  int32_t tile[8*8];
  ope_extract((uint64_t)tile, 8, 0, 1);
  for (int r = 0; r < 8; ++r) {
    int32_t* dst = &C[(size_t)r * ldc];
    const int32_t* src = &tile[r * 8];
    dst[0] = src[0]; 
    dst[1] = src[1]; 
    dst[2] = src[2]; 
    dst[3] = src[3];
    dst[4] = src[4]; 
    dst[5] = src[5]; 
    dst[6] = src[6]; 
    dst[7] = src[7];
  }
}

static inline void extract_clamped_to_C(int32_t* C, int ldc, int i_size, int j_size)
{
  int32_t tile[8*8];
  ope_extract((uint64_t)tile, 8, 0, 1);
  for (int r = 0; r < i_size; ++r) {
    int32_t* dst = &C[(size_t)r * ldc];
    const int32_t* src = &tile[r * 8];
    for (int c = 0; c < j_size; ++c) {
      dst[c] = src[c];
    }
  }
}

inline size_t ope_pack_workspace_size(int M, int N, int K)
{
  const int itiles = (M + 7) >> 3;
  const int jtiles = (N + 7) >> 3;
  size_t bytes = (size_t)itiles * K * 8 + (size_t)jtiles * K * 8;
  return (bytes + 7u) & ~7u;
}

inline int ope_pack_plan_init(ope_pack_plan* p,
                                     int M, int N, int K,
                                     void* workspace, size_t workspace_bytes)
{
  const int itiles = (M + 7) >> 3;
  const int jtiles = (N + 7) >> 3;
  const size_t need = (size_t)itiles * K * 8 + (size_t)jtiles * K * 8;
  if (workspace_bytes < ((need + 7u) & ~7u)) {
    return -1;
  }
  p->M = M; 
  p->N = N; 
  p->K = K;
  p->itiles = itiles; 
  p->jtiles = jtiles;

  uint8_t* base = (uint8_t*)workspace;
  p->A_pack = (int8_t*)base;
  p->B_pack = (int8_t*)(base + (size_t)itiles * K * 8);
  return 0;
}

// ===== High memory but fast OPE Driver Function =====

__attribute__((noinline))
void ope_matmul_prepacked(const int8_t* A, const int8_t* B, int32_t* C,
                          int M, int N, int K, int lda, int ldb, int ldc,
                          void* pack_workspace, size_t pack_workspace_bytes,
                          int buffered_full_tiles)
{
  ope_pack_plan plan;
  if (ope_pack_plan_init(&plan, M, N, K, pack_workspace, pack_workspace_bytes) != 0) {
    return ope_matmul(A, B, C, M, N, K, lda, ldb, ldc);
  }

  ope_pack_all_A(A, lda, &plan);
  ope_pack_all_B(B, ldb, &plan);

  const int rem_M = M & 7, rem_N = N & 7;

  for (int it = 0, i0 = 0; it < plan.itiles; ++it, i0 += 8) {
    const int i_size = (i0 + 8 <= M) ? 8 : rem_M;
    const int8_t* A_it = plan.A_pack + (size_t)it * K * 8;

    for (int jt = 0, j0 = 0; jt < plan.jtiles; ++jt, j0 += 8) {
      const int j_size = (j0 + 8 <= N) ? 8 : rem_N;
      const int8_t* B_jt = plan.B_pack + (size_t)jt * K * 8;

      // ACC from fully prepacked panels in K-chunks of up to 32
      ope_zero();
      for (int k0 = 0; k0 < K; k0 += 32) {
        int L = K - k0; if (L > 32) L = 32;
        ope_acc(A_it + k0*8, B_jt + k0*8, L);
      }

      int32_t* C_ij = &C[(size_t)i0 * ldc + j0];
      if (i_size == 8 && j_size == 8) {
        if (buffered_full_tiles) {
            extract_full_buffered_to_C(C_ij, ldc);
        } else {
          ope_extract(C_ij, ldc, 0, 1);
        }
      } else {
        extract_clamped_to_C(C_ij, ldc, i_size, j_size);
      }
    }
  }
}
