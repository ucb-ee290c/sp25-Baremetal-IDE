// #include "padding.h"

// // Pad (bottom/right) an int8 matrix into top-left of a larger buffer
// void pad_i8_bottom_right(const int8_t* src, size_t rows, size_t cols,
//                          int8_t* dst, size_t rows8, size_t cols8)
// {
//   vmemset_i8(dst, rows8 * cols8);

//   for (size_t r = 0; r < rows; ++r) {
//     const int8_t* s = src + r*cols;
//     int8_t*       d = dst + r*cols8;
//     size_t remain = cols;
//     while (remain) {
//       size_t vl = vsetvl_e8m8(remain);
//       vint8m8_t v = vle8_v_i8m8(s, vl);
//       vse8_v_i8m8(d, v, vl);
//       s += vl; d += vl; remain -= vl;
//     }
//   }
// }

// // Transpose(A) with bottom/right padding
// void transpose_pad_i8_to_AT_padded(const int8_t* A, size_t M, size_t K,
//                                    int8_t* AT, size_t M8, size_t K8)
// {
//   vmemset_i8(AT, K8 * M8);

//   for (size_t c = 0; c < K; ++c) {
//     const int8_t* col = A + c;
//     ptrdiff_t stride = (ptrdiff_t)K;
//     int8_t* out = AT + c * M8;
//     size_t remain = M;
//     while (remain) {
//       size_t vl = vsetvl_e8m8(remain);
//       vint8m8_t v = vlse8_v_i8m8(col, stride, vl);
//       vse8_v_i8m8(out, v, vl);
//       col += vl * stride;
//       out += vl;
//       remain -= vl;
//     }
//   }
// }

// // Pad (bottom/right) an int32 matrix (for C tiles)
// void pad_i32_bottom_right(const int32_t* src, size_t rows, size_t cols,
//                           int32_t* dst, size_t rows8, size_t cols8)
// {
//   vmemset_i32(dst, rows8 * cols8);
//   for (size_t r = 0; r < rows; ++r) {
//     const int32_t* s = src + r*cols;
//     int32_t*       d = dst + r*cols8;
//     size_t remain = cols;
//     while (remain) {
//       size_t vl = vsetvl_e32m8(remain);
//       vint32m8_t v = vle32_v_i32m8(s, vl);
//       vse32_v_i32m8(d, v, vl);
//       s += vl; d += vl; remain -= vl;
//     }
//   }
// }
