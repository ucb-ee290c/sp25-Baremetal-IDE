// #ifndef PADDING_H
// #define PADDING_H

// #include <stdint.h>
// #include <stddef.h>
// #include <riscv_vector.h>

// // Zero-fill helpers (int8 / int32)
// static inline void vmemset_i8(int8_t* dst, size_t n) {
//   while (n) {
//     size_t vl = vsetvl_e8m8(n);
//     vint8m8_t z = vmv_v_x_i8m8(0, vl);
//     vse8_v_i8m8(dst, z, vl);
//     dst += vl; n -= vl;
//   }
// }

// static inline void vmemset_i32(int32_t* dst, size_t n) {
//   while (n) {
//     size_t vl = vsetvl_e32m8(n);
//     vint32m8_t z = vmv_v_x_i32m8(0, vl);
//     vse32_v_i32m8(dst, z, vl);
//     dst += vl; n -= vl;
//   }
// }

// // Pad (bottom/right) an int8 matrix into top-left of a larger buffer
// void pad_i8_bottom_right(const int8_t* src, size_t rows, size_t cols,
//                          int8_t* dst, size_t rows8, size_t cols8);

// // Transpose(A) with bottom/right padding
// void transpose_pad_i8_to_AT_padded(const int8_t* A, size_t M, size_t K,
//                                    int8_t* AT, size_t M8, size_t K8);

// // Pad (bottom/right) an int32 matrix (for C tiles)
// void pad_i32_bottom_right(const int32_t* src, size_t rows, size_t cols,
//                           int32_t* dst, size_t rows8, size_t cols8);

// // Helper for rounding up to the nearest multiple of 8
// static inline size_t round_up8(size_t x) { return (x + 7) & ~((size_t)7); }

// #endif // PADDING_H
