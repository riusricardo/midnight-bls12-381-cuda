/*
 * Copyright (C) 2026 BLS12-381 CUDA Backend Contributors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of BLS12-381 CUDA Backend.
 *
 * BLS12-381 CUDA Backend is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * BLS12-381 CUDA Backend is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with BLS12-381 CUDA Backend.  If not, see <https://www.gnu.org/licenses/>.
 */

/**
 * @file bls12_381_constants.h
 * @brief BLS12-381 Curve Constants - Single Source of Truth
 * 
 * =============================================================================
 * AUDIT NOTICE
 * =============================================================================
 * This file contains all cryptographic constants for the BLS12-381 curve.
 * All values are defined here ONCE and consumed by both:
 *   - Host code (via constexpr arrays in bls12_381_params.cuh)
 *   - Device code (via static __device__ __constant__ arrays in bls12_381_params.cuh)
 * 
 * DO NOT define these constants anywhere else. Any modification to curve
 * parameters must be made in this file only.
 * 
 * Constants verified against:
 *   - BLST library (https://github.com/supranational/blst)
 *   - Arkworks (https://github.com/arkworks-rs/curves)
 *   - EIP-2537 specification
 * 
 * All multi-limb values are in LITTLE-ENDIAN order (least significant first).
 * Field elements are in Montgomery representation where noted.
 * =============================================================================
 */

#ifndef BLS12_381_CONSTANTS_H
#define BLS12_381_CONSTANTS_H

/* ============================================================================
 * Limb Configuration
 * ============================================================================ */

#define BLS12_381_FP_LIMBS_64  6   /* 384-bit base field Fq */
#define BLS12_381_FR_LIMBS_64  4   /* 256-bit scalar field Fr */

/* ============================================================================
 * Base Field Fq (381-bit prime)
 * ============================================================================
 * p = 0x1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaab
 * 
 * This is the field over which G1 points are defined.
 * ============================================================================ */

/* Modulus p in little-endian 64-bit limbs */
#define FQ_MODULUS_L0  0xb9feffffffffaaabULL
#define FQ_MODULUS_L1  0x1eabfffeb153ffffULL
#define FQ_MODULUS_L2  0x6730d2a0f6b0f624ULL
#define FQ_MODULUS_L3  0x64774b84f38512bfULL
#define FQ_MODULUS_L4  0x4b1ba7b6434bacd7ULL
#define FQ_MODULUS_L5  0x1a0111ea397fe69aULL

/* Montgomery constant: -p^{-1} mod 2^64 */
#define FQ_INV_VALUE   0x89f3fffcfffcfffdULL

/* R = 2^384 mod p (Montgomery one)
 * This is the multiplicative identity in Montgomery form */
#define FQ_ONE_L0      0x760900000002fffdULL
#define FQ_ONE_L1      0xebf4000bc40c0002ULL
#define FQ_ONE_L2      0x5f48985753c758baULL
#define FQ_ONE_L3      0x77ce585370525745ULL
#define FQ_ONE_L4      0x5c071a97a256ec6dULL
#define FQ_ONE_L5      0x15f65ec3fa80e493ULL

/* R^2 = 2^768 mod p (for converting integers to Montgomery form)
 * To convert x to Montgomery form: mont(x) = x * R^2 * R^{-1} mod p = x * R mod p */
#define FQ_R2_L0       0xf4df1f341c341746ULL
#define FQ_R2_L1       0x0a76e6a609d104f1ULL
#define FQ_R2_L2       0x8de5476c4c95b6d5ULL
#define FQ_R2_L3       0x67eb88a9939d83c0ULL
#define FQ_R2_L4       0x9a793e85b519952dULL
#define FQ_R2_L5       0x11988fe592cae3aaULL

/* ============================================================================
 * Scalar Field Fr (255-bit prime)
 * ============================================================================
 * r = 0x73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001
 * 
 * This is the order of the G1 and G2 groups (number of points on the curve).
 * ============================================================================ */

/* Modulus r in little-endian 64-bit limbs */
#define FR_MODULUS_L0  0xffffffff00000001ULL
#define FR_MODULUS_L1  0x53bda402fffe5bfeULL
#define FR_MODULUS_L2  0x3339d80809a1d805ULL
#define FR_MODULUS_L3  0x73eda753299d7d48ULL

/* Montgomery constant: -r^{-1} mod 2^64 */
#define FR_INV_VALUE   0xfffffffeffffffffULL

/* R = 2^256 mod r (Montgomery one) */
#define FR_ONE_L0      0x00000001fffffffeULL
#define FR_ONE_L1      0x5884b7fa00034802ULL
#define FR_ONE_L2      0x998c4fefecbc4ff5ULL
#define FR_ONE_L3      0x1824b159acc5056fULL

/* R^2 = 2^512 mod r (for converting integers to Montgomery form) */
#define FR_R2_L0       0xc999e990f3f29c6dULL
#define FR_R2_L1       0x2b6cedcb87925c23ULL
#define FR_R2_L2       0x05d314967254398fULL
#define FR_R2_L3       0x0748d9d99f59ff11ULL

/* Primitive 2^32-th root of unity in Fr (Montgomery form)
 * omega = generator^((r-1)/2^32)
 * Used for NTT operations up to size 2^32
 * Verified against BLST and Arkworks implementations */
#define FR_OMEGA_L0    0xb9b58d8c5f0e466aULL
#define FR_OMEGA_L1    0x5b1b4c801819d7ecULL
#define FR_OMEGA_L2    0x0af53ae352a31e64ULL
#define FR_OMEGA_L3    0x5bf3adda19e9b27bULL

/* ============================================================================
 * G1 Curve Parameters
 * ============================================================================
 * G1: y^2 = x^3 + 4 over Fq
 * ============================================================================ */

/* Curve coefficient b = 4 in Montgomery form */
#define G1_B_L0        0xaa270000000cfff3ULL
#define G1_B_L1        0x53cc0032fc34000aULL
#define G1_B_L2        0x478fe97a6b0a807fULL
#define G1_B_L3        0xb1d37ebee6ba24d7ULL
#define G1_B_L4        0x8ec9733bbf78ab2fULL
#define G1_B_L5        0x09d645513d83de7eULL

/* Generator point G1 - x coordinate in Montgomery form */
#define G1_GEN_X_L0    0x5cb38790fd530c16ULL
#define G1_GEN_X_L1    0x7817fc679976fff5ULL
#define G1_GEN_X_L2    0x154f95c7143ba1c1ULL
#define G1_GEN_X_L3    0xf0ae6acdf3d0e747ULL
#define G1_GEN_X_L4    0xedce6ecc21dbf440ULL
#define G1_GEN_X_L5    0x120177419e0bfb75ULL

/* Generator point G1 - y coordinate in Montgomery form */
#define G1_GEN_Y_L0    0xbaac93d50ce72271ULL
#define G1_GEN_Y_L1    0x8c22631a7918fd8eULL
#define G1_GEN_Y_L2    0xdd595f13570725ceULL
#define G1_GEN_Y_L3    0x51ac582950405194ULL
#define G1_GEN_Y_L4    0x0e1c8c3fad0059c0ULL
#define G1_GEN_Y_L5    0x0bbc3efc5008a26aULL

/* ============================================================================
 * G2 Curve Parameters
 * ============================================================================
 * G2: y^2 = x^3 + 4(u+1) over Fq2, where Fq2 = Fq[u]/(u^2+1)
 * 
 * Each G2 coordinate is an Fq2 element: c0 + c1*u
 * ============================================================================ */

/* Curve coefficient b' = 4(1+u) in Montgomery form
 * b'.c0 = 4 in Montgomery form (same as G1_B)
 * b'.c1 = 4 in Montgomery form (same as G1_B) */
#define G2_B_C0_L0     0xaa270000000cfff3ULL
#define G2_B_C0_L1     0x53cc0032fc34000aULL
#define G2_B_C0_L2     0x478fe97a6b0a807fULL
#define G2_B_C0_L3     0xb1d37ebee6ba24d7ULL
#define G2_B_C0_L4     0x8ec9733bbf78ab2fULL
#define G2_B_C0_L5     0x09d645513d83de7eULL

#define G2_B_C1_L0     0xaa270000000cfff3ULL
#define G2_B_C1_L1     0x53cc0032fc34000aULL
#define G2_B_C1_L2     0x478fe97a6b0a807fULL
#define G2_B_C1_L3     0xb1d37ebee6ba24d7ULL
#define G2_B_C1_L4     0x8ec9733bbf78ab2fULL
#define G2_B_C1_L5     0x09d645513d83de7eULL

/* Generator point G2 - x coordinate (Fq2 element)
 * x = x.c0 + x.c1 * u
 * Values from BLST/Arkworks in Montgomery form */

/* x.c0 */
#define G2_GEN_X_C0_L0 0xf5f28fa202940a10ULL
#define G2_GEN_X_C0_L1 0xb3f5fb2687b4961aULL
#define G2_GEN_X_C0_L2 0xa1a893b53e2ae580ULL
#define G2_GEN_X_C0_L3 0x9894999d1a3caee9ULL
#define G2_GEN_X_C0_L4 0x6f67b7631863366bULL
#define G2_GEN_X_C0_L5 0x058191924350bcd7ULL

/* x.c1 */
#define G2_GEN_X_C1_L0 0xa5a9c0759e23f606ULL
#define G2_GEN_X_C1_L1 0xaaa0c59dbccd60c3ULL
#define G2_GEN_X_C1_L2 0x3bb17e18e2867806ULL
#define G2_GEN_X_C1_L3 0x1b1ab6cc8541b367ULL
#define G2_GEN_X_C1_L4 0xc2b6ed0ef2158547ULL
#define G2_GEN_X_C1_L5 0x11922a097360edf3ULL

/* Generator point G2 - y coordinate (Fq2 element)
 * y = y.c0 + y.c1 * u */

/* y.c0 */
#define G2_GEN_Y_C0_L0 0x4c730af860494c4aULL
#define G2_GEN_Y_C0_L1 0x597cfa1f5e369c5aULL
#define G2_GEN_Y_C0_L2 0xe7e6856caa0a635aULL
#define G2_GEN_Y_C0_L3 0xbbefb5e96e0d495fULL
#define G2_GEN_Y_C0_L4 0x07d3a975f0ef25a2ULL
#define G2_GEN_Y_C0_L5 0x0083fd8e7e80dae5ULL

/* y.c1 */
#define G2_GEN_Y_C1_L0 0xadc0fc92df64b05dULL
#define G2_GEN_Y_C1_L1 0x18aa270a2b1461dcULL
#define G2_GEN_Y_C1_L2 0x86adac6a3be4eba0ULL
#define G2_GEN_Y_C1_L3 0x79495c4ec93da33aULL
#define G2_GEN_Y_C1_L4 0xe7175850a43ccaedULL
#define G2_GEN_Y_C1_L5 0x0b2bc2a163de1bf2ULL

/* ============================================================================
 * Convenience Macros for Array Initialization
 * ============================================================================ */

#define FQ_MODULUS_LIMBS  { FQ_MODULUS_L0, FQ_MODULUS_L1, FQ_MODULUS_L2, \
                            FQ_MODULUS_L3, FQ_MODULUS_L4, FQ_MODULUS_L5 }

#define FQ_ONE_LIMBS      { FQ_ONE_L0, FQ_ONE_L1, FQ_ONE_L2, \
                            FQ_ONE_L3, FQ_ONE_L4, FQ_ONE_L5 }

#define FQ_R2_LIMBS       { FQ_R2_L0, FQ_R2_L1, FQ_R2_L2, \
                            FQ_R2_L3, FQ_R2_L4, FQ_R2_L5 }

#define FR_MODULUS_LIMBS  { FR_MODULUS_L0, FR_MODULUS_L1, \
                            FR_MODULUS_L2, FR_MODULUS_L3 }

#define FR_ONE_LIMBS      { FR_ONE_L0, FR_ONE_L1, FR_ONE_L2, FR_ONE_L3 }

#define FR_R2_LIMBS       { FR_R2_L0, FR_R2_L1, FR_R2_L2, FR_R2_L3 }

#define FR_OMEGA_LIMBS    { FR_OMEGA_L0, FR_OMEGA_L1, FR_OMEGA_L2, FR_OMEGA_L3 }

#define G1_B_LIMBS        { G1_B_L0, G1_B_L1, G1_B_L2, \
                            G1_B_L3, G1_B_L4, G1_B_L5 }

#define G1_GEN_X_LIMBS    { G1_GEN_X_L0, G1_GEN_X_L1, G1_GEN_X_L2, \
                            G1_GEN_X_L3, G1_GEN_X_L4, G1_GEN_X_L5 }

#define G1_GEN_Y_LIMBS    { G1_GEN_Y_L0, G1_GEN_Y_L1, G1_GEN_Y_L2, \
                            G1_GEN_Y_L3, G1_GEN_Y_L4, G1_GEN_Y_L5 }

/* G2 coefficient b' = 4(1+u) components */
#define G2_B_C0_LIMBS     { G2_B_C0_L0, G2_B_C0_L1, G2_B_C0_L2, \
                            G2_B_C0_L3, G2_B_C0_L4, G2_B_C0_L5 }
#define G2_B_C1_LIMBS     { G2_B_C1_L0, G2_B_C1_L1, G2_B_C1_L2, \
                            G2_B_C1_L3, G2_B_C1_L4, G2_B_C1_L5 }

/* G2 generator x coordinate (Fq2 element) */
#define G2_GEN_X_C0_LIMBS { G2_GEN_X_C0_L0, G2_GEN_X_C0_L1, G2_GEN_X_C0_L2, \
                            G2_GEN_X_C0_L3, G2_GEN_X_C0_L4, G2_GEN_X_C0_L5 }
#define G2_GEN_X_C1_LIMBS { G2_GEN_X_C1_L0, G2_GEN_X_C1_L1, G2_GEN_X_C1_L2, \
                            G2_GEN_X_C1_L3, G2_GEN_X_C1_L4, G2_GEN_X_C1_L5 }

/* G2 generator y coordinate (Fq2 element) */
#define G2_GEN_Y_C0_LIMBS { G2_GEN_Y_C0_L0, G2_GEN_Y_C0_L1, G2_GEN_Y_C0_L2, \
                            G2_GEN_Y_C0_L3, G2_GEN_Y_C0_L4, G2_GEN_Y_C0_L5 }
#define G2_GEN_Y_C1_LIMBS { G2_GEN_Y_C1_L0, G2_GEN_Y_C1_L1, G2_GEN_Y_C1_L2, \
                            G2_GEN_Y_C1_L3, G2_GEN_Y_C1_L4, G2_GEN_Y_C1_L5 }

#endif /* BLS12_381_CONSTANTS_H */
