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
 * @file bls12_381_params.cuh
 * @brief BLS12-381 CUDA constants for host and device
 * 
 * =============================================================================
 * Architecture
 * =============================================================================
 * 
 * This file provides:
 *   1. Host-side constexpr arrays - for CPU code paths
 *   2. Device-side __constant__ arrays - for GPU code paths
 * 
 * All constant VALUES are defined in bls12_381_constants.h (single source of
 * truth). This file creates the appropriate C++/CUDA bindings.
 * 
 * Device constants are defined inline using static __device__ __constant__
 * which avoids the need for separate compilation and device linking.
 * 
 * =============================================================================
 */

#pragma once

#include <cstdint>
#include "bls12_381_constants.h"

namespace bls12_381 {

// =============================================================================
// Limb Configuration
// =============================================================================

constexpr int FP_LIMBS_64 = BLS12_381_FP_LIMBS_64;  // 384-bit base field
constexpr int FP_LIMBS_32 = FP_LIMBS_64 * 2;
constexpr int FR_LIMBS_64 = BLS12_381_FR_LIMBS_64;  // 256-bit scalar field
constexpr int FR_LIMBS_32 = FR_LIMBS_64 * 2;

// =============================================================================
// Base Field Fq - Host Constants
// =============================================================================

constexpr uint64_t FQ_MODULUS_HOST[FP_LIMBS_64] = FQ_MODULUS_LIMBS;
constexpr uint64_t FQ_ONE_HOST[FP_LIMBS_64]     = FQ_ONE_LIMBS;
constexpr uint64_t FQ_R2_HOST[FP_LIMBS_64]      = FQ_R2_LIMBS;
constexpr uint64_t FQ_INV                       = FQ_INV_VALUE;

// =============================================================================
// Base Field Fq - Device Constants
// =============================================================================

static __device__ __constant__ uint64_t FQ_MODULUS[FP_LIMBS_64] = FQ_MODULUS_LIMBS;
static __device__ __constant__ uint64_t FQ_ONE[FP_LIMBS_64]     = FQ_ONE_LIMBS;
static __device__ __constant__ uint64_t FQ_R2[FP_LIMBS_64]      = FQ_R2_LIMBS;

// =============================================================================
// Scalar Field Fr - Host Constants
// =============================================================================

constexpr uint64_t FR_MODULUS_HOST[FR_LIMBS_64] = FR_MODULUS_LIMBS;
constexpr uint64_t FR_ONE_HOST[FR_LIMBS_64]     = FR_ONE_LIMBS;
constexpr uint64_t FR_R2_HOST[FR_LIMBS_64]      = FR_R2_LIMBS;
constexpr uint64_t FR_INV                       = FR_INV_VALUE;

// =============================================================================
// Scalar Field Fr - Device Constants
// =============================================================================

static __device__ __constant__ uint64_t FR_MODULUS[FR_LIMBS_64] = FR_MODULUS_LIMBS;
static __device__ __constant__ uint64_t FR_ONE[FR_LIMBS_64]     = FR_ONE_LIMBS;
static __device__ __constant__ uint64_t FR_R2[FR_LIMBS_64]      = FR_R2_LIMBS;

// =============================================================================
// G1 Curve Parameters - Host Constants
// =============================================================================

constexpr uint64_t G1_B[FP_LIMBS_64]           = G1_B_LIMBS;
constexpr uint64_t G1_GENERATOR_X[FP_LIMBS_64] = G1_GEN_X_LIMBS;
constexpr uint64_t G1_GENERATOR_Y[FP_LIMBS_64] = G1_GEN_Y_LIMBS;

// =============================================================================
// G1 Curve Parameters - Device Constants
// =============================================================================

static __device__ __constant__ uint64_t G1_B_DEV[FP_LIMBS_64] = G1_B_LIMBS;

// =============================================================================
// G2 Curve Parameters - Host Constants
// =============================================================================
// G2 coordinates are Fq2 elements: each has c0 (real) and c1 (imaginary) parts

// Curve coefficient b' = 4(1+u) in Fq2
constexpr uint64_t G2_B_C0[FP_LIMBS_64]           = G2_B_C0_LIMBS;
constexpr uint64_t G2_B_C1[FP_LIMBS_64]           = G2_B_C1_LIMBS;

// =============================================================================
// G2 Curve Parameters - Device Constants
// =============================================================================

static __device__ __constant__ uint64_t G2_B_C0_DEV[FP_LIMBS_64] = G2_B_C0_LIMBS;
static __device__ __constant__ uint64_t G2_B_C1_DEV[FP_LIMBS_64] = G2_B_C1_LIMBS;

// Generator x coordinate (Fq2)
constexpr uint64_t G2_GENERATOR_X_C0[FP_LIMBS_64] = G2_GEN_X_C0_LIMBS;
constexpr uint64_t G2_GENERATOR_X_C1[FP_LIMBS_64] = G2_GEN_X_C1_LIMBS;

// Generator y coordinate (Fq2)
constexpr uint64_t G2_GENERATOR_Y_C0[FP_LIMBS_64] = G2_GEN_Y_C0_LIMBS;
constexpr uint64_t G2_GENERATOR_Y_C1[FP_LIMBS_64] = G2_GEN_Y_C1_LIMBS;

// =============================================================================
// NTT Parameters
// =============================================================================

// Maximum NTT size: 2^32 (determined by scalar field order r-1 = 2^32 * m)
constexpr int MAX_NTT_LOG_SIZE = 32;

// Primitive 2^32-th root of unity in Fr (Montgomery form) - Host
constexpr uint64_t FR_OMEGA_HOST[FR_LIMBS_64] = FR_OMEGA_LIMBS;

// Primitive 2^32-th root of unity in Fr (Montgomery form) - Device
static __device__ __constant__ uint64_t FR_OMEGA[FR_LIMBS_64] = FR_OMEGA_LIMBS;

} // namespace bls12_381
