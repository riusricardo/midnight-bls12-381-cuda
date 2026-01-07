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
 * @file msm.cuh
 * @brief Multi-Scalar Multiplication (MSM) - Types and Configuration
 * 
 * This header provides MSM types, configuration, and API declarations.
 * Kernel implementations are in src/curve/msm_kernels.cu.
 * 
 * ALGORITHM (Pippenger's Bucket Method):
 * ======================================
 * Computes: R = Σᵢ sᵢ × Pᵢ for scalars s and points P
 * 
 * 1. Decompose scalars into windows of c bits each
 * 2. For each window, accumulate points into 2^c buckets
 * 3. Compute weighted sum: Σⱼ j × bucket[j] (triangle sum)
 * 4. Combine windows: R = Σ 2^(c*w) × window_sum[w]
 * 
 * LIMITS:
 * =======
 * - Maximum MSM size: 2^24 points (16M)
 * - Window size c: 1-24 bits
 * 
 * MEMORY USAGE:
 * =============
 * Total GPU memory ≈ 3-4x input point array size
 * 
 * SECURITY:
 * =========
 * Uses Sort-Reduce pattern for constant-time bucket accumulation.
 */

#pragma once

#include "point.cuh"
#include "icicle_types.cuh"
#include "gpu_config.cuh"
#include <cuda_runtime.h>

namespace msm {

using namespace bls12_381;
using namespace icicle;
using namespace gpu;

// =============================================================================
// MSM Limits and Constants
// =============================================================================

// Maximum MSM size to prevent integer overflow in packed index encoding
constexpr int MAX_MSM_SIZE = (1 << 24);  // 16M points

// Window size limits
constexpr int MIN_WINDOW_SIZE = 1;
constexpr int MAX_WINDOW_SIZE = 24;

// Invalid bucket index marker (for zero contributions)
constexpr unsigned int INVALID_BUCKET_INDEX = 0xFFFFFFFF;

// =============================================================================
// Point Operation Dispatchers (inline device functions)
// =============================================================================
// These allow templated kernels to call correct point operations for G1/G2.

__device__ __forceinline__ void point_add(G1Projective& result, const G1Projective& a, const G1Projective& b) {
    g1_add(result, a, b);
}

__device__ __forceinline__ void point_add(G2Projective& result, const G2Projective& a, const G2Projective& b) {
    g2_add(result, a, b);
}

__device__ __forceinline__ void point_add_mixed(G1Projective& result, const G1Projective& a, const G1Affine& b) {
    g1_add_mixed(result, a, b);
}

__device__ __forceinline__ void point_add_mixed(G2Projective& result, const G2Projective& a, const G2Affine& b) {
    g2_add_mixed(result, a, b);
}

__device__ __forceinline__ void point_double(G1Projective& result, const G1Projective& a) {
    g1_double(result, a);
}

__device__ __forceinline__ void point_double(G2Projective& result, const G2Projective& a) {
    g2_double(result, a);
}

// =============================================================================
// MSM Configuration Helpers
// =============================================================================

/**
 * @brief Determine optimal window size based on MSM size
 */
__host__ __device__ inline int get_optimal_c(int msm_size) {
    if (msm_size <= 1) return 1;
    
    int log_size = 0;
    int temp = msm_size;
    while (temp > 1) {
        temp >>= 1;
        log_size++;
    }
    
    if (log_size <= 8)  return 7;
    if (log_size <= 10) return 8;
    if (log_size <= 12) return 10;
    if (log_size <= 14) return 12;
    if (log_size <= 16) return 13;
    if (log_size <= 18) return 14;
    if (log_size <= 20) return 15;
    return 16;
}

/**
 * @brief Calculate number of windows for given parameters
 */
__host__ __device__ inline int get_num_windows(int scalar_bits, int c) {
    return (scalar_bits + c - 1) / c;
}

// =============================================================================
// MSM API Declaration
// =============================================================================

/**
 * @brief Multi-Scalar Multiplication using Pippenger's algorithm
 * 
 * @tparam S Scalar type (Fr)
 * @tparam A Affine point type (G1Affine or G2Affine)
 * @tparam P Projective point type (G1Projective or G2Projective)
 * 
 * @param scalars Array of scalars
 * @param bases Array of affine points
 * @param msm_size Number of scalar-point pairs
 * @param config MSM configuration
 * @param result Output projective point
 * 
 * @return cudaSuccess on success, error code otherwise
 */
template<typename S, typename A, typename P>
cudaError_t msm_cuda(
    const S* scalars,
    const A* bases,
    int msm_size,
    const MSMConfig& config,
    P* result
);

// Extern template declarations (instantiated in msm_kernels.cu)
extern template cudaError_t msm_cuda<Fr, G1Affine, G1Projective>(
    const Fr* scalars,
    const G1Affine* bases,
    int msm_size,
    const MSMConfig& config,
    G1Projective* result
);

extern template cudaError_t msm_cuda<Fr, G2Affine, G2Projective>(
    const Fr* scalars,
    const G2Affine* bases,
    int msm_size,
    const MSMConfig& config,
    G2Projective* result
);

} // namespace msm
