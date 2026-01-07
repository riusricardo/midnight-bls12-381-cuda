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
 * @file icicle_types.cuh
 * @brief Icicle-compatible type definitions and interface
 * 
 * These types match the exact signatures expected by the Icicle Rust bindings.
 */

#pragma once

#include "field.cuh"
#include "point.cuh"
#include <cuda_runtime.h>

namespace icicle {

// =============================================================================
// Error codes (matching Icicle's eIcicleError)
// =============================================================================

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) return eIcicleError::ALLOCATION_FAILED; \
    } while(0)

enum class eIcicleError {
    SUCCESS = 0,
    INVALID_DEVICE = 1,
    OUT_OF_MEMORY = 2,
    INVALID_POINTER = 3,
    ALLOCATION_FAILED = 4,
    DEALLOCATION_FAILED = 5,
    COPY_FAILED = 6,
    SYNC_FAILED = 7,
    STREAM_CREATE_FAILED = 8,
    STREAM_DESTROY_FAILED = 9,
    API_NOT_IMPLEMENTED = 10,
    INVALID_ARGUMENT = 11,
    BACKEND_LOAD_FAILED = 12,
    LICENSE_CHECK_FAILED = 13,
    UNKNOWN_ERROR = 999
};

// =============================================================================
// Device context (matching Icicle's Device struct)
// =============================================================================

struct Device {
    const char* type;
    int id;
};

// =============================================================================
// Stream handle
// =============================================================================

using IcicleStreamHandle = cudaStream_t;

// =============================================================================
// NTT Direction and Ordering
// =============================================================================

enum class NTTDir {
    kForward = 0,
    kInverse = 1
};

enum class Ordering {
    kNN = 0,  // Natural-Natural
    kNR = 1,  // Natural-Reversed
    kRN = 2,  // Reversed-Natural
    kRR = 3,  // Reversed-Reversed
    kNM = 4,  // Mixed
    kMN = 5   // Mixed
};

// =============================================================================
// NTT Configuration (matching Icicle's NTTConfig<S> template exactly)
// =============================================================================

template <typename S>
struct NTTConfig {
    void* stream;                   // IcicleStreamHandle
    S coset_gen;                    // Coset generator (default: S::one())
    int batch_size;
    bool columns_batch;
    Ordering ordering;
    bool are_inputs_on_device;
    bool are_outputs_on_device;
    bool is_async;
    void* ext;
};

// Default NTT configuration
template <typename S>
inline NTTConfig<S> default_ntt_config() {
    NTTConfig<S> cfg;
    cfg.stream = nullptr;
    cfg.coset_gen = S::one();
    cfg.batch_size = 1;
    cfg.columns_batch = false;
    cfg.ordering = Ordering::kNN;
    cfg.are_inputs_on_device = false;
    cfg.are_outputs_on_device = false;
    cfg.is_async = false;
    cfg.ext = nullptr;
    return cfg;
}

// =============================================================================
// NTT Init Domain Configuration
// Matches ICICLE's NTTInitDomainConfig exactly for ABI compatibility
// =============================================================================

struct NTTInitDomainConfig {
    void* stream;                  // icicleStreamHandle for async execution
    bool is_async;                 // True if operation is asynchronous
    void* ext;                     // ConfigExtension* (can be nullptr)
};

inline NTTInitDomainConfig default_ntt_init_domain_config() {
    NTTInitDomainConfig cfg;
    cfg.stream = nullptr;
    cfg.is_async = false;
    cfg.ext = nullptr;
    return cfg;
}

// =============================================================================
// MSM Configuration (matching Icicle's MSMConfig exactly for ABI compatibility)
// Field order must match exactly for binary compatibility with Icicle runtime
// =============================================================================

struct MSMConfig {
    IcicleStreamHandle stream;       // icicleStreamHandle for async execution
    int precompute_factor;           // Points to pre-compute per input point
    int c;                           // Window bitsize (0 = auto)
    int bitsize;                     // Scalar bitsize (0 = use field size)
    int batch_size;                  // Number of MSMs to compute
    bool are_points_shared_in_batch; // Bases shared across batch
    bool are_scalars_on_device;      // Scalars in device memory
    bool are_scalars_montgomery_form;// Scalars in Montgomery form
    bool are_points_on_device;       // Points in device memory
    bool are_points_montgomery_form; // Points in Montgomery form
    bool are_results_on_device;      // Results in device memory
    bool is_async;                   // Async execution
    void* ext;                       // ConfigExtension* (can be nullptr)
};

// Default MSM configuration
inline MSMConfig default_msm_config() {
    MSMConfig cfg;
    cfg.stream = nullptr;
    cfg.precompute_factor = 1;
    cfg.c = 0;
    cfg.bitsize = 0;
    cfg.batch_size = 1;
    cfg.are_points_shared_in_batch = true;
    cfg.are_scalars_on_device = false;
    cfg.are_scalars_montgomery_form = false;
    cfg.are_points_on_device = false;
    cfg.are_points_montgomery_form = false;
    cfg.are_results_on_device = false;
    cfg.is_async = false;
    cfg.ext = nullptr;
    return cfg;
}

// =============================================================================
// Vector operations configuration
// =============================================================================

struct VecOpsConfig {
    IcicleStreamHandle stream;
    bool is_a_on_device;
    bool is_b_on_device;
    bool is_result_on_device;
    bool is_async;
    void* ext;
};

} // namespace icicle

// Make commonly used types available at global scope
using eIcicleError = icicle::eIcicleError;
using NTTDir = icicle::NTTDir;
template<typename S>
using NTTConfig = icicle::NTTConfig<S>;
using NTTInitDomainConfig = icicle::NTTInitDomainConfig;
using MSMConfig = icicle::MSMConfig;
using VecOpsConfig = icicle::VecOpsConfig;
using Ordering = icicle::Ordering;

// =============================================================================
// Type mappings for BLS12-381 to match Icicle symbol names
// =============================================================================

namespace bls12_381 {

// These are the exact type names used in the Icicle library symbol names
using scalar_t = Fr;
using point_t = G1Projective;
using affine_t = G1Affine;
using g2_point_t = G2Projective;
using g2_affine_t = G2Affine;

} // namespace bls12_381
