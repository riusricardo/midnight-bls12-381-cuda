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
 * @file icicle_curve_api.cu
 * @brief MSM Implementation, Backend Registration, and C API
 * 
 * This is the main MSM entry point file. It provides:
 * 
 * FILE STRUCTURE:
 * ===============
 * 1. Internal helper kernels (Montgomery/coordinate conversions)
 * 2. MSM implementation wrappers (G1 and G2)
 * 3. Precompute bases wrappers (G1 and G2)
 * 4. Template instantiations for G1 and G2 MSM
 * 5. Test C APIs (bls12_381_g1_msm_cuda, bls12_381_g2_msm_cuda)
 * 6. ICICLE backend registration (at bottom - static initializers)
 * 
 * ICICLE REGISTRATION:
 * ====================
 * The registration mechanism works as follows:
 * 1. This library is loaded via dlopen() by Icicle runtime
 * 2. Static initializers run the REGISTER_MSM_BACKEND macro
 * 3. The macro calls register_msm("CUDA", impl) 
 * 4. register_msm is resolved from libicicle_curve_bls12_381.so
 * 5. Our implementation is now available for the "CUDA" device type
 * 
 * MONTGOMERY FORM HANDLING:
 * =========================
 * Icicle's Rust bindings send field elements in NON-Montgomery form (via to_repr()).
 * 
 * CRITICAL INSIGHT:
 * - SCALARS are used for BIT EXTRACTION in MSM (to determine bucket indices)
 *   They must remain in STANDARD form - we extract bits from the actual value!
 *   Converting to Montgomery would make MSM compute (R*s mod r) * G instead of s * G
 * 
 * - POINTS use field arithmetic during bucket accumulation
 *   They need to be in MONTGOMERY form for efficient modular multiplication
 * 
 * - RESULTS are returned in standard form (converted from Montgomery)
 */

// Our minimal Icicle-compatible API declarations
#include "icicle_backend_api.cuh"

// Include field.cuh for Montgomery conversion functions
#include "field.cuh"
#include "point.cuh"

// Include GPU config for thread/block constants
#include "gpu_config.cuh"

// Include full MSM template definitions - this file provides the instantiations
#include "msm.cuh"

using namespace bls12_381;

// =============================================================================
// Internal Helper Kernels (Montgomery and Coordinate Conversions)
// =============================================================================

// Convert scalars from Montgomery form to standard form
// This is needed when are_scalars_montgomery_form=true, because MSM extracts
// bits from scalars to determine bucket indices - we need the actual scalar value.
__global__ void scalars_from_montgomery_kernel(Fr* scalars, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    Fr& s = scalars[idx];
    Fr s_std;
    field_from_montgomery(s_std, s);
    s = s_std;
}

// Convert G1 affine points from standard form to Montgomery form
__global__ void points_to_montgomery_kernel(G1Affine* points, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    G1Affine& p = points[idx];
    Fq x_mont, y_mont;
    field_to_montgomery(x_mont, p.x);
    field_to_montgomery(y_mont, p.y);
    p.x = x_mont;
    p.y = y_mont;
}

// Convert G2 affine points from standard form to Montgomery form
__global__ void g2_points_to_montgomery_kernel(G2Affine* points, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    G2Affine& p = points[idx];
    
    Fq2 x_mont, y_mont;
    
    // Convert both components of x (c0 and c1)
    field_to_montgomery(x_mont.c0, p.x.c0);
    field_to_montgomery(x_mont.c1, p.x.c1);
    
    // Convert both components of y (c0 and c1)
    field_to_montgomery(y_mont.c0, p.y.c0);
    field_to_montgomery(y_mont.c1, p.y.c1);
    
    p.x = x_mont;
    p.y = y_mont;
}

// Convert G1 Jacobian projective result (Montgomery) to standard projective (standard form)
// for ICICLE which expects standard projective: x = X/Z, y = Y/Z
//
// Our MSM uses Jacobian: x = X/Z², y = Y/Z³
// ICICLE uses Standard: x = X/Z, y = Y/Z
//
// Conversion: compute affine (x, y) then return as (x, y, 1) in standard form
__global__ void jacobian_to_standard_projective_kernel(G1Projective* result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    G1Projective& p = result[idx];
    
    // Check identity
    if (p.Z.is_zero()) {
        // Identity point: set to (0, 1, 0) in standard form
        // But first convert from Montgomery - identity in Montgomery is still identity
        p.X = Fq::zero();
        p.Y = Fq::one();  // This gives 1 in Montgomery form, need standard 1
        // Actually for identity we can just set all to zero except Y=1 in standard form
        for (int i = 0; i < 6; i++) {
            p.X.limbs[i] = 0;
            p.Y.limbs[i] = 0;
            p.Z.limbs[i] = 0;
        }
        p.Y.limbs[0] = 1;  // Y = 1 in standard form
        return;
    }
    
    // Compute affine coordinates from Jacobian: x = X/Z², y = Y/Z³
    Fq z_inv, z_inv_sq, z_inv_cu;
    field_inv(z_inv, p.Z);
    field_sqr(z_inv_sq, z_inv);
    field_mul(z_inv_cu, z_inv_sq, z_inv);
    
    Fq x_aff, y_aff;
    field_mul(x_aff, p.X, z_inv_sq);   // x = X/Z²
    field_mul(y_aff, p.Y, z_inv_cu);   // y = Y/Z³
    
    // Now x_aff, y_aff are in Montgomery form
    // Convert to standard form for ICICLE
    Fq x_std, y_std;
    field_from_montgomery(x_std, x_aff);
    field_from_montgomery(y_std, y_aff);
    
    // Return as standard projective with Z=1 (standard form 1)
    p.X = x_std;
    p.Y = y_std;
    // Z = 1 in standard form
    for (int i = 0; i < 6; i++) p.Z.limbs[i] = 0;
    p.Z.limbs[0] = 1;
}

// Convert G2 Jacobian projective result (Montgomery) to standard projective (standard form)
__global__ void g2_jacobian_to_standard_projective_kernel(G2Projective* result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    G2Projective& p = result[idx];
    
    // Check identity
    if (p.Z.is_zero()) {
        // Identity point in G2: (0, 1, 0) in standard form
        for (int i = 0; i < 6; i++) {
            p.X.c0.limbs[i] = 0;
            p.X.c1.limbs[i] = 0;
            p.Y.c0.limbs[i] = 0;
            p.Y.c1.limbs[i] = 0;
            p.Z.c0.limbs[i] = 0;
            p.Z.c1.limbs[i] = 0;
        }
        p.Y.c0.limbs[0] = 1;  // Y = 1 in standard form
        return;
    }
    
    // Compute affine coordinates from Jacobian: x = X/Z², y = Y/Z³
    Fq2 z_inv, z_inv_sq, z_inv_cu;
    fq2_inv(z_inv, p.Z);
    fq2_sqr(z_inv_sq, z_inv);
    fq2_mul(z_inv_cu, z_inv_sq, z_inv);
    
    Fq2 x_aff, y_aff;
    fq2_mul(x_aff, p.X, z_inv_sq);   // x = X/Z²
    fq2_mul(y_aff, p.Y, z_inv_cu);   // y = Y/Z³
    
    // Now x_aff, y_aff are in Montgomery form
    // Convert to standard form for ICICLE
    Fq2 x_std, y_std;
    field_from_montgomery(x_std.c0, x_aff.c0);
    field_from_montgomery(x_std.c1, x_aff.c1);
    field_from_montgomery(y_std.c0, y_aff.c0);
    field_from_montgomery(y_std.c1, y_aff.c1);
    
    // Return as standard projective with Z=1 (standard form 1)
    p.X = x_std;
    p.Y = y_std;
    // Z = 1 in standard form
    for (int i = 0; i < 6; i++) {
        p.Z.c0.limbs[i] = 0;
        p.Z.c1.limbs[i] = 0;
    }
    p.Z.c0.limbs[0] = 1;
}

// =============================================================================
// G1 MSM Implementation
// =============================================================================

/**
 * @brief G1 MSM wrapper for ICICLE backend
 * 
 * Handles Montgomery form conversions and coordinate system conversions:
 * - Input: Affine points + scalars (standard or Montgomery form per config flags)
 * - Internal: Montgomery form for point field arithmetic, standard form for scalars
 * - Output: Standard form standard projective (x=X/Z, y=Y/Z)
 */
static icicle::eIcicleError msm_cuda_impl(
    const icicle::Device& device,
    const icicle::icicle_scalar_t* scalars,
    const icicle::icicle_affine_t* bases,
    int msm_size,
    const icicle::MSMConfig& config,
    icicle::icicle_projective_t* results)
{
    (void)device; // Unused - we always use CUDA context
    
    if (msm_size == 0) {
        return icicle::eIcicleError::SUCCESS;
    }
    
    cudaStream_t stream = static_cast<cudaStream_t>(config.stream);
    cudaError_t err;
    
    // We need to convert inputs to appropriate form and outputs from Montgomery form.
    // To avoid modifying input data, we allocate temporary buffers when needed.
    Fr* d_scalars = nullptr;
    G1Affine* d_bases = nullptr;
    G1Projective* d_result = nullptr;
    
    bool allocated_scalars = false;
    bool allocated_bases = false;
    bool allocated_result = false;
    
    // --- Handle scalars ---
    // MSM extracts bits from scalars to determine bucket indices.
    // Scalars must be in STANDARD form so bits represent actual scalar value.
    // If are_scalars_montgomery_form=true, we need to convert from Montgomery to standard.
    if (config.are_scalars_on_device) {
        if (config.are_scalars_montgomery_form) {
            // Scalars are on device in Montgomery form - need to convert to standard
            err = cudaMalloc(&d_scalars, msm_size * sizeof(Fr));
            if (err != cudaSuccess) goto cleanup;
            allocated_scalars = true;
            
            err = cudaMemcpyAsync(d_scalars, scalars, msm_size * sizeof(Fr),
                                  cudaMemcpyDeviceToDevice, stream);
            if (err != cudaSuccess) goto cleanup;
            
            // Convert from Montgomery to standard form
            constexpr int threads = gpu::GPUConfig::DEFAULT_COMPUTE_THREADS;
            int blocks = (msm_size + threads - 1) / threads;
            scalars_from_montgomery_kernel<<<blocks, threads, 0, stream>>>(d_scalars, msm_size);
            err = cudaGetLastError();
            if (err != cudaSuccess) goto cleanup;
        } else {
            // Scalars already in standard form - use directly
            d_scalars = const_cast<Fr*>(reinterpret_cast<const Fr*>(scalars));
        }
    } else {
        // Data on host - copy to device
        err = cudaMalloc(&d_scalars, msm_size * sizeof(Fr));
        if (err != cudaSuccess) goto cleanup;
        allocated_scalars = true;
        
        err = cudaMemcpyAsync(d_scalars, scalars, msm_size * sizeof(Fr),
                              cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess) goto cleanup;
        
        if (config.are_scalars_montgomery_form) {
            // Convert from Montgomery to standard form
            constexpr int threads_host = gpu::GPUConfig::DEFAULT_COMPUTE_THREADS;
            int blocks = (msm_size + threads_host - 1) / threads_host;
            scalars_from_montgomery_kernel<<<blocks, threads_host, 0, stream>>>(d_scalars, msm_size);
            err = cudaGetLastError();
            if (err != cudaSuccess) goto cleanup;
        }
    }
    
    // --- Handle bases (points) ---
    if (config.are_points_on_device) {
        if (!config.are_points_montgomery_form) {
            err = cudaMalloc(&d_bases, msm_size * sizeof(G1Affine));
            if (err != cudaSuccess) goto cleanup;
            allocated_bases = true;
            
            err = cudaMemcpyAsync(d_bases, bases, msm_size * sizeof(G1Affine),
                                  cudaMemcpyDeviceToDevice, stream);
            if (err != cudaSuccess) goto cleanup;
            
            // Convert to Montgomery form
            constexpr int threads = gpu::GPUConfig::DEFAULT_COMPUTE_THREADS;
            int blocks = (msm_size + threads - 1) / threads;
            points_to_montgomery_kernel<<<blocks, threads, 0, stream>>>(d_bases, msm_size);
            err = cudaGetLastError();
            if (err != cudaSuccess) goto cleanup;
        } else {
            d_bases = const_cast<G1Affine*>(reinterpret_cast<const G1Affine*>(bases));
        }
    } else {
        err = cudaMalloc(&d_bases, msm_size * sizeof(G1Affine));
        if (err != cudaSuccess) goto cleanup;
        allocated_bases = true;
        
        err = cudaMemcpyAsync(d_bases, bases, msm_size * sizeof(G1Affine),
                              cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess) goto cleanup;
        
        if (!config.are_points_montgomery_form) {
            constexpr int threads = gpu::GPUConfig::DEFAULT_COMPUTE_THREADS;
            int blocks = (msm_size + threads - 1) / threads;
            points_to_montgomery_kernel<<<blocks, threads, 0, stream>>>(d_bases, msm_size);
            err = cudaGetLastError();
            if (err != cudaSuccess) goto cleanup;
        }
    }
    
    // --- Handle result ---
    if (config.are_results_on_device) {
        d_result = reinterpret_cast<G1Projective*>(results);
    } else {
        err = cudaMalloc(&d_result, sizeof(G1Projective));
        if (err != cudaSuccess) goto cleanup;
        allocated_result = true;
    }
    
    // --- Create modified config for internal MSM ---
    {
        icicle::MSMConfig internal_cfg = config;
        internal_cfg.are_scalars_on_device = true;
        internal_cfg.are_points_on_device = true;
        internal_cfg.are_results_on_device = true;
        // Scalars remain in standard form - MSM needs actual bit values
        internal_cfg.are_scalars_montgomery_form = false;  
        // Points are converted to Montgomery for field arithmetic
        internal_cfg.are_points_montgomery_form = true;
        
        // Call our MSM implementation
        err = msm::msm_cuda<Fr, G1Affine, G1Projective>(
            d_scalars, d_bases, msm_size, internal_cfg, d_result);
        if (err != cudaSuccess) goto cleanup;
    }
    
    // --- Convert result from Jacobian/Montgomery to Standard/Standard form ---
    // Our MSM uses Jacobian projective (x=X/Z², y=Y/Z³) in Montgomery form
    // ICICLE expects standard projective (x=X/Z, y=Y/Z) in standard form
    {
        jacobian_to_standard_projective_kernel<<<1, 1, 0, stream>>>(d_result, 1);
        err = cudaGetLastError();
        if (err != cudaSuccess) goto cleanup;
    }
    
    // --- Copy result back if needed ---
    if (!config.are_results_on_device) {
        err = cudaMemcpyAsync(results, d_result, sizeof(G1Projective),
                              cudaMemcpyDeviceToHost, stream);
        if (err != cudaSuccess) goto cleanup;
    }
    
    // Synchronize if not async
    if (!config.is_async) {
        err = cudaStreamSynchronize(stream);
        if (err != cudaSuccess) goto cleanup;
    }
    
cleanup:
    if (allocated_scalars && d_scalars) cudaFree(d_scalars);
    if (allocated_bases && d_bases) cudaFree(d_bases);
    if (allocated_result && d_result) cudaFree(d_result);
    
    return (err == cudaSuccess) ? icicle::eIcicleError::SUCCESS : icicle::eIcicleError::UNKNOWN_ERROR;
}

/**
 * @brief G1 MSM precompute bases wrapper
 * 
 * Our MSM doesn't require precomputation - this is a no-op that just
 * copies data if needed for ICICLE API compatibility.
 */
static icicle::eIcicleError msm_precompute_bases_cuda_impl(
    const icicle::Device& device,
    const icicle::icicle_affine_t* input_bases,
    int bases_size,
    const icicle::MSMConfig& config,
    icicle::icicle_affine_t* output_bases)
{
    (void)device; // Unused - we always use CUDA context
    
    // No-op: Our MSM doesn't use precomputation, but copy if different buffers
    if (output_bases != input_bases && config.are_points_on_device) {
        // Use Icicle's type size for the copy since that's what the caller expects
        // Icicle affine_t is 2 * 48 bytes = 96 bytes
        constexpr size_t ICICLE_AFFINE_SIZE = 96;  // 2 * 12 * 4 bytes (2 fields of 12 x 32-bit limbs)
        
        cudaStream_t stream = static_cast<cudaStream_t>(config.stream);
        cudaError_t err = cudaMemcpyAsync(
            output_bases, input_bases, 
            bases_size * ICICLE_AFFINE_SIZE,
            cudaMemcpyDeviceToDevice, stream
        );
        if (!config.is_async) cudaStreamSynchronize(stream);
        return (err == cudaSuccess) ? icicle::eIcicleError::SUCCESS : icicle::eIcicleError::COPY_FAILED;
    }
    return icicle::eIcicleError::SUCCESS;
}

// =============================================================================
// G2 MSM Implementation
// =============================================================================

/**
 * @brief G2 MSM wrapper for ICICLE backend
 * 
 * Handles Montgomery form conversions and coordinate system conversions:
 * - Input: G2 affine points (Fq2 coordinates) + scalars (standard or Montgomery form per config)
 * - Internal: Montgomery form for point field arithmetic, standard form for scalars
 * - Output: Standard form standard projective (x=X/Z, y=Y/Z)
 */
static icicle::eIcicleError msm_g2_cuda_impl(
    const icicle::Device& device,
    const icicle::icicle_scalar_t* scalars,
    const icicle::icicle_g2_affine_t* bases,
    int msm_size,
    const icicle::MSMConfig& config,
    icicle::icicle_g2_projective_t* results)
{
    (void)device; // Unused - we always use CUDA context
    
    if (msm_size == 0) {
        return icicle::eIcicleError::SUCCESS;
    }
    
    cudaStream_t stream = static_cast<cudaStream_t>(config.stream);
    cudaError_t err;
    
    // We need to convert inputs to appropriate form and outputs from Montgomery form.
    // To avoid modifying input data, we allocate temporary buffers when needed.
    Fr* d_scalars = nullptr;
    G2Affine* d_bases = nullptr;
    G2Projective* d_result = nullptr;
    
    bool allocated_scalars = false;
    bool allocated_bases = false;
    bool allocated_result = false;
    
    // --- Handle scalars ---
    // MSM extracts bits from scalars to determine bucket indices.
    // Scalars must be in STANDARD form so bits represent actual scalar value.
    // If are_scalars_montgomery_form=true, we need to convert from Montgomery to standard.
    if (config.are_scalars_on_device) {
        if (config.are_scalars_montgomery_form) {
            // Scalars are on device in Montgomery form - need to convert to standard
            err = cudaMalloc(&d_scalars, msm_size * sizeof(Fr));
            if (err != cudaSuccess) goto cleanup;
            allocated_scalars = true;
            
            err = cudaMemcpyAsync(d_scalars, scalars, msm_size * sizeof(Fr),
                                  cudaMemcpyDeviceToDevice, stream);
            if (err != cudaSuccess) goto cleanup;
            
            // Convert from Montgomery to standard form
            constexpr int threads = gpu::GPUConfig::DEFAULT_COMPUTE_THREADS;
            int blocks = (msm_size + threads - 1) / threads;
            scalars_from_montgomery_kernel<<<blocks, threads, 0, stream>>>(d_scalars, msm_size);
            err = cudaGetLastError();
            if (err != cudaSuccess) goto cleanup;
        } else {
            // Scalars already in standard form - use directly
            d_scalars = const_cast<Fr*>(reinterpret_cast<const Fr*>(scalars));
        }
    } else {
        // Data on host - copy to device
        err = cudaMalloc(&d_scalars, msm_size * sizeof(Fr));
        if (err != cudaSuccess) goto cleanup;
        allocated_scalars = true;
        
        err = cudaMemcpyAsync(d_scalars, scalars, msm_size * sizeof(Fr),
                              cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess) goto cleanup;
        
        if (config.are_scalars_montgomery_form) {
            // Convert from Montgomery to standard form
            constexpr int threads_host = gpu::GPUConfig::DEFAULT_COMPUTE_THREADS;
            int blocks = (msm_size + threads_host - 1) / threads_host;
            scalars_from_montgomery_kernel<<<blocks, threads_host, 0, stream>>>(d_scalars, msm_size);
            err = cudaGetLastError();
            if (err != cudaSuccess) goto cleanup;
        }
    }
    
    // --- Handle bases (points) ---
    if (config.are_points_on_device) {
        if (!config.are_points_montgomery_form) {
            err = cudaMalloc(&d_bases, msm_size * sizeof(G2Affine));
            if (err != cudaSuccess) goto cleanup;
            allocated_bases = true;
            
            err = cudaMemcpyAsync(d_bases, bases, msm_size * sizeof(G2Affine),
                                  cudaMemcpyDeviceToDevice, stream);
            if (err != cudaSuccess) goto cleanup;
            
            // Convert to Montgomery form
            constexpr int threads = gpu::GPUConfig::DEFAULT_COMPUTE_THREADS;
            int blocks = (msm_size + threads - 1) / threads;
            g2_points_to_montgomery_kernel<<<blocks, threads, 0, stream>>>(d_bases, msm_size);
            err = cudaGetLastError();
            if (err != cudaSuccess) goto cleanup;
        } else {
            d_bases = const_cast<G2Affine*>(reinterpret_cast<const G2Affine*>(bases));
        }
    } else {
        err = cudaMalloc(&d_bases, msm_size * sizeof(G2Affine));
        if (err != cudaSuccess) goto cleanup;
        allocated_bases = true;
        
        err = cudaMemcpyAsync(d_bases, bases, msm_size * sizeof(G2Affine),
                              cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess) goto cleanup;
        
        if (!config.are_points_montgomery_form) {
            constexpr int threads = gpu::GPUConfig::DEFAULT_COMPUTE_THREADS;
            int blocks = (msm_size + threads - 1) / threads;
            g2_points_to_montgomery_kernel<<<blocks, threads, 0, stream>>>(d_bases, msm_size);
            err = cudaGetLastError();
            if (err != cudaSuccess) goto cleanup;
        }
    }
    
    // --- Handle result ---
    if (config.are_results_on_device) {
        d_result = reinterpret_cast<G2Projective*>(results);
    } else {
        err = cudaMalloc(&d_result, sizeof(G2Projective));
        if (err != cudaSuccess) goto cleanup;
        allocated_result = true;
    }
    
    // --- Create modified config for internal MSM ---
    {
        icicle::MSMConfig internal_cfg = config;
        internal_cfg.are_scalars_on_device = true;
        internal_cfg.are_points_on_device = true;
        internal_cfg.are_results_on_device = true;
        // Scalars remain in standard form - MSM needs actual bit values
        internal_cfg.are_scalars_montgomery_form = false;  
        // Points are converted to Montgomery for field arithmetic
        internal_cfg.are_points_montgomery_form = true;
        
        // Call our MSM implementation (G2 version using template)
        err = msm::msm_cuda<Fr, G2Affine, G2Projective>(
            d_scalars, d_bases, msm_size, internal_cfg, d_result);
        if (err != cudaSuccess) goto cleanup;
    }
    
    // --- Convert result from Jacobian/Montgomery to Standard/Standard form ---
    // Our MSM uses Jacobian projective (x=X/Z², y=Y/Z³) in Montgomery form
    // ICICLE expects standard projective (x=X/Z, y=Y/Z) in standard form
    {
        g2_jacobian_to_standard_projective_kernel<<<1, 1, 0, stream>>>(d_result, 1);
        err = cudaGetLastError();
        if (err != cudaSuccess) goto cleanup;
    }
    
    // --- Copy result back if needed ---
    if (!config.are_results_on_device) {
        err = cudaMemcpyAsync(results, d_result, sizeof(G2Projective),
                              cudaMemcpyDeviceToHost, stream);
        if (err != cudaSuccess) goto cleanup;
    }
    
    // Synchronize if not async
    if (!config.is_async) {
        err = cudaStreamSynchronize(stream);
        if (err != cudaSuccess) goto cleanup;
    }
    
cleanup:
    if (allocated_scalars && d_scalars) cudaFree(d_scalars);
    if (allocated_bases && d_bases) cudaFree(d_bases);
    if (allocated_result && d_result) cudaFree(d_result);
    
    return (err == cudaSuccess) ? icicle::eIcicleError::SUCCESS : icicle::eIcicleError::UNKNOWN_ERROR;
}

/**
 * @brief G2 MSM precompute bases wrapper
 * 
 * Our MSM doesn't require precomputation - this is a no-op that just
 * copies data if needed for ICICLE API compatibility.
 */
static icicle::eIcicleError msm_g2_precompute_bases_cuda_impl(
    const icicle::Device& device,
    const icicle::icicle_g2_affine_t* input_bases,
    int bases_size,
    const icicle::MSMConfig& config,
    icicle::icicle_g2_affine_t* output_bases)
{
    (void)device; // Unused - we always use CUDA context
    
    // No-op: Our MSM doesn't use precomputation, but copy if different buffers
    if (output_bases != input_bases && config.are_points_on_device) {
        // G2 affine size: 2 fields * 2 components * 48 bytes = 192 bytes
        constexpr size_t ICICLE_G2_AFFINE_SIZE = 192;
        
        cudaStream_t stream = static_cast<cudaStream_t>(config.stream);
        cudaError_t err = cudaMemcpyAsync(
            output_bases, input_bases, 
            bases_size * ICICLE_G2_AFFINE_SIZE,
            cudaMemcpyDeviceToDevice, stream
        );
        if (!config.is_async) cudaStreamSynchronize(stream);
        return (err == cudaSuccess) ? icicle::eIcicleError::SUCCESS : icicle::eIcicleError::COPY_FAILED;
    }
    return icicle::eIcicleError::SUCCESS;
}

// =============================================================================
// ICICLE Backend Registration (must be at file scope for static initialization)
// =============================================================================

// Register our MSM implementation with the "CUDA" device type
// This uses Icicle's macro which creates a static initializer that runs at load time

// G1 MSM registration
REGISTER_MSM_PRE_COMPUTE_BASES_BACKEND("CUDA", msm_precompute_bases_cuda_impl);
REGISTER_MSM_BACKEND("CUDA", msm_cuda_impl);

// G2 MSM registration
REGISTER_MSM_G2_PRE_COMPUTE_BASES_BACKEND("CUDA", msm_g2_precompute_bases_cuda_impl);
REGISTER_MSM_G2_BACKEND("CUDA", msm_g2_cuda_impl);

// =============================================================================
// Test C APIs (used by cuda-backend tests)
// =============================================================================

extern "C" {

/**
 * @brief G1 MSM entry point for tests
 * 
 * Note: This API expects data already in Montgomery form.
 * For production use via ICICLE Rust bindings, use the registered backend instead.
 */
icicle::eIcicleError bls12_381_g1_msm_cuda(
    const Fr* scalars,
    const G1Affine* bases,
    int msm_size,
    const icicle::MSMConfig* config,
    G1Projective* result
) {
    cudaError_t err = msm::msm_cuda<Fr, G1Affine, G1Projective>(
        scalars, bases, msm_size, *config, result
    );
    return (err == cudaSuccess) ? icicle::eIcicleError::SUCCESS : icicle::eIcicleError::UNKNOWN_ERROR;
}

/**
 * @brief G2 MSM entry point for tests
 */
icicle::eIcicleError bls12_381_g2_msm_cuda(
    const Fr* scalars,
    const G2Affine* bases,
    int msm_size,
    const icicle::MSMConfig* config,
    G2Projective* result
) {
    cudaError_t err = msm::msm_cuda<Fr, G2Affine, G2Projective>(
        scalars, bases, msm_size, *config, result
    );
    return (err == cudaSuccess) ? icicle::eIcicleError::SUCCESS : icicle::eIcicleError::UNKNOWN_ERROR;
}

} // extern "C"