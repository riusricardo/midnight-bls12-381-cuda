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
 * @file ntt_kernels.cu
 * @brief NTT and Field Backend Implementation
 * 
 * This file contains the complete NTT implementation for the BLS12-381 scalar field.
 * 
 * ARCHITECTURE:
 * =============
 * CUDA static libraries require kernels to be defined in the same compilation unit
 * that calls them. This file is self-contained with:
 * 
 * 1. NTT Kernels (defined here, called here):
 *    - ntt_bit_reverse_kernel: Bit-reversal permutation
 *    - ntt_butterfly_kernel: Single Radix-2 butterfly stage
 *    - ntt_scale_kernel: Scaling for inverse NTT
 *    - ntt_shared_memory_kernel: Optimized for small sizes (≤256)
 *    - intt_shared_memory_kernel: Inverse version of above
 *    - ntt_butterfly_fused_2stage_kernel: Fused 2-stage for larger sizes

 * 
 * 2. Domain Management:
 *    - Static storage for Domain<Fr> instances
 *    - Twiddle factor precomputation kernels
 * 
 * 3. API Functions:
 *    - ntt_cuda: Main NTT entry point
 *    - coset_ntt_cuda: Coset NTT for polynomial evaluation
 *    - init_domain_cuda: Initialize twiddle factors
 *    - release_domain_cuda: Free resources
 * 
 * OPTIMIZATIONS:
 * ==============
 * - Small NTT (≤256): Uses shared memory kernel - entire transform in SRAM
 * - Large NTT (>256): Uses fused 2-stage butterfly - 50% fewer kernel launches
 * - All operations use Montgomery form for efficient modular arithmetic
 */

#include "field.cuh"
#include "ntt.cuh"
#include "icicle_types.cuh"
#include "gpu_config.cuh"
#include <cuda_runtime.h>
#include <iostream>

using namespace bls12_381;
using namespace ntt;
using namespace gpu;

// =============================================================================
// Global NTT Domain Registry - Storage Definitions
// =============================================================================

// Define static members for Domain<Fr> - must be outside namespace
namespace ntt {
    template<>
    Domain<Fr>* Domain<Fr>::domains[MAX_LOG_DOMAIN_SIZE] = {nullptr};

    template<>
    std::mutex Domain<Fr>::domains_mutex{};
}

template<>
Domain<Fr>* Domain<Fr>::get_domain(int log_size) {
    if (log_size >= MAX_LOG_DOMAIN_SIZE) return nullptr;
    std::lock_guard<std::mutex> lock(domains_mutex);
    return domains[log_size];
}

template<>
void Domain<Fr>::set_domain(int log_size, Domain<Fr>* domain) {
    if (log_size >= MAX_LOG_DOMAIN_SIZE) return;
    std::lock_guard<std::mutex> lock(domains_mutex);
    domains[log_size] = domain;
}

// =============================================================================
// NTT Core Implementation - Kernels and Host Functions
// =============================================================================

namespace ntt {

// =============================================================================
// NTT Kernels (must be in same compilation unit for proper linking)
// =============================================================================

/**
 * @brief Bit-reverse permutation kernel for Fr
 */
__global__ void ntt_bit_reverse_kernel(
    Fr* output,
    const Fr* input,
    int n,
    int log_n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= n) return;
    
    // Compute bit-reversed index
    unsigned int rev = 0;
    unsigned int temp = idx;
    for (int i = 0; i < log_n; i++) {
        rev = (rev << 1) | (temp & 1);
        temp >>= 1;
    }
    
    output[rev] = input[idx];
}

/**
 * @brief Single butterfly stage kernel for Fr
 */
__global__ void ntt_butterfly_kernel(
    Fr* data,
    const Fr* twiddles,
    int n,
    int m,           // Group size = 2^(stage+1)
    int half_m       // Half group size = 2^stage
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int num_butterflies = n / 2;
    
    if (idx >= num_butterflies) return;
    
    // Determine which butterfly group and position within group
    int group = idx / half_m;
    int pos = idx % half_m;
    
    // Indices of the two elements in the butterfly
    int i = group * m + pos;
    int j = i + half_m;
    
    // Twiddle factor index
    int twiddle_idx = pos * (n / m);
    
    Fr twiddle = twiddles[twiddle_idx];
    
    // Butterfly operation
    Fr u = data[i];
    Fr v = data[j] * twiddle;
    
    data[i] = u + v;
    data[j] = u - v;
}

/**
 * @brief Scale kernel for Fr (inverse NTT)
 */
__global__ void ntt_scale_kernel(
    Fr* output,
    const Fr* input,
    Fr scale_factor,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    output[idx] = input[idx] * scale_factor;
}

// =============================================================================
// Optimized NTT Kernels
// =============================================================================

/**
 * @brief Shared memory NTT kernel for small transforms
 * 
 * Processes entire NTT in shared memory when size <= 1024 (capability).
 * NOTE: Currently used only for sizes <= 256 (threshold) because fused
 * butterfly kernels achieve 3x better throughput for sizes 64-256.
 * 
 * This dramatically reduces global memory traffic - only 2 accesses per element
 * instead of 2*log_size accesses with the standard approach.
 * 
 * @param data Input/output array (in-place transform)
 * @param twiddles Precomputed twiddle factors
 * @param size NTT size (must be power of 2, <= 1024)
 * @param log_size log2(size)
 */
__global__ void ntt_shared_memory_kernel(
    Fr* data,
    const Fr* twiddles,
    int size,
    int log_size
) {
    extern __shared__ Fr sdata[];
    
    int tid = threadIdx.x;
    int block_offset = blockIdx.x * size;
    
    // Load data into shared memory with bit-reversal permutation
    if (tid < size) {
        unsigned int rev = 0;
        unsigned int n = tid;
        for (int i = 0; i < log_size; i++) {
            rev = (rev << 1) | (n & 1);
            n >>= 1;
        }
        sdata[tid] = data[block_offset + rev];
    }
    __syncthreads();
    
    // All butterfly stages in shared memory
    for (int s = 1; s <= log_size; s++) {
        int m = 1 << s;
        int half_m = m / 2;
        
        if (tid < size / 2) {
            int group = tid / half_m;
            int pos = tid % half_m;
            
            int i0 = group * m + pos;
            int i1 = i0 + half_m;
            
            int twiddle_idx = pos * (size / m);
            Fr omega = twiddles[twiddle_idx];
            
            Fr u = sdata[i0];
            Fr t = sdata[i1] * omega;
            
            sdata[i0] = u + t;
            sdata[i1] = u - t;
        }
        __syncthreads();
    }
    
    // Write back to global memory
    if (tid < size) {
        data[block_offset + tid] = sdata[tid];
    }
}

/**
 * @brief Batched small NTT kernel - processes multiple small NTTs in parallel
 * 
 * For very small NTTs (size <= 64), processing them individually
 * wastes GPU resources. This kernel batches multiple small NTTs into a single
 * block, dramatically improving throughput.
 * 
 * Each warp processes one complete small NTT using warp-level synchronization.
 * This avoids expensive __syncthreads() and maximizes parallelism.
 * 
 * @param data Input/output array [batch_size * ntt_size]
 * @param twiddles Precomputed twiddle factors
 * @param ntt_size Size of each individual NTT (must be <= 32)
 * @param log_size log2(ntt_size)
 * @param batch_size Number of NTTs to process
 */
__global__ void ntt_batch_warp_kernel(
    Fr* data,
    const Fr* twiddles,
    int ntt_size,
    int log_size,
    int batch_size
) {
    // Each warp (32 threads) processes one NTT
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane_id = threadIdx.x & 31;
    
    if (warp_id >= batch_size) return;
    
    // Offset to this NTT's data
    Fr* ntt_data = data + warp_id * ntt_size;
    
    // Load element with bit-reversal (each lane loads one element if lane < ntt_size)
    Fr val = Fr::zero();
    if (lane_id < ntt_size) {
        // Compute bit-reversed index
        unsigned int rev = 0;
        unsigned int temp = lane_id;
        for (int i = 0; i < log_size; i++) {
            rev = (rev << 1) | (temp & 1);
            temp >>= 1;
        }
        val = ntt_data[rev];
    }
    
    // Butterfly stages using warp shuffle
    for (int s = 1; s <= log_size; s++) {
        int m = 1 << s;
        int half_m = m / 2;
        
        // Determine butterfly pair
        int pos = lane_id % m;
        int partner = (pos < half_m) ? (lane_id + half_m) : (lane_id - half_m);
        
        // Get partner's value using warp shuffle
        Fr partner_val = Fr::zero();
        // Shuffle each limb separately
        for (int l = 0; l < Fr::LIMBS; l++) {
            partner_val.limbs[l] = __shfl_sync(0xFFFFFFFF, val.limbs[l], partner);
        }
        
        // Compute twiddle index
        int twiddle_pos = (pos < half_m) ? pos : (pos - half_m);
        int twiddle_idx = twiddle_pos * (ntt_size / m);
        Fr omega = (lane_id < ntt_size && twiddle_idx < ntt_size) ? twiddles[twiddle_idx] : Fr::one();
        
        // Butterfly operation
        if (lane_id < ntt_size) {
            if (pos < half_m) {
                // Upper element: val = val + partner * omega
                val = val + partner_val * omega;
            } else {
                // Lower element: val = partner - val * omega (but val was upper, partner is lower)
                // Actually: lower gets (upper - lower*omega), but we have lower in val
                // Need to recompute: result = shuffle(upper) - val * omega
                Fr upper_val = partner_val; // partner is the upper element
                val = upper_val - val * omega;
            }
        }
        __syncwarp();
    }
    
    // Store result
    if (lane_id < ntt_size) {
        ntt_data[lane_id] = val;
    }
}

/**
 * @brief Inverse batch warp-based NTT kernel
 * 
 * Like ntt_batch_warp_kernel but for inverse transform (DIF structure).
 * Each warp processes one complete small inverse NTT.
 * 
 * @param data Input/output array [batch_size * ntt_size]
 * @param inv_twiddles Precomputed inverse twiddle factors
 * @param scale_factor 1/n for normalization
 * @param ntt_size Size of each individual NTT (must be <= 32)
 * @param log_size log2(ntt_size)
 * @param batch_size Number of NTTs to process
 */
__global__ void intt_batch_warp_kernel(
    Fr* data,
    const Fr* inv_twiddles,
    Fr scale_factor,
    int ntt_size,
    int log_size,
    int batch_size
) {
    // Each warp (32 threads) processes one inverse NTT
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane_id = threadIdx.x & 31;
    
    if (warp_id >= batch_size) return;
    
    // Offset to this NTT's data
    Fr* ntt_data = data + warp_id * ntt_size;
    
    // Load element (no bit-reversal for DIF input)
    Fr val = Fr::zero();
    if (lane_id < ntt_size) {
        val = ntt_data[lane_id];
    }
    
    // Butterfly stages in reverse order (DIF structure for inverse)
    for (int s = log_size; s >= 1; s--) {
        int m = 1 << s;
        int half_m = m / 2;
        
        // Determine butterfly pair
        int pos = lane_id % m;
        int partner = (pos < half_m) ? (lane_id + half_m) : (lane_id - half_m);
        
        // Get partner's value using warp shuffle
        Fr partner_val = Fr::zero();
        for (int l = 0; l < Fr::LIMBS; l++) {
            partner_val.limbs[l] = __shfl_sync(0xFFFFFFFF, val.limbs[l], partner);
        }
        
        // Compute twiddle index
        int twiddle_pos = (pos < half_m) ? pos : (pos - half_m);
        int twiddle_idx = twiddle_pos * (ntt_size / m);
        Fr omega_inv = (lane_id < ntt_size && twiddle_idx < ntt_size) ? inv_twiddles[twiddle_idx] : Fr::one();
        
        // DIF butterfly: add/sub first, then multiply lower by inverse twiddle
        if (lane_id < ntt_size) {
            if (pos < half_m) {
                // Upper element: val = val + partner
                val = val + partner_val;
            } else {
                // Lower element: val = (partner - val) * omega_inv
                val = (partner_val - val) * omega_inv;
            }
        }
        __syncwarp();
    }
    
    // Apply bit-reversal and scale on output
    if (lane_id < ntt_size) {
        unsigned int rev = 0;
        unsigned int temp = lane_id;
        for (int i = 0; i < log_size; i++) {
            rev = (rev << 1) | (temp & 1);
            temp >>= 1;
        }
        ntt_data[rev] = val * scale_factor;
    }
}

/**
 * @brief Inverse shared memory NTT kernel
 * 
 * Like ntt_shared_memory_kernel but for inverse transform.
 * Uses inverse twiddles and scales by 1/n at the end.
 */
__global__ void intt_shared_memory_kernel(
    Fr* data,
    const Fr* inv_twiddles,
    Fr scale_factor,
    int size,
    int log_size
) {
    extern __shared__ Fr sdata[];
    
    int tid = threadIdx.x;
    int block_offset = blockIdx.x * size;
    
    // Load data into shared memory (no bit-reversal for inverse)
    if (tid < size) {
        sdata[tid] = data[block_offset + tid];
    }
    __syncthreads();
    
    // Butterfly stages in reverse order (DIF structure for inverse)
    for (int s = log_size; s >= 1; s--) {
        int m = 1 << s;
        int half_m = m / 2;
        
        if (tid < size / 2) {
            int group = tid / half_m;
            int pos = tid % half_m;
            
            int i0 = group * m + pos;
            int i1 = i0 + half_m;
            
            int twiddle_idx = pos * (size / m);
            Fr omega_inv = inv_twiddles[twiddle_idx];
            
            // DIF butterfly: add/sub first, then multiply
            Fr u = sdata[i0];
            Fr v = sdata[i1];
            
            sdata[i0] = u + v;
            sdata[i1] = (u - v) * omega_inv;
        }
        __syncthreads();
    }
    
    // DIF output is bit-reversed; bit-reverse and scale on write
    if (tid < size) {
        unsigned int rev = 0;
        unsigned int n = tid;
        for (int i = 0; i < log_size; i++) {
            rev = (rev << 1) | (n & 1);
            n >>= 1;
        }
        data[block_offset + rev] = sdata[tid] * scale_factor;
    }
}

/**
 * @brief Fused 2-stage butterfly kernel
 * 
 * Performs two consecutive butterfly stages in one kernel launch.
 * Reduces kernel launch overhead by 50% and halves global memory traffic
 * compared to running two separate butterfly kernels.
 * 
 * Each thread processes 4 elements through 2 stages.
 * 
 * @param data Input/output array
 * @param twiddles Precomputed twiddle factors
 * @param n Total NTT size
 * @param stride First stage half_m value (stride of first stage)
 */
__global__ void ntt_butterfly_fused_2stage_kernel(
    Fr* data,
    const Fr* twiddles,
    int n,
    int stride  // half_m of the first stage being processed
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int num_quads = n / 4;
    
    if (idx >= num_quads) return;
    
    // Each thread handles a group of 4 consecutive elements
    // Group size for fused 2 stages = 4 * stride
    int quad_m = 4 * stride;
    int group = idx / stride;
    int pos = idx % stride;
    
    int base = group * quad_m + pos;
    
    int i0 = base;
    int i1 = base + stride;
    int i2 = base + 2 * stride;
    int i3 = base + 3 * stride;
    
    // Load 4 elements
    Fr x0 = data[i0];
    Fr x1 = data[i1];
    Fr x2 = data[i2];
    Fr x3 = data[i3];
    
    // ---------------------------------------------------------
    // Stage 1 (group size = 2*stride, half_m = stride)
    // ---------------------------------------------------------
    // Pairs: (x0, x1) and (x2, x3)
    int tw_idx_1 = pos * (n / (2 * stride));
    Fr w1 = twiddles[tw_idx_1];
    
    // Butterfly on (x0, x1)
    Fr v1 = x1 * w1;
    Fr u1 = x0;
    x0 = u1 + v1;
    x1 = u1 - v1;
    
    // Butterfly on (x2, x3) with same twiddle
    Fr v2 = x3 * w1;
    Fr u2 = x2;
    x2 = u2 + v2;
    x3 = u2 - v2;
    
    // ---------------------------------------------------------
    // Stage 2 (group size = 4*stride, half_m = 2*stride)
    // ---------------------------------------------------------
    // Pairs: (x0, x2) and (x1, x3)
    int tw_idx_2a = pos * (n / (4 * stride));
    Fr w2a = twiddles[tw_idx_2a];
    
    // Butterfly on (x0, x2)
    Fr v3 = x2 * w2a;
    Fr u3 = x0;
    x0 = u3 + v3;
    x2 = u3 - v3;
    
    // Butterfly on (x1, x3)
    int tw_idx_2b = (pos + stride) * (n / (4 * stride));
    Fr w2b = twiddles[tw_idx_2b];
    
    Fr v4 = x3 * w2b;
    Fr u4 = x1;
    x1 = u4 + v4;
    x3 = u4 - v4;
    
    // Store 4 elements
    data[i0] = x0;
    data[i1] = x1;
    data[i2] = x2;
    data[i3] = x3;
}

/**
 * @brief Inverse fused 2-stage butterfly kernel (DIF structure)
 * 
 * For inverse NTT, we need DIF structure: add/sub first, then multiply by twiddle.
 * This is the inverse of the forward DIT kernel.
 * 
 * We process stages in REVERSE order compared to forward:
 *   - First: stage s (higher) with group_size = 4*stride, half_m = 2*stride
 *   - Then: stage s-1 (lower) with group_size = 2*stride, half_m = stride
 * 
 * @param data Input/output array
 * @param inv_twiddles Precomputed inverse twiddle factors
 * @param n Total NTT size
 * @param stride half_m of the LOWER stage = 2^(s-2)
 */
__global__ void intt_butterfly_fused_2stage_kernel(
    Fr* data,
    const Fr* inv_twiddles,
    int n,
    int stride  // half_m of the lower stage (same as forward kernel)
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int num_quads = n / 4;
    
    if (idx >= num_quads) return;
    
    // Each thread handles a group of 4 consecutive elements
    // Group size for fused 2 stages = 4 * stride (same indexing as forward)
    int quad_m = 4 * stride;
    int group = idx / stride;
    int pos = idx % stride;
    
    int base = group * quad_m + pos;
    
    int i0 = base;
    int i1 = base + stride;
    int i2 = base + 2 * stride;
    int i3 = base + 3 * stride;
    
    // Load 4 elements
    Fr x0 = data[i0];
    Fr x1 = data[i1];
    Fr x2 = data[i2];
    Fr x3 = data[i3];
    
    // ---------------------------------------------------------
    // Stage 1 (HIGHER stage first in DIF): group size = 4*stride, half_m = 2*stride
    // ---------------------------------------------------------
    // DIF butterfly: add/sub first, then multiply
    // Pairs: (x0, x2) and (x1, x3) - with distance 2*stride
    
    int tw_idx_1a = pos * (n / (4 * stride));
    Fr w1a = inv_twiddles[tw_idx_1a];
    
    // DIF butterfly on (x0, x2)
    Fr u0 = x0;
    Fr u2 = x2;
    x0 = u0 + u2;
    x2 = (u0 - u2) * w1a;
    
    // DIF butterfly on (x1, x3)
    int tw_idx_1b = (pos + stride) * (n / (4 * stride));
    Fr w1b = inv_twiddles[tw_idx_1b];
    
    Fr u1 = x1;
    Fr u3 = x3;
    x1 = u1 + u3;
    x3 = (u1 - u3) * w1b;
    
    // ---------------------------------------------------------
    // Stage 2 (LOWER stage in DIF): group size = 2*stride, half_m = stride
    // ---------------------------------------------------------
    // Pairs: (x0, x1) and (x2, x3) - with distance stride
    
    int tw_idx_2 = pos * (n / (2 * stride));
    Fr w2 = inv_twiddles[tw_idx_2];
    
    // DIF butterfly on (x0, x1)
    Fr v0 = x0;
    Fr v1 = x1;
    x0 = v0 + v1;
    x1 = (v0 - v1) * w2;
    
    // DIF butterfly on (x2, x3) with same twiddle
    Fr v2 = x2;
    Fr v3 = x3;
    x2 = v2 + v3;
    x3 = (v2 - v3) * w2;
    
    // Store 4 elements
    data[i0] = x0;
    data[i1] = x1;
    data[i2] = x2;
    data[i3] = x3;
}

/**
 * @brief Inverse single butterfly kernel (DIF structure)
 */
__global__ void intt_butterfly_kernel(
    Fr* data,
    const Fr* inv_twiddles,
    int n,
    int m,
    int half_m
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int num_pairs = n / 2;
    
    if (idx >= num_pairs) return;
    
    int group = idx / half_m;
    int pos = idx % half_m;
    
    int i0 = group * m + pos;
    int i1 = i0 + half_m;
    
    int twiddle_idx = pos * (n / m);
    Fr omega_inv = inv_twiddles[twiddle_idx];
    
    // DIF butterfly: add/sub first, then multiply
    Fr u = data[i0];
    Fr v = data[i1];
    
    data[i0] = u + v;
    data[i1] = (u - v) * omega_inv;
}

/**
 * @brief Forward NTT implementation with optimizations
 * 
 * Uses shared memory kernel for small sizes, fused 2-stage butterflies for larger sizes.
 */
template<typename F>
eIcicleError ntt_forward_impl(
    const F* input,
    int size,
    const NTTConfig<F>& config,
    F* output
) {
    if (size == 0) return eIcicleError::SUCCESS;
    if ((size & (size - 1)) != 0) return eIcicleError::INVALID_ARGUMENT;
    
    int log_size = 0;
    while ((1 << log_size) < size) log_size++;
    
    // Get domain
    Domain<F>* domain = Domain<F>::get_domain(log_size);
    if (!domain) {
        return eIcicleError::INVALID_ARGUMENT;
    }
    
    // Validate domain is fully initialized (twiddles allocated)
    if (!domain->twiddles || !domain->inv_twiddles) {
        return eIcicleError::INVALID_ARGUMENT;
    }
    
    // Allocate working buffer
    F* d_data = nullptr;
    bool need_alloc = !config.are_inputs_on_device;
    
    if (need_alloc) {
        CUDA_CHECK(cudaMalloc(&d_data, size * sizeof(F)));
        CUDA_CHECK(cudaMemcpy(d_data, input, size * sizeof(F), cudaMemcpyHostToDevice));
    } else if (input != output) {
        CUDA_CHECK(cudaMalloc(&d_data, size * sizeof(F)));
        CUDA_CHECK(cudaMemcpy(d_data, input, size * sizeof(F), cudaMemcpyDeviceToDevice));
    } else {
        d_data = const_cast<F*>(input);
    }
    
    const auto& gpu = GPUConfig::get();
    
    // Get NTT launch configuration - determines strategy and thread count
    auto ntt_cfg = get_ntt_launch_config(size, sizeof(F));
    
    // =========================================================================
    // STRATEGY:
    // - Size <= 32: Use warp-based kernel (when batched) - highest throughput
    // - Size <= 256: Use shared memory kernel - good for medium sizes
    // - Size > 256: Use fused butterfly kernels - best for large sizes
    // 
    // The shared memory kernel is kept at 256 threshold because:
    // 1. Butterfly fused kernel achieves 7.33% SM throughput vs 2.32% for shared
    // 2. For sizes 64-256, butterfly is 3× faster
    // =========================================================================
    
    if (ntt_cfg.strategy == NTTStrategy::SHARED_MEMORY) {
        // Shared memory approach: single kernel does bit-reversal + all butterflies
        ntt_shared_memory_kernel<<<1, size, ntt_cfg.shared_mem>>>(
            d_data, domain->twiddles, size, log_size
        );
        CUDA_CHECK(cudaDeviceSynchronize());
    } else {
        // Butterfly approach for larger sizes - higher SM throughput
        const int threads = ntt_cfg.threads;
        F* d_temp;
        CUDA_CHECK(cudaMalloc(&d_temp, size * sizeof(F)));
        
        int blocks = (size + threads - 1) / threads;
        ntt_bit_reverse_kernel<<<blocks, threads>>>(d_temp, d_data, size, log_size);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        CUDA_CHECK(cudaMemcpy(d_data, d_temp, size * sizeof(F), cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaFree(d_temp));
        
        // =====================================================================
        // Use fused 2-stage butterflies where possible
        // =====================================================================
        int s = 1;
        while (s + 1 <= log_size) {
            // Fuse stages s and s+1
            // stride = half_m of stage s = 2^(s-1)
            int stride = 1 << (s - 1);
            
            blocks = (size / 4 + threads - 1) / threads;
            ntt_butterfly_fused_2stage_kernel<<<blocks, threads>>>(
                d_data, domain->twiddles, size, stride
            );
            CUDA_CHECK(cudaDeviceSynchronize());
            
            s += 2;  // Advance by 2 stages
        }
        
        // Handle remaining stage if log_size is odd
        if (s == log_size) {
            int m = 1 << s;
            int half_m = m / 2;
            
            blocks = (size / 2 + threads - 1) / threads;
            ntt_butterfly_kernel<<<blocks, threads>>>(
                d_data, domain->twiddles, size, m, half_m
            );
            CUDA_CHECK(cudaDeviceSynchronize());
        }
    }
    
    // Copy output
    if (config.are_outputs_on_device) {
        if (d_data != output) {
            CUDA_CHECK(cudaMemcpy(output, d_data, size * sizeof(F), cudaMemcpyDeviceToDevice));
        }
    } else {
        CUDA_CHECK(cudaMemcpy(output, d_data, size * sizeof(F), cudaMemcpyDeviceToHost));
    }
    
    if (need_alloc || (input != output && config.are_inputs_on_device)) {
        if (d_data != input) CUDA_CHECK(cudaFree(d_data));
    }
    
    return eIcicleError::SUCCESS;
}

/**
 * @brief Inverse NTT implementation with optimizations
 * 
 * Uses shared memory kernel for small sizes, fused 2-stage butterflies for larger sizes.
 */
template<typename F>
eIcicleError ntt_inverse_impl(
    const F* input,
    int size,
    const NTTConfig<F>& config,
    F* output
) {
    if (size == 0) return eIcicleError::SUCCESS;
    if ((size & (size - 1)) != 0) return eIcicleError::INVALID_ARGUMENT;
    
    int log_size = 0;
    while ((1 << log_size) < size) log_size++;
    
    Domain<F>* domain = Domain<F>::get_domain(log_size);
    if (!domain) {
        return eIcicleError::INVALID_ARGUMENT;
    }
    
    // Validate domain is fully initialized (twiddles allocated)
    if (!domain->twiddles || !domain->inv_twiddles) {
        return eIcicleError::INVALID_ARGUMENT;
    }
    
    // Allocate working buffer
    F* d_data = nullptr;
    bool need_alloc = !config.are_inputs_on_device;
    
    if (need_alloc) {
        CUDA_CHECK(cudaMalloc(&d_data, size * sizeof(F)));
        CUDA_CHECK(cudaMemcpy(d_data, input, size * sizeof(F), cudaMemcpyHostToDevice));
    } else if (input != output) {
        CUDA_CHECK(cudaMalloc(&d_data, size * sizeof(F)));
        CUDA_CHECK(cudaMemcpy(d_data, input, size * sizeof(F), cudaMemcpyDeviceToDevice));
    } else {
        d_data = const_cast<F*>(input);
    }
    
    const auto& gpu = GPUConfig::get();
    
    // Get NTT launch configuration - determines strategy and thread count
    auto ntt_cfg = get_ntt_launch_config(size, sizeof(F));
    
    // =========================================================================
    // STRATEGY (same as forward NTT):
    // - Size <= 256: Use shared memory kernel
    // - Size > 256: Use fused butterfly kernels (7.33% vs 2.32% SM throughput)
    // =========================================================================
    
    if (ntt_cfg.strategy == NTTStrategy::SHARED_MEMORY) {
        // Shared memory approach: single kernel does all butterflies + bit-reversal + scaling
        intt_shared_memory_kernel<<<1, size, ntt_cfg.shared_mem>>>(
            d_data, domain->inv_twiddles, domain->domain_size_inv, size, log_size
        );
        CUDA_CHECK(cudaDeviceSynchronize());
    } else {
        // Butterfly approach for larger sizes - higher SM throughput
        const int threads = ntt_cfg.threads;
        // Use DIF fused 2-stage butterflies in reverse order
        // =====================================================================
        // For inverse NTT (DIF), we go from stage log_size down to 1
        // DIF = add/sub first, then multiply by inverse twiddle
        // Fuse pairs: (log_size, log_size-1), (log_size-2, log_size-3), ...
        
        int s = log_size;
        while (s - 1 >= 1) {
            // Fuse stages s and s-1 (going from high to low)
            // For the DIF fused kernel, we process:
            //   - First: stage s with half_m = 2^(s-1)
            //   - Then: stage s-1 with half_m = 2^(s-2)
            // The kernel uses stride = half_m of LOWER stage = 2^(s-2)
            int stride = 1 << (s - 2);
            
            int blocks = (size / 4 + threads - 1) / threads;
            intt_butterfly_fused_2stage_kernel<<<blocks, threads>>>(
                d_data, domain->inv_twiddles, size, stride
            );
            CUDA_CHECK(cudaDeviceSynchronize());
            
            s -= 2;  // Go back by 2 stages
        }
        
        // Handle remaining stage if log_size is odd
        if (s == 1) {
            int m = 1 << s;
            int half_m = m / 2;
            
            int blocks = (size / 2 + threads - 1) / threads;
            intt_butterfly_kernel<<<blocks, threads>>>(
                d_data, domain->inv_twiddles, size, m, half_m
            );
            CUDA_CHECK(cudaDeviceSynchronize());
        }
        
        // Bit reversal
        F* d_temp;
        CUDA_CHECK(cudaMalloc(&d_temp, size * sizeof(F)));
        
        int blocks = (size + threads - 1) / threads;
        ntt_bit_reverse_kernel<<<blocks, threads>>>(d_temp, d_data, size, log_size);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Scale by 1/n
        ntt_scale_kernel<<<blocks, threads>>>(
            d_data, d_temp, domain->domain_size_inv, size
        );
        CUDA_CHECK(cudaDeviceSynchronize());
        
        CUDA_CHECK(cudaFree(d_temp));
    }
    
    // Copy output
    if (config.are_outputs_on_device) {
        if (d_data != output) {
            CUDA_CHECK(cudaMemcpy(output, d_data, size * sizeof(F), cudaMemcpyDeviceToDevice));
        }
    } else {
        CUDA_CHECK(cudaMemcpy(output, d_data, size * sizeof(F), cudaMemcpyDeviceToHost));
    }
    
    if (need_alloc || (input != output && config.are_inputs_on_device)) {
        if (d_data != input) CUDA_CHECK(cudaFree(d_data));
    }
    
    return eIcicleError::SUCCESS;
}

/**
 * @brief Main NTT entry point
 * 
 * Handles batch processing with optimization:
 * - For small NTTs (size <= 32) with batch_size > 1: use warp-parallel kernel
 * - Otherwise: process each batch element individually
 */
template<typename F>
eIcicleError ntt_cuda_impl(
    const F* input,
    int size,
    NTTDir direction,
    const NTTConfig<F>& config,
    F* output
) {
    int batch_size = config.batch_size > 0 ? config.batch_size : 1;
    
    // =========================================================================
    // Use warp-parallel batch kernel for small NTTs
    // When processing multiple small NTTs, each warp handles one complete NTT
    // using shuffle instructions, avoiding shared memory and __syncthreads()
    // =========================================================================
    const int WARP_BATCH_THRESHOLD = 32;  // Max size for warp-parallel processing
    
    if (batch_size > 1 && size <= WARP_BATCH_THRESHOLD && (size & (size - 1)) == 0) {
        int log_size = 0;
        while ((1 << log_size) < size) log_size++;
        
        // Get domain for twiddles
        Domain<F>* domain = Domain<F>::get_domain(log_size);
        if (!domain || !domain->twiddles || !domain->inv_twiddles) {
            return eIcicleError::INVALID_ARGUMENT;
        }
        
        // Allocate device memory if needed
        F* d_data = nullptr;
        size_t total_size = batch_size * size * sizeof(F);
        bool need_alloc = !config.are_inputs_on_device;
        
        if (need_alloc) {
            CUDA_CHECK(cudaMalloc(&d_data, total_size));
            CUDA_CHECK(cudaMemcpy(d_data, input, total_size, cudaMemcpyHostToDevice));
        } else if (input != output) {
            CUDA_CHECK(cudaMalloc(&d_data, total_size));
            CUDA_CHECK(cudaMemcpy(d_data, input, total_size, cudaMemcpyDeviceToDevice));
        } else {
            d_data = const_cast<F*>(input);
        }
        
        // Launch warp-parallel batch kernel
        // Each warp (32 threads) processes one NTT
        const auto& gpu = GPUConfig::get();
        const int threads_per_block = gpu.get_optimal_threads(KernelType::NTT_SHARED);
        int warps_per_block = threads_per_block / 32;
        int blocks = (batch_size + warps_per_block - 1) / warps_per_block;
        
        if (direction == NTTDir::kForward) {
            ntt_batch_warp_kernel<<<blocks, threads_per_block>>>(
                d_data, domain->twiddles, size, log_size, batch_size
            );
        } else {
            intt_batch_warp_kernel<<<blocks, threads_per_block>>>(
                d_data, domain->inv_twiddles, domain->domain_size_inv, size, log_size, batch_size
            );
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Copy output
        if (config.are_outputs_on_device) {
            if (d_data != output) {
                CUDA_CHECK(cudaMemcpy(output, d_data, total_size, cudaMemcpyDeviceToDevice));
            }
        } else {
            CUDA_CHECK(cudaMemcpy(output, d_data, total_size, cudaMemcpyDeviceToHost));
        }
        
        if (need_alloc || (input != output && config.are_inputs_on_device)) {
            if (d_data != input) CUDA_CHECK(cudaFree(d_data));
        }
        
        return eIcicleError::SUCCESS;
    }
    
    // =========================================================================
    // Batched shared memory kernel for medium sizes (32 < size <= 256)
    // Each block processes one NTT, all blocks launched in parallel
    // =========================================================================
    const int SHARED_MEM_BATCH_THRESHOLD = 256;
    
    if (batch_size > 1 && size <= SHARED_MEM_BATCH_THRESHOLD && (size & (size - 1)) == 0) {
        int log_size = 0;
        while ((1 << log_size) < size) log_size++;
        
        Domain<F>* domain = Domain<F>::get_domain(log_size);
        if (!domain || !domain->twiddles || !domain->inv_twiddles) {
            return eIcicleError::INVALID_ARGUMENT;
        }
        
        // Check shared memory availability
        int device;
        cudaGetDevice(&device);
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device);
        size_t shared_mem_per_block = size * sizeof(F);
        
        if (shared_mem_per_block <= prop.sharedMemPerBlock) {
            // Allocate device memory for all batches
            F* d_data = nullptr;
            size_t total_size = batch_size * size * sizeof(F);
            bool need_alloc = !config.are_inputs_on_device;
            
            if (need_alloc) {
                CUDA_CHECK(cudaMalloc(&d_data, total_size));
                CUDA_CHECK(cudaMemcpy(d_data, input, total_size, cudaMemcpyHostToDevice));
            } else if (input != output) {
                CUDA_CHECK(cudaMalloc(&d_data, total_size));
                CUDA_CHECK(cudaMemcpy(d_data, input, total_size, cudaMemcpyDeviceToDevice));
            } else {
                d_data = const_cast<F*>(input);
            }
            
            // Launch batch_size blocks, each processing one NTT
            if (direction == NTTDir::kForward) {
                ntt_shared_memory_kernel<<<batch_size, size, shared_mem_per_block>>>(
                    d_data, domain->twiddles, size, log_size
                );
            } else {
                intt_shared_memory_kernel<<<batch_size, size, shared_mem_per_block>>>(
                    d_data, domain->inv_twiddles, domain->domain_size_inv, size, log_size
                );
            }
            CUDA_CHECK(cudaDeviceSynchronize());
            
            // Copy output
            if (config.are_outputs_on_device) {
                if (d_data != output) {
                    CUDA_CHECK(cudaMemcpy(output, d_data, total_size, cudaMemcpyDeviceToDevice));
                }
            } else {
                CUDA_CHECK(cudaMemcpy(output, d_data, total_size, cudaMemcpyDeviceToHost));
            }
            
            if (need_alloc || (input != output && config.are_inputs_on_device)) {
                if (d_data != input) CUDA_CHECK(cudaFree(d_data));
            }
            
            return eIcicleError::SUCCESS;
        }
    }
    
    // =========================================================================
    // Fallback: process each batch element individually
    // This path is taken for:
    // - batch_size == 1 (nothing to parallelize)
    // - size > 256 (large NTTs that saturate GPU per element)
    // =========================================================================
    for (int b = 0; b < batch_size; b++) {
        const F* batch_input = input + b * size;
        F* batch_output = output + b * size;
        
        eIcicleError err;
        if (direction == NTTDir::kForward) {
            err = ntt_forward_impl(batch_input, size, config, batch_output);
        } else {
            err = ntt_inverse_impl(batch_input, size, config, batch_output);
        }
        
        if (err != eIcicleError::SUCCESS) {
            return err;
        }
    }
    
    return eIcicleError::SUCCESS;
}

/**
 * @brief Kernel to compute field inverse (single element)
 */
template<typename F>
__global__ void field_inv_kernel(F* out, const F* in) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        field_inv(*out, *in);
    }
}

/**
 * @brief Coset NTT implementation
 * 
 * Forward coset NTT: multiply by g^i, then NTT
 * Inverse coset NTT: INTT, then divide by g^i
 * 
 * Uses precomputed coset powers when available for ~10x speedup on the
 * coset multiplication step.
 */
template<typename F>
eIcicleError coset_ntt_cuda_impl(
    const F* input,
    int size,
    NTTDir direction,
    const F& coset_gen,
    const NTTConfig<F>& config,
    F* output
) {
    if (size == 0) return eIcicleError::SUCCESS;
    if ((size & (size - 1)) != 0) return eIcicleError::INVALID_ARGUMENT;
    
    cudaStream_t stream = static_cast<cudaStream_t>(config.stream);
    auto cfg = get_launch_config(size, KernelType::VEC_ELEMENT_WISE);
    
    // Check if we have precomputed coset powers in the domain
    int log_size = 0;
    int temp_size = size;
    while (temp_size > 1) { temp_size >>= 1; log_size++; }
    
    Domain<F>* domain = Domain<F>::get_domain(log_size);
    bool use_precomputed = domain && domain->coset_initialized;
    
    // If domain exists but coset not initialized, try to initialize now
    if (domain && !domain->coset_initialized) {
        eIcicleError init_err = init_coset_powers(domain, coset_gen, stream);
        if (init_err == eIcicleError::SUCCESS) {
            use_precomputed = true;
        }
    }
    
    // Allocate temporary buffer
    F* d_temp = nullptr;
    CUDA_CHECK(cudaMalloc(&d_temp, size * sizeof(F)));
    
    if (direction == NTTDir::kForward) {
        // Step 1: Multiply by coset powers
        F* d_input = nullptr;
        bool need_alloc_input = !config.are_inputs_on_device;
        
        if (need_alloc_input) {
            CUDA_CHECK(cudaMalloc(&d_input, size * sizeof(F)));
            CUDA_CHECK(cudaMemcpy(d_input, input, size * sizeof(F), cudaMemcpyHostToDevice));
        } else {
            d_input = const_cast<F*>(input);
        }
        
        // Use precomputed powers if available (fast path)
        if (use_precomputed && domain->coset_powers) {
            coset_mul_precomputed_kernel<<<cfg.blocks, cfg.threads, 0, stream>>>(
                d_temp, d_input, domain->coset_powers, size);
        } else {
            // Fall back to on-the-fly computation
            coset_mul_kernel<<<cfg.blocks, cfg.threads, 0, stream>>>(d_temp, d_input, coset_gen, size);
        }
        
        // Check for kernel launch failure
        cudaError_t kernelErr = cudaGetLastError();
        if (kernelErr != cudaSuccess) {
            if (need_alloc_input) cudaFree(d_input);
            cudaFree(d_temp);
            return eIcicleError::UNKNOWN_ERROR;
        }
        
        kernelErr = cudaStreamSynchronize(stream);
        if (kernelErr != cudaSuccess) {
            if (need_alloc_input) cudaFree(d_input);
            cudaFree(d_temp);
            return eIcicleError::UNKNOWN_ERROR;
        }
        
        if (need_alloc_input) {
            CUDA_CHECK(cudaFree(d_input));
        }
        
        // Step 2: Apply regular NTT
        NTTConfig<F> modified_config = config;
        modified_config.are_inputs_on_device = true;
        
        eIcicleError err = ntt_forward_impl(d_temp, size, modified_config, output);
        
        CUDA_CHECK(cudaFree(d_temp));
        
        return err;
    } else {
        // Inverse coset NTT
        // Step 1: Apply inverse NTT
        NTTConfig<F> modified_config = config;
        modified_config.are_outputs_on_device = true;
        
        eIcicleError err = ntt_inverse_impl(input, size, modified_config, d_temp);
        if (err != eIcicleError::SUCCESS) {
            CUDA_CHECK(cudaFree(d_temp));
            return err;
        }
        
        // Step 2: Divide by coset powers (multiply by g^(-i))
        F* d_output = nullptr;
        bool need_alloc_output = !config.are_outputs_on_device;
        
        if (need_alloc_output) {
            CUDA_CHECK(cudaMalloc(&d_output, size * sizeof(F)));
        } else {
            d_output = output;
        }
        
        // Use precomputed powers if available (fast path)
        if (use_precomputed && domain->coset_powers_inv) {
            coset_div_precomputed_kernel<<<cfg.blocks, cfg.threads, 0, stream>>>(
                d_output, d_temp, domain->coset_powers_inv, size);
        } else {
            // Fall back to on-the-fly computation: compute g^(-1) first using device kernel
            F* d_gen;
            F* d_gen_inv;
            CUDA_CHECK(cudaMalloc(&d_gen, sizeof(F)));
            CUDA_CHECK(cudaMalloc(&d_gen_inv, sizeof(F)));
            CUDA_CHECK(cudaMemcpy(d_gen, &coset_gen, sizeof(F), cudaMemcpyHostToDevice));
            field_inv_kernel<<<1, 1, 0, stream>>>(d_gen_inv, d_gen);
            CUDA_CHECK(cudaStreamSynchronize(stream));
            
            F coset_gen_inv;
            CUDA_CHECK(cudaMemcpy(&coset_gen_inv, d_gen_inv, sizeof(F), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaFree(d_gen));
            CUDA_CHECK(cudaFree(d_gen_inv));
            
            coset_div_kernel<<<cfg.blocks, cfg.threads, 0, stream>>>(d_output, d_temp, coset_gen_inv, size);
        }
        
        // Check for kernel launch failure
        cudaError_t kernelErr = cudaGetLastError();
        if (kernelErr != cudaSuccess) {
            if (need_alloc_output) cudaFree(d_output);
            cudaFree(d_temp);
            return eIcicleError::UNKNOWN_ERROR;
        }
        
        kernelErr = cudaStreamSynchronize(stream);
        if (kernelErr != cudaSuccess) {
            if (need_alloc_output) cudaFree(d_output);
            cudaFree(d_temp);
            return eIcicleError::UNKNOWN_ERROR;
        }
        
        if (need_alloc_output) {
            CUDA_CHECK(cudaMemcpy(output, d_output, size * sizeof(F), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaFree(d_output));
        }
        
        CUDA_CHECK(cudaFree(d_temp));
        
        return eIcicleError::SUCCESS;
    }
}

// Explicit instantiations
template eIcicleError ntt_cuda_impl<Fr>(const Fr*, int, NTTDir, const NTTConfig<Fr>&, Fr*);
template eIcicleError coset_ntt_cuda_impl<Fr>(const Fr*, int, NTTDir, const Fr&, const NTTConfig<Fr>&, Fr*);

} // namespace ntt

// =============================================================================
// NTT Domain Management
// =============================================================================

namespace ntt {

// Kernel to compute twiddle factors on GPU
// Note: one_val is passed explicitly to avoid __constant__ memory issues across shared library boundaries
__global__ void compute_twiddles_kernel(
    Fr* twiddles,
    Fr* inv_twiddles,
    Fr omega,
    Fr omega_inv,
    Fr one_val,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    // Compute omega^idx using repeated squaring
    Fr pow = one_val;
    Fr base = omega;
    int exp = idx;
    
    while (exp > 0) {
        if (exp & 1) {
            pow = pow * base;
        }
        base = base * base;
        exp >>= 1;
    }
    twiddles[idx] = pow;
    
    // Compute omega_inv^idx
    pow = one_val;
    base = omega_inv;
    exp = idx;
    
    while (exp > 0) {
        if (exp & 1) {
            pow = pow * base;
        }
        base = base * base;
        exp >>= 1;
    }
    inv_twiddles[idx] = pow;
}

// Kernel to compute omega^(2^k)
__global__ void square_field_kernel(Fr* result, const Fr* input, int iterations) {
    Fr val = *input;
    for (int i = 0; i < iterations; i++) {
        val = val * val;
    }
    *result = val;
}

// =============================================================================
// Optimized Coset Power Computation
// =============================================================================

/**
 * @brief Phase 1: Compute powers within each block using chain multiplication
 * 
 * Each block computes: g^(block_start), g^(block_start+1), ..., g^(block_start+block_size-1)
 * Using chain multiplication: p[i] = p[i-1] * g (O(1) per element after first)
 * 
 * Block 0 computes g^0..g^(B-1) directly
 * Other blocks initially compute g^0..g^(B-1), then Phase 2 multiplies by g^(block_start)
 */
__global__ void compute_coset_powers_phase1_kernel(
    Fr* coset_powers,
    Fr* coset_powers_inv,
    Fr* block_bases,        // Output: g^(block_size) for each block (for Phase 2)
    Fr* block_bases_inv,
    Fr coset_gen,
    Fr coset_gen_inv,
    Fr one_val,
    int size,
    int block_size          // Elements per block (should equal blockDim.x)
) {
    extern __shared__ Fr shared[];
    Fr* s_powers = shared;                    // [0, block_size)
    Fr* s_powers_inv = shared + block_size;   // [block_size, 2*block_size)
    
    int tid = threadIdx.x;
    int block_start = blockIdx.x * block_size;
    int gid = block_start + tid;
    
    // Phase 1a: Thread 0 computes g^0 = 1, others wait
    if (tid == 0) {
        s_powers[0] = one_val;
        s_powers_inv[0] = one_val;
    }
    __syncthreads();
    
    // Phase 1b: Chain multiplication within block
    // Each thread i computes: s_powers[i] = s_powers[i-1] * g
    // We do this in log2(block_size) parallel steps using doubling
    // 
    // Use parallel prefix (scan) with multiplication
    // Step k: element i gets multiplied by element (i - 2^k) if it exists
    
    // First, each thread loads its "local" power = g (or g_inv)
    // We'll build up using inclusive scan
    
    // Simple approach: Thread 0 computes sequentially for small block (up to 256)
    // This is actually faster than complex parallel scan for the field multiply cost
    if (tid == 0) {
        Fr pow = one_val;
        Fr pow_inv = one_val;
        s_powers[0] = pow;
        s_powers_inv[0] = pow_inv;
        
        for (int i = 1; i < block_size && (block_start + i) < size; i++) {
            pow = pow * coset_gen;
            pow_inv = pow_inv * coset_gen_inv;
            s_powers[i] = pow;
            s_powers_inv[i] = pow_inv;
        }
        
        // Store block's final power for Phase 2 (g^block_size = g^B)
        if (block_size < size) {
            block_bases[blockIdx.x] = pow * coset_gen;       // g^block_size
            block_bases_inv[blockIdx.x] = pow_inv * coset_gen_inv;
        }
    }
    __syncthreads();
    
    // Write to global memory
    if (gid < size) {
        coset_powers[gid] = s_powers[tid];
        coset_powers_inv[gid] = s_powers_inv[tid];
    }
}

/**
 * @brief Phase 2: Multiply each block's powers by block base
 * 
 * Block k's elements should be: g^(k*B + i) = g^(k*B) * g^i
 * Phase 1 computed g^i, now multiply by g^(k*B)
 * 
 * g^(k*B) is computed from block_bases using repeated squaring of g^B
 */
__global__ void compute_coset_powers_phase2_kernel(
    Fr* coset_powers,
    Fr* coset_powers_inv,
    const Fr* block_bases,      // g^B precomputed
    const Fr* block_bases_inv,
    int size,
    int block_size
) {
    int tid = threadIdx.x;
    int block_idx = blockIdx.x;
    int gid = block_idx * block_size + tid;
    
    if (gid >= size) return;
    if (block_idx == 0) return;  // Block 0 already correct
    
    // Compute g^(block_idx * block_size) using repeated squaring
    // base_power = (g^block_size)^block_idx
    Fr base_power = block_bases[0];  // g^block_size
    Fr base_power_inv = block_bases_inv[0];
    
    // Compute base^block_idx via repeated squaring
    Fr acc = Fr::one();
    Fr acc_inv = Fr::one();
    int exp = block_idx;
    
    while (exp > 0) {
        if (exp & 1) {
            acc = acc * base_power;
            acc_inv = acc_inv * base_power_inv;
        }
        base_power = base_power * base_power;
        base_power_inv = base_power_inv * base_power_inv;
        exp >>= 1;
    }
    
    // Multiply this element's power by block base
    coset_powers[gid] = coset_powers[gid] * acc;
    coset_powers_inv[gid] = coset_powers_inv[gid] * acc_inv;
}

/**
 * @brief Optimized kernel for small sizes - single block chain multiplication
 * 
 * For sizes <= 1024, use a single block with shared memory for maximum efficiency.
 * Work complexity: O(n) instead of O(n log n)
 * 
 * NOTE: Requires 2*size*sizeof(Fr) = 64KB shared memory for size=1024.
 * GPUs with < 64KB shared memory will fall back to smaller block sizes.
 */
__global__ void compute_coset_powers_small_kernel(
    Fr* coset_powers,
    Fr* coset_powers_inv,
    Fr coset_gen,
    Fr coset_gen_inv,
    Fr one_val,
    int size
) {
    extern __shared__ Fr shared[];
    Fr* s_powers = shared;
    Fr* s_powers_inv = shared + size;
    
    int tid = threadIdx.x;
    
    // Thread 0 computes all powers sequentially using chain multiplication
    // This is O(n) work with O(n) depth but runs on a single SM efficiently
    if (tid == 0) {
        Fr pow = one_val;
        Fr pow_inv = one_val;
        
        for (int i = 0; i < size; i++) {
            s_powers[i] = pow;
            s_powers_inv[i] = pow_inv;
            pow = pow * coset_gen;
            pow_inv = pow_inv * coset_gen_inv;
        }
    }
    __syncthreads();
    
    // Parallel copy to global memory
    for (int i = tid; i < size; i += blockDim.x) {
        coset_powers[i] = s_powers[i];
        coset_powers_inv[i] = s_powers_inv[i];
    }
}

/**
 * @brief Legacy kernel for fallback (uses repeated squaring per thread)
 * 
 * Work complexity: O(n log n) but fully parallel
 * Use for very large sizes where memory bandwidth dominates
 */
__global__ void compute_coset_powers_kernel(
    Fr* coset_powers,
    Fr* coset_powers_inv,
    Fr coset_gen,
    Fr coset_gen_inv,
    Fr one_val,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    // Compute g^idx using repeated squaring
    Fr pow = one_val;
    Fr base = coset_gen;
    int exp = idx;
    
    while (exp > 0) {
        if (exp & 1) {
            pow = pow * base;
        }
        base = base * base;
        exp >>= 1;
    }
    coset_powers[idx] = pow;
    
    // Compute g^(-idx) = (g^(-1))^idx
    pow = one_val;
    base = coset_gen_inv;
    exp = idx;
    
    while (exp > 0) {
        if (exp & 1) {
            pow = pow * base;
        }
        base = base * base;
        exp >>= 1;
    }
    coset_powers_inv[idx] = pow;
}

// Kernel to compute inverse and size_inv
__global__ void compute_inv_kernel(Fr* omega_inv, Fr* size_inv, const Fr* omega, int size) {
    Fr inv;
    field_inv(inv, *omega);
    *omega_inv = inv;
    
    Fr size_field = Fr::from_int((uint64_t)size);
    field_inv(*size_inv, size_field);
}

/**
 * @brief Initialize NTT domain with given root of unity
 * 
 * The domain size is determined by the order of the root of unity.
 * For BLS12-381 Fr, the maximum is 2^32 (the 2-adicity of the field).
 * We initialize domains lazily on first use, but precompute common sizes.
 */
template<typename F>
eIcicleError init_domain_cuda_impl(
    const F& root_of_unity,
    const NTTInitDomainConfig& config
) {
    // For BLS12-381, the 2-adicity is 32, meaning max NTT size is 2^32
    // However, practical circuits rarely exceed 2^24, so we default to 24
    // to save GPU memory. Larger domains can be initialized on demand.
    constexpr int max_log_size = 24;  // Practical default for BLS12-381
    
    cudaStream_t stream = static_cast<cudaStream_t>(config.stream);
    
    // Copy root_of_unity to device
    F* d_root;
    CUDA_CHECK(cudaMalloc(&d_root, sizeof(F)));
    CUDA_CHECK(cudaMemcpy(d_root, &root_of_unity, sizeof(F), cudaMemcpyHostToDevice));
    
    // Precompute twiddle factors for each domain size
    for (int log_size = 1; log_size <= max_log_size && log_size < MAX_LOG_DOMAIN_SIZE; log_size++) {
        int size = 1 << log_size;
        
        // Allocate domain
        Domain<F>* domain = new Domain<F>();
        domain->log_size = log_size;
        domain->size = size;
        
        // Allocate twiddle factor arrays
        CUDA_CHECK(cudaMalloc(&domain->twiddles, size * sizeof(F)));
        CUDA_CHECK(cudaMalloc(&domain->inv_twiddles, size * sizeof(F)));
        
        // Compute omega = root_of_unity^(2^(max_log - log_size)) on GPU
        F* d_omega;
        F* d_omega_inv;
        F* d_size_inv;
        CUDA_CHECK(cudaMalloc(&d_omega, sizeof(F)));
        CUDA_CHECK(cudaMalloc(&d_omega_inv, sizeof(F)));
        CUDA_CHECK(cudaMalloc(&d_size_inv, sizeof(F)));
        
        int squarings = max_log_size - log_size;
        square_field_kernel<<<1, 1, 0, stream>>>(d_omega, d_root, squarings);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        
        // Compute inverse and size_inv
        compute_inv_kernel<<<1, 1, 0, stream>>>(d_omega_inv, d_size_inv, d_omega, size);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        
        // Copy omega values back to host to pass to twiddle kernel
        F h_omega, h_omega_inv;
        CUDA_CHECK(cudaMemcpy(&h_omega, d_omega, sizeof(F), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&h_omega_inv, d_omega_inv, sizeof(F), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&domain->domain_size_inv, d_size_inv, sizeof(F), cudaMemcpyDeviceToHost));
        
        // Get "one" value from host to avoid __constant__ memory issues
        F h_one = F::one_host();
        
        // Compute all twiddles in parallel on GPU
        auto cfg = get_launch_config(size, KernelType::COMPUTE_BOUND);
        compute_twiddles_kernel<<<cfg.blocks, cfg.threads, 0, stream>>>(
            domain->twiddles, domain->inv_twiddles, h_omega, h_omega_inv, h_one, size
        );
        CUDA_CHECK(cudaStreamSynchronize(stream));
        
        CUDA_CHECK(cudaFree(d_omega));
        CUDA_CHECK(cudaFree(d_omega_inv));
        CUDA_CHECK(cudaFree(d_size_inv));
        
        // Register domain using the accessor function
        Domain<F>::set_domain(log_size, domain);
    }
    
    CUDA_CHECK(cudaFree(d_root));
    
    return eIcicleError::SUCCESS;
}

template eIcicleError init_domain_cuda_impl<Fr>(const Fr&, const NTTInitDomainConfig&);

/**
 * @brief Initialize coset powers for a domain (lazy initialization)
 * 
 * Precomputes g^i and g^(-i) for all i in [0, n), enabling fast coset NTT.
 * Called on first use of coset NTT with a given domain.
 * 
 * @param domain The domain to initialize coset powers for
 * @param coset_gen The coset generator g
 * @param stream CUDA stream for async operations
 * @return eIcicleError::SUCCESS on success
 * 
 * STRATEGY:
 * - Small sizes (n <= 1024): Single-block chain multiplication, O(n) work
 * - Medium sizes (1024 < n <= 2^20): Two-phase parallel algorithm, O(n + B*log(n/B)) work
 * - Large sizes (n > 2^20): Parallel repeated squaring, O(n log n) work but fully parallel
 */
template<typename F>
eIcicleError init_coset_powers(Domain<F>* domain, const F& coset_gen, cudaStream_t stream) {
    if (domain->coset_initialized) {
        // Already initialized - check if same generator
        // For now, just return success (assume same generator)
        return eIcicleError::SUCCESS;
    }
    
    size_t size = domain->size;
    
    // Allocate coset power arrays
    CUDA_CHECK(cudaMalloc(&domain->coset_powers, size * sizeof(F)));
    CUDA_CHECK(cudaMalloc(&domain->coset_powers_inv, size * sizeof(F)));
    
    // Compute inverse of coset generator using existing kernel
    F* d_gen;
    F* d_gen_inv;
    CUDA_CHECK(cudaMalloc(&d_gen, sizeof(F)));
    CUDA_CHECK(cudaMalloc(&d_gen_inv, sizeof(F)));
    CUDA_CHECK(cudaMemcpy(d_gen, &coset_gen, sizeof(F), cudaMemcpyHostToDevice));
    
    // Use the existing field_inv_kernel to compute inverse on device
    field_inv_kernel<<<1, 1, 0, stream>>>(d_gen_inv, d_gen);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    
    // Copy inverse back to host for passing to coset powers kernel
    F coset_gen_inv;
    CUDA_CHECK(cudaMemcpy(&coset_gen_inv, d_gen_inv, sizeof(F), cudaMemcpyDeviceToHost));
    
    F h_one = F::one_host();
    
    // Choose optimal algorithm based on size
    cudaError_t err;
    
    if (size <= 1024) {
        // SMALL: Single-block chain multiplication - O(n) work, minimal overhead
        // Best for small domains where kernel launch overhead matters
        size_t shared_mem = 2 * size * sizeof(F);  // powers + powers_inv
        compute_coset_powers_small_kernel<<<1, 256, shared_mem, stream>>>(
            domain->coset_powers, domain->coset_powers_inv,
            coset_gen, coset_gen_inv, h_one, (int)size
        );
    }
    else if (size <= (1 << 20)) {  // Up to 1M elements
        // MEDIUM: Two-phase algorithm - O(n + B*log(n/B)) work
        // Phase 1: Each block computes local powers via chain multiplication
        // Phase 2: Multiply by block base powers
        
        const auto& gpu = GPUConfig::get();
        const int block_size = gpu.get_optimal_threads(KernelType::COMPUTE_BOUND);
        const int num_blocks = (size + block_size - 1) / block_size;
        
        // Allocate temporary storage for block bases
        F* d_block_bases;
        F* d_block_bases_inv;
        CUDA_CHECK(cudaMalloc(&d_block_bases, num_blocks * sizeof(F)));
        CUDA_CHECK(cudaMalloc(&d_block_bases_inv, num_blocks * sizeof(F)));
        
        // Phase 1: Compute local powers in each block
        size_t shared_mem = 2 * block_size * sizeof(F);
        compute_coset_powers_phase1_kernel<<<num_blocks, block_size, shared_mem, stream>>>(
            domain->coset_powers, domain->coset_powers_inv,
            d_block_bases, d_block_bases_inv,
            coset_gen, coset_gen_inv, h_one, (int)size, block_size
        );
        CUDA_CHECK(cudaStreamSynchronize(stream));
        
        // Phase 2: Multiply each block's elements by block base power
        // Block k needs to multiply by g^(k * block_size)
        if (num_blocks > 1) {
            compute_coset_powers_phase2_kernel<<<num_blocks, block_size, 0, stream>>>(
                domain->coset_powers, domain->coset_powers_inv,
                d_block_bases, d_block_bases_inv,
                (int)size, block_size
            );
        }
        
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaFree(d_block_bases));
        CUDA_CHECK(cudaFree(d_block_bases_inv));
    }
    else {
        // LARGE: Parallel repeated squaring - O(n log n) work but fully parallel
        // Best for very large domains where parallelism compensates for extra work
        auto cfg = get_launch_config(size, KernelType::COMPUTE_BOUND);
        compute_coset_powers_kernel<<<cfg.blocks, cfg.threads, 0, stream>>>(
            domain->coset_powers, domain->coset_powers_inv,
            coset_gen, coset_gen_inv, h_one, (int)size
        );
    }
    
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(domain->coset_powers);
        cudaFree(domain->coset_powers_inv);
        domain->coset_powers = nullptr;
        domain->coset_powers_inv = nullptr;
        cudaFree(d_gen);
        cudaFree(d_gen_inv);
        return eIcicleError::UNKNOWN_ERROR;
    }
    
    CUDA_CHECK(cudaStreamSynchronize(stream));
    
    // Store generator and mark as initialized
    domain->coset_gen = coset_gen;
    domain->coset_initialized = true;
    
    CUDA_CHECK(cudaFree(d_gen));
    CUDA_CHECK(cudaFree(d_gen_inv));
    
    return eIcicleError::SUCCESS;
}

template eIcicleError init_coset_powers<Fr>(Domain<Fr>*, const Fr&, cudaStream_t);

/**
 * @brief Release NTT domain resources
 * 
 * Uses a two-phase approach to avoid holding mutex during CUDA operations:
 * 1. Collect pointers under lock and clear the domain array
 * 2. Free resources outside the lock to avoid blocking other threads
 */
template<typename F>
eIcicleError release_domain_cuda_impl() {
    // Collect pointers under lock, then free outside lock
    Domain<F>* to_delete[MAX_LOG_DOMAIN_SIZE] = {nullptr};
    
    {
        std::lock_guard<std::mutex> lock(Domain<F>::domains_mutex);
        for (int i = 0; i < MAX_LOG_DOMAIN_SIZE; i++) {
            to_delete[i] = Domain<F>::domains[i];
            Domain<F>::domains[i] = nullptr;
        }
    }
    
    // Free outside lock to avoid blocking other threads
    for (int i = 0; i < MAX_LOG_DOMAIN_SIZE; i++) {
        Domain<F>* domain = to_delete[i];
        if (domain) {
            if (domain->twiddles) CUDA_CHECK(cudaFree(domain->twiddles));
            if (domain->inv_twiddles) CUDA_CHECK(cudaFree(domain->inv_twiddles));
            if (domain->coset_powers) CUDA_CHECK(cudaFree(domain->coset_powers));
            if (domain->coset_powers_inv) CUDA_CHECK(cudaFree(domain->coset_powers_inv));
            delete domain;
        }
    }
    
    return eIcicleError::SUCCESS;
}

template eIcicleError release_domain_cuda_impl<Fr>();

} // namespace ntt

// =============================================================================
// C++ Mangled Symbol Exports (matching ICICLE)
// =============================================================================

namespace ntt {

// These match the mangled symbols from the library
template<typename F>
eIcicleError ntt_cuda(
    const F* input,
    int size,
    NTTDir direction,
    const NTTConfig<F>& config,
    F* output
) {
    return ntt_cuda_impl<F>(input, size, direction, config, output);
}

template<typename F>
eIcicleError coset_ntt_cuda(
    const F* input,
    int size,
    NTTDir direction,
    const F& coset_gen,
    const NTTConfig<F>& config,
    F* output
) {
    return coset_ntt_cuda_impl<F>(input, size, direction, coset_gen, config, output);
}

template<typename F>
eIcicleError init_domain_cuda(
    const F& root_of_unity,
    const NTTInitDomainConfig& config
) {
    return init_domain_cuda_impl<F>(root_of_unity, config);
}

template<typename F>
eIcicleError release_domain_cuda() {
    return release_domain_cuda_impl<F>();
}

// Explicit instantiations for symbol export
template eIcicleError ntt_cuda<Fr>(const Fr*, int, NTTDir, const NTTConfig<Fr>&, Fr*);
template eIcicleError coset_ntt_cuda<Fr>(const Fr*, int, NTTDir, const Fr&, const NTTConfig<Fr>&, Fr*);
template eIcicleError init_domain_cuda<Fr>(const Fr&, const NTTInitDomainConfig&);
template eIcicleError release_domain_cuda<Fr>();

} // namespace ntt

// =============================================================================
// Icicle-Compatible C API Exports
// =============================================================================

extern "C" {

eIcicleError bls12_381_ntt_cuda(
    const bls12_381::Fr* input,
    int size,
    NTTDir dir,
    const NTTConfig<Fr>* config,
    bls12_381::Fr* output)
{
    return ntt::ntt_cuda<bls12_381::Fr>(input, size, dir, *config, output);
}

eIcicleError bls12_381_ntt_init_domain_cuda(
    const bls12_381::Fr* root_of_unity,
    const NTTInitDomainConfig* config)
{
    return ntt::init_domain_cuda<bls12_381::Fr>(*root_of_unity, *config);
}

eIcicleError bls12_381_ntt_release_domain_cuda()
{
    return ntt::release_domain_cuda<bls12_381::Fr>();
}

eIcicleError bls12_381_coset_ntt_cuda(
    const bls12_381::Fr* input,
    int size,
    NTTDir dir,
    const bls12_381::Fr* coset_gen,
    const NTTConfig<Fr>* config,
    bls12_381::Fr* output)
{
    return ntt::coset_ntt_cuda<bls12_381::Fr>(input, size, dir, *coset_gen, *config, output);
}

} // extern "C"
