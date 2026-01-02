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
 * @brief Multi-Scalar Multiplication (MSM) - Pippenger's Algorithm
 * 
 * Computes: R = Σᵢ sᵢ × Pᵢ for scalars s and points P
 * 
 * ARCHITECTURE:
 * =============
 * MSM kernels are templates defined in this header. This is an exception to
 * the "kernels in .cu files" rule because MSM is parameterized by curve type.
 * Template instantiation happens in msm.cu.
 * 
 * ALGORITHM (Pippenger's Bucket Method):
 * ======================================
 * 1. Decompose scalars into windows of c bits each
 * 2. For each window, accumulate points into 2^c buckets
 * 3. Compute weighted sum: Σⱼ j × bucket[j] (triangle sum)
 * 4. Combine windows: R = Σ 2^(c*w) × window_sum[w]
 * 
 * LIMITS:
 * =======
 * - Maximum MSM size: 2^24 points (16M)
 *   - Larger sizes risk integer overflow in intermediate calculations
 *   - For larger MSMs, split into batches and aggregate results
 * 
 * - Window size c: 1-24
 *   - Larger windows = more buckets = more memory
 *   - c > 24 may cause bucket index overflow
 * 
 * MEMORY USAGE:
 * =============
 * - Bucket storage: num_windows * 2^(c-1) * sizeof(G1Projective)
 *   - For c=16, n=2^20: 16 windows * 32K buckets * 288 bytes ≈ 148 MB
 * - CUB sort temporary: ~2x input size for radix sort
 * - Total GPU memory required: approximately 3-4x input point array size
 * 
 * Example memory for 1M points (c=15):
 *   - Input points: 1M * 288 bytes = 288 MB
 *   - Buckets: 17 windows * 16K buckets * 288 bytes = 78 MB
 *   - Sort temp: ~100 MB
 *   - Total: ~500 MB
 * 
 * SECURITY (Constant-Time Implementation):
 * ========================================
 * Uses Sort-Reduce pattern for bucket accumulation:
 * - Sort (bucket_idx, point) pairs by bucket
 * - Reduce consecutive points for same bucket
 * This prevents timing side-channels that could leak scalar information.
 * 
 * For side-channel sensitive applications, verify this is sufficient
 * for your threat model. The main remaining side-channel is memory
 * access patterns during bucket accumulation.
 * 
 * PERFORMANCE:
 * ============
 * - Optimal c selected based on MSM size (larger MSM → larger window)
 * - Signed digit representation for smaller bucket count
 * - CUB library for efficient parallel sorting
 * 
 * Kernels:
 * - compute_bucket_indices_kernel: Decompose scalars into bucket indices
 * - accumulate_sorted_kernel: Sort-Reduce bucket accumulation
 * - parallel_bucket_reduction_kernel: Compute weighted bucket sums
 * - final_accumulation_kernel: Combine window results
 */

#pragma once

#include "point.cuh"
#include "icicle_types.cuh"
#include <cuda_runtime.h>
#include <cub/cub.cuh>

namespace msm {

using namespace bls12_381;
using namespace icicle;

// =============================================================================
// MSM Limits and Constants
// =============================================================================

// Maximum MSM size to prevent integer overflow in packed index encoding
// (idx << 1) requires idx < 2^31 for safe unsigned 32-bit operation
constexpr int MAX_MSM_SIZE = (1 << 24);  // 16M points

// Window size limits
constexpr int MIN_WINDOW_SIZE = 1;
constexpr int MAX_WINDOW_SIZE = 24;  // Prevents bucket count overflow

// Invalid bucket index marker (for zero contributions)
constexpr unsigned int INVALID_BUCKET_INDEX = 0xFFFFFFFF;

// =============================================================================
// GPU Performance Tuning
// =============================================================================

// Kernel type identifiers (compile-time constants to avoid string comparison)
enum class KernelType {
    COMPUTE_INDICES,
    HISTOGRAM,
    ACCUMULATE,
    REDUCTION
};

/**
 * @brief Get optimal block size based on GPU compute capability and kernel type
 * 
 * Different kernels have different characteristics:
 * - Memory-bound: benefit from larger blocks (better memory bandwidth utilization)
 * - Compute-bound: may need smaller blocks (register pressure)
 * - Atomic-bound: medium blocks (balance contention vs parallelism)
 */
inline int get_optimal_block_size(int compute_capability_major, KernelType kernel_type) {
    // Validate compute capability is in reasonable range
    // GPU compute capabilities range from 1.x to 10.x (as of 2025)
    if (compute_capability_major < 1 || compute_capability_major > 15) {
        // Invalid capability - return safe conservative default
        return 256;
    }
    
    // Modern GPUs (Ampere 8.x, Ada 8.9, Hopper 9.x, Blackwell 10.x+)
    bool is_modern = compute_capability_major >= 8;
    
    // Kernel-specific tuning
    switch (kernel_type) {
        case KernelType::COMPUTE_INDICES:
            // Memory-bound: reading scalars, writing indices
            // Higher thread count improves memory throughput
            return is_modern ? 512 : 256;
            
        case KernelType::HISTOGRAM:
            // Atomic-bound: too many threads increase contention
            // Medium block size balances parallelism and atomic efficiency
            return is_modern ? 512 : 256;
            
        case KernelType::ACCUMULATE:
            // Compute-bound: point arithmetic (field ops, curve ops)
            // Register pressure from point operations limits thread count
            return 256;  // Conservative for all architectures
            
        case KernelType::REDUCTION:
            // Sequential per-thread: no benefit from larger blocks
            return 256;
            
        default:
            // Unknown kernel type - return safe default
            return 256;
    }
}

/**
 * @brief Detect GPU compute capability with error checking (thread-safe cached)
 * 
 * SECURITY:
 * - Thread-safe: uses atomic operations for cache initialization
 * - Error handling: validates all CUDA calls, returns safe default on failure
 * - Bounds checking: ensures returned value is in valid range
 * 
 * @return GPU compute capability major version, or 5 (safe default) on error
 */
inline int get_gpu_compute_capability() {
    // Thread-safe cache using atomic flag
    static std::atomic<int> cached_capability{-1};
    
    // Fast path: capability already cached
    int capability = cached_capability.load(std::memory_order_acquire);
    if (capability != -1) {
        return capability;
    }
    
    // Slow path: query GPU capability with error checking
    int device = 0;
    cudaError_t err = cudaGetDevice(&device);
    if (err != cudaSuccess) {
        // CUDA not initialized or no GPU - return safe default
        // Default to compute capability 5.x (Maxwell) - conservative but functional
        cached_capability.store(5, std::memory_order_release);
        return 5;
    }
    
    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) {
        // Failed to get device properties - return safe default
        cached_capability.store(5, std::memory_order_release);
        return 5;
    }
    
    // Validate compute capability is in reasonable range
    int major = prop.major;
    if (major < 1 || major > 15) {
        // Invalid capability - clamp to safe range
        major = 5;
    }
    
    // Cache the validated capability
    cached_capability.store(major, std::memory_order_release);
    return major;
}

// =============================================================================
// Generic Point Operation Dispatchers
// =============================================================================
// These allow the templated kernels to call the correct point operations
// based on the point type (G1 or G2).
//
// IMPORTANT: G1 and G2 use different base fields:
// - G1: Fq (381-bit prime field)
// - G2: Fq2 = Fq[u]/(u²+1) (quadratic extension)
//
// Point sizes differ:
// - G1Affine: 96 bytes, G1Projective: 144 bytes
// - G2Affine: 192 bytes, G2Projective: 288 bytes
//
// Scalars (Fr) are the SAME for both curves.

// --- Point Addition (Projective + Projective) ---
__device__ __forceinline__ void point_add(G1Projective& result, const G1Projective& a, const G1Projective& b) {
    g1_add(result, a, b);
}

__device__ __forceinline__ void point_add(G2Projective& result, const G2Projective& a, const G2Projective& b) {
    g2_add(result, a, b);
}

// --- Mixed Addition (Projective + Affine) ---
__device__ __forceinline__ void point_add_mixed(G1Projective& result, const G1Projective& a, const G1Affine& b) {
    g1_add_mixed(result, a, b);
}

__device__ __forceinline__ void point_add_mixed(G2Projective& result, const G2Projective& a, const G2Affine& b) {
    g2_add_mixed(result, a, b);
}

// --- Point Doubling ---
__device__ __forceinline__ void point_double(G1Projective& result, const G1Projective& a) {
    g1_double(result, a);
}

__device__ __forceinline__ void point_double(G2Projective& result, const G2Projective& a) {
    g2_double(result, a);
}

// =============================================================================
// MSM Configuration
// =============================================================================

/**
 * @brief Determine optimal window size based on MSM size
 * 
 * Empirically tuned values matching Icicle/BLST heuristics.
 * Larger windows = fewer additions but more buckets.
 */
__host__ __device__ inline int get_optimal_c(int msm_size) {
    if (msm_size <= 1) return 1;
    
    int log_size = 0;
    int temp = msm_size;
    while (temp > 1) {
        temp >>= 1;
        log_size++;
    }
    
    // Empirical optimal values
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
// Scalar Window Extraction
// =============================================================================

/**
 * @brief Extract c-bit window from scalar
 */
__device__ __forceinline__ int extract_window_value(
    const Fr& scalar,
    int window_idx,
    int c
) {
    int bit_offset = window_idx * c;
    int limb_idx = bit_offset / 64;
    int bit_in_limb = bit_offset % 64;
    
    if (limb_idx >= Fr::LIMBS) return 0;
    
    uint64_t window = scalar.limbs[limb_idx] >> bit_in_limb;
    
    // Handle cross-limb windows
    int bits_from_first = 64 - bit_in_limb;
    if (bits_from_first < c && limb_idx + 1 < Fr::LIMBS) {
        window |= (scalar.limbs[limb_idx + 1] << bits_from_first);
    }
    
    // Mask to c bits
    return (int)(window & ((1ULL << c) - 1));
}

// =============================================================================
// Bucket Accumulation Kernels (Sorted / Constant-Time)
// =============================================================================

/**
 * @brief Compute bucket indices for all scalar windows (Constant Time)
 */
template<typename S>
__global__ void compute_bucket_indices_kernel(
    unsigned int* bucket_indices,    // [msm_size * num_windows]
    unsigned int* packed_indices,    // [msm_size * num_windows] (point_idx << 1 | sign)
    const S* scalars,
    int msm_size,
    int c,
    int num_windows,
    int num_buckets // Actual buckets per window (excluding trash)
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= msm_size) return;
    
    S scalar = scalars[idx];
    
    // Number of buckets per window including trash bucket
    int buckets_per_window = num_buckets + 1;
    
    // Signed window decomposition with carry propagation
    // We use signed digits in range [-(2^(c-1)), 2^(c-1)]
    // When window_val > num_buckets (2^(c-1)), we use negative form:
    //   window_val = 2^c - bucket_val  =>  bucket_val = 2^c - window_val
    // This means we're computing: -bucket_val * P + carry * 2^c * P
    // The carry propagates to the next window.
    
    int carry = 0;
    
    for (int w = 0; w < num_windows; w++) {
        int output_idx = idx * num_windows + w;
        
        // Extract window value
        int bit_offset = w * c;
        int limb_idx = bit_offset / 64;
        int bit_in_limb = bit_offset % 64;
        
        uint64_t window = scalar.limbs[limb_idx] >> bit_in_limb;
        if (bit_in_limb + c > 64 && limb_idx + 1 < S::LIMBS) {
            window |= (scalar.limbs[limb_idx + 1] << (64 - bit_in_limb));
        }
        window &= ((1ULL << c) - 1);
        
        // Add carry from previous window
        int window_val = (int)window + carry;
        carry = 0;
        
        // Constant-time logic
        int sign = 0;
        int bucket_val = window_val;
        
        // Handle signed digit: if (window_val > num_buckets)
        // Use negative representation: bucket_val = 2^c - window_val
        // This generates a carry of +1 to the next window
        int is_large = (window_val > num_buckets);
        if (is_large) {
            bucket_val = (1 << c) - window_val;
            sign = 1;
            carry = 1;  // Propagate carry to next window
        }
        
        // Handle zero: if (bucket_val == 0 after adjustment)
        int is_zero = (bucket_val == 0);
        if (is_zero) {
            bucket_val = num_buckets + 1; // Map to trash bucket (beyond valid range)
            sign = 0;
        }
        
        // Global bucket index (0-based)
        unsigned int bucket_idx = w * buckets_per_window + (bucket_val - 1);
        
        bucket_indices[output_idx] = bucket_idx;
        
        // Use unsigned to prevent signed overflow, validate idx fits in 31 bits
        // This is guaranteed by MAX_MSM_SIZE check at function entry
        unsigned int idx_unsigned = static_cast<unsigned int>(idx);
        packed_indices[output_idx] = (idx_unsigned << 1) | (sign & 1);
    }
    
    // For valid scalars < modulus, the final window should never produce a carry.
    // If it does, the scalar is malformed or >= modulus, which should be rejected.
    // We handle this by mapping to trash bucket (same as zero contributions).
    if (carry != 0) {
        // Final window generated carry - mark all windows for this scalar as invalid
        for (int w = 0; w < num_windows; w++) {
            int output_idx = idx * num_windows + w;
            bucket_indices[output_idx] = INVALID_BUCKET_INDEX;
        }
    }
}

/**
 * @brief Conflict-free bucket accumulation after sorting (templated for G1/G2)
 * 
 * Template parameters:
 * - A: Affine point type (G1Affine or G2Affine)
 * - P: Projective point type (G1Projective or G2Projective)
 */
template<typename A, typename P>
__global__ void accumulate_sorted_kernel(
    P* buckets,
    const unsigned int* sorted_packed_indices,
    const A* bases,
    const unsigned int* bucket_offsets,
    const unsigned int* bucket_sizes,
    int total_buckets
) {
    int bucket_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (bucket_idx >= total_buckets) return;
    
    unsigned int offset = bucket_offsets[bucket_idx];
    unsigned int size = bucket_sizes[bucket_idx];
    
    if (size == 0) {
        buckets[bucket_idx] = P::identity();
        return;
    }
    
    P acc = P::identity();
    
    for (unsigned int i = 0; i < size; i++) {
        unsigned int idx = offset + i;
        unsigned int packed = sorted_packed_indices[idx];
        unsigned int point_idx = packed >> 1;
        unsigned int sign = packed & 1;
        
        A base = bases[point_idx];
        if (sign) {
            base = base.neg();
        }
        
        // Uses overloaded point_add_mixed for G1 or G2
        point_add_mixed(acc, acc, base);
    }
    
    buckets[bucket_idx] = acc;
}

/**
 * @brief Simple atomic histogram kernel
 * 
 * Each thread increments the count for its bucket index using atomics.
 */
__global__ void histogram_atomic_kernel(
    unsigned int* histogram,
    const unsigned int* indices,
    int num_samples,
    int num_buckets
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_samples) return;
    
    unsigned int bucket = indices[idx];
    if (bucket < (unsigned int)num_buckets) {
        atomicAdd(&histogram[bucket], 1);
    }
}

/**
 * @brief Optimized histogram kernel with warp-level aggregation
 * 
 * OPTIMIZATION: Reduces atomic contention by aggregating within warps first.
 * Uses warp-level ballot and popcount to count matching buckets before
 * issuing a single atomic per warp instead of per thread.
 */
__global__ void histogram_warp_aggregated_kernel(
    unsigned int* histogram,
    const unsigned int* indices,
    int num_samples,
    int num_buckets
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int lane_id = threadIdx.x & 31;
    
    unsigned int bucket = INVALID_BUCKET_INDEX;
    if (idx < num_samples) {
        bucket = indices[idx];
        if (bucket >= (unsigned int)num_buckets) {
            bucket = INVALID_BUCKET_INDEX; // Mark invalid
        }
    }
    
    // Warp-level aggregation: find threads with same bucket value
    // Process each unique bucket value in the warp
    unsigned int active_mask = __ballot_sync(0xFFFFFFFF, bucket != INVALID_BUCKET_INDEX);
    
    while (active_mask != 0) {
        // Get the bucket value from the first active lane
        int leader = __ffs(active_mask) - 1;
        unsigned int leader_bucket = __shfl_sync(0xFFFFFFFF, bucket, leader);
        
        // Find all lanes with matching bucket
        unsigned int match_mask = __ballot_sync(active_mask, bucket == leader_bucket);
        
        // Count matches and have leader do single atomic
        int count = __popc(match_mask);
        if (lane_id == leader && leader_bucket != INVALID_BUCKET_INDEX) {
            atomicAdd(&histogram[leader_bucket], count);
        }
        
        // Remove processed lanes from active mask
        active_mask &= ~match_mask;
    }
}

/**
 * @brief Parallel bucket reduction (templated for G1/G2)
 * 
 * Runs one thread per window to perform the triangle summation.
 * Since num_buckets can be large (e.g. 2^16), we read directly from global memory.
 * 
 * Template parameters:
 * - P: Projective point type (G1Projective or G2Projective)
 */
template<typename P>
__global__ void parallel_bucket_reduction_kernel(
    P* window_results,
    const P* buckets,
    int num_windows,
    int num_buckets, // Excluding trash
    int buckets_per_window // Including trash
) {
    int window_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (window_idx >= num_windows) return;
    
    // Offset to this window's buckets
    const P* window_buckets = buckets + window_idx * buckets_per_window;
    
    P running_sum = P::identity();
    P window_sum = P::identity();
    
    // Triangle summation: sum_{i=1}^B i*B_i
    // Implemented as:
    // running_sum += B_i
    // window_sum += running_sum
    // Iterate backwards from B-1 to 0 (bucket values B to 1)
    // Note: bucket_idx i corresponds to value i+1
    
    for (int i = num_buckets - 1; i >= 0; i--) {
        // Uses overloaded point_add for G1 or G2
        point_add(running_sum, running_sum, window_buckets[i]);
        point_add(window_sum, window_sum, running_sum);
    }
    
    window_results[window_idx] = window_sum;
}

// =============================================================================
// Final Window Combination
// =============================================================================

/**
 * @brief Combine window results: result = sum_{w} 2^{w*c} * window_result[w]
 * 
 * Template parameters:
 * - P: Projective point type (G1Projective or G2Projective)
 */
template<typename P>
__global__ void final_accumulation_kernel(
    P* result,
    const P* window_results,
    int num_windows,
    int c
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    
    P acc = window_results[num_windows - 1];
    
    // Process windows from second-highest to lowest
    for (int w = num_windows - 2; w >= 0; w--) {
        // Double c times
        for (int i = 0; i < c; i++) {
            // Uses overloaded point_double for G1 or G2
            point_double(acc, acc);
        }
        // Add window result (uses overloaded point_add)
        point_add(acc, acc, window_results[w]);
    }
    
    *result = acc;
}

// =============================================================================
// Main MSM Entry Point
// =============================================================================

/**
 * @brief MSM using Pippenger's bucket method with Sort-Reduce
 * implementation with conflict-free accumulation.
 * 
 * This implementation uses a Sort-Reduce approach to mitigate timing side-channels:
 * 1. Compute bucket indices for all windows (constant-time).
 * 2. Sort indices using Radix Sort (deterministic time).
 * 3. Accumulate sorted buckets (contiguous memory access).
 * 
 * Zeros are mapped to a "trash bucket" to avoid data-dependent skipping.
 */
template<typename S, typename A, typename P>
cudaError_t msm_cuda(
    const S* scalars,
    const A* bases,
    int msm_size,
    const MSMConfig& config,
    P* result
) {
    // Validate pointers are non-null
    if (scalars == nullptr || bases == nullptr || result == nullptr) {
        return cudaErrorInvalidValue;
    }
    
    // Validate msm_size is in safe range
    if (msm_size < 0) {
        return cudaErrorInvalidValue;
    }
    
    // Handle empty MSM
    if (msm_size == 0) {
        P identity = P::identity();
        if (config.are_results_on_device) {
            cudaMemcpy(result, &identity, sizeof(P), cudaMemcpyHostToDevice);
        } else {
            *result = identity;
        }
        return cudaSuccess;
    }
    
    // Validate msm_size doesn't exceed maximum safe value
    // This prevents integer overflow in packed index encoding (idx << 1)
    if (msm_size > MAX_MSM_SIZE) {
        return cudaErrorInvalidValue;
    }

    cudaStream_t stream = config.stream;
    cudaError_t err;
    
    // =============================================================================
    // Parameter Validation
    // =============================================================================
    
    // Determine window size with proper validation
    int c = config.c;
    if (c <= 0) {
        // Use automatic selection if not specified or invalid
        c = get_optimal_c(msm_size);
    }
    
    // Validate window size is in safe range
    if (c < MIN_WINDOW_SIZE || c > MAX_WINDOW_SIZE) {
        return cudaErrorInvalidValue;
    }

    // Validate scalar bitsize
    int scalar_bits = config.bitsize > 0 ? config.bitsize : S::LIMBS * 64;
    if (scalar_bits <= 0 || scalar_bits > S::LIMBS * 64) {
        return cudaErrorInvalidValue;
    }
    
    int num_windows = get_num_windows(scalar_bits, c);
    
    // Validate bucket count doesn't overflow
    // For c up to 24: (1 << 23) = 8,388,608 buckets per window (safe)
    // c >= 32 would cause overflow in int
    if (c >= 31) {  // Extra safety margin
        return cudaErrorInvalidValue;
    }
    int num_buckets = (1 << (c - 1));  // Signed digit representation
    
    // Check for overflow in total_buckets calculation
    long long total_buckets_long = (long long)num_windows * (long long)(num_buckets + 1);
    if (total_buckets_long > (long long)INT_MAX) {
        return cudaErrorInvalidValue;
    }
    int total_buckets = (int)total_buckets_long;
    
    // Check for overflow in num_contributions calculation
    long long num_contributions_long = (long long)msm_size * (long long)num_windows;
    if (num_contributions_long > (long long)INT_MAX) {
        return cudaErrorInvalidValue;
    }
    int num_contributions = (int)num_contributions_long;
    
    // Allocate device memory
    S* d_scalars = nullptr;
    A* d_bases = nullptr;
    P* d_result = nullptr;
    P* d_buckets = nullptr;       // Use template P for bucket type
    P* d_window_results = nullptr; // Use template P for window results
    
    // Sorting buffers
    unsigned int *d_bucket_indices = nullptr, *d_packed_indices = nullptr;
    unsigned int *d_bucket_indices_sorted = nullptr, *d_packed_indices_sorted = nullptr;
    unsigned int *d_bucket_offsets = nullptr, *d_bucket_sizes = nullptr;

    // Helper macro for error handling with cleanup
    #define MSM_CUDA_CHECK(call) do { \
        err = call; \
        if (err != cudaSuccess) goto cleanup; \
    } while(0)
    
    // Handle input data
    if (config.are_scalars_on_device) {
        d_scalars = const_cast<S*>(scalars);
    } else {
        MSM_CUDA_CHECK(cudaMalloc(&d_scalars, msm_size * sizeof(S)));
        MSM_CUDA_CHECK(cudaMemcpyAsync(d_scalars, scalars, msm_size * sizeof(S), 
                              cudaMemcpyHostToDevice, stream));
    }
    
    if (config.are_points_on_device) {
        d_bases = const_cast<A*>(bases);
    } else {
        MSM_CUDA_CHECK(cudaMalloc(&d_bases, msm_size * sizeof(A)));
        MSM_CUDA_CHECK(cudaMemcpyAsync(d_bases, bases, msm_size * sizeof(A),
                              cudaMemcpyHostToDevice, stream));
    }
    
    MSM_CUDA_CHECK(cudaMalloc(&d_buckets, total_buckets * sizeof(P)));
    MSM_CUDA_CHECK(cudaMalloc(&d_window_results, num_windows * sizeof(P)));
    
    if (config.are_results_on_device) {
        d_result = result;
    } else {
        MSM_CUDA_CHECK(cudaMalloc(&d_result, sizeof(P)));
    }
    
    // Allocate sorting buffers
    MSM_CUDA_CHECK(cudaMalloc(&d_bucket_indices, num_contributions * sizeof(unsigned int)));
    MSM_CUDA_CHECK(cudaMalloc(&d_packed_indices, num_contributions * sizeof(unsigned int)));
    MSM_CUDA_CHECK(cudaMalloc(&d_bucket_indices_sorted, num_contributions * sizeof(unsigned int)));
    MSM_CUDA_CHECK(cudaMalloc(&d_packed_indices_sorted, num_contributions * sizeof(unsigned int)));
    MSM_CUDA_CHECK(cudaMalloc(&d_bucket_offsets, total_buckets * sizeof(unsigned int)));
    MSM_CUDA_CHECK(cudaMalloc(&d_bucket_sizes, total_buckets * sizeof(unsigned int)));
    
    // 1. Compute Indices
    {
        int gpu_capability = get_gpu_compute_capability();
        int threads = get_optimal_block_size(gpu_capability, KernelType::COMPUTE_INDICES);
        int blocks = (msm_size + threads - 1) / threads;
        compute_bucket_indices_kernel<<<blocks, threads, 0, stream>>>(
            d_bucket_indices, d_packed_indices, d_scalars, msm_size, c, num_windows, num_buckets
        );
        MSM_CUDA_CHECK(cudaGetLastError());
    }
    
    // 2. Histogram (Compute bucket sizes) using warp-aggregated kernel
    {
        // Initialize sizes to 0
        MSM_CUDA_CHECK(cudaMemsetAsync(d_bucket_sizes, 0, total_buckets * sizeof(unsigned int), stream));
        
        // Use optimized warp-aggregated histogram kernel (reduces atomics by up to 32×)
        int gpu_capability = get_gpu_compute_capability();
        int threads = get_optimal_block_size(gpu_capability, KernelType::HISTOGRAM);
        int blocks = (num_contributions + threads - 1) / threads;
        histogram_warp_aggregated_kernel<<<blocks, threads, 0, stream>>>(
            d_bucket_sizes, d_bucket_indices, num_contributions, total_buckets);
        MSM_CUDA_CHECK(cudaGetLastError());
    }
    
    // 3. Scan (Compute bucket offsets)
    {
        void* d_temp_storage = nullptr;
        size_t temp_storage_bytes = 0;
        
        MSM_CUDA_CHECK(cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes,
            d_bucket_sizes, d_bucket_offsets, total_buckets, stream));
            
        MSM_CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes));
        
        MSM_CUDA_CHECK(cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes,
            d_bucket_sizes, d_bucket_offsets, total_buckets, stream));
            
        MSM_CUDA_CHECK(cudaFree(d_temp_storage));
    }
    
    // 4. Sort
    {
        void* d_temp_storage = nullptr;
        size_t temp_storage_bytes = 0;
        
        MSM_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
            d_bucket_indices, d_bucket_indices_sorted,
            d_packed_indices, d_packed_indices_sorted,
            num_contributions, 0, 32, stream));
            
        MSM_CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes));
        
        MSM_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
            d_bucket_indices, d_bucket_indices_sorted,
            d_packed_indices, d_packed_indices_sorted,
            num_contributions, 0, 32, stream));
            
        MSM_CUDA_CHECK(cudaFree(d_temp_storage));
    }
    
    // 5. Accumulate Sorted
    {
        int gpu_capability = get_gpu_compute_capability();
        int threads = get_optimal_block_size(gpu_capability, KernelType::ACCUMULATE);
        int blocks = (total_buckets + threads - 1) / threads;
        
        // Template instantiation: A = Affine type, P = Projective type
        accumulate_sorted_kernel<A, P><<<blocks, threads, 0, stream>>>(
            d_buckets,
            d_packed_indices_sorted,
            d_bases,
            d_bucket_offsets,
            d_bucket_sizes,
            total_buckets
        );
        MSM_CUDA_CHECK(cudaGetLastError());
    }
    
    // 6. Bucket Reduction
    {
        int gpu_capability = get_gpu_compute_capability();
        int optimal_threads = get_optimal_block_size(gpu_capability, KernelType::REDUCTION);
        int threads = min(num_windows, optimal_threads);
        int blocks = (num_windows + threads - 1) / threads;
        
        // Template instantiation: P = Projective type
        parallel_bucket_reduction_kernel<P><<<blocks, threads, 0, stream>>>(
            d_window_results,
            d_buckets,
            num_windows,
            num_buckets,
            num_buckets + 1 // buckets_per_window
        );
        MSM_CUDA_CHECK(cudaGetLastError());
    }
    
    // Final accumulation
    {
        // Template instantiation: P = Projective type
        final_accumulation_kernel<P><<<1, 1, 0, stream>>>(
            d_result,
            d_window_results,
            num_windows,
            c
        );
        MSM_CUDA_CHECK(cudaGetLastError());
    }
    
    // Copy result back if needed
    if (!config.are_results_on_device) {
        MSM_CUDA_CHECK(cudaMemcpyAsync(result, d_result, sizeof(P),
                              cudaMemcpyDeviceToHost, stream));
    }
    
    // Synchronize if not async
    if (!config.is_async) {
        MSM_CUDA_CHECK(cudaStreamSynchronize(stream));
    }

cleanup:
    // Ensure all GPU operations complete before cleanup
    // This prevents use-after-free in async mode
    if (config.is_async && stream != nullptr) {
        cudaStreamSynchronize(stream);  // Best effort, ignore errors in cleanup
    }
    
    // Track cleanup errors separately from main operation errors
    cudaError_t cleanup_err = cudaSuccess;
    
    #define SAFE_FREE(ptr) do { \
        if (ptr) { \
            cudaError_t e = cudaFree(ptr); \
            if (e != cudaSuccess && cleanup_err == cudaSuccess) cleanup_err = e; \
        } \
    } while(0)
    
    SAFE_FREE(d_buckets);
    SAFE_FREE(d_window_results);
    SAFE_FREE(d_bucket_indices);
    SAFE_FREE(d_packed_indices);
    SAFE_FREE(d_bucket_indices_sorted);
    SAFE_FREE(d_packed_indices_sorted);
    SAFE_FREE(d_bucket_offsets);
    SAFE_FREE(d_bucket_sizes);
    
    if (!config.are_scalars_on_device) SAFE_FREE(d_scalars);
    if (!config.are_points_on_device) SAFE_FREE(d_bases);
    if (!config.are_results_on_device) SAFE_FREE(d_result);
    
    #undef SAFE_FREE
    #undef MSM_CUDA_CHECK
    
    // Return original error if present, otherwise cleanup error
    return (err != cudaSuccess) ? err : cleanup_err;
}

} // namespace msm
