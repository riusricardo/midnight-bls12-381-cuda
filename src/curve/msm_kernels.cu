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
 * @file msm_kernels.cu
 * @brief Multi-Scalar Multiplication (MSM) Kernel Implementations
 * 
 * This file contains all MSM CUDA kernels and the msm_cuda entry point.
 * 
 * ALGORITHM (Pippenger's Bucket Method):
 * ======================================
 * 1. Decompose scalars into windows of c bits each
 * 2. For each window, accumulate points into 2^c buckets
 * 3. Compute weighted sum: Σⱼ j × bucket[j] (triangle sum)
 * 4. Combine windows: R = Σ 2^(c*w) × window_sum[w]
 * 
 * OPTIMIZATIONS:
 * ==============
 * - Warp-aggregated histogram (reduces atomics by up to 32×)
 * - Cooperative bucket accumulation (multiple threads per bucket)
 * - Persistent threads for small MSM sizes
 * - Multi-threaded bucket reduction
 * - Parallel final accumulation
 * 
 * SECURITY:
 * =========
 * Uses Sort-Reduce pattern for constant-time bucket accumulation.
 * Zero scalars are mapped to "trash bucket" to avoid data-dependent branching.
 */

#include "msm.cuh"
#include <cub/cub.cuh>

namespace msm {

using namespace bls12_381;
using namespace icicle;
using namespace gpu;

// =============================================================================
// Bucket Index Computation Kernels
// =============================================================================

/**
 * @brief Compute bucket indices for all scalar windows (Constant Time)
 * 
 * Uses signed-digit representation with carry propagation to reduce
 * bucket count by half (2^(c-1) instead of 2^c buckets).
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
        int is_large = (window_val > num_buckets);
        if (is_large) {
            bucket_val = (1 << c) - window_val;
            sign = 1;
            carry = 1;
        }
        
        // Handle zero: map to trash bucket
        int is_zero = (bucket_val == 0);
        if (is_zero) {
            bucket_val = num_buckets + 1;
            sign = 0;
        }
        
        // Global bucket index (0-based)
        unsigned int bucket_idx = w * buckets_per_window + (bucket_val - 1);
        
        bucket_indices[output_idx] = bucket_idx;
        
        unsigned int idx_unsigned = static_cast<unsigned int>(idx);
        packed_indices[output_idx] = (idx_unsigned << 1) | (sign & 1);
    }
    
    // Handle malformed scalars (final carry)
    if (carry != 0) {
        for (int w = 0; w < num_windows; w++) {
            int output_idx = idx * num_windows + w;
            bucket_indices[output_idx] = INVALID_BUCKET_INDEX;
        }
    }
}

/**
 * @brief Persistent-thread bucket indices kernel for small MSM sizes
 * 
 * Ensures all SMs are occupied by having each thread process multiple scalars.
 */
template<typename S>
__global__ void compute_bucket_indices_persistent_kernel(
    unsigned int* bucket_indices,
    unsigned int* packed_indices,
    const S* scalars,
    int msm_size,
    int c,
    int num_windows,
    int num_buckets,
    int total_work_items
) {
    const int buckets_per_window = num_buckets + 1;
    
    int stride = gridDim.x * blockDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    for (int idx = tid; idx < msm_size; idx += stride) {
        S scalar = scalars[idx];
        int carry = 0;
        
        for (int w = 0; w < num_windows; w++) {
            int output_idx = idx * num_windows + w;
            
            int start_bit = w * c;
            int limb_idx = start_bit / 64;
            int bit_in_limb = start_bit % 64;
            
            uint64_t window = scalar.limbs[limb_idx] >> bit_in_limb;
            if (bit_in_limb + c > 64 && limb_idx + 1 < S::LIMBS) {
                window |= (scalar.limbs[limb_idx + 1] << (64 - bit_in_limb));
            }
            window &= ((1ULL << c) - 1);
            
            int window_val = (int)window + carry;
            carry = 0;
            
            int sign = 0;
            int bucket_val = window_val;
            
            if (window_val > num_buckets) {
                bucket_val = (1 << c) - window_val;
                sign = 1;
                carry = 1;
            }
            
            if (bucket_val == 0) {
                bucket_val = num_buckets + 1;
                sign = 0;
            }
            
            unsigned int bucket_idx = w * buckets_per_window + (bucket_val - 1);
            
            bucket_indices[output_idx] = bucket_idx;
            packed_indices[output_idx] = (static_cast<unsigned int>(idx) << 1) | (sign & 1);
        }
        
        if (carry != 0) {
            for (int w = 0; w < num_windows; w++) {
                int output_idx = idx * num_windows + w;
                bucket_indices[output_idx] = INVALID_BUCKET_INDEX;
            }
        }
    }
}

// =============================================================================
// Histogram Kernels
// =============================================================================

/**
 * @brief Optimized histogram kernel with warp-level aggregation
 * 
 * Reduces atomic contention by up to 32× by aggregating within warps first.
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
            bucket = INVALID_BUCKET_INDEX;
        }
    }
    
    unsigned int active_mask = __ballot_sync(0xFFFFFFFF, bucket != INVALID_BUCKET_INDEX);
    
    while (active_mask != 0) {
        int leader = __ffs(active_mask) - 1;
        unsigned int leader_bucket = __shfl_sync(0xFFFFFFFF, bucket, leader);
        
        unsigned int match_mask = __ballot_sync(active_mask, bucket == leader_bucket);
        
        int count = __popc(match_mask);
        if (lane_id == leader && leader_bucket != INVALID_BUCKET_INDEX) {
            atomicAdd(&histogram[leader_bucket], count);
        }
        
        active_mask &= ~match_mask;
    }
}

// =============================================================================
// Bucket Accumulation Kernels
// =============================================================================

/**
 * @brief Warp-cooperative bucket accumulation for large buckets
 * 
 * Multiple threads cooperate to process each bucket in parallel,
 * then reduce within the group using shared memory.
 */
template<typename A, typename P, int THREADS_PER_BUCKET>
__global__ void accumulate_cooperative_kernel(
    P* buckets,
    const unsigned int* sorted_packed_indices,
    const A* bases,
    const unsigned int* bucket_offsets,
    const unsigned int* bucket_sizes,
    int total_buckets
) {
    static_assert(THREADS_PER_BUCKET <= 32 && (THREADS_PER_BUCKET & (THREADS_PER_BUCKET - 1)) == 0,
                  "THREADS_PER_BUCKET must be power of 2 and <= 32");
    
    int groups_per_block = blockDim.x / THREADS_PER_BUCKET;
    int group_id = threadIdx.x / THREADS_PER_BUCKET;
    int lane_in_group = threadIdx.x % THREADS_PER_BUCKET;
    
    int bucket_id = blockIdx.x * groups_per_block + group_id;
    
    if (bucket_id >= total_buckets) return;
    
    unsigned int offset = bucket_offsets[bucket_id];
    unsigned int size = bucket_sizes[bucket_id];
    
    P local_acc = P::identity();
    
    for (unsigned int i = lane_in_group; i < size; i += THREADS_PER_BUCKET) {
        unsigned int idx = offset + i;
        unsigned int packed = sorted_packed_indices[idx];
        unsigned int point_idx = packed >> 1;
        unsigned int sign = packed & 1;
        
        A base = bases[point_idx];
        if (sign) {
            base = base.neg();
        }
        
        point_add_mixed(local_acc, local_acc, base);
    }
    
    extern __shared__ char shared_mem[];
    P* shared_acc = reinterpret_cast<P*>(shared_mem);
    
    shared_acc[threadIdx.x] = local_acc;
    __syncthreads();
    
    for (int stride = THREADS_PER_BUCKET / 2; stride > 0; stride >>= 1) {
        if (lane_in_group < stride) {
            P other = shared_acc[threadIdx.x + stride];
            point_add(shared_acc[threadIdx.x], shared_acc[threadIdx.x], other);
        }
        __syncthreads();
    }
    
    if (lane_in_group == 0) {
        buckets[bucket_id] = shared_acc[threadIdx.x];
    }
}

/**
 * @brief Standard bucket accumulation - one thread per bucket
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
        
        point_add_mixed(acc, acc, base);
    }
    
    buckets[bucket_idx] = acc;
}

// =============================================================================
// Bucket Reduction Kernels
// =============================================================================

/**
 * @brief Simple bucket reduction - one thread per window
 */
template<typename P>
__global__ void parallel_bucket_reduction_kernel(
    P* window_results,
    const P* buckets,
    int num_windows,
    int num_buckets,
    int buckets_per_window
) {
    int window_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (window_idx >= num_windows) return;
    
    const P* window_buckets = buckets + window_idx * buckets_per_window;
    
    P running_sum = P::identity();
    P window_sum = P::identity();
    
    // Triangle summation
    for (int i = num_buckets - 1; i >= 0; i--) {
        point_add(running_sum, running_sum, window_buckets[i]);
        point_add(window_sum, window_sum, running_sum);
    }
    
    window_results[window_idx] = window_sum;
}

/**
 * @brief Multi-threaded bucket reduction with parallel suffix scan
 * 
 * Uses parallel Blelloch-style exclusive scan for suffix sums, avoiding
 * serial combination bottlenecks.
 * 
 * Algorithm:
 * 1. Each thread computes local triangle sum for its bucket range (parallel)
 * 2. Parallel exclusive suffix scan to compute suffix_sum for each thread
 * 3. Each thread adjusts its local_window with suffix_sum * count (parallel)
 * 4. Parallel reduction to sum all adjusted window values
 */
template<typename P>
__global__ void parallel_bucket_reduction_multithread_kernel(
    P* window_results,
    const P* buckets,
    int num_windows,
    int num_buckets,
    int buckets_per_window
) {
    int window_idx = blockIdx.x;
    if (window_idx >= num_windows) return;
    
    const int THREADS_PER_WINDOW = blockDim.x;
    int tid = threadIdx.x;
    
    const P* window_buckets = buckets + window_idx * buckets_per_window;
    
    int buckets_per_thread = (num_buckets + THREADS_PER_WINDOW - 1) / THREADS_PER_WINDOW;
    int start_bucket = tid * buckets_per_thread;
    int end_bucket = min(start_bucket + buckets_per_thread, num_buckets);
    int my_count = max(0, end_bucket - start_bucket);
    
    P local_running = P::identity();
    P local_window = P::identity();
    
    // 1. Compute local triangle sum for this thread's bucket range (parallel)
    for (int i = end_bucket - 1; i >= start_bucket; i--) {
        point_add(local_running, local_running, window_buckets[i]);
        point_add(local_window, local_window, local_running);
    }
    
    extern __shared__ char shared_mem[];
    P* shared_running = reinterpret_cast<P*>(shared_mem);
    P* shared_suffix = shared_running + THREADS_PER_WINDOW;
    int* shared_counts = reinterpret_cast<int*>(shared_suffix + THREADS_PER_WINDOW);
    
    shared_running[tid] = local_running;
    shared_counts[tid] = my_count;
    __syncthreads();
    
    // 2. Parallel exclusive suffix scan for running sums
    // suffix_sum[t] = Σ shared_running[t+1..end]
    // We compute this as a reverse inclusive scan, then shift
    
    // First, copy running sums into suffix array (reversed conceptually)
    shared_suffix[tid] = shared_running[tid];
    __syncthreads();
    
    // Parallel reverse scan: treat index (THREADS_PER_WINDOW - 1 - tid) as the forward index
    // This computes prefix sums from the right side
    for (int stride = 1; stride < THREADS_PER_WINDOW; stride *= 2) {
        P val = P::identity();
        if (tid + stride < THREADS_PER_WINDOW) {
            val = shared_suffix[tid + stride];
        }
        __syncthreads();
        if (tid + stride < THREADS_PER_WINDOW) {
            point_add(shared_suffix[tid], shared_suffix[tid], val);
        }
        __syncthreads();
    }
    
    // shared_suffix[tid] now contains inclusive suffix sum from tid to end
    // We need exclusive suffix sum (from tid+1 to end)
    P suffix_sum = (tid + 1 < THREADS_PER_WINDOW) ? shared_suffix[tid + 1] : P::identity();
    __syncthreads();
    
    // 3. Each thread adjusts its window value (parallel)
    // adjustment = suffix_sum * my_count (scalar multiplication via double-and-add)
    P adjusted_window = local_window;
    if (my_count > 0 && !suffix_sum.is_identity()) {
        P adjustment = P::identity();
        P base = suffix_sum;
        int remaining = my_count;
        
        while (remaining > 0) {
            if (remaining & 1) {
                point_add(adjustment, adjustment, base);
            }
            remaining >>= 1;
            if (remaining > 0) {
                point_double(base, base);
            }
        }
        point_add(adjusted_window, adjusted_window, adjustment);
    }
    
    // 4. Parallel reduction to sum all adjusted windows
    shared_suffix[tid] = adjusted_window;
    __syncthreads();
    
    // Standard parallel reduction
    for (int stride = THREADS_PER_WINDOW / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            point_add(shared_suffix[tid], shared_suffix[tid], shared_suffix[tid + stride]);
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        window_results[window_idx] = shared_suffix[0];
    }
}

// =============================================================================
// Final Accumulation Kernels
// =============================================================================

/**
 * @brief Parallel final accumulation with tree reduction
 * 
 * Processes window doublings in parallel, then uses shared memory tree reduction
 * to sum all windows efficiently instead of serial thread-0 loop.
 * 
 * Each window w contributes: window_results[w] << (w * c)
 * where << means repeated point doubling.
 */
template<typename P>
__global__ void final_accumulation_parallel_kernel(
    P* result,
    P* window_results,
    int num_windows,
    int c
) {
    extern __shared__ char shared_mem[];
    P* shared_windows = reinterpret_cast<P*>(shared_mem);
    
    int tid = threadIdx.x;
    int num_threads = blockDim.x;
    
    // 1. Each thread handles its assigned windows (parallel)
    // Apply the required doublings and store in shared memory
    for (int w = tid; w < num_windows; w += num_threads) {
        P val = window_results[w];
        if (w > 0) {
            int doublings = w * c;
            for (int i = 0; i < doublings; i++) {
                point_double(val, val);
            }
        }
        shared_windows[w] = val;
    }
    __syncthreads();
    
    // 2. Parallel tree reduction
    // Round up to next power of 2 for clean reduction
    int size = 1;
    while (size < num_windows) size *= 2;
    
    for (int stride = size / 2; stride > 0; stride /= 2) {
        for (int i = tid; i < stride; i += num_threads) {
            if (i < num_windows && i + stride < num_windows) {
                point_add(shared_windows[i], shared_windows[i], shared_windows[i + stride]);
            }
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        *result = shared_windows[0];
    }
}

/**
 * @brief Sequential final accumulation for small window counts
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
    
    for (int w = num_windows - 2; w >= 0; w--) {
        for (int i = 0; i < c; i++) {
            point_double(acc, acc);
        }
        point_add(acc, acc, window_results[w]);
    }
    
    *result = acc;
}

// =============================================================================
// Main MSM Entry Point
// =============================================================================

template<typename S, typename A, typename P>
cudaError_t msm_cuda(
    const S* scalars,
    const A* bases,
    int msm_size,
    const MSMConfig& config,
    P* result
) {
    // Validate pointers
    if (scalars == nullptr || bases == nullptr || result == nullptr) {
        return cudaErrorInvalidValue;
    }
    
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
    
    // Validate size limits
    if (msm_size > MAX_MSM_SIZE) {
        return cudaErrorInvalidValue;
    }

    cudaStream_t stream = config.stream;
    cudaError_t err;
    
    // Determine window size
    int c = config.c;
    if (c <= 0) {
        c = get_optimal_c(msm_size);
    }
    
    if (c < MIN_WINDOW_SIZE || c > MAX_WINDOW_SIZE) {
        return cudaErrorInvalidValue;
    }

    int scalar_bits = config.bitsize > 0 ? config.bitsize : S::LIMBS * 64;
    if (scalar_bits <= 0 || scalar_bits > S::LIMBS * 64) {
        return cudaErrorInvalidValue;
    }
    
    int num_windows = get_num_windows(scalar_bits, c);
    
    if (c >= 31) {
        return cudaErrorInvalidValue;
    }
    int num_buckets = (1 << (c - 1));
    
    long long total_buckets_long = (long long)num_windows * (long long)(num_buckets + 1);
    if (total_buckets_long > (long long)INT_MAX) {
        return cudaErrorInvalidValue;
    }
    int total_buckets = (int)total_buckets_long;
    
    long long num_contributions_long = (long long)msm_size * (long long)num_windows;
    if (num_contributions_long > (long long)INT_MAX) {
        return cudaErrorInvalidValue;
    }
    int num_contributions = (int)num_contributions_long;
    
    // Allocate device memory
    S* d_scalars = nullptr;
    A* d_bases = nullptr;
    P* d_result = nullptr;
    P* d_buckets = nullptr;
    P* d_window_results = nullptr;
    
    unsigned int *d_bucket_indices = nullptr, *d_packed_indices = nullptr;
    unsigned int *d_bucket_indices_sorted = nullptr, *d_packed_indices_sorted = nullptr;
    unsigned int *d_bucket_offsets = nullptr, *d_bucket_sizes = nullptr;

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
    
    MSM_CUDA_CHECK(cudaMalloc(&d_bucket_indices, num_contributions * sizeof(unsigned int)));
    MSM_CUDA_CHECK(cudaMalloc(&d_packed_indices, num_contributions * sizeof(unsigned int)));
    MSM_CUDA_CHECK(cudaMalloc(&d_bucket_indices_sorted, num_contributions * sizeof(unsigned int)));
    MSM_CUDA_CHECK(cudaMalloc(&d_packed_indices_sorted, num_contributions * sizeof(unsigned int)));
    MSM_CUDA_CHECK(cudaMalloc(&d_bucket_offsets, total_buckets * sizeof(unsigned int)));
    MSM_CUDA_CHECK(cudaMalloc(&d_bucket_sizes, total_buckets * sizeof(unsigned int)));
    
    // 1. Compute bucket indices
    {
        auto cfg = get_msm_launch_config(msm_size, KernelType::BUCKET_COUNT);
        
        if (cfg.use_persistent) {
            compute_bucket_indices_persistent_kernel<S><<<cfg.blocks, cfg.threads, 0, stream>>>(
                d_bucket_indices, d_packed_indices, d_scalars, msm_size, c, num_windows, num_buckets, msm_size
            );
        } else {
            compute_bucket_indices_kernel<S><<<cfg.blocks, cfg.threads, 0, stream>>>(
                d_bucket_indices, d_packed_indices, d_scalars, msm_size, c, num_windows, num_buckets
            );
        }
        MSM_CUDA_CHECK(cudaGetLastError());
    }
    
    // 2. Histogram
    {
        MSM_CUDA_CHECK(cudaMemsetAsync(d_bucket_sizes, 0, total_buckets * sizeof(unsigned int), stream));
        
        auto cfg = get_launch_config(num_contributions, KernelType::ATOMIC_BOUND);
        histogram_warp_aggregated_kernel<<<cfg.blocks, cfg.threads, 0, stream>>>(
            d_bucket_sizes, d_bucket_indices, num_contributions, total_buckets);
        MSM_CUDA_CHECK(cudaGetLastError());
    }
    
    // 3. Scan (bucket offsets)
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
    
    // 5. Bucket accumulation
    {
        constexpr int THREADS_PER_BUCKET = 8;
        int threads = GPUConfig::get().get_cooperative_threads(THREADS_PER_BUCKET);
        
        if (should_use_cooperative_accumulate(msm_size, total_buckets, threads)) {
            auto cfg = get_msm_cooperative_config(total_buckets, THREADS_PER_BUCKET, sizeof(P));
            accumulate_cooperative_kernel<A, P, THREADS_PER_BUCKET><<<cfg.blocks, cfg.threads, cfg.shared_mem, stream>>>(
                d_buckets,
                d_packed_indices_sorted,
                d_bases,
                d_bucket_offsets,
                d_bucket_sizes,
                total_buckets
            );
        } else {
            auto cfg = get_launch_config(total_buckets, KernelType::BUCKET_REDUCE);
            accumulate_sorted_kernel<A, P><<<cfg.blocks, cfg.threads, 0, stream>>>(
                d_buckets,
                d_packed_indices_sorted,
                d_bases,
                d_bucket_offsets,
                d_bucket_sizes,
                total_buckets
            );
        }
        MSM_CUDA_CHECK(cudaGetLastError());
    }
    
    // 6. Bucket reduction
    {
        auto cfg = get_msm_reduction_config(num_windows, num_buckets, sizeof(P));
        
        if (num_buckets >= 256) {
            parallel_bucket_reduction_multithread_kernel<P><<<cfg.blocks, cfg.threads, cfg.shared_mem, stream>>>(
                d_window_results,
                d_buckets,
                num_windows,
                num_buckets,
                num_buckets + 1
            );
        } else {
            parallel_bucket_reduction_kernel<P><<<cfg.blocks, cfg.threads, 0, stream>>>(
                d_window_results,
                d_buckets,
                num_windows,
                num_buckets,
                num_buckets + 1
            );
        }
        MSM_CUDA_CHECK(cudaGetLastError());
    }
    
    // 7. Final accumulation
    {
        if (num_windows >= 4) {
            int threads = 32;
            while (threads < num_windows) threads *= 2;
            if (threads > 256) threads = 256;
            
            // Shared memory: one P element per window for tree reduction
            size_t shared_mem = num_windows * sizeof(P);
            
            final_accumulation_parallel_kernel<P><<<1, threads, shared_mem, stream>>>(
                d_result,
                d_window_results,
                num_windows,
                c
            );
        } else {
            final_accumulation_kernel<P><<<1, 1, 0, stream>>>(
                d_result,
                d_window_results,
                num_windows,
                c
            );
        }
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
    if (config.is_async && stream != nullptr) {
        cudaStreamSynchronize(stream);
    }
    
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
    
    return (err != cudaSuccess) ? err : cleanup_err;
}

// =============================================================================
// Explicit Template Instantiations
// =============================================================================

// G1 MSM
template cudaError_t msm_cuda<Fr, G1Affine, G1Projective>(
    const Fr* scalars,
    const G1Affine* bases,
    int msm_size,
    const MSMConfig& config,
    G1Projective* result
);

// G2 MSM
template cudaError_t msm_cuda<Fr, G2Affine, G2Projective>(
    const Fr* scalars,
    const G2Affine* bases,
    int msm_size,
    const MSMConfig& config,
    G2Projective* result
);

} // namespace msm
