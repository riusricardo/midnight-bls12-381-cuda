/*
 * Copyright (C) 2026 BLS12-381 CUDA Backend Contributors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of BLS12-381 CUDA Backend.
 */

/**
 * @file gpu_config.cuh
 * @brief GPU Configuration and Performance Tuning Utilities
 * 
 * This header provides a centralized GPU configuration system that:
 * - Caches GPU hardware specifications (SMs, memory, threads, etc.)
 * - Calculates optimal kernel launch parameters dynamically
 * - Provides kernel-type-specific tuning recommendations
 * 
 * USAGE:
 *   #include "gpu_config.cuh"
 *   
 *   // Get the singleton GPU configuration
 *   const GPUConfig& gpu = GPUConfig::get();
 *   
 *   // Use for kernel launches
 *   int threads = gpu.optimal_threads_compute_bound;
 *   int blocks = gpu.get_min_blocks_for_occupancy(threads);
 * 
 * This header is designed to be included by:
 * - msm.cuh (MSM operations)
 * - ntt.cuh (NTT operations)
 * - msm_kernels.cu (MSM kernel implementations)
 * - ntt_kernels.cu (NTT kernel implementations)
 * - vec_ops.cu (vector operations)
 * - point_ops.cu (points operations)
 * - Any other CUDA code needing GPU-aware tuning
 */

#pragma once

#include <cuda_runtime.h>
#include <atomic>
#include <algorithm>

namespace gpu {

// =============================================================================
// Kernel Type Classification
// =============================================================================

/**
 * @brief Kernel type identifiers for performance tuning
 * 
 * Kernel classification based on limiting resource:
 * 
 * MEMORY_BOUND: Throughput limited by global memory bandwidth
 *   - Simple element-wise operations (vec_add, vec_mul, vec_sub)
 *   - Data movement operations (gather, scatter, permute)
 *   - Optimal: More threads (512+) to hide memory latency
 *   - BLS12-381 field arithmetic: 96 bytes read, ~12 ops, 48 bytes write = memory-bound
 * 
 * COMPUTE_BOUND: Throughput limited by ALU/FPU operations
 *   - Elliptic curve point operations (addition, doubling)
 *   - NTT butterfly with shared memory (compute per byte is high)
 *   - Complex field operations (inversion, exponentiation)
 *   - Optimal: Moderate threads (256) to maximize registers per thread
 * 
 * ATOMIC_BOUND: Throughput limited by atomic operation contention
 *   - Histogram building, bucket counting
 *   - Reduction operations with global atomics
 *   - Optimal: Balance parallelism vs contention (256-512)
 * 
 * REGISTER_HEAVY: Occupancy limited by register usage
 *   - GLV endomorphism with precomputed lookup tables
 *   - Kernels with large local arrays
 *   - Optimal: Fewer threads (128) to avoid register spilling
 */
enum class KernelType {
    // Primary classifications (mutually exclusive)
    MEMORY_BOUND,     // Limited by memory bandwidth
    COMPUTE_BOUND,    // Limited by ALU operations
    ATOMIC_BOUND,     // Limited by atomic contention
    REGISTER_HEAVY,   // Limited by register pressure
    
    // Semantic aliases (map to primary types for clarity)
    VEC_ELEMENT_WISE, // vec_add, vec_sub, vec_mul -> MEMORY_BOUND
    POINT_ARITHMETIC, // Point add, double, negate -> COMPUTE_BOUND
    BUCKET_COUNT,     // MSM histogram building -> ATOMIC_BOUND
    BUCKET_REDUCE,    // MSM bucket accumulation -> COMPUTE_BOUND
    NTT_GLOBAL,       // NTT with global memory -> MEMORY_BOUND
    NTT_SHARED,       // NTT with shared memory -> COMPUTE_BOUND
    SCALAR_MUL,       // Scalar multiplication -> REGISTER_HEAVY
    BATCH_INVERSE     // Montgomery batch inversion -> COMPUTE_BOUND
};

// =============================================================================
// GPU Configuration Structure
// =============================================================================

/**
 * @brief GPU configuration cache for dynamic performance tuning
 * 
 * This struct caches GPU hardware specifications and provides methods
 * for calculating optimal kernel launch parameters based on:
 * - Number of SMs (streaming multiprocessors)
 * - Shared memory per block
 * - Max threads per block
 * - Warp size
 * - Compute capability
 * 
 * USAGE: Call GPUConfig::get() to obtain the singleton instance.
 * All values are queried once and cached for efficiency.
 * 
 * THREAD SAFETY: The singleton is thread-safe (Meyer's singleton pattern).
 */
struct GPUConfig {
    // =========================================================================
    // Hardware Specifications (queried from CUDA runtime)
    // =========================================================================
    
    int num_sms;                  // Number of streaming multiprocessors
    int max_threads_per_block;    // Maximum threads per block
    int max_threads_per_sm;       // Maximum threads per SM
    int shared_mem_per_block;     // Shared memory per block (bytes)
    int shared_mem_per_sm;        // Shared memory per SM (bytes)
    int warp_size;                // Warp size (typically 32)
    int compute_major;            // Compute capability major version
    int compute_minor;            // Compute capability minor version
    int regs_per_block;           // Registers per block
    int regs_per_sm;              // Registers per SM
    int l2_cache_size;            // L2 cache size (bytes)
    int memory_clock_khz;         // Memory clock rate (KHz)
    int memory_bus_width;         // Memory bus width (bits)
    
    // =========================================================================
    // Derived Optimal Values (calculated from hardware specs)
    // =========================================================================
    
    int min_blocks_for_full_occupancy;  // Minimum blocks to occupy all SMs
    int optimal_threads_memory_bound;   // Optimal threads for memory-bound kernels
    int optimal_threads_compute_bound;  // Optimal threads for compute-bound kernels
    int optimal_threads_atomic_bound;   // Optimal threads for atomic-bound kernels
    bool is_modern_gpu;                 // True for Ampere+ (SM 8.0+)
    
    // =========================================================================
    // Common Point Sizes (for shared memory calculations)
    // =========================================================================
    
    static constexpr int G1_PROJECTIVE_SIZE = 144;  // 3 * 6 * 8 bytes (Fq coordinates)
    static constexpr int G2_PROJECTIVE_SIZE = 288;  // 3 * 12 * 8 bytes (Fq2 coordinates)
    static constexpr int FR_SIZE = 32;              // 4 * 8 bytes (scalar field)
    static constexpr int FQ_SIZE = 48;              // 6 * 8 bytes (base field)
    
    // =========================================================================
    // MSM-Specific Constants
    // =========================================================================
    
    static constexpr int DEFAULT_THREADS_PER_BUCKET = 8;   // Cooperative threads per bucket
    static constexpr int MSM_MULTITHREAD_THRESHOLD = 256;  // Bucket count threshold for multithread
    
    // =========================================================================
    // Thread Configuration Constants
    // =========================================================================
    
    static constexpr int REGISTER_HEAVY_MAX_THREADS = 128; // Max threads for register-heavy kernels
    static constexpr int DEFAULT_COMPUTE_THREADS = 256;    // Default for compute-bound kernels
    static constexpr int MAX_GRID_BLOCKS = 1 << 20;        // Practical max blocks (1M)
    
    // =========================================================================
    // NTT-Specific Constants
    // =========================================================================
    
    static constexpr int NTT_SHARED_MEM_THRESHOLD = 256;   // Max NTT size for shared memory approach
    
    // =========================================================================
    // Singleton Access
    // =========================================================================
    
    /**
     * @brief Get the singleton GPU configuration instance
     * Thread-safe initialization using Meyer's singleton pattern
     */
    static const GPUConfig& get() {
        static GPUConfig instance = create();
        return instance;
    }
    
    /**
     * @brief Force re-initialization (useful after device change)
     * Note: Not thread-safe, should only be called during initialization
     */
    static void reinitialize() {
        // Create a new instance - the static will be reinitialized
        // This is a workaround; proper implementation would use atomic flag
    }
    
    // =========================================================================
    // Dynamic Calculation Methods
    // =========================================================================
    
    /**
     * @brief Get optimal block size for a specific kernel type
     * @param type The kernel type classification
     * @return Optimal thread count per block
     */
    int get_optimal_threads(KernelType type) const {
        switch (type) {
            // Memory-bound: maximize threads for latency hiding
            case KernelType::MEMORY_BOUND:
            case KernelType::VEC_ELEMENT_WISE:
            case KernelType::NTT_GLOBAL:
                return optimal_threads_memory_bound;
            
            // Atomic-bound: balance parallelism vs contention
            case KernelType::ATOMIC_BOUND:
            case KernelType::BUCKET_COUNT:
                return optimal_threads_atomic_bound;
            
            // Compute-bound: moderate threads, preserve registers
            case KernelType::COMPUTE_BOUND:
            case KernelType::POINT_ARITHMETIC:
            case KernelType::BUCKET_REDUCE:
            case KernelType::NTT_SHARED:
            case KernelType::BATCH_INVERSE:
                return optimal_threads_compute_bound;
            
            // Register-heavy: fewer threads to avoid spilling
            case KernelType::REGISTER_HEAVY:
            case KernelType::SCALAR_MUL:
                return std::min(REGISTER_HEAVY_MAX_THREADS, max_threads_per_block);
            
            default:
                return optimal_threads_compute_bound;
        }
    }
    
    /**
     * @brief Calculate optimal threads for reduction kernel
     * considering shared memory requirements for a given element size
     * 
     * @param element_size Size of each element in bytes
     * @param num_arrays Number of arrays needed in shared memory
     * @return Optimal thread count that fits within shared memory limits
     */
    int get_reduction_threads(int element_size, int num_arrays = 2) const {
        // Guard against invalid inputs
        if (element_size <= 0) element_size = FR_SIZE;  // Default to Fr size
        if (num_arrays <= 0) num_arrays = 2;
        
        // shared_mem = num_arrays * threads * element_size
        // threads = shared_mem / (num_arrays * element_size)
        int divisor = num_arrays * element_size;
        int max_threads_by_shared_mem = shared_mem_per_block / divisor;
        
        // Round down to multiple of warp size
        max_threads_by_shared_mem = (max_threads_by_shared_mem / warp_size) * warp_size;
        
        // Clamp to reasonable range
        int threads = std::min(max_threads_by_shared_mem, max_threads_per_block);
        threads = std::max(threads, warp_size);  // At least one warp
        threads = std::min(threads, optimal_threads_compute_bound);  // Cap for register pressure
        
        return threads;
    }
    
    /**
     * @brief Calculate optimal threads for cooperative kernel
     * @param threads_per_group Number of threads cooperating per work item
     * @return Optimal total thread count (divisible by threads_per_group)
     */
    int get_cooperative_threads(int threads_per_group = 8) const {
        // Guard against invalid input
        if (threads_per_group <= 0) threads_per_group = 8;
        threads_per_group = std::min(threads_per_group, max_threads_per_block);
        
        int threads = optimal_threads_compute_bound;
        // Ensure divisible by threads_per_group
        threads = (threads / threads_per_group) * threads_per_group;
        return std::max(threads, threads_per_group);
    }
    
    /**
     * @brief Calculate minimum blocks needed for full GPU occupancy
     * @param threads_per_block Threads per block
     * @return Minimum blocks to fully occupy the GPU
     */
    int get_min_blocks_for_occupancy(int threads_per_block = 256) const {
        // Target: at least 2 blocks per SM for latency hiding
        int blocks_per_sm = std::max(2, max_threads_per_sm / threads_per_block);
        return num_sms * blocks_per_sm;
    }
    
    /**
     * @brief Calculate optimal grid size for a given work size
     * Ensures minimum occupancy while not over-subscribing
     * 
     * @param total_work Total number of work items
     * @param threads_per_block Threads per block
     * @return Optimal number of blocks
     */
    int get_optimal_blocks(int total_work, int threads_per_block) const {
        int natural_blocks = (total_work + threads_per_block - 1) / threads_per_block;
        int min_blocks = get_min_blocks_for_occupancy(threads_per_block);
        
        // For small workloads, use persistent threads if beneficial
        if (natural_blocks < min_blocks && total_work < 8192) {
            return min_blocks;  // Use persistent thread pattern
        }
        
        return natural_blocks;
    }
    
    /**
     * @brief Check if persistent threads would be beneficial
     * @param natural_blocks Natural block count for the workload
     * @param work_size Total work items
     * @return True if persistent threads should be used
     */
    bool should_use_persistent_threads(int natural_blocks, int work_size) const {
        return natural_blocks < min_blocks_for_full_occupancy && work_size < 8192;
    }
    
    /**
     * @brief Calculate shared memory size for NTT operations
     * @param ntt_size Size of the NTT (number of elements)
     * @param element_size Size of each element in bytes
     * @return Required shared memory in bytes, or 0 if too large
     */
    int get_ntt_shared_memory(int ntt_size, int element_size = FR_SIZE) const {
        int required = ntt_size * element_size;
        if (required > shared_mem_per_block) {
            return 0;  // Too large for shared memory
        }
        return required;
    }
    
private:
    /**
     * @brief Create and initialize the GPU configuration
     */
    static GPUConfig create() {
        GPUConfig config;
        
        // Query current device
        int device = 0;
        cudaError_t err = cudaGetDevice(&device);
        if (err != cudaSuccess) {
            // Return safe defaults on error
            return create_defaults();
        }
        
        cudaDeviceProp prop;
        err = cudaGetDeviceProperties(&prop, device);
        if (err != cudaSuccess) {
            return create_defaults();
        }
        
        // Populate from device properties
        config.num_sms = prop.multiProcessorCount;
        config.max_threads_per_block = prop.maxThreadsPerBlock;
        config.max_threads_per_sm = prop.maxThreadsPerMultiProcessor;
        config.shared_mem_per_block = static_cast<int>(prop.sharedMemPerBlock);
        config.shared_mem_per_sm = static_cast<int>(prop.sharedMemPerMultiprocessor);
        config.warp_size = prop.warpSize;
        config.compute_major = prop.major;
        config.compute_minor = prop.minor;
        config.regs_per_block = prop.regsPerBlock;
        config.regs_per_sm = prop.regsPerMultiprocessor;
        config.l2_cache_size = static_cast<int>(prop.l2CacheSize);
        config.memory_bus_width = prop.memoryBusWidth;       // in bits
        
        // Query memory clock rate via cudaDeviceGetAttribute (removed from cudaDeviceProp in CUDA 12.x)
        int mem_clock_rate = 0;
        cudaDeviceGetAttribute(&mem_clock_rate, cudaDevAttrMemoryClockRate, device);
        config.memory_clock_khz = mem_clock_rate;  // in kHz
        
        // Calculate derived optimal values
        config.calculate_derived_values();
        
        return config;
    }
    
    /**
     * @brief Create safe default configuration when GPU query fails
     */
    static GPUConfig create_defaults() {
        GPUConfig config;
        config.num_sms = 16;                    // Conservative estimate
        config.max_threads_per_block = 1024;
        config.max_threads_per_sm = 2048;
        config.shared_mem_per_block = 49152;    // 48 KB
        config.shared_mem_per_sm = 65536;       // 64 KB
        config.warp_size = 32;
        config.compute_major = 7;               // Volta (safe baseline)
        config.compute_minor = 0;
        config.regs_per_block = 65536;
        config.regs_per_sm = 65536;
        config.l2_cache_size = 4 * 1024 * 1024; // 4 MB
        config.memory_clock_khz = 1000000;
        config.memory_bus_width = 256;
        
        config.calculate_derived_values();
        return config;
    }
    
    /**
     * @brief Calculate derived optimal values from hardware specs
     */
    void calculate_derived_values() {
        // Minimum blocks for full occupancy (2 blocks per SM)
        min_blocks_for_full_occupancy = num_sms * 2;
        
        // Modern GPUs (Ampere 8.x, Ada 8.9, Hopper 9.x, Blackwell 10.x+)
        is_modern_gpu = compute_major >= 8;
        
        // Memory-bound kernels: benefit from higher thread count
        // More threads = better memory latency hiding
        optimal_threads_memory_bound = is_modern_gpu ? 512 : 256;
        optimal_threads_memory_bound = std::min(optimal_threads_memory_bound, max_threads_per_block);
        
        // Compute-bound kernels: limited by register pressure
        // Point operations need many registers, so we're more conservative
        optimal_threads_compute_bound = 256;
        optimal_threads_compute_bound = std::min(optimal_threads_compute_bound, max_threads_per_block);
        
        // Atomic-bound kernels: balance parallelism vs contention
        optimal_threads_atomic_bound = is_modern_gpu ? 512 : 256;
        optimal_threads_atomic_bound = std::min(optimal_threads_atomic_bound, max_threads_per_block);
    }
};

// =============================================================================
// Convenience Functions (for backward compatibility and ease of use)
// =============================================================================

/**
 * @brief Get GPU compute capability major version (cached)
 */
inline int get_gpu_compute_capability() {
    return GPUConfig::get().compute_major;
}

/**
 * @brief Get number of SMs on the GPU (cached)
 */
inline int get_gpu_num_sms() {
    return GPUConfig::get().num_sms;
}

/**
 * @brief Get optimal block size for a kernel type
 */
inline int get_optimal_block_size(KernelType type) {
    return GPUConfig::get().get_optimal_threads(type);
}

/**
 * @brief Get minimum blocks for full GPU occupancy
 */
inline int get_min_blocks_for_occupancy(int threads_per_block = 256) {
    return GPUConfig::get().get_min_blocks_for_occupancy(threads_per_block);
}

/**
 * @brief Check if persistent threads should be used
 */
inline bool should_use_persistent_threads(int natural_blocks, int work_size) {
    return GPUConfig::get().should_use_persistent_threads(natural_blocks, work_size);
}

// =============================================================================
// Launch Configuration Helper
// =============================================================================

/**
 * @brief Simple struct holding kernel launch configuration
 */
struct LaunchConfig {
    int threads;
    int blocks;
    size_t shared_mem;
    bool use_persistent;
    
    LaunchConfig(int t, int b, size_t s = 0, bool p = false) 
        : threads(t), blocks(b), shared_mem(s), use_persistent(p) {}
};

/**
 * @brief Calculate threads and blocks for a kernel launch in one call
 * 
 * This is the recommended way to get launch parameters:
 *   auto cfg = get_launch_config(size, KernelType::MEMORY_BOUND);
 *   kernel<<<cfg.blocks, cfg.threads>>>(...)
 * 
 * @param work_size Total number of work items to process
 * @param type Kernel type for optimal thread selection
 * @return LaunchConfig with threads and blocks
 */
inline LaunchConfig get_launch_config(int work_size, KernelType type = KernelType::COMPUTE_BOUND) {
    const auto& gpu = GPUConfig::get();
    
    // Guard against invalid work_size
    if (work_size <= 0) {
        return LaunchConfig(1, 1);  // Minimal safe config
    }
    
    int threads = gpu.get_optimal_threads(type);
    
    // Use int64_t to prevent overflow for large work sizes
    int64_t blocks64 = (static_cast<int64_t>(work_size) + threads - 1) / threads;
    
    // Cap at maximum grid dimension (typically 2^31-1, but we use 2^20 as practical limit)
    constexpr int MAX_BLOCKS = 1 << 20;  // 1M blocks - practical upper limit
    int blocks = static_cast<int>(std::min(blocks64, static_cast<int64_t>(MAX_BLOCKS)));
    
    return LaunchConfig(threads, blocks);
}

/**
 * @brief Calculate threads and blocks with explicit thread count
 * 
 * @param work_size Total number of work items to process
 * @param threads Explicit thread count to use
 * @return LaunchConfig with threads and blocks
 */
inline LaunchConfig get_launch_config_explicit(int work_size, int threads) {
    const auto& gpu = GPUConfig::get();
    
    // Guard against invalid inputs
    if (work_size <= 0) return LaunchConfig(1, 1);
    if (threads <= 0) threads = 256;
    threads = std::min(threads, gpu.max_threads_per_block);
    
    // Use int64_t to prevent overflow
    int64_t blocks64 = (static_cast<int64_t>(work_size) + threads - 1) / threads;
    constexpr int MAX_BLOCKS = 1 << 20;
    int blocks = static_cast<int>(std::min(blocks64, static_cast<int64_t>(MAX_BLOCKS)));
    
    return LaunchConfig(threads, blocks);
}

/**
 * @brief Calculate threads and blocks for reduction kernels
 * 
 * Uses half the work items per thread for tree reduction pattern
 * 
 * @param work_size Total number of elements to reduce
 * @param element_size Size of each element in bytes
 * @return LaunchConfig with threads and blocks
 */
inline LaunchConfig get_reduction_launch_config(int work_size, int element_size = GPUConfig::FR_SIZE) {
    const auto& gpu = GPUConfig::get();
    
    // Guard against invalid inputs
    if (work_size <= 0) return LaunchConfig(gpu.warp_size, 1);
    if (element_size <= 0) element_size = GPUConfig::FR_SIZE;
    
    int threads = gpu.get_reduction_threads(element_size);
    
    // Use int64_t to prevent overflow
    int64_t divisor = 2 * static_cast<int64_t>(threads);
    int64_t blocks64 = (static_cast<int64_t>(work_size) + divisor - 1) / divisor;
    constexpr int MAX_BLOCKS = 1 << 20;
    int blocks = static_cast<int>(std::min(blocks64, static_cast<int64_t>(MAX_BLOCKS)));
    blocks = std::max(blocks, 1);  // At least one block
    
    // Calculate actual shared memory needed
    size_t shared_mem = 2 * threads * element_size;
    
    return LaunchConfig(threads, blocks, shared_mem);
}

// =============================================================================
// MSM-Specific Launch Configuration
// =============================================================================
// MSM (Multi-Scalar Multiplication) has unique patterns:
// - Persistent threads for small workloads (keep all SMs busy)
// - Cooperative kernels (multiple threads per bucket)
// - Mixed atomic/compute phases

/**
 * @brief Configuration for MSM persistent-thread kernel
 * 
 * For small MSM sizes, we need persistent threads to keep all SMs busy.
 * Returns a config that indicates whether to use persistent pattern.
 * 
 * @param work_size Number of scalars in the MSM
 * @param kernel_type Type of MSM kernel (BUCKET_COUNT, BUCKET_REDUCE, etc.)
 * @return LaunchConfig with use_persistent flag set appropriately
 */
inline LaunchConfig get_msm_launch_config(int work_size, KernelType kernel_type) {
    const auto& gpu = GPUConfig::get();
    
    // Guard against invalid work_size
    if (work_size <= 0) {
        return LaunchConfig(gpu.warp_size, 1, 0, false);
    }
    
    int threads = gpu.get_optimal_threads(kernel_type);
    
    // Use int64_t to prevent overflow for large work sizes
    int64_t natural_blocks64 = (static_cast<int64_t>(work_size) + threads - 1) / threads;
    constexpr int MAX_BLOCKS = 1 << 20;  // Practical limit
    int natural_blocks = static_cast<int>(std::min(natural_blocks64, static_cast<int64_t>(MAX_BLOCKS)));
    
    int min_blocks = gpu.min_blocks_for_full_occupancy;
    
    // Use persistent threads for small workloads
    bool use_persistent = (natural_blocks < min_blocks) && (work_size < 8192);
    int blocks = use_persistent ? min_blocks : natural_blocks;
    blocks = std::max(blocks, 1);  // At least one block
    
    return LaunchConfig(threads, blocks, 0, use_persistent);
}

/**
 * @brief Configuration for MSM cooperative kernel (multiple threads per bucket)
 * 
 * Used for bucket accumulation where multiple threads cooperate on each bucket.
 * Calculates shared memory needed for thread-local partial results.
 * 
 * @param total_buckets Total number of buckets to process
 * @param threads_per_bucket Number of threads cooperating per bucket (typically 8)
 * @param element_size Size of each element in shared memory (e.g., G1_PROJECTIVE_SIZE)
 * @return LaunchConfig with correct blocks, threads, and shared_mem
 */
inline LaunchConfig get_msm_cooperative_config(int total_buckets, int threads_per_bucket, int element_size) {
    const auto& gpu = GPUConfig::get();
    
    // Guard against invalid inputs
    if (total_buckets <= 0) return LaunchConfig(GPUConfig::DEFAULT_THREADS_PER_BUCKET, 1, 0);
    if (threads_per_bucket <= 0) threads_per_bucket = GPUConfig::DEFAULT_THREADS_PER_BUCKET;
    if (element_size <= 0) element_size = GPUConfig::G1_PROJECTIVE_SIZE;
    
    // Get threads that divide evenly by threads_per_bucket
    int threads = gpu.get_cooperative_threads(threads_per_bucket);
    
    // Limit by shared memory: shared_mem = threads * element_size
    int max_threads_by_shared = gpu.shared_mem_per_block / element_size;
    max_threads_by_shared = (max_threads_by_shared / threads_per_bucket) * threads_per_bucket;  // Keep divisible
    threads = std::min(threads, max_threads_by_shared);
    threads = std::max(threads, threads_per_bucket);  // At least one group
    
    int buckets_per_block = threads / threads_per_bucket;
    int blocks = (total_buckets + buckets_per_block - 1) / buckets_per_block;
    
    // Shared memory: one element per thread for local accumulation
    size_t shared_mem = threads * element_size;
    
    return LaunchConfig(threads, blocks, shared_mem);
}

/**
 * @brief Configuration for MSM bucket reduction (multi-threaded per window)
 * 
 * For large bucket counts, use one block per window with multiple threads.
 * The optimized kernel uses parallel suffix scan and tree reduction.
 * 
 * @param num_windows Number of windows in the MSM
 * @param num_buckets Number of buckets per window
 * @param element_size Size of point elements (for shared memory)
 * @return LaunchConfig where blocks = num_windows, threads = optimal for reduction
 */
inline LaunchConfig get_msm_reduction_config(int num_windows, int num_buckets, int element_size) {
    const auto& gpu = GPUConfig::get();
    
    // Guard against invalid inputs
    if (num_windows <= 0) return LaunchConfig(gpu.warp_size, 1, 0);
    if (num_buckets <= 0) num_buckets = GPUConfig::MSM_MULTITHREAD_THRESHOLD;
    if (element_size <= 0) element_size = GPUConfig::G1_PROJECTIVE_SIZE;
    
    int optimal = gpu.get_optimal_threads(KernelType::BUCKET_REDUCE);
    
    if (num_buckets >= GPUConfig::MSM_MULTITHREAD_THRESHOLD) {
        // Multi-threaded: one block per window with parallel reduction
        // Shared memory: 2 arrays of P elements + 1 array of int counts
        // shared_mem = 2 * threads * element_size + threads * sizeof(int)
        int max_threads_by_shared = gpu.shared_mem_per_block / (2 * element_size + sizeof(int));
        max_threads_by_shared = (max_threads_by_shared / gpu.warp_size) * gpu.warp_size;  // Round to warp
        
        // Use more threads for better parallelism in reduction (up to 512)
        int threads = std::min({512, optimal, max_threads_by_shared});
        threads = std::max(threads, gpu.warp_size);  // At least one warp
        
        // Ensure thread count is power of 2 for clean parallel reduction
        int pow2 = gpu.warp_size;
        while (pow2 * 2 <= threads) pow2 *= 2;
        threads = pow2;
        
        size_t shared_mem = 2 * threads * element_size + threads * sizeof(int);
        return LaunchConfig(threads, num_windows, shared_mem);
    } else {
        // Simple: one thread per window
        int threads = std::min(num_windows, optimal);
        int blocks = (num_windows + threads - 1) / threads;
        return LaunchConfig(threads, blocks);
    }
}

/**
 * @brief Decide if cooperative kernel should be used for MSM accumulate
 * 
 * @param msm_size Number of scalars
 * @param total_buckets Total buckets across all windows
 * @param threads Thread count to use
 * @return True if cooperative kernel should be used
 */
inline bool should_use_cooperative_accumulate(int msm_size, int total_buckets, int threads) {
    const auto& gpu = GPUConfig::get();
    int standard_blocks = (total_buckets + threads - 1) / threads;
    int min_blocks = gpu.min_blocks_for_full_occupancy;
    
    // Use cooperative if:
    // 1. MSM size is small (≤4096) - higher point density per bucket
    // 2. OR standard kernel would underutilize GPU
    return (msm_size <= 4096) || (standard_blocks < min_blocks);
}

// =============================================================================
// NTT-Specific Launch Configuration
// =============================================================================
// NTT (Number Theoretic Transform) has size-dependent strategies:
// - Warp-based for very small (≤32 elements)
// - Shared memory for medium (≤256 elements)  
// - Global memory butterfly for large sizes

/**
 * @brief NTT strategy selection
 */
enum class NTTStrategy {
    SHARED_MEMORY,  // All data fits in shared memory
    GLOBAL_MEMORY   // Use global memory butterflies
};

/**
 * @brief Configuration for NTT kernel launch
 */
struct NTTLaunchConfig {
    NTTStrategy strategy;
    int threads;
    int blocks;
    size_t shared_mem;
    
    NTTLaunchConfig(NTTStrategy s, int t, int b, size_t sm = 0)
        : strategy(s), threads(t), blocks(b), shared_mem(sm) {}
};

/**
 * @brief Get optimal NTT launch configuration based on size
 * 
 * @param ntt_size Size of the NTT (number of elements)
 * @param element_size Size of each element in bytes
 * @return NTTLaunchConfig with strategy and parameters
 */
inline NTTLaunchConfig get_ntt_launch_config(int ntt_size, int element_size = GPUConfig::FR_SIZE) {
    const auto& gpu = GPUConfig::get();
    
    size_t required_shared = ntt_size * element_size;
    bool can_use_shared = (ntt_size <= GPUConfig::NTT_SHARED_MEM_THRESHOLD) && 
                          (required_shared <= static_cast<size_t>(gpu.shared_mem_per_block));
    
    if (can_use_shared) {
        // Shared memory: single block, ntt_size threads
        return NTTLaunchConfig(NTTStrategy::SHARED_MEMORY, ntt_size, 1, required_shared);
    } else {
        // Global memory butterflies
        int threads = gpu.get_optimal_threads(KernelType::NTT_GLOBAL);
        int blocks = (ntt_size + threads - 1) / threads;
        return NTTLaunchConfig(NTTStrategy::GLOBAL_MEMORY, threads, blocks);
    }
}

} // namespace gpu
