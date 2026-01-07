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
 * @file ntt.cuh
 * @brief Number Theoretic Transform (NTT) - Header
 * 
 * This header provides:
 * - NTT Domain class for managing precomputed twiddle factors
 * - API declarations for NTT functions (implementations in ntt_kernels.cu)
 * - Coset NTT helper kernels (template kernels used by coset operations)
 * 
 * ARCHITECTURE NOTE:
 * ==================
 * CUDA static libraries require kernels to be defined in the same compilation
 * unit that calls them. Therefore:
 * 
 * - Core NTT kernels (bit-reverse, butterfly, scale) are defined in ntt_kernels.cu
 * - Only coset helper kernels are templates here (they get instantiated in ntt_kernels.cu)
 * - DO NOT add __global__ kernel implementations here unless they are templates
 *   that will be instantiated by each .cu file that uses them
 * 
 * Algorithm: Cooley-Tukey decimation-in-time with optimizations:
 * - Shared memory kernel for small sizes (≤ 1024)
 * - Fused 2-stage butterfly for larger sizes (50% fewer kernel launches)
 */

#pragma once

#include "field.cuh"
#include "icicle_types.cuh"
#include "gpu_config.cuh"
#include <cuda_runtime.h>
#include <mutex>

namespace ntt {

using namespace bls12_381;
using namespace gpu;

// Maximum log domain size supported
constexpr int MAX_LOG_DOMAIN_SIZE = 33;

// =============================================================================
// NTT Domain Management
// =============================================================================

/**
 * @brief NTT Domain containing precomputed twiddle factors
 * 
 * Stores precomputed values for efficient NTT and coset NTT operations:
 * - twiddles: powers of the primitive root of unity omega
 * - inv_twiddles: powers of omega^(-1) for inverse NTT
 * - coset_powers: powers of the coset generator g (g^0, g^1, ..., g^(n-1))
 * - coset_powers_inv: powers of g^(-1) for inverse coset NTT
 * 
 * Coset powers are lazily initialized on first use of coset NTT.
 */
template<typename F>
class Domain {
public:
    F* twiddles;              // Twiddle factors (powers of omega)
    F* inv_twiddles;          // Inverse twiddle factors
    F* coset_powers;          // Precomputed g^i for i in [0, n)
    F* coset_powers_inv;      // Precomputed g^(-i) for i in [0, n)
    F coset_gen;              // The coset generator g (if initialized)
    F domain_size_inv;        // 1/n for this domain
    int log_size;
    size_t size;
    bool coset_initialized;   // Whether coset powers have been computed
    
    // Global domain registry - declared extern, defined in ntt_kernels.cu
    static Domain* domains[MAX_LOG_DOMAIN_SIZE];
    static std::mutex domains_mutex;
    
    __host__ Domain() : twiddles(nullptr), inv_twiddles(nullptr), 
                        coset_powers(nullptr), coset_powers_inv(nullptr),
                        log_size(0), size(0), coset_initialized(false) {}
    
    __host__ ~Domain() {
        if (twiddles) cudaFree(twiddles);
        if (inv_twiddles) cudaFree(inv_twiddles);
        if (coset_powers) cudaFree(coset_powers);
        if (coset_powers_inv) cudaFree(coset_powers_inv);
    }
    
    /**
     * @brief Get domain for given log size
     */
    static Domain* get_domain(int log_size);
    
    /**
     * @brief Set domain for given log size
     */
    static void set_domain(int log_size, Domain* domain);
};

// =============================================================================
// Coset NTT Helper Kernels (Template - instantiated in ntt_kernels.cu)
// =============================================================================
// NOTE: Core NTT kernels (bit-reverse, butterfly, scale, fused) are defined
// directly in ntt_kernels.cu to ensure proper CUDA device code linking.
// Only these coset helpers need to be templates since coset_ntt_cuda uses them.

/**
 * @brief Multiply by coset generator powers: output[i] = input[i] * g^i
 * 
 * Used for coset FFT: evaluates polynomial at g*omega^i instead of omega^i
 */
template<typename F>
__global__ void coset_mul_kernel(
    F* output,
    const F* input,
    F coset_gen,      // The coset generator g
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    // Compute g^idx by repeated squaring
    F power = F::one();
    F base = coset_gen;
    int exp = idx;
    
    while (exp > 0) {
        if (exp & 1) {
            power = power * base;
        }
        base = base * base;
        exp >>= 1;
    }
    
    output[idx] = input[idx] * power;
}

/**
 * @brief Divide by coset generator powers: output[i] = input[i] * g^(-i)
 * 
 * Used after inverse coset FFT to convert back from coset domain
 */
template<typename F>
__global__ void coset_div_kernel(
    F* output,
    const F* input,
    F coset_gen_inv,  // The inverse of coset generator: g^(-1)
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    // Compute g^(-idx) = (g^(-1))^idx
    F power = F::one();
    F base = coset_gen_inv;
    int exp = idx;
    
    while (exp > 0) {
        if (exp & 1) {
            power = power * base;
        }
        base = base * base;
        exp >>= 1;
    }
    
    output[idx] = input[idx] * power;
}

/**
 * @brief Fast multiply by precomputed coset powers: output[i] = input[i] * powers[i]
 * 
 * Uses precomputed coset_powers array from Domain, avoiding per-element exponentiation.
 * This is ~10x faster than the on-the-fly computation version.
 */
template<typename F>
__global__ void coset_mul_precomputed_kernel(
    F* output,
    const F* input,
    const F* coset_powers,  // Precomputed g^i array
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    output[idx] = input[idx] * coset_powers[idx];
}

/**
 * @brief Fast divide by precomputed coset powers: output[i] = input[i] * powers_inv[i]
 * 
 * Uses precomputed coset_powers_inv array from Domain.
 */
template<typename F>
__global__ void coset_div_precomputed_kernel(
    F* output,
    const F* input,
    const F* coset_powers_inv,  // Precomputed g^(-i) array
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    output[idx] = input[idx] * coset_powers_inv[idx];
}

// =============================================================================
// NTT API Declarations (Implementations in ntt_kernels.cu)
// =============================================================================

/**
 * @brief NTT using Cooley-Tukey algorithm
 * 
 * This matches the Icicle ntt_cuda signature.
 * Implementation is in ntt_kernels.cu
 */
template<typename F>
eIcicleError ntt_cuda(
    const F* input,
    int size,
    NTTDir direction,
    const NTTConfig<F>& config,
    F* output
);

/**
 * @brief Coset NTT: evaluates at g*omega^i
 * 
 * Multiplies input by coset powers, then applies regular NTT
 */
template<typename F>
eIcicleError coset_ntt_cuda(
    const F* input,
    int size,
    NTTDir direction,
    const F& coset_gen,
    const NTTConfig<F>& config,
    F* output
);

/**
 * @brief Initialize NTT domain with root of unity
 */
template<typename F>
eIcicleError init_domain_cuda(
    const F& root_of_unity,
    const NTTInitDomainConfig& config
);

/**
 * @brief Release NTT domain resources
 */
template<typename F>
eIcicleError release_domain_cuda();

} // namespace ntt
