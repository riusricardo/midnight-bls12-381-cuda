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
 * @file icicle_field_api.cu
 * @brief ICICLE field backend registration for NTT and vector operations
 * 
 * This file registers our CUDA implementations with ICICLE's dispatcher system.
 * The pattern follows ICICLE's backend registration API:
 * 
 * 1. Registration functions and types are declared in icicle_backend_api.cuh
 * 2. Create wrapper implementations matching ICICLE's signatures
 * 3. Use REGISTER_*_BACKEND macros for static initialization
 * 
 * The actual kernel implementations are in:
 * - ntt_kernels.cu: NTT operations (ntt_cuda, init_domain_cuda, etc.)
 * - vec_ops.cu: Vector operations (vec_add, vec_sub, vec_mul, etc.)
 * 
 * Build: Compiled together with ntt_kernels.cu, vec_ops.cu
 */

#include "field.cuh"
#include "ntt.cuh"
#include "icicle_backend_api.cuh"

using namespace bls12_381;

// Bring in commonly used icicle types explicitly to avoid ambiguity
using icicle::eIcicleError;
using icicle::Device;
using icicle::NTTDir;
using icicle::NTTInitDomainConfig;
using icicle::VecOpsConfig;

// =============================================================================
// Forward declarations to implementation functions in other .cu files
// =============================================================================

// From ntt_kernels.cu
namespace ntt {
    template<typename Fr>
    eIcicleError ntt_cuda(
        const Fr* input,
        int size,
        NTTDir dir,
        const icicle::NTTConfig<Fr>& config,
        Fr* output);
    
    template<typename Fr>
    eIcicleError init_domain_cuda(
        const Fr& root_of_unity,
        const NTTInitDomainConfig& config);
    
    template<typename Fr>
    eIcicleError release_domain_cuda();
}

// =============================================================================
// ICICLE-Compatible Wrapper Implementations
// =============================================================================
// These wrappers add the Device& parameter that ICICLE's dispatcher expects

// Forward declarations for kernel functions from vec_ops.cu
// Must be outside anonymous namespace to match actual definitions
namespace vec_ops {
    __global__ void vec_add_kernel(Fr* out, const Fr* a, const Fr* b, int size);
    __global__ void vec_sub_kernel(Fr* out, const Fr* a, const Fr* b, int size);
    __global__ void vec_mul_kernel(Fr* out, const Fr* a, const Fr* b, int size);
    __global__ void scalar_vec_mul_kernel(Fr* out, const Fr scalar, const Fr* vec, int size);
    __global__ void scalar_vec_add_kernel(Fr* out, const Fr scalar, const Fr* vec, int size);
}

namespace {

// -----------------------------------------------------------------------------
// NTT Wrappers
// -----------------------------------------------------------------------------

eIcicleError ntt_cuda_impl(
    const Device& device,
    const Fr* input,
    int size,
    NTTDir dir,
    const icicle::NTTConfig<Fr>& config,
    Fr* output)
{
    (void)device;  // CUDA device already set, device param unused
    return ntt::ntt_cuda<Fr>(input, size, dir, config, output);
}

eIcicleError ntt_init_domain_cuda_impl(
    const Device& device,
    const Fr& primitive_root,
    const NTTInitDomainConfig& config)
{
    (void)device;
    return ntt::init_domain_cuda<Fr>(primitive_root, config);
}

eIcicleError ntt_release_domain_cuda_impl(
    const Device& device,
    const Fr& phantom)
{
    (void)device;
    (void)phantom;
    return ntt::release_domain_cuda<Fr>();
}

// -----------------------------------------------------------------------------
// Vector Operations Wrappers
// -----------------------------------------------------------------------------

// Helper function to handle device memory allocation based on config
template<typename KernelFunc>
eIcicleError run_vec_op(
    const Device& device,
    const Fr* a,
    const Fr* b,
    uint64_t size,
    const VecOpsConfig& config,
    Fr* output,
    KernelFunc kernel)
{
    (void)device;  // CUDA device already set
    cudaStream_t stream = static_cast<cudaStream_t>(config.stream);
    
    const Fr* d_a = a;
    const Fr* d_b = b;
    Fr* d_output = output;
    
    bool alloc_a = !config.is_a_on_device;
    bool alloc_b = b != nullptr && !config.is_b_on_device;
    bool alloc_out = !config.is_result_on_device;
    
    // Allocate device memory if input is on host
    if (alloc_a) {
        Fr* temp;
        cudaMalloc(&temp, size * sizeof(Fr));
        cudaMemcpyAsync(temp, a, size * sizeof(Fr), cudaMemcpyHostToDevice, stream);
        d_a = temp;
    }
    if (alloc_b) {
        Fr* temp;
        cudaMalloc(&temp, size * sizeof(Fr));
        cudaMemcpyAsync(temp, b, size * sizeof(Fr), cudaMemcpyHostToDevice, stream);
        d_b = temp;
    }
    if (alloc_out) {
        cudaMalloc(&d_output, size * sizeof(Fr));
    }
    
    // Launch kernel
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;
    kernel<<<blocks, threads, 0, stream>>>(d_output, d_a, d_b, (int)size);
    
    // Copy output back if needed
    if (alloc_out) {
        cudaMemcpyAsync(output, d_output, size * sizeof(Fr), cudaMemcpyDeviceToHost, stream);
    }
    
    // Sync if not async
    if (!config.is_async) {
        cudaStreamSynchronize(stream);
    }
    
    // Free temporary allocations
    if (alloc_out) cudaFree(d_output);
    if (alloc_a) cudaFree(const_cast<Fr*>(d_a));
    if (alloc_b) cudaFree(const_cast<Fr*>(d_b));
    
    return cudaGetLastError() == cudaSuccess ? 
           eIcicleError::SUCCESS : eIcicleError::UNKNOWN_ERROR;
}

eIcicleError vector_add_cuda_impl(
    const Device& device,
    const Fr* a,
    const Fr* b,
    uint64_t size,
    const VecOpsConfig& config,
    Fr* output)
{
    return run_vec_op(device, a, b, size, config, output, vec_ops::vec_add_kernel);
}

eIcicleError vector_sub_cuda_impl(
    const Device& device,
    const Fr* a,
    const Fr* b,
    uint64_t size,
    const VecOpsConfig& config,
    Fr* output)
{
    return run_vec_op(device, a, b, size, config, output, vec_ops::vec_sub_kernel);
}

eIcicleError vector_mul_cuda_impl(
    const Device& device,
    const Fr* a,
    const Fr* b,
    uint64_t size,
    const VecOpsConfig& config,
    Fr* output)
{
    return run_vec_op(device, a, b, size, config, output, vec_ops::vec_mul_kernel);
}

eIcicleError scalar_mul_vec_cuda_impl(
    const Device& device,
    const Fr* scalar,
    const Fr* vec,
    uint64_t size,
    const VecOpsConfig& config,
    Fr* output)
{
    (void)device;  // CUDA device already set
    cudaStream_t stream = static_cast<cudaStream_t>(config.stream);
    
    const Fr* d_vec = vec;
    Fr* d_output = output;
    Fr h_scalar;
    
    bool alloc_vec = !config.is_b_on_device;  // scalar uses a, vec uses b
    bool alloc_out = !config.is_result_on_device;
    
    // Get scalar value (scalar is single element, usually on host)
    if (config.is_a_on_device) {
        cudaMemcpy(&h_scalar, scalar, sizeof(Fr), cudaMemcpyDeviceToHost);
    } else {
        h_scalar = *scalar;
    }
    
    if (alloc_vec) {
        Fr* temp;
        cudaMalloc(&temp, size * sizeof(Fr));
        cudaMemcpyAsync(temp, vec, size * sizeof(Fr), cudaMemcpyHostToDevice, stream);
        d_vec = temp;
    }
    if (alloc_out) {
        cudaMalloc(&d_output, size * sizeof(Fr));
    }
    
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;
    vec_ops::scalar_vec_mul_kernel<<<blocks, threads, 0, stream>>>(
        d_output, h_scalar, d_vec, (int)size);
    
    if (alloc_out) {
        cudaMemcpyAsync(output, d_output, size * sizeof(Fr), cudaMemcpyDeviceToHost, stream);
    }
    
    if (!config.is_async) {
        cudaStreamSynchronize(stream);
    }
    
    if (alloc_out) cudaFree(d_output);
    if (alloc_vec) cudaFree(const_cast<Fr*>(d_vec));
    
    return cudaGetLastError() == cudaSuccess ? 
           eIcicleError::SUCCESS : eIcicleError::UNKNOWN_ERROR;
}

eIcicleError scalar_add_vec_cuda_impl(
    const Device& device,
    const Fr* scalar,
    const Fr* vec,
    uint64_t size,
    const VecOpsConfig& config,
    Fr* output)
{
    (void)device;  // CUDA device already set
    cudaStream_t stream = static_cast<cudaStream_t>(config.stream);
    
    const Fr* d_vec = vec;
    Fr* d_output = output;
    Fr h_scalar;
    
    bool alloc_vec = !config.is_b_on_device;
    bool alloc_out = !config.is_result_on_device;
    
    if (config.is_a_on_device) {
        cudaMemcpy(&h_scalar, scalar, sizeof(Fr), cudaMemcpyDeviceToHost);
    } else {
        h_scalar = *scalar;
    }
    
    if (alloc_vec) {
        Fr* temp;
        cudaMalloc(&temp, size * sizeof(Fr));
        cudaMemcpyAsync(temp, vec, size * sizeof(Fr), cudaMemcpyHostToDevice, stream);
        d_vec = temp;
    }
    if (alloc_out) {
        cudaMalloc(&d_output, size * sizeof(Fr));
    }
    
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;
    vec_ops::scalar_vec_add_kernel<<<blocks, threads, 0, stream>>>(
        d_output, h_scalar, d_vec, (int)size);
    
    if (alloc_out) {
        cudaMemcpyAsync(output, d_output, size * sizeof(Fr), cudaMemcpyDeviceToHost, stream);
    }
    
    if (!config.is_async) {
        cudaStreamSynchronize(stream);
    }
    
    if (alloc_out) cudaFree(d_output);
    if (alloc_vec) cudaFree(const_cast<Fr*>(d_vec));
    
    return cudaGetLastError() == cudaSuccess ? 
           eIcicleError::SUCCESS : eIcicleError::UNKNOWN_ERROR;
}

} // anonymous namespace

// =============================================================================
// Backend Registration (Static Initialization)
// =============================================================================
// These macros create static variables whose initializers call the registration
// functions when the library is loaded

REGISTER_NTT_BACKEND("CUDA", ntt_cuda_impl);
REGISTER_NTT_INIT_DOMAIN_BACKEND("CUDA", ntt_init_domain_cuda_impl);
REGISTER_NTT_RELEASE_DOMAIN_BACKEND("CUDA", ntt_release_domain_cuda_impl);

REGISTER_VECTOR_ADD_BACKEND("CUDA", vector_add_cuda_impl);
REGISTER_VECTOR_SUB_BACKEND("CUDA", vector_sub_cuda_impl);
REGISTER_VECTOR_MUL_BACKEND("CUDA", vector_mul_cuda_impl);
REGISTER_SCALAR_MUL_VEC_BACKEND("CUDA", scalar_mul_vec_cuda_impl);
REGISTER_SCALAR_ADD_VEC_BACKEND("CUDA", scalar_add_vec_cuda_impl);

// =============================================================================
// C API Exports (for standalone testing and dlsym discovery)
// =============================================================================

extern "C" {

// These symbols are for backward compatibility / standalone testing
// The primary integration path is through ICICLE's dispatcher above

eIcicleError bls12_381_field_ntt_cuda(
    const Fr* input,
    int size,
    NTTDir dir,
    const icicle::NTTConfig<Fr>* config,
    Fr* output)
{
    return ntt::ntt_cuda<Fr>(input, size, dir, *config, output);
}

eIcicleError bls12_381_field_ntt_init_domain_cuda(
    const Fr* root_of_unity,
    const NTTInitDomainConfig* config)
{
    return ntt::init_domain_cuda<Fr>(*root_of_unity, *config);
}

eIcicleError bls12_381_field_ntt_release_domain_cuda()
{
    return ntt::release_domain_cuda<Fr>();
}

} // extern "C"
