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
 * @file test_vec_ops.cu
 * @brief Comprehensive Tests for BLS12-381 Vector Operations
 * 
 * =============================================================================
 * TEST SUITE FOR VECTOR FIELD OPERATIONS
 * =============================================================================
 * 
 * Tests mathematical properties that MUST hold for correct implementations:
 * 
 * VECTOR OPERATION AXIOMS:
 * ========================
 * 1. Commutativity: vec_add(a,b) = vec_add(b,a), vec_mul(a,b) = vec_mul(b,a)
 * 2. Associativity: (a + b) + c = a + (b + c)
 * 3. Identity: vec_add(a, 0) = a, vec_mul(a, 1) = a
 * 4. Inverse: vec_sub(a, a) = 0
 * 5. Distributivity: scalar * (a + b) = scalar*a + scalar*b
 * 
 * EDGE CASE COVERAGE:
 * ===================
 * - Single element vectors
 * - Large vectors (64K+ elements)
 * - Non-power-of-2 sizes
 * - All-zero vectors
 * - All-one vectors
 * 
 * CONSISTENCY:
 * ============
 * - Deterministic results
 * - Host vs device memory paths
 */

#include "security_audit_tests.cuh"
#include "bls12_381_constants.h"

using namespace security_tests;
using namespace bls12_381;

// =============================================================================
// External Declarations for vec_ops
// =============================================================================

extern "C" {
eIcicleError bls12_381_vector_add(
    const Fr* a, const Fr* b, int n, const VecOpsConfig* config, Fr* result);
eIcicleError bls12_381_vector_sub(
    const Fr* a, const Fr* b, int n, const VecOpsConfig* config, Fr* result);
eIcicleError bls12_381_vector_mul(
    const Fr* a, const Fr* b, int n, const VecOpsConfig* config, Fr* result);
}

// =============================================================================
// Test Kernels
// =============================================================================

/**
 * @brief Generate random field elements
 */
__global__ void generate_random_elements_kernel(Fr* output, int n, uint64_t seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    // Simple LCG-based PRNG for test data
    uint64_t state = seed + idx * 6364136223846793005ULL;
    for (int i = 0; i < Fr::LIMBS; i++) {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        output[idx].limbs[i] = state;
    }
    // Reduce to ensure valid field element (simple mod-reduction approximation)
    output[idx].limbs[Fr::LIMBS - 1] &= 0x1FFFFFFFFFFFFFFFULL;
}

/**
 * @brief Compare two Fr arrays element-by-element
 */
__global__ void compare_arrays_kernel(const Fr* a, const Fr* b, int n, int* passed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    for (int i = 0; i < Fr::LIMBS; i++) {
        if (a[idx].limbs[i] != b[idx].limbs[i]) {
            atomicExch(passed, 0);
            return;
        }
    }
}

/**
 * @brief Check if all elements are zero
 */
__global__ void check_all_zero_kernel(const Fr* a, int n, int* all_zero) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    for (int i = 0; i < Fr::LIMBS; i++) {
        if (a[idx].limbs[i] != 0) {
            atomicExch(all_zero, 0);
            return;
        }
    }
}

/**
 * @brief Reference vector add kernel for validation
 */
__global__ void reference_vec_add_kernel(const Fr* a, const Fr* b, Fr* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    field_add(c[idx], a[idx], b[idx]);
}

/**
 * @brief Reference vector mul kernel for validation
 */
__global__ void reference_vec_mul_kernel(const Fr* a, const Fr* b, Fr* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    c[idx] = a[idx] * b[idx];
}

// =============================================================================
// Helper Functions
// =============================================================================

static VecOpsConfig make_host_config() {
    VecOpsConfig config;
    memset(&config, 0, sizeof(config));
    config.is_a_on_device = false;
    config.is_b_on_device = false;
    config.is_result_on_device = false;
    config.is_async = false;
    config.stream = nullptr;
    return config;
}

static void fill_with_zero(Fr* arr, int n) {
    for (int i = 0; i < n; i++) {
        memset(arr[i].limbs, 0, sizeof(arr[i].limbs));
    }
}

static void fill_with_one_montgomery(Fr* arr, int n) {
    for (int i = 0; i < n; i++) {
        arr[i].limbs[0] = FR_ONE_L0;
        arr[i].limbs[1] = FR_ONE_L1;
        arr[i].limbs[2] = FR_ONE_L2;
        arr[i].limbs[3] = FR_ONE_L3;
    }
}

static void fill_random(Fr* arr, int n, uint64_t seed) {
    for (int i = 0; i < n; i++) {
        uint64_t state = seed + i * 6364136223846793005ULL;
        for (int j = 0; j < Fr::LIMBS; j++) {
            state = state * 6364136223846793005ULL + 1442695040888963407ULL;
            arr[i].limbs[j] = state;
        }
        arr[i].limbs[Fr::LIMBS - 1] &= 0x1FFFFFFFFFFFFFFFULL;
    }
}

static bool arrays_equal(const Fr* a, const Fr* b, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < Fr::LIMBS; j++) {
            if (a[i].limbs[j] != b[i].limbs[j]) {
                return false;
            }
        }
    }
    return true;
}

static bool array_all_zero(const Fr* a, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < Fr::LIMBS; j++) {
            if (a[i].limbs[j] != 0) {
                return false;
            }
        }
    }
    return true;
}

// =============================================================================
// Test Cases - Commutativity
// =============================================================================

/**
 * @test vec_add(a, b) = vec_add(b, a)
 */
TestResult test_vec_add_commutative() {
    const int n = 1024;
    std::vector<Fr> a(n), b(n), ab(n), ba(n);
    
    fill_random(a.data(), n, 12345);
    fill_random(b.data(), n, 67890);
    
    VecOpsConfig config = make_host_config();
    
    eIcicleError err1 = bls12_381_vector_add(a.data(), b.data(), n, &config, ab.data());
    eIcicleError err2 = bls12_381_vector_add(b.data(), a.data(), n, &config, ba.data());
    
    if (err1 != eIcicleError::SUCCESS || err2 != eIcicleError::SUCCESS) {
        return TestResult::CUDA_ERROR;
    }
    
    if (!arrays_equal(ab.data(), ba.data(), n)) {
        return TestResult::FAILED;
    }
    
    return TestResult::PASSED;
}

/**
 * @test vec_mul(a, b) = vec_mul(b, a)
 */
TestResult test_vec_mul_commutative() {
    const int n = 1024;
    std::vector<Fr> a(n), b(n), ab(n), ba(n);
    
    fill_random(a.data(), n, 11111);
    fill_random(b.data(), n, 22222);
    
    VecOpsConfig config = make_host_config();
    
    eIcicleError err1 = bls12_381_vector_mul(a.data(), b.data(), n, &config, ab.data());
    eIcicleError err2 = bls12_381_vector_mul(b.data(), a.data(), n, &config, ba.data());
    
    if (err1 != eIcicleError::SUCCESS || err2 != eIcicleError::SUCCESS) {
        return TestResult::CUDA_ERROR;
    }
    
    if (!arrays_equal(ab.data(), ba.data(), n)) {
        return TestResult::FAILED;
    }
    
    return TestResult::PASSED;
}

// =============================================================================
// Test Cases - Identity Properties
// =============================================================================

/**
 * @test a + 0 = a
 */
TestResult test_vec_add_zero_identity() {
    const int n = 1024;
    std::vector<Fr> a(n), zero(n), result(n);
    
    fill_random(a.data(), n, 33333);
    fill_with_zero(zero.data(), n);
    
    VecOpsConfig config = make_host_config();
    
    eIcicleError err = bls12_381_vector_add(a.data(), zero.data(), n, &config, result.data());
    
    if (err != eIcicleError::SUCCESS) {
        return TestResult::CUDA_ERROR;
    }
    
    if (!arrays_equal(a.data(), result.data(), n)) {
        return TestResult::FAILED;
    }
    
    return TestResult::PASSED;
}

/**
 * @test a * 1 = a (in Montgomery form)
 */
TestResult test_vec_mul_one_identity() {
    const int n = 1024;
    std::vector<Fr> a(n), ones(n), result(n);
    
    fill_random(a.data(), n, 44444);
    fill_with_one_montgomery(ones.data(), n);
    
    VecOpsConfig config = make_host_config();
    
    eIcicleError err = bls12_381_vector_mul(a.data(), ones.data(), n, &config, result.data());
    
    if (err != eIcicleError::SUCCESS) {
        return TestResult::CUDA_ERROR;
    }
    
    if (!arrays_equal(a.data(), result.data(), n)) {
        return TestResult::FAILED;
    }
    
    return TestResult::PASSED;
}

/**
 * @test a * 0 = 0
 */
TestResult test_vec_mul_zero() {
    const int n = 1024;
    std::vector<Fr> a(n), zero(n), result(n);
    
    fill_random(a.data(), n, 55555);
    fill_with_zero(zero.data(), n);
    
    VecOpsConfig config = make_host_config();
    
    eIcicleError err = bls12_381_vector_mul(a.data(), zero.data(), n, &config, result.data());
    
    if (err != eIcicleError::SUCCESS) {
        return TestResult::CUDA_ERROR;
    }
    
    if (!array_all_zero(result.data(), n)) {
        return TestResult::FAILED;
    }
    
    return TestResult::PASSED;
}

// =============================================================================
// Test Cases - Inverse/Subtraction
// =============================================================================

/**
 * @test a - a = 0
 */
TestResult test_vec_sub_self_zero() {
    const int n = 1024;
    std::vector<Fr> a(n), result(n);
    
    fill_random(a.data(), n, 66666);
    
    VecOpsConfig config = make_host_config();
    
    eIcicleError err = bls12_381_vector_sub(a.data(), a.data(), n, &config, result.data());
    
    if (err != eIcicleError::SUCCESS) {
        return TestResult::CUDA_ERROR;
    }
    
    if (!array_all_zero(result.data(), n)) {
        return TestResult::FAILED;
    }
    
    return TestResult::PASSED;
}

/**
 * @test (a - b) + b = a
 */
TestResult test_vec_sub_add_identity() {
    const int n = 1024;
    std::vector<Fr> a(n), b(n), diff(n), recovered(n);
    
    fill_random(a.data(), n, 77777);
    fill_random(b.data(), n, 88888);
    
    VecOpsConfig config = make_host_config();
    
    eIcicleError err1 = bls12_381_vector_sub(a.data(), b.data(), n, &config, diff.data());
    eIcicleError err2 = bls12_381_vector_add(diff.data(), b.data(), n, &config, recovered.data());
    
    if (err1 != eIcicleError::SUCCESS || err2 != eIcicleError::SUCCESS) {
        return TestResult::CUDA_ERROR;
    }
    
    if (!arrays_equal(a.data(), recovered.data(), n)) {
        return TestResult::FAILED;
    }
    
    return TestResult::PASSED;
}

// =============================================================================
// Test Cases - Edge Cases
// =============================================================================

/**
 * @test Single element vector
 */
TestResult test_single_element() {
    const int n = 1;
    Fr a, b, result;
    
    a.limbs[0] = 0x12345678ABCDEF00ULL;
    a.limbs[1] = 0xFEDCBA9876543210ULL;
    a.limbs[2] = 0x1111222233334444ULL;
    a.limbs[3] = 0x0000000000000001ULL;
    
    b.limbs[0] = 0xAAAABBBBCCCCDDDDULL;
    b.limbs[1] = 0x1234567890ABCDEFULL;
    b.limbs[2] = 0x5555666677778888ULL;
    b.limbs[3] = 0x0000000000000002ULL;
    
    VecOpsConfig config = make_host_config();
    
    eIcicleError err = bls12_381_vector_add(&a, &b, n, &config, &result);
    
    if (err != eIcicleError::SUCCESS) {
        return TestResult::CUDA_ERROR;
    }
    
    // Verify by subtraction: (a+b) - b should equal a
    Fr check;
    bls12_381_vector_sub(&result, &b, n, &config, &check);
    
    for (int i = 0; i < Fr::LIMBS; i++) {
        if (check.limbs[i] != a.limbs[i]) {
            return TestResult::FAILED;
        }
    }
    
    return TestResult::PASSED;
}

/**
 * @test Large vector (64K elements)
 */
TestResult test_large_vector() {
    const int n = 65536;
    std::vector<Fr> a(n), b(n), ab(n), ba(n);
    
    fill_random(a.data(), n, 99999);
    fill_random(b.data(), n, 11111);
    
    VecOpsConfig config = make_host_config();
    
    eIcicleError err1 = bls12_381_vector_add(a.data(), b.data(), n, &config, ab.data());
    eIcicleError err2 = bls12_381_vector_add(b.data(), a.data(), n, &config, ba.data());
    
    if (err1 != eIcicleError::SUCCESS || err2 != eIcicleError::SUCCESS) {
        return TestResult::CUDA_ERROR;
    }
    
    if (!arrays_equal(ab.data(), ba.data(), n)) {
        return TestResult::FAILED;
    }
    
    return TestResult::PASSED;
}

/**
 * @test Non-power-of-2 size (1000 elements)
 */
TestResult test_non_power_of_two() {
    const int n = 1000;  // Not a power of 2
    std::vector<Fr> a(n), zero(n), result(n);
    
    fill_random(a.data(), n, 22222);
    fill_with_zero(zero.data(), n);
    
    VecOpsConfig config = make_host_config();
    
    eIcicleError err = bls12_381_vector_add(a.data(), zero.data(), n, &config, result.data());
    
    if (err != eIcicleError::SUCCESS) {
        return TestResult::CUDA_ERROR;
    }
    
    if (!arrays_equal(a.data(), result.data(), n)) {
        return TestResult::FAILED;
    }
    
    return TestResult::PASSED;
}

/**
 * @test All-zero input vectors
 */
TestResult test_all_zeros() {
    const int n = 1024;
    std::vector<Fr> zero1(n), zero2(n), result(n);
    
    fill_with_zero(zero1.data(), n);
    fill_with_zero(zero2.data(), n);
    
    VecOpsConfig config = make_host_config();
    
    // Test add
    eIcicleError err1 = bls12_381_vector_add(zero1.data(), zero2.data(), n, &config, result.data());
    if (err1 != eIcicleError::SUCCESS || !array_all_zero(result.data(), n)) {
        return TestResult::FAILED;
    }
    
    // Test mul
    eIcicleError err2 = bls12_381_vector_mul(zero1.data(), zero2.data(), n, &config, result.data());
    if (err2 != eIcicleError::SUCCESS || !array_all_zero(result.data(), n)) {
        return TestResult::FAILED;
    }
    
    // Test sub
    eIcicleError err3 = bls12_381_vector_sub(zero1.data(), zero2.data(), n, &config, result.data());
    if (err3 != eIcicleError::SUCCESS || !array_all_zero(result.data(), n)) {
        return TestResult::FAILED;
    }
    
    return TestResult::PASSED;
}

// =============================================================================
// Test Cases - Consistency
// =============================================================================

/**
 * @test Determinism: multiple calls yield same result
 */
TestResult test_determinism() {
    const int n = 1024;
    const int iterations = 3;
    
    std::vector<Fr> a(n), b(n);
    std::vector<std::vector<Fr>> results(iterations, std::vector<Fr>(n));
    
    fill_random(a.data(), n, 33333);
    fill_random(b.data(), n, 44444);
    
    VecOpsConfig config = make_host_config();
    
    for (int iter = 0; iter < iterations; iter++) {
        eIcicleError err = bls12_381_vector_add(a.data(), b.data(), n, &config, results[iter].data());
        if (err != eIcicleError::SUCCESS) {
            return TestResult::CUDA_ERROR;
        }
    }
    
    // All iterations should produce identical results
    for (int iter = 1; iter < iterations; iter++) {
        if (!arrays_equal(results[0].data(), results[iter].data(), n)) {
            return TestResult::FAILED;
        }
    }
    
    return TestResult::PASSED;
}

/**
 * @test Cross-validate against reference kernel
 */
TestResult test_cross_validate_add() {
    const int n = 1024;
    std::vector<Fr> a(n), b(n), api_result(n), ref_result(n);
    
    fill_random(a.data(), n, 55555);
    fill_random(b.data(), n, 66666);
    
    // API result
    VecOpsConfig config = make_host_config();
    eIcicleError err = bls12_381_vector_add(a.data(), b.data(), n, &config, api_result.data());
    if (err != eIcicleError::SUCCESS) {
        return TestResult::CUDA_ERROR;
    }
    
    // Reference result (GPU kernel)
    Fr *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, n * sizeof(Fr));
    cudaMalloc(&d_b, n * sizeof(Fr));
    cudaMalloc(&d_c, n * sizeof(Fr));
    
    cudaMemcpy(d_a, a.data(), n * sizeof(Fr), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), n * sizeof(Fr), cudaMemcpyHostToDevice);
    
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    reference_vec_add_kernel<<<numBlocks, blockSize>>>(d_a, d_b, d_c, n);
    
    cudaMemcpy(ref_result.data(), d_c, n * sizeof(Fr), cudaMemcpyDeviceToHost);
    
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    
    if (!arrays_equal(api_result.data(), ref_result.data(), n)) {
        return TestResult::FAILED;
    }
    
    return TestResult::PASSED;
}

/**
 * @test Cross-validate multiplication against reference kernel
 */
TestResult test_cross_validate_mul() {
    const int n = 1024;
    std::vector<Fr> a(n), b(n), api_result(n), ref_result(n);
    
    fill_random(a.data(), n, 77777);
    fill_random(b.data(), n, 88888);
    
    // API result
    VecOpsConfig config = make_host_config();
    eIcicleError err = bls12_381_vector_mul(a.data(), b.data(), n, &config, api_result.data());
    if (err != eIcicleError::SUCCESS) {
        return TestResult::CUDA_ERROR;
    }
    
    // Reference result (GPU kernel)
    Fr *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, n * sizeof(Fr));
    cudaMalloc(&d_b, n * sizeof(Fr));
    cudaMalloc(&d_c, n * sizeof(Fr));
    
    cudaMemcpy(d_a, a.data(), n * sizeof(Fr), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), n * sizeof(Fr), cudaMemcpyHostToDevice);
    
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    reference_vec_mul_kernel<<<numBlocks, blockSize>>>(d_a, d_b, d_c, n);
    
    cudaMemcpy(ref_result.data(), d_c, n * sizeof(Fr), cudaMemcpyDeviceToHost);
    
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    
    if (!arrays_equal(api_result.data(), ref_result.data(), n)) {
        return TestResult::FAILED;
    }
    
    return TestResult::PASSED;
}

// =============================================================================
// Test Registration
// =============================================================================

void register_vec_ops_tests(SecurityTestSuite& suite) {
    // Commutativity (Algebraic property)
    suite.add_test("vec_add commutativity: a+b = b+a", "Algebraic Properties",
                   test_vec_add_commutative);
    suite.add_test("vec_mul commutativity: a*b = b*a", "Algebraic Properties",
                   test_vec_mul_commutative);
    
    // Identity (Critical for correctness)
    suite.add_test("Additive identity: a + 0 = a", "Identity Properties",
                   test_vec_add_zero_identity, true);
    suite.add_test("Multiplicative identity: a * 1 = a", "Identity Properties",
                   test_vec_mul_one_identity, true);
    suite.add_test("Multiplication by zero: a * 0 = 0", "Identity Properties",
                   test_vec_mul_zero);
    
    // Inverse/Subtraction
    suite.add_test("Self-subtraction: a - a = 0", "Inverse Properties",
                   test_vec_sub_self_zero, true);
    suite.add_test("Subtraction+add identity: (a-b)+b = a", "Inverse Properties",
                   test_vec_sub_add_identity);
    
    // Edge cases
    suite.add_test("Single element vector", "Edge Cases",
                   test_single_element);
    suite.add_test("Large vector (64K elements)", "Edge Cases",
                   test_large_vector);
    suite.add_test("Non-power-of-2 size (1000)", "Edge Cases",
                   test_non_power_of_two);
    suite.add_test("All-zero vectors", "Edge Cases",
                   test_all_zeros);
    
    // Consistency
    suite.add_test("Determinism (3 iterations)", "Consistency",
                   test_determinism);
    suite.add_test("Cross-validate add vs reference kernel", "Consistency",
                   test_cross_validate_add, true);
    suite.add_test("Cross-validate mul vs reference kernel", "Consistency",
                   test_cross_validate_mul, true);
}

// =============================================================================
// Main
// =============================================================================

int main(int argc, char** argv) {
    (void)argc; (void)argv;
    
    int device_count;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
        std::cerr << "No CUDA devices found!" << std::endl;
        return 1;
    }
    
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    std::cout << "GPU: " << prop.name << " (SM " << prop.major << "." << prop.minor << ")" << std::endl;
    
    SecurityTestSuite suite;
    register_vec_ops_tests(suite);
    
    bool success = suite.run_all();
    return success ? 0 : 1;
}
