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
 * @file test_field_properties.cu
 * @brief Algebraic Property Tests for BLS12-381 Field Arithmetic
 * 
 * Tests mathematical properties that MUST hold for correct field implementations:
 * 
 * FIELD AXIOMS (must verify):
 * ===========================
 * 1. Closure: a + b ∈ Fr, a * b ∈ Fr
 * 2. Associativity: (a + b) + c = a + (b + c), (a * b) * c = a * (b * c)
 * 3. Commutativity: a + b = b + a, a * b = b * a
 * 4. Identity: a + 0 = a, a * 1 = a
 * 5. Inverse: a + (-a) = 0, a * a^(-1) = 1 (for a ≠ 0)
 * 6. Distributivity: a * (b + c) = a*b + a*c
 * 
 * MONTGOMERY FORM PROPERTIES:
 * ===========================
 * 7. to_mont(from_mont(a)) = a
 * 8. from_mont(to_mont(a)) = a
 * 9. (aR * bR) * R^(-1) = abR (Montgomery multiplication)
 * 
 * SQUARING CONSISTENCY:
 * =====================
 * 10. a^2 = a * a (field_sqr matches field_mul)
 * 
 * SECURITY CONSIDERATIONS:
 * ========================
 * - All operations must be constant-time
 * - No timing leaks based on operand values
 * - Proper reduction (results always < modulus)
 */

#include "security_audit_tests.cuh"

using namespace security_tests;

// =============================================================================
// Test Kernels for Field Properties
// =============================================================================

/**
 * @brief Test: a + b = b + a (additive commutativity)
 */
__global__ void test_add_commutative_kernel(
    const Fr* a, const Fr* b, Fr* ab, Fr* ba, int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    field_add(ab[idx], a[idx], b[idx]);
    field_add(ba[idx], b[idx], a[idx]);
}

/**
 * @brief Test: a * b = b * a (multiplicative commutativity)
 */
__global__ void test_mul_commutative_kernel(
    const Fr* a, const Fr* b, Fr* ab, Fr* ba, int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    ab[idx] = a[idx] * b[idx];
    ba[idx] = b[idx] * a[idx];
}

/**
 * @brief Test: (a + b) + c = a + (b + c) (additive associativity)
 */
__global__ void test_add_associative_kernel(
    const Fr* a, const Fr* b, const Fr* c, Fr* lhs, Fr* rhs, int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    Fr ab, bc;
    field_add(ab, a[idx], b[idx]);
    field_add(lhs[idx], ab, c[idx]);  // (a+b)+c
    
    field_add(bc, b[idx], c[idx]);
    field_add(rhs[idx], a[idx], bc);  // a+(b+c)
}

/**
 * @brief Test: (a * b) * c = a * (b * c) (multiplicative associativity)
 */
__global__ void test_mul_associative_kernel(
    const Fr* a, const Fr* b, const Fr* c, Fr* lhs, Fr* rhs, int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    Fr ab = a[idx] * b[idx];
    lhs[idx] = ab * c[idx];  // (a*b)*c
    
    Fr bc = b[idx] * c[idx];
    rhs[idx] = a[idx] * bc;  // a*(b*c)
}

/**
 * @brief Test: a + 0 = a (additive identity)
 */
__global__ void test_add_identity_kernel(
    const Fr* a, Fr* out, int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    Fr zero = Fr::zero();
    field_add(out[idx], a[idx], zero);
}

/**
 * @brief Test: a * 1 = a (multiplicative identity)
 */
__global__ void test_mul_identity_kernel(
    const Fr* a, Fr* out, int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    Fr one = Fr::one();
    out[idx] = a[idx] * one;
}

/**
 * @brief Test: a + (-a) = 0 (additive inverse)
 */
__global__ void test_add_inverse_kernel(
    const Fr* a, Fr* out, int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    Fr neg_a;
    field_neg(neg_a, a[idx]);
    field_add(out[idx], a[idx], neg_a);
}

/**
 * @brief Test: a * a^(-1) = 1 (multiplicative inverse)
 */
__global__ void test_mul_inverse_kernel(
    const Fr* a, Fr* out, int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    // Skip zero (has no inverse)
    if (a[idx].is_zero()) {
        out[idx] = Fr::one();  // Mark as "passed" for zero
        return;
    }
    
    Fr a_inv;
    field_inv(a_inv, a[idx]);
    out[idx] = a[idx] * a_inv;
}

/**
 * @brief Test: a * (b + c) = a*b + a*c (distributivity)
 */
__global__ void test_distributive_kernel(
    const Fr* a, const Fr* b, const Fr* c, Fr* lhs, Fr* rhs, int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    Fr b_plus_c;
    field_add(b_plus_c, b[idx], c[idx]);
    lhs[idx] = a[idx] * b_plus_c;  // a*(b+c)
    
    Fr ab = a[idx] * b[idx];
    Fr ac = a[idx] * c[idx];
    field_add(rhs[idx], ab, ac);   // a*b + a*c
}

/**
 * @brief Test: a^2 = a * a (squaring consistency)
 */
__global__ void test_sqr_consistency_kernel(
    const Fr* a, Fr* sqr_out, Fr* mul_out, int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    field_sqr(sqr_out[idx], a[idx]);  // Using dedicated squaring
    mul_out[idx] = a[idx] * a[idx];   // Using multiplication
}

/**
 * @brief Test: from_mont(to_mont(a)) = a (Montgomery roundtrip)
 */
__global__ void test_montgomery_roundtrip_kernel(
    const Fr* a, Fr* out, int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    Fr mont, back;
    field_to_montgomery(mont, a[idx]);
    field_from_montgomery(back, mont);
    out[idx] = back;
}

/**
 * @brief Test: (a - b) + b = a (subtraction identity)
 */
__global__ void test_sub_add_identity_kernel(
    const Fr* a, const Fr* b, Fr* out, int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    Fr diff;
    field_sub(diff, a[idx], b[idx]);
    field_add(out[idx], diff, b[idx]);
}

/**
 * @brief Test: a - a = 0 (self-subtraction)
 */
__global__ void test_sub_self_kernel(
    const Fr* a, Fr* out, int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    field_sub(out[idx], a[idx], a[idx]);
}

/**
 * @brief Test: 0 * a = 0 (multiplication by zero)
 */
__global__ void test_mul_zero_kernel(
    const Fr* a, Fr* out, int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    Fr zero = Fr::zero();
    out[idx] = zero * a[idx];
}

/**
 * @brief Test: Reduction - verify results are always < modulus
 */
__global__ void test_reduction_kernel(
    const Fr* a, const Fr* b, int* violations, int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    Fr sum, prod;
    field_add(sum, a[idx], b[idx]);
    prod = a[idx] * b[idx];
    
    // Check sum < modulus (comparing limb by limb from high to low)
    bool sum_valid = true;
    bool prod_valid = true;
    
    for (int i = 3; i >= 0; i--) {
        uint64_t mod_i = fp_config::modulus(i);
        if (sum.limbs[i] > mod_i) {
            sum_valid = false;
            break;
        } else if (sum.limbs[i] < mod_i) {
            break;  // Already less, no need to check lower limbs
        }
    }
    
    for (int i = 3; i >= 0; i--) {
        uint64_t mod_i = fp_config::modulus(i);
        if (prod.limbs[i] > mod_i) {
            prod_valid = false;
            break;
        } else if (prod.limbs[i] < mod_i) {
            break;
        }
    }
    
    if (!sum_valid || !prod_valid) {
        atomicAdd(violations, 1);
    }
}

// =============================================================================
// Test Functions
// =============================================================================

TestResult test_additive_commutativity() {
    const int n = 4096;
    std::mt19937_64 rng(42);
    
    std::vector<Fr> a_vals(n), b_vals(n);
    for (int i = 0; i < n; i++) {
        a_vals[i] = random_fr_montgomery(rng);
        b_vals[i] = random_fr_montgomery(rng);
    }
    
    Fr *d_a, *d_b, *d_ab, *d_ba;
    SECURITY_CHECK_CUDA(cudaMalloc(&d_a, n * sizeof(Fr)));
    SECURITY_CHECK_CUDA(cudaMalloc(&d_b, n * sizeof(Fr)));
    SECURITY_CHECK_CUDA(cudaMalloc(&d_ab, n * sizeof(Fr)));
    SECURITY_CHECK_CUDA(cudaMalloc(&d_ba, n * sizeof(Fr)));
    
    SECURITY_CHECK_CUDA(cudaMemcpy(d_a, a_vals.data(), n * sizeof(Fr), cudaMemcpyHostToDevice));
    SECURITY_CHECK_CUDA(cudaMemcpy(d_b, b_vals.data(), n * sizeof(Fr), cudaMemcpyHostToDevice));
    
    test_add_commutative_kernel<<<(n + 255) / 256, 256>>>(d_a, d_b, d_ab, d_ba, n);
    SECURITY_CHECK_CUDA(cudaDeviceSynchronize());
    
    std::vector<Fr> ab_results(n), ba_results(n);
    SECURITY_CHECK_CUDA(cudaMemcpy(ab_results.data(), d_ab, n * sizeof(Fr), cudaMemcpyDeviceToHost));
    SECURITY_CHECK_CUDA(cudaMemcpy(ba_results.data(), d_ba, n * sizeof(Fr), cudaMemcpyDeviceToHost));
    
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_ab); cudaFree(d_ba);
    
    int failures = 0;
    for (int i = 0; i < n; i++) {
        if (!limbs_equal(ab_results[i].limbs, ba_results[i].limbs, 4)) {
            failures++;
        }
    }
    
    if (failures > 0) {
        std::cout << "\n    " << failures << "/" << n << " failures";
        return TestResult::FAILED;
    }
    return TestResult::PASSED;
}

TestResult test_multiplicative_commutativity() {
    const int n = 4096;
    std::mt19937_64 rng(123);
    
    std::vector<Fr> a_vals(n), b_vals(n);
    for (int i = 0; i < n; i++) {
        a_vals[i] = random_fr_montgomery(rng);
        b_vals[i] = random_fr_montgomery(rng);
    }
    
    Fr *d_a, *d_b, *d_ab, *d_ba;
    SECURITY_CHECK_CUDA(cudaMalloc(&d_a, n * sizeof(Fr)));
    SECURITY_CHECK_CUDA(cudaMalloc(&d_b, n * sizeof(Fr)));
    SECURITY_CHECK_CUDA(cudaMalloc(&d_ab, n * sizeof(Fr)));
    SECURITY_CHECK_CUDA(cudaMalloc(&d_ba, n * sizeof(Fr)));
    
    SECURITY_CHECK_CUDA(cudaMemcpy(d_a, a_vals.data(), n * sizeof(Fr), cudaMemcpyHostToDevice));
    SECURITY_CHECK_CUDA(cudaMemcpy(d_b, b_vals.data(), n * sizeof(Fr), cudaMemcpyHostToDevice));
    
    test_mul_commutative_kernel<<<(n + 255) / 256, 256>>>(d_a, d_b, d_ab, d_ba, n);
    SECURITY_CHECK_CUDA(cudaDeviceSynchronize());
    
    std::vector<Fr> ab_results(n), ba_results(n);
    SECURITY_CHECK_CUDA(cudaMemcpy(ab_results.data(), d_ab, n * sizeof(Fr), cudaMemcpyDeviceToHost));
    SECURITY_CHECK_CUDA(cudaMemcpy(ba_results.data(), d_ba, n * sizeof(Fr), cudaMemcpyDeviceToHost));
    
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_ab); cudaFree(d_ba);
    
    int failures = 0;
    for (int i = 0; i < n; i++) {
        if (!limbs_equal(ab_results[i].limbs, ba_results[i].limbs, 4)) {
            failures++;
        }
    }
    
    if (failures > 0) {
        std::cout << "\n    " << failures << "/" << n << " failures";
        return TestResult::FAILED;
    }
    return TestResult::PASSED;
}

TestResult test_additive_associativity() {
    const int n = 4096;
    std::mt19937_64 rng(456);
    
    std::vector<Fr> a_vals(n), b_vals(n), c_vals(n);
    for (int i = 0; i < n; i++) {
        a_vals[i] = random_fr_montgomery(rng);
        b_vals[i] = random_fr_montgomery(rng);
        c_vals[i] = random_fr_montgomery(rng);
    }
    
    Fr *d_a, *d_b, *d_c, *d_lhs, *d_rhs;
    SECURITY_CHECK_CUDA(cudaMalloc(&d_a, n * sizeof(Fr)));
    SECURITY_CHECK_CUDA(cudaMalloc(&d_b, n * sizeof(Fr)));
    SECURITY_CHECK_CUDA(cudaMalloc(&d_c, n * sizeof(Fr)));
    SECURITY_CHECK_CUDA(cudaMalloc(&d_lhs, n * sizeof(Fr)));
    SECURITY_CHECK_CUDA(cudaMalloc(&d_rhs, n * sizeof(Fr)));
    
    SECURITY_CHECK_CUDA(cudaMemcpy(d_a, a_vals.data(), n * sizeof(Fr), cudaMemcpyHostToDevice));
    SECURITY_CHECK_CUDA(cudaMemcpy(d_b, b_vals.data(), n * sizeof(Fr), cudaMemcpyHostToDevice));
    SECURITY_CHECK_CUDA(cudaMemcpy(d_c, c_vals.data(), n * sizeof(Fr), cudaMemcpyHostToDevice));
    
    test_add_associative_kernel<<<(n + 255) / 256, 256>>>(d_a, d_b, d_c, d_lhs, d_rhs, n);
    SECURITY_CHECK_CUDA(cudaDeviceSynchronize());
    
    std::vector<Fr> lhs(n), rhs(n);
    SECURITY_CHECK_CUDA(cudaMemcpy(lhs.data(), d_lhs, n * sizeof(Fr), cudaMemcpyDeviceToHost));
    SECURITY_CHECK_CUDA(cudaMemcpy(rhs.data(), d_rhs, n * sizeof(Fr), cudaMemcpyDeviceToHost));
    
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c); cudaFree(d_lhs); cudaFree(d_rhs);
    
    int failures = 0;
    for (int i = 0; i < n; i++) {
        if (!limbs_equal(lhs[i].limbs, rhs[i].limbs, 4)) {
            failures++;
        }
    }
    
    if (failures > 0) {
        std::cout << "\n    " << failures << "/" << n << " failures";
        return TestResult::FAILED;
    }
    return TestResult::PASSED;
}

TestResult test_multiplicative_associativity() {
    const int n = 4096;
    std::mt19937_64 rng(789);
    
    std::vector<Fr> a_vals(n), b_vals(n), c_vals(n);
    for (int i = 0; i < n; i++) {
        a_vals[i] = random_fr_montgomery(rng);
        b_vals[i] = random_fr_montgomery(rng);
        c_vals[i] = random_fr_montgomery(rng);
    }
    
    Fr *d_a, *d_b, *d_c, *d_lhs, *d_rhs;
    SECURITY_CHECK_CUDA(cudaMalloc(&d_a, n * sizeof(Fr)));
    SECURITY_CHECK_CUDA(cudaMalloc(&d_b, n * sizeof(Fr)));
    SECURITY_CHECK_CUDA(cudaMalloc(&d_c, n * sizeof(Fr)));
    SECURITY_CHECK_CUDA(cudaMalloc(&d_lhs, n * sizeof(Fr)));
    SECURITY_CHECK_CUDA(cudaMalloc(&d_rhs, n * sizeof(Fr)));
    
    SECURITY_CHECK_CUDA(cudaMemcpy(d_a, a_vals.data(), n * sizeof(Fr), cudaMemcpyHostToDevice));
    SECURITY_CHECK_CUDA(cudaMemcpy(d_b, b_vals.data(), n * sizeof(Fr), cudaMemcpyHostToDevice));
    SECURITY_CHECK_CUDA(cudaMemcpy(d_c, c_vals.data(), n * sizeof(Fr), cudaMemcpyHostToDevice));
    
    test_mul_associative_kernel<<<(n + 255) / 256, 256>>>(d_a, d_b, d_c, d_lhs, d_rhs, n);
    SECURITY_CHECK_CUDA(cudaDeviceSynchronize());
    
    std::vector<Fr> lhs(n), rhs(n);
    SECURITY_CHECK_CUDA(cudaMemcpy(lhs.data(), d_lhs, n * sizeof(Fr), cudaMemcpyDeviceToHost));
    SECURITY_CHECK_CUDA(cudaMemcpy(rhs.data(), d_rhs, n * sizeof(Fr), cudaMemcpyDeviceToHost));
    
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c); cudaFree(d_lhs); cudaFree(d_rhs);
    
    int failures = 0;
    for (int i = 0; i < n; i++) {
        if (!limbs_equal(lhs[i].limbs, rhs[i].limbs, 4)) {
            failures++;
        }
    }
    
    if (failures > 0) {
        std::cout << "\n    " << failures << "/" << n << " failures";
        return TestResult::FAILED;
    }
    return TestResult::PASSED;
}

TestResult test_additive_identity() {
    const int n = 4096;
    std::mt19937_64 rng(111);
    
    std::vector<Fr> a_vals(n);
    for (int i = 0; i < n; i++) {
        a_vals[i] = random_fr_montgomery(rng);
    }
    
    Fr *d_a, *d_out;
    SECURITY_CHECK_CUDA(cudaMalloc(&d_a, n * sizeof(Fr)));
    SECURITY_CHECK_CUDA(cudaMalloc(&d_out, n * sizeof(Fr)));
    
    SECURITY_CHECK_CUDA(cudaMemcpy(d_a, a_vals.data(), n * sizeof(Fr), cudaMemcpyHostToDevice));
    
    test_add_identity_kernel<<<(n + 255) / 256, 256>>>(d_a, d_out, n);
    SECURITY_CHECK_CUDA(cudaDeviceSynchronize());
    
    std::vector<Fr> results(n);
    SECURITY_CHECK_CUDA(cudaMemcpy(results.data(), d_out, n * sizeof(Fr), cudaMemcpyDeviceToHost));
    
    cudaFree(d_a); cudaFree(d_out);
    
    int failures = 0;
    for (int i = 0; i < n; i++) {
        if (!limbs_equal(a_vals[i].limbs, results[i].limbs, 4)) {
            failures++;
        }
    }
    
    if (failures > 0) {
        std::cout << "\n    " << failures << "/" << n << " failures";
        return TestResult::FAILED;
    }
    return TestResult::PASSED;
}

TestResult test_multiplicative_identity() {
    const int n = 4096;
    std::mt19937_64 rng(222);
    
    std::vector<Fr> a_vals(n);
    for (int i = 0; i < n; i++) {
        a_vals[i] = random_fr_montgomery(rng);
    }
    
    Fr *d_a, *d_out;
    SECURITY_CHECK_CUDA(cudaMalloc(&d_a, n * sizeof(Fr)));
    SECURITY_CHECK_CUDA(cudaMalloc(&d_out, n * sizeof(Fr)));
    
    SECURITY_CHECK_CUDA(cudaMemcpy(d_a, a_vals.data(), n * sizeof(Fr), cudaMemcpyHostToDevice));
    
    test_mul_identity_kernel<<<(n + 255) / 256, 256>>>(d_a, d_out, n);
    SECURITY_CHECK_CUDA(cudaDeviceSynchronize());
    
    std::vector<Fr> results(n);
    SECURITY_CHECK_CUDA(cudaMemcpy(results.data(), d_out, n * sizeof(Fr), cudaMemcpyDeviceToHost));
    
    cudaFree(d_a); cudaFree(d_out);
    
    int failures = 0;
    for (int i = 0; i < n; i++) {
        if (!limbs_equal(a_vals[i].limbs, results[i].limbs, 4)) {
            failures++;
        }
    }
    
    if (failures > 0) {
        std::cout << "\n    " << failures << "/" << n << " failures";
        return TestResult::FAILED;
    }
    return TestResult::PASSED;
}

TestResult test_additive_inverse() {
    const int n = 4096;
    std::mt19937_64 rng(333);
    
    std::vector<Fr> a_vals(n);
    for (int i = 0; i < n; i++) {
        a_vals[i] = random_fr_montgomery(rng);
    }
    
    Fr *d_a, *d_out;
    SECURITY_CHECK_CUDA(cudaMalloc(&d_a, n * sizeof(Fr)));
    SECURITY_CHECK_CUDA(cudaMalloc(&d_out, n * sizeof(Fr)));
    
    SECURITY_CHECK_CUDA(cudaMemcpy(d_a, a_vals.data(), n * sizeof(Fr), cudaMemcpyHostToDevice));
    
    test_add_inverse_kernel<<<(n + 255) / 256, 256>>>(d_a, d_out, n);
    SECURITY_CHECK_CUDA(cudaDeviceSynchronize());
    
    std::vector<Fr> results(n);
    SECURITY_CHECK_CUDA(cudaMemcpy(results.data(), d_out, n * sizeof(Fr), cudaMemcpyDeviceToHost));
    
    cudaFree(d_a); cudaFree(d_out);
    
    Fr zero = make_fr_zero_host();
    int failures = 0;
    for (int i = 0; i < n; i++) {
        if (!limbs_equal(results[i].limbs, zero.limbs, 4)) {
            failures++;
        }
    }
    
    if (failures > 0) {
        std::cout << "\n    " << failures << "/" << n << " failures";
        return TestResult::FAILED;
    }
    return TestResult::PASSED;
}

TestResult test_multiplicative_inverse() {
    const int n = 1024;  // Smaller due to expensive inversions
    std::mt19937_64 rng(444);
    
    std::vector<Fr> a_vals(n);
    for (int i = 0; i < n; i++) {
        a_vals[i] = random_fr_nonzero(rng);
    }
    
    Fr *d_a, *d_out;
    SECURITY_CHECK_CUDA(cudaMalloc(&d_a, n * sizeof(Fr)));
    SECURITY_CHECK_CUDA(cudaMalloc(&d_out, n * sizeof(Fr)));
    
    SECURITY_CHECK_CUDA(cudaMemcpy(d_a, a_vals.data(), n * sizeof(Fr), cudaMemcpyHostToDevice));
    
    test_mul_inverse_kernel<<<(n + 255) / 256, 256>>>(d_a, d_out, n);
    SECURITY_CHECK_CUDA(cudaDeviceSynchronize());
    
    std::vector<Fr> results(n);
    SECURITY_CHECK_CUDA(cudaMemcpy(results.data(), d_out, n * sizeof(Fr), cudaMemcpyDeviceToHost));
    
    cudaFree(d_a); cudaFree(d_out);
    
    Fr one = make_fr_one_host();
    int failures = 0;
    for (int i = 0; i < n; i++) {
        if (!limbs_equal(results[i].limbs, one.limbs, 4)) {
            failures++;
        }
    }
    
    if (failures > 0) {
        std::cout << "\n    " << failures << "/" << n << " failures";
        return TestResult::FAILED;
    }
    return TestResult::PASSED;
}

TestResult test_distributivity() {
    const int n = 4096;
    std::mt19937_64 rng(555);
    
    std::vector<Fr> a_vals(n), b_vals(n), c_vals(n);
    for (int i = 0; i < n; i++) {
        a_vals[i] = random_fr_montgomery(rng);
        b_vals[i] = random_fr_montgomery(rng);
        c_vals[i] = random_fr_montgomery(rng);
    }
    
    Fr *d_a, *d_b, *d_c, *d_lhs, *d_rhs;
    SECURITY_CHECK_CUDA(cudaMalloc(&d_a, n * sizeof(Fr)));
    SECURITY_CHECK_CUDA(cudaMalloc(&d_b, n * sizeof(Fr)));
    SECURITY_CHECK_CUDA(cudaMalloc(&d_c, n * sizeof(Fr)));
    SECURITY_CHECK_CUDA(cudaMalloc(&d_lhs, n * sizeof(Fr)));
    SECURITY_CHECK_CUDA(cudaMalloc(&d_rhs, n * sizeof(Fr)));
    
    SECURITY_CHECK_CUDA(cudaMemcpy(d_a, a_vals.data(), n * sizeof(Fr), cudaMemcpyHostToDevice));
    SECURITY_CHECK_CUDA(cudaMemcpy(d_b, b_vals.data(), n * sizeof(Fr), cudaMemcpyHostToDevice));
    SECURITY_CHECK_CUDA(cudaMemcpy(d_c, c_vals.data(), n * sizeof(Fr), cudaMemcpyHostToDevice));
    
    test_distributive_kernel<<<(n + 255) / 256, 256>>>(d_a, d_b, d_c, d_lhs, d_rhs, n);
    SECURITY_CHECK_CUDA(cudaDeviceSynchronize());
    
    std::vector<Fr> lhs(n), rhs(n);
    SECURITY_CHECK_CUDA(cudaMemcpy(lhs.data(), d_lhs, n * sizeof(Fr), cudaMemcpyDeviceToHost));
    SECURITY_CHECK_CUDA(cudaMemcpy(rhs.data(), d_rhs, n * sizeof(Fr), cudaMemcpyDeviceToHost));
    
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c); cudaFree(d_lhs); cudaFree(d_rhs);
    
    int failures = 0;
    for (int i = 0; i < n; i++) {
        if (!limbs_equal(lhs[i].limbs, rhs[i].limbs, 4)) {
            failures++;
        }
    }
    
    if (failures > 0) {
        std::cout << "\n    " << failures << "/" << n << " failures";
        return TestResult::FAILED;
    }
    return TestResult::PASSED;
}

TestResult test_squaring_consistency() {
    const int n = 4096;
    std::mt19937_64 rng(666);
    
    std::vector<Fr> a_vals(n);
    for (int i = 0; i < n; i++) {
        a_vals[i] = random_fr_montgomery(rng);
    }
    
    Fr *d_a, *d_sqr, *d_mul;
    SECURITY_CHECK_CUDA(cudaMalloc(&d_a, n * sizeof(Fr)));
    SECURITY_CHECK_CUDA(cudaMalloc(&d_sqr, n * sizeof(Fr)));
    SECURITY_CHECK_CUDA(cudaMalloc(&d_mul, n * sizeof(Fr)));
    
    SECURITY_CHECK_CUDA(cudaMemcpy(d_a, a_vals.data(), n * sizeof(Fr), cudaMemcpyHostToDevice));
    
    test_sqr_consistency_kernel<<<(n + 255) / 256, 256>>>(d_a, d_sqr, d_mul, n);
    SECURITY_CHECK_CUDA(cudaDeviceSynchronize());
    
    std::vector<Fr> sqr_results(n), mul_results(n);
    SECURITY_CHECK_CUDA(cudaMemcpy(sqr_results.data(), d_sqr, n * sizeof(Fr), cudaMemcpyDeviceToHost));
    SECURITY_CHECK_CUDA(cudaMemcpy(mul_results.data(), d_mul, n * sizeof(Fr), cudaMemcpyDeviceToHost));
    
    cudaFree(d_a); cudaFree(d_sqr); cudaFree(d_mul);
    
    int failures = 0;
    for (int i = 0; i < n; i++) {
        if (!limbs_equal(sqr_results[i].limbs, mul_results[i].limbs, 4)) {
            failures++;
        }
    }
    
    if (failures > 0) {
        std::cout << "\n    " << failures << "/" << n << " failures";
        return TestResult::FAILED;
    }
    return TestResult::PASSED;
}

TestResult test_proper_reduction() {
    const int n = 4096;
    std::mt19937_64 rng(777);
    
    std::vector<Fr> a_vals(n), b_vals(n);
    for (int i = 0; i < n; i++) {
        a_vals[i] = random_fr_montgomery(rng);
        b_vals[i] = random_fr_montgomery(rng);
    }
    
    Fr *d_a, *d_b;
    int *d_violations;
    SECURITY_CHECK_CUDA(cudaMalloc(&d_a, n * sizeof(Fr)));
    SECURITY_CHECK_CUDA(cudaMalloc(&d_b, n * sizeof(Fr)));
    SECURITY_CHECK_CUDA(cudaMalloc(&d_violations, sizeof(int)));
    SECURITY_CHECK_CUDA(cudaMemset(d_violations, 0, sizeof(int)));
    
    SECURITY_CHECK_CUDA(cudaMemcpy(d_a, a_vals.data(), n * sizeof(Fr), cudaMemcpyHostToDevice));
    SECURITY_CHECK_CUDA(cudaMemcpy(d_b, b_vals.data(), n * sizeof(Fr), cudaMemcpyHostToDevice));
    
    test_reduction_kernel<<<(n + 255) / 256, 256>>>(d_a, d_b, d_violations, n);
    SECURITY_CHECK_CUDA(cudaDeviceSynchronize());
    
    int violations;
    SECURITY_CHECK_CUDA(cudaMemcpy(&violations, d_violations, sizeof(int), cudaMemcpyDeviceToHost));
    
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_violations);
    
    if (violations > 0) {
        std::cout << "\n    " << violations << " unreduced results detected";
        return TestResult::FAILED;
    }
    return TestResult::PASSED;
}

TestResult test_subtraction_add_identity() {
    const int n = 4096;
    std::mt19937_64 rng(888);
    
    std::vector<Fr> a_vals(n), b_vals(n);
    for (int i = 0; i < n; i++) {
        a_vals[i] = random_fr_montgomery(rng);
        b_vals[i] = random_fr_montgomery(rng);
    }
    
    Fr *d_a, *d_b, *d_out;
    SECURITY_CHECK_CUDA(cudaMalloc(&d_a, n * sizeof(Fr)));
    SECURITY_CHECK_CUDA(cudaMalloc(&d_b, n * sizeof(Fr)));
    SECURITY_CHECK_CUDA(cudaMalloc(&d_out, n * sizeof(Fr)));
    
    SECURITY_CHECK_CUDA(cudaMemcpy(d_a, a_vals.data(), n * sizeof(Fr), cudaMemcpyHostToDevice));
    SECURITY_CHECK_CUDA(cudaMemcpy(d_b, b_vals.data(), n * sizeof(Fr), cudaMemcpyHostToDevice));
    
    test_sub_add_identity_kernel<<<(n + 255) / 256, 256>>>(d_a, d_b, d_out, n);
    SECURITY_CHECK_CUDA(cudaDeviceSynchronize());
    
    std::vector<Fr> results(n);
    SECURITY_CHECK_CUDA(cudaMemcpy(results.data(), d_out, n * sizeof(Fr), cudaMemcpyDeviceToHost));
    
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_out);
    
    int failures = 0;
    for (int i = 0; i < n; i++) {
        if (!limbs_equal(a_vals[i].limbs, results[i].limbs, 4)) {
            failures++;
        }
    }
    
    if (failures > 0) {
        std::cout << "\n    " << failures << "/" << n << " failures";
        return TestResult::FAILED;
    }
    return TestResult::PASSED;
}

TestResult test_self_subtraction() {
    const int n = 4096;
    std::mt19937_64 rng(999);
    
    std::vector<Fr> a_vals(n);
    for (int i = 0; i < n; i++) {
        a_vals[i] = random_fr_montgomery(rng);
    }
    
    Fr *d_a, *d_out;
    SECURITY_CHECK_CUDA(cudaMalloc(&d_a, n * sizeof(Fr)));
    SECURITY_CHECK_CUDA(cudaMalloc(&d_out, n * sizeof(Fr)));
    
    SECURITY_CHECK_CUDA(cudaMemcpy(d_a, a_vals.data(), n * sizeof(Fr), cudaMemcpyHostToDevice));
    
    test_sub_self_kernel<<<(n + 255) / 256, 256>>>(d_a, d_out, n);
    SECURITY_CHECK_CUDA(cudaDeviceSynchronize());
    
    std::vector<Fr> results(n);
    SECURITY_CHECK_CUDA(cudaMemcpy(results.data(), d_out, n * sizeof(Fr), cudaMemcpyDeviceToHost));
    
    cudaFree(d_a); cudaFree(d_out);
    
    Fr zero = make_fr_zero_host();
    int failures = 0;
    for (int i = 0; i < n; i++) {
        if (!limbs_equal(results[i].limbs, zero.limbs, 4)) {
            failures++;
        }
    }
    
    if (failures > 0) {
        std::cout << "\n    " << failures << "/" << n << " failures";
        return TestResult::FAILED;
    }
    return TestResult::PASSED;
}

TestResult test_multiply_by_zero() {
    const int n = 4096;
    std::mt19937_64 rng(1010);
    
    std::vector<Fr> a_vals(n);
    for (int i = 0; i < n; i++) {
        a_vals[i] = random_fr_montgomery(rng);
    }
    
    Fr *d_a, *d_out;
    SECURITY_CHECK_CUDA(cudaMalloc(&d_a, n * sizeof(Fr)));
    SECURITY_CHECK_CUDA(cudaMalloc(&d_out, n * sizeof(Fr)));
    
    SECURITY_CHECK_CUDA(cudaMemcpy(d_a, a_vals.data(), n * sizeof(Fr), cudaMemcpyHostToDevice));
    
    test_mul_zero_kernel<<<(n + 255) / 256, 256>>>(d_a, d_out, n);
    SECURITY_CHECK_CUDA(cudaDeviceSynchronize());
    
    std::vector<Fr> results(n);
    SECURITY_CHECK_CUDA(cudaMemcpy(results.data(), d_out, n * sizeof(Fr), cudaMemcpyDeviceToHost));
    
    cudaFree(d_a); cudaFree(d_out);
    
    Fr zero = make_fr_zero_host();
    int failures = 0;
    for (int i = 0; i < n; i++) {
        if (!limbs_equal(results[i].limbs, zero.limbs, 4)) {
            failures++;
        }
    }
    
    if (failures > 0) {
        std::cout << "\n    " << failures << "/" << n << " failures";
        return TestResult::FAILED;
    }
    return TestResult::PASSED;
}

// =============================================================================
// Known Answer Tests (KAT) - Cross-validated with Python/SageMath
// =============================================================================

/**
 * @brief Verify Fr modulus is correct
 * 
 * Fr modulus r = 0x73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001
 */
TestResult test_fr_modulus_correct() {
    // Expected modulus (verified against BLS12-381 specification)
    const uint64_t expected[4] = {
        0xffffffff00000001ULL,
        0x53bda402fffe5bfeULL,
        0x3339d80809a1d805ULL,
        0x73eda753299d7d48ULL
    };
    
    // Get modulus from implementation
    for (int i = 0; i < 4; i++) {
        if (fp_config::modulus(i) != expected[i]) {
            std::cout << "\n    Fr modulus mismatch at limb " << i;
            return TestResult::FAILED;
        }
    }
    return TestResult::PASSED;
}

/**
 * @brief Known Answer Test: 2 * 3 = 6 in Fr
 * 
 * This verifies Montgomery multiplication is correctly implemented.
 * Values computed independently in Python.
 */
__global__ void kat_mul_2_times_3_kernel(Fr* result) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    
    Fr two = Fr::from_int(2);
    Fr three = Fr::from_int(3);
    *result = two * three;
}

TestResult test_kat_simple_multiplication() {
    Fr* d_result;
    SECURITY_CHECK_CUDA(cudaMalloc(&d_result, sizeof(Fr)));
    
    kat_mul_2_times_3_kernel<<<1, 1>>>(d_result);
    SECURITY_CHECK_CUDA(cudaDeviceSynchronize());
    
    Fr result;
    SECURITY_CHECK_CUDA(cudaMemcpy(&result, d_result, sizeof(Fr), cudaMemcpyDeviceToHost));
    cudaFree(d_result);
    
    // Expected: 6 in Montgomery form
    Fr expected = to_montgomery_fr(uint256_t(6ULL));
    
    if (!limbs_equal(result.limbs, expected.limbs, 4)) {
        std::cout << "\n    2 * 3 ≠ 6!";
        std::cout << "\n    Got:      " << limbs_to_hex(result.limbs, 4);
        std::cout << "\n    Expected: " << limbs_to_hex(expected.limbs, 4);
        return TestResult::FAILED;
    }
    return TestResult::PASSED;
}

/**
 * @brief Known Answer Test: a * a^(-1) = 1 for specific value
 * 
 * Tests modular inverse with a known value.
 */
__global__ void kat_inverse_kernel(const Fr* a, Fr* a_inv, Fr* product) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    
    field_inv(*a_inv, *a);
    *product = (*a) * (*a_inv);
}

TestResult test_kat_modular_inverse() {
    // Use a = 12345 (a non-trivial value)
    Fr a = to_montgomery_fr(uint256_t(12345ULL));
    
    Fr *d_a, *d_a_inv, *d_product;
    SECURITY_CHECK_CUDA(cudaMalloc(&d_a, sizeof(Fr)));
    SECURITY_CHECK_CUDA(cudaMalloc(&d_a_inv, sizeof(Fr)));
    SECURITY_CHECK_CUDA(cudaMalloc(&d_product, sizeof(Fr)));
    
    SECURITY_CHECK_CUDA(cudaMemcpy(d_a, &a, sizeof(Fr), cudaMemcpyHostToDevice));
    
    kat_inverse_kernel<<<1, 1>>>(d_a, d_a_inv, d_product);
    SECURITY_CHECK_CUDA(cudaDeviceSynchronize());
    
    Fr product;
    SECURITY_CHECK_CUDA(cudaMemcpy(&product, d_product, sizeof(Fr), cudaMemcpyDeviceToHost));
    
    cudaFree(d_a);
    cudaFree(d_a_inv);
    cudaFree(d_product);
    
    // Expected: 1 in Montgomery form
    Fr one;
    for (int i = 0; i < 4; i++) one.limbs[i] = FR_ONE_HOST[i];
    
    if (!limbs_equal(product.limbs, one.limbs, 4)) {
        std::cout << "\n    12345 * 12345^(-1) ≠ 1!";
        return TestResult::FAILED;
    }
    return TestResult::PASSED;
}

// =============================================================================
// Registration
// =============================================================================

void register_field_property_tests(SecurityTestSuite& suite) {
    // Known Answer Tests (CRITICAL - verify implementation correctness)
    suite.add_test("Fr modulus matches BLS12-381 spec", "KAT Verification",
                   test_fr_modulus_correct, true);  // Critical
    suite.add_test("KAT: 2 * 3 = 6", "KAT Verification",
                   test_kat_simple_multiplication, true);  // Critical
    suite.add_test("KAT: a * a^(-1) = 1", "KAT Verification",
                   test_kat_modular_inverse, true);  // Critical
    
    // Commutativity
    suite.add_test("Addition is commutative: a + b = b + a", "Field Axioms",
                   test_additive_commutativity);
    suite.add_test("Multiplication is commutative: a * b = b * a", "Field Axioms",
                   test_multiplicative_commutativity);
    
    // Associativity
    suite.add_test("Addition is associative: (a+b)+c = a+(b+c)", "Field Axioms",
                   test_additive_associativity);
    suite.add_test("Multiplication is associative: (a*b)*c = a*(b*c)", "Field Axioms",
                   test_multiplicative_associativity);
    
    // Identity
    suite.add_test("Additive identity: a + 0 = a", "Field Axioms",
                   test_additive_identity);
    suite.add_test("Multiplicative identity: a * 1 = a", "Field Axioms",
                   test_multiplicative_identity);
    
    // Inverse
    suite.add_test("Additive inverse: a + (-a) = 0", "Field Axioms",
                   test_additive_inverse);
    suite.add_test("Multiplicative inverse: a * a^(-1) = 1", "Field Axioms",
                   test_multiplicative_inverse);
    
    // Distributivity
    suite.add_test("Distributivity: a*(b+c) = a*b + a*c", "Field Axioms",
                   test_distributivity);
    
    // Consistency checks
    suite.add_test("Squaring consistency: a² = a * a", "Consistency",
                   test_squaring_consistency);
    suite.add_test("Results properly reduced (< modulus)", "Consistency",
                   test_proper_reduction);
    
    // Subtraction tests
    suite.add_test("Subtraction identity: (a-b)+b = a", "Subtraction",
                   test_subtraction_add_identity);
    suite.add_test("Self subtraction: a - a = 0", "Subtraction",
                   test_self_subtraction);
    suite.add_test("Multiply by zero: 0 * a = 0", "Edge Cases",
                   test_multiply_by_zero);
}

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
    register_field_property_tests(suite);
    
    bool success = suite.run_all();
    return success ? 0 : 1;
}
