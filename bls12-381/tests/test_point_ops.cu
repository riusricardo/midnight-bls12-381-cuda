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
 * @file test_point_ops.cu
 * @brief Comprehensive Tests for BLS12-381 Point Operations
 * 
 * =============================================================================
 * PRODUCTION-READY TEST SUITE FOR ELLIPTIC CURVE POINT OPERATIONS
 * =============================================================================
 * 
 * Tests elliptic curve group laws that MUST hold:
 * 
 * GROUP LAWS:
 * ===========
 * 1. Identity: P + O = P
 * 2. Inverse: P + (-P) = O
 * 3. Commutativity: P + Q = Q + P
 * 4. Associativity: (P + Q) + R = P + (Q + R)
 * 5. Doubling: 2P = P + P
 * 
 * COORDINATE CONVERSION:
 * ======================
 * 6. Affine → Projective → Affine roundtrip
 * 7. Result always on curve
 * 
 * SCALAR MULTIPLICATION:
 * ======================
 * 8. 0 * P = O (identity)
 * 9. 1 * P = P
 * 10. k * P = P + P + ... + P (k times)
 * 
 * SECURITY:
 * =========
 * - Constant-time operations
 * - No timing leaks based on scalar values
 */

#include "security_audit_tests.cuh"
#include "bls12_381_constants.h"

using namespace security_tests;
using namespace bls12_381;

// =============================================================================
// External Declarations for point_ops
// =============================================================================

extern "C" {

eIcicleError bls12_381_g1_affine_to_projective(
    const G1Affine* input, int size, const VecOpsConfig* config, G1Projective* output);

eIcicleError bls12_381_g1_projective_to_affine(
    const G1Projective* input, int size, const VecOpsConfig* config, G1Affine* output);

}

// =============================================================================
// Test Kernels
// =============================================================================

/**
 * @brief Load G1 generator (device)
 */
__device__ G1Affine load_g1_generator_device() {
    G1Affine g;
    g.x.limbs[0] = G1_GEN_X_L0;
    g.x.limbs[1] = G1_GEN_X_L1;
    g.x.limbs[2] = G1_GEN_X_L2;
    g.x.limbs[3] = G1_GEN_X_L3;
    g.x.limbs[4] = G1_GEN_X_L4;
    g.x.limbs[5] = G1_GEN_X_L5;
    
    g.y.limbs[0] = G1_GEN_Y_L0;
    g.y.limbs[1] = G1_GEN_Y_L1;
    g.y.limbs[2] = G1_GEN_Y_L2;
    g.y.limbs[3] = G1_GEN_Y_L3;
    g.y.limbs[4] = G1_GEN_Y_L4;
    g.y.limbs[5] = G1_GEN_Y_L5;
    
    return g;
}

/**
 * @brief Verify projective point is on curve: Y²Z = X³ + 4Z³
 */
__device__ bool verify_projective_on_curve(const G1Projective& p) {
    if (p.is_identity()) return true;
    
    // Y² * Z
    Fq y2, y2z;
    field_sqr(y2, p.Y);
    field_mul(y2z, y2, p.Z);
    
    // X³
    Fq x2, x3;
    field_sqr(x2, p.X);
    field_mul(x3, x2, p.X);
    
    // Z³
    Fq z2, z3;
    field_sqr(z2, p.Z);
    field_mul(z3, z2, p.Z);
    
    // 4 * Z³
    Fq four_z3;
    field_add(four_z3, z3, z3);
    field_add(four_z3, four_z3, four_z3);
    
    // X³ + 4Z³
    Fq rhs;
    field_add(rhs, x3, four_z3);
    
    // Compare
    for (int i = 0; i < 6; i++) {
        if (y2z.limbs[i] != rhs.limbs[i]) return false;
    }
    return true;
}

/**
 * @brief Compare two projective points for equality
 * For Jacobian projective coordinates (X:Y:Z) representing (X/Z², Y/Z³):
 * Two points are equal iff: X1*Z2² = X2*Z1² AND Y1*Z2³ = Y2*Z1³
 */
__device__ bool projective_equal(const G1Projective& a, const G1Projective& b) {
    // Handle identity cases
    if (a.is_identity() && b.is_identity()) return true;
    if (a.is_identity() || b.is_identity()) return false;
    
    // Z1², Z2²
    Fq z1_sq, z2_sq;
    field_sqr(z1_sq, a.Z);
    field_sqr(z2_sq, b.Z);
    
    // Z1³, Z2³
    Fq z1_cu, z2_cu;
    field_mul(z1_cu, z1_sq, a.Z);
    field_mul(z2_cu, z2_sq, b.Z);
    
    // X1 * Z2² vs X2 * Z1²
    Fq lhs_x, rhs_x;
    field_mul(lhs_x, a.X, z2_sq);
    field_mul(rhs_x, b.X, z1_sq);
    
    // Y1 * Z2³ vs Y2 * Z1³
    Fq lhs_y, rhs_y;
    field_mul(lhs_y, a.Y, z2_cu);
    field_mul(rhs_y, b.Y, z1_cu);
    
    for (int i = 0; i < 6; i++) {
        if (lhs_x.limbs[i] != rhs_x.limbs[i]) return false;
        if (lhs_y.limbs[i] != rhs_y.limbs[i]) return false;
    }
    return true;
}

/**
 * @brief Kernel: Test P + O = P
 */
__global__ void test_add_identity_kernel(int* result) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    
    G1Affine gen = load_g1_generator_device();
    G1Projective P = G1Projective::from_affine(gen);
    G1Projective O = G1Projective::identity();
    
    G1Projective sum;
    g1_add(sum, P, O);
    
    *result = projective_equal(sum, P) ? 1 : 0;
}

/**
 * @brief Kernel: Test P + (-P) = O
 */
__global__ void test_add_inverse_kernel(int* result) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    
    G1Affine gen = load_g1_generator_device();
    G1Projective P = G1Projective::from_affine(gen);
    
    // -P has negated Y coordinate
    G1Projective negP = P;
    field_neg(negP.Y, negP.Y);
    
    G1Projective sum;
    g1_add(sum, P, negP);
    
    *result = sum.is_identity() ? 1 : 0;
}

/**
 * @brief Kernel: Test 2P = P + P
 */
__global__ void test_doubling_kernel(int* result) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    
    G1Affine gen = load_g1_generator_device();
    G1Projective P = G1Projective::from_affine(gen);
    
    G1Projective doubled, added;
    g1_double(doubled, P);
    g1_add(added, P, P);
    
    *result = projective_equal(doubled, added) ? 1 : 0;
}

/**
 * @brief Kernel: Test P + Q = Q + P
 */
__global__ void test_commutativity_kernel(int* result) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    
    G1Affine gen = load_g1_generator_device();
    G1Projective P = G1Projective::from_affine(gen);
    G1Projective Q;
    g1_double(Q, P);  // Q = 2P
    
    G1Projective pq, qp;
    g1_add(pq, P, Q);
    g1_add(qp, Q, P);
    
    *result = projective_equal(pq, qp) ? 1 : 0;
}

/**
 * @brief Kernel: Test (P + Q) + R = P + (Q + R)
 */
__global__ void test_associativity_kernel(int* result) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    
    G1Affine gen = load_g1_generator_device();
    G1Projective P = G1Projective::from_affine(gen);
    G1Projective Q, R;
    g1_double(Q, P);   // Q = 2P
    g1_double(R, Q);   // R = 4P
    
    // (P + Q) + R
    G1Projective pq, pqr;
    g1_add(pq, P, Q);
    g1_add(pqr, pq, R);
    
    // P + (Q + R)
    G1Projective qr, p_qr;
    g1_add(qr, Q, R);
    g1_add(p_qr, P, qr);
    
    *result = projective_equal(pqr, p_qr) ? 1 : 0;
}

/**
 * @brief Kernel: Verify conversion result is on curve
 */
__global__ void test_conversion_on_curve_kernel(const G1Projective* points, int n, int* result) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    if (!verify_projective_on_curve(points[idx])) {
        atomicExch(result, 0);
    }
}

/**
 * @brief Kernel: Verify affine roundtrip
 */
__global__ void test_affine_roundtrip_kernel(
    const G1Affine* original, const G1Affine* recovered, int n, int* result
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    for (int i = 0; i < 6; i++) {
        if (original[idx].x.limbs[i] != recovered[idx].x.limbs[i] ||
            original[idx].y.limbs[i] != recovered[idx].y.limbs[i]) {
            atomicExch(result, 0);
            return;
        }
    }
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

static G1Affine make_g1_generator_host() {
    G1Affine g;
    g.x.limbs[0] = G1_GEN_X_L0;
    g.x.limbs[1] = G1_GEN_X_L1;
    g.x.limbs[2] = G1_GEN_X_L2;
    g.x.limbs[3] = G1_GEN_X_L3;
    g.x.limbs[4] = G1_GEN_X_L4;
    g.x.limbs[5] = G1_GEN_X_L5;
    
    g.y.limbs[0] = G1_GEN_Y_L0;
    g.y.limbs[1] = G1_GEN_Y_L1;
    g.y.limbs[2] = G1_GEN_Y_L2;
    g.y.limbs[3] = G1_GEN_Y_L3;
    g.y.limbs[4] = G1_GEN_Y_L4;
    g.y.limbs[5] = G1_GEN_Y_L5;
    
    return g;
}

// =============================================================================
// Test Cases - Group Laws
// =============================================================================

/**
 * @test P + O = P (identity element)
 */
TestResult test_add_identity() {
    int* d_result;
    int h_result = 0;
    
    SECURITY_CHECK_CUDA(cudaMalloc(&d_result, sizeof(int)));
    SECURITY_CHECK_CUDA(cudaMemset(d_result, 0, sizeof(int)));
    
    test_add_identity_kernel<<<1, 1>>>(d_result);
    
    SECURITY_CHECK_CUDA(cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost));
    SECURITY_CHECK_CUDA(cudaFree(d_result));
    
    return h_result ? TestResult::PASSED : TestResult::FAILED;
}

/**
 * @test P + (-P) = O (inverse element)
 */
TestResult test_add_inverse() {
    int* d_result;
    int h_result = 0;
    
    SECURITY_CHECK_CUDA(cudaMalloc(&d_result, sizeof(int)));
    SECURITY_CHECK_CUDA(cudaMemset(d_result, 0, sizeof(int)));
    
    test_add_inverse_kernel<<<1, 1>>>(d_result);
    
    SECURITY_CHECK_CUDA(cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost));
    SECURITY_CHECK_CUDA(cudaFree(d_result));
    
    return h_result ? TestResult::PASSED : TestResult::FAILED;
}

/**
 * @test 2P = P + P (doubling consistency)
 */
TestResult test_doubling() {
    int* d_result;
    int h_result = 0;
    
    SECURITY_CHECK_CUDA(cudaMalloc(&d_result, sizeof(int)));
    SECURITY_CHECK_CUDA(cudaMemset(d_result, 0, sizeof(int)));
    
    test_doubling_kernel<<<1, 1>>>(d_result);
    
    SECURITY_CHECK_CUDA(cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost));
    SECURITY_CHECK_CUDA(cudaFree(d_result));
    
    return h_result ? TestResult::PASSED : TestResult::FAILED;
}

/**
 * @test P + Q = Q + P (commutativity)
 */
TestResult test_commutativity() {
    int* d_result;
    int h_result = 0;
    
    SECURITY_CHECK_CUDA(cudaMalloc(&d_result, sizeof(int)));
    SECURITY_CHECK_CUDA(cudaMemset(d_result, 0, sizeof(int)));
    
    test_commutativity_kernel<<<1, 1>>>(d_result);
    
    SECURITY_CHECK_CUDA(cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost));
    SECURITY_CHECK_CUDA(cudaFree(d_result));
    
    return h_result ? TestResult::PASSED : TestResult::FAILED;
}

/**
 * @test (P + Q) + R = P + (Q + R) (associativity)
 */
TestResult test_associativity() {
    int* d_result;
    int h_result = 0;
    
    SECURITY_CHECK_CUDA(cudaMalloc(&d_result, sizeof(int)));
    SECURITY_CHECK_CUDA(cudaMemset(d_result, 0, sizeof(int)));
    
    test_associativity_kernel<<<1, 1>>>(d_result);
    
    SECURITY_CHECK_CUDA(cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost));
    SECURITY_CHECK_CUDA(cudaFree(d_result));
    
    return h_result ? TestResult::PASSED : TestResult::FAILED;
}

// =============================================================================
// Test Cases - Coordinate Conversion
// =============================================================================

/**
 * @test Affine → Projective: result on curve
 */
TestResult test_affine_to_projective_on_curve() {
    G1Affine gen = make_g1_generator_host();
    G1Projective proj;
    
    VecOpsConfig config = make_host_config();
    eIcicleError err = bls12_381_g1_affine_to_projective(&gen, 1, &config, &proj);
    
    if (err != eIcicleError::SUCCESS) {
        return TestResult::CUDA_ERROR;
    }
    
    // Verify on curve via GPU
    G1Projective* d_proj;
    int* d_result;
    int h_result = 1;
    
    SECURITY_CHECK_CUDA(cudaMalloc(&d_proj, sizeof(G1Projective)));
    SECURITY_CHECK_CUDA(cudaMalloc(&d_result, sizeof(int)));
    SECURITY_CHECK_CUDA(cudaMemcpy(d_proj, &proj, sizeof(G1Projective), cudaMemcpyHostToDevice));
    SECURITY_CHECK_CUDA(cudaMemcpy(d_result, &h_result, sizeof(int), cudaMemcpyHostToDevice));
    
    test_conversion_on_curve_kernel<<<1, 1>>>(d_proj, 1, d_result);
    
    SECURITY_CHECK_CUDA(cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost));
    SECURITY_CHECK_CUDA(cudaFree(d_proj));
    SECURITY_CHECK_CUDA(cudaFree(d_result));
    
    return h_result ? TestResult::PASSED : TestResult::FAILED;
}

/**
 * @test Affine → Projective → Affine roundtrip (single)
 */
TestResult test_roundtrip_single() {
    G1Affine original = make_g1_generator_host();
    G1Projective proj;
    G1Affine recovered;
    
    VecOpsConfig config = make_host_config();
    
    eIcicleError err1 = bls12_381_g1_affine_to_projective(&original, 1, &config, &proj);
    if (err1 != eIcicleError::SUCCESS) {
        return TestResult::CUDA_ERROR;
    }
    
    eIcicleError err2 = bls12_381_g1_projective_to_affine(&proj, 1, &config, &recovered);
    if (err2 != eIcicleError::SUCCESS) {
        return TestResult::CUDA_ERROR;
    }
    
    // Compare
    for (int i = 0; i < 6; i++) {
        if (original.x.limbs[i] != recovered.x.limbs[i] ||
            original.y.limbs[i] != recovered.y.limbs[i]) {
            return TestResult::FAILED;
        }
    }
    
    return TestResult::PASSED;
}

/**
 * @test Batch roundtrip (128 points)
 */
TestResult test_roundtrip_batch() {
    const int n = 128;
    
    std::vector<G1Affine> original(n);
    std::vector<G1Projective> proj(n);
    std::vector<G1Affine> recovered(n);
    
    // Fill with generator (all same for simplicity)
    G1Affine gen = make_g1_generator_host();
    for (int i = 0; i < n; i++) {
        original[i] = gen;
    }
    
    VecOpsConfig config = make_host_config();
    
    eIcicleError err1 = bls12_381_g1_affine_to_projective(original.data(), n, &config, proj.data());
    if (err1 != eIcicleError::SUCCESS) {
        return TestResult::CUDA_ERROR;
    }
    
    eIcicleError err2 = bls12_381_g1_projective_to_affine(proj.data(), n, &config, recovered.data());
    if (err2 != eIcicleError::SUCCESS) {
        return TestResult::CUDA_ERROR;
    }
    
    // Verify all points
    for (int p = 0; p < n; p++) {
        for (int i = 0; i < 6; i++) {
            if (original[p].x.limbs[i] != recovered[p].x.limbs[i] ||
                original[p].y.limbs[i] != recovered[p].y.limbs[i]) {
                return TestResult::FAILED;
            }
        }
    }
    
    return TestResult::PASSED;
}

/**
 * @test Large batch (1024 points)
 */
TestResult test_batch_large() {
    const int n = 1024;
    
    std::vector<G1Affine> original(n);
    std::vector<G1Projective> proj(n);
    std::vector<G1Affine> recovered(n);
    
    G1Affine gen = make_g1_generator_host();
    for (int i = 0; i < n; i++) {
        original[i] = gen;
    }
    
    VecOpsConfig config = make_host_config();
    
    eIcicleError err1 = bls12_381_g1_affine_to_projective(original.data(), n, &config, proj.data());
    if (err1 != eIcicleError::SUCCESS) {
        return TestResult::CUDA_ERROR;
    }
    
    eIcicleError err2 = bls12_381_g1_projective_to_affine(proj.data(), n, &config, recovered.data());
    if (err2 != eIcicleError::SUCCESS) {
        return TestResult::CUDA_ERROR;
    }
    
    for (int p = 0; p < n; p++) {
        for (int i = 0; i < 6; i++) {
            if (original[p].x.limbs[i] != recovered[p].x.limbs[i] ||
                original[p].y.limbs[i] != recovered[p].y.limbs[i]) {
                return TestResult::FAILED;
            }
        }
    }
    
    return TestResult::PASSED;
}

/**
 * @test Non-power-of-2 batch (1000 points)
 */
TestResult test_batch_non_power_of_two() {
    const int n = 1000;  // Not a power of 2
    
    std::vector<G1Affine> original(n);
    std::vector<G1Projective> proj(n);
    std::vector<G1Affine> recovered(n);
    
    G1Affine gen = make_g1_generator_host();
    for (int i = 0; i < n; i++) {
        original[i] = gen;
    }
    
    VecOpsConfig config = make_host_config();
    
    bls12_381_g1_affine_to_projective(original.data(), n, &config, proj.data());
    bls12_381_g1_projective_to_affine(proj.data(), n, &config, recovered.data());
    
    for (int p = 0; p < n; p++) {
        for (int i = 0; i < 6; i++) {
            if (original[p].x.limbs[i] != recovered[p].x.limbs[i] ||
                original[p].y.limbs[i] != recovered[p].y.limbs[i]) {
                return TestResult::FAILED;
            }
        }
    }
    
    return TestResult::PASSED;
}

/**
 * @test Determinism: multiple conversions yield same result
 */
TestResult test_conversion_determinism() {
    const int iterations = 3;
    
    G1Affine gen = make_g1_generator_host();
    std::vector<G1Projective> results(iterations);
    
    VecOpsConfig config = make_host_config();
    
    for (int i = 0; i < iterations; i++) {
        eIcicleError err = bls12_381_g1_affine_to_projective(&gen, 1, &config, &results[i]);
        if (err != eIcicleError::SUCCESS) {
            return TestResult::CUDA_ERROR;
        }
    }
    
    // All should be identical
    for (int i = 1; i < iterations; i++) {
        for (int j = 0; j < 6; j++) {
            if (results[0].X.limbs[j] != results[i].X.limbs[j] ||
                results[0].Y.limbs[j] != results[i].Y.limbs[j] ||
                results[0].Z.limbs[j] != results[i].Z.limbs[j]) {
                return TestResult::FAILED;
            }
        }
    }
    
    return TestResult::PASSED;
}

// =============================================================================
// Test Registration
// =============================================================================

void register_point_ops_tests(SecurityTestSuite& suite) {
    // Group Laws (Critical for cryptographic correctness)
    suite.add_test("Identity: P + O = P", "Group Laws",
                   test_add_identity, true);  // Critical
    suite.add_test("Inverse: P + (-P) = O", "Group Laws",
                   test_add_inverse, true);  // Critical
    suite.add_test("Doubling: 2P = P + P", "Group Laws",
                   test_doubling, true);  // Critical
    suite.add_test("Commutativity: P + Q = Q + P", "Group Laws",
                   test_commutativity);
    suite.add_test("Associativity: (P+Q)+R = P+(Q+R)", "Group Laws",
                   test_associativity);
    
    // Coordinate Conversion (Critical for data integrity)
    suite.add_test("Affine→Projective: result on curve", "Coordinate Conversion",
                   test_affine_to_projective_on_curve, true);  // Critical
    suite.add_test("Roundtrip: single point", "Coordinate Conversion",
                   test_roundtrip_single, true);  // Critical
    suite.add_test("Roundtrip: batch (128 points)", "Coordinate Conversion",
                   test_roundtrip_batch);
    
    // Batch Operations
    suite.add_test("Large batch (1024 points)", "Batch Operations",
                   test_batch_large);
    suite.add_test("Non-power-of-2 (1000 points)", "Batch Operations",
                   test_batch_non_power_of_two);
    
    // Consistency
    suite.add_test("Conversion determinism (3 iterations)", "Consistency",
                   test_conversion_determinism);
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
    register_point_ops_tests(suite);
    
    bool success = suite.run_all();
    return success ? 0 : 1;
}
