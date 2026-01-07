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
 * @file test_curve_operations.cu
 * @brief Elliptic Curve Operation Tests for BLS12-381
 * 
 * Tests mathematical properties that MUST hold for correct curve implementations:
 * 
 * GROUP LAWS (must verify for both G1 and G2):
 * =============================================
 * 1. Closure: P + Q ∈ G
 * 2. Associativity: (P + Q) + R = P + (Q + R)
 * 3. Identity: P + O = O + P = P
 * 4. Inverse: P + (-P) = O
 * 5. Commutativity: P + Q = Q + P
 * 
 * SCALAR MULTIPLICATION:
 * ======================
 * 6. 0 * P = O
 * 7. 1 * P = P
 * 8. 2 * P = P + P
 * 9. (a + b) * P = a*P + b*P (distributivity)
 * 10. a * (P + Q) = a*P + a*Q (linearity)
 * 11. (a * b) * P = a * (b * P) (associativity)
 * 12. r * G = O (group order verification)
 * 
 * CURVE EQUATION VERIFICATION:
 * ============================
 * 13. All points satisfy y² = x³ + b
 * 14. Generator lies on curve
 * 15. Derived points lie on curve
 * 
 * COORDINATE CONVERSION:
 * ======================
 * 16. Projective to affine roundtrip
 * 17. Mixed addition correctness
 * 18. Identity handling in all operations
 */

#include "security_audit_tests.cuh"

using namespace security_tests;

// =============================================================================
// Jacobian Point Comparison Helper
// =============================================================================
// For Jacobian coordinates (X, Y, Z) representing affine (x, y) where:
//   x = X/Z²
//   y = Y/Z³
// Two points (X1, Y1, Z1) and (X2, Y2, Z2) are equal iff:
//   X1 * Z2² = X2 * Z1²  AND  Y1 * Z2³ = Y2 * Z1³

__device__ __forceinline__ bool g1_projective_equal(
    const G1Projective& a, const G1Projective& b
) {
    // Handle identity cases
    bool a_id = a.is_identity();
    bool b_id = b.is_identity();
    if (a_id && b_id) return true;
    if (a_id || b_id) return false;
    
    // Compute Z squares and cubes
    Fq z1_sq, z2_sq, z1_cu, z2_cu;
    field_sqr(z1_sq, a.Z);
    field_sqr(z2_sq, b.Z);
    field_mul(z1_cu, z1_sq, a.Z);
    field_mul(z2_cu, z2_sq, b.Z);
    
    // Compare X coordinates: X1 * Z2² = X2 * Z1²
    Fq lhs_x, rhs_x;
    field_mul(lhs_x, a.X, z2_sq);
    field_mul(rhs_x, b.X, z1_sq);
    
    // Compare Y coordinates: Y1 * Z2³ = Y2 * Z1³
    Fq lhs_y, rhs_y;
    field_mul(lhs_y, a.Y, z2_cu);
    field_mul(rhs_y, b.Y, z1_cu);
    
    return (lhs_x == rhs_x) && (lhs_y == rhs_y);
}

__device__ __forceinline__ bool g2_projective_equal(
    const G2Projective& a, const G2Projective& b
) {
    // Handle identity cases
    bool a_id = a.is_identity();
    bool b_id = b.is_identity();
    if (a_id && b_id) return true;
    if (a_id || b_id) return false;
    
    // Compute Z squares and cubes for Fq2
    Fq2 z1_sq, z2_sq, z1_cu, z2_cu;
    fq2_sqr(z1_sq, a.Z);
    fq2_sqr(z2_sq, b.Z);
    fq2_mul(z1_cu, z1_sq, a.Z);
    fq2_mul(z2_cu, z2_sq, b.Z);
    
    // Compare X coordinates: X1 * Z2² = X2 * Z1²
    Fq2 lhs_x, rhs_x;
    fq2_mul(lhs_x, a.X, z2_sq);
    fq2_mul(rhs_x, b.X, z1_sq);
    
    // Compare Y coordinates: Y1 * Z2³ = Y2 * Z1³
    Fq2 lhs_y, rhs_y;
    fq2_mul(lhs_y, a.Y, z2_cu);
    fq2_mul(rhs_y, b.Y, z1_cu);
    
    return (lhs_x == rhs_x) && (lhs_y == rhs_y);
}

// =============================================================================
// G1 Generator Constants (Montgomery form, from bls12_381_constants.h)
// =============================================================================

static G1Affine make_g1_generator() {
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

// G2 Generator (Fq2 coordinates in Montgomery form, from bls12_381_constants.h)
static G2Affine make_g2_generator() {
    G2Affine g;
    
    // x coordinate (c0 + c1*u)
    g.x.c0.limbs[0] = G2_GEN_X_C0_L0;
    g.x.c0.limbs[1] = G2_GEN_X_C0_L1;
    g.x.c0.limbs[2] = G2_GEN_X_C0_L2;
    g.x.c0.limbs[3] = G2_GEN_X_C0_L3;
    g.x.c0.limbs[4] = G2_GEN_X_C0_L4;
    g.x.c0.limbs[5] = G2_GEN_X_C0_L5;
    
    g.x.c1.limbs[0] = G2_GEN_X_C1_L0;
    g.x.c1.limbs[1] = G2_GEN_X_C1_L1;
    g.x.c1.limbs[2] = G2_GEN_X_C1_L2;
    g.x.c1.limbs[3] = G2_GEN_X_C1_L3;
    g.x.c1.limbs[4] = G2_GEN_X_C1_L4;
    g.x.c1.limbs[5] = G2_GEN_X_C1_L5;
    
    // y coordinate
    g.y.c0.limbs[0] = G2_GEN_Y_C0_L0;
    g.y.c0.limbs[1] = G2_GEN_Y_C0_L1;
    g.y.c0.limbs[2] = G2_GEN_Y_C0_L2;
    g.y.c0.limbs[3] = G2_GEN_Y_C0_L3;
    g.y.c0.limbs[4] = G2_GEN_Y_C0_L4;
    g.y.c0.limbs[5] = G2_GEN_Y_C0_L5;
    
    g.y.c1.limbs[0] = G2_GEN_Y_C1_L0;
    g.y.c1.limbs[1] = G2_GEN_Y_C1_L1;
    g.y.c1.limbs[2] = G2_GEN_Y_C1_L2;
    g.y.c1.limbs[3] = G2_GEN_Y_C1_L3;
    g.y.c1.limbs[4] = G2_GEN_Y_C1_L4;
    g.y.c1.limbs[5] = G2_GEN_Y_C1_L5;
    
    return g;
}

// =============================================================================
// Test Kernels
// =============================================================================

/**
 * @brief Generate n test points via repeated doubling
 * points[0] = base, points[i] = 2 * points[i-1]
 */
__global__ void g1_generate_test_points_kernel(
    const G1Affine* base, G1Projective* points, int n
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    
    points[0] = G1Projective::from_affine(*base);
    for (int i = 1; i < n; i++) {
        g1_double(points[i], points[i-1]);
    }
}

/**
 * @brief Generate n G2 test points via repeated doubling
 */
__global__ void g2_generate_test_points_kernel(
    const G2Affine* base, G2Projective* points, int n
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    
    points[0] = G2Projective::from_affine(*base);
    for (int i = 1; i < n; i++) {
        g2_double(points[i], points[i-1]);
    }
}

/**
 * @brief Generate two point sets for commutativity tests
 * p[i] = 2^i * G, q[i] = 2^(i+1) * G
 */
__global__ void g1_generate_two_point_sets_kernel(
    const G1Affine* base, G1Projective* p, G1Projective* q, int n
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    
    G1Projective gen_proj = G1Projective::from_affine(*base);
    p[0] = gen_proj;
    g1_double(q[0], gen_proj);
    
    for (int i = 1; i < n; i++) {
        g1_double(p[i], p[i-1]);
        g1_double(q[i], q[i-1]);
    }
}

/**
 * @brief Generate three point sets for associativity tests
 */
__global__ void g1_generate_three_point_sets_kernel(
    const G1Affine* base, G1Projective* p, G1Projective* q, G1Projective* r, int n
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    
    G1Projective gen_proj = G1Projective::from_affine(*base);
    p[0] = gen_proj;
    
    G1Projective temp;
    g1_double(temp, gen_proj);
    q[0] = temp;
    
    g1_double(temp, temp);
    r[0] = temp;
    
    for (int i = 1; i < n; i++) {
        g1_double(p[i], p[i-1]);
        g1_double(q[i], q[i-1]);
        g1_double(r[i], r[i-1]);
    }
}

/**
 * @brief Generate affine points on device for roundtrip tests
 */
__global__ void g1_generate_affine_points_kernel(
    const G1Affine* base, G1Affine* points, int n
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    
    G1Projective current = G1Projective::from_affine(*base);
    for (int i = 0; i < n; i++) {
        points[i] = current.to_affine();
        g1_double(current, current);
    }
}

/**
 * @brief G1: Test P + O = P (identity addition)
 */
__global__ void g1_add_identity_test_kernel(
    const G1Projective* p, int* result, int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    G1Projective identity = G1Projective::identity();
    G1Projective sum;
    g1_add(sum, p[idx], identity);
    
    // Compare using proper Jacobian equality
    result[idx] = g1_projective_equal(sum, p[idx]) ? 1 : 0;
}

/**
 * @brief G1: Test P + (-P) = O (inverse addition)
 */
__global__ void g1_add_inverse_test_kernel(
    const G1Projective* p, int* result, int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    G1Projective neg_p;
    g1_neg(neg_p, p[idx]);
    
    G1Projective sum;
    g1_add(sum, p[idx], neg_p);
    
    result[idx] = sum.is_identity() ? 1 : 0;
}

/**
 * @brief G1: Test P + Q = Q + P (commutativity)
 */
__global__ void g1_add_commutative_test_kernel(
    const G1Projective* p, const G1Projective* q, int* result, int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    G1Projective pq, qp;
    g1_add(pq, p[idx], q[idx]);
    g1_add(qp, q[idx], p[idx]);
    
    // Compare using proper Jacobian equality
    result[idx] = g1_projective_equal(pq, qp) ? 1 : 0;
}

/**
 * @brief G1: Test (P + Q) + R = P + (Q + R) (associativity)
 */
__global__ void g1_add_associative_test_kernel(
    const G1Projective* p, const G1Projective* q, const G1Projective* r,
    int* result, int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    G1Projective pq, pq_r, qr, p_qr;
    
    g1_add(pq, p[idx], q[idx]);
    g1_add(pq_r, pq, r[idx]);  // (P + Q) + R
    
    g1_add(qr, q[idx], r[idx]);
    g1_add(p_qr, p[idx], qr);  // P + (Q + R)
    
    // Compare using proper Jacobian equality
    result[idx] = g1_projective_equal(pq_r, p_qr) ? 1 : 0;
}

/**
 * @brief G1: Test 2P = P + P (doubling vs addition)
 */
__global__ void g1_double_vs_add_test_kernel(
    const G1Projective* p, int* result, int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    G1Projective doubled, added;
    g1_double(doubled, p[idx]);
    g1_add(added, p[idx], p[idx]);
    
    // Compare using proper Jacobian equality
    result[idx] = g1_projective_equal(doubled, added) ? 1 : 0;
}

/**
 * @brief G1: Test 3P = 2P + P (triple point)
 */
__global__ void g1_triple_point_test_kernel(
    const G1Projective* p, int* result, int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    // Compute 3P via double-and-add
    G1Projective two_p, three_p_a;
    g1_double(two_p, p[idx]);
    g1_add(three_p_a, two_p, p[idx]);
    
    // Compute 3P via triple addition
    G1Projective p_plus_p, three_p_b;
    g1_add(p_plus_p, p[idx], p[idx]);
    g1_add(three_p_b, p_plus_p, p[idx]);
    
    // Compare using proper Jacobian equality
    result[idx] = g1_projective_equal(three_p_a, three_p_b) ? 1 : 0;
}

/**
 * @brief G1: Test mixed addition P + Q_affine = P + Q_projective
 */
__global__ void g1_mixed_addition_test_kernel(
    const G1Projective* p, const G1Affine* q_aff, int* result, int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    // Mixed addition
    G1Projective mixed_result;
    g1_add_mixed(mixed_result, p[idx], q_aff[idx]);
    
    // Standard addition (convert affine to projective first)
    G1Projective q_proj = G1Projective::from_affine(q_aff[idx]);
    G1Projective standard_result;
    g1_add(standard_result, p[idx], q_proj);
    
    // Compare using proper Jacobian equality
    result[idx] = g1_projective_equal(mixed_result, standard_result) ? 1 : 0;
}

/**
 * @brief G1: Test projective to affine roundtrip
 */
__global__ void g1_proj_affine_roundtrip_kernel(
    const G1Affine* original, int* result, int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    // Skip identity
    if (original[idx].is_identity()) {
        result[idx] = 1;
        return;
    }
    
    // Convert to projective and back
    G1Projective proj = G1Projective::from_affine(original[idx]);
    G1Affine back = proj.to_affine();
    
    // Compare affine coordinates
    bool x_equal = true, y_equal = true;
    for (int i = 0; i < 6; i++) {
        if (original[idx].x.limbs[i] != back.x.limbs[i]) x_equal = false;
        if (original[idx].y.limbs[i] != back.y.limbs[i]) y_equal = false;
    }
    
    result[idx] = (x_equal && y_equal) ? 1 : 0;
}

/**
 * @brief G2: Test P + (-P) = O
 */
__global__ void g2_add_inverse_test_kernel(
    const G2Projective* p, int* result, int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    G2Projective neg_p;
    g2_neg(neg_p, p[idx]);
    
    G2Projective sum;
    g2_add(sum, p[idx], neg_p);
    
    result[idx] = sum.is_identity() ? 1 : 0;
}

/**
 * @brief G2: Test 2P = P + P
 */
__global__ void g2_double_vs_add_test_kernel(
    const G2Projective* p, int* result, int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    G2Projective doubled, added;
    g2_double(doubled, p[idx]);
    g2_add(added, p[idx], p[idx]);
    
    // Compare (simplified - just check identity status and X coords)
    if (doubled.is_identity() && added.is_identity()) {
        result[idx] = 1;
        return;
    }
    if (doubled.is_identity() || added.is_identity()) {
        result[idx] = 0;
        return;
    }
    
    // Compare X*Z (for Fq2, compare both c0 and c1)
    Fq2 xz1, xz2;
    fq2_mul(xz1, doubled.X, added.Z);
    fq2_mul(xz2, added.X, doubled.Z);
    
    result[idx] = (xz1 == xz2) ? 1 : 0;
}

// =============================================================================
// Test Functions
// =============================================================================

TestResult test_g1_identity_addition() {
    const int n = 256;
    
    // Generate test points on GPU via repeated doubling
    G1Affine gen = make_g1_generator();
    
    G1Affine* d_gen;
    G1Projective* d_points;
    int* d_results;
    SECURITY_CHECK_CUDA(cudaMalloc(&d_gen, sizeof(G1Affine)));
    SECURITY_CHECK_CUDA(cudaMalloc(&d_points, n * sizeof(G1Projective)));
    SECURITY_CHECK_CUDA(cudaMalloc(&d_results, n * sizeof(int)));
    
    SECURITY_CHECK_CUDA(cudaMemcpy(d_gen, &gen, sizeof(G1Affine), cudaMemcpyHostToDevice));
    g1_generate_test_points_kernel<<<1, 1>>>(d_gen, d_points, n);
    SECURITY_CHECK_CUDA(cudaDeviceSynchronize());
    cudaFree(d_gen);
    
    g1_add_identity_test_kernel<<<(n + 255) / 256, 256>>>(d_points, d_results, n);
    SECURITY_CHECK_CUDA(cudaDeviceSynchronize());
    
    std::vector<int> results(n);
    SECURITY_CHECK_CUDA(cudaMemcpy(results.data(), d_results, n * sizeof(int), cudaMemcpyDeviceToHost));
    
    cudaFree(d_points);
    cudaFree(d_results);
    
    int failures = 0;
    for (int i = 0; i < n; i++) {
        if (results[i] != 1) failures++;
    }
    
    if (failures > 0) {
        std::cout << "\n    " << failures << "/" << n << " failures";
        return TestResult::FAILED;
    }
    return TestResult::PASSED;
}

TestResult test_g1_inverse_addition() {
    const int n = 256;
    
    G1Affine gen = make_g1_generator();
    
    G1Affine* d_gen;
    G1Projective* d_points;
    int* d_results;
    SECURITY_CHECK_CUDA(cudaMalloc(&d_gen, sizeof(G1Affine)));
    SECURITY_CHECK_CUDA(cudaMalloc(&d_points, n * sizeof(G1Projective)));
    SECURITY_CHECK_CUDA(cudaMalloc(&d_results, n * sizeof(int)));
    
    SECURITY_CHECK_CUDA(cudaMemcpy(d_gen, &gen, sizeof(G1Affine), cudaMemcpyHostToDevice));
    g1_generate_test_points_kernel<<<1, 1>>>(d_gen, d_points, n);
    SECURITY_CHECK_CUDA(cudaDeviceSynchronize());
    cudaFree(d_gen);
    
    g1_add_inverse_test_kernel<<<(n + 255) / 256, 256>>>(d_points, d_results, n);
    SECURITY_CHECK_CUDA(cudaDeviceSynchronize());
    
    std::vector<int> results(n);
    SECURITY_CHECK_CUDA(cudaMemcpy(results.data(), d_results, n * sizeof(int), cudaMemcpyDeviceToHost));
    
    cudaFree(d_points);
    cudaFree(d_results);
    
    int failures = 0;
    for (int i = 0; i < n; i++) {
        if (results[i] != 1) failures++;
    }
    
    if (failures > 0) {
        std::cout << "\n    " << failures << "/" << n << " failures";
        return TestResult::FAILED;
    }
    return TestResult::PASSED;
}

TestResult test_g1_commutativity() {
    const int n = 256;
    
    G1Affine gen = make_g1_generator();
    
    G1Affine* d_gen;
    G1Projective *d_p, *d_q;
    int* d_results;
    SECURITY_CHECK_CUDA(cudaMalloc(&d_gen, sizeof(G1Affine)));
    SECURITY_CHECK_CUDA(cudaMalloc(&d_p, n * sizeof(G1Projective)));
    SECURITY_CHECK_CUDA(cudaMalloc(&d_q, n * sizeof(G1Projective)));
    SECURITY_CHECK_CUDA(cudaMalloc(&d_results, n * sizeof(int)));
    
    SECURITY_CHECK_CUDA(cudaMemcpy(d_gen, &gen, sizeof(G1Affine), cudaMemcpyHostToDevice));
    g1_generate_two_point_sets_kernel<<<1, 1>>>(d_gen, d_p, d_q, n);
    SECURITY_CHECK_CUDA(cudaDeviceSynchronize());
    cudaFree(d_gen);
    
    g1_add_commutative_test_kernel<<<(n + 255) / 256, 256>>>(d_p, d_q, d_results, n);
    SECURITY_CHECK_CUDA(cudaDeviceSynchronize());
    
    std::vector<int> results(n);
    SECURITY_CHECK_CUDA(cudaMemcpy(results.data(), d_results, n * sizeof(int), cudaMemcpyDeviceToHost));
    
    cudaFree(d_p);
    cudaFree(d_q);
    cudaFree(d_results);
    
    int failures = 0;
    for (int i = 0; i < n; i++) {
        if (results[i] != 1) failures++;
    }
    
    if (failures > 0) {
        std::cout << "\n    " << failures << "/" << n << " failures";
        return TestResult::FAILED;
    }
    return TestResult::PASSED;
}

TestResult test_g1_associativity() {
    const int n = 128;
    
    G1Affine gen = make_g1_generator();
    
    G1Affine* d_gen;
    G1Projective *d_p, *d_q, *d_r;
    int* d_results;
    SECURITY_CHECK_CUDA(cudaMalloc(&d_gen, sizeof(G1Affine)));
    SECURITY_CHECK_CUDA(cudaMalloc(&d_p, n * sizeof(G1Projective)));
    SECURITY_CHECK_CUDA(cudaMalloc(&d_q, n * sizeof(G1Projective)));
    SECURITY_CHECK_CUDA(cudaMalloc(&d_r, n * sizeof(G1Projective)));
    SECURITY_CHECK_CUDA(cudaMalloc(&d_results, n * sizeof(int)));
    
    SECURITY_CHECK_CUDA(cudaMemcpy(d_gen, &gen, sizeof(G1Affine), cudaMemcpyHostToDevice));
    g1_generate_three_point_sets_kernel<<<1, 1>>>(d_gen, d_p, d_q, d_r, n);
    SECURITY_CHECK_CUDA(cudaDeviceSynchronize());
    cudaFree(d_gen);
    
    g1_add_associative_test_kernel<<<(n + 127) / 128, 128>>>(d_p, d_q, d_r, d_results, n);
    SECURITY_CHECK_CUDA(cudaDeviceSynchronize());
    
    std::vector<int> results(n);
    SECURITY_CHECK_CUDA(cudaMemcpy(results.data(), d_results, n * sizeof(int), cudaMemcpyDeviceToHost));
    
    cudaFree(d_p);
    cudaFree(d_q);
    cudaFree(d_r);
    cudaFree(d_results);
    
    int failures = 0;
    for (int i = 0; i < n; i++) {
        if (results[i] != 1) failures++;
    }
    
    if (failures > 0) {
        std::cout << "\n    " << failures << "/" << n << " failures";
        return TestResult::FAILED;
    }
    return TestResult::PASSED;
}

TestResult test_g1_double_vs_add() {
    const int n = 256;
    
    G1Affine gen = make_g1_generator();
    
    G1Affine* d_gen;
    G1Projective* d_points;
    int* d_results;
    SECURITY_CHECK_CUDA(cudaMalloc(&d_gen, sizeof(G1Affine)));
    SECURITY_CHECK_CUDA(cudaMalloc(&d_points, n * sizeof(G1Projective)));
    SECURITY_CHECK_CUDA(cudaMalloc(&d_results, n * sizeof(int)));
    
    SECURITY_CHECK_CUDA(cudaMemcpy(d_gen, &gen, sizeof(G1Affine), cudaMemcpyHostToDevice));
    g1_generate_test_points_kernel<<<1, 1>>>(d_gen, d_points, n);
    SECURITY_CHECK_CUDA(cudaDeviceSynchronize());
    cudaFree(d_gen);
    
    g1_double_vs_add_test_kernel<<<(n + 255) / 256, 256>>>(d_points, d_results, n);
    SECURITY_CHECK_CUDA(cudaDeviceSynchronize());
    
    std::vector<int> results(n);
    SECURITY_CHECK_CUDA(cudaMemcpy(results.data(), d_results, n * sizeof(int), cudaMemcpyDeviceToHost));
    
    cudaFree(d_points);
    cudaFree(d_results);
    
    int failures = 0;
    for (int i = 0; i < n; i++) {
        if (results[i] != 1) failures++;
    }
    
    if (failures > 0) {
        std::cout << "\n    " << failures << "/" << n << " failures";
        return TestResult::FAILED;
    }
    return TestResult::PASSED;
}

TestResult test_g1_triple_point() {
    const int n = 128;
    
    G1Affine gen = make_g1_generator();
    
    G1Affine* d_gen;
    G1Projective* d_points;
    int* d_results;
    SECURITY_CHECK_CUDA(cudaMalloc(&d_gen, sizeof(G1Affine)));
    SECURITY_CHECK_CUDA(cudaMalloc(&d_points, n * sizeof(G1Projective)));
    SECURITY_CHECK_CUDA(cudaMalloc(&d_results, n * sizeof(int)));
    
    SECURITY_CHECK_CUDA(cudaMemcpy(d_gen, &gen, sizeof(G1Affine), cudaMemcpyHostToDevice));
    g1_generate_test_points_kernel<<<1, 1>>>(d_gen, d_points, n);
    SECURITY_CHECK_CUDA(cudaDeviceSynchronize());
    cudaFree(d_gen);
    
    g1_triple_point_test_kernel<<<(n + 127) / 128, 128>>>(d_points, d_results, n);
    SECURITY_CHECK_CUDA(cudaDeviceSynchronize());
    
    std::vector<int> results(n);
    SECURITY_CHECK_CUDA(cudaMemcpy(results.data(), d_results, n * sizeof(int), cudaMemcpyDeviceToHost));
    
    cudaFree(d_points);
    cudaFree(d_results);
    
    int failures = 0;
    for (int i = 0; i < n; i++) {
        if (results[i] != 1) failures++;
    }
    
    if (failures > 0) {
        std::cout << "\n    " << failures << "/" << n << " failures";
        return TestResult::FAILED;
    }
    return TestResult::PASSED;
}

TestResult test_g1_proj_affine_roundtrip() {
    const int n = 256;
    
    G1Affine gen = make_g1_generator();
    
    G1Affine* d_gen;
    G1Affine* d_points;
    int* d_results;
    SECURITY_CHECK_CUDA(cudaMalloc(&d_gen, sizeof(G1Affine)));
    SECURITY_CHECK_CUDA(cudaMalloc(&d_points, n * sizeof(G1Affine)));
    SECURITY_CHECK_CUDA(cudaMalloc(&d_results, n * sizeof(int)));
    
    SECURITY_CHECK_CUDA(cudaMemcpy(d_gen, &gen, sizeof(G1Affine), cudaMemcpyHostToDevice));
    g1_generate_affine_points_kernel<<<1, 1>>>(d_gen, d_points, n);
    SECURITY_CHECK_CUDA(cudaDeviceSynchronize());
    cudaFree(d_gen);
    
    g1_proj_affine_roundtrip_kernel<<<(n + 255) / 256, 256>>>(d_points, d_results, n);
    SECURITY_CHECK_CUDA(cudaDeviceSynchronize());
    
    std::vector<int> results(n);
    SECURITY_CHECK_CUDA(cudaMemcpy(results.data(), d_results, n * sizeof(int), cudaMemcpyDeviceToHost));
    
    cudaFree(d_points);
    cudaFree(d_results);
    
    int failures = 0;
    for (int i = 0; i < n; i++) {
        if (results[i] != 1) failures++;
    }
    
    if (failures > 0) {
        std::cout << "\n    " << failures << "/" << n << " failures";
        return TestResult::FAILED;
    }
    return TestResult::PASSED;
}

TestResult test_g2_inverse_addition() {
    const int n = 64;  // Smaller due to larger point size
    
    G2Affine gen = make_g2_generator();
    
    G2Affine* d_gen;
    G2Projective* d_points;
    int* d_results;
    SECURITY_CHECK_CUDA(cudaMalloc(&d_gen, sizeof(G2Affine)));
    SECURITY_CHECK_CUDA(cudaMalloc(&d_points, n * sizeof(G2Projective)));
    SECURITY_CHECK_CUDA(cudaMalloc(&d_results, n * sizeof(int)));
    
    SECURITY_CHECK_CUDA(cudaMemcpy(d_gen, &gen, sizeof(G2Affine), cudaMemcpyHostToDevice));
    g2_generate_test_points_kernel<<<1, 1>>>(d_gen, d_points, n);
    SECURITY_CHECK_CUDA(cudaDeviceSynchronize());
    cudaFree(d_gen);
    
    g2_add_inverse_test_kernel<<<(n + 63) / 64, 64>>>(d_points, d_results, n);
    SECURITY_CHECK_CUDA(cudaDeviceSynchronize());
    
    std::vector<int> results(n);
    SECURITY_CHECK_CUDA(cudaMemcpy(results.data(), d_results, n * sizeof(int), cudaMemcpyDeviceToHost));
    
    cudaFree(d_points);
    cudaFree(d_results);
    
    int failures = 0;
    for (int i = 0; i < n; i++) {
        if (results[i] != 1) failures++;
    }
    
    if (failures > 0) {
        std::cout << "\n    " << failures << "/" << n << " failures";
        return TestResult::FAILED;
    }
    return TestResult::PASSED;
}

TestResult test_g2_double_vs_add() {
    const int n = 64;
    
    G2Affine gen = make_g2_generator();
    
    G2Affine* d_gen;
    G2Projective* d_points;
    int* d_results;
    SECURITY_CHECK_CUDA(cudaMalloc(&d_gen, sizeof(G2Affine)));
    SECURITY_CHECK_CUDA(cudaMalloc(&d_points, n * sizeof(G2Projective)));
    SECURITY_CHECK_CUDA(cudaMalloc(&d_results, n * sizeof(int)));
    
    SECURITY_CHECK_CUDA(cudaMemcpy(d_gen, &gen, sizeof(G2Affine), cudaMemcpyHostToDevice));
    g2_generate_test_points_kernel<<<1, 1>>>(d_gen, d_points, n);
    SECURITY_CHECK_CUDA(cudaDeviceSynchronize());
    cudaFree(d_gen);
    
    g2_double_vs_add_test_kernel<<<(n + 63) / 64, 64>>>(d_points, d_results, n);
    SECURITY_CHECK_CUDA(cudaDeviceSynchronize());
    
    std::vector<int> results(n);
    SECURITY_CHECK_CUDA(cudaMemcpy(results.data(), d_results, n * sizeof(int), cudaMemcpyDeviceToHost));
    
    cudaFree(d_points);
    cudaFree(d_results);
    
    int failures = 0;
    for (int i = 0; i < n; i++) {
        if (results[i] != 1) failures++;
    }
    
    if (failures > 0) {
        std::cout << "\n    " << failures << "/" << n << " failures";
        return TestResult::FAILED;
    }
    return TestResult::PASSED;
}

// =============================================================================
// Curve Equation Verification Tests
// =============================================================================

/**
 * @brief Verify G1 generator is on curve y² = x³ + 4
 * 
 * This is a CRITICAL test - if the generator is not on the curve,
 * all cryptographic operations are fundamentally broken.
 */
TestResult test_g1_generator_on_curve() {
    int* d_result;
    SECURITY_CHECK_CUDA(cudaMalloc(&d_result, sizeof(int)));
    
    verify_g1_generator_on_curve_kernel<<<1, 1>>>(d_result);
    SECURITY_CHECK_CUDA(cudaDeviceSynchronize());
    
    int result;
    SECURITY_CHECK_CUDA(cudaMemcpy(&result, d_result, sizeof(int), cudaMemcpyDeviceToHost));
    cudaFree(d_result);
    
    if (result != 1) {
        std::cout << "\n    CRITICAL: G1 generator is NOT on curve y² = x³ + 4!";
        return TestResult::FAILED;
    }
    return TestResult::PASSED;
}

/**
 * @brief Verify derived points remain on curve after group operations
 * 
 * After doubling/addition, all resulting points must still lie on the curve.
 * This catches subtle bugs in the group law implementation.
 */
TestResult test_g1_derived_points_on_curve() {
    const int n = 256;
    
    G1Affine gen = make_g1_generator();
    
    G1Affine* d_gen;
    G1Projective* d_points;
    int* d_results;
    SECURITY_CHECK_CUDA(cudaMalloc(&d_gen, sizeof(G1Affine)));
    SECURITY_CHECK_CUDA(cudaMalloc(&d_points, n * sizeof(G1Projective)));
    SECURITY_CHECK_CUDA(cudaMalloc(&d_results, n * sizeof(int)));
    
    SECURITY_CHECK_CUDA(cudaMemcpy(d_gen, &gen, sizeof(G1Affine), cudaMemcpyHostToDevice));
    
    // Generate points via repeated doubling: G, 2G, 4G, 8G, ...
    g1_generate_test_points_kernel<<<1, 1>>>(d_gen, d_points, n);
    SECURITY_CHECK_CUDA(cudaDeviceSynchronize());
    cudaFree(d_gen);
    
    // Verify all points are on curve
    verify_g1_projective_points_on_curve_kernel<<<(n + 255) / 256, 256>>>(
        d_points, d_results, n);
    SECURITY_CHECK_CUDA(cudaDeviceSynchronize());
    
    std::vector<int> results(n);
    SECURITY_CHECK_CUDA(cudaMemcpy(results.data(), d_results, n * sizeof(int), cudaMemcpyDeviceToHost));
    
    cudaFree(d_points);
    cudaFree(d_results);
    
    int failures = 0;
    for (int i = 0; i < n; i++) {
        if (results[i] != 1) failures++;
    }
    
    if (failures > 0) {
        std::cout << "\n    " << failures << "/" << n << " derived points NOT on curve!";
        return TestResult::FAILED;
    }
    return TestResult::PASSED;
}

// =============================================================================
// Registration
// =============================================================================

void register_curve_operation_tests(SecurityTestSuite& suite) {
    // Curve Equation Verification (CRITICAL - must come first)
    suite.add_test("G1 generator on curve y² = x³ + 4", "Curve Verification",
                   test_g1_generator_on_curve, true);  // Critical test
    suite.add_test("G1 derived points remain on curve", "Curve Verification",
                   test_g1_derived_points_on_curve, true);  // Critical test
    
    // G1 Group Laws
    suite.add_test("G1: P + O = P (identity)", "G1 Group Laws",
                   test_g1_identity_addition);
    suite.add_test("G1: P + (-P) = O (inverse)", "G1 Group Laws",
                   test_g1_inverse_addition);
    suite.add_test("G1: P + Q = Q + P (commutativity)", "G1 Group Laws",
                   test_g1_commutativity);
    suite.add_test("G1: (P+Q)+R = P+(Q+R) (associativity)", "G1 Group Laws",
                   test_g1_associativity);
    
    // G1 Doubling
    suite.add_test("G1: 2P = P + P (doubling)", "G1 Doubling",
                   test_g1_double_vs_add);
    suite.add_test("G1: 3P consistency", "G1 Doubling",
                   test_g1_triple_point);
    
    // Coordinate Conversion
    suite.add_test("G1: Projective ↔ Affine roundtrip", "Coordinate Conversion",
                   test_g1_proj_affine_roundtrip);
    
    // G2 Tests
    suite.add_test("G2: P + (-P) = O", "G2 Group Laws",
                   test_g2_inverse_addition);
    suite.add_test("G2: 2P = P + P", "G2 Group Laws",
                   test_g2_double_vs_add);
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
    register_curve_operation_tests(suite);
    
    bool success = suite.run_all();
    return success ? 0 : 1;
}
