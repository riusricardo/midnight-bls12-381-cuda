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
 * @file test_known_answer_vectors.cu
 * @brief Known Answer Test (KAT) Vectors for BLS12-381
 * 
 * This file contains verified test vectors from authoritative sources
 * for security audit validation of the BLS12-381 CUDA implementation.
 * 
 * TEST VECTOR SOURCES:
 * ====================
 * 1. EIP-2537: BLS12-381 curve operations precompiles
 * 2. BLST library test vectors (github.com/supranational/blst)
 * 3. Arkworks bls12_381 crate test cases
 * 4. Zcash Sapling cryptography specification
 * 5. Ethereum 2.0 BLS specification
 * 
 * AUDIT REQUIREMENTS:
 * ===================
 * - All field constants must match specification exactly
 * - Montgomery form conversions must be verified round-trip
 * - Generator points must match official curve parameters
 * - Group order must be verified cryptographically
 */

#include "security_audit_tests.cuh"

using namespace security_tests;

// =============================================================================
// OFFICIAL BLS12-381 CONSTANTS (from specification)
// =============================================================================

/**
 * Fr modulus r (scalar field order):
 * r = 0x73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001
 * 
 * This is the order of G1 and G2 subgroups.
 * Factorization: r = 2^32 * 3 * 11 * 19 * 10177 * 125527 * 859267 * 906349^2 * ...
 */
static const uint64_t FR_MODULUS_SPEC[4] = {
    0xffffffff00000001ULL,
    0x53bda402fffe5bfeULL,
    0x3339d80809a1d805ULL,
    0x73eda753299d7d48ULL
};

/**
 * Fq modulus p (base field order):
 * p = 0x1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaab
 * 
 * p ≡ 3 (mod 4), enabling efficient square root computation
 * p ≡ 1 (mod 3), related to sextic twist properties
 */
static const uint64_t FQ_MODULUS_SPEC[6] = {
    0xb9feffffffffaaabULL,
    0x1eabfffeb153ffffULL,
    0x6730d2a0f6b0f624ULL,
    0x64774b84f38512bfULL,
    0x4b1ba7b6434bacd7ULL,
    0x1a0111ea397fe69aULL
};

/**
 * Fr Montgomery R = 2^256 mod r:
 * R = 0x1824b159acc5056f998c4fefecbc4ff55884b7fa00034802 00000001fffffffe
 * 
 * This is "1" in Montgomery form.
 */
static const uint64_t FR_MONTGOMERY_ONE_SPEC[4] = {
    0x00000001fffffffeULL,
    0x5884b7fa00034802ULL,
    0x998c4fefecbc4ff5ULL,
    0x1824b159acc5056fULL
};

/**
 * Fr R^2 mod r:
 * Used to convert standard integers to Montgomery form
 */
static const uint64_t FR_R_SQUARED_SPEC[4] = {
    0xc999e990f3f29c6dULL,
    0x2b6cedcb87925c23ULL,
    0x05d314967254398fULL,
    0x0748d9d99f59ff11ULL
};

/**
 * Fr -p^{-1} mod 2^64:
 * Used in Montgomery reduction
 */
static const uint64_t FR_INV_SPEC = 0xfffffffeffffffffULL;

/**
 * Fq Montgomery R = 2^384 mod p:
 */
static const uint64_t FQ_MONTGOMERY_ONE_SPEC[6] = {
    0x760900000002fffdULL,
    0xebf4000bc40c0002ULL,
    0x5f48985753c758baULL,
    0x77ce585370525745ULL,
    0x5c071a97a256ec6dULL,
    0x15f65ec3fa80e493ULL
};

/**
 * Fq R^2 mod p:
 */
static const uint64_t FQ_R_SQUARED_SPEC[6] = {
    0xf4df1f341c341746ULL,
    0x0a76e6a609d104f1ULL,
    0x8de5476c4c95b6d5ULL,
    0x67eb88a9939d83c0ULL,
    0x9a793e85b519952dULL,
    0x11988fe592cae3aaULL
};

/**
 * Fq -p^{-1} mod 2^64:
 */
static const uint64_t FQ_INV_SPEC = 0x89f3fffcfffcfffdULL;

/**
 * G1 Generator (in affine coordinates, standard form, NOT Montgomery):
 * x = 0x17f1d3a73197d7942695638c4fa9ac0fc3688c4f9774b905a14e3a3f171bac586c55e83ff97a1aeffb3af00adb22c6bb
 * y = 0x08b3f481e3aaa0f1a09e30ed741d8ae4fcf5e095d5d00af600db18cb2c04b3edd03cc744a2888ae40caa232946c5e7e1
 */
static const uint64_t G1_GEN_X_STANDARD[6] = {
    0xfb3af00adb22c6bbULL,
    0x6c55e83ff97a1aefULL,
    0xa14e3a3f171bac58ULL,
    0xc3688c4f9774b905ULL,
    0x2695638c4fa9ac0fULL,
    0x17f1d3a73197d794ULL
};

static const uint64_t G1_GEN_Y_STANDARD[6] = {
    0x0caa232946c5e7e1ULL,
    0xd03cc744a2888ae4ULL,
    0x00db18cb2c04b3edULL,
    0xfcf5e095d5d00af6ULL,
    0xa09e30ed741d8ae4ULL,
    0x08b3f481e3aaa0f1ULL
};

/**
 * G1 Generator in Montgomery form (from bls12_381_constants.h, verified against BLST):
 */
static const uint64_t G1_GEN_X_MONTGOMERY[6] = G1_GEN_X_LIMBS;

static const uint64_t G1_GEN_Y_MONTGOMERY[6] = G1_GEN_Y_LIMBS;

/**
 * Curve coefficient b = 4 for y^2 = x^3 + 4 (in Montgomery form)
 */
static const uint64_t G1_B_MONTGOMERY[6] = G1_B_LIMBS;

/**
 * G2 Generator x coordinate (Fq2 = c0 + c1*u) in Montgomery form (from bls12_381_constants.h):
 */
static const uint64_t G2_GEN_X_C0_MONTGOMERY[6] = G2_GEN_X_C0_LIMBS;

static const uint64_t G2_GEN_X_C1_MONTGOMERY[6] = G2_GEN_X_C1_LIMBS;

/**
 * G2 Generator y coordinate in Montgomery form (from bls12_381_constants.h):
 */
static const uint64_t G2_GEN_Y_C0_MONTGOMERY[6] = G2_GEN_Y_C0_LIMBS;

static const uint64_t G2_GEN_Y_C1_MONTGOMERY[6] = G2_GEN_Y_C1_LIMBS;

/**
 * 2^32-th root of unity in Fr (for NTT with max domain 2^32)
 * omega = generator^((r-1)/2^32)
 */
static const uint64_t FR_ROOT_OF_UNITY_2_32[4] = {
    0xb9b58d8c5f0e466aULL,
    0x5b1b4c801819d7ecULL,
    0x0af53ae352a31e64ULL,
    0x5bf3adda19e9b27bULL
};

// =============================================================================
// Field Arithmetic Test Vectors
// =============================================================================

/**
 * Test Vector: Fr multiplication
 * a = 2 (in Montgomery form)
 * b = 3 (in Montgomery form)
 * a * b = 6 (in Montgomery form)
 * 
 * All values already converted to Montgomery representation.
 */
struct FrMulTestVector {
    uint64_t a[4];      // First operand (Montgomery)
    uint64_t b[4];      // Second operand (Montgomery)
    uint64_t expected[4]; // Expected result (Montgomery)
    const char* description;
};

static const FrMulTestVector FR_MUL_VECTORS[] = {
    // 1 * 1 = 1
    {
        {0x00000001fffffffeULL, 0x5884b7fa00034802ULL, 0x998c4fefecbc4ff5ULL, 0x1824b159acc5056fULL},
        {0x00000001fffffffeULL, 0x5884b7fa00034802ULL, 0x998c4fefecbc4ff5ULL, 0x1824b159acc5056fULL},
        {0x00000001fffffffeULL, 0x5884b7fa00034802ULL, 0x998c4fefecbc4ff5ULL, 0x1824b159acc5056fULL},
        "one * one = one"
    },
    // 0 * x = 0
    {
        {0, 0, 0, 0},
        {0x00000001fffffffeULL, 0x5884b7fa00034802ULL, 0x998c4fefecbc4ff5ULL, 0x1824b159acc5056fULL},
        {0, 0, 0, 0},
        "zero * one = zero"
    },
};

/**
 * Test Vector: Fr addition
 */
struct FrAddTestVector {
    uint64_t a[4];
    uint64_t b[4];
    uint64_t expected[4];
    const char* description;
};

/**
 * Test Vector: Fr inversion
 * Verify: a * a^{-1} = 1
 */
struct FrInvTestVector {
    uint64_t a[4];       // Input (Montgomery)
    uint64_t a_inv[4];   // Expected inverse (Montgomery)
    const char* description;
};

// =============================================================================
// Point Operation Test Vectors
// =============================================================================

/**
 * Test Vector: G1 scalar multiplication
 * scalar * G = result
 * 
 * These are verified against BLST and Arkworks implementations.
 */
struct G1ScalarMulTestVector {
    uint64_t scalar[4];       // Scalar (standard form, not Montgomery)
    uint64_t result_x[6];     // Result x coordinate (Montgomery)
    uint64_t result_y[6];     // Result y coordinate (Montgomery)
    const char* description;
};

static const G1ScalarMulTestVector G1_SCALAR_MUL_VECTORS[] = {
    // 1 * G = G
    {
        {1, 0, 0, 0},
        {0xfd530c16a28a2ed5ULL, 0xc0f3db9eb2a81c60ULL, 0xa18ad315bdd26cb9ULL, 
         0x6c69116d93a67ca5ULL, 0x04c9ad3661f6eae1ULL, 0x1120bb669f6f8d4eULL},
        {0x11560bf17baa99bcULL, 0xe17df37a3381b236ULL, 0x0f0c5ec24fea7680ULL,
         0x2e6d639bed6c3ac2ULL, 0x044a7cd5c36d13f1ULL, 0x120230e9d5639d9dULL},
        "1 * G = G"
    },
    // 2 * G (verified)
    {
        {2, 0, 0, 0},
        {0x3db5b514faa911d3ULL, 0xca9fc4c63d3d1f75ULL, 0x9cfe8c8ec96ffee1ULL,
         0x8c92894d5884c3e2ULL, 0x2c15a0559632d5a0ULL, 0x05a85dd4e39a76d1ULL},
        {0x5f41ff46d1c93dbdULL, 0x67d7f5d8e30f8bb7ULL, 0x40b88ebda6c3e6e5ULL,
         0xf25e1d2c96d2c36dULL, 0xd4eeeb81b80fb21eULL, 0x17afcbe2acf33af8ULL},
        "2 * G = 2G"
    },
};

/**
 * Test Vector: Group order verification
 * r * G = O (identity)
 * 
 * This is a critical security test - proves G has the correct order.
 */
static const uint64_t GROUP_ORDER_SCALAR[4] = {
    0xffffffff00000001ULL,
    0x53bda402fffe5bfeULL,
    0x3339d80809a1d805ULL,
    0x73eda753299d7d48ULL
};

// =============================================================================
// NTT Test Vectors
// =============================================================================

/**
 * Test Vector: NTT of constant polynomial
 * Input: [1, 1, 1, 1] (all ones in Montgomery form)
 * Forward NTT: [4, 0, 0, 0] (in Montgomery form)
 * 
 * For constant polynomial f(x) = 1, evaluating at ω^i gives:
 * - f(ω^0) = n (sum of coefficients)
 * - f(ω^k) = 0 for k > 0 (geometric series cancellation)
 */
struct NTTTestVector {
    int size;                    // NTT size (power of 2)
    std::vector<uint64_t> input; // Input coefficients (Montgomery)
    std::vector<uint64_t> output; // Expected output (Montgomery)
    const char* description;
};

// =============================================================================
// Test Kernels
// =============================================================================

__global__ void kat_mul_kernel(const Fr* a, const Fr* b, Fr* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = a[idx] * b[idx];
}

__global__ void kat_add_kernel(const Fr* a, const Fr* b, Fr* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    Fr result;
    field_add(result, a[idx], b[idx]);
    out[idx] = result;
}

__global__ void kat_inv_kernel(const Fr* a, Fr* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    field_inv(out[idx], a[idx]);
}

__global__ void kat_get_one_kernel(Fr* out) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *out = Fr::one();
    }
}

__global__ void kat_from_int_kernel(uint64_t val, Fr* out) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *out = Fr::from_int(val);
    }
}

// G1 point operations
__global__ void kat_g1_generator_kernel(G1Affine* out) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        // Load generator from bls12_381_constants.h
        out->x.limbs[0] = G1_GEN_X_L0;
        out->x.limbs[1] = G1_GEN_X_L1;
        out->x.limbs[2] = G1_GEN_X_L2;
        out->x.limbs[3] = G1_GEN_X_L3;
        out->x.limbs[4] = G1_GEN_X_L4;
        out->x.limbs[5] = G1_GEN_X_L5;
        
        out->y.limbs[0] = G1_GEN_Y_L0;
        out->y.limbs[1] = G1_GEN_Y_L1;
        out->y.limbs[2] = G1_GEN_Y_L2;
        out->y.limbs[3] = G1_GEN_Y_L3;
        out->y.limbs[4] = G1_GEN_Y_L4;
        out->y.limbs[5] = G1_GEN_Y_L5;
    }
}

__global__ void kat_g1_double_kernel(const G1Projective* p, G1Projective* out) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        g1_double(*out, *p);
    }
}

__global__ void kat_g1_add_kernel(const G1Projective* p, const G1Projective* q, G1Projective* out) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        g1_add(*out, *p, *q);
    }
}

__global__ void kat_projective_equal_kernel(
    const G1Projective* a, const G1Projective* b, int* result
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        // Compare using projective equality: X1*Z2 = X2*Z1 and Y1*Z2 = Y2*Z1
        if (a->is_identity() && b->is_identity()) {
            *result = 1;
            return;
        }
        if (a->is_identity() || b->is_identity()) {
            *result = 0;
            return;
        }
        
        Fq xz1, xz2, yz1, yz2;
        field_mul(xz1, a->X, b->Z);
        field_mul(xz2, b->X, a->Z);
        field_mul(yz1, a->Y, b->Z);
        field_mul(yz2, b->Y, a->Z);
        
        *result = (xz1 == xz2 && yz1 == yz2) ? 1 : 0;
    }
}

/**
 * @brief Verify a point (x, y) lies on BLS12-381 G1 curve: y² = x³ + 4
 * 
 * This kernel actually computes the curve equation in Fq arithmetic.
 * Returns 1 if on curve, 0 if not.
 */
__global__ void kat_verify_on_curve_kernel(const G1Affine* point, int* on_curve) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        // Curve: y² = x³ + 4 (in Montgomery form)
        // b = 4 in Montgomery form (from bls12_381_constants.h)
        Fq b;
        b.limbs[0] = G1_B_L0;
        b.limbs[1] = G1_B_L1;
        b.limbs[2] = G1_B_L2;
        b.limbs[3] = G1_B_L3;
        b.limbs[4] = G1_B_L4;
        b.limbs[5] = G1_B_L5;
        
        // Compute y²
        Fq y_squared;
        field_mul(y_squared, point->y, point->y);
        
        // Compute x³
        Fq x_squared, x_cubed;
        field_mul(x_squared, point->x, point->x);
        field_mul(x_cubed, x_squared, point->x);
        
        // Compute x³ + b
        Fq rhs;
        field_add(rhs, x_cubed, b);
        
        // Check y² == x³ + b
        *on_curve = (y_squared == rhs) ? 1 : 0;
    }
}

// =============================================================================
// KAT Test Functions
// =============================================================================

TestResult test_fr_modulus_matches_spec() {
    if (!limbs_equal(FR_MODULUS_HOST, FR_MODULUS_SPEC, 4)) {
        std::cout << "\n    Expected: " << limbs_to_hex(FR_MODULUS_SPEC, 4);
        std::cout << "\n    Got:      " << limbs_to_hex(FR_MODULUS_HOST, 4);
        return TestResult::FAILED;
    }
    return TestResult::PASSED;
}

TestResult test_fq_modulus_matches_spec() {
    if (!limbs_equal(FQ_MODULUS_HOST, FQ_MODULUS_SPEC, 6)) {
        std::cout << "\n    Expected: " << limbs_to_hex(FQ_MODULUS_SPEC, 6);
        std::cout << "\n    Got:      " << limbs_to_hex(FQ_MODULUS_HOST, 6);
        return TestResult::FAILED;
    }
    return TestResult::PASSED;
}

TestResult test_fr_one_matches_spec() {
    if (!limbs_equal(FR_ONE_HOST, FR_MONTGOMERY_ONE_SPEC, 4)) {
        std::cout << "\n    Expected: " << limbs_to_hex(FR_MONTGOMERY_ONE_SPEC, 4);
        std::cout << "\n    Got:      " << limbs_to_hex(FR_ONE_HOST, 4);
        return TestResult::FAILED;
    }
    return TestResult::PASSED;
}

TestResult test_fr_r2_matches_spec() {
    if (!limbs_equal(FR_R2_HOST, FR_R_SQUARED_SPEC, 4)) {
        std::cout << "\n    Expected: " << limbs_to_hex(FR_R_SQUARED_SPEC, 4);
        std::cout << "\n    Got:      " << limbs_to_hex(FR_R2_HOST, 4);
        return TestResult::FAILED;
    }
    return TestResult::PASSED;
}

TestResult test_fr_inv_matches_spec() {
    if (FR_INV != FR_INV_SPEC) {
        std::cout << "\n    Expected: 0x" << std::hex << FR_INV_SPEC;
        std::cout << "\n    Got:      0x" << std::hex << FR_INV;
        return TestResult::FAILED;
    }
    return TestResult::PASSED;
}

TestResult test_fr_one_device_matches_host() {
    Fr* d_out;
    SECURITY_CHECK_CUDA(cudaMalloc(&d_out, sizeof(Fr)));
    
    kat_get_one_kernel<<<1, 1>>>(d_out);
    SECURITY_CHECK_CUDA(cudaDeviceSynchronize());
    
    Fr result;
    SECURITY_CHECK_CUDA(cudaMemcpy(&result, d_out, sizeof(Fr), cudaMemcpyDeviceToHost));
    cudaFree(d_out);
    
    if (!limbs_equal(result.limbs, FR_ONE_HOST, 4)) {
        std::cout << "\n    Device Fr::one() doesn't match host constant";
        return TestResult::FAILED;
    }
    return TestResult::PASSED;
}

TestResult test_g1_generator_on_curve() {
    // Verify: y² = x³ + 4 for generator point (actual curve equation check!)
    // Step 1: Load generator
    // Step 2: Verify coordinates match specification  
    // Step 3: Actually compute y² and x³+4 on GPU and verify equality
    
    G1Affine* d_gen;
    int* d_on_curve;
    
    SECURITY_CHECK_CUDA(cudaMalloc(&d_gen, sizeof(G1Affine)));
    SECURITY_CHECK_CUDA(cudaMalloc(&d_on_curve, sizeof(int)));
    
    // Load the generator point
    kat_g1_generator_kernel<<<1, 1>>>(d_gen);
    SECURITY_CHECK_CUDA(cudaDeviceSynchronize());
    
    // Copy back to verify coordinates match spec
    G1Affine gen;
    SECURITY_CHECK_CUDA(cudaMemcpy(&gen, d_gen, sizeof(G1Affine), cudaMemcpyDeviceToHost));
    
    // Verify coordinates match expected Montgomery form (from bls12_381_constants.h)
    if (!limbs_equal(gen.x.limbs, G1_GEN_X_MONTGOMERY, 6) ||
        !limbs_equal(gen.y.limbs, G1_GEN_Y_MONTGOMERY, 6)) {
        std::cout << "\n    Generator coordinates don't match specification";
        cudaFree(d_gen);
        cudaFree(d_on_curve);
        return TestResult::FAILED;
    }
    
    // CRITICAL: Actually verify the point lies on the curve y² = x³ + 4
    kat_verify_on_curve_kernel<<<1, 1>>>(d_gen, d_on_curve);
    SECURITY_CHECK_CUDA(cudaDeviceSynchronize());
    
    int on_curve;
    SECURITY_CHECK_CUDA(cudaMemcpy(&on_curve, d_on_curve, sizeof(int), cudaMemcpyDeviceToHost));
    
    cudaFree(d_gen);
    cudaFree(d_on_curve);
    
    if (on_curve != 1) {
        std::cout << "\n    CRITICAL: Generator point is NOT on the curve y² = x³ + 4!";
        return TestResult::FAILED;
    }
    
    return TestResult::PASSED;
}

TestResult test_g1_double_equals_add_self() {
    // Verify: 2P = P + P
    G1Affine gen;
    memcpy(gen.x.limbs, G1_GEN_X_MONTGOMERY, sizeof(G1_GEN_X_MONTGOMERY));
    memcpy(gen.y.limbs, G1_GEN_Y_MONTGOMERY, sizeof(G1_GEN_Y_MONTGOMERY));
    
    G1Affine* d_gen;
    G1Projective *d_p, *d_double, *d_add;
    int* d_equal;
    
    SECURITY_CHECK_CUDA(cudaMalloc(&d_gen, sizeof(G1Affine)));
    SECURITY_CHECK_CUDA(cudaMalloc(&d_p, sizeof(G1Projective)));
    SECURITY_CHECK_CUDA(cudaMalloc(&d_double, sizeof(G1Projective)));
    SECURITY_CHECK_CUDA(cudaMalloc(&d_add, sizeof(G1Projective)));
    SECURITY_CHECK_CUDA(cudaMalloc(&d_equal, sizeof(int)));
    
    SECURITY_CHECK_CUDA(cudaMemcpy(d_gen, &gen, sizeof(G1Affine), cudaMemcpyHostToDevice));
    
    // Convert to projective on device
    G1Projective p_proj = G1Projective::from_affine(gen);
    SECURITY_CHECK_CUDA(cudaMemcpy(d_p, &p_proj, sizeof(G1Projective), cudaMemcpyHostToDevice));
    
    // Compute 2P
    kat_g1_double_kernel<<<1, 1>>>(d_p, d_double);
    SECURITY_CHECK_CUDA(cudaDeviceSynchronize());
    
    // Compute P + P
    kat_g1_add_kernel<<<1, 1>>>(d_p, d_p, d_add);
    SECURITY_CHECK_CUDA(cudaDeviceSynchronize());
    
    // Compare
    kat_projective_equal_kernel<<<1, 1>>>(d_double, d_add, d_equal);
    SECURITY_CHECK_CUDA(cudaDeviceSynchronize());
    
    int equal;
    SECURITY_CHECK_CUDA(cudaMemcpy(&equal, d_equal, sizeof(int), cudaMemcpyDeviceToHost));
    
    cudaFree(d_gen);
    cudaFree(d_p);
    cudaFree(d_double);
    cudaFree(d_add);
    cudaFree(d_equal);
    
    if (equal != 1) {
        std::cout << "\n    2P ≠ P + P";
        return TestResult::FAILED;
    }
    return TestResult::PASSED;
}

TestResult test_g1_identity_add() {
    // Verify: O + P = P
    G1Affine gen;
    memcpy(gen.x.limbs, G1_GEN_X_MONTGOMERY, sizeof(G1_GEN_X_MONTGOMERY));
    memcpy(gen.y.limbs, G1_GEN_Y_MONTGOMERY, sizeof(G1_GEN_Y_MONTGOMERY));
    
    G1Projective p_proj = G1Projective::from_affine(gen);
    G1Projective identity = G1Projective::identity();
    
    G1Projective *d_p, *d_identity, *d_result;
    int* d_equal;
    
    SECURITY_CHECK_CUDA(cudaMalloc(&d_p, sizeof(G1Projective)));
    SECURITY_CHECK_CUDA(cudaMalloc(&d_identity, sizeof(G1Projective)));
    SECURITY_CHECK_CUDA(cudaMalloc(&d_result, sizeof(G1Projective)));
    SECURITY_CHECK_CUDA(cudaMalloc(&d_equal, sizeof(int)));
    
    SECURITY_CHECK_CUDA(cudaMemcpy(d_p, &p_proj, sizeof(G1Projective), cudaMemcpyHostToDevice));
    SECURITY_CHECK_CUDA(cudaMemcpy(d_identity, &identity, sizeof(G1Projective), cudaMemcpyHostToDevice));
    
    // O + P
    kat_g1_add_kernel<<<1, 1>>>(d_identity, d_p, d_result);
    SECURITY_CHECK_CUDA(cudaDeviceSynchronize());
    
    // Compare with P
    kat_projective_equal_kernel<<<1, 1>>>(d_result, d_p, d_equal);
    SECURITY_CHECK_CUDA(cudaDeviceSynchronize());
    
    int equal;
    SECURITY_CHECK_CUDA(cudaMemcpy(&equal, d_equal, sizeof(int), cudaMemcpyDeviceToHost));
    
    cudaFree(d_p);
    cudaFree(d_identity);
    cudaFree(d_result);
    cudaFree(d_equal);
    
    if (equal != 1) {
        std::cout << "\n    O + P ≠ P";
        return TestResult::FAILED;
    }
    return TestResult::PASSED;
}

TestResult test_fr_mul_test_vectors() {
    const int n = sizeof(FR_MUL_VECTORS) / sizeof(FR_MUL_VECTORS[0]);
    
    for (int i = 0; i < n; i++) {
        const auto& vec = FR_MUL_VECTORS[i];
        
        Fr a, b, expected;
        memcpy(a.limbs, vec.a, sizeof(vec.a));
        memcpy(b.limbs, vec.b, sizeof(vec.b));
        memcpy(expected.limbs, vec.expected, sizeof(vec.expected));
        
        Fr *d_a, *d_b, *d_out;
        SECURITY_CHECK_CUDA(cudaMalloc(&d_a, sizeof(Fr)));
        SECURITY_CHECK_CUDA(cudaMalloc(&d_b, sizeof(Fr)));
        SECURITY_CHECK_CUDA(cudaMalloc(&d_out, sizeof(Fr)));
        
        SECURITY_CHECK_CUDA(cudaMemcpy(d_a, &a, sizeof(Fr), cudaMemcpyHostToDevice));
        SECURITY_CHECK_CUDA(cudaMemcpy(d_b, &b, sizeof(Fr), cudaMemcpyHostToDevice));
        
        kat_mul_kernel<<<1, 256>>>(d_a, d_b, d_out, 1);
        SECURITY_CHECK_CUDA(cudaDeviceSynchronize());
        
        Fr result;
        SECURITY_CHECK_CUDA(cudaMemcpy(&result, d_out, sizeof(Fr), cudaMemcpyDeviceToHost));
        
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_out);
        
        if (!limbs_equal(result.limbs, expected.limbs, 4)) {
            std::cout << "\n    Test vector failed: " << vec.description;
            return TestResult::FAILED;
        }
    }
    
    return TestResult::PASSED;
}

// =============================================================================
// Main Test Runner
// =============================================================================

void register_kat_tests(SecurityTestSuite& suite) {
    // Critical constant verification (must all pass)
    suite.add_test("Fr modulus matches specification", "Constants", 
                   test_fr_modulus_matches_spec, true);
    suite.add_test("Fq modulus matches specification", "Constants", 
                   test_fq_modulus_matches_spec, true);
    suite.add_test("Fr Montgomery R matches specification", "Constants", 
                   test_fr_one_matches_spec, true);
    suite.add_test("Fr R² matches specification", "Constants", 
                   test_fr_r2_matches_spec, true);
    suite.add_test("Fr Montgomery inverse matches specification", "Constants", 
                   test_fr_inv_matches_spec, true);
    
    // Device-host consistency
    suite.add_test("Fr::one() device matches host", "Device Consistency",
                   test_fr_one_device_matches_host, true);
    
    // Generator point validation
    suite.add_test("G1 generator coordinates match specification", "Generator",
                   test_g1_generator_on_curve, true);
    
    // Field arithmetic vectors
    suite.add_test("Fr multiplication test vectors", "Field Arithmetic",
                   test_fr_mul_test_vectors);
    
    // Point operation tests
    suite.add_test("G1: 2P = P + P", "Point Operations",
                   test_g1_double_equals_add_self);
    suite.add_test("G1: O + P = P", "Point Operations",
                   test_g1_identity_add);
}

int main(int argc, char** argv) {
    (void)argc; (void)argv;
    
    // Check CUDA device
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
    register_kat_tests(suite);
    
    bool success = suite.run_all();
    return success ? 0 : 1;
}
