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
 * @file point.cuh
 * @brief BLS12-381 Elliptic Curve Point Types and Operations
 * 
 * Implements G1 and G2 point operations in both affine and projective coordinates.
 * 
 * ARCHITECTURE:
 * =============
 * This header defines:
 * 
 * 1. Point Types:
 *    - Affine<F>: (x, y) coordinates, efficient storage
 *    - Projective<F, S, Tag>: (X, Y, Z) coordinates, efficient arithmetic
 *    - G1Affine/G1Projective: Points on E(Fq)
 *    - G2Affine/G2Projective: Points on E'(Fq2)
 * 
 * 2. Extension Field:
 *    - Fq2 = Fq[u]/(u² + 1): Complex extension for G2
 *    - ComplexExtensionField template with arithmetic
 * 
 * 3. Inline __device__ Functions:
 *    - Point addition (g1_add, g2_add)
 *    - Point doubling (g1_double, g2_double)
 *    - Point negation (g1_neg, g2_neg)
 *    - Affine ↔ Projective conversion
 * 
 * Curve Equations:
 * - G1: y² = x³ + 4 over Fq
 * - G2: y² = x³ + 4(u + 1) over Fq2
 * 
 * All coordinate operations assume Montgomery form.
 */

#pragma once

#include "field.cuh"

namespace bls12_381 {

// =============================================================================
// Forward declarations
// =============================================================================

template<typename BaseField> struct Affine;
template<typename BaseField, typename ScalarField, typename CurveTag> struct Projective;
struct G1;
struct G2;

// =============================================================================
// Curve tags
// =============================================================================

struct G1 {};
struct G2 {};

// =============================================================================
// Complex extension field for G2 (Fq2 = Fq[u] / (u^2 + 1))
// =============================================================================

template<typename BaseConfig, typename BaseField>
struct ComplexExtensionField {
    BaseField c0;  // Real part
    BaseField c1;  // Imaginary part (coefficient of u)

    __host__ __device__ ComplexExtensionField() : c0(), c1() {}
    
    __host__ __device__ ComplexExtensionField(const BaseField& real, const BaseField& imag)
        : c0(real), c1(imag) {}

    __host__ __device__ static ComplexExtensionField zero() {
        return ComplexExtensionField();
    }

    __host__ __device__ static ComplexExtensionField one() {
        return ComplexExtensionField(BaseField::one(), BaseField::zero());
    }

    /**
     * Constant-time zero check.
     * Uses bitwise OR to avoid short-circuit evaluation.
     */
    __host__ __device__ bool is_zero() const {
        // Avoid short-circuit && which leaks timing
        // Both is_zero() calls execute regardless of first result
        bool c0_zero = c0.is_zero();
        bool c1_zero = c1.is_zero();
        return c0_zero & c1_zero;  // Bitwise AND, no short-circuit
    }

    /**
     * Constant-time equality check.
     * Both component comparisons execute regardless of first result.
     */
    __host__ __device__ bool operator==(const ComplexExtensionField& other) const {
        // Avoid short-circuit && which leaks timing
        bool c0_eq = (c0 == other.c0);
        bool c1_eq = (c1 == other.c1);
        return c0_eq & c1_eq;  // Bitwise AND, no short-circuit
    }

    __host__ __device__ bool operator!=(const ComplexExtensionField& other) const {
        return !(*this == other);
    }
};

// Fq2 = Fq[u] / (u^2 + 1)
using Fq2 = ComplexExtensionField<fq_config, Fq>;

// Fq2 arithmetic
__device__ __forceinline__ void fq2_add(Fq2& result, const Fq2& a, const Fq2& b) {
    field_add(result.c0, a.c0, b.c0);
    field_add(result.c1, a.c1, b.c1);
}

__device__ __forceinline__ void fq2_sub(Fq2& result, const Fq2& a, const Fq2& b) {
    field_sub(result.c0, a.c0, b.c0);
    field_sub(result.c1, a.c1, b.c1);
}

__device__ __forceinline__ void fq2_mul(Fq2& result, const Fq2& a, const Fq2& b) {
    // (a0 + a1*u) * (b0 + b1*u) = (a0*b0 - a1*b1) + (a0*b1 + a1*b0)*u
    // Using Karatsuba: 3 multiplications instead of 4
    Fq t0, t1, t2;
    
    field_mul(t0, a.c0, b.c0);           // a0 * b0
    field_mul(t1, a.c1, b.c1);           // a1 * b1
    
    Fq a_sum, b_sum;
    field_add(a_sum, a.c0, a.c1);        // a0 + a1
    field_add(b_sum, b.c0, b.c1);        // b0 + b1
    field_mul(t2, a_sum, b_sum);         // (a0 + a1)(b0 + b1)
    
    field_sub(result.c0, t0, t1);        // a0*b0 - a1*b1
    field_sub(result.c1, t2, t0);        // (a0+a1)(b0+b1) - a0*b0
    field_sub(result.c1, result.c1, t1); // ... - a1*b1
}

__device__ __forceinline__ void fq2_sqr(Fq2& result, const Fq2& a) {
    // (a0 + a1*u)^2 = (a0^2 - a1^2) + 2*a0*a1*u
    // Optimized: (a0 + a1)(a0 - a1) for real part
    // 
    // NOTE: This function is aliasing-safe (result can be the same as a)
    Fq t0, t1;
    
    field_add(t0, a.c0, a.c1);           // t0 = a0 + a1
    field_sub(t1, a.c0, a.c1);           // t1 = a0 - a1
    
    // Compute c1 BEFORE c0 to handle aliasing (result == a)
    Fq c1_tmp;
    field_mul(c1_tmp, a.c0, a.c1);       // c1_tmp = a0 * a1
    field_add(c1_tmp, c1_tmp, c1_tmp);   // c1_tmp = 2 * a0 * a1
    
    // Now safe to write c0 (we've finished reading a.c0 and a.c1)
    field_mul(result.c0, t0, t1);        // (a0+a1)(a0-a1) = a0^2 - a1^2
    result.c1 = c1_tmp;
}

__device__ __forceinline__ void fq2_neg(Fq2& result, const Fq2& a) {
    field_neg(result.c0, a.c0);
    field_neg(result.c1, a.c1);
}

/**
 * @brief Fq2 inversion: result = 1/a in Fq2
 *
 * This function is constant-time. The inversion is always
 * computed, and edge cases (zero input, zero norm) are handled using
 * constant-time selection. Returns zero for invalid inputs.
 */
__device__ __forceinline__ void fq2_inv(Fq2& result, const Fq2& a) {
    // (a0 + a1*u)^-1 = (a0 - a1*u) / (a0^2 + a1^2)
    // For u^2 = -1: norm = a0^2 + a1^2
    Fq t0, t1, norm, norm_inv;
    
    field_sqr(t0, a.c0);                 // a0^2
    field_sqr(t1, a.c1);                 // a1^2
    field_add(norm, t0, t1);             // a0^2 + a1^2
    
    // Always compute inversion (field_inv handles zero input safely)
    field_inv(norm_inv, norm);           // 1 / (a0^2 + a1^2)
    
    // Compute the inverse components
    Fq2 computed_inv;
    field_mul(computed_inv.c0, a.c0, norm_inv);  // a0 / norm
    field_neg(t0, a.c1);
    field_mul(computed_inv.c1, t0, norm_inv);    // -a1 / norm
    
    // Check if input was zero in constant-time
    // Accumulate OR of all limbs from both components
    uint64_t nonzero_acc = 0;
    for (int i = 0; i < fq_config::LIMBS; i++) {
        nonzero_acc |= a.c0.limbs[i];
        nonzero_acc |= a.c1.limbs[i];
    }
    int input_is_zero = (nonzero_acc == 0) ? 1 : 0;
    
    // Constant-time selection: if input was zero, return zero
    Fq2 zero_result = Fq2::zero();
    
    // Select between computed inverse and zero based on input validity
    // Use field_cmov on each component
    field_cmov(result.c0, zero_result.c0, computed_inv.c0, input_is_zero);
    field_cmov(result.c1, zero_result.c1, computed_inv.c1, input_is_zero);
}

// Overloads for Fq2 to work with generic field templates
// This allows G2 operations to use the same template code as G1

__device__ __forceinline__ void field_add(Fq2& result, const Fq2& a, const Fq2& b) {
    fq2_add(result, a, b);
}

__device__ __forceinline__ void field_sub(Fq2& result, const Fq2& a, const Fq2& b) {
    fq2_sub(result, a, b);
}

__device__ __forceinline__ void field_mul(Fq2& result, const Fq2& a, const Fq2& b) {
    fq2_mul(result, a, b);
}

__device__ __forceinline__ void field_sqr(Fq2& result, const Fq2& a) {
    fq2_sqr(result, a);
}

__device__ __forceinline__ void field_inv(Fq2& result, const Fq2& a) {
    fq2_inv(result, a);
}

__device__ __forceinline__ void field_neg(Fq2& result, const Fq2& a) {
    fq2_neg(result, a);
}

// Fq2 operators (for convenience)
__device__ __forceinline__ Fq2 operator+(const Fq2& a, const Fq2& b) {
    Fq2 result;
    fq2_add(result, a, b);
    return result;
}

__device__ __forceinline__ Fq2 operator-(const Fq2& a, const Fq2& b) {
    Fq2 result;
    fq2_sub(result, a, b);
    return result;
}

__device__ __forceinline__ Fq2 operator*(const Fq2& a, const Fq2& b) {
    Fq2 result;
    fq2_mul(result, a, b);
    return result;
}

__device__ __forceinline__ Fq2 operator-(const Fq2& a) {
    Fq2 result;
    fq2_neg(result, a);
    return result;
}

// Forward declaration for Projective
template<typename BaseField, typename ScalarField, typename CurveTag> struct Projective;

// =============================================================================
// Affine point representation
// =============================================================================

template<typename BaseField>
struct Affine {
    BaseField x;
    BaseField y;

    __host__ __device__ Affine() : x(), y() {}
    
    __host__ __device__ Affine(const BaseField& _x, const BaseField& _y) : x(_x), y(_y) {}

    __host__ __device__ static Affine identity() {
        // Point at infinity represented as (0, 0)
        return Affine();
    }

    __host__ __device__ bool is_identity() const {
        return x.is_zero() && y.is_zero();
    }

    __host__ __device__ bool operator==(const Affine& other) const {
        return x == other.x && y == other.y;
    }

    __host__ __device__ Affine neg() const {
        Affine result;
        result.x = x;
        result.y = -y;  // Use unary minus operator
        return result;
    }
    
    // Convert to projective coordinates (defined after Projective is complete)
    template<typename ScalarField, typename CurveTag>
    __device__ __forceinline__ Projective<BaseField, ScalarField, CurveTag> to_projective() const;
};

// Type aliases
using G1Affine = Affine<Fq>;
using G2Affine = Affine<Fq2>;

// =============================================================================
// Point Validation Functions
// =============================================================================
// These functions verify that points are on the correct curve.
// MUST be called for any externally-provided points before use.

/**
 * @brief Check if a G1 affine point is on the curve E(Fq): y² = x³ + 4
 * 
 * This function should be called for all untrusted point inputs.
 * Failure to validate points can enable invalid curve attacks.
 * 
 * @param p The point to validate
 * @return true if p is on the curve or is the identity point
 */
__device__ __forceinline__ bool g1_is_on_curve(const G1Affine& p) {
    // Identity is always valid
    if (p.is_identity()) return true;
    
    // Check y² = x³ + b where b = 4
    Fq x2, x3, y2, rhs, b;
    
    // Load curve parameter b (Montgomery form)
    for (int i = 0; i < fq_config::LIMBS; i++) {
        b.limbs[i] = G1_B_DEV[i];
    }
    
    field_sqr(x2, p.x);          // x²
    field_mul(x3, x2, p.x);       // x³
    field_sqr(y2, p.y);          // y²
    field_add(rhs, x3, b);        // x³ + b
    
    return y2 == rhs;
}

/**
 * @brief Check if a G2 affine point is on the twist curve E'(Fq2): y² = x³ + 4(1+u)
 * 
 * This function should be called for all untrusted point inputs.
 * Failure to validate points can enable invalid curve attacks.
 * 
 * @param p The point to validate
 * @return true if p is on the curve or is the identity point
 */
__device__ __forceinline__ bool g2_is_on_curve(const G2Affine& p) {
    // Identity is always valid
    if (p.is_identity()) return true;
    
    // Check y² = x³ + b' where b' = 4(1+u) in Fq2
    Fq2 x2, x3, y2, rhs, b;
    
    // Load curve parameter b' = 4(1+u) in Montgomery form
    for (int i = 0; i < fq_config::LIMBS; i++) {
        b.c0.limbs[i] = G2_B_C0_DEV[i];
        b.c1.limbs[i] = G2_B_C1_DEV[i];
    }
    
    fq2_sqr(x2, p.x);            // x²
    fq2_mul(x3, x2, p.x);         // x³
    fq2_sqr(y2, p.y);            // y²
    fq2_add(rhs, x3, b);          // x³ + b'
    
    return y2 == rhs;
}

// =============================================================================
// Subgroup Membership Checks
// =============================================================================
// BLS12-381 G2 has a large cofactor. Points on the twist curve E'(Fq2)
// may not be in the prime-order subgroup G2. Small subgroup attacks
// can recover private keys when operating on malicious points.
//
// For G1: The cofactor is small (h ≈ 2^126) but non-trivial.
//         Subgroup check: [h]P ≠ O (multiply by cofactor)
//
// For G2: The cofactor is very large (~300 bits).
//         Efficient check uses the Frobenius endomorphism:
//         P ∈ G2 ⟺ ψ(P) = [x]P  where x is the BLS parameter

/**
 * @brief Check if a G1 point is in the prime-order subgroup
 * 
 * For G1, the cofactor h = (z-1)²/3 where z is the BLS parameter.
 * A point P is in G1 iff [r]P = O (where r is the subgroup order).
 * 
 * Implementation: Multiply by the cofactor h. If result is not identity,
 * the original point was not in G1.
 * 
 * NOTE: For most BLS12-381 use cases, G1 subgroup membership is not
 * strictly required because the cofactor is relatively prime to r.
 * However, it's good practice to validate externally-provided points.
 * 
 * @param p The G1 point to check (must already be on curve)
 * @return true if p is in the prime-order subgroup G1
 */
// TODO: Implement g1_is_in_subgroup() using cofactor multiplication
// For now, applications should call g1_is_on_curve() which catches most attacks

/**
 * @brief Check if a G2 point is in the prime-order subgroup
 * 
 * SECURITY CRITICAL: This check is REQUIRED for all externally-provided
 * G2 points. Failure to perform this check enables small subgroup attacks.
 * 
 * Efficient implementation uses the Frobenius endomorphism ψ:
 *   P ∈ G2 ⟺ ψ(P) = [x]P
 * where x = -0xd201000000010000 is the BLS parameter.
 * 
 * The Frobenius endomorphism is:
 *   ψ(x, y) = (x^p * c₁, y^p * c₂)
 * where c₁, c₂ are the Frobenius coefficients.
 * 
 * @param p The G2 projective point to check (must already be on curve)
 * @return true if p is in the prime-order subgroup G2
 */
// TODO: Implement g2_is_in_subgroup() using Frobenius endomorphism check
// Requires:
//   1. Frobenius coefficients in Fq2 (conjugate + multiply by coefficients)
//   2. Scalar multiplication by x (64-bit scalar, efficient)
//   3. Comparison of ψ(P) with [x]P
//
// For production use, either:
//   A) Implement the endomorphism check (fast, O(log x) point operations)
//   B) Multiply by the full cofactor (slow, O(log h') where h' is ~300 bits)
//   C) Use only trusted point sources (generator, hash-to-curve)

// =============================================================================
// Projective point representation using Jacobian coordinates
// (X, Y, Z) where x = X/Z², y = Y/Z³
// =============================================================================

template<typename BaseField, typename ScalarField, typename CurveTag>
struct Projective {
    BaseField X;
    BaseField Y;
    BaseField Z;

    __host__ __device__ Projective() : X(), Y(), Z() {
        // Proper identity point is (0:1:0) in projective coordinates
        Y = BaseField::one();
    }

    __host__ __device__ Projective(const BaseField& x, const BaseField& y, const BaseField& z)
        : X(x), Y(y), Z(z) {}

    __host__ __device__ static Projective identity() {
        Projective p;
        p.X = BaseField::zero();
        p.Y = BaseField::one();  // Convention: (0:1:0) for identity
        p.Z = BaseField::zero();
        return p;
    }

    __host__ __device__ static Projective from_affine(const Affine<BaseField>& p) {
        if (p.is_identity()) {
            return identity();
        }
        return Projective(p.x, p.y, BaseField::one());
    }

    __host__ __device__ bool is_identity() const {
        return Z.is_zero();
    }

    __host__ __device__ bool operator==(const Projective& other) const {
        if (is_identity() && other.is_identity()) return true;
        if (is_identity() || other.is_identity()) return false;
        
        // Compare X1*Z2 == X2*Z1 and Y1*Z2 == Y2*Z1
        BaseField lhs_x, rhs_x, lhs_y, rhs_y;
        field_mul(lhs_x, X, other.Z);
        field_mul(rhs_x, other.X, Z);
        field_mul(lhs_y, Y, other.Z);
        field_mul(rhs_y, other.Y, Z);
        
        return lhs_x == rhs_x && lhs_y == rhs_y;
    }
    
    // Convert to affine coordinates
    // For Jacobian: x = X/Z², y = Y/Z³
    __device__ __forceinline__ Affine<BaseField> to_affine() const {
        if (is_identity()) {
            return Affine<BaseField>::identity();
        }
        
        // Defensive check - Z should never be zero for non-identity points
        // but this guards against bugs in point arithmetic
        if (Z.is_zero()) {
            return Affine<BaseField>::identity();
        }
        
        BaseField z_inv, z_inv_sq, z_inv_cu;
        field_inv(z_inv, Z);
        field_mul(z_inv_sq, z_inv, z_inv);   // z_inv^2
        field_mul(z_inv_cu, z_inv_sq, z_inv); // z_inv^3
        
        BaseField ax, ay;
        field_mul(ax, X, z_inv_sq);   // x = X / Z²
        field_mul(ay, Y, z_inv_cu);   // y = Y / Z³
        return Affine<BaseField>(ax, ay);
    }
};

// Type aliases
using G1Projective = Projective<Fq, Fr, G1>;
using G2Projective = Projective<Fq2, Fr, G2>;

// Implementation of Affine::to_projective
template<typename BaseField>
template<typename ScalarField, typename CurveTag>
__device__ __forceinline__ Projective<BaseField, ScalarField, CurveTag> 
Affine<BaseField>::to_projective() const {
    return Projective<BaseField, ScalarField, CurveTag>::from_affine(*this);
}

// Convenience methods for G1Affine and G2Affine
__device__ __forceinline__ G1Projective g1_affine_to_projective(const G1Affine& p) {
    return G1Projective::from_affine(p);
}

__device__ __forceinline__ G2Projective g2_affine_to_projective(const G2Affine& p) {
    return G2Projective::from_affine(p);
}

// =============================================================================
// Constant-Time Point Selection (for side-channel resistance)
// =============================================================================

/**
 * @brief Constant-time conditional selection for G1Projective: result = cond ? a : b
 * 
 * This function executes in constant time regardless of the value
 * of `cond`. It prevents timing side-channels by avoiding branches.
 */
__device__ __forceinline__ void g1_cmov(
    G1Projective& result,
    const G1Projective& a,
    const G1Projective& b,
    int cond
) {
    field_cmov(result.X, a.X, b.X, cond);
    field_cmov(result.Y, a.Y, b.Y, cond);
    field_cmov(result.Z, a.Z, b.Z, cond);
}

/**
 * @brief Constant-time conditional copy for Fq2 (used for G2)
 */
__device__ __forceinline__ void fq2_cmov(
    Fq2& result,
    const Fq2& a,
    const Fq2& b,
    int cond
) {
    field_cmov(result.c0, a.c0, b.c0, cond);
    field_cmov(result.c1, a.c1, b.c1, cond);
}

/**
 * @brief Constant-time conditional selection for G2Projective: result = cond ? a : b
 */
__device__ __forceinline__ void g2_cmov(
    G2Projective& result,
    const G2Projective& a,
    const G2Projective& b,
    int cond
) {
    fq2_cmov(result.X, a.X, b.X, cond);
    fq2_cmov(result.Y, a.Y, b.Y, cond);
    fq2_cmov(result.Z, a.Z, b.Z, cond);
}

// =============================================================================
// G1 Point Operations
// =============================================================================

/**
 * @brief Double a projective point: result = 2 * P (Constant-Time)
 * 
 * Uses the doubling formula for short Weierstrass curves y^2 = x^3 + b
 * Cost: 4S + 3M + 8add + 4sub
 * 
 * NOTE: This function is safe for in-place operation (result == p)
 * Uses constant-time selection for identity handling to prevent
 *           timing side-channels.
 */
__device__ __forceinline__ void g1_double(G1Projective& result, const G1Projective& p) {
    // Pre-compute identity condition (will use cmov at end)
    int is_identity = p.is_identity() ? 1 : 0;
    
    // Save original p for identity case (handles aliasing when result == p)
    G1Projective saved_p = p;

    Fq t0, t1, t2, t3;
    
    // t0 = Y^2
    field_sqr(t0, p.Y);
    
    // t1 = 4 * X * Y^2
    field_mul(t1, p.X, t0);
    field_add(t1, t1, t1);
    field_add(t1, t1, t1);
    
    // t2 = 8 * Y^4
    field_sqr(t2, t0);
    field_add(t2, t2, t2);
    field_add(t2, t2, t2);
    field_add(t2, t2, t2);
    
    // t3 = 3 * X^2 (for a = 0)
    field_sqr(t3, p.X);
    Fq t3_2;
    field_add(t3_2, t3, t3);
    field_add(t3, t3_2, t3);
    
    // Compute all outputs to temporaries first (for in-place safety)
    Fq new_X, new_Y, new_Z;
    
    // X3 = t3^2 - 2*t1
    field_sqr(new_X, t3);
    field_sub(new_X, new_X, t1);
    field_sub(new_X, new_X, t1);
    
    // Y3 = t3 * (t1 - X3) - t2
    field_sub(t0, t1, new_X);
    field_mul(new_Y, t3, t0);
    field_sub(new_Y, new_Y, t2);
    
    // Z3 = 2 * Y * Z (must use original p.Y and p.Z!)
    field_mul(new_Z, p.Y, p.Z);
    field_add(new_Z, new_Z, new_Z);
    
    // Computed result
    G1Projective computed;
    computed.X = new_X;
    computed.Y = new_Y;
    computed.Z = new_Z;
    
    // Constant-time selection: if identity, return saved_p (which is identity)
    // Note: We use saved_p instead of G1Projective::identity() to handle
    // the case where result == p and we've already computed into p
    g1_cmov(result, saved_p, computed, is_identity);
}

/**
 * @brief Add two projective points: result = P + Q (Constant-Time)
 * 
 * Uses complete addition formula with constant-time edge case handling.
 * Computes all possible results and uses cmov to select the correct one.
 * 
 * NOTE: This function is safe for in-place operation (result == p or result == q)
 * 
 * No branches that depend on point values. All edge cases handled
 *           via constant-time selection (cmov) to prevent timing side-channels.
 */
__device__ __forceinline__ void g1_add(G1Projective& result, const G1Projective& p, const G1Projective& q) {
    // Pre-compute identity conditions
    int p_is_identity = p.is_identity() ? 1 : 0;
    int q_is_identity = q.is_identity() ? 1 : 0;
    
    // CRITICAL: Save original inputs FIRST to handle aliasing (result == p or result == q)
    // These copies are used in the final cmov selection
    G1Projective saved_p = p;
    G1Projective saved_q = q;

    Fq z1z1, z2z2, u1, u2, s1, s2, h, i, j, r, v;
    
    // Z1Z1 = Z1^2
    field_sqr(z1z1, p.Z);
    // Z2Z2 = Z2^2
    field_sqr(z2z2, q.Z);
    
    // U1 = X1 * Z2Z2
    field_mul(u1, p.X, z2z2);
    // U2 = X2 * Z1Z1
    field_mul(u2, q.X, z1z1);
    
    // S1 = Y1 * Z2 * Z2Z2
    field_mul(s1, p.Y, q.Z);
    field_mul(s1, s1, z2z2);
    // S2 = Y2 * Z1 * Z1Z1
    field_mul(s2, q.Y, p.Z);
    field_mul(s2, s2, z1z1);
    
    // H = U2 - U1
    field_sub(h, u2, u1);
    
    // Compute edge case conditions (constant-time)
    int h_is_zero = h.is_zero() ? 1 : 0;
    Fq s_diff;
    field_sub(s_diff, s2, s1);
    int s_diff_is_zero = s_diff.is_zero() ? 1 : 0;
    
    // I = (2*H)^2
    field_add(i, h, h);
    field_sqr(i, i);
    
    // J = H * I
    field_mul(j, h, i);
    
    // r = 2 * (S2 - S1)
    field_sub(r, s2, s1);
    field_add(r, r, r);
    
    // V = U1 * I
    field_mul(v, u1, i);
    
    // Save values that will be needed after we start writing to result
    // (for in-place safety)
    Fq saved_Z1 = p.Z;
    Fq saved_Z2 = q.Z;
    Fq saved_S1 = s1;
    
    // Compute to temporaries
    Fq new_X, new_Y, new_Z;
    
    // X3 = r^2 - J - 2*V
    field_sqr(new_X, r);
    field_sub(new_X, new_X, j);
    field_sub(new_X, new_X, v);
    field_sub(new_X, new_X, v);
    
    // Y3 = r * (V - X3) - 2*S1*J
    field_sub(v, v, new_X);
    field_mul(new_Y, r, v);
    field_mul(saved_S1, saved_S1, j);
    field_add(saved_S1, saved_S1, saved_S1);
    field_sub(new_Y, new_Y, saved_S1);
    
    // Z3 = ((Z1 + Z2)^2 - Z1Z1 - Z2Z2) * H
    field_add(new_Z, saved_Z1, saved_Z2);
    field_sqr(new_Z, new_Z);
    field_sub(new_Z, new_Z, z1z1);
    field_sub(new_Z, new_Z, z2z2);
    field_mul(new_Z, new_Z, h);
    
    // Standard addition result
    G1Projective add_result;
    add_result.X = new_X;
    add_result.Y = new_Y;
    add_result.Z = new_Z;
    
    // Compute doubled result (for when P == Q)
    // Use saved_p since p might be aliased with result
    G1Projective doubled;
    g1_double(doubled, saved_p);
    
    // Identity point (for when P == -Q)
    G1Projective identity_point = G1Projective::identity();
    
    // Constant-time selection of result:
    // 1. Start with standard addition result
    // 2. If h == 0 and s_diff == 0: P == Q, use doubled
    // 3. If h == 0 and s_diff != 0: P == -Q, use identity
    // 4. If p is identity: return q (use saved_q)
    // 5. If q is identity: return p (use saved_p)
    
    result = add_result;
    
    int use_doubled = h_is_zero & s_diff_is_zero;
    int use_identity = h_is_zero & (!s_diff_is_zero ? 1 : 0);
    
    g1_cmov(result, doubled, result, use_doubled);
    g1_cmov(result, identity_point, result, use_identity);
    
    // Handle input identity cases (MUST use saved copies due to aliasing)
    g1_cmov(result, saved_q, result, p_is_identity);
    g1_cmov(result, saved_p, result, q_is_identity);
}

/**
 * @brief Mixed addition: result = P + Q where Q is affine (Constant-Time)
 * 
 * More efficient than general addition when one point is affine.
 * NOTE: This function is safe for in-place operation (result == p)
 * 
 * Uses constant-time selection for all edge cases to prevent
 *           timing side-channels.
 */
__device__ __forceinline__ void g1_add_mixed(G1Projective& result, const G1Projective& p, const G1Affine& q) {
    // Pre-compute identity conditions
    int p_is_identity = p.is_identity() ? 1 : 0;
    int q_is_identity = q.is_identity() ? 1 : 0;
    
    // CRITICAL: Save original p FIRST to handle aliasing (result == p)
    // This copy is used in the final cmov selection
    G1Projective saved_p = p;

    Fq z1z1, u2, s2, h, hh, i, j, r, v;
    
    // Z1Z1 = Z1^2
    field_sqr(z1z1, p.Z);
    
    // U2 = X2 * Z1Z1 (U1 = X1)
    field_mul(u2, q.x, z1z1);
    
    // S2 = Y2 * Z1 * Z1Z1 (S1 = Y1)
    field_mul(s2, q.y, p.Z);
    field_mul(s2, s2, z1z1);
    
    // H = U2 - X1
    field_sub(h, u2, p.X);
    
    // Compute edge case conditions (constant-time)
    int h_is_zero = h.is_zero() ? 1 : 0;
    Fq s_diff;
    field_sub(s_diff, s2, p.Y);
    int s_diff_is_zero = s_diff.is_zero() ? 1 : 0;
    
    // HH = H^2
    field_sqr(hh, h);
    
    // I = 4 * HH
    field_add(i, hh, hh);
    field_add(i, i, i);
    
    // J = H * I
    field_mul(j, h, i);
    
    // r = 2 * (S2 - Y1)
    field_sub(r, s2, p.Y);
    field_add(r, r, r);
    
    // V = X1 * I
    field_mul(v, p.X, i);
    
    // Save p.Y and p.Z before we potentially overwrite (for in-place safety)
    Fq saved_Y = p.Y;
    Fq saved_Z = p.Z;
    
    // Compute to temporaries for in-place safety
    Fq new_X, new_Y, new_Z;
    
    // X3 = r^2 - J - 2*V
    field_sqr(new_X, r);
    field_sub(new_X, new_X, j);
    field_sub(new_X, new_X, v);
    field_sub(new_X, new_X, v);
    
    // Y3 = r * (V - X3) - 2*Y1*J
    Fq t;
    field_sub(t, v, new_X);
    field_mul(new_Y, r, t);
    field_mul(t, saved_Y, j);
    field_add(t, t, t);
    field_sub(new_Y, new_Y, t);
    
    // Z3 = (Z1 + H)^2 - Z1Z1 - HH
    field_add(new_Z, saved_Z, h);
    field_sqr(new_Z, new_Z);
    field_sub(new_Z, new_Z, z1z1);
    field_sub(new_Z, new_Z, hh);
    
    // Standard addition result
    G1Projective add_result;
    add_result.X = new_X;
    add_result.Y = new_Y;
    add_result.Z = new_Z;
    
    // Compute doubled result (for when P == Q)
    // Use saved_p since p might be aliased with result
    G1Projective doubled;
    g1_double(doubled, saved_p);
    
    // Identity point (for when P == -Q)
    G1Projective identity_point = G1Projective::identity();
    
    // Convert q to projective (for when p is identity)
    G1Projective q_proj = G1Projective::from_affine(q);
    
    // Constant-time selection of result:
    // 1. Start with standard addition result
    // 2. If h == 0 and s_diff == 0: P == Q, use doubled
    // 3. If h == 0 and s_diff != 0: P == -Q, use identity
    // 4. If p is identity: return q_proj
    // 5. If q is identity: return p (use saved_p)
    
    result = add_result;
    
    int use_doubled = h_is_zero & s_diff_is_zero;
    int use_identity = h_is_zero & (!s_diff_is_zero ? 1 : 0);
    
    g1_cmov(result, doubled, result, use_doubled);
    g1_cmov(result, identity_point, result, use_identity);
    
    // Handle input identity cases (MUST use saved_p due to aliasing)
    g1_cmov(result, q_proj, result, p_is_identity);
    g1_cmov(result, saved_p, result, q_is_identity);
}

/**
 * @brief Negate a projective point
 */
__device__ __forceinline__ void g1_neg(G1Projective& result, const G1Projective& p) {
    result.X = p.X;
    field_neg(result.Y, p.Y);
    result.Z = p.Z;
}

/**
 * @brief Convert projective to affine using template method
 * 
 * NOTE: This function is kept for API compatibility but delegates to
 * the template Projective::to_affine() method which implements
 * Jacobian to affine conversion: x = X/Z², y = Y/Z³
 */
__device__ __forceinline__ void g1_to_affine(G1Affine& result, const G1Projective& p) {
    result = p.to_affine();
}

// =============================================================================
// G2 Point Operations
// =============================================================================

/**
 * @brief Double a G2 projective point: result = 2 * P (Constant-Time)
 * 
 * Uses the doubling formula for y^2 = x^3 + b where b = 4(1+i) for G2.
 * Cost: 4S + 3M + 8add + 4sub (in Fq2)
 * 
 * NOTE: This function is safe for in-place operation (result == p)
 * Uses constant-time selection for identity handling to prevent
 *           timing side-channels.
 */
__device__ __forceinline__ void g2_double(G2Projective& result, const G2Projective& p) {
    // Pre-compute identity condition (will use cmov at end)
    int is_identity = p.is_identity() ? 1 : 0;
    
    // Save original p for identity case (handles aliasing when result == p)
    G2Projective saved_p = p;

    Fq2 t0, t1, t2, t3;
    
    // t0 = Y^2
    field_sqr(t0, p.Y);
    
    // t1 = 4 * X * Y^2
    field_mul(t1, p.X, t0);
    field_add(t1, t1, t1);
    field_add(t1, t1, t1);
    
    // t2 = 8 * Y^4
    field_sqr(t2, t0);
    field_add(t2, t2, t2);
    field_add(t2, t2, t2);
    field_add(t2, t2, t2);
    
    // t3 = 3 * X^2 (for a = 0)
    field_sqr(t3, p.X);
    Fq2 t3_2;
    field_add(t3_2, t3, t3);
    field_add(t3, t3_2, t3);
    
    // Compute all outputs to temporaries first (for in-place safety)
    Fq2 new_X, new_Y, new_Z;
    
    // X3 = t3^2 - 2*t1
    field_sqr(new_X, t3);
    field_sub(new_X, new_X, t1);
    field_sub(new_X, new_X, t1);
    
    // Y3 = t3 * (t1 - X3) - t2
    field_sub(t0, t1, new_X);
    field_mul(new_Y, t3, t0);
    field_sub(new_Y, new_Y, t2);
    
    // Z3 = 2 * Y * Z (must use original p.Y and p.Z!)
    field_mul(new_Z, p.Y, p.Z);
    field_add(new_Z, new_Z, new_Z);
    
    // Computed result
    G2Projective computed;
    computed.X = new_X;
    computed.Y = new_Y;
    computed.Z = new_Z;
    
    // Constant-time selection: if identity, return saved_p (which is identity)
    // Note: We use saved_p instead of G2Projective::identity() to handle
    // the case where result == p and we've already computed into p
    g2_cmov(result, saved_p, computed, is_identity);
}

/**
 * @brief Add two G2 projective points: result = P + Q (Constant-Time)
 * 
 * Uses complete addition formula with constant-time edge case handling.
 * Computes all possible results and uses cmov to select the correct one.
 * 
 * NOTE: This function is safe for in-place operation (result == p or result == q)
 * 
 * No branches that depend on point values. All edge cases handled
 *           via constant-time selection (cmov) to prevent timing side-channels.
 */
__device__ __forceinline__ void g2_add(G2Projective& result, const G2Projective& p, const G2Projective& q) {
    // Pre-compute identity conditions
    int p_is_identity = p.is_identity() ? 1 : 0;
    int q_is_identity = q.is_identity() ? 1 : 0;
    
    // CRITICAL: Save original inputs FIRST to handle aliasing (result == p or result == q)
    // These copies are used in the final cmov selection
    G2Projective saved_p = p;
    G2Projective saved_q = q;

    Fq2 z1z1, z2z2, u1, u2, s1, s2, h, i, j, r, v;
    
    // Z1Z1 = Z1^2
    field_sqr(z1z1, p.Z);
    // Z2Z2 = Z2^2
    field_sqr(z2z2, q.Z);
    
    // U1 = X1 * Z2Z2
    field_mul(u1, p.X, z2z2);
    // U2 = X2 * Z1Z1
    field_mul(u2, q.X, z1z1);
    
    // S1 = Y1 * Z2 * Z2Z2
    field_mul(s1, p.Y, q.Z);
    field_mul(s1, s1, z2z2);
    // S2 = Y2 * Z1 * Z1Z1
    field_mul(s2, q.Y, p.Z);
    field_mul(s2, s2, z1z1);
    
    // H = U2 - U1
    field_sub(h, u2, u1);
    
    // Compute edge case conditions (constant-time)
    int h_is_zero = h.is_zero() ? 1 : 0;
    Fq2 s_diff;
    field_sub(s_diff, s2, s1);
    int s_diff_is_zero = s_diff.is_zero() ? 1 : 0;
    
    // I = (2*H)^2
    field_add(i, h, h);
    field_sqr(i, i);
    
    // J = H * I
    field_mul(j, h, i);
    
    // r = 2 * (S2 - S1)
    field_sub(r, s2, s1);
    field_add(r, r, r);
    
    // V = U1 * I
    field_mul(v, u1, i);
    
    // Save values that will be needed after we start writing to result
    // (for in-place safety)
    Fq2 saved_Z1 = p.Z;
    Fq2 saved_Z2 = q.Z;
    Fq2 saved_S1 = s1;
    
    // Compute to temporaries
    Fq2 new_X, new_Y, new_Z;
    
    // X3 = r^2 - J - 2*V
    field_sqr(new_X, r);
    field_sub(new_X, new_X, j);
    field_sub(new_X, new_X, v);
    field_sub(new_X, new_X, v);
    
    // Y3 = r * (V - X3) - 2*S1*J
    field_sub(v, v, new_X);
    field_mul(new_Y, r, v);
    field_mul(saved_S1, saved_S1, j);
    field_add(saved_S1, saved_S1, saved_S1);
    field_sub(new_Y, new_Y, saved_S1);
    
    // Z3 = ((Z1 + Z2)^2 - Z1Z1 - Z2Z2) * H
    field_add(new_Z, saved_Z1, saved_Z2);
    field_sqr(new_Z, new_Z);
    field_sub(new_Z, new_Z, z1z1);
    field_sub(new_Z, new_Z, z2z2);
    field_mul(new_Z, new_Z, h);
    
    // Standard addition result
    G2Projective add_result;
    add_result.X = new_X;
    add_result.Y = new_Y;
    add_result.Z = new_Z;
    
    // Compute doubled result (for when P == Q)
    // Use saved_p since p might be aliased with result
    G2Projective doubled;
    g2_double(doubled, saved_p);
    
    // Identity point (for when P == -Q)
    G2Projective identity_point = G2Projective::identity();
    
    // Constant-time selection of result:
    // 1. Start with standard addition result
    // 2. If h == 0 and s_diff == 0: P == Q, use doubled
    // 3. If h == 0 and s_diff != 0: P == -Q, use identity
    // 4. If p is identity: return q (use saved_q)
    // 5. If q is identity: return p (use saved_p)
    
    result = add_result;
    
    int use_doubled = h_is_zero & s_diff_is_zero;
    int use_identity = h_is_zero & (!s_diff_is_zero ? 1 : 0);
    
    g2_cmov(result, doubled, result, use_doubled);
    g2_cmov(result, identity_point, result, use_identity);
    
    // Handle input identity cases (MUST use saved copies due to aliasing)
    g2_cmov(result, saved_q, result, p_is_identity);
    g2_cmov(result, saved_p, result, q_is_identity);
}

/**
 * @brief Mixed addition for G2: result = P + Q where Q is affine (Constant-Time)
 * 
 * More efficient than general addition when one point is affine.
 * NOTE: This function is safe for in-place operation (result == p)
 * 
 * Uses constant-time selection for all edge cases to prevent
 *           timing side-channels.
 */
__device__ __forceinline__ void g2_add_mixed(G2Projective& result, const G2Projective& p, const G2Affine& q) {
    // Pre-compute identity conditions
    int p_is_identity = p.is_identity() ? 1 : 0;
    int q_is_identity = q.is_identity() ? 1 : 0;
    
    // CRITICAL: Save original p FIRST to handle aliasing (result == p)
    // This copy is used in the final cmov selection
    G2Projective saved_p = p;

    Fq2 z1z1, u2, s2, h, hh, i, j, r, v;
    
    // Z1Z1 = Z1^2
    field_sqr(z1z1, p.Z);
    
    // U2 = X2 * Z1Z1 (U1 = X1)
    field_mul(u2, q.x, z1z1);
    
    // S2 = Y2 * Z1 * Z1Z1 (S1 = Y1)
    field_mul(s2, q.y, p.Z);
    field_mul(s2, s2, z1z1);
    
    // H = U2 - X1
    field_sub(h, u2, p.X);
    
    // Compute edge case conditions (constant-time)
    int h_is_zero = h.is_zero() ? 1 : 0;
    Fq2 s_diff;
    field_sub(s_diff, s2, p.Y);
    int s_diff_is_zero = s_diff.is_zero() ? 1 : 0;
    
    // HH = H^2
    field_sqr(hh, h);
    
    // I = 4 * HH
    field_add(i, hh, hh);
    field_add(i, i, i);
    
    // J = H * I
    field_mul(j, h, i);
    
    // r = 2 * (S2 - Y1)
    field_sub(r, s2, p.Y);
    field_add(r, r, r);
    
    // V = X1 * I
    field_mul(v, p.X, i);
    
    // Save p.Y and p.Z before we potentially overwrite (for in-place safety)
    Fq2 saved_Y = p.Y;
    Fq2 saved_Z = p.Z;
    
    // Compute to temporaries for in-place safety
    Fq2 new_X, new_Y, new_Z;
    
    // X3 = r^2 - J - 2*V
    field_sqr(new_X, r);
    field_sub(new_X, new_X, j);
    field_sub(new_X, new_X, v);
    field_sub(new_X, new_X, v);
    
    // Y3 = r * (V - X3) - 2*Y1*J
    Fq2 t;
    field_sub(t, v, new_X);
    field_mul(new_Y, r, t);
    field_mul(t, saved_Y, j);
    field_add(t, t, t);
    field_sub(new_Y, new_Y, t);
    
    // Z3 = (Z1 + H)^2 - Z1Z1 - HH
    field_add(new_Z, saved_Z, h);
    field_sqr(new_Z, new_Z);
    field_sub(new_Z, new_Z, z1z1);
    field_sub(new_Z, new_Z, hh);
    
    // Standard addition result
    G2Projective add_result;
    add_result.X = new_X;
    add_result.Y = new_Y;
    add_result.Z = new_Z;
    
    // Compute doubled result (for when P == Q)
    // Use saved_p since p might be aliased with result
    G2Projective doubled;
    g2_double(doubled, saved_p);
    
    // Identity point (for when P == -Q)
    G2Projective identity_point = G2Projective::identity();
    
    // Convert q to projective (for when p is identity)
    G2Projective q_proj = G2Projective::from_affine(q);
    
    // Constant-time selection of result:
    // 1. Start with standard addition result
    // 2. If h == 0 and s_diff == 0: P == Q, use doubled
    // 3. If h == 0 and s_diff != 0: P == -Q, use identity
    // 4. If p is identity: return q_proj
    // 5. If q is identity: return p (use saved_p)
    
    result = add_result;
    
    int use_doubled = h_is_zero & s_diff_is_zero;
    int use_identity = h_is_zero & (!s_diff_is_zero ? 1 : 0);
    
    g2_cmov(result, doubled, result, use_doubled);
    g2_cmov(result, identity_point, result, use_identity);
    
    // Handle input identity cases (MUST use saved_p due to aliasing)
    g2_cmov(result, q_proj, result, p_is_identity);
    g2_cmov(result, saved_p, result, q_is_identity);
}

/**
 * @brief Negate a G2 projective point
 */
__device__ __forceinline__ void g2_neg(G2Projective& result, const G2Projective& p) {
    result.X = p.X;
    field_neg(result.Y, p.Y);
    result.Z = p.Z;
}

/**
 * @brief Convert G2 projective to affine using template method
 * 
 * NOTE: This function implements Jacobian to affine conversion: x = X/Z², y = Y/Z³
 * Previously this used direct fq2_* calls, but now works via field_* overloads.
 */
__device__ __forceinline__ void g2_to_affine(G2Affine& result, const G2Projective& p) {
    result = p.to_affine();
}

} // namespace bls12_381
