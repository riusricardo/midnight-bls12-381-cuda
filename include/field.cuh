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
 * @file field.cuh
 * @brief BLS12-381 Field Element Types and Arithmetic
 * 
 * Implements both Fq (base field, 381-bit) and Fr (scalar field, 255-bit)
 * using Montgomery representation for efficient modular arithmetic.
 * 
 * ARCHITECTURE:
 * =============
 * This header defines:
 * 
 * 1. Field Configuration Traits (fp_config, fq_config):
 *    - Constants as inline switch functions (CUDA device-compatible)
 *    - Modulus, R, R², INV for Montgomery operations
 * 
 * 2. Field<Config> Template Class:
 *    - Limb storage (4×64-bit for Fr, 6×64-bit for Fq)
 *    - Montgomery arithmetic operators (+, -, *, inverse)
 *    - Helper methods (one, zero, from_int)
 * 
 * 3. Inline __device__ Functions:
 *    - Montgomery multiplication/reduction
 *    - Field inversion via Fermat's little theorem
 *    - Helper operations (neg, add, sub, mul)
 * 
 * Type Aliases:
 * - Fr = Field<fp_config>: Scalar field (255-bit, for scalars/exponents)
 * - Fq = Field<fq_config>: Base field (381-bit, for curve coordinates)
 * 
 * CUDA Compatibility:
 * ===================
 * All functions are marked __host__ __device__ for use on both CPU and GPU.
 * Constants use switch statements instead of constexpr arrays for device compatibility.
 */

#pragma once

#include "bls12_381_params.cuh"
#include <cuda_runtime.h>

// Portable unroll pragma - only effective in CUDA device code
#ifdef __CUDA_ARCH__
  #define UNROLL_LOOP _Pragma("unroll")
#else
  #define UNROLL_LOOP
#endif

// =============================================================================
// Forward declarations - Field is in global namespace (like ICICLE)
// =============================================================================
template<typename Config> struct Field;

namespace bls12_381 {

// =============================================================================
// Forward declarations for config types
// =============================================================================

struct fp_config;
struct fq_config;

// =============================================================================
// Field configuration traits - using inline device functions for constants
// =============================================================================

// Use switch statements to avoid constexpr array issues in device code
struct fp_config {
    static constexpr int LIMBS = FR_LIMBS_64;
    static constexpr int NBITS = 255;
    static constexpr uint64_t INV = FR_INV;
    
    // Use switch for device compatibility - each case is a compile-time constant
    __host__ __device__ __forceinline__ static uint64_t modulus(int i) {
        switch(i) {
            case 0: return FR_MODULUS_L0;
            case 1: return FR_MODULUS_L1;
            case 2: return FR_MODULUS_L2;
            case 3: return FR_MODULUS_L3;
            default: return 0;
        }
    }
    
    __host__ __device__ __forceinline__ static uint64_t one(int i) {
        switch(i) {
            case 0: return FR_ONE_L0;
            case 1: return FR_ONE_L1;
            case 2: return FR_ONE_L2;
            case 3: return FR_ONE_L3;
            default: return 0;
        }
    }
    
    __host__ __device__ __forceinline__ static uint64_t r2(int i) {
        switch(i) {
            case 0: return FR_R2_L0;
            case 1: return FR_R2_L1;
            case 2: return FR_R2_L2;
            case 3: return FR_R2_L3;
            default: return 0;
        }
    }
    
    __host__ static uint64_t modulus_host(int i) {
        return FR_MODULUS_HOST[i];
    }
    
    __host__ static uint64_t one_host(int i) {
        return FR_ONE_HOST[i];
    }
    
    __host__ static uint64_t r2_host(int i) {
        return FR_R2_HOST[i];
    }
};

struct fq_config {
    static constexpr int LIMBS = FP_LIMBS_64;
    static constexpr int NBITS = 381;
    static constexpr uint64_t INV = FQ_INV;
    
    __host__ __device__ __forceinline__ static uint64_t modulus(int i) {
        switch(i) {
            case 0: return FQ_MODULUS_L0;
            case 1: return FQ_MODULUS_L1;
            case 2: return FQ_MODULUS_L2;
            case 3: return FQ_MODULUS_L3;
            case 4: return FQ_MODULUS_L4;
            case 5: return FQ_MODULUS_L5;
            default: return 0;
        }
    }
    
    __host__ __device__ __forceinline__ static uint64_t one(int i) {
        switch(i) {
            case 0: return FQ_ONE_L0;
            case 1: return FQ_ONE_L1;
            case 2: return FQ_ONE_L2;
            case 3: return FQ_ONE_L3;
            case 4: return FQ_ONE_L4;
            case 5: return FQ_ONE_L5;
            default: return 0;
        }
    }
    
    __host__ __device__ __forceinline__ static uint64_t r2(int i) {
        switch(i) {
            case 0: return FQ_R2_L0;
            case 1: return FQ_R2_L1;
            case 2: return FQ_R2_L2;
            case 3: return FQ_R2_L3;
            case 4: return FQ_R2_L4;
            case 5: return FQ_R2_L5;
            default: return 0;
        }
    }
    
    __host__ static uint64_t modulus_host(int i) {
        return FQ_MODULUS_HOST[i];
    }
    
    __host__ static uint64_t one_host(int i) {
        return FQ_ONE_HOST[i];
    }
    
    __host__ static uint64_t r2_host(int i) {
        return FQ_R2_HOST[i];
    }
};

} // namespace bls12_381

// =============================================================================
// Field element storage - in global namespace to match ICICLE ABI
// =============================================================================

template<typename Config>
struct Field {
    static constexpr int LIMBS = Config::LIMBS;
    uint64_t limbs[LIMBS];

    // Default constructor (zero)
    __host__ __device__ Field() {
        UNROLL_LOOP
        for (int i = 0; i < LIMBS; i++) {
            limbs[i] = 0;
        }
    }

    // Constructor from limbs array
    __host__ __device__ explicit Field(const uint64_t* data) {
        UNROLL_LOOP
        for (int i = 0; i < LIMBS; i++) {
            limbs[i] = data[i];
        }
    }

    // Zero element
    __host__ __device__ static Field zero() {
        return Field();
    }

    /**
     * @brief Returns the multiplicative identity (1) in Montgomery form
     * 
     * Uses __CUDA_ARCH__ to detect at compile time whether we're on host or device:
     * - On device: reads from __constant__ memory (FQ_ONE, FR_ONE)
     * - On host: reads from constexpr arrays (FQ_ONE_HOST, FR_ONE_HOST)
     */
    __host__ __device__ static Field one() {
        Field result;
        #ifdef __CUDA_ARCH__
        // Device code path - read from GPU constant memory
        UNROLL_LOOP
        for (int i = 0; i < LIMBS; i++) {
            result.limbs[i] = Config::one(i);
        }
        #else
        // Host code path - read from CPU constexpr arrays
        for (int i = 0; i < LIMBS; i++) {
            result.limbs[i] = Config::one_host(i);
        }
        #endif
        return result;
    }
    
    // One element - explicit host version (kept for backward compatibility)
    __host__ static Field one_host() {
        Field result;
        for (int i = 0; i < LIMBS; i++) {
            result.limbs[i] = Config::one_host(i);
        }
        return result;
    }
    
    /**
     * @brief Returns R^2 mod p for Montgomery conversion
     * 
     * Used to convert standard integers to Montgomery form:
     * to_mont(a) = a * R^2 * R^{-1} mod p = a * R mod p
     */
    __host__ __device__ static Field R_SQUARED() {
        Field result;
        #ifdef __CUDA_ARCH__
        UNROLL_LOOP
        for (int i = 0; i < LIMBS; i++) {
            result.limbs[i] = Config::r2(i);
        }
        #else
        for (int i = 0; i < LIMBS; i++) {
            result.limbs[i] = Config::r2_host(i);
        }
        #endif
        return result;
    }
    
    /**
     * @brief Create field element from integer
     * 
     * Converts a small integer to Montgomery form: result = val * R mod p
     */
    __host__ __device__ static Field from_int(uint64_t val) {
        Field result;
        result.limbs[0] = val;
        for (int i = 1; i < LIMBS; i++) {
            result.limbs[i] = 0;
        }
        // Convert to Montgomery form by multiplying by R^2
        // Note: This requires field_mul to be available on host
        // For host usage, prefer using the host-side Montgomery conversion
        #ifdef __CUDA_ARCH__
        return result * R_SQUARED();
        #else
        // On host, we cannot use operator* (which calls device field_mul)
        // Return raw value - caller should use host-side conversion
        // This is a limitation; for production, implement host-side Montgomery mul
        return result;
        #endif
    }

    // Check if zero
    __host__ __device__ bool is_zero() const {
        uint64_t acc = 0;
        UNROLL_LOOP
        for (int i = 0; i < LIMBS; i++) {
            acc |= limbs[i];
        }
        return acc == 0;
    }

    // Equality
    __host__ __device__ bool operator==(const Field& other) const {
        UNROLL_LOOP
        for (int i = 0; i < LIMBS; i++) {
            if (limbs[i] != other.limbs[i]) return false;
        }
        return true;
    }

    __host__ __device__ bool operator!=(const Field& other) const {
        return !(*this == other);
    }
};

// =============================================================================
// BLS12-381 Type Aliases - inside namespace for compatibility
// =============================================================================

namespace bls12_381 {

// Type aliases - Field is in global namespace, aliases are in bls12_381
using Fr = ::Field<fp_config>;  // Scalar field
using Fq = ::Field<fq_config>;  // Base field

// For ICICLE compatibility: scalar_t is the standard name
using scalar_t = Fr;

} // namespace bls12_381

// =============================================================================
// Constant-Time Selection Helpers (for side-channel resistance)
// =============================================================================

/**
 * @brief Constant-time conditional selection: result = cond ? a : b
 * 
 * This function executes in constant time regardless of the value
 * of `cond`. It prevents timing side-channels by avoiding branches.
 * 
 * @param result Output field element
 * @param a Value selected when cond is true (non-zero)
 * @param b Value selected when cond is false (zero)
 * @param cond Condition (0 = select b, non-zero = select a)
 */
template<typename Config>
__device__ __forceinline__ void field_cmov(
    Field<Config>& result,
    const Field<Config>& a,
    const Field<Config>& b,
    int cond
) {
    constexpr int LIMBS = Config::LIMBS;
    
    // Convert condition to all-ones or all-zeros mask
    uint64_t mask = (uint64_t)(-(int64_t)(cond != 0));
    
    UNROLL_LOOP
    for (int i = 0; i < LIMBS; i++) {
        // result[i] = (mask & a[i]) | (~mask & b[i])
        result.limbs[i] = (mask & a.limbs[i]) | (~mask & b.limbs[i]);
    }
}

// =============================================================================
// Montgomery arithmetic - Device functions
// =============================================================================

/**
 * @brief Add two field elements: result = a + b mod p
 * 
 * This function is constant-time. Both the unreduced and
 * reduced results are always computed, and the final selection uses
 * field_cmov to avoid data-dependent branches.
 */
template<typename Config>
__device__ __forceinline__ void field_add(
    Field<Config>& result,
    const Field<Config>& a,
    const Field<Config>& b
) {
    constexpr int LIMBS = Config::LIMBS;
    
    uint64_t carry = 0;
    uint64_t temp[LIMBS];
    
    // First addition: a + b
    UNROLL_LOOP
    for (int i = 0; i < LIMBS; i++) {
        uint64_t ai = a.limbs[i];
        uint64_t bi = b.limbs[i];
        uint64_t sum = ai + bi;
        uint64_t new_carry = (sum < ai) ? 1ULL : 0ULL;
        sum += carry;
        new_carry += (sum < carry) ? 1ULL : 0ULL;
        temp[i] = sum;
        carry = new_carry;
    }
    
    // Conditional subtraction if >= modulus
    uint64_t borrow = 0;
    uint64_t reduced[LIMBS];
    
    UNROLL_LOOP
    for (int i = 0; i < LIMBS; i++) {
        uint64_t mod_i = Config::modulus(i);
        uint64_t t = temp[i];
        uint64_t diff = t - mod_i;
        uint64_t new_borrow = (t < mod_i) ? 1ULL : 0ULL;
        uint64_t diff2 = diff - borrow;
        new_borrow += (diff < borrow) ? 1ULL : 0ULL;
        reduced[i] = diff2;
        borrow = new_borrow;
    }
    
    // Constant-time selection using field_cmov
    // Select reduced if carry set or no borrow (result >= modulus)
    Field<Config> temp_field, reduced_field;
    UNROLL_LOOP
    for (int i = 0; i < LIMBS; i++) {
        temp_field.limbs[i] = temp[i];
        reduced_field.limbs[i] = reduced[i];
    }
    
    int use_reduced = ((carry != 0) || (borrow == 0)) ? 1 : 0;
    field_cmov(result, reduced_field, temp_field, use_reduced);
}

/**
 * @brief Subtract two field elements: result = a - b mod p
 * 
 * This function is constant-time. Both the uncorrected and
 * corrected results are always computed, and the final selection uses
 * field_cmov to avoid data-dependent branches.
 */
template<typename Config>
__device__ __forceinline__ void field_sub(
    Field<Config>& result,
    const Field<Config>& a,
    const Field<Config>& b
) {
    constexpr int LIMBS = Config::LIMBS;
    
    uint64_t borrow = 0;
    uint64_t temp[LIMBS];
    
    // Subtraction: a - b
    UNROLL_LOOP
    for (int i = 0; i < LIMBS; i++) {
        uint64_t ai = a.limbs[i];
        uint64_t bi = b.limbs[i];
        uint64_t diff = ai - bi;
        uint64_t new_borrow = (ai < bi) ? 1ULL : 0ULL;
        uint64_t diff2 = diff - borrow;
        new_borrow += (diff < borrow) ? 1ULL : 0ULL;
        temp[i] = diff2;
        borrow = new_borrow;
    }
    
    // Always compute corrected version (add modulus)
    // This ensures constant-time execution regardless of borrow value
    uint64_t corrected[LIMBS];
    uint64_t carry = 0;
    UNROLL_LOOP
    for (int i = 0; i < LIMBS; i++) {
        uint64_t t = temp[i];
        uint64_t mod_i = Config::modulus(i);
        uint64_t sum = t + mod_i;
        uint64_t new_carry = (sum < t) ? 1ULL : 0ULL;
        sum += carry;
        new_carry += (sum < carry) ? 1ULL : 0ULL;
        corrected[i] = sum;
        carry = new_carry;
    }
    
    // Constant-time selection using field_cmov
    // Select corrected if borrow occurred, else use temp
    Field<Config> temp_field, corrected_field;
    UNROLL_LOOP
    for (int i = 0; i < LIMBS; i++) {
        temp_field.limbs[i] = temp[i];
        corrected_field.limbs[i] = corrected[i];
    }
    
    int needs_correction = (borrow != 0) ? 1 : 0;
    field_cmov(result, corrected_field, temp_field, needs_correction);
}

/**
 * @brief Montgomery multiplication: result = a * b * R^{-1} mod p
 * 
 * Uses the CIOS (Coarsely Integrated Operand Scanning) algorithm
 * optimized for GPU execution.
 * 
 * Final reduction uses field_cmov for constant-time selection.
 */
template<typename Config>
__device__ __forceinline__ void field_mul(
    Field<Config>& result,
    const Field<Config>& a,
    const Field<Config>& b
) {
    constexpr int LIMBS = Config::LIMBS;
    const uint64_t inv = Config::INV;
    
    uint64_t t[LIMBS + 2] = {0};
    
    UNROLL_LOOP
    for (int i = 0; i < LIMBS; i++) {
        // Multiply accumulate: t += a[i] * b
        uint64_t carry = 0;
        UNROLL_LOOP
        for (int j = 0; j < LIMBS; j++) {
            // t[j] += a[i] * b[j] + carry
            unsigned __int128 prod = (unsigned __int128)a.limbs[i] * b.limbs[j];
            prod += t[j];
            prod += carry;
            t[j] = (uint64_t)prod;
            carry = (uint64_t)(prod >> 64);
        }
        t[LIMBS] += carry;
        t[LIMBS + 1] = (t[LIMBS] < carry) ? 1 : 0;
        
        // Montgomery reduction step
        uint64_t m = t[0] * inv;
        
        carry = 0;
        UNROLL_LOOP
        for (int j = 0; j < LIMBS; j++) {
            unsigned __int128 prod = (unsigned __int128)m * Config::modulus(j);
            prod += t[j];
            prod += carry;
            if (j > 0) t[j - 1] = (uint64_t)prod;
            carry = (uint64_t)(prod >> 64);
        }
        t[LIMBS - 1] = t[LIMBS] + carry;
        t[LIMBS] = t[LIMBS + 1] + ((t[LIMBS - 1] < carry) ? 1 : 0);
        t[LIMBS + 1] = 0;
    }
    
    // Final reduction
    uint64_t borrow = 0;
    uint64_t reduced[LIMBS];
    
    UNROLL_LOOP
    for (int i = 0; i < LIMBS; i++) {
        uint64_t mod_i = Config::modulus(i);
        uint64_t diff = t[i] - mod_i - borrow;
        borrow = (t[i] < mod_i + borrow) ? 1 : 0;
        reduced[i] = diff;
    }
    
    // Constant-time selection using field_cmov
    // Select reduced if overflow or no borrow (result >= modulus)
    Field<Config> unreduced_field, reduced_field;
    UNROLL_LOOP
    for (int i = 0; i < LIMBS; i++) {
        unreduced_field.limbs[i] = t[i];
        reduced_field.limbs[i] = reduced[i];
    }
    
    int use_reduced = ((t[LIMBS] != 0) || (borrow == 0)) ? 1 : 0;
    field_cmov(result, reduced_field, unreduced_field, use_reduced);
}

/**
 * @brief Montgomery squaring: result = a^2 * R^{-1} mod p
 * 
 * Optimized squaring exploiting a[i]*a[j] = a[j]*a[i] symmetry.
 * Saves ~40% compared to general multiplication.
 * 
 * Final reduction uses field_cmov for constant-time selection.
 */
template<typename Config>
__device__ __forceinline__ void field_sqr(
    Field<Config>& result,
    const Field<Config>& a
) {
    constexpr int LIMBS = Config::LIMBS;
    const uint64_t inv = Config::INV;
    
    uint64_t t[2 * LIMBS] = {0};
    
    // Step 1: Compute off-diagonal products (doubled due to symmetry)
    UNROLL_LOOP
    for (int i = 0; i < LIMBS; i++) {
        uint64_t carry = 0;
        UNROLL_LOOP
        for (int j = i + 1; j < LIMBS; j++) {
            unsigned __int128 prod = (unsigned __int128)a.limbs[i] * a.limbs[j];
            prod += t[i + j];
            prod += carry;
            t[i + j] = (uint64_t)prod;
            carry = (uint64_t)(prod >> 64);
        }
        t[i + LIMBS] += carry;
    }
    
    // Step 2: Double the off-diagonal terms
    uint64_t carry = 0;
    UNROLL_LOOP
    for (int i = 1; i < 2 * LIMBS - 1; i++) {
        uint64_t val = t[i];
        t[i] = (val << 1) | carry;
        carry = val >> 63;
    }
    t[2 * LIMBS - 1] = (t[2 * LIMBS - 1] << 1) | carry;
    
    // Step 3: Add diagonal terms a[i]^2
    carry = 0;
    UNROLL_LOOP
    for (int i = 0; i < LIMBS; i++) {
        unsigned __int128 sq = (unsigned __int128)a.limbs[i] * a.limbs[i];
        sq += t[2 * i];
        sq += carry;
        t[2 * i] = (uint64_t)sq;
        
        unsigned __int128 high = (sq >> 64) + t[2 * i + 1];
        t[2 * i + 1] = (uint64_t)high;
        carry = (uint64_t)(high >> 64);
    }
    
    // Step 4: Montgomery reduction
    UNROLL_LOOP
    for (int i = 0; i < LIMBS; i++) {
        uint64_t m = t[i] * inv;
        
        uint64_t red_carry = 0;
        UNROLL_LOOP
        for (int j = 0; j < LIMBS; j++) {
            unsigned __int128 prod = (unsigned __int128)m * Config::modulus(j);
            prod += t[i + j];
            prod += red_carry;
            t[i + j] = (uint64_t)prod;
            red_carry = (uint64_t)(prod >> 64);
        }
        
        // Propagate carry
        for (int j = i + LIMBS; j < 2 * LIMBS && red_carry; j++) {
            uint64_t sum = t[j] + red_carry;
            red_carry = (sum < t[j]) ? 1 : 0;
            t[j] = sum;
        }
    }
    
    // Step 5: Final reduction - result is in t[LIMBS..2*LIMBS-1]
    uint64_t borrow = 0;
    uint64_t reduced[LIMBS];
    
    UNROLL_LOOP
    for (int i = 0; i < LIMBS; i++) {
        uint64_t mod_i = Config::modulus(i);
        uint64_t val = t[i + LIMBS];
        uint64_t diff = val - mod_i;
        uint64_t new_borrow = (val < mod_i) ? 1ULL : 0ULL;
        uint64_t diff2 = diff - borrow;
        new_borrow += (diff < borrow) ? 1ULL : 0ULL;
        reduced[i] = diff2;
        borrow = new_borrow;
    }
    
    // Constant-time selection using field_cmov
    // Select reduced if no borrow (result >= modulus)
    Field<Config> unreduced_field, reduced_field;
    UNROLL_LOOP
    for (int i = 0; i < LIMBS; i++) {
        unreduced_field.limbs[i] = t[i + LIMBS];
        reduced_field.limbs[i] = reduced[i];
    }
    
    int use_reduced = (borrow == 0) ? 1 : 0;
    field_cmov(result, reduced_field, unreduced_field, use_reduced);
}

/**
 * @brief Negate field element: result = -a mod p
 * 
 * This function is constant-time. The negation is always
 * computed, and a constant-time selection chooses between the negated
 * value and zero based on whether the input was zero.
 */
template<typename Config>
__device__ __forceinline__ void field_neg(
    Field<Config>& result,
    const Field<Config>& a
) {
    constexpr int LIMBS = Config::LIMBS;
    
    // Always compute negation (p - a)
    Field<Config> negated;
    uint64_t borrow = 0;
    UNROLL_LOOP
    for (int i = 0; i < LIMBS; i++) {
        uint64_t mod_i = Config::modulus(i);
        uint64_t ai = a.limbs[i];
        // Compute mod_i - ai - borrow carefully to avoid overflow issues
        uint64_t diff = mod_i - ai;
        uint64_t new_borrow = (mod_i < ai) ? 1ULL : 0ULL;
        uint64_t diff2 = diff - borrow;
        new_borrow += (diff < borrow) ? 1ULL : 0ULL;
        negated.limbs[i] = diff2;
        borrow = new_borrow;
    }
    
    // Constant-time check if input is zero
    // Accumulate OR of all limbs (constant-time, no early exit)
    uint64_t nonzero_acc = 0;
    UNROLL_LOOP
    for (int i = 0; i < LIMBS; i++) {
        nonzero_acc |= a.limbs[i];
    }
    
    // If input is zero, result should be zero; otherwise result is negated
    // is_zero = 1 if a == 0, else 0
    int is_zero = (nonzero_acc == 0) ? 1 : 0;
    
    // Constant-time selection using field_cmov
    // result = is_zero ? a : negated  (where a is zero if is_zero is true)
    field_cmov(result, a, negated, is_zero);
}

/**
 * @brief Modular inversion using constant-time exponentiation
 * 
 * Computes a^(-1) mod p using Fermat's little theorem: a^(-1) = a^(p-2) mod p.
 * 
 * This function is constant-time with respect to the input value.
 * All operations are performed regardless of the input, and selection between
 * results uses bitwise masking to avoid timing side-channels.
 * 
 * For 4-limb fields (Fr), uses windowed exponentiation with window size 4.
 * For 6-limb fields (Fq), uses constant-time binary exponentiation.
 * 
 * Note: The exponent (p-2) is public, so the number of iterations is fixed.
 * The security requirement is that timing does not depend on the INPUT value.
 */
template<typename Config>
__device__ void field_inv(
    Field<Config>& result,
    const Field<Config>& a
) {
    constexpr int LIMBS = Config::LIMBS;
    
    // Check for zero input in constant-time
    // (accumulate OR of all limbs, no early exit)
    uint64_t nonzero_acc = 0;
    UNROLL_LOOP
    for (int i = 0; i < LIMBS; i++) {
        nonzero_acc |= a.limbs[i];
    }
    int input_is_zero = (nonzero_acc == 0) ? 1 : 0;
    
    // We still compute the inversion even for zero input (constant-time)
    // and select zero result at the end if input was zero
    
    // Compute exponent = p - 2
    uint64_t exp[LIMBS];
    {
        uint64_t borrow = 0;
        uint64_t mod_0 = Config::modulus(0);
        exp[0] = mod_0 - 2;
        borrow = (mod_0 < 2) ? 1 : 0;
        
        UNROLL_LOOP
        for (int i = 1; i < LIMBS; i++) {
            uint64_t mod_i = Config::modulus(i);
            exp[i] = mod_i - borrow;
            borrow = (mod_i < borrow) ? 1 : 0;
        }
    }
    
    if constexpr (LIMBS == 4) {
        // =====================================================================
        // CONSTANT-TIME Windowed Exponentiation for Fr (4 limbs)
        // =====================================================================
        // Uses window size 4: precompute a^1, a^2, ..., a^15
        // Then process exponent in 4-bit windows
        
        Field<Config> x = a;
        
        // Build power table: powers[i] = a^i for i in [0, 15]
        Field<Config> powers[16];
        powers[0] = Field<Config>::one();
        powers[1] = x;
        
        // Compute powers 2-15 using squaring and multiplication
        field_sqr(powers[2], x);                    // x^2
        field_mul(powers[3], powers[2], x);         // x^3
        field_sqr(powers[4], powers[2]);            // x^4
        field_mul(powers[5], powers[4], x);         // x^5
        field_sqr(powers[6], powers[3]);            // x^6
        field_mul(powers[7], powers[6], x);         // x^7
        field_sqr(powers[8], powers[4]);            // x^8
        field_mul(powers[9], powers[8], x);         // x^9
        field_mul(powers[10], powers[9], x);        // x^10
        field_mul(powers[11], powers[10], x);       // x^11
        field_mul(powers[12], powers[11], x);       // x^12
        field_mul(powers[13], powers[12], x);       // x^13
        field_mul(powers[14], powers[13], x);       // x^14
        field_mul(powers[15], powers[14], x);       // x^15
        
        // Constant-time windowed exponentiation
        // Process all windows, always perform operations, use cmov for selection
        
        Field<Config> acc = Field<Config>::one();
        
        // For p-2 in Fr, the highest nibble of the highest limb is known
        // We can find the first non-zero nibble position at compile time
        // For Fr: p-2 = 0x73eda753299d7d483339d80809a1d80553bda402fffe5bfefffffffeffffffff
        // Highest limb is 0x73eda753299d7d48, highest nibble is 0x7
        
        // Process all 64 nibbles (16 nibbles per limb * 4 limbs)
        // The first few operations are effectively multiplying by 1, which is fine
        
        for (int limb = LIMBS - 1; limb >= 0; limb--) {
            for (int nibble = 15; nibble >= 0; nibble--) {
                // Always perform 4 squarings
                field_sqr(acc, acc);
                field_sqr(acc, acc);
                field_sqr(acc, acc);
                field_sqr(acc, acc);
                
                // Get current window value
                int window = (exp[limb] >> (nibble * 4)) & 0xF;
                
                // Constant-time table lookup using sequential cmov
                // This is O(16) per window, but guarantees constant-time
                Field<Config> power_to_use = powers[0];  // Default to 1
                
                UNROLL_LOOP
                for (int i = 1; i < 16; i++) {
                    int is_match = (window == i) ? 1 : 0;
                    field_cmov(power_to_use, powers[i], power_to_use, is_match);
                }
                
                // Always perform multiplication
                Field<Config> multiplied;
                field_mul(multiplied, acc, power_to_use);
                
                // Select multiplied result if window != 0, else keep acc
                // Note: multiplying by powers[0] = 1 gives acc, so this is safe
                // But we use explicit cmov for clarity and guaranteed constant-time
                int window_nonzero = (window != 0) ? 1 : 0;
                field_cmov(acc, multiplied, acc, window_nonzero);
            }
        }
        
        result = acc;
        
    } else {
        // =====================================================================
        // CONSTANT-TIME Binary Exponentiation for Fq (6 limbs)
        // =====================================================================
        // Process each bit, always perform both multiply and square
        
        Field<Config> base = a;
        Field<Config> acc = Field<Config>::one();
        
        for (int i = 0; i < LIMBS; i++) {
            for (int bit = 0; bit < 64; bit++) {
                // Get current bit
                int exp_bit = (exp[i] >> bit) & 1;
                
                // Always compute the multiplication
                Field<Config> multiplied;
                field_mul(multiplied, acc, base);
                
                // Constant-time selection based on exponent bit
                field_cmov(acc, multiplied, acc, exp_bit);
                
                // Always square the base
                field_sqr(base, base);
            }
        }
        
        result = acc;
    }
    
    // If input was zero, return zero (constant-time selection)
    Field<Config> zero_result = Field<Config>::zero();
    field_cmov(result, zero_result, result, input_is_zero);
    
    #ifdef ICICLE_DEBUG
    if (input_is_zero) {
        printf("WARNING: field_inv called with zero input, returning zero\n");
    }
    #endif
}

/**
 * @brief Convert from standard to Montgomery form: result = a * R mod p
 */
template<typename Config>
__device__ __forceinline__ void field_to_montgomery(
    Field<Config>& result,
    const Field<Config>& a
) {
    Field<Config> r2 = Field<Config>::R_SQUARED();
    field_mul(result, a, r2);
}

/**
 * @brief Convert from Montgomery to standard form: result = a * R^{-1} mod p
 */
template<typename Config>
__device__ __forceinline__ void field_from_montgomery(
    Field<Config>& result,
    const Field<Config>& a
) {
    Field<Config> one_raw;
    one_raw.limbs[0] = 1;
    for (int i = 1; i < Config::LIMBS; i++) {
        one_raw.limbs[i] = 0;
    }
    field_mul(result, a, one_raw);
}

// =============================================================================
// Convenience wrapper functions (return value versions)
// =============================================================================

/**
 * @brief Negate field element (returns value)
 */
template<typename Config>
__device__ __forceinline__ Field<Config> field_neg(const Field<Config>& a) {
    Field<Config> result;
    field_neg(result, a);
    return result;
}

/**
 * @brief Invert field element (returns value)
 */
template<typename Config>
__device__ __forceinline__ Field<Config> field_inv(const Field<Config>& a) {
    Field<Config> result;
    field_inv(result, a);
    return result;
}

// =============================================================================
// Operator overloads for cleaner syntax
// =============================================================================

template<typename Config>
__device__ __forceinline__ Field<Config> operator+(
    const Field<Config>& a,
    const Field<Config>& b
) {
    Field<Config> result;
    field_add(result, a, b);
    return result;
}

template<typename Config>
__device__ __forceinline__ Field<Config> operator-(
    const Field<Config>& a,
    const Field<Config>& b
) {
    Field<Config> result;
    field_sub(result, a, b);
    return result;
}

template<typename Config>
__device__ __forceinline__ Field<Config> operator*(
    const Field<Config>& a,
    const Field<Config>& b
) {
    Field<Config> result;
    field_mul(result, a, b);
    return result;
}

template<typename Config>
__device__ __forceinline__ Field<Config> operator-(const Field<Config>& a) {
    Field<Config> result;
    field_neg(result, a);
    return result;
}

// End of field.cuh - helper functions are in global namespace
