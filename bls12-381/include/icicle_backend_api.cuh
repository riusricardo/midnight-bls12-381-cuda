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
 * @file icicle_backend_api.cuh
 * @brief Icicle backend registration API
 * 
 * This file provides the registration functions and macros needed to
 * register our CUDA backend implementations with Icicle's dispatcher.
 * 
 * COVERAGE:
 * =========
 * - NTT (Number Theoretic Transform) and domain management
 * - Vector operations (add, sub, mul, sum)
 * - MSM (Multi-Scalar Multiplication) for G1 and G2
 * 
 * The registration functions are resolved at runtime from ICICLE core libraries:
 * - libicicle_field_bls12_381.so  (NTT, VecOps)
 * - libicicle_curve_bls12_381.so  (MSM)
 * 
 * Type definitions are in icicle_types.cuh - this file only adds
 * registration-specific declarations.
 * 
 * IMPORTANT: The type declarations here must produce the EXACT same
 * C++ mangled symbols as Icicle's core library. This means we need
 * to use the same namespaces, template names, and struct names.
 * 
 * G1 vs G2 REGISTRATION:
 * ======================
 * G1 and G2 require separate registration functions because they operate
 * on different types:
 * - G1: Uses Fq base field, G1Affine/G1Projective points
 * - G2: Uses Fq2 extension field, G2Affine/G2Projective points
 * 
 * Icicle's dispatcher routes to the correct implementation based on
 * the registered device type and curve type.
 */

#pragma once

// Get all Icicle-compatible type definitions
#include "icicle_types.cuh"
#include <functional>
#include <string>

// =============================================================================
// Forward declarations matching Icicle's type names EXACTLY for symbol mangling
// =============================================================================
// These MUST match Icicle's internal type names for correct symbol resolution

namespace bls12_381 {
    struct fp_config;  // Scalar field config
    struct fq_config;  // Base field config
    struct G1;         // G1 curve marker
    struct G2;         // G2 curve marker
}

// Template declarations matching Icicle's templates
// These are forward declarations only - we use them for type signatures
template<typename Config> class Field;
template<typename BaseField> class Affine;
template<typename BaseField, typename ScalarField, typename Gen> class Projective;

// G2 extension field type (Fq2)
template<typename BaseConfig, typename BaseField> class ComplexExtensionField;

// =============================================================================
// Registration Function Declarations
// =============================================================================
// These are resolved at runtime from libicicle_curve_bls12_381.so
// The backend library has these as undefined symbols (U) that get
// resolved when loaded into a process that has the Icicle core loaded.
//
// CRITICAL: The function signatures MUST match Icicle's exactly for
// proper symbol name mangling. This includes:
// - Template instantiation types (Field<fp_config>, Affine<...>, etc.)
// - Namespace (icicle::)
// - Parameter order and types

namespace icicle {

// Type aliases matching Icicle's internal types for BLS12-381 G1
using icicle_scalar_t = Field<bls12_381::fp_config>;
using icicle_point_field_t = Field<bls12_381::fq_config>;
using icicle_affine_t = Affine<icicle_point_field_t>;
using icicle_projective_t = Projective<icicle_point_field_t, icicle_scalar_t, bls12_381::G1>;

// Type aliases for G2 (uses Fq2 extension field)
// Note: These may need adjustment based on actual Icicle symbol names
using icicle_g2_field_t = ComplexExtensionField<bls12_381::fq_config, icicle_point_field_t>;
using icicle_g2_affine_t = Affine<icicle_g2_field_t>;
using icicle_g2_projective_t = Projective<icicle_g2_field_t, icicle_scalar_t, bls12_381::G2>;

// =============================================================================
// NTT Registration (Field Operations)
// =============================================================================
// These are provided by libicicle_field_bls12_381.so

// NTT registration function types
using NttImpl = std::function<eIcicleError(
    const Device& device,
    const icicle_scalar_t* input,
    int size,
    NTTDir dir,
    const NTTConfig<icicle_scalar_t>& config,
    icicle_scalar_t* output)>;

using NttInitDomainImpl = std::function<eIcicleError(
    const Device& device,
    const icicle_scalar_t& primitive_root,
    const NTTInitDomainConfig& config)>;

using NttReleaseDomainImpl = std::function<eIcicleError(
    const Device& device,
    const icicle_scalar_t& phantom)>;

using NttGetRouFromDomainImpl = std::function<eIcicleError(
    const Device& device,
    uint64_t logn,
    icicle_scalar_t* rou)>;

// NTT registration functions - weak symbols provided by ICICLE core
__attribute__((weak)) void register_ntt(const std::string& deviceType, NttImpl impl);
__attribute__((weak)) void register_ntt_init_domain(const std::string& deviceType, NttInitDomainImpl impl);
__attribute__((weak)) void register_ntt_release_domain(const std::string& deviceType, NttReleaseDomainImpl impl);
__attribute__((weak)) void register_ntt_get_rou_from_domain(const std::string& deviceType, NttGetRouFromDomainImpl impl);

// =============================================================================
// Vector Operations Registration (Field Operations)
// =============================================================================
// These are provided by libicicle_field_bls12_381.so

// VecOps registration function types
using scalarVectorOpImpl = std::function<eIcicleError(
    const Device& device,
    const icicle_scalar_t* scalar_a,
    const icicle_scalar_t* vec_b,
    uint64_t size,
    const VecOpsConfig& config,
    icicle_scalar_t* output)>;

using VectorReduceOpImpl = std::function<eIcicleError(
    const Device& device,
    const icicle_scalar_t* vec_a,
    uint64_t size,
    const VecOpsConfig& config,
    icicle_scalar_t* output)>;

// VecOps registration functions - weak symbols provided by ICICLE core
__attribute__((weak)) void register_vector_add(const std::string& deviceType, scalarVectorOpImpl impl);
__attribute__((weak)) void register_vector_sub(const std::string& deviceType, scalarVectorOpImpl impl);
__attribute__((weak)) void register_vector_mul(const std::string& deviceType, scalarVectorOpImpl impl);
__attribute__((weak)) void register_scalar_mul_vec(const std::string& deviceType, scalarVectorOpImpl impl);
__attribute__((weak)) void register_scalar_add_vec(const std::string& deviceType, scalarVectorOpImpl impl);
__attribute__((weak)) void register_vector_sum(const std::string& deviceType, VectorReduceOpImpl impl);

// =============================================================================
// G1 MSM Registration
// =============================================================================

// MSM registration function types matching Icicle's exact signatures
using MsmImpl = std::function<eIcicleError(
    const Device& device,
    const icicle_scalar_t* scalars,
    const icicle_affine_t* bases,
    int msm_size,
    const MSMConfig& config,
    icicle_projective_t* results)>;

using MsmPreComputeImpl = std::function<eIcicleError(
    const Device& device,
    const icicle_affine_t* input_bases,
    int bases_size,
    const MSMConfig& config,
    icicle_affine_t* output_bases)>;

// G1 MSM registration functions - provided by ICICLE runtime (libicicle_curve_bls12_381.so)
// These are called at library load time when our backend is dlopen'd by ICICLE.
// Declared as weak symbols so they can be undefined when testing standalone.
__attribute__((weak)) void register_msm(const std::string& deviceType, MsmImpl impl);
__attribute__((weak)) void register_msm_precompute_bases(const std::string& deviceType, MsmPreComputeImpl impl);

// =============================================================================
// G2 MSM Registration
// =============================================================================

// G2 MSM registration function types
using MsmG2Impl = std::function<eIcicleError(
    const Device& device,
    const icicle_scalar_t* scalars,
    const icicle_g2_affine_t* bases,
    int msm_size,
    const MSMConfig& config,
    icicle_g2_projective_t* results)>;

using MsmG2PreComputeImpl = std::function<eIcicleError(
    const Device& device,
    const icicle_g2_affine_t* input_bases,
    int bases_size,
    const MSMConfig& config,
    icicle_g2_affine_t* output_bases)>;

// G2 registration functions - we provide our own implementation since
// ICICLE core doesn't export these (G2 support is behind G2_ENABLED flag)
// See g2_registry.cu for the implementation
void register_g2_msm(const std::string& deviceType, MsmG2Impl impl);
void register_g2_msm_precompute_bases(const std::string& deviceType, MsmG2PreComputeImpl impl);

} // namespace icicle

// =============================================================================
// Registration Macros
// =============================================================================
// These create static initializers that run when the library is loaded

// Unique variable name generator
#define ICICLE_CONCAT_INNER(a, b) a##b
#define ICICLE_CONCAT(a, b) ICICLE_CONCAT_INNER(a, b)
#define ICICLE_UNIQUE(prefix) ICICLE_CONCAT(prefix, __COUNTER__)

// G1 MSM Registration
// Check if registration functions are available (they're weak symbols)
#define REGISTER_MSM_BACKEND(DEVICE_TYPE, FUNC)                    \
    namespace {                                                     \
        static bool ICICLE_UNIQUE(_reg_msm_) = []() -> bool {      \
            if (icicle::register_msm) {                            \
                icicle::register_msm(DEVICE_TYPE, FUNC);           \
            }                                                       \
            return true;                                            \
        }();                                                        \
    }

#define REGISTER_MSM_PRE_COMPUTE_BASES_BACKEND(DEVICE_TYPE, FUNC)  \
    namespace {                                                     \
        static bool ICICLE_UNIQUE(_reg_msm_pre_) = []() -> bool {  \
            if (icicle::register_msm_precompute_bases) {           \
                icicle::register_msm_precompute_bases(DEVICE_TYPE, FUNC); \
            }                                                       \
            return true;                                            \
        }();                                                        \
    }

// G2 MSM Registration (matches ICICLE's naming: register_g2_msm)
#define REGISTER_MSM_G2_BACKEND(DEVICE_TYPE, FUNC)                 \
    namespace {                                                     \
        static bool ICICLE_UNIQUE(_reg_msm_g2_) = []() -> bool {   \
            icicle::register_g2_msm(DEVICE_TYPE, FUNC);            \
            return true;                                            \
        }();                                                        \
    }

#define REGISTER_MSM_G2_PRE_COMPUTE_BASES_BACKEND(DEVICE_TYPE, FUNC) \
    namespace {                                                      \
        static bool ICICLE_UNIQUE(_reg_msm_g2_pre_) = []() -> bool { \
            icicle::register_g2_msm_precompute_bases(DEVICE_TYPE, FUNC); \
            return true;                                             \
        }();                                                         \
    }
// =============================================================================
// NTT Registration Macros
// =============================================================================

#define REGISTER_NTT_BACKEND(DEVICE_TYPE, FUNC)                    \
    namespace {                                                     \
        static bool ICICLE_UNIQUE(_reg_ntt_) = []() -> bool {      \
            if (icicle::register_ntt) {                            \
                icicle::register_ntt(DEVICE_TYPE, FUNC);           \
            }                                                       \
            return true;                                            \
        }();                                                        \
    }

#define REGISTER_NTT_INIT_DOMAIN_BACKEND(DEVICE_TYPE, FUNC)        \
    namespace {                                                     \
        static bool ICICLE_UNIQUE(_reg_ntt_init_) = []() -> bool { \
            if (icicle::register_ntt_init_domain) {                \
                icicle::register_ntt_init_domain(DEVICE_TYPE, FUNC); \
            }                                                       \
            return true;                                            \
        }();                                                        \
    }

#define REGISTER_NTT_RELEASE_DOMAIN_BACKEND(DEVICE_TYPE, FUNC)     \
    namespace {                                                     \
        static bool ICICLE_UNIQUE(_reg_ntt_rel_) = []() -> bool {  \
            if (icicle::register_ntt_release_domain) {             \
                icicle::register_ntt_release_domain(DEVICE_TYPE, FUNC); \
            }                                                       \
            return true;                                            \
        }();                                                        \
    }

// =============================================================================
// Vector Operations Registration Macros
// =============================================================================

#define REGISTER_VECTOR_ADD_BACKEND(DEVICE_TYPE, FUNC)             \
    namespace {                                                     \
        static bool ICICLE_UNIQUE(_reg_vec_add_) = []() -> bool {  \
            if (icicle::register_vector_add) {                     \
                icicle::register_vector_add(DEVICE_TYPE, FUNC);    \
            }                                                       \
            return true;                                            \
        }();                                                        \
    }

#define REGISTER_VECTOR_SUB_BACKEND(DEVICE_TYPE, FUNC)             \
    namespace {                                                     \
        static bool ICICLE_UNIQUE(_reg_vec_sub_) = []() -> bool {  \
            if (icicle::register_vector_sub) {                     \
                icicle::register_vector_sub(DEVICE_TYPE, FUNC);    \
            }                                                       \
            return true;                                            \
        }();                                                        \
    }

#define REGISTER_VECTOR_MUL_BACKEND(DEVICE_TYPE, FUNC)             \
    namespace {                                                     \
        static bool ICICLE_UNIQUE(_reg_vec_mul_) = []() -> bool {  \
            if (icicle::register_vector_mul) {                     \
                icicle::register_vector_mul(DEVICE_TYPE, FUNC);    \
            }                                                       \
            return true;                                            \
        }();                                                        \
    }

#define REGISTER_SCALAR_MUL_VEC_BACKEND(DEVICE_TYPE, FUNC)         \
    namespace {                                                     \
        static bool ICICLE_UNIQUE(_reg_scl_mul_) = []() -> bool {  \
            if (icicle::register_scalar_mul_vec) {                 \
                icicle::register_scalar_mul_vec(DEVICE_TYPE, FUNC); \
            }                                                       \
            return true;                                            \
        }();                                                        \
    }

#define REGISTER_SCALAR_ADD_VEC_BACKEND(DEVICE_TYPE, FUNC)         \
    namespace {                                                     \
        static bool ICICLE_UNIQUE(_reg_scl_add_) = []() -> bool {  \
            if (icicle::register_scalar_add_vec) {                 \
                icicle::register_scalar_add_vec(DEVICE_TYPE, FUNC); \
            }                                                       \
            return true;                                            \
        }();                                                        \
    }

#define REGISTER_VECTOR_SUM_BACKEND(DEVICE_TYPE, FUNC)             \
    namespace {                                                     \
        static bool ICICLE_UNIQUE(_reg_vec_sum_) = []() -> bool {  \
            if (icicle::register_vector_sum) {                     \
                icicle::register_vector_sum(DEVICE_TYPE, FUNC);    \
            }                                                       \
            return true;                                            \
        }();                                                        \
    }

