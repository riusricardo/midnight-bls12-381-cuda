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

//! GPU-Accelerated Vector Operations
//!
//! This module provides GPU-accelerated element-wise vector operations using ICICLE's
//! VecOps API. From the Ingonyama Halo2 article:
//!
//! > "Calling VecOps API in ICICLE instead of using CPU to process long arrays
//! > gave 200x boost on average (size 2^22)"
//!
//! # Supported Operations
//!
//! - `vector_add`: Element-wise addition of two vectors
//! - `vector_sub`: Element-wise subtraction of two vectors
//! - `vector_mul`: Element-wise multiplication of two vectors
//! - `scalar_mul`: Multiply all elements by a scalar
//!
//! # Usage
//!
//! ```rust,ignore
//! use midnight_proofs::gpu::vecops::{vector_add, vector_mul};
//!
//! // GPU-accelerated element-wise operations
//! let result = vector_add(&a, &b)?;
//! let product = vector_mul(&a, &b)?;
//! ```
//!
//! # Performance Notes
//!
//! - Operations automatically fall back to CPU for small vectors (< threshold)
//! - Zero-copy type conversion via transmute (Montgomery form preserved)
//! - Async operations available for pipelining

use ff::Field;
use midnight_curves::Fq;
#[cfg(not(feature = "gpu"))]
use crate::GpuError;

#[cfg(feature = "gpu")]
use {
    crate::{GpuError, TypeConverter},
    icicle_bls12_381::curve::ScalarField as IcicleScalar,
    icicle_core::bignum::BigNum,
    icicle_core::vec_ops::{VecOps, VecOpsConfig},
    icicle_runtime::memory::{DeviceVec, HostSlice},
};

/// Errors specific to vector operations
#[derive(Debug)]
pub enum VecOpsError {
    /// Size mismatch between input vectors
    SizeMismatch {
        /// Expected vector size
        expected: usize,
        /// Actual vector size received
        got: usize,
    },
    /// GPU operation failed
    ExecutionFailed(String),
    /// GPU not available
    GpuUnavailable,
}

impl std::fmt::Display for VecOpsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VecOpsError::SizeMismatch { expected, got } => {
                write!(f, "Vector size mismatch: expected {}, got {}", expected, got)
            }
            VecOpsError::ExecutionFailed(msg) => write!(f, "VecOps execution failed: {}", msg),
            VecOpsError::GpuUnavailable => write!(f, "GPU not available for vector operations"),
        }
    }
}

impl std::error::Error for VecOpsError {}

impl From<GpuError> for VecOpsError {
    fn from(e: GpuError) -> Self {
        VecOpsError::ExecutionFailed(e.to_string())
    }
}

/// Minimum vector size for GPU acceleration.
/// Below this, CPU is faster due to transfer overhead.
const MIN_VECOPS_SIZE: usize = 4096; // 2^12

/// Check if GPU should be used for vector operations
#[inline]
pub fn should_use_gpu_vecops(size: usize) -> bool {
    #[cfg(feature = "gpu")]
    {
        use crate::config::device_type;
        use crate::config::DeviceType;
        
        match device_type() {
            DeviceType::Gpu => true,
            DeviceType::Cpu => false,
            DeviceType::Auto => size >= MIN_VECOPS_SIZE,
        }
    }
    
    #[cfg(not(feature = "gpu"))]
    {
        let _ = size;
        false
    }
}

/// GPU-accelerated element-wise vector addition.
///
/// Computes result[i] = a[i] + b[i] for all i.
///
/// # Arguments
/// * `a` - First input vector
/// * `b` - Second input vector (must be same length as `a`)
///
/// # Returns
/// New vector containing element-wise sum
#[cfg(feature = "gpu")]
pub fn vector_add(a: &[Fq], b: &[Fq]) -> Result<Vec<Fq>, VecOpsError> {
    if a.len() != b.len() {
        return Err(VecOpsError::SizeMismatch {
            expected: a.len(),
            got: b.len(),
        });
    }

    if a.is_empty() {
        return Ok(vec![]);
    }

    // Fall back to CPU for small vectors
    if !should_use_gpu_vecops(a.len()) {
        return Ok(a.iter().zip(b.iter()).map(|(x, y)| *x + *y).collect());
    }

    use crate::backend::ensure_backend_loaded;
    use icicle_runtime::{Device, set_device};

    ensure_backend_loaded()?;
    
    let device = Device::new("CUDA", 0);
    set_device(&device)
        .map_err(|e| VecOpsError::ExecutionFailed(format!("Failed to set device: {:?}", e)))?;

    // Zero-copy conversion to ICICLE types
    let icicle_a = TypeConverter::scalar_slice_as_icicle(a);
    let icicle_b = TypeConverter::scalar_slice_as_icicle(b);

    // Allocate result on device
    let mut device_result = DeviceVec::<IcicleScalar>::device_malloc(a.len())
        .map_err(|e| VecOpsError::ExecutionFailed(format!("Device malloc failed: {:?}", e)))?;

    // Configure VecOps
    let cfg = VecOpsConfig::default();

    // Execute GPU vector add
    IcicleScalar::add(
        HostSlice::from_slice(icicle_a),
        HostSlice::from_slice(icicle_b),
        &mut device_result[..],
        &cfg,
    )
    .map_err(|e| VecOpsError::ExecutionFailed(format!("vector_add failed: {:?}", e)))?;

    // Copy result back to host
    let mut host_result = vec![<IcicleScalar as BigNum>::zero(); a.len()];
    device_result
        .copy_to_host(HostSlice::from_mut_slice(&mut host_result))
        .map_err(|e| VecOpsError::ExecutionFailed(format!("Copy to host failed: {:?}", e)))?;

    // Zero-copy conversion back to Fq
    Ok(TypeConverter::icicle_slice_as_scalar(&host_result).to_vec())
}

/// GPU-accelerated element-wise vector subtraction.
///
/// Computes result[i] = a[i] - b[i] for all i.
#[cfg(feature = "gpu")]
pub fn vector_sub(a: &[Fq], b: &[Fq]) -> Result<Vec<Fq>, VecOpsError> {
    if a.len() != b.len() {
        return Err(VecOpsError::SizeMismatch {
            expected: a.len(),
            got: b.len(),
        });
    }

    if a.is_empty() {
        return Ok(vec![]);
    }

    // Fall back to CPU for small vectors
    if !should_use_gpu_vecops(a.len()) {
        return Ok(a.iter().zip(b.iter()).map(|(x, y)| *x - *y).collect());
    }

    use crate::backend::ensure_backend_loaded;
    use icicle_runtime::{Device, set_device};

    ensure_backend_loaded()?;
    
    let device = Device::new("CUDA", 0);
    set_device(&device)
        .map_err(|e| VecOpsError::ExecutionFailed(format!("Failed to set device: {:?}", e)))?;

    let icicle_a = TypeConverter::scalar_slice_as_icicle(a);
    let icicle_b = TypeConverter::scalar_slice_as_icicle(b);

    let mut device_result = DeviceVec::<IcicleScalar>::device_malloc(a.len())
        .map_err(|e| VecOpsError::ExecutionFailed(format!("Device malloc failed: {:?}", e)))?;

    let cfg = VecOpsConfig::default();

    IcicleScalar::sub(
        HostSlice::from_slice(icicle_a),
        HostSlice::from_slice(icicle_b),
        &mut device_result[..],
        &cfg,
    )
    .map_err(|e| VecOpsError::ExecutionFailed(format!("vector_sub failed: {:?}", e)))?;

    let mut host_result = vec![<IcicleScalar as BigNum>::zero(); a.len()];
    device_result
        .copy_to_host(HostSlice::from_mut_slice(&mut host_result))
        .map_err(|e| VecOpsError::ExecutionFailed(format!("Copy to host failed: {:?}", e)))?;

    Ok(TypeConverter::icicle_slice_as_scalar(&host_result).to_vec())
}

/// GPU-accelerated element-wise vector multiplication.
///
/// Computes result[i] = a[i] * b[i] for all i.
#[cfg(feature = "gpu")]
pub fn vector_mul(a: &[Fq], b: &[Fq]) -> Result<Vec<Fq>, VecOpsError> {
    if a.len() != b.len() {
        return Err(VecOpsError::SizeMismatch {
            expected: a.len(),
            got: b.len(),
        });
    }

    if a.is_empty() {
        return Ok(vec![]);
    }

    // Fall back to CPU for small vectors
    if !should_use_gpu_vecops(a.len()) {
        return Ok(a.iter().zip(b.iter()).map(|(x, y)| *x * *y).collect());
    }

    use crate::backend::ensure_backend_loaded;
    use icicle_runtime::{Device, set_device};

    ensure_backend_loaded()?;
    
    let device = Device::new("CUDA", 0);
    set_device(&device)
        .map_err(|e| VecOpsError::ExecutionFailed(format!("Failed to set device: {:?}", e)))?;

    let icicle_a = TypeConverter::scalar_slice_as_icicle(a);
    let icicle_b = TypeConverter::scalar_slice_as_icicle(b);

    let mut device_result = DeviceVec::<IcicleScalar>::device_malloc(a.len())
        .map_err(|e| VecOpsError::ExecutionFailed(format!("Device malloc failed: {:?}", e)))?;

    let cfg = VecOpsConfig::default();

    IcicleScalar::mul(
        HostSlice::from_slice(icicle_a),
        HostSlice::from_slice(icicle_b),
        &mut device_result[..],
        &cfg,
    )
    .map_err(|e| VecOpsError::ExecutionFailed(format!("vector_mul failed: {:?}", e)))?;

    let mut host_result = vec![<IcicleScalar as BigNum>::zero(); a.len()];
    device_result
        .copy_to_host(HostSlice::from_mut_slice(&mut host_result))
        .map_err(|e| VecOpsError::ExecutionFailed(format!("Copy to host failed: {:?}", e)))?;

    Ok(TypeConverter::icicle_slice_as_scalar(&host_result).to_vec())
}

/// GPU-accelerated scalar multiplication of vector.
///
/// Computes result[i] = scalar * a[i] for all i.
///
/// # Performance Note
///
/// This uses ICICLE's `scalar_mul` with `batch_size` to broadcast the scalar
/// efficiently without creating a full repeated vector. The scalar is passed
/// as a single-element slice and `batch_size = a.len()` tells ICICLE to
/// broadcast it across all elements.
#[cfg(feature = "gpu")]
pub fn scalar_mul(scalar: Fq, a: &[Fq]) -> Result<Vec<Fq>, VecOpsError> {
    use icicle_core::vec_ops::scalar_mul as icicle_scalar_mul;

    if a.is_empty() {
        return Ok(vec![]);
    }

    // Fall back to CPU for small vectors
    if !should_use_gpu_vecops(a.len()) {
        return Ok(a.iter().map(|x| scalar * *x).collect());
    }

    use crate::backend::ensure_backend_loaded;
    use icicle_runtime::{Device, set_device};

    ensure_backend_loaded()?;
    
    let device = Device::new("CUDA", 0);
    set_device(&device)
        .map_err(|e| VecOpsError::ExecutionFailed(format!("Failed to set device: {:?}", e)))?;

    // Zero-copy conversion
    let icicle_scalar = TypeConverter::scalar_slice_as_icicle(std::slice::from_ref(&scalar));
    let icicle_a = TypeConverter::scalar_slice_as_icicle(a);

    // Allocate result on device
    let mut device_result = DeviceVec::<IcicleScalar>::device_malloc(a.len())
        .map_err(|e| VecOpsError::ExecutionFailed(format!("Device malloc failed: {:?}", e)))?;

    // Configure VecOps with batch_size to broadcast the single scalar
    let mut cfg = VecOpsConfig::default();
    cfg.batch_size = a.len() as i32;

    // Execute GPU scalar multiply with broadcast
    // icicle_scalar is length 1, batch_size = a.len() makes ICICLE broadcast it
    icicle_scalar_mul(
        HostSlice::from_slice(icicle_scalar),
        HostSlice::from_slice(icicle_a),
        &mut device_result[..],
        &cfg,
    )
    .map_err(|e| VecOpsError::ExecutionFailed(format!("scalar_mul failed: {:?}", e)))?;

    // Copy result back to host
    let mut host_result = vec![<IcicleScalar as BigNum>::zero(); a.len()];
    device_result
        .copy_to_host(HostSlice::from_mut_slice(&mut host_result))
        .map_err(|e| VecOpsError::ExecutionFailed(format!("Copy to host failed: {:?}", e)))?;

    Ok(TypeConverter::icicle_slice_as_scalar(&host_result).to_vec())
}

// =========================================================================
// Bit-Reverse Operations
// =========================================================================
//
// Bit-reversal is used in NTT/FFT algorithms to reorder data between
// different ordering conventions. ICICLE's bit_reverse is optimized for GPU.
//
// Use cases:
// - Converting between natural (kNN) and bit-reversed (kRN/kNR) orderings
// - Explicit reordering when using mixed orderings in NTT pipelines

/// GPU-accelerated bit-reverse permutation.
///
/// Reorders the input vector according to bit-reversed indices.
/// For a vector of size N = 2^k, element at index i is moved to index bit_reverse(i).
///
/// # Example
/// For N=8: [a0, a1, a2, a3, a4, a5, a6, a7] → [a0, a4, a2, a6, a1, a5, a3, a7]
///
/// # Arguments
/// * `input` - Input vector (must be power of 2 length)
///
/// # Returns
/// New vector with bit-reversed ordering
#[cfg(feature = "gpu")]
pub fn bit_reverse(input: &[Fq]) -> Result<Vec<Fq>, VecOpsError> {
    use icicle_core::vec_ops::bit_reverse as icicle_bit_reverse;

    if input.is_empty() {
        return Ok(vec![]);
    }

    if !input.len().is_power_of_two() {
        return Err(VecOpsError::ExecutionFailed(format!(
            "bit_reverse requires power of 2 length, got {}", input.len()
        )));
    }

    // Fall back to CPU for small vectors
    if !should_use_gpu_vecops(input.len()) {
        return bit_reverse_cpu(input);
    }

    use crate::backend::ensure_backend_loaded;
    use icicle_runtime::{Device, set_device};

    ensure_backend_loaded()?;
    
    let device = Device::new("CUDA", 0);
    set_device(&device)
        .map_err(|e| VecOpsError::ExecutionFailed(format!("Failed to set device: {:?}", e)))?;

    // Zero-copy conversion to ICICLE types
    let icicle_input = TypeConverter::scalar_slice_as_icicle(input);

    // Allocate result on device
    let mut device_result = DeviceVec::<IcicleScalar>::device_malloc(input.len())
        .map_err(|e| VecOpsError::ExecutionFailed(format!("Device malloc failed: {:?}", e)))?;

    // Configure VecOps
    let cfg = VecOpsConfig::default();

    // Execute GPU bit-reverse
    icicle_bit_reverse(
        HostSlice::from_slice(icicle_input),
        &cfg,
        &mut device_result[..],
    )
    .map_err(|e| VecOpsError::ExecutionFailed(format!("bit_reverse failed: {:?}", e)))?;

    // Copy result back to host
    let mut host_result = vec![<IcicleScalar as BigNum>::zero(); input.len()];
    device_result
        .copy_to_host(HostSlice::from_mut_slice(&mut host_result))
        .map_err(|e| VecOpsError::ExecutionFailed(format!("Copy to host failed: {:?}", e)))?;

    Ok(TypeConverter::icicle_slice_as_scalar(&host_result).to_vec())
}

/// GPU-accelerated in-place bit-reverse permutation.
///
/// Reorders the input vector in-place according to bit-reversed indices.
/// More memory-efficient than `bit_reverse` as it doesn't allocate a new vector.
///
/// # Arguments
/// * `input` - Input/output vector (must be power of 2 length)
#[cfg(feature = "gpu")]
pub fn bit_reverse_inplace(input: &mut [Fq]) -> Result<(), VecOpsError> {
    use icicle_core::vec_ops::bit_reverse_inplace as icicle_bit_reverse_inplace;

    if input.is_empty() {
        return Ok(());
    }

    if !input.len().is_power_of_two() {
        return Err(VecOpsError::ExecutionFailed(format!(
            "bit_reverse_inplace requires power of 2 length, got {}", input.len()
        )));
    }

    // Fall back to CPU for small vectors
    if !should_use_gpu_vecops(input.len()) {
        bit_reverse_inplace_cpu(input);
        return Ok(());
    }

    use crate::backend::ensure_backend_loaded;
    use icicle_runtime::{Device, set_device};

    ensure_backend_loaded()?;
    
    let device = Device::new("CUDA", 0);
    set_device(&device)
        .map_err(|e| VecOpsError::ExecutionFailed(format!("Failed to set device: {:?}", e)))?;

    // Zero-copy conversion (for GPU we need device memory)
    let icicle_input = TypeConverter::scalar_slice_as_icicle(input);

    // Allocate device buffer and copy input
    let mut device_data = DeviceVec::<IcicleScalar>::device_malloc(input.len())
        .map_err(|e| VecOpsError::ExecutionFailed(format!("Device malloc failed: {:?}", e)))?;
    
    device_data
        .copy_from_host(HostSlice::from_slice(icicle_input))
        .map_err(|e| VecOpsError::ExecutionFailed(format!("Copy to device failed: {:?}", e)))?;

    // Configure VecOps
    let mut cfg = VecOpsConfig::default();
    cfg.is_a_on_device = true;
    cfg.is_result_on_device = true;

    // Execute GPU bit-reverse in-place
    icicle_bit_reverse_inplace(
        &mut device_data[..],
        &cfg,
    )
    .map_err(|e| VecOpsError::ExecutionFailed(format!("bit_reverse_inplace failed: {:?}", e)))?;

    // Copy result back to host
    let icicle_output = TypeConverter::scalar_slice_as_icicle_mut(input);
    device_data
        .copy_to_host(HostSlice::from_mut_slice(icicle_output))
        .map_err(|e| VecOpsError::ExecutionFailed(format!("Copy to host failed: {:?}", e)))?;

    Ok(())
}

/// CPU fallback for bit-reverse (used for small vectors or non-GPU builds)
fn bit_reverse_cpu(input: &[Fq]) -> Result<Vec<Fq>, VecOpsError> {
    let n = input.len();
    let log_n = n.trailing_zeros() as usize;
    
    let mut result = vec![Fq::ZERO; n];
    for i in 0..n {
        let rev_i = bit_reverse_index(i, log_n);
        result[rev_i] = input[i];
    }
    Ok(result)
}

/// CPU fallback for in-place bit-reverse
fn bit_reverse_inplace_cpu(input: &mut [Fq]) {
    let n = input.len();
    let log_n = n.trailing_zeros() as usize;
    
    for i in 0..n {
        let rev_i = bit_reverse_index(i, log_n);
        if i < rev_i {
            input.swap(i, rev_i);
        }
    }
}

/// Compute bit-reversed index for a given bit width
#[inline]
fn bit_reverse_index(mut i: usize, log_n: usize) -> usize {
    let mut rev = 0;
    for _ in 0..log_n {
        rev = (rev << 1) | (i & 1);
        i >>= 1;
    }
    rev
}

// CPU fallback implementations for non-GPU builds
#[cfg(not(feature = "gpu"))]
pub fn bit_reverse(input: &[Fq]) -> Result<Vec<Fq>, VecOpsError> {
    if !input.len().is_power_of_two() && !input.is_empty() {
        return Err(VecOpsError::ExecutionFailed(format!(
            "bit_reverse requires power of 2 length, got {}", input.len()
        )));
    }
    bit_reverse_cpu(input)
}

#[cfg(not(feature = "gpu"))]
pub fn bit_reverse_inplace(input: &mut [Fq]) -> Result<(), VecOpsError> {
    if !input.len().is_power_of_two() && !input.is_empty() {
        return Err(VecOpsError::ExecutionFailed(format!(
            "bit_reverse_inplace requires power of 2 length, got {}", input.len()
        )));
    }
    bit_reverse_inplace_cpu(input);
    Ok(())
}

// CPU fallback implementations for non-GPU builds
#[cfg(not(feature = "gpu"))]
pub fn vector_add(a: &[Fq], b: &[Fq]) -> Result<Vec<Fq>, VecOpsError> {
    if a.len() != b.len() {
        return Err(VecOpsError::SizeMismatch {
            expected: a.len(),
            got: b.len(),
        });
    }
    Ok(a.iter().zip(b.iter()).map(|(x, y)| *x + *y).collect())
}

#[cfg(not(feature = "gpu"))]
pub fn vector_sub(a: &[Fq], b: &[Fq]) -> Result<Vec<Fq>, VecOpsError> {
    if a.len() != b.len() {
        return Err(VecOpsError::SizeMismatch {
            expected: a.len(),
            got: b.len(),
        });
    }
    Ok(a.iter().zip(b.iter()).map(|(x, y)| *x - *y).collect())
}

#[cfg(not(feature = "gpu"))]
pub fn vector_mul(a: &[Fq], b: &[Fq]) -> Result<Vec<Fq>, VecOpsError> {
    if a.len() != b.len() {
        return Err(VecOpsError::SizeMismatch {
            expected: a.len(),
            got: b.len(),
        });
    }
    Ok(a.iter().zip(b.iter()).map(|(x, y)| *x * *y).collect())
}

#[cfg(not(feature = "gpu"))]
pub fn scalar_mul(scalar: Fq, a: &[Fq]) -> Result<Vec<Fq>, VecOpsError> {
    Ok(a.iter().map(|x| scalar * *x).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ff::Field;

    #[test]
    fn test_vector_add_small() {
        let a = vec![Fq::ONE, Fq::ONE + Fq::ONE, Fq::ONE + Fq::ONE + Fq::ONE];
        let b = vec![Fq::ONE, Fq::ONE, Fq::ONE];
        
        let result = vector_add(&a, &b).unwrap();
        
        assert_eq!(result.len(), 3);
        assert_eq!(result[0], Fq::ONE + Fq::ONE);
        assert_eq!(result[1], Fq::ONE + Fq::ONE + Fq::ONE);
    }

    #[test]
    fn test_vector_sub_small() {
        let a = vec![Fq::ONE + Fq::ONE, Fq::ONE + Fq::ONE + Fq::ONE];
        let b = vec![Fq::ONE, Fq::ONE];
        
        let result = vector_sub(&a, &b).unwrap();
        
        assert_eq!(result.len(), 2);
        assert_eq!(result[0], Fq::ONE);
        assert_eq!(result[1], Fq::ONE + Fq::ONE);
    }

    #[test]
    fn test_size_mismatch() {
        let a = vec![Fq::ONE; 10];
        let b = vec![Fq::ONE; 5];
        
        assert!(matches!(
            vector_add(&a, &b),
            Err(VecOpsError::SizeMismatch { .. })
        ));
    }

    #[test]
    fn test_empty_vectors() {
        let empty: Vec<Fq> = vec![];
        
        assert!(vector_add(&empty, &empty).unwrap().is_empty());
        assert!(vector_sub(&empty, &empty).unwrap().is_empty());
        assert!(vector_mul(&empty, &empty).unwrap().is_empty());
    }

    #[test]
    fn test_bit_reverse_index() {
        // For log_n = 3 (N=8):
        // 0 (000) -> 0 (000)
        // 1 (001) -> 4 (100)
        // 2 (010) -> 2 (010)
        // 3 (011) -> 6 (110)
        // 4 (100) -> 1 (001)
        // 5 (101) -> 5 (101)
        // 6 (110) -> 3 (011)
        // 7 (111) -> 7 (111)
        assert_eq!(bit_reverse_index(0, 3), 0);
        assert_eq!(bit_reverse_index(1, 3), 4);
        assert_eq!(bit_reverse_index(2, 3), 2);
        assert_eq!(bit_reverse_index(3, 3), 6);
        assert_eq!(bit_reverse_index(4, 3), 1);
        assert_eq!(bit_reverse_index(5, 3), 5);
        assert_eq!(bit_reverse_index(6, 3), 3);
        assert_eq!(bit_reverse_index(7, 3), 7);
    }

    #[test]
    fn test_bit_reverse_small() {
        // Create a vector [0, 1, 2, 3, 4, 5, 6, 7] as field elements
        let input: Vec<Fq> = (0..8u64).map(|i| Fq::from(i)).collect();
        
        let result = bit_reverse(&input).unwrap();
        
        // Expected: [0, 4, 2, 6, 1, 5, 3, 7]
        assert_eq!(result.len(), 8);
        assert_eq!(result[0], Fq::from(0u64));
        assert_eq!(result[1], Fq::from(4u64));
        assert_eq!(result[2], Fq::from(2u64));
        assert_eq!(result[3], Fq::from(6u64));
        assert_eq!(result[4], Fq::from(1u64));
        assert_eq!(result[5], Fq::from(5u64));
        assert_eq!(result[6], Fq::from(3u64));
        assert_eq!(result[7], Fq::from(7u64));
    }

    #[test]
    fn test_bit_reverse_inplace_small() {
        let mut input: Vec<Fq> = (0..8u64).map(|i| Fq::from(i)).collect();
        
        bit_reverse_inplace(&mut input).unwrap();
        
        // Same expected result as above
        assert_eq!(input[0], Fq::from(0u64));
        assert_eq!(input[1], Fq::from(4u64));
        assert_eq!(input[2], Fq::from(2u64));
        assert_eq!(input[3], Fq::from(6u64));
        assert_eq!(input[4], Fq::from(1u64));
        assert_eq!(input[5], Fq::from(5u64));
        assert_eq!(input[6], Fq::from(3u64));
        assert_eq!(input[7], Fq::from(7u64));
    }

    #[test]
    fn test_bit_reverse_double_is_identity() {
        let input: Vec<Fq> = (0..16u64).map(|i| Fq::from(i * 7 + 3)).collect();
        
        let once = bit_reverse(&input).unwrap();
        let twice = bit_reverse(&once).unwrap();
        
        assert_eq!(input, twice, "Double bit-reverse should be identity");
    }

    #[test]
    fn test_bit_reverse_non_power_of_two_fails() {
        let input = vec![Fq::ONE; 5]; // Not a power of 2
        
        assert!(matches!(
            bit_reverse(&input),
            Err(VecOpsError::ExecutionFailed(_))
        ));
    }

    #[test]
    fn test_bit_reverse_empty() {
        let empty: Vec<Fq> = vec![];
        assert!(bit_reverse(&empty).unwrap().is_empty());
        
        let mut empty_mut: Vec<Fq> = vec![];
        bit_reverse_inplace(&mut empty_mut).unwrap();
        assert!(empty_mut.is_empty());
    }
}
