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
#[cfg(feature = "gpu")]
pub fn scalar_mul(scalar: Fq, a: &[Fq]) -> Result<Vec<Fq>, VecOpsError> {
    if a.is_empty() {
        return Ok(vec![]);
    }

    // Fall back to CPU for small vectors
    if !should_use_gpu_vecops(a.len()) {
        return Ok(a.iter().map(|x| scalar * *x).collect());
    }

    // Create a vector of the scalar repeated
    let scalars = vec![scalar; a.len()];
    vector_mul(&scalars, a)
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
}
