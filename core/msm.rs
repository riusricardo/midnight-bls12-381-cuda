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

//! GPU-Accelerated Multi-Scalar Multiplication (MSM)
//!
//! This module provides GPU-accelerated MSM operations using ICICLE's CUDA backend.
//! MSM computes sum(scalars[i] * points[i]) - the core operation for polynomial commitments.
//!
//! # Architecture
//!
//! Following the icicle-halo2 pattern, we provide:
//! 1. **Sync API**: Uses `IcicleStream::default()` for simple blocking operations
//! 2. **Async API**: Creates per-operation streams for pipelining
//!
//! # Reference
//!
//! Pattern derived from:
//! - **ICICLE Rust Guide**: https://dev.ingonyama.com/start/programmers_guide/rust
//!
//! # Sync Usage
//!
//! ```rust,ignore
//! use midnight_proofs::gpu::msm::GpuMsmContext;
//!
//! let ctx = GpuMsmContext::new()?;
//! let result = ctx.msm(&scalars, &points)?;  // Blocking
//! ```
//!
//! # Async Usage (icicle-halo2 pattern)
//!
//! ```rust,ignore
//! // Launch async MSM
//! let handle = ctx.msm_async(&scalars, &device_bases)?;
//!
//! // ... do other work while GPU computes ...
//!
//! // Wait for result
//! let result = handle.wait()?;
//! ```

use crate::GpuError;

#[cfg(feature = "gpu")]
use crate::{TypeConverter, stream::ManagedStream};

#[cfg(feature = "gpu")]
use midnight_curves::{Fq as Scalar, G1Affine, G1Projective, G2Affine, G2Projective};

#[cfg(feature = "gpu")]
use group::Group;
use tracing::debug;

#[cfg(feature = "gpu")]
use icicle_bls12_381::curve::{
    G1Affine as IcicleG1Affine,
    G1Projective as IcicleG1Projective,
    G2Affine as IcicleG2Affine,
    G2Projective as IcicleG2Projective,
};
#[cfg(feature = "gpu")]
use icicle_core::ecntt::Projective;
#[cfg(feature = "gpu")]
use icicle_core::msm::{msm, MSMConfig};
#[cfg(feature = "gpu")]
use icicle_runtime::{
    Device,
    memory::{DeviceVec, HostSlice, HostOrDeviceSlice},
};

/// Errors specific to MSM operations
#[derive(Debug)]
pub enum MsmError {
    /// GPU context initialization failed
    ContextInitFailed(String),
    /// MSM execution failed
    ExecutionFailed(String),
    /// Invalid input (size mismatch, empty, etc.)
    InvalidInput(String),
    /// Underlying GPU error
    GpuError(GpuError),
}

impl std::fmt::Display for MsmError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MsmError::ContextInitFailed(msg) => write!(f, "MSM context init failed: {}", msg),
            MsmError::ExecutionFailed(msg) => write!(f, "MSM execution failed: {}", msg),
            MsmError::InvalidInput(msg) => write!(f, "Invalid MSM input: {}", msg),
            MsmError::GpuError(e) => write!(f, "GPU error: {}", e),
        }
    }
}

impl std::error::Error for MsmError {}

impl From<GpuError> for MsmError {
    fn from(e: GpuError) -> Self {
        MsmError::GpuError(e)
    }
}

/// GPU MSM Context
///
/// Manages GPU resources for MSM operations:
/// - Device handle for CUDA operations
/// - Backend initialization state
///
/// # Thread Safety
///
/// This context can be safely shared between threads. Each MSM call
/// uses synchronous execution on the default stream.
#[cfg(feature = "gpu")]
pub struct GpuMsmContext {
    /// Device reference
    device: Device,
}

#[cfg(feature = "gpu")]
impl std::fmt::Debug for GpuMsmContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuMsmContext")
            .field("device", &"<Device>")
            .finish()
    }
}

// Implement Send and Sync for GpuMsmContext
// Safe because Device is just an identifier (string + int), no GPU resources
#[cfg(feature = "gpu")]
unsafe impl Send for GpuMsmContext {}
#[cfg(feature = "gpu")]
unsafe impl Sync for GpuMsmContext {}

#[cfg(feature = "gpu")]
impl GpuMsmContext {
    /// Create a new GPU MSM context
    ///
    /// Initializes the ICICLE backend and sets the device.
    pub fn new() -> Result<Self, MsmError> {
        use crate::backend::ensure_backend_loaded;
        use icicle_runtime::set_device;

        // Ensure ICICLE backend is loaded
        ensure_backend_loaded()
            .map_err(|e| MsmError::GpuError(e))?;

        // Set device context
        let device = Device::new("CUDA", 0);
        set_device(&device)
            .map_err(|e| MsmError::ContextInitFailed(format!("Failed to set device: {:?}", e)))?;

        debug!("GpuMsmContext created");

        Ok(Self { device })
    }

    /// Get the device handle
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Upload G1 bases to GPU memory in Montgomery form (zero-copy conversion)
    ///
    /// This is the optimized path for repeated MSMs with the same bases.
    /// Bases are kept in Montgomery form on GPU, eliminating per-MSM conversion.
    ///
    /// # Arguments
    /// * `points` - G1 affine points to upload
    ///
    /// # Returns
    /// Device vector of bases in GPU memory
    pub fn upload_g1_bases(&self, points: &[G1Affine]) -> Result<DeviceVec<IcicleG1Affine>, MsmError> {
        use icicle_runtime::set_device;

        if points.is_empty() {
            return Err(MsmError::InvalidInput("Empty points array".to_string()));
        }

        // Set device context
        set_device(&self.device)
            .map_err(|e| MsmError::ExecutionFailed(format!("Failed to set device: {:?}", e)))?;

        // Zero-copy conversion: reinterpret as ICICLE points (keeps Montgomery form)
        let icicle_points = TypeConverter::g1_slice_as_icicle(points);

        // Allocate device memory
        let mut device_bases = DeviceVec::<IcicleG1Affine>::device_malloc(points.len())
            .map_err(|e| MsmError::ExecutionFailed(format!("Device malloc failed: {:?}", e)))?;

        // Upload to GPU
        device_bases
            .copy_from_host(HostSlice::from_slice(icicle_points))
            .map_err(|e| MsmError::ExecutionFailed(format!("Copy to device failed: {:?}", e)))?;

        Ok(device_bases)
    }

    /// Upload G2 bases to GPU memory in Montgomery form (zero-copy conversion)
    ///
    /// Same as `upload_g1_bases()` but for G2 points.
    pub fn upload_g2_bases(&self, points: &[G2Affine]) -> Result<DeviceVec<IcicleG2Affine>, MsmError> {
        use icicle_runtime::set_device;

        if points.is_empty() {
            return Err(MsmError::InvalidInput("Empty points array".to_string()));
        }

        // Set device context
        set_device(&self.device)
            .map_err(|e| MsmError::ExecutionFailed(format!("Failed to set device: {:?}", e)))?;

        // Zero-copy conversion: reinterpret as ICICLE points (keeps Montgomery form)
        let icicle_points = TypeConverter::g2_slice_as_icicle(points);

        // Allocate device memory
        let mut device_bases = DeviceVec::<IcicleG2Affine>::device_malloc(points.len())
            .map_err(|e| MsmError::ExecutionFailed(format!("Device malloc failed: {:?}", e)))?;

        // Upload to GPU
        device_bases
            .copy_from_host(HostSlice::from_slice(icicle_points))
            .map_err(|e| MsmError::ExecutionFailed(format!("Copy to device failed: {:?}", e)))?;

        Ok(device_bases)
    }

    /// Compute G1 MSM: sum(scalars[i] * points[i])
    ///
    /// Points are uploaded to GPU for this call. For repeated MSMs with the same
    /// bases, use `msm_with_device_bases()` instead.
    ///
    /// # Arguments
    /// * `scalars` - Scalar multipliers (in Montgomery form)
    /// * `points` - G1 affine points
    ///
    /// # Returns
    /// The MSM result as a G1 projective point
    pub fn msm(&self, scalars: &[Scalar], points: &[G1Affine]) -> Result<G1Projective, MsmError> {
        use icicle_runtime::set_device;

        if scalars.len() != points.len() {
            return Err(MsmError::InvalidInput(format!(
                "Scalar and point count mismatch: {} vs {}",
                scalars.len(),
                points.len()
            )));
        }

        if scalars.is_empty() {
            return Ok(G1Projective::identity());
        }

        // Set device context
        set_device(&self.device)
            .map_err(|e| MsmError::ExecutionFailed(format!("Failed to set device: {:?}", e)))?;

        #[cfg(feature = "trace-msm")]
        let start = std::time::Instant::now();

        // Zero-copy scalar conversion - O(1) pointer cast
        let icicle_scalars = TypeConverter::scalar_slice_as_icicle(scalars);

        // Convert points (TODO: zero-copy when layout verified)
        let icicle_points = TypeConverter::g1_affine_slice_to_icicle_vec(points);

        // Allocate device result buffer
        let mut device_result = DeviceVec::<IcicleG1Projective>::device_malloc(1)
            .map_err(|e| MsmError::ExecutionFailed(format!("Device malloc failed: {:?}", e)))?;

        // Configure MSM - synchronous on default stream
        // Note: are_bases_montgomery_form = false because g1_affine_to_icicle uses to_repr()
        // which converts out of Montgomery form. Scalars remain in Montgomery form.
        let mut cfg = MSMConfig::default();
        cfg.are_scalars_montgomery_form = true;
        cfg.is_async = false;

        // Execute MSM
        msm(
            HostSlice::from_slice(icicle_scalars),
            HostSlice::from_slice(&icicle_points),
            &cfg,
            &mut device_result[..],
        )
        .map_err(|e| MsmError::ExecutionFailed(format!("MSM failed: {:?}", e)))?;

        // Copy result back to host
        let mut host_result = vec![IcicleG1Projective::zero(); 1];
        device_result
            .copy_to_host(HostSlice::from_mut_slice(&mut host_result))
            .map_err(|e| MsmError::ExecutionFailed(format!("Copy to host failed: {:?}", e)))?;

        #[cfg(feature = "trace-msm")]
        debug!("G1 MSM completed for {} points in {:?}", scalars.len(), start.elapsed());

        Ok(TypeConverter::icicle_to_g1_projective(&host_result[0]))
    }

    /// Compute G1 MSM with pre-uploaded device bases
    ///
    /// This is the most efficient path when bases are cached on GPU (e.g., SRS).
    /// Eliminates per-call point upload overhead.
    ///
    /// # Arguments
    /// * `scalars` - Scalar multipliers (in Montgomery form)
    /// * `device_bases` - G1 points already in GPU memory
    ///
    /// # Returns
    /// The MSM result as a G1 projective point
    pub fn msm_with_device_bases(
        &self,
        scalars: &[Scalar],
        device_bases: &DeviceVec<IcicleG1Affine>,
    ) -> Result<G1Projective, MsmError> {
        use icicle_runtime::set_device;

        if scalars.is_empty() {
            return Ok(G1Projective::identity());
        }

        if scalars.len() > device_bases.len() {
            return Err(MsmError::InvalidInput(format!(
                "More scalars ({}) than bases ({})",
                scalars.len(),
                device_bases.len()
            )));
        }

        // Set device context
        set_device(&self.device)
            .map_err(|e| MsmError::ExecutionFailed(format!("Failed to set device: {:?}", e)))?;

        #[cfg(feature = "trace-msm")]
        let start = std::time::Instant::now();

        // Zero-copy scalar conversion
        let icicle_scalars = TypeConverter::scalar_slice_as_icicle(scalars);

        // Allocate device result buffer
        let mut device_result = DeviceVec::<IcicleG1Projective>::device_malloc(1)
            .map_err(|e| MsmError::ExecutionFailed(format!("Device malloc failed: {:?}", e)))?;

        // Configure MSM - synchronous on default stream
        // CRITICAL: Both scalars AND bases are in Montgomery form!
        // - Scalars: midnight-curves stores Fq in Montgomery form
        // - Bases: uploaded in Montgomery form via get_or_upload_gpu_bases()
        // This eliminates per-MSM D2D copy + Montgomery conversion in CUDA backend.
        //
        // ICICLE performance tuning (from docs):
        // - precompute_factor: trades memory for ~20-30% speedup
        // - batch_size=1: single MSM per call (batching done at higher level)
        // - are_points_shared_in_batch: N/A for batch_size=1
        use crate::config::{precompute_factor, msm_window_size};
        let mut cfg = MSMConfig::default();
        cfg.are_scalars_montgomery_form = true;
        cfg.are_bases_montgomery_form = true;  // Bases pre-uploaded in Montgomery form
        cfg.is_async = false;
        cfg.precompute_factor = precompute_factor();
        if msm_window_size() > 0 {
            cfg.c = msm_window_size();
        }

        // Execute MSM with device bases - zero-copy, no conversion!
        msm(
            HostSlice::from_slice(icicle_scalars),
            &device_bases[..scalars.len()],
            &cfg,
            &mut device_result[..],
        )
        .map_err(|e| MsmError::ExecutionFailed(format!("MSM failed: {:?}", e)))?;

        // Copy result back to host
        let mut host_result = vec![IcicleG1Projective::zero(); 1];
        device_result
            .copy_to_host(HostSlice::from_mut_slice(&mut host_result))
            .map_err(|e| MsmError::ExecutionFailed(format!("Copy to host failed: {:?}", e)))?;

        #[cfg(feature = "trace-msm")]
        debug!("G1 MSM (device bases) completed for {} points in {:?}", scalars.len(), start.elapsed());

        Ok(TypeConverter::icicle_to_g1_projective(&host_result[0]))
    }

    /// Compute G1 MSM asynchronously with pre-uploaded device bases
    ///
    /// This launches the MSM on the GPU and returns immediately, allowing the
    /// CPU to continue with other work while the GPU computes.
    ///
    /// # Performance
    ///
    /// Async mode eliminates synchronization overhead:
    /// - GPU can pipeline multiple operations
    /// - CPU is free to prepare next batch
    /// - Expected 3-5x speedup vs synchronous mode
    ///
    /// # Arguments
    /// * `scalars` - Scalar multipliers (in Montgomery form)
    /// * `device_bases` - G1 points already in GPU memory
    ///
    /// # Returns
    /// A handle to the async operation. Call `wait()` to get the result.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// // Launch async MSM
    /// let handle = ctx.msm_with_device_bases_async(&scalars, &device_bases)?;
    ///
    /// // Do other CPU work while GPU computes
    /// prepare_next_batch();
    ///
    /// // Wait for GPU result
    /// let result = handle.wait(|p| TypeConverter::icicle_to_g1_projective(p))?;
    /// ```
    pub fn msm_with_device_bases_async(
        &self,
        scalars: &[Scalar],
        device_bases: &DeviceVec<IcicleG1Affine>,
    ) -> Result<MsmHandle, MsmError> {
        use icicle_runtime::set_device;

        if scalars.is_empty() {
            return Err(MsmError::InvalidInput("Empty scalars array".to_string()));
        }

        if scalars.len() > device_bases.len() {
            return Err(MsmError::InvalidInput(format!(
                "More scalars ({}) than bases ({})",
                scalars.len(),
                device_bases.len()
            )));
        }

        // Set device context
        set_device(&self.device)
            .map_err(|e| MsmError::ExecutionFailed(format!("Failed to set device: {:?}", e)))?;

        #[cfg(feature = "trace-msm")]
        let start = std::time::Instant::now();

        // Create dedicated stream for async operation
        let stream = ManagedStream::create()
            .map_err(|e| MsmError::ExecutionFailed(format!("Stream creation failed: {:?}", e)))?;

        // Zero-copy scalar conversion
        let icicle_scalars = TypeConverter::scalar_slice_as_icicle(scalars);

        // Allocate device result buffer
        let mut device_result = DeviceVec::<IcicleG1Projective>::device_malloc(1)
            .map_err(|e| MsmError::ExecutionFailed(format!("Device malloc failed: {:?}", e)))?;

        // Configure MSM - ASYNC mode with dedicated stream
        // ICICLE performance tuning parameters applied
        use crate::config::{precompute_factor, msm_window_size};
        let mut cfg = MSMConfig::default();
        cfg.are_scalars_montgomery_form = true;
        cfg.are_bases_montgomery_form = true;
        cfg.is_async = true;  // Enable async execution!
        cfg.stream_handle = stream.as_ref().into();
        cfg.precompute_factor = precompute_factor();
        if msm_window_size() > 0 {
            cfg.c = msm_window_size();
        }

        // Execute MSM - returns immediately, GPU continues in background
        msm(
            HostSlice::from_slice(icicle_scalars),
            &device_bases[..scalars.len()],
            &cfg,
            &mut device_result[..],
        )
        .map_err(|e| MsmError::ExecutionFailed(format!("MSM launch failed: {:?}", e)))?;

        #[cfg(feature = "trace-msm")]
        debug!("G1 MSM async launched for {} points in {:?}", scalars.len(), start.elapsed());

        // Return handle - GPU is still computing
        Ok(MsmHandle {
            stream,
            device_result,
            is_identity: false,
        })
    }

    /// 
    /// Compute G2 MSM: sum(scalars[i] * points[i])
    ///
    /// # Arguments
    /// * `scalars` - Scalar multipliers (in Montgomery form)
    /// * `points` - G2 affine points
    ///
    /// # Returns
    /// The MSM result as a G2 projective point
    pub fn g2_msm(&self, scalars: &[Scalar], points: &[G2Affine]) -> Result<G2Projective, MsmError> {
        use icicle_runtime::set_device;

        if scalars.len() != points.len() {
            return Err(MsmError::InvalidInput(format!(
                "Scalar and point count mismatch: {} vs {}",
                scalars.len(),
                points.len()
            )));
        }

        if scalars.is_empty() {
            return Ok(G2Projective::identity());
        }

        // Set device context
        set_device(&self.device)
            .map_err(|e| MsmError::ExecutionFailed(format!("Failed to set device: {:?}", e)))?;

        #[cfg(feature = "trace-msm")]
        let start = std::time::Instant::now();

        // Zero-copy scalar conversion
        let icicle_scalars = TypeConverter::scalar_slice_as_icicle(scalars);

        // Convert points (TODO: zero-copy when layout verified)
        let icicle_points = TypeConverter::g2_affine_slice_to_icicle_vec(points);

        // Allocate device result buffer
        let mut device_result = DeviceVec::<IcicleG2Projective>::device_malloc(1)
            .map_err(|e| MsmError::ExecutionFailed(format!("Device malloc failed: {:?}", e)))?;

        // Configure MSM - synchronous on default stream
        // Note: are_bases_montgomery_form = false because g2_affine_to_icicle uses to_repr()
        // which converts out of Montgomery form. Scalars remain in Montgomery form.
        let mut cfg = MSMConfig::default();
        cfg.are_scalars_montgomery_form = true;
        cfg.is_async = false;

        // Execute G2 MSM
        msm(
            HostSlice::from_slice(icicle_scalars),
            HostSlice::from_slice(&icicle_points),
            &cfg,
            &mut device_result[..],
        )
        .map_err(|e| MsmError::ExecutionFailed(format!("G2 MSM failed: {:?}", e)))?;

        // Copy result back to host
        let mut host_result = vec![IcicleG2Projective::zero(); 1];
        device_result
            .copy_to_host(HostSlice::from_mut_slice(&mut host_result))
            .map_err(|e| MsmError::ExecutionFailed(format!("Copy to host failed: {:?}", e)))?;

        #[cfg(feature = "trace-msm")]
        debug!("G2 MSM completed for {} points in {:?}", scalars.len(), start.elapsed());

        Ok(TypeConverter::icicle_to_g2_projective(&host_result[0]))
    }

    /// Compute G2 MSM with pre-uploaded device bases
    pub fn g2_msm_with_device_bases(
        &self,
        scalars: &[Scalar],
        device_bases: &DeviceVec<IcicleG2Affine>,
    ) -> Result<G2Projective, MsmError> {
        use icicle_runtime::set_device;

        if scalars.is_empty() {
            return Ok(G2Projective::identity());
        }

        if scalars.len() > device_bases.len() {
            return Err(MsmError::InvalidInput(format!(
                "More scalars ({}) than bases ({})",
                scalars.len(),
                device_bases.len()
            )));
        }

        // Set device context
        set_device(&self.device)
            .map_err(|e| MsmError::ExecutionFailed(format!("Failed to set device: {:?}", e)))?;

        #[cfg(feature = "trace-msm")]
        let start = std::time::Instant::now();

        // Zero-copy scalar conversion
        let icicle_scalars = TypeConverter::scalar_slice_as_icicle(scalars);

        // Allocate device result buffer
        let mut device_result = DeviceVec::<IcicleG2Projective>::device_malloc(1)
            .map_err(|e| MsmError::ExecutionFailed(format!("Device malloc failed: {:?}", e)))?;

        // Configure MSM - synchronous on default stream
        // CRITICAL: Both scalars AND bases are in Montgomery form!
        // ICICLE performance tuning parameters applied
        use crate::config::{precompute_factor, msm_window_size};
        let mut cfg = MSMConfig::default();
        cfg.are_scalars_montgomery_form = true;
        cfg.are_bases_montgomery_form = true;  // Bases pre-uploaded in Montgomery form
        cfg.is_async = false;
        cfg.precompute_factor = precompute_factor();
        if msm_window_size() > 0 {
            cfg.c = msm_window_size();
        }

        // Execute MSM with device bases - zero-copy, no conversion!
        msm(
            HostSlice::from_slice(icicle_scalars),
            &device_bases[..scalars.len()],
            &cfg,
            &mut device_result[..],
        )
        .map_err(|e| MsmError::ExecutionFailed(format!("G2 MSM failed: {:?}", e)))?;

        // Copy result back to host
        let mut host_result = vec![IcicleG2Projective::zero(); 1];
        device_result
            .copy_to_host(HostSlice::from_mut_slice(&mut host_result))
            .map_err(|e| MsmError::ExecutionFailed(format!("Copy to host failed: {:?}", e)))?;

        #[cfg(feature = "trace-msm")]
        debug!("G2 MSM (device bases) completed for {} points in {:?}", scalars.len(), start.elapsed());

        Ok(TypeConverter::icicle_to_g2_projective(&host_result[0]))
    }

    /// Warmup the GPU by running a small MSM
    ///
    /// Call this at application startup to pay initialization costs upfront.
    pub fn warmup(&self) -> Result<std::time::Duration, MsmError> {
        use ff::Field;
        use group::prime::PrimeCurveAffine;
        use std::time::Instant;

        let start = Instant::now();

        // Small warmup MSM
        let warmup_size = 256;
        let scalars: Vec<Scalar> = (0..warmup_size)
            .map(|i| {
                let mut s = Scalar::ONE;
                for _ in 0..i % 8 {
                    s = s.double();
                }
                s
            })
            .collect();
        let points: Vec<G1Affine> = (0..warmup_size).map(|_| G1Affine::generator()).collect();

        let _ = self.msm(&scalars, &points)?;

        let elapsed = start.elapsed();
        debug!("GPU MSM warmup complete in {:?}", elapsed);
        Ok(elapsed)
    }

    // =========================================================================
    // Async API (icicle-halo2 pattern)
    // =========================================================================

    /// Launch async G1 MSM with device bases, returns handle to wait on result.
    ///
    /// This follows the icicle-halo2 pattern of creating a stream per operation:
    /// ```rust,ignore
    /// // From icicle-halo2 evaluation.rs:
    /// let mut stream = IcicleStream::create().unwrap();
    /// let mut d_result = DeviceVec::device_malloc_async(size, &stream).unwrap();
    /// cfg.stream_handle = stream.into();
    /// cfg.is_async = true;
    /// // ... launch operations ...
    /// stream.synchronize().unwrap();
    /// stream.destroy().unwrap();
    /// ```
    ///
    /// # Arguments
    /// * `scalars` - Scalar multipliers (in Montgomery form)
    /// * `device_bases` - G1 points already in GPU memory
    ///
    /// # Returns
    /// A handle that can be waited on to get the result
    ///
    /// # Example
    /// ```rust,ignore
    /// let handle = ctx.msm_async(&scalars, &device_bases)?;
    /// // ... do other work ...
    /// let result = handle.wait()?;
    /// ```
    pub fn msm_async(
        &self,
        scalars: &[Scalar],
        device_bases: &DeviceVec<IcicleG1Affine>,
    ) -> Result<MsmHandle, MsmError> {
        use icicle_runtime::set_device;

        if scalars.is_empty() {
            return Ok(MsmHandle::identity());
        }

        if scalars.len() > device_bases.len() {
            return Err(MsmError::InvalidInput(format!(
                "More scalars ({}) than bases ({})",
                scalars.len(),
                device_bases.len()
            )));
        }

        // Set device context
        set_device(&self.device)
            .map_err(|e| MsmError::ExecutionFailed(format!("Failed to set device: {:?}", e)))?;

        // Create stream for this operation (icicle-halo2 pattern)
        let stream = ManagedStream::create()
            .map_err(|e| MsmError::ExecutionFailed(format!("Stream creation failed: {:?}", e)))?;

        #[cfg(feature = "trace-msm")]
        let _start = std::time::Instant::now();

        // Zero-copy scalar conversion
        let icicle_scalars = TypeConverter::scalar_slice_as_icicle(scalars);

        // Allocate device result buffer (async)
        let mut device_result = DeviceVec::<IcicleG1Projective>::device_malloc_async(1, stream.as_ref())
            .map_err(|e| MsmError::ExecutionFailed(format!("Device malloc failed: {:?}", e)))?;

        // Configure MSM - async on our stream
        // CRITICAL: Both scalars AND bases are in Montgomery form
        // ICICLE performance tuning parameters applied
        use crate::config::{precompute_factor, msm_window_size};
        let mut cfg = MSMConfig::default();
        cfg.stream_handle = stream.as_ref().into();
        cfg.are_scalars_montgomery_form = true;
        cfg.are_bases_montgomery_form = true;
        cfg.is_async = true;
        cfg.precompute_factor = precompute_factor();
        if msm_window_size() > 0 {
            cfg.c = msm_window_size();
        }

        // Launch async MSM with device bases
        msm(
            HostSlice::from_slice(icicle_scalars),
            &device_bases[..scalars.len()],
            &cfg,
            &mut device_result[..],
        )
        .map_err(|e| MsmError::ExecutionFailed(format!("MSM launch failed: {:?}", e)))?;

        #[cfg(feature = "trace-msm")]
        debug!("G1 MSM async launched for {} points", scalars.len());

        Ok(MsmHandle {
            stream,
            device_result,
            is_identity: false,
        })
    }

    /// Launch async G2 MSM with device bases
    pub fn g2_msm_async(
        &self,
        scalars: &[Scalar],
        device_bases: &DeviceVec<IcicleG2Affine>,
    ) -> Result<G2MsmHandle, MsmError> {
        use icicle_runtime::set_device;

        if scalars.is_empty() {
            return Ok(G2MsmHandle::identity());
        }

        if scalars.len() > device_bases.len() {
            return Err(MsmError::InvalidInput(format!(
                "More scalars ({}) than bases ({})",
                scalars.len(),
                device_bases.len()
            )));
        }

        set_device(&self.device)
            .map_err(|e| MsmError::ExecutionFailed(format!("Failed to set device: {:?}", e)))?;

        let stream = ManagedStream::create()
            .map_err(|e| MsmError::ExecutionFailed(format!("Stream creation failed: {:?}", e)))?;

        let icicle_scalars = TypeConverter::scalar_slice_as_icicle(scalars);

        let mut device_result = DeviceVec::<IcicleG2Projective>::device_malloc_async(1, stream.as_ref())
            .map_err(|e| MsmError::ExecutionFailed(format!("Device malloc failed: {:?}", e)))?;

        // CRITICAL: Both scalars AND bases are in Montgomery form
        // ICICLE performance tuning parameters applied
        use crate::config::{precompute_factor, msm_window_size};
        let mut cfg = MSMConfig::default();
        cfg.stream_handle = stream.as_ref().into();
        cfg.are_scalars_montgomery_form = true;
        cfg.are_bases_montgomery_form = true;
        cfg.is_async = true;
        cfg.precompute_factor = precompute_factor();
        if msm_window_size() > 0 {
            cfg.c = msm_window_size();
        }

        msm(
            HostSlice::from_slice(icicle_scalars),
            &device_bases[..scalars.len()],
            &cfg,
            &mut device_result[..],
        )
        .map_err(|e| MsmError::ExecutionFailed(format!("G2 MSM launch failed: {:?}", e)))?;

        Ok(G2MsmHandle {
            stream,
            device_result,
            is_identity: false,
        })
    }
}

// =============================================================================
// Async Handles (icicle-halo2 pattern)
// =============================================================================

/// Handle for an in-flight async G1 MSM operation.
///
/// This implements the icicle-halo2 pattern where each async operation owns
/// its stream and result buffer. Call `wait()` to synchronize and get the result.
///
/// # Reference
///
/// From icicle-halo2:
/// ```rust,ignore
/// stream.synchronize().unwrap();
/// msm_results.copy_to_host_async(HostSlice::from_mut_slice(&mut result), stream).unwrap();
/// stream.destroy().unwrap();
/// ```
#[cfg(feature = "gpu")]
pub struct MsmHandle {
    /// Owned stream for this operation
    stream: ManagedStream,
    /// Device buffer holding the result
    device_result: DeviceVec<IcicleG1Projective>,
    /// True if this represents identity (empty input)
    is_identity: bool,
}

#[cfg(feature = "gpu")]
impl std::fmt::Debug for MsmHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MsmHandle")
            .field("stream", &self.stream)
            .field("is_identity", &self.is_identity)
            .finish()
    }
}

#[cfg(feature = "gpu")]
impl MsmHandle {
    /// Create a handle representing identity result (for empty input)
    fn identity() -> Self {
        // Note: We allocate 1 element instead of 0 because CUDA doesn't allow zero-size allocations.
        // The is_identity flag ensures we return identity without reading this buffer.
        Self {
            stream: ManagedStream::default_stream(),
            device_result: DeviceVec::<IcicleG1Projective>::device_malloc(1).unwrap(),
            is_identity: true,
        }
    }

    /// Wait for the MSM to complete and return the result.
    ///
    /// This synchronizes the stream, copies the result to host, and cleans up.
    /// The stream is automatically destroyed.
    ///
    /// # Example
    /// ```rust,ignore
    /// let handle = ctx.msm_async(&scalars, &device_bases)?;
    /// // ... do other work ...
    /// let result = handle.wait()?;
    /// ```
    pub fn wait(mut self) -> Result<G1Projective, MsmError> {
        if self.is_identity {
            return Ok(G1Projective::identity());
        }

        // Synchronize stream (wait for GPU to finish)
        self.stream.synchronize()
            .map_err(|e| MsmError::ExecutionFailed(format!("Stream sync failed: {:?}", e)))?;

        // Copy result to host
        let mut host_result = vec![IcicleG1Projective::zero(); 1];
        self.device_result
            .copy_to_host(HostSlice::from_mut_slice(&mut host_result))
            .map_err(|e| MsmError::ExecutionFailed(format!("Copy to host failed: {:?}", e)))?;

        // Stream is destroyed automatically by ManagedStream::Drop

        Ok(TypeConverter::icicle_to_g1_projective(&host_result[0]))
    }
}

/// Handle for an in-flight async G2 MSM operation.
#[cfg(feature = "gpu")]
pub struct G2MsmHandle {
    stream: ManagedStream,
    device_result: DeviceVec<IcicleG2Projective>,
    is_identity: bool,
}

#[cfg(feature = "gpu")]
impl std::fmt::Debug for G2MsmHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("G2MsmHandle")
            .field("stream", &self.stream)
            .field("is_identity", &self.is_identity)
            .finish()
    }
}

#[cfg(feature = "gpu")]
impl G2MsmHandle {
    fn identity() -> Self {
        // Note: We allocate 1 element instead of 0 because CUDA doesn't allow zero-size allocations.
        Self {
            stream: ManagedStream::default_stream(),
            device_result: DeviceVec::<IcicleG2Projective>::device_malloc(1).unwrap(),
            is_identity: true,
        }
    }

    /// Wait for the G2 MSM to complete and return the result.
    pub fn wait(mut self) -> Result<G2Projective, MsmError> {
        if self.is_identity {
            return Ok(G2Projective::identity());
        }

        self.stream.synchronize()
            .map_err(|e| MsmError::ExecutionFailed(format!("Stream sync failed: {:?}", e)))?;

        let mut host_result = vec![IcicleG2Projective::zero(); 1];
        self.device_result
            .copy_to_host(HostSlice::from_mut_slice(&mut host_result))
            .map_err(|e| MsmError::ExecutionFailed(format!("Copy to host failed: {:?}", e)))?;

        Ok(TypeConverter::icicle_to_g2_projective(&host_result[0]))
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
#[cfg(feature = "gpu")]
mod tests {
    use super::*;
    use ff::Field;
    use group::prime::PrimeCurveAffine;

    #[test]
    fn test_msm_context_creation() {
        let ctx = GpuMsmContext::new();
        assert!(ctx.is_ok(), "Should create MSM context");
    }

    #[test]
    fn test_msm_empty() {
        let ctx = GpuMsmContext::new().expect("Failed to create context");
        let result = ctx.msm(&[], &[]).unwrap();
        assert_eq!(result, G1Projective::identity());
    }

    #[test]
    fn test_msm_single_point() {
        let ctx = GpuMsmContext::new().expect("Failed to create context");

        let scalar = Scalar::from(5u64);
        let point = G1Affine::generator();

        let result = ctx.msm(&[scalar], &[point]).expect("MSM failed");

        // Expected: 5 * G
        let expected = G1Projective::from(point) * scalar;
        assert_eq!(result, expected);
    }

    #[test]
    fn test_msm_multiple_points() {
        let ctx = GpuMsmContext::new().expect("Failed to create context");

        let n = 64;
        let scalars: Vec<Scalar> = (1..=n).map(|i| Scalar::from(i as u64)).collect();
        let points: Vec<G1Affine> = (0..n).map(|_| G1Affine::generator()).collect();

        let result = ctx.msm(&scalars, &points).expect("MSM failed");

        // Expected: sum(i * G) for i = 1..n = (n*(n+1)/2) * G
        let sum = n * (n + 1) / 2;
        let expected = G1Projective::from(G1Affine::generator()) * Scalar::from(sum as u64);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_msm_size_mismatch() {
        let ctx = GpuMsmContext::new().expect("Failed to create context");

        let scalars = vec![Scalar::ONE];
        let points = vec![];

        let result = ctx.msm(&scalars, &points);
        assert!(matches!(result, Err(MsmError::InvalidInput(_))));
    }

    /// Test async MSM with device bases
    #[test]
    fn test_msm_async_with_device_bases() {
        let ctx = GpuMsmContext::new().expect("Failed to create context");

        let n = 64;
        let scalars: Vec<Scalar> = (1..=n).map(|i| Scalar::from(i as u64)).collect();
        let points: Vec<G1Affine> = (0..n).map(|_| G1Affine::generator()).collect();

        // Upload points to GPU
        let icicle_points = TypeConverter::g1_affine_slice_to_icicle_vec(&points);
        let mut device_bases = DeviceVec::<IcicleG1Affine>::device_malloc(n as usize)
            .expect("Device malloc failed");
        device_bases
            .copy_from_host(HostSlice::from_slice(&icicle_points))
            .expect("Copy to device failed");

        // Launch async MSM
        let handle = ctx.msm_async(&scalars, &device_bases).expect("Async MSM launch failed");

        // Wait for result
        let result = handle.wait().expect("Async MSM wait failed");

        // Expected: sum(i * G) for i = 1..n = (n*(n+1)/2) * G
        let sum = n * (n + 1) / 2;
        let expected = G1Projective::from(G1Affine::generator()) * Scalar::from(sum as u64);
        assert_eq!(result, expected);
    }

    /// Test async MSM with empty input returns identity
    #[test]
    fn test_msm_async_empty() {
        let ctx = GpuMsmContext::new().expect("Failed to create context");

        // Create minimal device bases (CUDA doesn't allow zero-size allocations)
        // The msm_async function checks for empty scalars before using bases
        let device_bases = DeviceVec::<IcicleG1Affine>::device_malloc(1)
            .expect("Device malloc failed");

        // Launch async MSM with empty scalars
        let handle = ctx.msm_async(&[], &device_bases).expect("Async MSM launch failed");
        let result = handle.wait().expect("Async MSM wait failed");

        assert_eq!(result, G1Projective::identity());
    }

    /// Test MsmHandle debug implementation
    #[test]
    fn test_msm_handle_debug() {
        let ctx = GpuMsmContext::new().expect("Failed to create context");

        let n = 16;
        let scalars: Vec<Scalar> = (1..=n).map(|i| Scalar::from(i as u64)).collect();
        let points: Vec<G1Affine> = (0..n).map(|_| G1Affine::generator()).collect();

        let icicle_points = TypeConverter::g1_affine_slice_to_icicle_vec(&points);
        let mut device_bases = DeviceVec::<IcicleG1Affine>::device_malloc(n as usize)
            .expect("Device malloc failed");
        device_bases
            .copy_from_host(HostSlice::from_slice(&icicle_points))
            .expect("Copy to device failed");

        let handle = ctx.msm_async(&scalars, &device_bases).expect("Async MSM launch failed");

        // Test Debug implementation
        let debug_str = format!("{:?}", handle);
        assert!(debug_str.contains("MsmHandle"));

        // Consume handle
        let _ = handle.wait();
    }
}
