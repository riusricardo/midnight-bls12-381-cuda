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

//! GPU-Accelerated Number Theoretic Transform (NTT)
//!
//! This module provides GPU-accelerated NTT operations using ICICLE's CUDA backend.
//! NTT is the core operation for converting between coefficient and evaluation 
//! (Lagrange) representations of polynomials.
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
//! - **icicle-halo2**: https://github.com/ingonyama-zk/halo2/blob/main/halo2_proofs/src/icicle.rs
//! - **ICICLE Rust Guide**: https://dev.ingonyama.com/start/programmers_guide/rust
//!
//! # Sync Usage
//!
//! ```rust,ignore
//! use midnight_proofs::gpu::ntt::GpuNttContext;
//!
//! let ctx = GpuNttContext::new(16)?;  // K=16, domain size = 2^16
//! let evaluations = ctx.forward_ntt(&coefficients)?;  // Blocking
//! let coefficients = ctx.inverse_ntt(&evaluations)?;   // Blocking
//! ```
//!
//! # Async Usage (icicle-halo2 pattern)
//!
//! ```rust,ignore
//! // Launch async NTT
//! let handle = ctx.forward_ntt_async(&coefficients)?;
//! // ... do other work while GPU computes ...
//! let evaluations = handle.wait()?;
//! ```

use crate::GpuError;

#[cfg(feature = "gpu")]
use crate::{TypeConverter, stream::ManagedStream};

#[cfg(feature = "gpu")]
use midnight_curves::Fq as Scalar;

#[cfg(feature = "gpu")]
use once_cell::sync::OnceCell;
use tracing::debug;
#[cfg(feature = "trace-fft")]
use tracing::info;

#[cfg(feature = "gpu")]
use icicle_bls12_381::curve::ScalarField as IcicleScalar;
#[cfg(feature = "gpu")]
use icicle_core::bignum::BigNum;
#[cfg(feature = "gpu")]
use icicle_core::ntt::{ntt, ntt_inplace, NTTConfig, NTTDir, NTTDomain, NTTInitDomainConfig, Ordering};
#[cfg(feature = "gpu")]
use icicle_runtime::{
    Device, 
    memory::{DeviceVec, HostSlice, HostOrDeviceSlice},
};

/// Errors specific to NTT operations
#[derive(Debug)]
pub enum NttError {
    /// Domain initialization failed
    DomainInitFailed(String),
    /// NTT execution failed
    ExecutionFailed(String),
    /// Invalid input size
    InvalidSize(String),
    /// GPU not available
    GpuNotAvailable,
    /// Underlying GPU error
    GpuError(GpuError),
}

impl std::fmt::Display for NttError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NttError::DomainInitFailed(s) => write!(f, "NTT domain init failed: {}", s),
            NttError::ExecutionFailed(s) => write!(f, "NTT execution failed: {}", s),
            NttError::InvalidSize(s) => write!(f, "Invalid NTT size: {}", s),
            NttError::GpuNotAvailable => write!(f, "GPU not available for NTT"),
            NttError::GpuError(e) => write!(f, "GPU error in NTT: {}", e),
        }
    }
}

impl std::error::Error for NttError {}

impl From<GpuError> for NttError {
    fn from(e: GpuError) -> Self {
        NttError::GpuError(e)
    }
}

/// Global NTT domain state - initialized once per max_log_size
/// 
/// ICICLE's NTT domain is a global singleton per field. We track whether
/// it has been initialized and for what maximum size.
#[cfg(feature = "gpu")]
static NTT_DOMAIN_INITIALIZED: OnceCell<u32> = OnceCell::new();

/// GPU NTT Context for polynomial transformations
/// 
/// This context manages NTT operations on GPU, including:
/// - Domain initialization with roots of unity
/// - Forward NTT (coefficients -> evaluations)
/// - Inverse NTT (evaluations -> coefficients)
/// - In-place operations for memory efficiency
#[cfg(feature = "gpu")]
pub struct GpuNttContext {
    /// Maximum log size this context supports (2^max_log_size elements)
    max_log_size: u32,
    /// Device reference
    device: Device,
}

// Implement Send and Sync for GpuNttContext
// Safe because Device is just an identifier (string + int)
#[cfg(feature = "gpu")]
unsafe impl Send for GpuNttContext {}
#[cfg(feature = "gpu")]
unsafe impl Sync for GpuNttContext {}

#[cfg(feature = "gpu")]
impl std::fmt::Debug for GpuNttContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuNttContext")
            .field("max_log_size", &self.max_log_size)
            .field("device", &"<Device>")
            .finish()
    }
}

#[cfg(feature = "gpu")]
impl GpuNttContext {
    /// Create a new GPU NTT context for domains up to size 2^max_log_size
    /// 
    /// This initializes the ICICLE NTT domain with the appropriate roots of unity.
    /// The domain is a global resource and will be reused for all NTT operations
    /// of size <= 2^max_log_size.
    /// 
    /// # Arguments
    /// * `max_log_size` - Maximum K value (domain size = 2^K). Typically 16-20 for circuits.
    /// 
    /// # Example
    /// ```rust,ignore
    /// let ctx = GpuNttContext::new(16)?; // Supports up to 2^16 = 65536 elements
    /// ```
    pub fn new(max_log_size: u32) -> Result<Self, NttError> {
        use crate::backend::ensure_backend_loaded;
        use icicle_runtime::set_device;
        
        // Ensure ICICLE backend is loaded
        ensure_backend_loaded()
            .map_err(|e| NttError::GpuError(e))?;
        
        // Set device context
        let device = Device::new("CUDA", 0);
        set_device(&device)
            .map_err(|e| NttError::DomainInitFailed(format!("Failed to set device: {:?}", e)))?;
        
        // Initialize NTT domain if not already done (or if larger size needed)
        Self::ensure_domain_initialized(max_log_size)?;
        
        debug!("GpuNttContext created for max_log_size={}", max_log_size);
        
        Ok(Self {
            max_log_size,
            device,
        })
    }
    
    /// Ensure NTT domain is initialized for at least the given size
    fn ensure_domain_initialized(max_log_size: u32) -> Result<(), NttError> {
        // Check if already initialized with sufficient size
        if let Some(&current_size) = NTT_DOMAIN_INITIALIZED.get() {
            if current_size >= max_log_size {
                debug!("NTT domain already initialized for size 2^{}", current_size);
                return Ok(());
            }
            // Need larger domain - release and reinitialize
            debug!("Releasing NTT domain to reinitialize with larger size");
            <IcicleScalar as NTTDomain<IcicleScalar>>::release_domain()
                .map_err(|e| NttError::DomainInitFailed(format!("Failed to release domain: {:?}", e)))?;
        }
        
        // Get primitive root of unity for the domain
        let max_size = 1u64 << max_log_size;
        let primitive_root = <IcicleScalar as NTTDomain<IcicleScalar>>::get_root_of_unity(max_size)
            .map_err(|e| NttError::DomainInitFailed(format!("Failed to get root of unity: {:?}", e)))?;
        
        // Initialize domain with roots of unity
        let init_cfg = NTTInitDomainConfig::default();
        
        #[cfg(feature = "trace-fft")]
        let start = std::time::Instant::now();
        
        <IcicleScalar as NTTDomain<IcicleScalar>>::initialize_domain(primitive_root, &init_cfg)
            .map_err(|e| NttError::DomainInitFailed(format!("Failed to initialize domain: {:?}", e)))?;
        
        #[cfg(feature = "trace-fft")]
        info!("NTT domain initialized for size 2^{} in {:?}", max_log_size, start.elapsed());
        
        // Update global state (only once - OnceCell)
        let _ = NTT_DOMAIN_INITIALIZED.set(max_log_size);
        
        Ok(())
    }
    
    /// Forward NTT: Convert coefficients to evaluations
    /// 
    /// Computes polynomial evaluations at powers of the root of unity.
    /// 
    /// # Arguments
    /// * `coefficients` - Polynomial coefficients (length must be power of 2)
    /// 
    /// # Returns
    /// Polynomial evaluations at omega^0, omega^1, ..., omega^(n-1)
    pub fn forward_ntt(&self, coefficients: &[Scalar]) -> Result<Vec<Scalar>, NttError> {
        self.ntt_internal(coefficients, NTTDir::kForward)
    }
    
    /// Inverse NTT: Convert evaluations to coefficients
    /// 
    /// Recovers polynomial coefficients from evaluations.
    /// 
    /// # Arguments
    /// * `evaluations` - Polynomial evaluations (length must be power of 2)
    /// 
    /// # Returns
    /// Polynomial coefficients
    pub fn inverse_ntt(&self, evaluations: &[Scalar]) -> Result<Vec<Scalar>, NttError> {
        self.ntt_internal(evaluations, NTTDir::kInverse)
    }
    
    /// Forward NTT in-place: Convert coefficients to evaluations without allocation
    /// 
    /// # Arguments
    /// * `data` - Coefficients on input, evaluations on output
    pub fn forward_ntt_inplace(&self, data: &mut [Scalar]) -> Result<(), NttError> {
        self.ntt_inplace_internal(data, NTTDir::kForward)
    }
    
    /// Inverse NTT in-place: Convert evaluations to coefficients without allocation
    /// 
    /// # Arguments
    /// * `data` - Evaluations on input, coefficients on output
    pub fn inverse_ntt_inplace(&self, data: &mut [Scalar]) -> Result<(), NttError> {
        self.ntt_inplace_internal(data, NTTDir::kInverse)
    }
    
    /// Internal NTT implementation (allocating version)
    fn ntt_internal(&self, input: &[Scalar], direction: NTTDir) -> Result<Vec<Scalar>, NttError> {
        use icicle_runtime::set_device;
        
        let n = input.len();
        if !n.is_power_of_two() {
            return Err(NttError::InvalidSize(format!(
                "NTT size must be power of 2, got {}", n
            )));
        }
        
        let log_n = n.trailing_zeros();
        if log_n > self.max_log_size {
            return Err(NttError::InvalidSize(format!(
                "NTT size 2^{} exceeds max 2^{}", log_n, self.max_log_size
            )));
        }
        
        // Set device context (important for multi-threaded scenarios)
        set_device(&self.device)
            .map_err(|e| NttError::ExecutionFailed(format!("Failed to set device: {:?}", e)))?;
        
        #[cfg(feature = "trace-fft")]
        let start = std::time::Instant::now();
        
        // Zero-copy transmute to ICICLE scalars
        let icicle_input = TypeConverter::scalar_slice_as_icicle(input);
        
        // Allocate output buffer
        let mut output = vec![<IcicleScalar as BigNum>::zero(); n];
        
        // Configure NTT
        let mut cfg = NTTConfig::<IcicleScalar>::default();
        cfg.ordering = Ordering::kNN;  // Natural-Natural ordering
        cfg.are_inputs_on_device = false;
        cfg.are_outputs_on_device = false;
        cfg.is_async = false;
        
        // Execute NTT
        ntt(
            HostSlice::from_slice(icicle_input),
            direction,
            &cfg,
            HostSlice::from_mut_slice(&mut output),
        ).map_err(|e| NttError::ExecutionFailed(format!("NTT failed: {:?}", e)))?;
        
        #[cfg(feature = "trace-fft")]
        debug!("NTT {:?} completed for {} elements in {:?}", direction, n, start.elapsed());
        
        // Zero-copy transmute back to midnight scalars
        let result = TypeConverter::icicle_slice_as_scalar(&output);
        Ok(result.to_vec())
    }
    
    /// Internal in-place NTT implementation
    fn ntt_inplace_internal(&self, data: &mut [Scalar], direction: NTTDir) -> Result<(), NttError> {
        use icicle_runtime::set_device;
        
        let n = data.len();
        if !n.is_power_of_two() {
            return Err(NttError::InvalidSize(format!(
                "NTT size must be power of 2, got {}", n
            )));
        }
        
        let log_n = n.trailing_zeros();
        if log_n > self.max_log_size {
            return Err(NttError::InvalidSize(format!(
                "NTT size 2^{} exceeds max 2^{}", log_n, self.max_log_size
            )));
        }
        
        // Set device context
        set_device(&self.device)
            .map_err(|e| NttError::ExecutionFailed(format!("Failed to set device: {:?}", e)))?;
        
        #[cfg(feature = "trace-fft")]
        let start = std::time::Instant::now();
        
        // Zero-copy transmute - this is the key optimization!
        // We can operate directly on the midnight scalar data
        let icicle_data = TypeConverter::scalar_slice_as_icicle_mut(data);
        
        // Configure NTT
        let mut cfg = NTTConfig::<IcicleScalar>::default();
        cfg.ordering = Ordering::kNN;
        cfg.are_inputs_on_device = false;
        cfg.are_outputs_on_device = false;
        cfg.is_async = false;
        
        // Execute in-place NTT
        ntt_inplace(
            HostSlice::from_mut_slice(icicle_data),
            direction,
            &cfg,
        ).map_err(|e| NttError::ExecutionFailed(format!("NTT inplace failed: {:?}", e)))?;
        
        #[cfg(feature = "trace-fft")]
        debug!("NTT inplace {:?} completed for {} elements in {:?}", direction, n, start.elapsed());
        
        Ok(())
    }
    
    /// Execute NTT on data already in GPU memory
    /// 
    /// This is the most efficient path when data is already on device,
    /// avoiding any PCIe transfer overhead.
    /// 
    /// # Arguments
    /// * `device_data` - Mutable reference to data in GPU memory
    /// * `direction` - Forward or inverse NTT
    pub fn ntt_on_device(
        &self, 
        device_data: &mut DeviceVec<IcicleScalar>,
        direction: NTTDir,
    ) -> Result<(), NttError> {
        use icicle_runtime::set_device;
        
        let n = device_data.len();
        if !n.is_power_of_two() {
            return Err(NttError::InvalidSize(format!(
                "NTT size must be power of 2, got {}", n
            )));
        }
        
        // Set device context
        set_device(&self.device)
            .map_err(|e| NttError::ExecutionFailed(format!("Failed to set device: {:?}", e)))?;
        
        #[cfg(feature = "trace-fft")]
        let start = std::time::Instant::now();
        
        // Configure NTT for device data - synchronous on default stream
        let mut cfg = NTTConfig::<IcicleScalar>::default();
        cfg.ordering = Ordering::kNN;
        cfg.are_inputs_on_device = true;
        cfg.are_outputs_on_device = true;
        cfg.is_async = false;
        
        // Execute in-place on device
        ntt_inplace(
            &mut device_data[..],
            direction,
            &cfg,
        ).map_err(|e| NttError::ExecutionFailed(format!("Device NTT failed: {:?}", e)))?;
        
        #[cfg(feature = "trace-fft")]
        debug!("Device NTT {:?} completed for {} elements in {:?}", direction, n, start.elapsed());
        
        Ok(())
    }
    
    /// Get the maximum log size this context supports
    pub fn max_log_size(&self) -> u32 {
        self.max_log_size
    }

    // =========================================================================
    // Async API (icicle-halo2 pattern)
    // =========================================================================

    /// Launch async forward NTT, returns handle to wait on result.
    ///
    /// This follows the icicle-halo2 pattern of creating a stream per operation:
    /// ```rust,ignore
    /// // From icicle-halo2 icicle.rs:
    /// let dir = if inverse { NTTDir::kInverse } else { NTTDir::kForward };
    /// let mut cfg = NTTConfig::<ScalarField>::default();
    /// cfg.stream_handle = stream.into();
    /// cfg.is_async = true;
    /// ```
    ///
    /// # Arguments
    /// * `coefficients` - Polynomial coefficients (length must be power of 2)
    ///
    /// # Returns
    /// A handle that can be waited on to get the evaluations
    pub fn forward_ntt_async(&self, coefficients: &[Scalar]) -> Result<NttHandle, NttError> {
        self.ntt_async_internal(coefficients, NTTDir::kForward)
    }

    /// Launch async inverse NTT, returns handle to wait on result.
    pub fn inverse_ntt_async(&self, evaluations: &[Scalar]) -> Result<NttHandle, NttError> {
        self.ntt_async_internal(evaluations, NTTDir::kInverse)
    }

    /// Internal async NTT implementation
    fn ntt_async_internal(&self, input: &[Scalar], direction: NTTDir) -> Result<NttHandle, NttError> {
        use icicle_runtime::set_device;

        let n = input.len();
        if !n.is_power_of_two() {
            return Err(NttError::InvalidSize(format!(
                "NTT size must be power of 2, got {}", n
            )));
        }

        let log_n = n.trailing_zeros();
        if log_n > self.max_log_size {
            return Err(NttError::InvalidSize(format!(
                "NTT size 2^{} exceeds max 2^{}", log_n, self.max_log_size
            )));
        }

        // Set device context
        set_device(&self.device)
            .map_err(|e| NttError::ExecutionFailed(format!("Failed to set device: {:?}", e)))?;

        // Create stream for this operation (icicle-halo2 pattern)
        let stream = ManagedStream::create()
            .map_err(|e| NttError::ExecutionFailed(format!("Stream creation failed: {:?}", e)))?;

        #[cfg(feature = "trace-fft")]
        debug!("Launching async NTT {:?} for {} elements", direction, n);

        // Zero-copy transmute to ICICLE scalars
        let icicle_input = TypeConverter::scalar_slice_as_icicle(input);

        // Allocate device buffer for input (async)
        let mut device_data = DeviceVec::<IcicleScalar>::device_malloc_async(n, stream.as_ref())
            .map_err(|e| NttError::ExecutionFailed(format!("Device malloc failed: {:?}", e)))?;

        // Copy input to device (async)
        device_data
            .copy_from_host_async(HostSlice::from_slice(icicle_input), stream.as_ref())
            .map_err(|e| NttError::ExecutionFailed(format!("Copy to device failed: {:?}", e)))?;

        // Configure NTT - async on our stream
        let mut cfg = NTTConfig::<IcicleScalar>::default();
        cfg.stream_handle = stream.as_ref().into();
        cfg.ordering = Ordering::kNN;
        cfg.are_inputs_on_device = true;
        cfg.are_outputs_on_device = true;
        cfg.is_async = true;

        // Launch async NTT (in-place on device)
        ntt_inplace(
            &mut device_data[..],
            direction,
            &cfg,
        ).map_err(|e| NttError::ExecutionFailed(format!("NTT launch failed: {:?}", e)))?;

        Ok(NttHandle {
            stream,
            device_data,
            size: n,
        })
    }
}

// =============================================================================
// Async Handle (icicle-halo2 pattern)
// =============================================================================

/// Handle for an in-flight async NTT operation.
///
/// This implements the icicle-halo2 pattern where each async operation owns
/// its stream and result buffer. Call `wait()` to synchronize and get the result.
///
/// # Reference
///
/// From icicle-halo2 `icicle.rs`:
/// ```rust,ignore
/// ntt_inplace::<ScalarField, ScalarField>(&mut icicle_scalars, dir, &cfg).unwrap();
/// let c_scalars = &c_scalars_from_device_vec::<G>(&mut icicle_scalars, stream)[..];
/// ```
#[cfg(feature = "gpu")]
pub struct NttHandle {
    /// Owned stream for this operation
    stream: ManagedStream,
    /// Device buffer holding input/output (in-place)
    device_data: DeviceVec<IcicleScalar>,
    /// Size of the NTT
    size: usize,
}

#[cfg(feature = "gpu")]
impl std::fmt::Debug for NttHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NttHandle")
            .field("stream", &self.stream)
            .field("size", &self.size)
            .finish()
    }
}

#[cfg(feature = "gpu")]
impl NttHandle {
    /// Wait for the NTT to complete and return the result.
    ///
    /// This synchronizes the stream, copies the result to host, and cleans up.
    /// The stream is automatically destroyed when the handle is dropped.
    ///
    /// # Example
    /// ```rust,ignore
    /// let handle = ctx.forward_ntt_async(&coefficients)?;
    /// // ... do other work ...
    /// let evaluations = handle.wait()?;
    /// ```
    pub fn wait(mut self) -> Result<Vec<Scalar>, NttError> {
        // Synchronize stream (wait for GPU to finish)
        self.stream.synchronize()
            .map_err(|e| NttError::ExecutionFailed(format!("Stream sync failed: {:?}", e)))?;

        // Allocate host buffer
        let mut host_result = vec![<IcicleScalar as BigNum>::zero(); self.size];

        // Copy result to host
        self.device_data
            .copy_to_host(HostSlice::from_mut_slice(&mut host_result))
            .map_err(|e| NttError::ExecutionFailed(format!("Copy to host failed: {:?}", e)))?;

        // Zero-copy transmute back to midnight scalars
        let result = TypeConverter::icicle_slice_as_scalar(&host_result);
        Ok(result.to_vec())
    }

    /// Get the size of this NTT operation
    pub fn size(&self) -> usize {
        self.size
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
    
    /// Test NTT context creation
    #[test]
    fn test_ntt_context_creation() {
        let ctx = GpuNttContext::new(12);
        assert!(ctx.is_ok(), "Should create NTT context for K=12");
        
        let ctx = ctx.unwrap();
        assert_eq!(ctx.max_log_size(), 12);
    }
    
    /// Test forward/inverse NTT roundtrip
    #[test]
    fn test_ntt_roundtrip() {
        let ctx = GpuNttContext::new(10).expect("Failed to create context");
        
        // Create test polynomial
        let n = 1 << 8; // 256 elements
        let original: Vec<Scalar> = (0..n).map(|i| Scalar::from(i as u64 + 1)).collect();
        
        // Forward NTT
        let evaluations = ctx.forward_ntt(&original).expect("Forward NTT failed");
        assert_eq!(evaluations.len(), n);
        
        // Inverse NTT
        let recovered = ctx.inverse_ntt(&evaluations).expect("Inverse NTT failed");
        assert_eq!(recovered.len(), n);
        
        // Verify roundtrip
        for (orig, rec) in original.iter().zip(recovered.iter()) {
            assert_eq!(*orig, *rec, "NTT roundtrip should preserve values");
        }
    }
    
    /// Test in-place NTT
    #[test]
    fn test_ntt_inplace() {
        let ctx = GpuNttContext::new(10).expect("Failed to create context");
        
        let n = 1 << 8;
        let original: Vec<Scalar> = (0..n).map(|i| Scalar::from(i as u64 + 1)).collect();
        let mut data = original.clone();
        
        // Forward in-place
        ctx.forward_ntt_inplace(&mut data).expect("Forward NTT inplace failed");
        
        // Inverse in-place
        ctx.inverse_ntt_inplace(&mut data).expect("Inverse NTT inplace failed");
        
        // Verify roundtrip
        for (orig, rec) in original.iter().zip(data.iter()) {
            assert_eq!(*orig, *rec, "In-place NTT roundtrip should preserve values");
        }
    }
    
    /// Test NTT with identity polynomial (all zeros except first coefficient)
    #[test]
    fn test_ntt_identity() {
        let ctx = GpuNttContext::new(8).expect("Failed to create context");
        
        let n = 1 << 4; // 16 elements
        let mut coeffs = vec![Scalar::ZERO; n];
        coeffs[0] = Scalar::ONE;
        
        let evaluations = ctx.forward_ntt(&coeffs).expect("Forward NTT failed");
        
        // For constant polynomial c₀, all evaluations should equal c₀
        for eval in evaluations.iter() {
            assert_eq!(*eval, Scalar::ONE, "Constant polynomial should have constant evaluations");
        }
    }

    /// Test async forward/inverse NTT roundtrip
    #[test]
    fn test_ntt_async_roundtrip() {
        let ctx = GpuNttContext::new(10).expect("Failed to create context");
        
        // Create test polynomial
        let n = 1 << 8; // 256 elements
        let original: Vec<Scalar> = (0..n).map(|i| Scalar::from(i as u64 + 1)).collect();
        
        // Launch async forward NTT
        let handle = ctx.forward_ntt_async(&original).expect("Async forward NTT launch failed");
        
        // Wait for result
        let evaluations = handle.wait().expect("Async forward NTT wait failed");
        assert_eq!(evaluations.len(), n);
        
        // Launch async inverse NTT
        let handle = ctx.inverse_ntt_async(&evaluations).expect("Async inverse NTT launch failed");
        let recovered = handle.wait().expect("Async inverse NTT wait failed");
        assert_eq!(recovered.len(), n);
        
        // Verify roundtrip
        for (orig, rec) in original.iter().zip(recovered.iter()) {
            assert_eq!(*orig, *rec, "Async NTT roundtrip should preserve values");
        }
    }

    /// Test async NTT handle debug impl
    #[test]
    fn test_ntt_handle_debug() {
        let ctx = GpuNttContext::new(8).expect("Failed to create context");
        
        let n = 1 << 4;
        let coeffs: Vec<Scalar> = (0..n).map(|i| Scalar::from(i as u64 + 1)).collect();
        
        let handle = ctx.forward_ntt_async(&coeffs).expect("Async NTT launch failed");
        
        // Test Debug implementation
        let debug_str = format!("{:?}", handle);
        assert!(debug_str.contains("NttHandle"));
        assert!(debug_str.contains("size"));
        
        // Consume handle
        let _ = handle.wait();
    }
}
