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

//! GPU Stream Management (RAII Wrapper)
//!
//! This module provides safe stream management following the icicle-halo2 pattern.
//! 
//! # Reference
//! 
//! Pattern derived from:
//! - **ICICLE Rust Guide**: https://dev.ingonyama.com/start/programmers_guide/rust
//!
//! # Key ICICLE Stream Rules
//!
//! From the ICICLE documentation:
//! 1. Streams must be explicitly destroyed with `stream.destroy()` before drop
//! 2. `IcicleStream::default()` returns a null/default stream (synchronous)
//! 3. Created streams are non-blocking and allow async operations
//!
//! # Usage Pattern (from icicle-halo2/evaluation.rs)
//!
//! ```rust,ignore
//! // Create stream per operation batch
//! let mut stream = IcicleStream::create().unwrap();
//! let mut d_result = DeviceVec::device_malloc_async(size, &stream).unwrap();
//!
//! let mut cfg = VecOpsConfig::default();
//! cfg.stream_handle = stream.into();
//! cfg.is_async = true;
//!
//! // ... perform async operations ...
//!
//! stream.synchronize().unwrap();
//! stream.destroy().unwrap();  // MUST call before drop!
//! ```
//!
//! This module wraps this pattern in a safe RAII type.

use crate::GpuError;

#[cfg(feature = "gpu")]
use icicle_runtime::stream::IcicleStream;

/// RAII wrapper for ICICLE streams ensuring proper lifecycle management.
///
/// ICICLE streams **must** be explicitly destroyed before they are dropped.
/// This wrapper ensures that `destroy()` is called automatically in the 
/// destructor, preventing resource leaks.
///
/// # Pattern Reference
///
/// From icicle-halo2 `evaluation.rs`:
/// ```rust,ignore
/// let mut stream = IcicleStream::create().unwrap();
/// // ... use stream ...
/// stream.synchronize().unwrap();
/// stream.destroy().unwrap();  // <-- We automate this
/// ```
///
/// # Example
///
/// ```rust,ignore
/// use midnight_proofs::gpu::stream::ManagedStream;
///
/// let mut stream = ManagedStream::create()?;
///
/// // Use stream for async operations
/// let mut cfg = MSMConfig::default();
/// cfg.stream_handle = stream.as_ref().into();
/// cfg.is_async = true;
///
/// // ... perform operations ...
///
/// stream.synchronize()?;
/// // Stream is automatically destroyed when dropped
/// ```
#[cfg(feature = "gpu")]
pub struct ManagedStream {
    /// The underlying ICICLE stream
    stream: IcicleStream,
    /// Whether the stream has been destroyed (to prevent double-destroy)
    destroyed: bool,
}

#[cfg(feature = "gpu")]
impl ManagedStream {
    /// Create a new CUDA stream for async operations.
    ///
    /// This allocates a new stream on the GPU. Operations submitted to this
    /// stream will execute asynchronously with respect to other streams.
    ///
    /// # Errors
    ///
    /// Returns an error if stream creation fails (e.g., CUDA not available).
    pub fn create() -> Result<Self, GpuError> {
        let stream = IcicleStream::create()
            .map_err(|e| GpuError::StreamError(format!("Failed to create stream: {:?}", e)))?;
        
        Ok(Self {
            stream,
            destroyed: false,
        })
    }

    /// Create a wrapper around the default (null) stream.
    ///
    /// The default stream is synchronous - operations on it block until complete.
    /// This is suitable for simple, single-operation use cases.
    ///
    /// # Note
    ///
    /// The default stream should NOT be destroyed, so we mark it as already
    /// destroyed to prevent the destructor from calling destroy().
    pub fn default_stream() -> Self {
        Self {
            stream: IcicleStream::default(),
            destroyed: true, // Don't destroy the default stream
        }
    }

    /// Synchronize the stream, waiting for all operations to complete.
    ///
    /// This blocks the CPU until all GPU operations submitted to this stream
    /// have finished executing.
    ///
    /// # Pattern Reference
    ///
    /// From icicle-halo2:
    /// ```rust,ignore
    /// stream.synchronize().unwrap();
    /// ```
    pub fn synchronize(&mut self) -> Result<(), GpuError> {
        self.stream.synchronize()
            .map_err(|e| GpuError::StreamError(format!("Stream synchronize failed: {:?}", e)))
    }

    /// Get a reference to the underlying ICICLE stream.
    ///
    /// Use this to pass the stream to ICICLE configuration structs:
    /// ```rust,ignore
    /// cfg.stream_handle = stream.as_ref().into();
    /// ```
    pub fn as_ref(&self) -> &IcicleStream {
        &self.stream
    }

    /// Explicitly destroy the stream.
    ///
    /// This is called automatically by `Drop`, but can be called manually
    /// if you want to handle errors.
    ///
    /// After calling this, the stream is marked as destroyed and the
    /// destructor will not attempt to destroy it again.
    pub fn destroy(&mut self) -> Result<(), GpuError> {
        if !self.destroyed {
            self.stream.destroy()
                .map_err(|e| GpuError::StreamError(format!("Stream destroy failed: {:?}", e)))?;
            self.destroyed = true;
        }
        Ok(())
    }

    /// Check if the stream has been destroyed.
    pub fn is_destroyed(&self) -> bool {
        self.destroyed
    }
}

#[cfg(feature = "gpu")]
impl Drop for ManagedStream {
    fn drop(&mut self) {
        if !self.destroyed {
            // Best-effort destroy - can't propagate errors from Drop
            if let Err(e) = self.stream.destroy() {
                tracing::warn!("Failed to destroy stream in Drop: {:?}", e);
            }
            self.destroyed = true;
        }
    }
}

#[cfg(feature = "gpu")]
impl std::fmt::Debug for ManagedStream {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ManagedStream")
            .field("destroyed", &self.destroyed)
            .finish()
    }
}

// Note: ManagedStream is NOT Send/Sync because IcicleStream contains a raw pointer.
// This is intentional - streams should be used within a single thread.
// For multi-threaded scenarios, create a new stream per thread.

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(feature = "gpu")]
    fn test_managed_stream_lifecycle() {
        use crate::backend::ensure_backend_loaded;
        use icicle_runtime::{Device, set_device};

        // Setup
        if ensure_backend_loaded().is_err() {
            return; // Skip if no GPU
        }
        let device = Device::new("CUDA", 0);
        if set_device(&device).is_err() {
            return; // Skip if device not available
        }

        // Test stream creation and destruction
        let mut stream = ManagedStream::create().expect("Should create stream");
        assert!(!stream.is_destroyed());
        
        stream.synchronize().expect("Should synchronize");
        stream.destroy().expect("Should destroy");
        assert!(stream.is_destroyed());
        
        // Destructor should not double-destroy
        drop(stream);
    }

    #[test]
    #[cfg(feature = "gpu")]
    fn test_default_stream() {
        let stream = ManagedStream::default_stream();
        // Default stream is marked as destroyed to prevent destroy() call
        assert!(stream.is_destroyed());
    }

    #[test]
    #[cfg(feature = "gpu")]
    fn test_stream_drop_without_explicit_destroy() {
        use crate::backend::ensure_backend_loaded;
        use icicle_runtime::{Device, set_device};

        if ensure_backend_loaded().is_err() {
            return;
        }
        let device = Device::new("CUDA", 0);
        if set_device(&device).is_err() {
            return;
        }

        // Stream should be cleaned up by Drop even without explicit destroy()
        let stream = ManagedStream::create().expect("Should create stream");
        assert!(!stream.is_destroyed());
        drop(stream); // Should call destroy() in destructor
    }
}
