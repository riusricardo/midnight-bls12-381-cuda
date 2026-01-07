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

//! GPU acceleration for midnight-zk proof generation
//!
//! This module provides GPU-accelerated implementations of computationally intensive
//! operations in the PLONK prover, specifically Multi-Scalar Multiplication (MSM)
//! and Number Theoretic Transform (NTT).
//!
//! # Features
//!
//! Enable GPU support by compiling with the `gpu` feature:
//! ```toml
//! midnight-proofs = { version = "*", features = ["gpu"] }
//! ```
//!
//! # Backend Support
//!
//! - **CUDA**: NVIDIA GPUs via ICICLE CUDA backend
//! - **BLST**: CPU fallback for small operations or when GPU disabled
//!
//! Note: There is NO ICICLE CPU backend. When GPU is not used, operations
//! fall back to BLST.
//!
//! # Device Selection
//!
//! Control via `MIDNIGHT_DEVICE` environment variable:
//! - `auto` (default): GPU for large ops (>= 2^14 points), BLST for small
//! - `gpu`: Force GPU for all operations
//! - `cpu`: Force BLST for all operations (disable GPU)
//!
//! # Setup
//!
//! 1. Build and install the CUDA backend:
//!    ```bash
//!    cd bls12-381-cuda-backend
//!    mkdir build && cd build
//!    cmake .. -DCMAKE_BUILD_TYPE=Release
//!    make icicle -j$(nproc)
//!    sudo make icicle-install
//!    ```
//!
//! 2. The backend will be automatically loaded when the prover is initialized.
//!
//! # Example
//!
//! ```rust,no_run
//! use midnight_proofs::gpu::{ensure_backend_loaded, should_use_gpu, GpuMsmContext};
//!
//! // Initialize GPU backend
//! ensure_backend_loaded()?;
//!
//! // Check if GPU should be used for this size
//! if should_use_gpu(points.len()) {
//!     let ctx = GpuMsmContext::new()?;
//!     // Use GPU...
//! } else {
//!     // Use BLST...
//! }
//! ```

pub mod backend;
pub mod config;
pub mod msm;
pub mod ntt;
pub mod stream;
pub mod types;
pub mod vecops;

// Core exports
pub use backend::{ensure_backend_loaded, is_gpu_available, GpuError};
pub use config::{device_type, min_gpu_size, should_use_gpu, should_use_gpu_batch, backend_path, device_id, DeviceType};

// GPU-specific exports (only when gpu feature is enabled)
#[cfg(feature = "gpu")]
pub use msm::{GpuMsmContext, MsmError, MsmHandle, G2MsmHandle};
#[cfg(feature = "gpu")]
pub use ntt::{GpuNttContext, NttError, NttHandle};
#[cfg(feature = "gpu")]
pub use stream::ManagedStream;
#[cfg(feature = "gpu")]
pub use types::TypeConverter;

// VecOps exports
pub use vecops::{vector_add, vector_sub, vector_mul, scalar_mul, VecOpsError, should_use_gpu_vecops};

/// Check if GPU support is compiled in
pub const GPU_SUPPORT: bool = cfg!(feature = "gpu");

/// Initialize and warmup GPU backend for proof generation
/// 
/// Call this at application startup to avoid first-request latency.
/// This function:
/// 1. Initializes the ICICLE CUDA backend
/// 2. Creates the global MSM executor
/// 3. Runs a small MSM to trigger GPU memory allocation
/// 
/// Returns the warmup duration, or None if GPU is not available.
/// 
/// # Example
/// ```rust,no_run
/// use midnight_proofs::gpu::warmup_gpu;
/// 
/// #[tokio::main]
/// async fn main() {
///     // Warmup GPU before starting server
///     if let Some(duration) = warmup_gpu() {
///         println!("GPU ready in {:?}", duration);
///     }
///     
///     // Start server...
/// }
/// ```
pub fn warmup_gpu() -> Option<std::time::Duration> {
    use tracing::info;
    
    #[cfg(feature = "gpu")]
    {
        info!("Warming up GPU backend...");
        
        match GpuMsmContext::new() {
            Ok(ctx) => match ctx.warmup() {
                Ok(duration) => {
                    info!("GPU warmup successful: backend=CUDA, duration={:?}", duration);
                    Some(duration)
                }
                Err(e) => {
                    tracing::warn!("GPU warmup failed: {:?}", e);
                    None
                }
            },
            Err(e) => {
                tracing::warn!("GPU context creation failed: {:?}", e);
                None
            }
        }
    }
    
    #[cfg(not(feature = "gpu"))]
    {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_support() {
        // This test just verifies the module compiles
        assert_eq!(GPU_SUPPORT, cfg!(feature = "gpu"));
    }
}
