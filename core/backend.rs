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

//! ICICLE GPU backend initialization
//!
//! Simple backend loading matching icicle-halo2's approach.
//! The backend is loaded once and cached globally.

use super::config::backend_path;
use std::sync::OnceLock;
use tracing::{debug, error, info};

#[cfg(feature = "gpu")]
use icicle_runtime::load_backend_from_env_or_default;

/// Errors that can occur during GPU operations
#[derive(Debug, Clone)]
pub enum GpuError {
    /// GPU backend failed to load
    BackendLoadFailed(String),
    /// Failed to set device
    DeviceSetFailed(String),
    /// GPU operation failed
    OperationFailed(String),
    /// Stream operation failed
    StreamError(String),
    /// GPU not available
    NotAvailable,
}

impl std::fmt::Display for GpuError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GpuError::BackendLoadFailed(msg) => write!(f, "Backend load failed: {}", msg),
            GpuError::DeviceSetFailed(msg) => write!(f, "Device set failed: {}", msg),
            GpuError::OperationFailed(msg) => write!(f, "GPU operation failed: {}", msg),
            GpuError::StreamError(msg) => write!(f, "Stream error: {}", msg),
            GpuError::NotAvailable => write!(f, "GPU not available"),
        }
    }
}

impl std::error::Error for GpuError {}

/// Global singleton for ICICLE backend initialization
/// Ensures the backend is loaded exactly once across all threads
#[cfg(feature = "gpu")]
static BACKEND_INITIALIZED: OnceLock<Result<(), String>> = OnceLock::new();

/// Ensure the ICICLE backend is loaded (called exactly once)
///
/// This function is safe to call from multiple threads - only the first
/// call will actually load the backend, subsequent calls return immediately.
///
/// Returns Ok(()) if backend is loaded, Err with message if loading failed.
#[cfg(feature = "gpu")]
pub fn ensure_backend_loaded() -> Result<(), GpuError> {
    let result = BACKEND_INITIALIZED.get_or_init(|| {
        let path = backend_path();

        // Set environment variable for ICICLE
        std::env::set_var("ICICLE_BACKEND_INSTALL_DIR", &path);

        debug!("Loading ICICLE backend from {}", path);

        match load_backend_from_env_or_default() {
            Ok(_) => {
                info!("ICICLE backend loaded successfully from {}", path);
                Ok(())
            }
            Err(e) => {
                error!("Failed to load ICICLE backend: {:?}", e);
                Err(format!("{:?}", e))
            }
        }
    });

    result.clone().map_err(GpuError::BackendLoadFailed)
}

/// Check if GPU is available (backend loaded successfully)
#[cfg(feature = "gpu")]
pub fn is_gpu_available() -> bool {
    ensure_backend_loaded().is_ok()
}

#[cfg(not(feature = "gpu"))]
pub fn ensure_backend_loaded() -> Result<(), GpuError> {
    Err(GpuError::NotAvailable)
}

#[cfg(not(feature = "gpu"))]
pub fn is_gpu_available() -> bool {
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_error_display() {
        let err = GpuError::NotAvailable;
        assert_eq!(err.to_string(), "GPU not available");

        let err = GpuError::BackendLoadFailed("test".to_string());
        assert_eq!(err.to_string(), "Backend load failed: test");

        let err = GpuError::StreamError("sync failed".to_string());
        assert_eq!(err.to_string(), "Stream error: sync failed");
    }

    #[cfg(feature = "gpu")]
    #[test]
    #[ignore] // Only run when GPU is available
    fn test_backend_initialization() {
        let result = ensure_backend_loaded();

        match result {
            Ok(()) => {
                println!("ICICLE backend loaded successfully");
                // Second call should also succeed (cached)
                assert!(ensure_backend_loaded().is_ok());
            }
            Err(e) => {
                println!("GPU not available (expected on some systems): {}", e);
            }
        }
    }
}
