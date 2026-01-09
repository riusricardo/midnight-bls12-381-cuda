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

//! GPU device configuration (icicle-halo2 pattern)
//!
//! Simple device selection following icicle-halo2's approach.
//! When GPU is not used, operations fall back to BLST (not ICICLE CPU).
//!
//! # Environment Variables
//!
//! - `ICICLE_BACKEND_INSTALL_DIR`: Path to ICICLE backend (default: `/opt/icicle/lib/backend`)
//! - `MIDNIGHT_GPU_MIN_K`: Minimum K for GPU usage (default: 15, meaning 2^15 = 32768 points)
//! - `MIDNIGHT_DEVICE`: Device selection ("auto", "gpu", or "cpu")
//!   - `auto` (default): Use GPU for large operations, BLST for small ones
//!   - `gpu`: Force GPU for all operations regardless of size
//!   - `cpu`: Force BLST for all operations (disable GPU)

use std::sync::OnceLock;
use tracing::{debug, info, warn};

/// Device selection mode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceType {
    /// Automatically select: GPU for large operations, BLST for small ones
    Auto,
    /// Force GPU for all operations (ignore size threshold)
    Gpu,
    /// Force BLST for all operations (disable GPU)
    Cpu,
}

impl DeviceType {
    /// Parse device type from environment variable MIDNIGHT_DEVICE
    pub fn from_env() -> Self {
        std::env::var("MIDNIGHT_DEVICE")
            .ok()
            .and_then(|s| match s.to_lowercase().as_str() {
                "gpu" | "cuda" => Some(DeviceType::Gpu),
                "cpu" | "blst" => Some(DeviceType::Cpu),
                "auto" => Some(DeviceType::Auto),
                other => {
                    warn!("Unknown MIDNIGHT_DEVICE value '{}', using Auto", other);
                    None
                }
            })
            .unwrap_or(DeviceType::Auto)
    }
}

impl std::fmt::Display for DeviceType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DeviceType::Auto => write!(f, "Auto"),
            DeviceType::Gpu => write!(f, "GPU"),
            DeviceType::Cpu => write!(f, "CPU (BLST)"),
        }
    }
}

// ============================================================================
// Simple configuration functions (icicle-halo2 pattern)
// ============================================================================

/// Get the configured device type.
///
/// Reads from `MIDNIGHT_DEVICE` environment variable.
/// Returns cached value after first call.
pub fn device_type() -> DeviceType {
    static DEVICE_TYPE: OnceLock<DeviceType> = OnceLock::new();
    *DEVICE_TYPE.get_or_init(|| {
        let dt = DeviceType::from_env();
        if dt != DeviceType::Auto {
            info!("Device type from MIDNIGHT_DEVICE: {}", dt);
        }
        dt
    })
}

// ============================================================================
// MSM Performance Tuning
// ============================================================================

/// MSM precompute factor.
///
/// Trades GPU memory for ~20-30% MSM speedup by precomputing point multiples.
/// From ICICLE docs: "Determines the number of extra points to pre-compute for
/// each point, affecting memory footprint and performance."
///
/// Parsed from `MIDNIGHT_GPU_PRECOMPUTE` environment variable.
/// Default: 1 (no precomputation)
/// Recommended: 4 (good balance of memory vs speed)
pub fn precompute_factor() -> i32 {
    static PRECOMPUTE: OnceLock<i32> = OnceLock::new();
    *PRECOMPUTE.get_or_init(|| {
        std::env::var("MIDNIGHT_GPU_PRECOMPUTE")
            .ok()
            .and_then(|s| s.parse::<i32>().ok())
            .map(|v| {
                let factor = v.max(1).min(8); // Clamp to 1-8
                if factor > 1 {
                    info!(
                        "MSM base precomputation enabled: factor={} (20-30% faster, {}x memory)",
                        factor, factor
                    );
                }
                factor
            })
            .unwrap_or(1)
    })
}

/// Window bitsize (c parameter) for MSM bucket method.
///
/// From ICICLE docs: "The 'window bitsize', a parameter controlling the
/// computational complexity and memory footprint of the MSM operation."
///
/// Default: 0 (let ICICLE choose optimal based on size)
/// Values: typically 14-18 for large MSMs
pub fn msm_window_size() -> i32 {
    static WINDOW: OnceLock<i32> = OnceLock::new();
    *WINDOW.get_or_init(|| {
        std::env::var("MIDNIGHT_MSM_WINDOW")
            .ok()
            .and_then(|s| s.parse::<i32>().ok())
            .map(|v| {
                debug!("MSM window_size={} (from MIDNIGHT_MSM_WINDOW)", v);
                v
            })
            .unwrap_or(0) // 0 = auto
    })
}

// ============================================================================
// NTT Performance Tuning
// ============================================================================

/// NTT algorithm selection.
///
/// ICICLE provides two NTT algorithms:
/// - `Auto` (0): Heuristic selection based on size and batch
/// - `Radix2` (1): Better for small NTTs (log_n ≤ 16, batch_size = 1)
/// - `MixedRadix` (2): Better for large NTTs and batch operations
///
/// From ICICLE docs:
/// > "Radix 2 is faster for small NTTs (around logN = 16 and batch size 1).
/// > Mixed radix works better for larger NTTs with larger input sizes."
///
/// Parsed from `MIDNIGHT_NTT_ALGORITHM` environment variable.
/// Default: 0 (Auto - let ICICLE choose)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum NttAlgorithm {
    Auto = 0,
    Radix2 = 1,
    MixedRadix = 2,
}

impl NttAlgorithm {
    pub fn from_i32(v: i32) -> Self {
        match v {
            1 => NttAlgorithm::Radix2,
            2 => NttAlgorithm::MixedRadix,
            _ => NttAlgorithm::Auto,
        }
    }
}

/// Get the configured NTT algorithm.
///
/// Reads from `MIDNIGHT_NTT_ALGORITHM` environment variable.
/// Values: "auto" (default), "radix2", "mixed" or "mixedradix"
pub fn ntt_algorithm() -> NttAlgorithm {
    static ALGORITHM: OnceLock<NttAlgorithm> = OnceLock::new();
    *ALGORITHM.get_or_init(|| {
        std::env::var("MIDNIGHT_NTT_ALGORITHM")
            .ok()
            .and_then(|s| match s.to_lowercase().as_str() {
                "radix2" | "radix-2" | "1" => Some(NttAlgorithm::Radix2),
                "mixed" | "mixedradix" | "mixed-radix" | "2" => Some(NttAlgorithm::MixedRadix),
                "auto" | "0" => Some(NttAlgorithm::Auto),
                other => {
                    warn!("Unknown MIDNIGHT_NTT_ALGORITHM value '{}', using Auto", other);
                    None
                }
            })
            .map(|alg| {
                if alg != NttAlgorithm::Auto {
                    info!("NTT algorithm from MIDNIGHT_NTT_ALGORITHM: {:?}", alg);
                }
                alg
            })
            .unwrap_or(NttAlgorithm::Auto)
    })
}

/// Enable fast twiddles mode for NTT domain initialization.
///
/// From ICICLE docs:
/// > "When using the Mixed-radix algorithm, it is recommended to initialize 
/// > the domain in 'fast-twiddles' mode. This is essentially allocating the 
/// > domain using extra memory but enables faster NTT."
///
/// This trades GPU memory for ~10-20% faster NTT operations.
///
/// Parsed from `MIDNIGHT_NTT_FAST_TWIDDLES` environment variable.
/// Default: true (enabled, recommended for Mixed-Radix)
pub fn ntt_fast_twiddles() -> bool {
    static FAST_TWIDDLES: OnceLock<bool> = OnceLock::new();
    *FAST_TWIDDLES.get_or_init(|| {
        std::env::var("MIDNIGHT_NTT_FAST_TWIDDLES")
            .ok()
            .and_then(|s| match s.to_lowercase().as_str() {
                "true" | "1" | "yes" | "on" => Some(true),
                "false" | "0" | "no" | "off" => Some(false),
                other => {
                    warn!("Unknown MIDNIGHT_NTT_FAST_TWIDDLES value '{}', using default", other);
                    None
                }
            })
            .map(|enabled| {
                info!("NTT fast twiddles mode: {}", if enabled { "ENABLED" } else { "DISABLED" });
                enabled
            })
            .unwrap_or(true) // Default: enabled for performance
    })
}

// =========================================================================
// NTT Ordering Configuration
// =========================================================================
//
// ICICLE supports multiple orderings for NTT inputs/outputs:
// - kNN: Natural-Natural (standard, both in natural order)
// - kNR/kRN: Bit-reversed orderings (radix-2 specific)
// - kNM/kMN: Mixed-digit orderings (most efficient for mixed-radix)
//
// For the common workflow "(1) NTT, (2) elementwise ops, (3) INTT":
// Using kNM for forward NTT and kMN for inverse avoids reordering overhead.

/// NTT ordering options (matches ICICLE's Ordering enum)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(C)]
pub enum NttOrdering {
    /// Natural-Natural: Inputs and outputs in natural order (default)
    #[default]
    NN,
    /// Natural-Reversed: Inputs natural, outputs bit-reversed
    NR,
    /// Reversed-Natural: Inputs bit-reversed, outputs natural
    RN,
    /// Reversed-Reversed: Both bit-reversed
    RR,
    /// Natural-Mixed: Inputs natural, outputs mixed-digit-reversed (efficient for mixed-radix)
    NM,
    /// Mixed-Natural: Inputs mixed-digit-reversed, outputs natural (efficient for mixed-radix)
    MN,
}

impl NttOrdering {
    /// Convert from i32 representation
    pub fn from_i32(val: i32) -> Self {
        match val {
            0 => NttOrdering::NN,
            1 => NttOrdering::NR,
            2 => NttOrdering::RN,
            3 => NttOrdering::RR,
            4 => NttOrdering::NM,
            5 => NttOrdering::MN,
            _ => NttOrdering::NN,
        }
    }
}

/// Get the configured NTT ordering mode.
///
/// Reads from `MIDNIGHT_NTT_ORDERING` environment variable.
/// Values:
/// - "nn" (default): Natural-Natural
/// - "nr": Natural-Reversed  
/// - "rn": Reversed-Natural
/// - "rr": Reversed-Reversed
/// - "nm": Natural-Mixed (recommended for forward NTT with mixed-radix)
/// - "mn": Mixed-Natural (recommended for inverse NTT with mixed-radix)
/// - "mixed": Enable mixed ordering mode (uses kNM for forward, kMN for inverse)
///
/// Note: "mixed" mode requires using the ordering-aware NTT APIs
pub fn ntt_ordering() -> NttOrdering {
    static ORDERING: OnceLock<NttOrdering> = OnceLock::new();
    *ORDERING.get_or_init(|| {
        std::env::var("MIDNIGHT_NTT_ORDERING")
            .ok()
            .and_then(|s| match s.to_lowercase().as_str() {
                "nn" | "natural" | "0" => Some(NttOrdering::NN),
                "nr" | "1" => Some(NttOrdering::NR),
                "rn" | "2" => Some(NttOrdering::RN),
                "rr" | "3" => Some(NttOrdering::RR),
                "nm" | "4" => Some(NttOrdering::NM),
                "mn" | "5" => Some(NttOrdering::MN),
                other => {
                    warn!("Unknown MIDNIGHT_NTT_ORDERING value '{}', using NN", other);
                    None
                }
            })
            .map(|ord| {
                if ord != NttOrdering::NN {
                    info!("NTT ordering from MIDNIGHT_NTT_ORDERING: {:?}", ord);
                }
                ord
            })
            .unwrap_or(NttOrdering::NN)
    })
}

/// Check if mixed ordering mode should be used.
///
/// When enabled, forward NTT uses kNM ordering and inverse NTT uses kMN ordering.
/// This is the most efficient mode for the pattern:
/// 1. Forward NTT (kNM)
/// 2. Element-wise operations in frequency domain
/// 3. Inverse NTT (kMN)
///
/// Parsed from `MIDNIGHT_NTT_MIXED_ORDERING` environment variable.
/// Default: false (use kNN for compatibility)
pub fn ntt_use_mixed_ordering() -> bool {
    static USE_MIXED: OnceLock<bool> = OnceLock::new();
    *USE_MIXED.get_or_init(|| {
        std::env::var("MIDNIGHT_NTT_MIXED_ORDERING")
            .ok()
            .and_then(|s| match s.to_lowercase().as_str() {
                "true" | "1" | "yes" | "on" => Some(true),
                "false" | "0" | "no" | "off" => Some(false),
                other => {
                    warn!("Unknown MIDNIGHT_NTT_MIXED_ORDERING value '{}', using default", other);
                    None
                }
            })
            .map(|enabled| {
                if enabled {
                    info!("NTT mixed ordering mode ENABLED (kNM forward, kMN inverse)");
                }
                enabled
            })
            .unwrap_or(false) // Default: disabled for compatibility
    })
}

// =========================================================================
// NTT GPU/CPU Decision
// =========================================================================
//
// NTT has different transfer characteristics than MSM:
// - Only transfers scalars (32 bytes each), no points
// - O(n log n) computation scales better on GPU
// - GPU NTT benefits significantly from batching
//
// The crossover point is typically lower than MSM.

/// Minimum NTT size threshold for GPU usage.
///
/// NTTs smaller than this will use CPU (halo2curves FFT) due to GPU transfer overhead.
/// Parsed from `MIDNIGHT_NTT_MIN_K` environment variable (as log2 value).
/// Default: 2^12 = 4096 elements (more aggressive than MSM since less transfer overhead)
pub fn min_ntt_gpu_size() -> usize {
    static MIN_SIZE: OnceLock<usize> = OnceLock::new();

    *MIN_SIZE.get_or_init(|| {
        std::env::var("MIDNIGHT_NTT_MIN_K")
            .ok()
            .and_then(|s| s.parse::<u8>().ok())
            .map(|k| {
                let size = 1usize << k;
                info!("MIDNIGHT_NTT_MIN_K={} -> min_ntt_gpu_size={}", k, size);
                size
            })
            .unwrap_or(4096) // Default: K >= 12
    })
}

/// Check if NTT should use GPU for a given size.
///
/// Returns true if GPU should be used, false means use CPU FFT (halo2curves).
///
/// - `DeviceType::Auto`: Use GPU if size >= NTT threshold
/// - `DeviceType::Gpu`: Always use GPU
/// - `DeviceType::Cpu`: Always use CPU FFT (never GPU)
///
/// # Arguments
/// * `size` - Number of elements in the NTT
#[inline]
pub fn should_use_gpu_ntt(size: usize) -> bool {
    match device_type() {
        DeviceType::Gpu => true,  // Force GPU regardless of size
        DeviceType::Cpu => false, // Force CPU FFT (disable GPU)
        DeviceType::Auto => size >= min_ntt_gpu_size(),
    }
}

/// Minimum problem size threshold for GPU usage.

///
/// Problems smaller than this will use BLST due to GPU transfer overhead.
/// Parsed from `MIDNIGHT_GPU_MIN_K` environment variable (as log2 value).
/// Default: 2^15 = 32768 points
pub fn min_gpu_size() -> usize {
    static MIN_SIZE: OnceLock<usize> = OnceLock::new();

    *MIN_SIZE.get_or_init(|| {
        std::env::var("MIDNIGHT_GPU_MIN_K")
            .ok()
            .and_then(|s| s.parse::<u8>().ok())
            .map(|k| {
                let size = 1usize << k;
                info!("MIDNIGHT_GPU_MIN_K={} -> min_gpu_size={}", k, size);
                size
            })
            .unwrap_or(32768) // Default: K >= 15
    })
}

/// Check if a problem size should use GPU.
///
/// Returns true if GPU should be used, false means use BLST.
///
/// - `DeviceType::Auto`: Use GPU if size >= threshold
/// - `DeviceType::Gpu`: Always use GPU
/// - `DeviceType::Cpu`: Always use BLST (never GPU)
///
/// # Arguments
/// * `size` - Number of elements (scalars, points, etc.)
#[inline]
pub fn should_use_gpu(size: usize) -> bool {
    match device_type() {
        DeviceType::Gpu => true,  // Force GPU regardless of size
        DeviceType::Cpu => false, // Force BLST (disable GPU)
        DeviceType::Auto => size >= min_gpu_size(),
    }
}

/// Check if GPU should be used for a batch of operations.
///
/// This uses the **same threshold** as single operations because benchmarking shows
/// that GPU overhead for small MSMs is significant even when batched.
///
/// # Decision Logic
///
/// For batch operations, GPU is beneficial when:
/// - Individual MSM size is >= threshold (same as single operation)
///
/// # Benchmarking Results
///
/// - 4096 points: CPU is faster (even batched)
/// - 8192 points: CPU is still faster
/// - 16384 points: CPU is still faster
/// - 32768+ points: GPU wins (threshold K=15)
///
/// # Arguments
/// * `individual_size` - Size of each individual operation (e.g., points per MSM)
/// * `batch_count` - Number of operations in the batch
///
/// # Returns
/// `true` if GPU should be used for this batch
#[inline]
pub fn should_use_gpu_batch(individual_size: usize, batch_count: usize) -> bool {
    // For batch operations, we use the same threshold as single operations.
    // GPU overhead for small MSMs is significant, so batching small MSMs
    // on GPU is actually slower than BLST on CPU.
    //
    // The key insight from benchmarking:
    // - 4096 points: CPU is faster (even batched)
    // - 8192 points: CPU is still faster 
    // - 16384 points: CPU is still faster
    // - 32768+ points: GPU wins (threshold K=15)
    //
    // With ICICLE's batch_size parameter, we can batch multiple MSMs into
    // a single kernel launch. This is beneficial when:
    // 1. Individual MSMs are above the base threshold, OR
    // 2. Total work (batch * individual) is large enough to amortize overhead
    //
    // ICICLE recommendation: Use batch_size parameter with are_points_shared_in_batch=true
    // when all MSMs use the same bases (e.g., SRS commitments)
    
    // Ignore batch_count and total_work - benchmarking shows GPU only wins
    // when individual MSM size meets threshold, regardless of batch size
    let _ = individual_size.saturating_mul(batch_count);
    
    match device_type() {
        DeviceType::Gpu => true,  // Force GPU for all batches
        DeviceType::Cpu => false, // Force BLST for all batches
        DeviceType::Auto => {
            // GPU beneficial only if individual size meets threshold
            // Batching small MSMs on GPU is slower than BLST on CPU
            should_use_gpu(individual_size)
        }
    }
}

/// Get the ICICLE backend installation path.
///
/// Reads from `ICICLE_BACKEND_INSTALL_DIR` environment variable.
/// Falls back to `/opt/icicle/lib/backend` if not set.
pub fn backend_path() -> String {
    std::env::var("ICICLE_BACKEND_INSTALL_DIR")
        .unwrap_or_else(|_| "/opt/icicle/lib/backend".to_string())
}

/// Get the device ID to use (for multi-GPU systems).
///
/// Currently always returns 0 (first GPU).
/// Future: Could be configurable via environment variable.
#[inline]
pub const fn device_id() -> i32 {
    0
}

/// Log current GPU configuration.
///
/// Useful for debugging and verifying configuration at startup.
pub fn log_config() {
    debug!("GPU Configuration:");
    debug!("  Backend path: {}", backend_path());
    debug!("  Device type: {}", device_type());
    debug!("  Device ID: {}", device_id());
    debug!(
        "  Min GPU size: {} (K >= {})",
        min_gpu_size(),
        min_gpu_size().trailing_zeros()
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_type_display() {
        assert_eq!(DeviceType::Auto.to_string(), "Auto");
        assert_eq!(DeviceType::Gpu.to_string(), "GPU");
        assert_eq!(DeviceType::Cpu.to_string(), "CPU (BLST)");
    }

    #[test]
    fn test_min_gpu_size_default() {
        let size = min_gpu_size();
        assert!(size > 0);
        assert!(size.is_power_of_two());
        assert_eq!(size, 32768); // Default K=15
    }

    #[test]
    fn test_should_use_gpu_threshold() {
        let threshold = min_gpu_size();

        // In Auto mode (default), should respect threshold
        if device_type() == DeviceType::Auto {
            assert!(!should_use_gpu(threshold - 1));
            assert!(should_use_gpu(threshold));
            assert!(should_use_gpu(threshold * 2));
        }
    }

    #[test]
    fn test_should_use_gpu_batch() {
        let threshold = min_gpu_size(); // 32768
        
        // In Auto mode, batch uses same threshold as single operation
        if device_type() == DeviceType::Auto {
            // Below threshold - should not use GPU regardless of batch count
            assert!(!should_use_gpu_batch(threshold - 1, 1));
            assert!(!should_use_gpu_batch(threshold - 1, 10));
            assert!(!should_use_gpu_batch(4096, 100)); // 4096 < 32768
            
            // At or above threshold - should use GPU
            assert!(should_use_gpu_batch(threshold, 1));
            assert!(should_use_gpu_batch(threshold, 10));
            assert!(should_use_gpu_batch(threshold * 2, 5));
        }
    }

    #[test]
    fn test_backend_path() {
        let path = backend_path();
        assert!(!path.is_empty());
    }

    #[test]
    fn test_device_id() {
        assert_eq!(device_id(), 0);
    }
}
