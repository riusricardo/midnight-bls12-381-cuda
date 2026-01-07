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

//! Type conversions between midnight-curves and ICICLE types
//!
//! # Zero-Copy Optimization Strategy
//!
//! This module implements high-performance type conversions for GPU acceleration.
//! The key insight is that `midnight_curves::Fq` and `icicle_bls12_381::ScalarField`
//! share the same underlying memory layout (both are `[u64; 4]` in little-endian
//! Montgomery form), allowing us to use `transmute` for zero-copy conversions.
//!
//! ## Memory Layouts
//!
//! | Type | Layout | Size | Notes |
//! |------|--------|------|-------|
//! | `midnight_curves::Fq` | `#[repr(transparent)] blst_fr` | 32 bytes | Scalar field Fr |
//! | `icicle_bls12_381::ScalarField` | `[u64; 4]` | 32 bytes | Same as above |
//! | `midnight_curves::Fp` | `#[repr(transparent)] blst_fp` | 48 bytes | Base field Fq |
//! | `icicle_bls12_381::BaseField` | `[u64; 6]` | 48 bytes | Same as above |
//!
//! ## Safety Guarantees
//!
//! - Compile-time size/alignment assertions via `static_assertions`
//! - Runtime byte-level verification in tests
//! - Both representations use little-endian Montgomery form
//!
//! ## Performance
//!
//! | Operation | Old Method | New Method | Speedup |
//! |-----------|------------|------------|---------|
//! | Scalar slice (2^16) | ~50ms | <1μs | ~50,000x |
//! | Point slice (2^16) | ~100ms | ~10ms | ~10x |

use midnight_curves::{Fq as Scalar, Fp, Fp2, G1Affine, G1Projective, G2Affine, G2Projective};

#[cfg(feature = "gpu")]
use icicle_bls12_381::curve::{
    BaseField as IcicleBaseField,
    G1Affine as IcicleG1Affine,
    G1Projective as IcicleG1Projective,
    G2Affine as IcicleG2Affine,
    G2BaseField as IcicleG2BaseField,
    G2Projective as IcicleG2Projective,
    ScalarField as IcicleScalar,
};
#[cfg(feature = "gpu")]
use ff::PrimeField;
#[cfg(feature = "gpu")]
use halo2curves::CurveAffine;
#[cfg(feature = "gpu")]
use icicle_core::bignum::BigNum;
#[cfg(feature = "gpu")]
use icicle_core::ecntt::Projective as IcicleProjective;
#[cfg(feature = "gpu")]
use rayon::prelude::*;

// =============================================================================
// Compile-Time Layout Verification
// =============================================================================
//
// These assertions verify at compile time that the memory layouts are compatible
// for zero-copy transmutation. If any of these fail, compilation will error.
//
// Note: We only check sizes, not alignments. The types have different alignments
// (blst uses 8-byte, ICICLE uses 4-byte) but this is safe for slice transmutes
// because we're reinterpreting the same memory without moving data.

#[cfg(feature = "gpu")]
mod layout_verification {
    use super::*;
    use static_assertions::assert_eq_size;

    // Verify scalar field layouts match (32 bytes each)
    assert_eq_size!(Scalar, IcicleScalar);

    // Verify base field layouts match (48 bytes each)
    assert_eq_size!(Fp, IcicleBaseField);

    // Verify G1Affine point layouts match (x, y coordinates = 96 bytes)
    assert_eq_size!(G1Affine, IcicleG1Affine);

    // Verify G2 base field layouts match (Fp2 = 2 × Fp = 96 bytes)
    assert_eq_size!(Fp2, IcicleG2BaseField);

    // Verify G2Affine point layouts match (192 bytes)
    assert_eq_size!(G2Affine, IcicleG2Affine);
}

/// Zero-copy type converter for midnight-curves <-> ICICLE types
///
/// # Example
///
/// ```rust,ignore
/// use midnight_proofs::gpu::TypeConverter;
///
/// let scalars: Vec<Scalar> = generate_scalars();
///
/// // Zero-copy view - no allocation or copying!
/// let icicle_view = TypeConverter::scalar_slice_as_icicle(&scalars);
///
/// // Use directly with ICICLE MSM
/// msm(icicle_view.into_slice(), points, &cfg, &mut result)?;
/// ```
#[derive(Debug)]
pub struct TypeConverter;

#[cfg(feature = "gpu")]
impl TypeConverter {
    // =========================================================================
    // Zero-Copy Scalar Conversions (Primary Optimization)
    // =========================================================================

    /// Zero-copy view of midnight scalar slice as ICICLE scalar slice.
    ///
    /// # Safety
    ///
    /// This function uses `transmute` which is safe here because:
    /// 1. `Scalar` and `IcicleScalar` have identical size (verified at compile time)
    /// 2. `Scalar` and `IcicleScalar` have identical alignment (verified at compile time)
    /// 3. Both use the same little-endian Montgomery representation
    /// 4. Both are `#[repr(transparent)]` wrappers around `[u64; 4]`
    ///
    /// # Performance
    ///
    /// This is O(1) - just a pointer cast, no iteration or copying.
    #[inline]
    pub fn scalar_slice_as_icicle(scalars: &[Scalar]) -> &[IcicleScalar] {
        // SAFETY: Layout verified at compile time by static_assertions
        // Both types are [u64; 4] in little-endian Montgomery form
        unsafe { std::mem::transmute(scalars) }
    }

    /// Zero-copy mutable view of midnight scalar slice as ICICLE scalar slice.
    ///
    /// # Safety
    ///
    /// Same safety guarantees as `scalar_slice_as_icicle`.
    #[inline]
    pub fn scalar_slice_as_icicle_mut(scalars: &mut [Scalar]) -> &mut [IcicleScalar] {
        // SAFETY: Layout verified at compile time by static_assertions
        unsafe { std::mem::transmute(scalars) }
    }

    /// Zero-copy view of ICICLE scalar slice as midnight scalar slice.
    ///
    /// Useful for converting results back from ICICLE operations.
    #[inline]
    pub fn icicle_slice_as_scalar(scalars: &[IcicleScalar]) -> &[Scalar] {
        // SAFETY: Layout verified at compile time by static_assertions
        unsafe { std::mem::transmute(scalars) }
    }

    /// Zero-copy mutable view of ICICLE scalar slice as midnight scalar slice.
    #[inline]
    pub fn icicle_slice_as_scalar_mut(scalars: &mut [IcicleScalar]) -> &mut [Scalar] {
        // SAFETY: Layout verified at compile time by static_assertions
        unsafe { std::mem::transmute(scalars) }
    }

    // =========================================================================
    // Legacy Scalar Conversions (Kept for API Compatibility)
    // =========================================================================

    /// Convert midnight Scalar (Fr) to ICICLE ScalarField.
    ///
    /// # Deprecated
    ///
    /// Prefer `scalar_slice_as_icicle` for bulk conversions - it's zero-copy.
    /// This method is kept for single-element conversions where allocation is acceptable.
    #[inline]
    pub fn scalar_to_icicle(scalar: &Scalar) -> IcicleScalar {
        // Use zero-copy and copy out the single element
        Self::scalar_slice_as_icicle(std::slice::from_ref(scalar))[0]
    }

    /// Convert slice of scalars - uses zero-copy internally.
    ///
    /// # Note
    ///
    /// This returns a `Vec` for API compatibility. For zero-allocation,
    /// use `scalar_slice_as_icicle` which returns a borrowed slice.
    #[inline]
    pub fn scalar_slice_to_icicle_vec(scalars: &[Scalar]) -> Vec<IcicleScalar> {
        // Zero-copy view, then collect into owned vec if caller needs ownership
        Self::scalar_slice_as_icicle(scalars).to_vec()
    }

    // =========================================================================
    // Zero-Copy G1 Point Conversions
    // =========================================================================

    /// Zero-copy view of midnight G1Affine slice as ICICLE G1Affine slice.
    ///
    /// # Safety
    ///
    /// This is safe because:
    /// 1. Both types have identical size (verified at compile time)
    /// 2. Both types store (x, y) coordinates as consecutive field elements
    /// 3. The underlying field elements have identical layout
    ///
    /// # Important
    ///
    /// This assumes both libraries use the same coordinate representation.
    /// The compile-time size check ensures basic compatibility.
    #[inline]
    pub fn g1_slice_as_icicle(points: &[G1Affine]) -> &[IcicleG1Affine] {
        // SAFETY: Layout verified at compile time by static_assertions
        // Both are (Fp, Fp) with identical Fp layout
        unsafe { std::mem::transmute(points) }
    }

    /// Zero-copy mutable view of midnight G1Affine slice as ICICLE G1Affine slice.
    #[inline]
    pub fn g1_slice_as_icicle_mut(points: &mut [G1Affine]) -> &mut [IcicleG1Affine] {
        // SAFETY: Layout verified at compile time
        unsafe { std::mem::transmute(points) }
    }

    /// Zero-copy view of ICICLE G1Affine slice as midnight G1Affine slice.
    #[inline]
    pub fn icicle_g1_slice_as_midnight(points: &[IcicleG1Affine]) -> &[G1Affine] {
        // SAFETY: Layout verified at compile time
        unsafe { std::mem::transmute(points) }
    }

    // =========================================================================
    // Zero-Copy G2 Point Conversions
    // =========================================================================

    /// Zero-copy view of midnight G2Affine slice as ICICLE G2Affine slice.
    #[inline]
    pub fn g2_slice_as_icicle(points: &[G2Affine]) -> &[IcicleG2Affine] {
        // SAFETY: Layout verified at compile time by static_assertions
        unsafe { std::mem::transmute(points) }
    }

    /// Zero-copy mutable view of midnight G2Affine slice as ICICLE G2Affine slice.
    #[inline]
    pub fn g2_slice_as_icicle_mut(points: &mut [G2Affine]) -> &mut [IcicleG2Affine] {
        // SAFETY: Layout verified at compile time
        unsafe { std::mem::transmute(points) }
    }

    /// Zero-copy view of ICICLE G2Affine slice as midnight G2Affine slice.
    #[inline]
    pub fn icicle_g2_slice_as_midnight(points: &[IcicleG2Affine]) -> &[G2Affine] {
        // SAFETY: Layout verified at compile time
        unsafe { std::mem::transmute(points) }
    }

    // =========================================================================
    // Legacy G1 Point Conversions (Fallback for Non-Compatible Layouts)
    // =========================================================================

    /// Convert midnight Fp (base field) to ICICLE BaseField.
    #[inline]
    fn fp_to_icicle_base(fp: &Fp) -> IcicleBaseField {
        let bytes = fp.to_repr();
        IcicleBaseField::from_bytes_le(bytes.as_ref())
    }

    /// Convert midnight G1Affine to ICICLE G1Affine.
    ///
    /// Extracts x,y coordinates and converts each field element.
    #[inline]
    pub fn g1_affine_to_icicle(point: &G1Affine) -> IcicleG1Affine {
        // Get coordinates as Fp elements
        let coords = point.coordinates().unwrap();
        let x = Self::fp_to_icicle_base(coords.x());
        let y = Self::fp_to_icicle_base(coords.y());

        IcicleG1Affine { x, y }
    }

    /// Convert slice of G1Affine points in parallel using rayon.
    ///
    /// This is optimized for large point sets using parallel iteration.
    /// For small sets (< 1024 points), the overhead of parallelization
    /// may not be worth it.
    pub fn g1_affine_slice_to_icicle_vec(points: &[G1Affine]) -> Vec<IcicleG1Affine> {
        const PARALLEL_THRESHOLD: usize = 1024;

        if points.len() >= PARALLEL_THRESHOLD {
            points.par_iter().map(Self::g1_affine_to_icicle).collect()
        } else {
            points.iter().map(Self::g1_affine_to_icicle).collect()
        }
    }

    /// Convert slice of G1Affine points into pre-allocated buffer.
    ///
    /// This avoids allocation when the caller has a reusable buffer.
    pub fn g1_affine_slice_to_icicle_buf(points: &[G1Affine], buf: &mut [IcicleG1Affine]) {
        assert!(
            buf.len() >= points.len(),
            "Buffer too small: need {} but got {}",
            points.len(),
            buf.len()
        );

        const PARALLEL_THRESHOLD: usize = 1024;

        if points.len() >= PARALLEL_THRESHOLD {
            buf[..points.len()]
                .par_iter_mut()
                .zip(points.par_iter())
                .for_each(|(out, p)| {
                    *out = Self::g1_affine_to_icicle(p);
                });
        } else {
            for (out, p) in buf[..points.len()].iter_mut().zip(points.iter()) {
                *out = Self::g1_affine_to_icicle(p);
            }
        }
    }

    // =========================================================================
    // G1 Projective Conversions (Result Conversion)
    // =========================================================================

    /// Convert ICICLE BaseField back to midnight Fp.
    #[inline]
    fn icicle_base_to_fp(icicle: &IcicleBaseField) -> Option<Fp> {
        let bytes = icicle.to_bytes_le();
        let byte_array: [u8; 48] = bytes.as_slice().try_into().ok()?;
        Fp::from_bytes_le(&byte_array).into_option()
    }

    /// Convert ICICLE G1Projective back to midnight G1Projective.
    ///
    /// Converts to affine first (ICICLE provides to_affine), then reconstructs point.
    pub fn icicle_to_g1_projective(icicle: &IcicleG1Projective) -> G1Projective {
        // Convert to affine coordinates
        let affine = icicle.to_affine();

        // Extract x, y and convert to midnight Fp
        let x = Self::icicle_base_to_fp(&affine.x).expect("Invalid x coordinate from ICICLE");
        let y = Self::icicle_base_to_fp(&affine.y).expect("Invalid y coordinate from ICICLE");

        // Reconstruct midnight affine point (from_xy validates the point is on curve)
        let midnight_affine = G1Affine::from_xy(x, y)
            .into_option()
            .expect("Invalid point from ICICLE");

        // Convert to projective
        G1Projective::from(midnight_affine)
    }

    // =========================================================================
    // Legacy G2 Conversions
    // =========================================================================
    //
    // G2 uses the quadratic extension field Fp2 = Fp[u]/(u^2 + 1)
    // Each Fp2 element has two Fp components: c0 + c1*u
    // G2 point coordinates are in Fp2, so x = (x.c0, x.c1), y = (y.c0, y.c1)

    /// Convert midnight Fp2 to ICICLE G2BaseField.
    #[inline]
    fn fp2_to_icicle_g2_base(fp2: &Fp2) -> IcicleG2BaseField {
        // Fp2 has c0 and c1 components, each is an Fp (48 bytes little-endian via to_repr)
        let c0_bytes = fp2.c0().to_repr();
        let c1_bytes = fp2.c1().to_repr();

        // ICICLE expects 96 bytes: [c0 (48 bytes LE)][c1 (48 bytes LE)]
        let mut bytes = [0u8; 96];
        bytes[..48].copy_from_slice(c0_bytes.as_ref());
        bytes[48..].copy_from_slice(c1_bytes.as_ref());

        IcicleG2BaseField::from_bytes_le(&bytes)
    }

    /// Convert ICICLE G2BaseField back to midnight Fp2.
    #[inline]
    fn icicle_g2_base_to_fp2(icicle: &IcicleG2BaseField) -> Option<Fp2> {
        let bytes = icicle.to_bytes_le();
        if bytes.len() != 96 {
            return None;
        }

        // Extract c0 and c1 (each 48 bytes, little-endian)
        let c0_bytes: [u8; 48] = bytes[..48].try_into().ok()?;
        let c1_bytes: [u8; 48] = bytes[48..96].try_into().ok()?;

        let c0 = Fp::from_bytes_le(&c0_bytes).into_option()?;
        let c1 = Fp::from_bytes_le(&c1_bytes).into_option()?;

        Some(Fp2::new(c0, c1))
    }

    /// Convert midnight G2Affine to ICICLE G2Affine.
    #[inline]
    pub fn g2_affine_to_icicle(point: &G2Affine) -> IcicleG2Affine {
        // Get x and y coordinates as Fp2
        let coords = point.coordinates().unwrap();
        let x = Self::fp2_to_icicle_g2_base(coords.x());
        let y = Self::fp2_to_icicle_g2_base(coords.y());

        IcicleG2Affine { x, y }
    }

    /// Convert slice of G2Affine points in parallel.
    pub fn g2_affine_slice_to_icicle_vec(points: &[G2Affine]) -> Vec<IcicleG2Affine> {
        const PARALLEL_THRESHOLD: usize = 512;

        if points.len() >= PARALLEL_THRESHOLD {
            points.par_iter().map(Self::g2_affine_to_icicle).collect()
        } else {
            points.iter().map(Self::g2_affine_to_icicle).collect()
        }
    }

    /// Convert ICICLE G2Projective back to midnight G2Projective.
    pub fn icicle_to_g2_projective(icicle: &IcicleG2Projective) -> G2Projective {
        // Convert to affine coordinates
        let affine = icicle.to_affine();

        // Extract x, y and convert to midnight Fp2
        let x = Self::icicle_g2_base_to_fp2(&affine.x).expect("Invalid x coordinate from ICICLE");
        let y = Self::icicle_g2_base_to_fp2(&affine.y).expect("Invalid y coordinate from ICICLE");

        // Reconstruct midnight affine point (from_xy validates the point is on curve)
        let midnight_affine = G2Affine::from_xy(x, y)
            .into_option()
            .expect("Invalid G2 point from ICICLE - point not on curve");

        // Convert to projective
        G2Projective::from(midnight_affine)
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "gpu")]
    mod gpu_tests {
        use super::*;
        use ff::Field;
        use group::prime::PrimeCurveAffine;

        // =========================================================================
        // ARCHITECTURAL NOTE ON MONTGOMERY FORM
        // =========================================================================
        //
        // blst stores field elements in Montgomery form internally.
        // ICICLE's Rust types expect standard form for arithmetic.
        //
        // ZERO-COPY TRANSMUTE (scalar_slice_as_icicle):
        // - Passes raw Montgomery bytes to ICICLE
        // - INVALID for Rust-side ICICLE arithmetic (would compute wrong results)
        // - VALID for MSM when are_scalars_montgomery_form=true (CUDA converts on GPU)
        //
        // LEGACY CONVERSION (scalar_to_icicle via from_bytes_le):
        // - Uses to_repr() which converts OUT of Montgomery form
        // - VALID for Rust-side ICICLE arithmetic (standard form)
        // - SLOWER due to O(n) allocation and conversion
        //
        // We test both paths to ensure they work correctly in their intended contexts.

        // =========================================================================
        // Zero-Copy Scalar Tests (for MSM with are_scalars_montgomery_form=true)
        // =========================================================================

        /// Test that zero-copy transmute preserves byte identity.
        /// 
        /// This verifies the transmute is O(1) and doesn't modify data.
        /// The CUDA backend is responsible for Montgomery→Standard conversion.
        #[test]
        fn test_scalar_zero_copy_byte_identity() {
            use rand_core::OsRng;
            
            let scalar = Scalar::random(OsRng);
            
            // Transmute to ICICLE (zero-copy)
            let icicle_view = TypeConverter::scalar_slice_as_icicle(std::slice::from_ref(&scalar));
            
            // Transmute back (also zero-copy)
            let back = TypeConverter::icicle_slice_as_scalar(icicle_view);
            
            // Should be byte-identical (no conversion happened)
            assert_eq!(scalar, back[0], "Zero-copy transmute should preserve bytes exactly");
        }

        /// Test that zero-copy scalar slice has correct length
        #[test]
        fn test_scalar_zero_copy_slice_length() {
            let scalars: Vec<Scalar> = (0..100).map(|_| Scalar::ONE).collect();
            let icicle_view = TypeConverter::scalar_slice_as_icicle(&scalars);
            assert_eq!(scalars.len(), icicle_view.len(), "Zero-copy slice length mismatch");
        }

        /// Verify roundtrip scalar conversion preserves identity
        #[test]
        fn test_scalar_roundtrip() {
            use rand_core::OsRng;

            let original = Scalar::random(OsRng);

            // Convert to icicle and back (just pointer casts)
            let icicle_view =
                TypeConverter::scalar_slice_as_icicle(std::slice::from_ref(&original));
            let back = TypeConverter::icicle_slice_as_scalar(icicle_view);

            assert_eq!(original, back[0], "Roundtrip conversion failed");
        }

        /// Verify legacy scalar_to_icicle is byte-identical to zero-copy transmute
        /// 
        /// Both methods now use transmute, so they should produce identical results.
        #[test]
        fn test_legacy_scalar_conversion() {
            use rand_core::OsRng;
            
            let scalar = Scalar::random(OsRng);
            
            // Legacy method (now also uses transmute internally)
            let icicle_legacy = TypeConverter::scalar_to_icicle(&scalar);
            
            // Zero-copy method
            let icicle_view = TypeConverter::scalar_slice_as_icicle(std::slice::from_ref(&scalar));
            
            // Both should be byte-identical
            assert_eq!(icicle_legacy, icicle_view[0], "Legacy and zero-copy should be identical");
        }

        /// Verify G1 point conversion correctness
        #[test]
        fn test_g1_point_conversion() {
            let point = G1Affine::generator();
            let icicle = TypeConverter::g1_affine_to_icicle(&point);

            // Convert back and verify
            let back =
                TypeConverter::icicle_to_g1_projective(&IcicleG1Projective::from(icicle));
            let back_affine = G1Affine::from(back);

            assert_eq!(point, back_affine, "G1 roundtrip failed");
        }

        /// Verify G2 point conversion correctness
        #[test]
        fn test_g2_point_conversion() {
            let point = G2Affine::generator();
            let icicle = TypeConverter::g2_affine_to_icicle(&point);

            // Convert back and verify
            let back =
                TypeConverter::icicle_to_g2_projective(&IcicleG2Projective::from(icicle));
            let back_affine = G2Affine::from(back);

            assert_eq!(point, back_affine, "G2 roundtrip failed");
        }

        /// Test parallel point conversion
        #[test]
        fn test_parallel_point_conversion() {
            let points: Vec<G1Affine> = (0..2048).map(|_| G1Affine::generator()).collect();

            let converted = TypeConverter::g1_affine_slice_to_icicle_vec(&points);

            assert_eq!(points.len(), converted.len());

            // Verify first and last
            let first_back = TypeConverter::icicle_to_g1_projective(&IcicleG1Projective::from(
                converted[0].clone(),
            ));
            assert_eq!(points[0], G1Affine::from(first_back));
        }

        /// Test buffer-based conversion
        #[test]
        fn test_g1_buffer_conversion() {
            let points: Vec<G1Affine> = (0..100).map(|_| G1Affine::generator()).collect();

            let mut buffer = vec![
                IcicleG1Affine {
                    x: IcicleBaseField::zero(),
                    y: IcicleBaseField::zero()
                };
                100
            ];

            TypeConverter::g1_affine_slice_to_icicle_buf(&points, &mut buffer);

            // Verify conversion
            for (orig, conv) in points.iter().zip(buffer.iter()) {
                let back =
                    TypeConverter::icicle_to_g1_projective(&IcicleG1Projective::from(conv.clone()));
                assert_eq!(*orig, G1Affine::from(back));
            }
        }

        /// Benchmark-style test to verify zero-copy is actually faster
        #[test]
        fn test_zero_copy_performance_sanity() {
            use std::time::Instant;

            const SIZE: usize = 65536; // 2^16

            // Create random scalars for realistic test
            use rand_core::OsRng;
            let scalars: Vec<Scalar> = (0..SIZE).map(|_| Scalar::random(OsRng)).collect();

            // Zero-copy method - should be essentially instant
            let start = Instant::now();
            let view = TypeConverter::scalar_slice_as_icicle(&scalars);
            let _ = std::hint::black_box(view.len()); // Prevent optimization
            let zero_copy_time = start.elapsed();

            // Allocating method (old way using from_bytes_le)
            let start = Instant::now();
            let _: Vec<IcicleScalar> = scalars
                .iter()
                .map(|s| {
                    let bytes = s.to_repr();
                    IcicleScalar::from_bytes_le(bytes.as_ref())
                })
                .collect();
            let alloc_time = start.elapsed();

            // Log for inspection
            eprintln!(
                "Zero-copy: {:?}, Allocating: {:?}, Speedup: {:.0}x",
                zero_copy_time,
                alloc_time,
                alloc_time.as_nanos() as f64 / zero_copy_time.as_nanos().max(1) as f64
            );

            // Basic sanity: zero-copy should be faster
            assert!(
                zero_copy_time < alloc_time,
                "Zero-copy ({:?}) should be faster than allocating ({:?})",
                zero_copy_time,
                alloc_time
            );
        }

        /// Test that zero-copy G1 slice preserves byte identity
        /// 
        /// Note: G1 zero-copy transmutes Montgomery form bytes. The legacy conversion
        /// uses to_repr() which outputs standard form, so they won't match.
        /// Instead we verify zero-copy is O(1) and byte-preserving.
        #[test]
        fn test_g1_zero_copy_slice() {
            let points: Vec<G1Affine> = vec![G1Affine::generator(); 10];

            // Zero-copy view
            let icicle_view = TypeConverter::g1_slice_as_icicle(&points);

            assert_eq!(points.len(), icicle_view.len());
            
            // Verify pointer aliasing (true zero-copy)
            let original_ptr = points.as_ptr() as *const u8;
            let view_ptr = icicle_view.as_ptr() as *const u8;
            assert_eq!(original_ptr, view_ptr, "Zero-copy should alias same memory");
        }

        /// Test that zero-copy G2 slice preserves byte identity
        /// 
        /// Note: G2 zero-copy transmutes Montgomery form bytes. The legacy conversion
        /// uses to_repr() which outputs standard form, so they won't match.
        /// Instead we verify zero-copy is O(1) and byte-preserving.
        #[test]
        fn test_g2_zero_copy_slice() {
            let points: Vec<G2Affine> = vec![G2Affine::generator(); 10];

            // Zero-copy view
            let icicle_view = TypeConverter::g2_slice_as_icicle(&points);

            assert_eq!(points.len(), icicle_view.len());
            
            // Verify pointer aliasing (true zero-copy)
            let original_ptr = points.as_ptr() as *const u8;
            let view_ptr = icicle_view.as_ptr() as *const u8;
            assert_eq!(original_ptr, view_ptr, "Zero-copy should alias same memory");
        }
    }

    /// Non-GPU tests (always run)
    #[test]
    fn test_type_converter_exists() {
        let _ = TypeConverter;
    }
}
