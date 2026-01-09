//! Test comparing ICICLE NTT with halo2curves FFT
//!
//! This test investigates the relationship between ICICLE's NTT and halo2curves' FFT
//! to understand if there's a transformation that makes them compatible.

#[cfg(feature = "gpu")]
#[test]
fn compare_icicle_vs_halo2_fft() {
    use midnight_bls12_381_cuda::GpuNttContext;
    use midnight_curves::Fq;
    use ff::{Field, PrimeField};
    
    println!("\n=== Comparing ICICLE NTT vs halo2curves FFT ===\n");
    
    let k = 10u32;
    let n = 1 << k;
    
    // Create test data: simple sequence 1, 2, 3, ...
    let original: Vec<Fq> = (0..n).map(|i| Fq::from((i + 1) as u64)).collect();
    
    // ========== Halo2curves FFT ==========
    let mut halo2_data = original.clone();
    
    // Compute omega for halo2curves (using midnight-curves ROOT_OF_UNITY)
    let mut halo2_omega = Fq::ROOT_OF_UNITY;
    for _ in k..Fq::S {
        halo2_omega = halo2_omega.square();
    }
    
    println!("Halo2curves omega (from midnight ROOT_OF_UNITY):");
    println!("  {:?}\n", halo2_omega);
    
    halo2curves::fft::best_fft(&mut halo2_data, halo2_omega, k);
    
    // ========== ICICLE NTT ==========
    let mut icicle_data = original.clone();
    
    let ctx = GpuNttContext::new(k).expect("Failed to create GPU NTT context");
    ctx.forward_ntt_inplace(&mut icicle_data)
        .expect("Failed to run ICICLE NTT");
    
    // ========== Compare Results ==========
    println!("Original data (first 8):");
    println!("  {:?}\n", &original[..8.min(n)]);
    
    println!("Halo2curves FFT result (first 8):");
    println!("  {:?}\n", &halo2_data[..8.min(n)]);
    
    println!("ICICLE NTT result (first 8):");
    println!("  {:?}\n", &icicle_data[..8.min(n)]);
    
    // Check if results match
    let mut matches = 0;
    let mut mismatches = 0;
    
    for (i, (h, ic)) in halo2_data.iter().zip(icicle_data.iter()).enumerate() {
        if h == ic {
            matches += 1;
        } else {
            mismatches += 1;
            if mismatches <= 5 {
                println!("Mismatch at index {}: halo2={:?}, icicle={:?}", i, h, ic);
            }
        }
    }
    
    println!("\n=== Summary ===");
    println!("Matches: {}/{}", matches, n);
    println!("Mismatches: {}/{}", mismatches, n);
    
    if matches == n {
        println!("\n✅ Results MATCH! ICICLE and halo2curves use compatible roots!");
    } else {
        println!("\n❌ Results DIFFER. Different roots of unity are being used.");
        
        // Try to find a permutation relationship
        println!("\nSearching for permutation pattern...");
        
        // Check if it's a simple index permutation (bit reversal, etc.)
        let mut found_permutation = true;
        for i in 0..n.min(16) {
            let halo2_val = halo2_data[i];
            if let Some(pos) = icicle_data.iter().position(|&x| x == halo2_val) {
                if i < 8 {
                    println!("  halo2[{}] == icicle[{}]", i, pos);
                }
            } else {
                found_permutation = false;
                break;
            }
        }
        
        if found_permutation {
            println!("\n✨ Looks like a permutation! Values appear in different order.");
        } else {
            println!("\n🔍 Not a simple permutation. Values may be scaled/transformed.");
            
            // Check if there's a scaling relationship
            if let Some(ratio) = compute_scaling_ratio(&halo2_data, &icicle_data) {
                println!("\nFound potential scaling factor: {:?}", ratio);
            }
        }
    }
    
    // ========== Verify ICICLE Roundtrip ==========
    ctx.inverse_ntt_inplace(&mut icicle_data)
        .expect("Failed to run ICICLE inverse NTT");
    
    let roundtrip_ok = icicle_data == original;
    println!("\nICICLE NTT roundtrip: {}", if roundtrip_ok { "✅ OK" } else { "❌ FAILED" });
    
    assert!(roundtrip_ok, "ICICLE NTT roundtrip should work");
}

#[cfg(feature = "gpu")]
fn compute_scaling_ratio(a: &[midnight_curves::Fq], b: &[midnight_curves::Fq]) -> Option<midnight_curves::Fq> {
    use ff::Field;
    
    // Find first non-zero pair
    for (x, y) in a.iter().zip(b.iter()) {
        if !bool::from(x.is_zero()) && !bool::from(y.is_zero()) {
            // ratio = y / x
            let x_inv_opt: Option<midnight_curves::Fq> = x.invert().into();
            if let Some(x_inv) = x_inv_opt {
                return Some(*y * x_inv);
            }
        }
    }
    None
}

#[cfg(feature = "gpu")]
#[test]
fn test_get_icicle_root_compatibility() {
    use midnight_bls12_381_cuda::ntt::cpu::compute_omega;
    use midnight_curves::Fq;
    use ff::{Field, PrimeField};
    
    println!("\n=== Testing ICICLE Root of Unity ===\n");
    
    let k = 16u32;
    let domain_size = 1u64 << k;
    
    // Get our computed root (same as what we use for CPU NTT)
    let computed_root = compute_omega(k);
    
    // Get midnight's root
    let mut midnight_root = Fq::ROOT_OF_UNITY;
    for _ in k..Fq::S {
        midnight_root = midnight_root.square();
    }
    
    println!("Domain size: 2^{} = {}", k, domain_size);
    println!("\nComputed root:  {:?}", computed_root);
    println!("Midnight root:  {:?}", midnight_root);
    println!("\nRoots match: {}", computed_root == midnight_root);
    
    // Verify both are primitive nth roots
    let mut computed_test = computed_root;
    let mut midnight_test = midnight_root;
    
    for _ in 0..k {
        computed_test = computed_test.square();
        midnight_test = midnight_test.square();
    }
    
    println!("\nComputed root^(2^{}) = {:?}", k, computed_test);
    println!("Midnight root^(2^{}) = {:?}", k, midnight_test);
    
    assert_eq!(computed_test, Fq::ONE, "Computed root should be primitive {}th root", domain_size);
    assert_eq!(midnight_test, Fq::ONE, "Midnight root should be primitive {}th root", domain_size);
    assert_eq!(computed_root, midnight_root, "Roots should match");
}
