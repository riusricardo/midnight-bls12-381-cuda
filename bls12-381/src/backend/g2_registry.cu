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

/**
 * @file g2_registry.cu
 * @brief G2 MSM Backend Registration Implementation
 * 
 * ICICLE's core library doesn't include G2 registration functions
 * (register_g2_msm, register_g2_msm_precompute_bases) because G2 support
 * is optional and behind a G2_ENABLED flag.
 * 
 * Since we need G2 MSM support, we provide our own implementation of
 * these registration functions. This creates a simple dispatcher that
 * stores the registered backend implementation and calls it when needed.
 * 
 * NOTE: G1 registration (register_msm, register_msm_precompute_bases) is
 * provided by the ICICLE runtime library (libicicle_curve_bls12_381.so).
 * Tests should be run through ICICLE/Rust integration, not standalone.
 */

#include "icicle_backend_api.cuh"
#include <unordered_map>
#include <mutex>

namespace icicle {

// =============================================================================
// G2 MSM Dispatcher Registry
// =============================================================================
// Uses function-local statics (Meyer's Singleton) to avoid static initialization
// order fiasco. The maps are created on first use, not at library load time.

namespace {
    // Thread-safe accessor for G2 MSM backends registry
    std::unordered_map<std::string, MsmG2Impl>& get_g2_msm_backends() {
        static std::unordered_map<std::string, MsmG2Impl> backends;
        return backends;
    }
    
    // Thread-safe accessor for G2 precompute backends registry
    std::unordered_map<std::string, MsmG2PreComputeImpl>& get_g2_precompute_backends() {
        static std::unordered_map<std::string, MsmG2PreComputeImpl> backends;
        return backends;
    }
    
    // Thread-safe accessor for the registry mutex
    std::mutex& get_g2_registry_mutex() {
        static std::mutex mtx;
        return mtx;
    }
}

// Register a G2 MSM implementation for a device type
void register_g2_msm(const std::string& deviceType, MsmG2Impl impl) {
    std::lock_guard<std::mutex> lock(get_g2_registry_mutex());
    get_g2_msm_backends()[deviceType] = impl;
}

// Register a G2 MSM precompute implementation for a device type
void register_g2_msm_precompute_bases(const std::string& deviceType, MsmG2PreComputeImpl impl) {
    std::lock_guard<std::mutex> lock(get_g2_registry_mutex());
    get_g2_precompute_backends()[deviceType] = impl;
}

// Get the registered G2 MSM implementation for a device type
MsmG2Impl get_g2_msm_backend(const std::string& deviceType) {
    std::lock_guard<std::mutex> lock(get_g2_registry_mutex());
    auto it = get_g2_msm_backends().find(deviceType);
    if (it != get_g2_msm_backends().end()) {
        return it->second;
    }
    return nullptr;
}

// Get the registered G2 MSM precompute implementation for a device type
MsmG2PreComputeImpl get_g2_precompute_backend(const std::string& deviceType) {
    std::lock_guard<std::mutex> lock(get_g2_registry_mutex());
    auto it = get_g2_precompute_backends().find(deviceType);
    if (it != get_g2_precompute_backends().end()) {
        return it->second;
    }
    return nullptr;
}

} // namespace icicle
