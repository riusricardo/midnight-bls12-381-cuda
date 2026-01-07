# =============================================================================
# BLS12-381 CUDA Backend - CMake Build System
# =============================================================================
#
# This builds the complete BLS12-381 CUDA cryptographic library including
# open-source replacements for closed-source ICICLE CUDA backend libraries.
#
# BUILD:
#   mkdir build && cd build
#   cmake .. -DCMAKE_BUILD_TYPE=Release
#   make -j$(nproc)
#
# INSTALL TO ICICLE:
#   sudo make icicle-install
#
# ARTIFACTS:
#   lib/libicicle_backend_cuda_field_bls12_381.so  - Field operations (NTT, vec_ops)
#   lib/libicicle_backend_cuda_curve_bls12_381.so  - Curve operations (MSM, EC-NTT)
#   lib/libicicle_backend_cuda_device.so           - Device API (memory, streams)
#
# =============================================================================

cmake_minimum_required(VERSION 3.18)
project(icicle_backend_cuda_bls12_381 LANGUAGES CXX CUDA)

# =============================================================================
# Configuration Options
# =============================================================================

# For distribution: ON for both (larger binaries but works everywhere)
# For development:  OFF for both (smaller binaries, faster builds)
option(CUDA_STATIC_RUNTIME "Use static CUDA runtime for redistributability" ON)
option(BUILD_TESTS "Build test executables" OFF)
option(G2_ENABLED "Enable G2 curve support" OFF)
option(GLV_ENABLED "Enable GLV endomorphism and batch scalar mul - experimental, not used by MSM" OFF)
option(MULTI_GPU_ARCH "Build for multiple GPU architectures (larger binary)" ON)

# ICICLE installation directory (for icicle-install target)
set(ICICLE_INSTALL_DIR "/opt/icicle" CACHE PATH "ICICLE installation directory")

# Mark overrides for summary
if(NOT CUDA_STATIC_RUNTIME)
    set(CMAKE_STATIC_RUNTIME_OVERRIDE " (overridden)")
endif()
if(NOT MULTI_GPU_ARCH)
    set(CMAKE_MULTI_GPU_ARCH_OVERRIDE " (overridden)")
endif()

# =============================================================================
# CUDA Configuration
# =============================================================================

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CUDA architectures
# Default: Build only for current/latest GPU (smaller binary, faster build)
# With MULTI_GPU_ARCH=ON: Build for all supported GPUs (larger binary, more portable)
if(MULTI_GPU_ARCH)
    # Build for all supported architectures (Turing through Blackwell)
    # 75 = Turing (RTX 20xx), 80 = Ampere (A100), 86 = Ampere (RTX 30xx)
    # 89 = Ada Lovelace (RTX 40xx), 100 = Hopper (H100), 120 = Blackwell (RTX 50xx)
    set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89 100 120 CACHE STRING "CUDA architectures" FORCE)
    message(STATUS "Building for multiple GPU architectures: ${CMAKE_CUDA_ARCHITECTURES}")
else()
    # Default: Build only for Blackwell (RTX 50 series) - smallest binary
    # Users can override with: cmake -DCMAKE_CUDA_ARCHITECTURES="120"
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 120 CACHE STRING "CUDA architectures" FORCE)
    endif()
    message(STATUS "Building for GPU architecture: ${CMAKE_CUDA_ARCHITECTURES}")
endif()

find_package(CUDAToolkit REQUIRED)

if(CUDA_STATIC_RUNTIME)
    set(CUDA_RUNTIME_LIB CUDA::cudart_static)
    find_package(Threads REQUIRED)
    message(STATUS "Using static CUDA runtime (larger binary, more portable)")
else()
    set(CUDA_RUNTIME_LIB CUDA::cudart)
    message(STATUS "Using dynamic CUDA runtime")
endif()

# =============================================================================
# Compiler Flags
# =============================================================================

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -use_fast_math")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -fPIC")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -Wall -Xcompiler -Wextra")

if(G2_ENABLED)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DG2_ENABLED")
    message(STATUS "G2 curve support enabled")
endif()

if(GLV_ENABLED)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DGLV_ENABLED")
    message(STATUS "GLV endomorphism support enabled (experimental)")
else()
    message(STATUS "GLV endomorphism disabled (MSM uses Pippenger instead)")
endif()

set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -use_fast_math -Xptxas -O3 --extra-device-vectorization")
set(CMAKE_CUDA_FLAGS_DEBUG "-g -G -DDEBUG -lineinfo")

# =============================================================================
# Include Directories
# =============================================================================

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include/icicle)

# =============================================================================
# Source Files
# =============================================================================

# Field library sources
set(FIELD_SOURCES
    src/field/vec_ops.cu
    src/field/ntt_kernels.cu
    src/backend/icicle_field_api.cu
)

# Curve library sources (depends on field library)
set(CURVE_SOURCES
    src/curve/point_ops.cu
    src/curve/msm_kernels.cu
    src/backend/icicle_curve_api.cu
    src/backend/g2_registry.cu
)

# Device library sources (standalone)
set(DEVICE_SOURCES
    src/device/cuda_device_api.cu
)

# =============================================================================
# Helper macro for library properties
# =============================================================================

macro(set_icicle_library_properties target)
    set_target_properties(${target} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RUNTIME_LIBRARY $<IF:$<BOOL:${CUDA_STATIC_RUNTIME}>,Static,Shared>
        POSITION_INDEPENDENT_CODE ON
        BUILD_WITH_INSTALL_RPATH ON
        INSTALL_RPATH "$ORIGIN"
    )
endmacro()

# =============================================================================
# Field Backend Library
# =============================================================================

add_library(icicle_backend_cuda_field_bls12_381 SHARED ${FIELD_SOURCES})
target_link_libraries(icicle_backend_cuda_field_bls12_381 
    ${CUDA_RUNTIME_LIB}
    $<$<BOOL:${CUDA_STATIC_RUNTIME}>:Threads::Threads>
    $<$<BOOL:${CUDA_STATIC_RUNTIME}>:${CMAKE_DL_LIBS}>
)
set_icicle_library_properties(icicle_backend_cuda_field_bls12_381)

# =============================================================================
# Curve Backend Library
# =============================================================================

add_library(icicle_backend_cuda_curve_bls12_381 SHARED ${CURVE_SOURCES})
target_link_libraries(icicle_backend_cuda_curve_bls12_381 
    icicle_backend_cuda_field_bls12_381
    ${CUDA_RUNTIME_LIB}
    $<$<BOOL:${CUDA_STATIC_RUNTIME}>:Threads::Threads>
    $<$<BOOL:${CUDA_STATIC_RUNTIME}>:${CMAKE_DL_LIBS}>
)
set_icicle_library_properties(icicle_backend_cuda_curve_bls12_381)

# =============================================================================
# Device Backend Library (replaces closed-source ICICLE device library)
# =============================================================================

add_library(icicle_backend_cuda_device SHARED ${DEVICE_SOURCES})
target_link_libraries(icicle_backend_cuda_device
    ${CUDA_RUNTIME_LIB}
    $<$<BOOL:${CUDA_STATIC_RUNTIME}>:Threads::Threads>
    $<$<BOOL:${CUDA_STATIC_RUNTIME}>:${CMAKE_DL_LIBS}>
)
set_icicle_library_properties(icicle_backend_cuda_device)

# =============================================================================
# Convenience target to build all ICICLE libraries
# =============================================================================

add_custom_target(icicle
    DEPENDS 
        icicle_backend_cuda_field_bls12_381
        icicle_backend_cuda_curve_bls12_381
        icicle_backend_cuda_device
    COMMENT "Building all ICICLE backend libraries"
)

# Print summary after building icicle target
add_custom_command(TARGET icicle POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E echo ""
    COMMAND ${CMAKE_COMMAND} -E echo "=================================================="
    COMMAND ${CMAKE_COMMAND} -E echo " ICICLE-compatible backends built!"
    COMMAND ${CMAKE_COMMAND} -E echo " Field:  $<TARGET_FILE:icicle_backend_cuda_field_bls12_381>"
    COMMAND ${CMAKE_COMMAND} -E echo " Curve:  $<TARGET_FILE:icicle_backend_cuda_curve_bls12_381>"
    COMMAND ${CMAKE_COMMAND} -E echo " Device: $<TARGET_FILE:icicle_backend_cuda_device>"
    COMMAND ${CMAKE_COMMAND} -E echo "=================================================="
    COMMAND ${CMAKE_COMMAND} -E echo ""
    COMMAND ${CMAKE_COMMAND} -E echo "To install - replace closed-source ICICLE libraries:"
    COMMAND ${CMAKE_COMMAND} -E echo "  sudo make icicle-install"
    COMMAND ${CMAKE_COMMAND} -E echo ""
)

# =============================================================================
# ICICLE Installation Target (replaces closed-source libraries)
# =============================================================================

# Use a shell script for installation to properly handle backups and errors
add_custom_target(icicle-install
    DEPENDS icicle
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/scripts/icicle_install.sh 
        "${CMAKE_BINARY_DIR}" "${ICICLE_INSTALL_DIR}"
    COMMENT "Installing ICICLE backend libraries to ${ICICLE_INSTALL_DIR}"
    VERBATIM
)

# =============================================================================
# Standard Installation (for non-ICICLE use)
# =============================================================================

install(TARGETS 
    icicle_backend_cuda_field_bls12_381 
    icicle_backend_cuda_curve_bls12_381
    LIBRARY DESTINATION lib/backend/bls12_381/cuda
)

install(TARGETS
    icicle_backend_cuda_device
    LIBRARY DESTINATION lib/backend/cuda
)

# =============================================================================
# Tests
# =============================================================================

if(BUILD_TESTS)
    enable_testing()
    
    # Macro for test executables
    macro(add_cuda_test name source)
        add_executable(${name} ${source})
        target_link_libraries(${name}
            icicle_backend_cuda_curve_bls12_381
            icicle_backend_cuda_field_bls12_381
            ${CUDA_RUNTIME_LIB}
            $<$<BOOL:${CUDA_STATIC_RUNTIME}>:Threads::Threads>
            $<$<BOOL:${CUDA_STATIC_RUNTIME}>:${CMAKE_DL_LIBS}>
        )
        set_target_properties(${name} PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_RUNTIME_LIBRARY $<IF:$<BOOL:${CUDA_STATIC_RUNTIME}>,Static,Shared>
        )
        add_test(NAME ${name} COMMAND ${name})
    endmacro()
    
    # =========================================================================
    # Security Audit Test Suite
    # =========================================================================
    # These tests are designed for comprehensive security audits of the
    # BLS12-381 CUDA implementation. They cover:
    # - Known Answer Tests (KAT) from official specifications
    # - Field arithmetic algebraic properties
    # - Curve operation correctness
    # - MSM correctness and edge cases
    # - NTT correctness and properties
    # - Security edge cases and constant-time operations
    
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_known_answer_vectors.cu)
        add_cuda_test(test_known_answer_vectors tests/test_known_answer_vectors.cu)
    endif()
    
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_field_properties.cu)
        add_cuda_test(test_field_properties tests/test_field_properties.cu)
    endif()
    
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_curve_operations.cu)
        add_cuda_test(test_curve_operations tests/test_curve_operations.cu)
    endif()
    
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_msm_security.cu)
        add_cuda_test(test_msm_security tests/test_msm_security.cu)
    endif()
    
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_ntt_security.cu)
        add_cuda_test(test_ntt_security tests/test_ntt_security.cu)
    endif()
    
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_security_edge_cases.cu)
        add_cuda_test(test_security_edge_cases tests/test_security_edge_cases.cu)
    endif()
    
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_vec_ops.cu)
        add_cuda_test(test_vec_ops tests/test_vec_ops.cu)
    endif()
    
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_point_ops.cu)
        add_cuda_test(test_point_ops tests/test_point_ops.cu)
    endif()
    
    # Convenience target to run all security audit tests
    add_custom_target(security-audit
        COMMAND ${CMAKE_CTEST_COMMAND} -R "test_known_answer_vectors|test_field_properties|test_curve_operations|test_msm_security|test_ntt_security|test_security_edge_cases|test_vec_ops|test_point_ops" --output-on-failure
        DEPENDS 
            test_known_answer_vectors
            test_field_properties
            test_curve_operations
            test_msm_security
            test_ntt_security
            test_security_edge_cases
            test_vec_ops
            test_point_ops
        COMMENT "Running security audit test suite"
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    )
endif()

# =============================================================================
# Summary
# =============================================================================

message(STATUS "")
message(STATUS "=== CUDA Backend Configuration ===")
message(STATUS "  CUDA Architectures:  ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  Multi-GPU Arch:      ${MULTI_GPU_ARCH}${CMAKE_MULTI_GPU_ARCH_OVERRIDE}")
message(STATUS "  Static Runtime:      ${CUDA_STATIC_RUNTIME}${CMAKE_STATIC_RUNTIME_OVERRIDE}")
message(STATUS "  G2 Support:          ${G2_ENABLED}")
message(STATUS "  Build Tests:         ${BUILD_TESTS}")
message(STATUS "  ICICLE Install Dir:  ${ICICLE_INSTALL_DIR}")
message(STATUS "")
message(STATUS "Build targets:")
message(STATUS "  make                       - Build all libraries")
message(STATUS "  make icicle                - Build ICICLE-compatible libraries")
message(STATUS "  sudo make icicle-install   - Install to ICICLE directory")
message(STATUS "  make test                  - Run tests")
message(STATUS "")
message(STATUS "Build options (prefix with cmake .. -DCMAKE_BUILD_TYPE=Release):")
message(STATUS "  -DMULTI_GPU_ARCH=ON/OFF               - Multi-GPU support")
message(STATUS "  -DCUDA_STATIC_RUNTIME=ON/OFF          - Static CUDA runtime")
message(STATUS "  -DCMAKE_CUDA_ARCHITECTURES=\"89\"       - Specific GPU arch")
message(STATUS "  -DG2_ENABLED=ON/OFF                   - G2 curve support")
message(STATUS "  -DBUILD_TESTS=ON/OFF                  - Build test executables")
message(STATUS "")
