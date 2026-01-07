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
 * CUDA Device API for ICICLE
 * 
 * This file implements the DeviceAPI interface using standard CUDA runtime API calls.
 */

#include <iostream>
#include <cstring>
#include "icicle/device_api.h"
#include "icicle/errors.h"
#include "icicle/utils/log.h"
#include "icicle/runtime.h"
#include "cuda_runtime.h"

using namespace icicle;

class CudaDeviceAPI : public DeviceAPI
{
public:
  eIcicleError set_device(const Device& device) override
  {
    cudaError_t err = cudaSetDevice(device.id);
    return (err == cudaSuccess) ? eIcicleError::SUCCESS : eIcicleError::INVALID_DEVICE;
  }

  eIcicleError get_device_count(int& device_count) const override
  {
    cudaError_t err = cudaGetDeviceCount(&device_count);
    return (err == cudaSuccess) ? eIcicleError::SUCCESS : eIcicleError::INVALID_DEVICE;
  }

  // Memory management
  eIcicleError allocate_memory(void** ptr, size_t size) const override
  {
    cudaError_t err = cudaMalloc(ptr, size);
    return (err == cudaSuccess) ? eIcicleError::SUCCESS : eIcicleError::ALLOCATION_FAILED;
  }

  eIcicleError allocate_memory_async(void** ptr, size_t size, icicleStreamHandle stream) const override
  {
    cudaError_t err = cudaMallocAsync(ptr, size, reinterpret_cast<cudaStream_t>(stream));
    return (err == cudaSuccess) ? eIcicleError::SUCCESS : eIcicleError::ALLOCATION_FAILED;
  }

  eIcicleError free_memory(void* ptr) const override
  {
    cudaError_t err = cudaFree(ptr);
    return (err == cudaSuccess) ? eIcicleError::SUCCESS : eIcicleError::DEALLOCATION_FAILED;
  }

  eIcicleError free_memory_async(void* ptr, icicleStreamHandle stream) const override
  {
    cudaError_t err = cudaFreeAsync(ptr, reinterpret_cast<cudaStream_t>(stream));
    return (err == cudaSuccess) ? eIcicleError::SUCCESS : eIcicleError::DEALLOCATION_FAILED;
  }

  eIcicleError get_available_memory(size_t& total /*OUT*/, size_t& free /*OUT*/) const override
  {
    cudaError_t err = cudaMemGetInfo(&free, &total);
    return (err == cudaSuccess) ? eIcicleError::SUCCESS : eIcicleError::UNKNOWN_ERROR;
  }

  eIcicleError memset(void* ptr, int value, size_t size) const override
  {
    cudaError_t err = cudaMemset(ptr, value, size);
    return (err == cudaSuccess) ? eIcicleError::SUCCESS : eIcicleError::UNKNOWN_ERROR;
  }

  eIcicleError memset_async(void* ptr, int value, size_t size, icicleStreamHandle stream) const override
  {
    cudaError_t err = cudaMemsetAsync(ptr, value, size, reinterpret_cast<cudaStream_t>(stream));
    return (err == cudaSuccess) ? eIcicleError::SUCCESS : eIcicleError::UNKNOWN_ERROR;
  }

  // Data transfer
  eIcicleError copy(void* dst, const void* src, size_t size, eCopyDirection direction) const override
  {
    cudaMemcpyKind cuda_copy_kind = direction == eCopyDirection::HostToDevice   ? cudaMemcpyHostToDevice
                                    : direction == eCopyDirection::DeviceToHost ? cudaMemcpyDeviceToHost
                                                                                : cudaMemcpyDeviceToDevice;
    cudaError_t err = cudaMemcpy(dst, src, size, cuda_copy_kind);
    return (err == cudaSuccess) ? eIcicleError::SUCCESS : eIcicleError::COPY_FAILED;
  }

  eIcicleError copy_async(
    void* dst, const void* src, size_t size, eCopyDirection direction, icicleStreamHandle stream) const override
  {
    cudaMemcpyKind cuda_copy_kind = direction == eCopyDirection::HostToDevice   ? cudaMemcpyHostToDevice
                                    : direction == eCopyDirection::DeviceToHost ? cudaMemcpyDeviceToHost
                                                                                : cudaMemcpyDeviceToDevice;
    cudaError_t err = cudaMemcpyAsync(dst, src, size, cuda_copy_kind, reinterpret_cast<cudaStream_t>(stream));
    return (err == cudaSuccess) ? eIcicleError::SUCCESS : eIcicleError::COPY_FAILED;
  }

  // Synchronization
  eIcicleError synchronize(icicleStreamHandle stream = nullptr) const override
  {
    cudaError_t err =
      (stream == nullptr) ? cudaDeviceSynchronize() : cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream));
    return (err == cudaSuccess) ? eIcicleError::SUCCESS : eIcicleError::SYNCHRONIZATION_FAILED;
  }

  // Stream management
  eIcicleError create_stream(icicleStreamHandle* stream) const override
  {
    cudaStream_t cudaStream;
    cudaError_t err = cudaStreamCreate(&cudaStream);
    *stream = reinterpret_cast<icicleStreamHandle>(cudaStream);
    return (err == cudaSuccess) ? eIcicleError::SUCCESS : eIcicleError::STREAM_CREATION_FAILED;
  }

  eIcicleError destroy_stream(icicleStreamHandle stream) const override
  {
    cudaError_t err = cudaStreamDestroy(reinterpret_cast<cudaStream_t>(stream));
    return (err == cudaSuccess) ? eIcicleError::SUCCESS : eIcicleError::STREAM_DESTRUCTION_FAILED;
  }

  eIcicleError get_device_properties(DeviceProperties& properties) const override
  {
    properties.using_host_memory = false;
    properties.num_memory_regions = 1;
    properties.supports_pinned_memory = true;
    return eIcicleError::SUCCESS;
  }
};

// Register as "CUDA"
REGISTER_DEVICE_API("CUDA", CudaDeviceAPI);
