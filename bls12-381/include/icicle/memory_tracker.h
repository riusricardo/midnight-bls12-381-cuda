/*
 * Copyright (C) 2023 Ingonyama (original MIT-licensed ICICLE code)
 * Copyright (C) 2026 BLS12-381 CUDA Backend Contributors (modifications and integration)
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of BLS12-381 CUDA Backend.
 * Originally from ICICLE library (https://github.com/ingonyama-zk/icicle)
 * Copied from: icicle/include/icicle/memory_tracker.h
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

#pragma once

#include <iostream>
#include <map>
#include <mutex>
#include <optional>
#include <thread>

namespace icicle {

  template <typename T>
  class MemoryTracker
  {
  public:
    // Add a new allocation with a void* address and an associated data of type T
    void add_allocation(const void* address, size_t size, T associated_data)
    {
      std::lock_guard<std::mutex> lock(mutex_);
      allocations_.insert(std::make_pair(address, AllocationInfo{size, associated_data}));
    }

    // Remove an allocation
    void remove_allocation(const void* address)
    {
      std::lock_guard<std::mutex> lock(mutex_);
      allocations_.erase(address);
    }

    // Identify the base address and offset for a given address
    std::optional<std::pair<const T*, size_t /*offset*/>> identify(const void* address)
    {
      std::lock_guard<std::mutex> lock(mutex_);
      auto it = allocations_.upper_bound(address);
      if (it == allocations_.begin()) { return std::nullopt; }
      --it;
      const char* start = static_cast<const char*>(it->first);
      const char* end = start + it->second.size_;
      if (start <= static_cast<const char*>(address) && static_cast<const char*>(address) < end) {
        size_t offset = static_cast<const char*>(address) - start;
        return std::make_pair(&it->second.associated_data_, offset);
      }
      return std::nullopt;
    }

  private:
    struct AllocationInfo {
      size_t size_;
      const T associated_data_;

      AllocationInfo(size_t size, T associated_data) : size_{size}, associated_data_{associated_data} {}
    };

    std::map<const void*, AllocationInfo> allocations_;
    std::mutex mutex_;
  };

} // namespace icicle