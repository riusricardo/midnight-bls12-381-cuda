/*
 * Copyright (C) 2023 Ingonyama (original MIT-licensed ICICLE code)
 * Copyright (C) 2026 BLS12-381 CUDA Backend Contributors (modifications and integration)
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of BLS12-381 CUDA Backend.
 * Originally from ICICLE library (https://github.com/ingonyama-zk/icicle)
 * Copied from: icicle/include/icicle/device.h
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
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

#include <cstring>
#include <string>
#include <iostream>
#include <memory>

namespace icicle {

  /**
   * @brief Represents a device with a type and id.
   */
  struct Device {
    char type[32]; // Device type string (e.g., "CUDA", "CPU", "CUDA-PQC")
    int id;        // Device ID

    Device() : id(0) { strcpy(type, "CPU"); }

    Device(const char* type_input, int id_input = 0) : id(id_input) { copy_str(type, type_input); }

    Device(const std::string& type_input, int id_input = 0) : id(id_input) { copy_str(type, type_input.c_str()); }

    static void copy_str(char* dst, const char* src)
    {
      if (src != nullptr) {
        strncpy(dst, src, sizeof(Device::type) - 1);
        dst[sizeof(Device::type) - 1] = '\0';
      }
    }

    bool operator==(const Device& other) const { return strcmp(type, other.type) == 0 && id == other.id; }

    bool operator!=(const Device& other) const { return !(*this == other); }
  };

  inline std::ostream& operator<<(std::ostream& os, const Device& device)
  {
    os << "Device(type: " << device.type << ", id: " << device.id << ")";
    return os;
  }

  /**
   * @brief Struct to hold the properties of a device.
   */
  struct DeviceProperties {
    bool using_host_memory;
    int num_memory_regions;
    bool supports_pinned_memory;
  };

} // namespace icicle
