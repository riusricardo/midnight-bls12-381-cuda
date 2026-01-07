/*
 * Copyright (C) 2023 Ingonyama (original MIT-licensed ICICLE code)
 * Copyright (C) 2026 BLS12-381 CUDA Backend Contributors (modifications and integration)
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of BLS12-381 CUDA Backend.
 * Originally from ICICLE library (https://github.com/ingonyama-zk/icicle)
 * Copied from: icicle/include/icicle/utils/utils.h
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
#include <string>
#include <memory>
#include <cxxabi.h>
#include <iomanip>
#include <iostream>

#define CONCAT_DIRECT(a, b) a##_##b
#define CONCAT_EXPAND(a, b) CONCAT_DIRECT(a, b)

// Define a separate macro to ensure expansion before stringification
#define STRINGIFY_EXPAND(x) STRINGIFY(x)
#define STRINGIFY(x)        #x

#define UNIQUE(a) CONCAT_EXPAND(a, __LINE__)

// Template function to demangle the name of the type
template <typename T>
std::string demangle()
{
  int status = -4;
  std::unique_ptr<char, void (*)(void*)> res{
    abi::__cxa_demangle(typeid(T).name(), nullptr, nullptr, &status), std::free};

  return (status == 0) ? res.get() : typeid(T).name();
}

// Debug
__attribute__((unused))
static void print_bytes(const std::byte* data, const uint nof_elements, const uint element_size)
{
  for (uint element_idx = 0; element_idx < nof_elements; ++element_idx) {
    std::cout << "0x";
    for (uint byte_idx = 0; byte_idx < element_size; ++byte_idx) {
      std::cout << std::hex << std::setw(2) << std::setfill('0')
                << static_cast<int>(data[element_idx * element_size + byte_idx]);
    }
    std::cout << std::dec << ",\n";
  }
}