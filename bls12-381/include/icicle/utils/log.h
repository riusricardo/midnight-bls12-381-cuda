/*
 * Copyright (C) 2023 Ingonyama (original MIT-licensed ICICLE code)
 * Copyright (C) 2026 BLS12-381 CUDA Backend Contributors (modifications and integration)
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of BLS12-381 CUDA Backend.
 * Originally from ICICLE library (https://github.com/ingonyama-zk/icicle)
 * Copied from: icicle/include/icicle/utils/log.h
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
#include <sstream>

#ifdef __ANDROID__
  #include <android/log.h>
#endif

#define ICICLE_LOG_VERBOSE Log(Log::Verbose)
#define ICICLE_LOG_DEBUG   Log(Log::Debug)
#define ICICLE_LOG_INFO    Log(Log::Info)
#define ICICLE_LOG_WARNING Log(Log::Warning)
#define ICICLE_LOG_ERROR   Log(Log::Error)

class Log
{
public:
  enum eLogLevel { Verbose, Debug, Info, Warning, Error };

  Log(eLogLevel level) : level{level}
  {
    if (level >= s_min_log_level) { oss << "[" << logLevelToString(level) << "] "; }
  }

  ~Log()
  {
    if (level >= s_min_log_level) {
#ifdef __ANDROID__
      // Use Android logcat
      android_LogPriority androidPriority = logLevelToAndroidPriority(level);
      __android_log_print(androidPriority, "ICICLE", "%s", oss.str().c_str());
#else
      // Use standard error stream for other platforms
      std::cerr << oss.str() << std::endl;
#endif
    }
  }

  template <typename T>
  Log& operator<<(const T& msg)
  {
    if (level >= s_min_log_level) { oss << msg; }
    return *this;
  }

  // Static method to set the minimum log level
  static void set_min_log_level(eLogLevel level) { s_min_log_level = level; }

private:
  eLogLevel level;
  std::ostringstream oss;

  // Convert log level to string
  const char* logLevelToString(eLogLevel level) const
  {
    switch (level) {
    case Verbose:
      return "VERBOSE";
    case Debug:
      return "DEBUG";
    case Info:
      return "INFO";
    case Warning:
      return "WARNING";
    case Error:
      return "ERROR";
    default:
      return "";
    }
  }

#ifdef __ANDROID__
  // Map custom log level to Android log priority
  android_LogPriority logLevelToAndroidPriority(eLogLevel level) const
  {
    switch (level) {
    case Verbose:
      return ANDROID_LOG_VERBOSE;
    case Debug:
      return ANDROID_LOG_DEBUG;
    case Info:
      return ANDROID_LOG_INFO;
    case Warning:
      return ANDROID_LOG_WARN;
    case Error:
      return ANDROID_LOG_ERROR;
    default:
      return ANDROID_LOG_UNKNOWN;
    }
  }
#endif

  // Static member to hold the minimum log level
#if defined(NDEBUG)
  static inline eLogLevel s_min_log_level = eLogLevel::Info;
#else
  static inline eLogLevel s_min_log_level = eLogLevel::Debug;
#endif

  // Note: for verbose, need to explicitly call `set_min_log_level(eLogLevel::Verbose)`
};