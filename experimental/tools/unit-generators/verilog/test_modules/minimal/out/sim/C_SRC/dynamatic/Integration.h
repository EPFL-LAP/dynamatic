//===- Integration.h - Utilities for integration tests ----------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// To be included by all integration tests to get access to utilities necessary
// to run, for example, profiling for smart buffer placement.
//
// The file mainly defines the CALL_KERNEL macro to wrap a kernel function call
// as well as it arguments, which may be seen as some kind of code
// instrumentation. This is useful when one needs do something right before or
// after a kernel call. All the "instrumentation" code is written in
// clang-compliant (the version built through Polygeist) C++ and hidden behind
// off-by-default compile-time boolean macros to maintain pure-C-compatibility
// and so that one may choose which (if any) "instrumentation" to apply at
// compile-time. By default, the macro does nothing and just produces the kernel
// call as one would have written it.
//
// Right now there are two available instrumentations.
// 1. `PRINT_PROFILING_INFO` | Prints the value of kernel arguments to
// standard output to serve as profiling inputs for `frequency-profiler`.
// Every argument is printed on a single line, with arrays printed as a
// comma-separated list of their elements.
// 2. `HLS_VERIFICATION` | Stores value of kernel arguments before and after the
// kernel call (as well as the kernel's return value, if any) into individual
// files for use during C/VHDL co-simulation by our `hls-verifier`. The
// `HLS_VERIFICATION_PATH` macro indicates the relative path to the directory
// where arguments' value will be stored (`HLS_VERIFICATION_PATH/INPUT_VECTORS`
// for values before the kernel call and `HLS_VERIFICATION_PATH/C_OUT` for
// values after the kernel call).
//
//===- IMPORTANT NOTE -----------------------------------------------------===//
//
// There are two "limitations" when instrumenting a kernel call using
// CALL_KERNEL. However, they can easily be lifted should there be a need.
// 1. Statically-sized arrays are supported only up to 5 dimensions. Adding
// support for higher-dimensional arrays is as simple as adding more `dumpArg`
// functions similar to those already present.
// 2. Kernels must have a maximum of 16 arguments. Adding support for kernels
// with more arguments requires modifying a couple macros.
//   - HAS_ARGS_IMPL (before argument N, add _17, _18, ... )
//   - HAS_ARGS (add ARGS as many times as the number of extra arguments you
//   want to support before NO_ARGS)
//   - VA_NUM_ARGS_IMPL (before argument N, add _17, _18, ... )
//   - VA_NUM_ARGS (after variadic argument in expansion, add ..., 18, 17)
//   - DUMP_ARG_* (add DUMP_ARG_17, DUMP_ARG_18, ...)
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_INTEGRATION_H
#define DYNAMATIC_INTEGRATION_H

#define STRINGIFY_IMPL(str) #str
#define STRINGIFY(str) STRINGIFY_IMPL(str)

#define CONCAT_IMPL(x, y) x##y
#define CONCAT(x, y) CONCAT_IMPL(x, y)

/// Expands to the 18th macro argument.
#define HAS_ARGS_IMPL(_kernel, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11,   \
                      _12, _13, _14, _15, _16, N, ...)                         \
  N

/// HAS_ARGS will expand to NO_ARGS if kernelAndArgs has size 1 (just the kernel
/// name). Otherwise it will expand to ARGS.
#define HAS_ARGS(kernelAndArgs...)                                             \
  HAS_ARGS_IMPL(kernelAndArgs, ARGS, ARGS, ARGS, ARGS, ARGS, ARGS, ARGS, ARGS, \
                ARGS, ARGS, ARGS, ARGS, ARGS, ARGS, ARGS, ARGS, NO_ARGS)

//===----------------------------------------------------------------------===//
// PRINT_PROFILING_INFO/HLS_VERIFICATION - Common code
//===----------------------------------------------------------------------===//

#if defined(PRINT_PROFILING_INFO) || defined(HLS_VERIFICATION)
#include <cstddef>
#include <ostream>

using OS = std::basic_ostream<char>;

template <typename T>
static void scalarPrinter(const T &arg, OS &os);

template <typename T>
static void arrayPrinter(const T *arrayPtr, size_t size, OS &os);

/// Dumps the contents of a scalar type.
template <typename T>
static void dumpArg(const T &arg, OS &os) {
  scalarPrinter(arg, os);
}

/// Dumps the contents of a statically sized 1-dimensional array.
template <typename T, size_t Size1>
static void dumpArg(const T (&arrayArg)[Size1], OS &os) {
  arrayPrinter((T *)arrayArg, Size1, os);
}

/// Dumps the contents of a statically sized 2-dimensional array.
template <typename T, size_t Size1, size_t Size2>
static void dumpArg(const T (&arrayArg)[Size1][Size2], OS &os) {
  arrayPrinter((T *)arrayArg, Size1 * Size2, os);
}

/// Dumps the contents of a statically sized 3-dimensional array.
template <typename T, size_t Size1, size_t Size2, size_t Size3>
static void dumpArg(const T (&arrayArg)[Size1][Size2][Size3], OS &os) {
  arrayPrinter((T *)arrayArg, Size1 * Size2 * Size3, os);
}

/// Dumps the contents of a statically sized 4-dimensional array.
template <typename T, size_t Size1, size_t Size2, size_t Size3, size_t Size4>
static void dumpArg(const T (&arrayArg)[Size1][Size2][Size3][Size4], OS &os) {
  arrayPrinter((T *)arrayArg, Size1 * Size2 * Size3 * Size4, os);
}

/// Dumps the contents of a statically sized 5-dimensional array.
template <typename T, size_t Size1, size_t Size2, size_t Size3, size_t Size4,
          size_t Size5>
static void dumpArg(const T (&arrayArg)[Size1][Size2][Size3][Size4][Size5],
                    OS &os) {
  arrayPrinter((T *)arrayArg, Size1 * Size2 * Size3 * Size4 * Size5, os);
}

/// And on and on... Go further with higher-dimensional arrays if you want!
#endif // defined(PRINT_PROFILING_INFO) || defined (HLS_VERIFICATION)

//===----------------------------------------------------------------------===//
// PRINT_PROFILING_INFO - Compile for profiling using frequency-profiler
//===----------------------------------------------------------------------===//

#ifdef PRINT_PROFILING_INFO
#include "stdint.h"
#include <cassert>
#include <iostream>
#include <sstream>
#include <string>

// Dummy template for generating a compile-time error when the branch is
// initiantiated.
template <typename T>
inline constexpr bool always_false = false;

template <typename T>
std::string formatElement(const T &element) {
  std::ostringstream oss;
  if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double> ||
                std::is_same_v<T, int> || std::is_same_v<T, unsigned>) {
    // We can use the default handler for printing float, double, and int.
    oss << element;
  } else if constexpr (std::is_same_v<T, int8_t> ||
                       std::is_same_v<T, uint8_t> ||
                       std::is_same_v<T, int16_t> ||
                       std::is_same_v<T, uint16_t>) {
    // C++ can correctly print the value of int, float, double, etc..  However,
    // int8_t might be interpreted and printed as a char, so we need to convert
    // it to an int before printing it to stdout.
    oss << static_cast<int>(element);
  } else if constexpr (std::is_same_v<T, char>) {
    // A char can be directly printed as a integer (i.e., its ASCII code)
    oss << int(element);
  } else {
    static_assert(always_false<T>, "Unsupported type!");
  }
  return oss.str();
}

/// Writes the argument's directly to the stream.
template <typename T>
static void scalarPrinter(const T &arg, OS &os) {
  os << formatElement(arg) << std::endl;
}

/// Writes the array's content in row-major-order as a comma-separated list of
/// individual array elements.
template <typename T>
static void arrayPrinter(const T *arrayPtr, size_t size, OS &os) {
  if (size == 0) {
    os << std::endl;
    return;
  }
  for (size_t idx = 0; idx < size - 1; ++idx)
    os << formatElement(arrayPtr[idx]) << ",";
  os << formatElement(arrayPtr[size - 1]) << std::endl;
}

/// After dumping the contents of all kernel arguments to stdout, calls the
/// kernel with the arguments and returns its result. We use two variadic
/// template types (for what should semantically be the same thing) because
/// there may be slight differences in the types of arguments to *this* function
/// compared to the kernel's signature. For example, statically sized arrays
/// in function declarations are automatically converted to their corresponding
/// pointer type, but we still want to explicitly provide statically sized
/// arrays as arguments to this function so that we can match the version of
/// `dumpArg` that operates on those.
template <typename Res, typename... FunArgs, typename... RealArgs>
static Res callKernel(Res (*kernel)(FunArgs...), RealArgs &&...args) {
  (dumpArg(args, std::cout), ...);
  return kernel(std::forward<RealArgs>(args)...);
}

template <typename Res>
static Res callKernel(Res (*kernel)(void)) {
  return kernel();
}

#define CALL_NO_ARGS(kernel) callKernel(kernel)
#define CALL_ARGS(kernel, args...) callKernel(kernel, args)
#define CALL_KERNEL(kernelAndArgs...)                                          \
  CONCAT(CALL_, HAS_ARGS(kernelAndArgs))(kernelAndArgs)
#endif // PRINT_PROFILING_INFO

//===----------------------------------------------------------------------===//
// HLS_VERIFICATION - Compile for HLS verification using hls-verifier
//===----------------------------------------------------------------------===//

/// CALL_KERNEL macro expansion when compiling for HLS verification: logs the
/// values of all function arguments to dedicated folders on disk before and
/// after kernel execution. Also logs the kernel's return value, if it has one.
#ifdef HLS_VERIFICATION
#include "stdint.h"
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>

/// Whenever HLS_VERIFICATION is defined, this macro must contain the path to
/// the directory where the INPUT_VECTORS and C_OUT directories have been
/// created. By default, it points to the current working directory.
#ifndef HLS_VERIFICATION_PATH
#define HLS_VERIFICATION_PATH .
#endif // HLS_VERIFICATION_PATH

// NOLINTBEGIN(readability-identifier-naming)

/// Incremented once at each use of CALL_KERNEL.
static unsigned _transactionID_ = 0;
/// Outpath path prefix for storing the value of a function argument to a file
/// on disk.
static std::string _outPrefix_;

// NOLINTEND(readability-identifier-naming)

/// Specialization of the scalar printer for char.
template <>
void scalarPrinter<char>(const char &arg, OS &os) {
  // Print the char as a 2-digit hexadecimal number.
  os << "0x" << std::hex << std::setfill('0') << std::setw(2)
     << (static_cast<int>(arg)) << std::endl;
}

/// Specialization of the scalar printer for int8_t.
template <>
void scalarPrinter<int8_t>(const int8_t &arg, OS &os) {
  // Since int8_t only has 8 bits, it is sufficient to print it as a 2-digits
  // hexadecimal number.
  os << "0x" << std::hex << std::setfill('0') << std::setw(2)
     << static_cast<uint16_t>(static_cast<uint8_t>(arg)) << std::endl;
}

/// Specialization of the scalar printer for uint8_t.
template <>
void scalarPrinter<uint8_t>(const uint8_t &arg, OS &os) {
  // Since uint8_t only has 8 bits, it is sufficient to print it as a 2-digits
  // hexadecimal number.
  os << "0x" << std::hex << std::setfill('0') << std::setw(2)
     << static_cast<uint16_t>(static_cast<uint8_t>(arg)) << std::endl;
}

template <>
void scalarPrinter<float>(const float &arg, OS &os) {
  uint32_t bits;
  std::memcpy(&bits, &arg, sizeof(bits));
  os << "0x" << std::hex << std::setfill('0') << std::setw(8) << bits
     << std::endl;
}

/// Specialization of the scalar printer for double.
template <>
void scalarPrinter<double>(const double &arg, OS &os) {
  os << "0x" << std::hex << std::setfill('0') << std::setw(8)
     << *((const unsigned int *)(&arg)) << std::endl;
}

/// Writes the argument's as an 8-digits hexadecimal number padded with zeros
/// directly to stdout.
template <typename T>
static void scalarPrinter(const T &arg, OS &os) {
  os << "0x" << std::hex << std::setfill('0') << std::setw(8) << arg
     << std::endl;
}

/// Writes the array's content in row-major-order; one element per line,
/// formatted as 8-digits hexadecimal number padded with zeros.
template <typename T>
static void arrayPrinter(const T *arrayPtr, size_t size, OS &os) {
  for (size_t idx = 0; idx < size; ++idx)
    scalarPrinter(arrayPtr[idx], os);
}

template <typename T>
void dumpHLSArg(const T &arg, const char *argName) {
  std::string filepath = _outPrefix_ + argName + ".dat";
  std::ofstream outFile(filepath);
  if (!outFile.is_open()) {
    std::cerr << "Failed to open " << filepath << std::endl;
    exit(1);
  }

  outFile << "[[[runtime]]]" << std::endl
          << "[[transaction]] " << _transactionID_ << std::endl;
  dumpArg(arg, outFile);
  outFile << "[[/transaction]]" << std::endl << "[[[/runtime]]]" << std::endl;
  outFile.close();
}

/// Calls the kernel with the provided arguments.
template <typename... FunArgs, typename... RealArgs>
static void callKernel(void (*kernel)(FunArgs...), RealArgs &&...args) {
  return kernel(std::forward<RealArgs>(args)...);
}

/// Calls the kernel.
static void callKernel(void (*kernel)(void)) { return kernel(); }

/// Calls the kernel with the provided arguments and dumps the function's result
/// to a file.
template <typename Res, typename... FunArgs, typename... RealArgs>
static void callKernel(Res (*kernel)(FunArgs...), RealArgs &&...args) {
  Res res = kernel(std::forward<RealArgs>(args)...);
  dumpHLSArg(res, "out0");
}

/// Calls the kernel and dumps the function's result to a file.
template <typename Res>
static void callKernel(Res (*kernel)(void)) {
  Res res = kernel();
  dumpHLSArg(res, "out0");
}

// Following macro definitions strongly inspired by
// https://stackoverflow.com/questions/46725369/how-to-get-name-for-each-argument-in-variadic-macros

// This works for kernels with at most 16 arguments, but can be trivially
// extended to more if needed.

#define VA_NUM_ARGS_IMPL(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12,    \
                         _13, _14, _15, _16, N, ...)                           \
  N
#define VA_NUM_ARGS(...)                                                       \
  VA_NUM_ARGS_IMPL(__VA_ARGS__, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4,  \
                   3, 2, 1, 0)

#define HLS_DUMP(arg) dumpHLSArg(arg, #arg);
#define DUMP_ARG_0
#define DUMP_ARG_1(arg) HLS_DUMP(arg)
#define DUMP_ARG_2(arg, ...) HLS_DUMP(arg) DUMP_ARG_1(__VA_ARGS__)
#define DUMP_ARG_3(arg, ...) HLS_DUMP(arg) DUMP_ARG_2(__VA_ARGS__)
#define DUMP_ARG_4(arg, ...) HLS_DUMP(arg) DUMP_ARG_3(__VA_ARGS__)
#define DUMP_ARG_5(arg, ...) HLS_DUMP(arg) DUMP_ARG_4(__VA_ARGS__)
#define DUMP_ARG_6(arg, ...) HLS_DUMP(arg) DUMP_ARG_5(__VA_ARGS__)
#define DUMP_ARG_7(arg, ...) HLS_DUMP(arg) DUMP_ARG_6(__VA_ARGS__)
#define DUMP_ARG_8(arg, ...) HLS_DUMP(arg) DUMP_ARG_7(__VA_ARGS__)
#define DUMP_ARG_9(arg, ...) HLS_DUMP(arg) DUMP_ARG_8(__VA_ARGS__)
#define DUMP_ARG_10(arg, ...) HLS_DUMP(arg) DUMP_ARG_9(__VA_ARGS__)
#define DUMP_ARG_11(arg, ...) HLS_DUMP(arg) DUMP_ARG_10(__VA_ARGS__)
#define DUMP_ARG_12(arg, ...) HLS_DUMP(arg) DUMP_ARG_11(__VA_ARGS__)
#define DUMP_ARG_13(arg, ...) HLS_DUMP(arg) DUMP_ARG_12(__VA_ARGS__)
#define DUMP_ARG_14(arg, ...) HLS_DUMP(arg) DUMP_ARG_13(__VA_ARGS__)
#define DUMP_ARG_15(arg, ...) HLS_DUMP(arg) DUMP_ARG_14(__VA_ARGS__)
#define DUMP_ARG_16(arg, ...) HLS_DUMP(arg) DUMP_ARG_15(__VA_ARGS__)
#define DUMP_ARGS(args...) CONCAT(DUMP_ARG_, VA_NUM_ARGS(args))(args)

#define CALL_NO_ARGS(kernel)                                                   \
  {                                                                            \
    _outPrefix_ = std::string{STRINGIFY(HLS_VERIFICATION_PATH)} +              \
                  std::filesystem::path::preferred_separator + "C_OUT" +       \
                  std::filesystem::path::preferred_separator + "output_";      \
    callKernel(kernel);                                                        \
    ++_transactionID_;                                                         \
  }

#define CALL_ARGS(kernel, args...)                                             \
  {                                                                            \
    _outPrefix_ = std::string{STRINGIFY(HLS_VERIFICATION_PATH)} +              \
                  std::filesystem::path::preferred_separator +                 \
                  "INPUT_VECTORS" +                                            \
                  std::filesystem::path::preferred_separator + "input_";       \
    DUMP_ARGS(args)                                                            \
    _outPrefix_ = std::string{STRINGIFY(HLS_VERIFICATION_PATH)} +              \
                  std::filesystem::path::preferred_separator + "C_OUT" +       \
                  std::filesystem::path::preferred_separator + "output_";      \
    callKernel(kernel, args);                                                  \
    DUMP_ARGS(args)                                                            \
    ++_transactionID_;                                                         \
  }

#define CALL_KERNEL(kernelAndArgs...)                                          \
  CONCAT(CALL_, HAS_ARGS(kernelAndArgs))(kernelAndArgs)
#endif // HLS_VERIFICATION

//===----------------------------------------------------------------------===//
// Default compilation
//===----------------------------------------------------------------------===//

/// CALL_KERNEL macro expansion when compiling in no specific mode: simply
/// calls the kernel.
#ifndef PRINT_PROFILING_INFO
#ifndef HLS_VERIFICATION
#define CALL_KERNEL(kernel, args...) kernel(args)
#endif // HLS_VERIFICATION
#endif // PRINT_PROFILING_INFO

#endif // INTEGRATION_UTILS_H
