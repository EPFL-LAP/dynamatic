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
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_INTEGRATION_H
#define DYNAMATIC_INTEGRATION_H

//===----------------------------------------------------------------------===//
// PRINT_PROFILING_INFO/HLS_VERIFICATION - Common code
//===----------------------------------------------------------------------===//

#if defined(PRINT_PROFILING_INFO) || defined(HLS_VERIFICATION)
#include <cstddef>
#include <ostream>
#include <type_traits>

using OS = std::basic_ostream<char>;

template <typename T>
static void scalarPrinter(const T &arg, OS &os);

template <typename T>
static void arrayPrinter(const T *arrayPtr, size_t size, OS &os);

/// Compile-time function to compute the total number of elements
/// in a multidimensional array type.
///
/// Example:
///   getArraySize<int[2][3][4]>() == 24 (2 * 3 * 4)
///
/// It works recursively:
///   - If T is one-dimensional (rank == 1), return its extent (size).
///   - Otherwise, multiply the first dimension by the size of the remaining
///   dimensions.
template <typename T>
constexpr size_t getArraySize() {
  if constexpr (std::rank_v<T> == 1)
    return std::extent_v<T>;
  else
    return std::extent_v<T> * getArraySize<std::remove_extent_t<T>>();
}

/// Helper metafunction to deduce the "value type" of an array type.
///
/// Behavior:
///   - If T is not an array, the value type is T itself.
///   - If T is an array type, one dimension is peeled off at a time
///     (via specialization) until the base element type is reached.
///
/// Examples:
///   getValueType<int[2][3][4]> == int
///   getValueType<double[5]>    == double
///   getValueType<char>         == char
template <typename T>
struct getValueTypeImpl {
  using type = T;
};
template <typename T, size_t TSize>
struct getValueTypeImpl<T[TSize]> {
  using type = typename getValueTypeImpl<T>::type;
};

/// Alias to simplify usage of getValueTypeImpl.
/// Instead of writing getValueTypeImpl<T>::type,
/// you can just use getValueType<T>.
template <typename T>
using getValueType = typename getValueTypeImpl<T>::type;

/// Dumps the contents of a scalar type.
template <typename T>
static std::enable_if_t<std::rank_v<T> == 0> dumpArg(const T &arg, OS &os) {
  scalarPrinter(arg, os);
}

/// Dumps the contents of a statically sized n-dimensional array.
template <typename T>
static std::enable_if_t<std::rank_v<T> != 0> dumpArg(const T &argArray,
                                                     OS &os) {
  constexpr size_t totalSize = getArraySize<T>();
  arrayPrinter((getValueType<T> *)argArray, totalSize, os);
}

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

#ifndef __clang__
#error "This code requires Clang (for _BitInt builtins)"
#endif

#if !__has_builtin(__is_integral)
#error "Your Clang version does not support __is_integral!"
#endif

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
  } else if constexpr (__is_integral(T) && !std::is_same_v<T, bool>) {
    // Fallback case for the other integral types (e.g., _BitInt(N) introduced
    // in C23). In this case, we simply cast it to "long long"
    // TODO: maybe check if "long long" actually fits the value?
    oss << static_cast<long long>(element);
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

#define CALL_KERNEL(kernelAndArgs...) callKernel(kernelAndArgs)
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
#include <vector>

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

/// Specialization of the scalar printer for int.
template <>
void scalarPrinter<int>(const int &arg, OS &os) {
  // Print the char as a 8-digit hexadecimal number.
  os << "0x" << std::hex << std::setfill('0') << std::setw(8) << arg
     << std::endl;
}

/// Writes the argument's as an 8-digits hexadecimal number padded with zeros
/// directly to stdout.
template <typename T>
static void scalarPrinter(const T &arg, OS &os) {
  os << "0x" << std::hex << std::setfill('0') << std::setw(8)
     << static_cast<unsigned long long>(arg) << std::endl;
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

/// Helper function to dump a parameter pack of arguments with their
/// corresponding names.
///
/// Usage:
///   dumpArgsImpl("fir, a, b, c", someFunc, argA, argB, argC);
///
/// Behavior:
///   - The first token in the string (e.g. "fir") is ignored.
///   - The first function parameter (`Func`) is also ignored.
///     It is passed redundantly as both a name and an argument
///     to simplify the caller macro.
///   - The remaining tokens (e.g. "a", "b", "c") are extracted by splitting on
///   commas.
///   - Each argument in the parameter pack is passed to `dumpHLSArg` together
///   with
///     the matching name from the parsed list.
///   - Whitespace around names is trimmed before use.
///
/// Example:
///   Input:  names = "fir, a, b, c"
///           args  = { argA, argB, argC }
///   Effect: calls
///              dumpHLSArg(argA, "a");
///              dumpHLSArg(argB, "b");
///              dumpHLSArg(argC, "c");
template <typename Func, typename... Args>
void dumpArgsImpl(const char *names, Func, Args &&...args) {
  std::string s(names); // e.g. "fir, a, b, c"
  std::stringstream ss(s);
  std::string name;

  // Split the input string on ',' to extract argument names
  std::vector<std::string> nameList;

  // Discard the first token (the function name)
  std::getline(ss, name, ',');
  while (std::getline(ss, name, ',')) {
    // Trim leading and trailing whitespace
    name.erase(0, name.find_first_not_of(" \t"));
    name.erase(name.find_last_not_of(" \t") + 1);
    nameList.push_back(name);
  }

  size_t i = 0;
  // At this point, nameList should be {"a", "b", "c"}
  // Iterate over args and pair each with the corresponding name
  ((dumpHLSArg(args, nameList[i++].c_str())), ...);
}

/// Calls the kernel with the provided arguments.
template <typename... FunArgs, typename... RealArgs>
static void callKernelImpl(void (*kernel)(FunArgs...), RealArgs &&...args) {
  return kernel(std::forward<RealArgs>(args)...);
}

/// Calls the kernel with the provided arguments and dumps the function's result
/// to a file.
template <typename Res, typename... FunArgs, typename... RealArgs>
static void callKernelImpl(Res (*kernel)(FunArgs...), RealArgs &&...args) {
  Res res = kernel(std::forward<RealArgs>(args)...);
  dumpHLSArg(res, "out0");
}

#define STRINGIFY_IMPL(str) #str
#define STRINGIFY(str) STRINGIFY_IMPL(str)

#define CALL_KERNEL(kernelAndArgs...)                                          \
  {                                                                            \
    _outPrefix_ = std::string{(STRINGIFY(HLS_VERIFICATION_PATH))} +            \
                  std::filesystem::path::preferred_separator +                 \
                  "INPUT_VECTORS" +                                            \
                  std::filesystem::path::preferred_separator + "input_";       \
    dumpArgsImpl(#kernelAndArgs, kernelAndArgs);                               \
    _outPrefix_ = std::string{(STRINGIFY(HLS_VERIFICATION_PATH))} +            \
                  std::filesystem::path::preferred_separator + "C_OUT" +       \
                  std::filesystem::path::preferred_separator + "output_";      \
    callKernelImpl(kernelAndArgs);                                             \
    dumpArgsImpl(#kernelAndArgs, kernelAndArgs);                               \
    ++_transactionID_;                                                         \
  }

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
