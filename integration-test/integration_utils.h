//===- integration_utils.h - Utilities for integration tests ------*- C -*-===//
//
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
// The file mainly defines the CALL macro to wrap a kernel function call as well
// as it arguments, which may be seen as some kind of code instrumentation. This
// is occasionally useful when one needs do something right before or after a
// kernel call. All the "instrumentation" code is written in C++ and hidden
// behind off-by-default compile boolean macros to maintain pure-C-compatibility
// and so that one may choose which (if any) "instrumentation" to apply at
// compile-time. By default, the macro does nothing and just produces the kernel
// call as one would have written it.
//
// Right now there is only a single instrumentation controlled by the
// `PRINT_PROFILING_INFO` boolean macro. Its effect is to print the value of
// wrapped function arguments to standard output to serve as profiling inputs to
// our MLIR std-level profiler. Every argument is printed on a single line, with
// arrays printed as a comma-separated list of their elements.
//
//===----------------------------------------------------------------------===//

#ifndef INTEGRATION_UTILS_H
#define INTEGRATION_UTILS_H

#ifdef PRINT_PROFILING_INFO
#include <cstddef>
#include <iostream>

/// Dumps the contents of an array of known size.
template <typename T>
static void dumpArray(T *arrayPtr, size_t size) {
  if (size == 0) {
    std::cout << std::endl;
    return;
  }
  for (size_t idx = 0; idx < size - 1; ++idx)
    std::cout << arrayPtr[idx] << ",";
  std::cout << arrayPtr[size - 1] << std::endl;
}

/// Dumps the value of the argument.
template <typename T>
static void dumpArg(T arg) {
  std::cout << arg << std::endl;
}

/// Dumps the contents of a statically sized 1-dimensional array.
template <typename T, size_t Size1>
static void dumpArg(T (&arrayArg)[Size1]) {
  dumpArray((T *)arrayArg, Size1);
}

/// Dumps the contents of a statically sized 2-dimensional array.
template <typename T, size_t Size1, size_t Size2>
static void dumpArg(T (&arrayArg)[Size1][Size2]) {
  dumpArray((T *)arrayArg, Size1 * Size2);
}

/// Dumps the contents of a statically sized 3-dimensional array.
template <typename T, size_t Size1, size_t Size2, size_t Size3>
static void dumpArg(T (&arrayArg)[Size1][Size2][Size3]) {
  dumpArray((T *)arrayArg, Size1 * Size2 * Size3);
}

/// Dumps the contents of a statically sized 4-dimensional array.
template <typename T, size_t Size1, size_t Size2, size_t Size3, size_t Size4>
static void dumpArg(T (&arrayArg)[Size1][Size2][Size3][Size4]) {
  dumpArray((T *)arrayArg, Size1 * Size2 * Size3 * Size4);
}

/// Dumps the contents of a statically sized 5-dimensional array.
template <typename T, size_t Size1, size_t Size2, size_t Size3, size_t Size4,
          size_t Size5>
static void dumpArg(T (&arrayArg)[Size1][Size2][Size3][Size4][Size5]) {
  dumpArray((T *)arrayArg, Size1 * Size2 * Size3 * Size4 * Size5);
}

/// And on and on... Go further with higher-dimensional arrays if you want!

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
  (dumpArg(args), ...);
  return kernel(std::forward<RealArgs>(args)...);
}
#endif // PRINT_PROFILING_INFO

#ifdef PRINT_PROFILING_INFO
#define CALL_KERNEL(KERNEL, ARGS...) callKernel(KERNEL, ARGS);
#else
#define CALL_KERNEL(KERNEL, ARGS...) KERNEL(ARGS)
#endif // PRINT_PROFILING_INFO

#endif // INTEGRATION_UTILS_H
