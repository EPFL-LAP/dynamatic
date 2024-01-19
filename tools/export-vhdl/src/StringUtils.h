//===- StringUtils.h - Utilities to manipulate strings ----------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef EXPORT_VHDL_STRING_UTILS_H
#define EXPORT_VHDL_STRING_UTILS_H

#include <string>
#include <vector>

void stringSplit(const std::string &s, char c, std::vector<std::string> &v);
std::string stringRemoveBlank(std::string stringInput);
std::string stringClean(std::string stringInput);
std::string stringConstant(unsigned long int value, int size);
std::string cleanEntity(const std::string &filename);
std::string stripExtension(std::string stringInput,
                           const std::string &extension);

#endif // EXPORT_VHDL_STRING_UTILS_H
