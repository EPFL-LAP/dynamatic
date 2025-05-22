//===- Utilities.cpp --------------------------------------------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Utilities.h"
#include "HlsLogging.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/Support/raw_ostream.h"
#include <cmath>
#include <dirent.h>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <unistd.h>

const string LOG_TAG = "UTIL";
using namespace mlir;

// @Jiahui17: Should we go for more digits?
const float DEFAULT_FLOAT_COMPARE_THRESHOLD = 0.00001;
const double DEFAULT_DOUBLE_COMPARE_THRESHOLD = 0.000000000001;

bool TokenCompare::compare(const string &token1, const string &token2) const {
  return token1 == token2;
}

IntegerCompare::IntegerCompare(bool isSigned) : isSigned(isSigned) {}

bool IntegerCompare::compare(const string &token1, const string &token2) const {
  string t1 = token1.substr(2);
  string t2 = token2.substr(2);
  while (t1.length() < t2.length())
    t1 = ((t2[0] >= '8' && isSigned) ? "f" : "0") + t1;

  while (t2.length() < t1.length())
    t2 = ((t1[0] >= '8' && isSigned) ? "f" : "0") + t2;

  return (t1 == t2);
}

// A safe and protable way of decoding a hex string into a 32-bit IEEE single
// precision float number
float hexStringToFloat(std::string hexStr) {
  if (hexStr.rfind("0x", 0) == 0 || hexStr.rfind("0X", 0) == 0) {
    hexStr = hexStr.substr(2);
  }

  uint32_t intVal;
  std::stringstream ss;
  ss << std::hex << hexStr;
  ss >> intVal;

  llvm::APInt bits(32, intVal);
  llvm::APFloat f(llvm::APFloat::IEEEsingle(), bits);

  return f.convertToFloat();
}

bool FloatCompare::compare(const string &token1, const string &token2) const {

  float f1 = hexStringToFloat(token1);
  float f2 = hexStringToFloat(token2);

  if (isnan(f1) && isnan(f2))
    return true;

  float diff = abs(f1 - f2);
  return (diff < DEFAULT_FLOAT_COMPARE_THRESHOLD);
}

// A safe and protable way of decoding a hex string into a 32-bit IEEE double
// precision float number
double hexStringToDouble(std::string hexStr) {
  if (hexStr.rfind("0x", 0) == 0 || hexStr.rfind("0X", 0) == 0) {
    hexStr = hexStr.substr(2);
  }
  uint64_t intVal;
  std::stringstream ss;
  ss << std::hex << hexStr;
  ss >> intVal;

  llvm::APInt bits(64, intVal);
  llvm::APFloat f(llvm::APFloat::IEEEdouble(), bits);

  return f.convertToDouble();
}

bool DoubleCompare::compare(const string &token1, const string &token2) const {

  double d1 = hexStringToDouble(token1);
  double d2 = hexStringToDouble(token2);

  if (isnan(d1) && isnan(d2))
    return true;

  double diff = abs(d1 - d2);
  return (diff < DEFAULT_DOUBLE_COMPARE_THRESHOLD);
}

string extractParentDirectoryPath(string filepath) {
  for (int i = filepath.length() - 1; i >= 0; i--)
    if (filepath[i] == '/')
      return filepath.substr(0, i);

  return ".";
}

string getApplicationDirectory() {
  char result[MAX_PATH_LENGTH];
  int count = readlink("/proc/self/exe", result, MAX_PATH_LENGTH);
  return string(result, (count > 0) ? count : 0);
}

mlir::LogicalResult compareFiles(const string &refFile, const string &outFile,
                                 std::unique_ptr<TokenCompare> tokenCompare) {
  ifstream ref(refFile.c_str());
  ifstream out(outFile.c_str());
  if (!ref.is_open()) {
    logErr(LOG_TAG, "Reference file does not exist: " + refFile);
    return failure();
  }
  if (!out.is_open()) {
    logErr(LOG_TAG, "Output file does not exist: " + outFile);
    return failure();
  }
  string str1, str2;
  int tn1, tn2;
  while (ref >> str1) {
    out >> str2;
    if (str1 == "[[[runtime]]]") {
      if (str2 == "[[[runtime]]]")
        continue;
      logErr("COMPARE", "Token mismatch: [" + str1 + "] and [" + str2 +
                            "] are not equal.");
      return failure();
    }
    if (str1 == "[[[/runtime]]]") {
      break;
    }
    if (str1 == "[[transaction]]") {
      ref >> tn1;
      out >> tn2;
      if (tn1 != tn2) {
        logErr("COMPARE", "Transaction number mismatch!");
        return failure();
      }
      continue;
    }
    if (str1 == "[[/transaction]]") {
      continue;
    }
    if (!tokenCompare->compare(str1, str2)) {
      logErr("COMPARE", "Token mismatch: [" + str1 + "] and [" + str2 +
                            "] are not equal (at transaction id " +
                            to_string(tn1) + ").");
      return failure();
    }
  }
  return success();
}

string trim(const string &str) {
  size_t first = str.find_first_not_of(" \t\n\r\f");
  if (string::npos == first)
    return str;

  size_t last = str.find_last_not_of(" \t\n\r\f");
  return str.substr(first, (last - first + 1));
}

vector<string> split(const string &str, const string &delims) {
  std::size_t current, previous = 0;
  current = str.find_first_of(delims);
  vector<string> cont;
  while (current != std::string::npos) {
    cont.push_back(trim(str.substr(previous, current - previous)));
    previous = str.find_first_not_of(delims, current);
    current = str.find_first_of(delims, previous);
  }
  std::string remainder = trim(str.substr(previous, current - previous));
  if (!remainder.empty())
    cont.push_back(remainder);
  return cont;
}

bool executeCommand(const string &command) {
  int status = system(command.c_str());
  if (status != 0) {
    logErr(LOG_TAG, "Execution failed for command [" + command + "]");

    assert(false && "Command execution failed!");
  }

  return (status == 0);
}

vector<string> getListOfFilesInDirectory(const string &directory,
                                         const string &extension) {
  vector<string> result;
  char sep = std::filesystem::path::preferred_separator;
  DIR *dirp = opendir(directory.c_str());
  struct dirent *dp;
  if (dirp != nullptr) {
    while ((dp = readdir(dirp)) != nullptr) {
      string temp(dp->d_name);
      if (temp.length() >= extension.length() &&
          temp.substr(temp.length() - extension.length(), extension.length())
                  .compare(extension) == 0)
        result.emplace_back(directory + sep + dp->d_name);
    }
    closedir(dirp);
  }
  return result;
}
