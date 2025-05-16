//===- Utilities.h ----------------------------------------------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef HLS_VERIFIER_UTILITIES_H
#define HLS_VERIFIER_UTILITIES_H

#include <string>
#include <vector>

#define MAX_PATH_LENGTH 5000

using namespace std;

namespace hls_verify {

class TokenCompare {
public:
  virtual bool compare(const string &token1, const string &token2) const;
  virtual ~TokenCompare() = default;
};

class IntegerCompare : public TokenCompare {
public:
  IntegerCompare() = default;
  IntegerCompare(bool isSigned);
  bool compare(const string &token1, const string &token2) const override;

private:
  bool isSigned;
};

class FloatCompare : public TokenCompare {
public:
  FloatCompare() = default;
  FloatCompare(float threshold);
  bool compare(const string &token1, const string &token2) const override;

private:
  float threshold;
  float decodeToken(unsigned int token) const;
};

class DoubleCompare : public TokenCompare {
public:
  DoubleCompare() = default;
  DoubleCompare(double threshold);
  bool compare(const string &token1, const string &token2) const override;

private:
  double threshold;
  double decodeToken(unsigned int token) const;
};

/**
 * Extracts the parent directory from a file path.
 * @param file_path path of the file.
 * @return parent directory path or "." if @file_path has no directory
 * separators.
 */
string extractParentDirectoryPath(string filepath);

/**
 * Get the directory of the executable file.
 * @return the directory of the executable file.
 */
string getApplicationDirectory();

/**
 * Compares two data files using a given token comparator.
 * @param refFilePath path of the reference data file
 * @param outFile path of the other data file
 * @param token_compare the comparator to be used for comparing two tokens
 * @return true if all comparisons succeed, false otherwise.
 */
bool compareFiles(const string &refFilePath, const string &outFile,
                  const TokenCompare *tokenCompare);

/**
 * Trims the leading and trailing white spaces.
 * @param str the string to be trimmed
 * @return trimmed string
 */
string trim(const string &str);

/**
 * Splits the given string using one of the delimiters.
 * @param str the string to be splitted
 * @param delims the list of delimitters
 * @return a vector of strings which are the parts after splitting.
 */
vector<string> split(const string &str, const string &delims = " \t\n\r\f");

/**
 * Get the number of transactions in a data file.
 * @param input_path path of the data file
 * @return the number of transactions in the given file.
 */
int getNumberOfTransactions(const string &inputPath);

/**
 * Execute the given command in a shell.
 * @param command the command to be executed
 * @return true if execution returns 0, false otherwise.
 */
bool executeCommand(const string &command);

/**
 * Get a list of file paths in the given directory that has the given extension.
 * @param directory the path of the directory
 * @param extension extension of the required files including '.' character
 * @return a vector of strings denoting the paths of the files in @directory
 * that has extension @extension.
 */
vector<string> getListOfFilesInDirectory(const string &directory,
                                         const string &extension = "");

} // namespace hls_verify

#endif // HLS_VERIFIER_UTILITIES_H
