//===- CAnalyser.cpp --------------------------------------------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <fstream>
#include <sstream>

#define BOOST_EXCEPTION_DISABLE
#define BOOST_NO_EXCEPTIONS
#include <boost/regex.hpp>

#include "CAnalyser.h"
#include "HlsLogging.h"
#include "Utilities.h"

/// Just assert on an exception.
void boost::throw_exception(std::exception const &e) { assert(false); }

namespace hls_verify {
static const string LOG_TAG = "CAnalyzer";

/**
 * Trying to parse a function parameter declaration assuming it is of form
 * "data_type param_name[arrayLength]".
 * @param str
 * @param param
 * @return true if successful
 */
bool CAnalyser::parseAsArrayType(const string &str, CFunctionParameter &param) {
  boost::smatch m;
  boost::regex e("^\\s*(\\w+)\\s*([a-zA-Z_][a-zA-Z0-9_]*)\\s*((\\[\\s*(\\d+)"
                 "\\s*\\])+)\\s*$");

  // m[0] whole pattern, m[1] param type, m[2] param name, m[3] array length

  if (boost::regex_search(str, m, e)) {
    param.parameterName = m[2];
    param.isPointer = true;
    param.isReturn = false;

    string arrLenStr = m[3].str();
    vector<string> dimensions =
        split(arrLenStr.substr(1, arrLenStr.length() - 2), "[]");
    param.arrayLength = 1;
    param.dims = vector<int>();
    for (auto &dimension : dimensions) {
      int len = 0;
      if (dimension.substr(0, 2) == "0x") {
        len = stoi(dimension, nullptr, 16);
      } else {
        len = stoi(dimension, nullptr, 10);
      }
      param.dims.push_back(len);
      param.arrayLength *= len;
    }
    cout << "array length " << param.arrayLength << endl;

    param.immediateType = m[1];
    if (param.immediateType.substr(0, 3) == "in_") {
      param.isInput = true;
      param.isOutput = false;
    } else if (param.immediateType.substr(0, 4) == "out_") {
      param.isInput = false;
      param.isOutput = true;
    } else if (param.immediateType.substr(0, 6) == "inout_") {
      param.isInput = true;
      param.isOutput = true;
    } else {
      logErr(LOG_TAG,
             "Unable to identify IO type for parameter " + param.parameterName);
      return false;
    }

    logInf(LOG_TAG, "Parameter identified: " + param.parameterName +
                        " of type " + param.immediateType + "[" +
                        to_string(param.arrayLength) + "].");
    return true;
  }
  return false;
}

/**
 * Trying to parse a function parameter declaration assuming it is of form
 * "data_type * param_name".
 * @param str
 * @param param
 * @return true if successful
 */
bool CAnalyser::parseAsPointerType(const string &str,
                                   CFunctionParameter &param) {
  boost::smatch m;
  boost::regex e(R"(^\s*(\w+)\s*\*\s*(\w+)\s*$)");

  // m[0] whole pattern, m[1] param type, m[2] param name

  if (boost::regex_search(str, m, e)) {
    param.parameterName = m[2];
    param.isPointer = true;
    param.isReturn = false;

    param.arrayLength = 1;
    param.dims = vector<int>();
    param.dims.push_back(1);

    param.immediateType = m[1];
    if (param.immediateType.substr(0, 3) == "in_") {
      param.isInput = true;
      param.isOutput = false;
    } else if (param.immediateType.substr(0, 4) == "out_") {
      param.isInput = false;
      param.isOutput = true;
    } else if (param.immediateType.substr(0, 6) == "inout_") {
      param.isInput = true;
      param.isOutput = true;
    } else {
      logErr(LOG_TAG,
             "Unable to identify IO type for parameter " + param.parameterName);
      return false;
    }
    logInf(LOG_TAG, "Parameter identified: " + param.parameterName +
                        " of type " + param.immediateType + "[" +
                        to_string(param.arrayLength) + "].");
    return true;
  }
  return false;
}

/**
 * Trying to parse a function parameter declaration assuming it is of form
 * "data_type * param_name".
 * @param str
 * @param param
 * @return true if successful
 */
bool CAnalyser::parseAsSimpleType(const string &str,
                                  CFunctionParameter &param) {
  boost::smatch m;
  boost::regex e(R"(^\s*(\w+)\s*(\w+)\s*\s*$)");

  // m[0] whole pattern, m[1] param type, m[2] param name

  if (boost::regex_search(str, m, e)) {
    param.parameterName = m[2];
    param.isPointer = false;
    param.isReturn = false;

    param.arrayLength = 0;

    param.immediateType = m[1];
    if (param.immediateType.substr(0, 3) == "in_") {
      param.isInput = true;
      param.isOutput = false;
    } else if (param.immediateType.substr(0, 4) == "out_") {
      param.isInput = false;
      param.isOutput = true;
    } else if (param.immediateType.substr(0, 6) == "inout_") {
      param.isInput = true;
      param.isOutput = true;
    } else {
      logErr(LOG_TAG,
             "Unable to identify IO type for parameter " + param.parameterName);
      return false;
    }
    logInf(LOG_TAG, "Parameter identified: " + param.parameterName +
                        " of type " + param.immediateType + ".");
    return true;
  }
  return false;
}

/**
 * Trying to parse a function parameter and identify its attributes.
 * @param str
 * @param param
 * @return true if successful
 */
bool CAnalyser::paramFromString(const string &str, CFunctionParameter &param) {
  logInf(LOG_TAG, "Parsing parameter " + str);

  if (parseAsArrayType(str, param))
    return true;
  if (parseAsPointerType(str, param))
    return true;
  if (parseAsSimpleType(str, param))
    return true;

  logErr(LOG_TAG, "Unable to parse parameter from string \"" + str + "\".");

  return false;
}

/**
 * Finds the actual type of a user defined type. (Does not work with pointer
 * types. E.g., typedef int *int_ptr;)
 * @param cSrc C source code to search for the typedef
 * @param type User defined type for which the actual type should be found
 * @return actual primitive type of parameter 'type'
 */
string CAnalyser::getActualType(const string &cSrc, string type) {
  boost::smatch m;
  boost::regex e(R"(typedef\s+(\w+(\s+\w+)*)\s+)" + type + "\\s*;");

  while (boost::regex_search(cSrc, m, e)) {
    type = m[1].str();
    e = boost::regex(R"(typedef\s+(\w+(\s+\w+)*)\s+)" + type + "\\s*;");
  }

  istringstream stream(type);
  string result;
  stream >> result;
  string word;
  while (stream >> word) {
    result = result + " " + word;
  }
  return result;
}

/**
 * Returns the bit width for a given premitive type.
 * @param type
 * @return number of bits needed to represent a value of type 'type'
 */
int CAnalyser::getBitWidth(const string &type) {

  if (type == "char")
    return 8;
  if (type == "signed char")
    return 8;
  if (type == "unsigned char")
    return 8;

  if (type == "short")
    return 16;
  if (type == "short int")
    return 16;
  if (type == "signed short")
    return 16;
  if (type == "signed short int")
    return 16;
  if (type == "unsigned short")
    return 16;
  if (type == "unsigned short int")
    return 16;

  if (type == "int")
    return 32;
  if (type == "unsigned int")
    return 32;
  if (type == "signed int")
    return 32;
  if (type == "long")
    return 32;
  if (type == "unsigned long")
    return 32;
  if (type == "signed long")
    return 32;
  if (type == "long int")
    return 32;
  if (type == "unsigned long int")
    return 32;
  if (type == "signed long int")
    return 32;
  if (type == "unsigned")
    return 32;
  if (type == "signed")
    return 32;

  if (type == "long long")
    return 64;
  if (type == "unsigned long long")
    return 64;
  if (type == "signed long long")
    return 64;
  if (type == "long long int")
    return 64;
  if (type == "unsigned long long int")
    return 64;
  if (type == "signed long long int")
    return 64;

  if (type == "float")
    return 32;
  if (type == "double")
    return 64;
  if (type == "long double")
    return 128;

  return -1;
}

/**
 * Checks if a given type is a floating point type
 * @param type
 * @return true if 'type' is a floating point type
 */
bool CAnalyser::isFloatType(const string &type) {
  if (type == "float")
    return true;
  if (type == "double")
    return true;
  if (type == "long double")
    return true;
  return false;
}

string CAnalyser::getPreprocOutput(const string &cFilePath,
                                   const string &cIncludeDir) {
  string preProcCmd = "gcc -E " + cFilePath + " -I ";
  if (cIncludeDir.empty()) {
    preProcCmd += extractParentDirectoryPath(cFilePath);
  } else {
    preProcCmd += cIncludeDir;
  }
  preProcCmd += " -o hls_preproc.tmp";
  int status = system(preProcCmd.c_str());
  if (status != 0) {
    logErr(LOG_TAG, "Preprocessing failed for " + cFilePath + ".");
  }
  ifstream t("hls_preproc.tmp");
  std::string cSrc((istreambuf_iterator<char>(t)), istreambuf_iterator<char>());
  t.close();
  system("rm hls_preproc.tmp");

  return cSrc;
}

bool CAnalyser::parseCFunction(const string &cSrc, const string &fuvName,
                               CFunction &func) {
  func.functionName = fuvName;

  boost::smatch m;
  boost::regex eFuncDec(R"(^\s*((\w+)(\s+\w+)*)\s*)" + fuvName +
                        R"(\s*\(([^\)]*)\)\s*;)"); //\\s+

  // m[0] whole pattern, m[1] return type, m[2] param list

  if (!boost::regex_search(cSrc, m, eFuncDec)) {
    return false;
  }

  string returnType = m[1];
  string paramList = m[4];

  CFunctionParameter fuvRet;
  fuvRet.parameterName = "out0";
  fuvRet.isPointer = false;
  fuvRet.isReturn = true;
  fuvRet.isInput = false;
  fuvRet.arrayLength = 0;
  if (returnType != "void") {
    fuvRet.isOutput = true;
    fuvRet.immediateType = returnType;
    fuvRet.actualType = getActualType(cSrc, fuvRet.immediateType);
    fuvRet.dtWidth = getBitWidth(fuvRet.actualType);
    fuvRet.isFloatType = isFloatType(fuvRet.actualType);
    fuvRet.isIntType = !fuvRet.isFloatType;
    logInf(LOG_TAG, "Return identified: " + fuvRet.parameterName + " of type " +
                        fuvRet.immediateType + " (return value).");
  } else {
    fuvRet.isOutput = false;
    fuvRet.immediateType = "void";
    fuvRet.actualType = "void";
    fuvRet.dtWidth = 0;
    fuvRet.isFloatType = false;
    fuvRet.isIntType = false;
    logInf(LOG_TAG, "Function does not return any value.");
  }
  func.returnVal = fuvRet;

  vector<CFunctionParameter> params;

  vector<string> paramStrings = split(paramList, ",");
  for (const auto &i : paramStrings) {
    CFunctionParameter param;
    if (paramFromString(i, param)) {
      params.push_back(param);
    } else {
      logErr(LOG_TAG, "Function parameter \"" + i + "\" could not be parsed.");
      return false;
    }
  }

  for (size_t i = 0; i < params.size(); i++) {
    cout << "i" << i << endl;
    params[i].actualType = getActualType(cSrc, params[i].immediateType);
    params[i].dtWidth = getBitWidth(params[i].actualType);
    params[i].isFloatType = isFloatType(params[i].actualType);
    params[i].isIntType = !params[i].isFloatType;
    logInf(LOG_TAG, "Actual type of \"" + params[i].parameterName + "\" is \"" +
                        params[i].actualType + "\" (" +
                        to_string(params[i].dtWidth) + " bits).");
  }

  func.params = params;

  return true;
}
} // namespace hls_verify
