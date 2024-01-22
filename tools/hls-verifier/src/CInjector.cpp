//===- CInjector.cpp --------------------------------------------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CInjector.h"
#include <boost/regex.hpp>
#include <fstream>
#include <iostream>
#include <sstream>
#include <utility>

using namespace std;

namespace hls_verify {

int findEndBraceIndex(const string &cFuv, int startIndex) {
  int openCount = 1;
  for (size_t i = startIndex + 1; i < cFuv.length(); i++) {
    if (cFuv[i] == '{')
      openCount++;
    else if (cFuv[i] == '}')
      openCount--;
    if (openCount == 0) {
      return i;
    }
  }
  return cFuv.length() - 1;
}

CInjector::CInjector(const VerificationContext &ctx) : ctx(ctx) {

  const CFunction &func = ctx.getCFuv();
  // add stdio.h include and get preprocessor output
  ifstream cFuvSrcFileIn(ctx.getCFuvPath());
  string cFuv((istreambuf_iterator<char>(cFuvSrcFileIn)),
              istreambuf_iterator<char>());
  cFuvSrcFileIn.close();
  ofstream cFuvSrcFileOut(ctx.getCFuvPath() + ".tmp.c");
  cFuvSrcFileOut << "#include <stdio.h>" << endl << cFuv;
  cFuvSrcFileOut.close();

  cFuv = CAnalyser::getPreprocOutput(ctx.getCFuvPath() + ".tmp.c");
  system(("rm " + ctx.getCFuvPath() + ".tmp.c").c_str());

  boost::smatch m;
  boost::regex e;

  string prefix;
  string head;
  string body;
  string suffix;

  stringstream out;

  e = boost::regex("^\\s*" + func.returnVal.immediateType + "\\s*" +
                   func.functionName + R"(\s*\([^\)]*\)\s*\{)");
  boost::regex_search(cFuv, m, e);

  if (m.empty())
    return;

  int funcBe = m.position() + m.length();
  int funcEn = findEndBraceIndex(cFuv, funcBe);

  prefix = cFuv.substr(0, m.position());
  head = m[0];
  body = cFuv.substr(funcBe, funcEn - funcBe);
  suffix = cFuv.substr(funcEn, cFuv.length() - funcEn);

  if (func.returnVal.actualType == "void") {
    body = body + "\n\treturn;\n";
  }

  out << prefix;
  out << endl << "int hlsv_transaction_id = -1;" << endl << endl;
  out << head << endl << endl;
  out << getVariableDeclarations(func);
  out << getFileIoCodeForInput(func);

  e = boost::regex("return([^;]*);");
  while (boost::regex_search(body, m, e)) {
    out << body.substr(0, m.position()) << endl;
    out << getFileIoCodeForOutput(func, string(m[1]));
    body = body.substr(m.position() + string(m[0]).length());
  }

  out << body;
  out << suffix;

  injectedFuvSrc = out.str();
}

string CInjector::getVariableDeclarations(const CFunction &func) {
  stringstream out;
  if (func.returnVal.actualType != "void") {
    out << "\t" << func.returnVal.immediateType + " hlsv_return;" << endl;
    out << "\tFILE * hlsv_of_return;" << endl;
  }
  for (const auto &param : func.params) {
    if (param.isInput) {
      out << "\tFILE * hlsv_if_" << param.parameterName << ";" << endl;
    }
  }
  for (const auto &param : func.params) {
    if (param.isOutput) {
      out << "\tFILE * hlsv_of_" << param.parameterName << ";" << endl;
    }
  }
  out << endl << "\tint hlsv_i = 0;" << endl;

  size_t maxDim = 0;
  for (const auto &param : func.params) {
    if (param.isPointer && param.dims.size() > maxDim) {
      maxDim = param.dims.size();
    }
  }
  for (size_t i = 0; i < maxDim; i++) {
    out << endl << "\tint hlsv_i" << i << " = 0;" << endl;
  }

  return out.str();
}

string CInjector::getFileIoCodeForInput(const CFunction &func) {
  stringstream out;
  out << endl << "\thlsv_transaction_id++;" << endl << endl;
  for (const auto &param : func.params) {
    if (param.isInput) {
      out << getFileIoCodeForInputParam(param);
    }
  }
  return out.str();
}

string arrayPrintCode(const CFunctionParameter &param,
                      const string &fileNamePrefix) {
  stringstream out;
  for (size_t d = 0; d < param.dims.size(); d++) {
    out << string(d, '\t') << "\tfor(hlsv_i" << d << " = 0; hlsv_i" << d
        << " < " << param.dims[d] << "; hlsv_i" << d << "++){" << endl;
  }
  out << string(param.dims.size(), '\t');
  // TODO: edit the following line for non integer types
  if (param.isFloatType) {
    if (param.dtWidth == 32) {
      out << "\tfprintf(" << fileNamePrefix << param.parameterName
          << R"(, "0x%08x\n", *((unsigned int *)&)" << param.parameterName;
      for (size_t d = 0; d < param.dims.size(); d++) {
        out << "[hlsv_i" << to_string(d) << "]";
      }
      out << "));" << endl;
    } else {
      out << "\tfprintf(" << fileNamePrefix << param.parameterName
          << R"(, "0x%08llx\n", *((unsigned long long *)&)"
          << param.parameterName;
      for (size_t d = 0; d < param.dims.size(); d++) {
        out << "[hlsv_i" << to_string(d) << "]";
      }
      out << "));" << endl;
    }
  } else {
    out << "\t\tfprintf(" << fileNamePrefix << param.parameterName
        << R"(, "0x%08llx\n", (long long))" << param.parameterName;
    for (size_t d = 0; d < param.dims.size(); d++) {
      out << "[hlsv_i" << to_string(d) << "]";
    }
    out << ");" << endl;
  }
  for (size_t d = 0; d < param.dims.size(); d++) {
    out << string(param.dims.size() - d, '\t') << "\t}" << endl;
  }
  return out.str();
}

string CInjector::getFileIoCodeForInputParam(const CFunctionParameter &param) {
  stringstream out;
  out << "\thlsv_if_" << param.parameterName << " = fopen(\""
      << ctx.getInputVectorPath(param) << R"(", "a");)" << endl;
  out << "\tfprintf(hlsv_if_" << param.parameterName
      << R"(, "[[transaction]] %d\n", hlsv_transaction_id);)" << endl;
  if (param.isPointer) {
    out << arrayPrintCode(param, "hlsv_if_");
  } else {
    if (param.isFloatType) {
      if (param.dtWidth == 32) {
        out << "\tfprintf(hlsv_if_" << param.parameterName
            << R"(, "0x%08x\n",  *((unsigned int *)&)" << param.parameterName
            << "));" << endl;

      } else {
        out << "\tfprintf(hlsv_if_" << param.parameterName
            << R"(, "0x%08llx\n",  *((unsigned long long *)&)"
            << param.parameterName << "));" << endl;
      }
    } else {
      out << "\tfprintf(hlsv_if_" << param.parameterName
          << R"(, "0x%08llx\n", (long long))" << param.parameterName << ");"
          << endl;
    }
  }
  out << "\tfprintf(hlsv_if_" << param.parameterName
      << R"(, "[[/transaction]]\n");)" << endl;
  out << "\tfclose(hlsv_if_" << param.parameterName << ");" << endl << endl;
  return out.str();
}

string CInjector::getFileIoCodeForOutput(const CFunction &func,
                                         const string &actualReturnValue) {
  stringstream out;
  out << "\t{" << endl;
  for (const auto &param : func.params) {
    if (param.isOutput) {
      out << getFileIoCodeForOutputParam(param);
    }
  }
  out << getFileIoCodeForReturnValue(func.returnVal, actualReturnValue);
  out << "\t}" << endl;
  return out.str();
}

string CInjector::getFileIoCodeForOutputParam(const CFunctionParameter &param) {
  stringstream out;
  out << "\t\thlsv_of_" << param.parameterName << " = fopen(\""
      << ctx.getCOutPath(param) << R"(", "a");)" << endl;
  out << "\t\tfprintf(hlsv_of_" << param.parameterName
      << R"(, "[[transaction]] %d\n", hlsv_transaction_id);)" << endl;
  if (param.isPointer) {
    out << arrayPrintCode(param, "hlsv_of_");
  } else {
    if (param.isFloatType) {
      if (param.dtWidth == 32) {
        out << "\tfprintf(hlsv_of_" << param.parameterName
            << R"(, "0x%08x\n",  *((unsigned int *)&)" << param.parameterName
            << "));" << endl;

      } else {
        out << "\tfprintf(hlsv_of_" << param.parameterName
            << R"(, "0x%08llx\n",  *((unsigned long long *)&)"
            << param.parameterName << "));" << endl;
      }
    } else {
      out << "\tfprintf(hlsv_of_" << param.parameterName
          << R"(, "0x%08llx\n",  (long long))" << param.parameterName << ");"
          << endl;
    }
  }
  out << "\t\tfprintf(hlsv_of_" << param.parameterName
      << R"(, "[[/transaction]]\n");)" << endl;
  out << "\t\tfclose(hlsv_of_" << param.parameterName << ");" << endl << endl;
  return out.str();
}

string CInjector::getFileIoCodeForReturnValue(const CFunctionParameter &param,
                                              const string &actualReturnValue) {
  stringstream out;

  if (param.actualType == "void") {
    out << "\t\treturn;" << endl;
    return out.str();
  }

  out << "\t\thlsv_return = " << actualReturnValue << ";" << endl;

  out << "\t\thlsv_of_return = fopen(\"" << ctx.getCOutPath(param)
      << R"(", "a");)" << endl;
  out << "\t\tfprintf(hlsv_of_return, \"[[transaction]] %d\\n\", "
         "hlsv_transaction_id);"
      << endl;
  if (param.isFloatType) {
    if (param.dtWidth == 32) {
      out << "\t\tfprintf(hlsv_of_return, \"0x%08x\\n\",  *((unsigned int "
             "*)&hlsv_return));"
          << endl;

    } else {
      out << "\t\tfprintf(hlsv_of_return, \"0x%08llx\\n\",  *((unsigned long "
             "long*)&hlsv_return));"
          << endl;
    }
  } else {
    out << "\t\tfprintf(hlsv_of_return, \"0x%08llx\\n\", (long "
           "long)hlsv_return);"
        << endl;
  }
  out << "\t\tfprintf(hlsv_of_return, \"[[/transaction]]\\n\");" << endl;
  out << "\t\tfclose(hlsv_of_return);" << endl << endl;

  out << "\t\treturn hlsv_return;" << endl;
  return out.str();
}

string CInjector::getInjectedCFuv() { return injectedFuvSrc; }

} // namespace hls_verify
