//===- StringUtils.cpp - Utilities to manipulate strings --------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "StringUtils.h"
#include "DOTParser.h"
#include "VHDLWriter.h"
#include <algorithm>
#include <bitset>
#include <cctype>
#include <fstream>
#include <iostream>
#include <list>
#include <sstream>
#include <stdexcept>
#include <stdlib.h>
#include <string>
#include <vector>

using namespace std;

static void eraseAllSubStr(std::string &mainStr, const std::string &toErase) {
  size_t pos = std::string::npos;

  // Search for the substring in string in a loop untill nothing is found
  while ((pos = mainStr.find(toErase)) != std::string::npos) {
    // If found then erase it from string
    mainStr.erase(pos, toErase.length());
  }
}

void stringSplit(const string &s, char c, vector<string> &v) {
  string::size_type i = 0;
  string::size_type j = s.find(c);

  while (j != string::npos) {
    v.push_back(s.substr(i, j - i));
    i = ++j;
    j = s.find(c, j);

    if (j == string::npos)
      v.push_back(s.substr(i, s.length()));
  }
}

string stringRemoveBlank(string stringInput) {
  stringInput.erase(remove(stringInput.begin(), stringInput.end(), ' '),
                    stringInput.end());
  return stringInput;
}

string stringClean(string stringInput) {
  stringInput.erase(remove(stringInput.begin(), stringInput.end(), '\t'),
                    stringInput.end());

  stringInput.erase(remove(stringInput.begin(), stringInput.end(), ' '),
                    stringInput.end());
  stringInput.erase(remove(stringInput.begin(), stringInput.end(), '"'),
                    stringInput.end());
  stringInput.erase(remove(stringInput.begin(), stringInput.end(), ']'),
                    stringInput.end());
  stringInput.erase(remove(stringInput.begin(), stringInput.end(), ';'),
                    stringInput.end());

  return stringInput;
}

string stripExtension(string stringInput, const string &extension) {
  size_t dot = stringInput.rfind(extension);
  if (dot != std::string::npos)
    stringInput.resize(dot);
  return stringInput;
}

string stringConstant(unsigned long int value, int size) {
  stringstream ss;
  switch (size) {
  case 1:
    ss << std::bitset<1>(value);
    break;
  case 2:
    ss << std::bitset<2>(value);
    break;
  case 3:
    ss << std::bitset<3>(value);
    break;
  case 4:
    ss << std::bitset<4>(value);
    break;
  case 5:
    ss << std::bitset<5>(value);
    break;
  case 6:
    ss << std::bitset<6>(value);
    break;
  case 7:
    ss << std::bitset<7>(value);
    break;
  case 8:
    ss << std::bitset<8>(value);
    break;
  case 9:
    ss << std::bitset<9>(value);
    break;
  case 10:
    ss << std::bitset<10>(value);
    break;
  case 11:
    ss << std::bitset<11>(value);
    break;
  case 12:
    ss << std::bitset<12>(value);
    break;
  case 13:
    ss << std::bitset<13>(value);
    break;
  case 14:
    ss << std::bitset<14>(value);
    break;
  case 15:
    ss << std::bitset<15>(value);
    break;
  case 16:
    ss << std::bitset<16>(value);
    break;
  case 17:
    ss << std::bitset<17>(value);
    break;
  case 18:
    ss << std::bitset<18>(value);
    break;
  case 19:
    ss << std::bitset<19>(value);
    break;
  case 20:
    ss << std::bitset<20>(value);
    break;
  case 21:
    ss << std::bitset<21>(value);
    break;
  case 22:
    ss << std::bitset<22>(value);
    break;
  case 23:
    ss << std::bitset<23>(value);
    break;
  case 24:
    ss << std::bitset<24>(value);
    break;
  case 25:
    ss << std::bitset<25>(value);
    break;
  case 26:
    ss << std::bitset<26>(value);
    break;
  case 27:
    ss << std::bitset<27>(value);
    break;
  case 28:
    ss << std::bitset<28>(value);
    break;
  case 29:
    ss << std::bitset<29>(value);
    break;
  case 30:
    ss << std::bitset<30>(value);
    break;
  case 31:
    ss << std::bitset<31>(value);
    break;
  case 32:
    ss << std::bitset<32>(value);
    break;
  default:
    if ((value > 0xFFFFFFFF) && (size > 32)) {
      ss << std::bitset<64>(value);
    } else {
      ss << std::bitset<32>(value);
    }
    break;
  }
  return ss.str();
}

string cleanEntity(const string &filename) {
  vector<string> v;
  string returnString;
  stringSplit(filename, '/', v);

  if (!v.empty()) {
    for (const auto &indx : v)
      returnString = indx;
  } else {
    returnString = filename;
  }

  eraseAllSubStr(returnString, "_elaborated");
  eraseAllSubStr(returnString, "_optimized");
  eraseAllSubStr(returnString, "_area");

  return returnString;
}
