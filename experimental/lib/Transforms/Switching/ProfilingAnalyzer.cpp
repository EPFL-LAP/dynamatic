//===- SwitchingSupport.cpp - Estimate Swithicng Activities ------*- C++ -*-===//
//
// Implements the Analyzer Class for the functional profiling results
//
//===----------------------------------------------------------------------===//

#include "experimental/Transforms/Switching/ProfilingAnalyzer.h"

using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::handshake;

// Constructor for the SCF parsing class
SCFProfilingResult::SCFProfilingResult(StringRef dataTrace, StringRef bbList, SwitchingInfo& switchInfo) {
  // Step 0: Get the directory path
  // TODO: Need to add this to the pass interface, instead of hardcoding the file name here
  std::filesystem::path pathObj(bbList.str());
  std::string resultDir = pathObj.parent_path().string();
  std::string scfFilePath = resultDir + "/cf_dyn_transformed.mlir";
  llvm::dbgs() << "[DEBUG] \tResult Dir : " << resultDir << "\n";

  // Step 0: Parse the SCF level mlir file to get the op name vector
  SCFFile scfFile(scfFilePath);
  buildScfToHSMap(switchInfo, scfFile);

  // Step 1: Parse the BBlist file and reconstruct the execution BBlist
  parseBBListFile(bbList, switchInfo);

  // Step 2: Parse the actual data log file
  parseDataLogFile(dataTrace, switchInfo);

  // Step 3: Construct the map for seg execution count
  constructSegExeCount();

  //! Testing, 01/09/2024
  // for (const auto& [key, value]: execPhaseToSegExecNumMap) {
  //   llvm::dbgs() << "[DEBUG] Op Name: " << key << "\n";

  //   llvm::dbgs() << "[DEBUG] \tValue: " << value.first << ", Iter Index: " << value.second << "\n";
    
  // }
}

void SCFProfilingResult::constructSegExeCount() {
  unsigned segCounter = 0;
  unsigned numExecPhase = 0;
  std::string prevSeg = "S";

  for (const auto& selSeg: executedSegTrace) {
    if (selSeg != prevSeg) {
      execPhaseToSegExecNumMap[numExecPhase] = std::make_pair(prevSeg, segCounter);
      prevSeg = selSeg;
      numExecPhase++;
      segCounter = 1;
    } else {
      segCounter++;
    }
  }

  // Add the ending section
  execPhaseToSegExecNumMap[numExecPhase] = std::make_pair("E", 1);
}

void SCFProfilingResult::insertValuePair(int opValue, unsigned iterIndex, std::string opName) {
  // Check wheter the key exist in the map or not
  if (opNameToValueListMap.find(opName) != opNameToValueListMap.end()) {
    opNameToValueListMap[opName].push_back(std::make_pair(opValue, iterIndex));
  } else {
    std::vector<std::pair<int, unsigned>> newVector = {std::make_pair(opValue, iterIndex)};
    opNameToValueListMap[opName] = newVector;
  }
}

void SCFProfilingResult::parseBBListFile(StringRef bbList, SwitchingInfo& switchInfo) {
  llvm::dbgs() << "[DEBUG] \t[PARSING BBLIST LOG FILE]\n";

  // STEP 0 : Variable initialization
  std::vector<unsigned> tmpMGTrace;
  std::vector<unsigned> tmpBBTrace; // List of traversed BB (temporary structure)
  int numTransSections = 0;
  executedBBTrace.push_back(0);    // Always start with BB 0
  tmpBBTrace.push_back(0);

  // Record all transaction section
  std::vector<std::vector<unsigned>> transactionBBLists;

  // Read the BB list log file
  std::ifstream file(bbList.str());
  if (!file.is_open()) {
    llvm::errs() << "[ERROR] Can't Open file " << bbList << "\n";
  }

  // STEP 1: Traverse the BB list log
  std::string line;
  while (std::getline(file, line)) {
    // Remove unnecessary information
    line = strip(line, "");
    auto lineSplit = split(line, " ");

    // Clean the string
    std::string BBTupleStr = strip(lineSplit.back(), "(");
    BBTupleStr = strip(BBTupleStr, ")");

    // If edge encountered
    if (lineSplit[0] == "[Edge]") {
      auto edgeTuple = split(BBTupleStr, ",");

      executedBBTrace.push_back(std::stoul(edgeTuple[1]));
      tmpBBTrace.push_back(std::stoul(edgeTuple[1]));
    } else if (lineSplit[0] == "[BEdge]") {
      auto backEdgeTuple = split(BBTupleStr, ",");
      executedBBTrace.push_back(std::stoul(backEdgeTuple[1]));

      // Get the executed MG
      unsigned srcBB = std::stoul(backEdgeTuple[0]);
      unsigned dstBB = std::stoul(backEdgeTuple[1]);
      auto CFDFCVec = switchInfo.backEdgeToCFDFCMap[std::make_pair(srcBB, dstBB)];
      unsigned travMG = 0;

      if (CFDFCVec.size() > 1) {
        // Multiple CFDFC cancidates
        bool matchedFlag = false;
        unsigned selMGBBNumber = 0;
        unsigned selMGIndex = 0;

        for (const auto& selMG: CFDFCVec) {
          // Get the BBlist vector
          // We choose the one with the most number of bbs matched with the selBBList
          std::vector<unsigned> selBBList = switchInfo.segToBBListMap[std::to_string(selMG)];
          // Create sets from traversed BB and the potential MG
          std::set<unsigned> selMGBBSet(selBBList.begin(), selBBList.end());
          std::set<unsigned> compareSet(tmpBBTrace.begin(), tmpBBTrace.end());

          // TODO: Verify the comparison below
          if (std::includes(compareSet.begin(), compareSet.end(), selMGBBSet.begin(), selMGBBSet.end())) {
            matchedFlag = true;
            if (selMGBBSet.size() > selMGBBNumber) {
              selMGIndex = selMG;
              selMGBBNumber = selMGBBSet.size();
            }
          }
        }

        if (matchedFlag) {
          travMG = selMGIndex;
        } else {
          llvm::errs() << "[ERROR] Cant's match the backedge (" << srcBB << ", " << dstBB << ") to the corresponding MG.\n" ;
        }
      } else {
        travMG = CFDFCVec[0];
      }

      tmpMGTrace.push_back(travMG);

      // Update the tmpBBTrace
      std::vector<unsigned> newVector = {dstBB};
      tmpBBTrace = std::move(newVector);
    }
  }

  file.close();

  // STEP 2: Partition the executedBB Trace into different sections
  // Identify all segments in the execution trace
  // Including both MGs and Transitions sections, which will not be repeatedly executed
  unsigned curIter = 0;
  unsigned tracePointer = 0;

  for (const auto& selMG : tmpMGTrace) {
    auto curCFDFCBBList = switchInfo.segToBBListMap[std::to_string(selMG)];

    if (curIter == 0) {
      std::vector<unsigned> tmpStartBBList;

      while (true) {
        if (std::find(curCFDFCBBList.begin(), curCFDFCBBList.end(), executedBBTrace[tracePointer]) == curCFDFCBBList.end()) {
          tmpStartBBList.push_back(executedBBTrace[tracePointer]);
          tracePointer++;
        } else {
          break;
        }
      }

      // Update the CFDFC BB storing Dict
      switchInfo.segToBBListMap["S"] = tmpStartBBList;

      // Store the end of the initialization
      executedSegTrace.push_back("S");

      // Update the iteration end stataus
      iterEndIndex.push_back(tracePointer - 1);

      curIter++;

      // First Marked Graph entered, update the trace pointer
      tracePointer += curCFDFCBBList.size();
      iterEndIndex.push_back(tracePointer - 1);

      executedSegTrace.push_back(std::to_string(selMG));

      // Update the curIter
      curIter++;
    } else {
      // Check whether a transation section is encountered
      if (std::find(curCFDFCBBList.begin(), curCFDFCBBList.end(), executedBBTrace[tracePointer]) == curCFDFCBBList.end()) {
        // Transaction section encountered
        std::vector<unsigned> tmpTransBBList;

        while (true) {
          if (std::find(curCFDFCBBList.begin(), curCFDFCBBList.end(), executedBBTrace[tracePointer]) == curCFDFCBBList.end()) {
            tmpTransBBList.push_back(executedBBTrace[tracePointer]);
            tracePointer++;
          } else break;
        }

        // Add the transaction section to the execution trace and bblist mapping dict
        bool presentFlag = false;

        for (int i = 0; i < transactionBBLists.size(); i++) {
          if (tmpTransBBList == transactionBBLists[i]) {
            // We have stored the transaction section
            std::string sectionName = "T" + std::to_string(i);
            executedSegTrace.push_back(sectionName);

            presentFlag = true;
            break;
          }
        }

        // Update iter end status
        iterEndIndex.push_back(tracePointer - 1);
        curIter++;

        // If this is a new transaction section
        if (!presentFlag) {
          std::string sectionName = "T" + std::to_string(numTransSections);

          // Update the storing structure
          transactionBBLists.push_back(tmpTransBBList);
          executedSegTrace.push_back(sectionName);
          switchInfo.segToBBListMap[sectionName] = tmpTransBBList;

          numTransSections++;

          // Update the successing list
          switchInfo.transToSucMGMap[sectionName] = std::to_string(selMG);
        }
      }

      tracePointer += curCFDFCBBList.size();
      iterEndIndex.push_back(tracePointer - 1);
      executedSegTrace.push_back(std::to_string(selMG));
      curIter++;
    }
  }

  // Updatge the ending BB list
  std::vector<unsigned> tmpEndBBList;
  for (int i = 0; i < executedBBTrace.size() - tracePointer; i++) {
    tmpEndBBList.push_back(executedBBTrace[tracePointer + i]);
  }

  // Store the ending section
  executedSegTrace.push_back("E");
  switchInfo.segToBBListMap["E"] = tmpEndBBList;

  // Construct the BB iter map
  unsigned iterMapCounter = 0;
  unsigned iterCounter = 0;
  for (const auto& selSeg: executedSegTrace) {
    for (int i = 0; i < switchInfo.segToBBListMap[selSeg].size(); i++) {
      bbToIterMap[iterMapCounter] = iterCounter;
    }
    iterCounter++;
  }
}

void SCFProfilingResult::parseDataLogFile(StringRef dataTrace, SwitchingInfo& switchInfo) {
  llvm::dbgs() << "[DEBUG] \t[PARSING DATA LOG FILE]\n";

  // STEP 0: Variable initialization
  unsigned curIter = 0;
  
  // Read the Data log file
  std::ifstream file(dataTrace.str());
  if (!file.is_open()) {
    llvm::errs() << "[ERROR] Can't Open file " << dataTrace << "\n";
  }

  // STEP 1: Traverse the Data log
  std::string line;
  while (std::getline(file, line)) {
    // Remove unnecessary information
    line = strip(line, "");
    auto lineSplit = split(line, " ");

    if (lineSplit[0] == "[DATA]") {
      // Get the (op_name, value) tuple
      std::string opValueStr = strip(lineSplit[1], "(");
      opValueStr = strip(opValueStr, ")");

      auto opValueTuple = split(opValueStr, ",");
      std::string scfOPName = strip(opValueTuple[0], "\"");

      std::string opName = scfToHandshakeNameMap[scfOPName];

      // TODO: Remove the [ARG] in the data profiler
      if (lineSplit.size() > 1) {
        int opValue = std::stoi(opValueTuple.back());
        insertValuePair(opValue, curIter, opName);
      } else {
        continue;
      }
    } else if (lineSplit[0] == "[Edge]") {
      unsigned edgeIndex = std::stoul(lineSplit[1]);

      if (std::find(iterEndIndex.begin(), iterEndIndex.end(), edgeIndex) != iterEndIndex.end()) {
        curIter++;
      }
    } else if (lineSplit[0] == "[BEdge]") {
      unsigned edgeIndex = std::stoul(lineSplit[1]);

      if (std::find(iterEndIndex.begin(), iterEndIndex.end(), edgeIndex) != iterEndIndex.end()) {
        curIter++;
      }
    }
  }

  // Close the file
  file.close();
} 

void SCFProfilingResult::buildScfToHSMap(SwitchingInfo& switchInfo, SCFFile& scfFile) {
  for (int i = 0; i < switchInfo.funcOpNames.size(); i++) {
    std::string scfName = scfFile.opNameList[i];
    std::string hsName = switchInfo.funcOpNames[i];

    scfToHandshakeNameMap[scfName] = hsName;

    // llvm::dbgs() << "[DEBUG] \t SCF OP Name : " << scfName << "; Handshake Name: " << hsName << "\n";
  }
}

//===----------------------------------------------------------------------===//
//
// Helper Class
//
//===----------------------------------------------------------------------===//
SCFFile::SCFFile(StringRef scfFile) {
  // Define the matching pattern
  std::regex nonNameExpr("[^a-zA-Z_0-9]+");
  std::ifstream file(scfFile.str());

  if (!file.is_open()) {
    llvm::errs() << "[ERROR] Can't Open file " << scfFile << "\n";
    return;
  }

  // Get the operation name
  std::string line;
  while (std::getline(file, line)) {
    line = strip(line, "");
    auto lineSplit = split(line, " ");

    if ((lineSplit[0].find("%") != std::string::npos) || (lineSplit[0].find("memref") != std::string::npos)) {
      std::regex handshakeExpr("#handshake\\.name.*");
      std::vector<std::string>::iterator iterBegin = lineSplit.begin();
      std::vector<std::string>::iterator iterEnd = lineSplit.end();

      // Iterate through the found line
      for (std::vector<std::string>::iterator vecIter = iterBegin; vecIter != iterEnd; vecIter++) {
        if (*vecIter == "{handshake.name" || *vecIter == "handshake.name") {
          // Handshake name matched
          // TODO: Change the name analysis process to make sure the name of the ALU is not changed
          vecIter += 2;

          // If not constant
          // Remove unwanted characters
          std::string tmpName = std::regex_replace(*vecIter, nonNameExpr, "");
        
          if (tmpName.find("constant") == std::string::npos) {
            opNameList.push_back(tmpName);
          }
        }
      }
    }
  }
  // Close the file
  file.close();
}
