//===- SwitchingEstimation.cpp - Estimate Swithicng Activities ------*- C++ -*-===//
//
// Implements the switching estimation pass for all untis in the generated
// dataflow circuit
//
//===----------------------------------------------------------------------===//

#include "experimental/Transforms/Switching/SwitchingEstimation.h"
#include "experimental/Transforms/Switching/SwitchingSupport.h"
#include "experimental/Transforms/Switching/ProfilingAnalyzer.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Dialect/Handshake/HandshakeAttributes.h"
#include "dynamatic/Support/DynamaticPass.h"
#include "dynamatic/Support/LLVM.h"
#include "dynamatic/Support/Logging.h"
#include "dynamatic/Support/Backedge.h"
#include "dynamatic/Support/TimingModels.h"
#include "dynamatic/Support/Attribute.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/IR/Value.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Debug.h"

using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::handshake;
using namespace dynamatic::buffer;
using namespace dynamatic::experimental;
using namespace dynamatic::experimental::switching;

namespace {
// Define Switching Estimation pass driver
struct SwitchingEstimationPass
    : public dynamatic::experimental::switching::impl::SwitchingEstimationBase<
        SwitchingEstimationPass> {

  SwitchingEstimationPass(StringRef dataTrace,
                          StringRef bbList,
                          StringRef frequencies,
                          StringRef timingModels) {
    this->dataTrace = dataTrace.str();
    this->bbList = bbList.str();
    this->frequencies = frequencies.str();
    this->timingModels = timingModels.str();
  }

  // Main Interface
  void runDynamaticPass() override;

  // 
  //  Define global storing structure
  //
  SwitchingInfo switchInfo;

  // 
  //  Information Extraction Related Functions
  //
  // This function extract the backedge archs from all archs in the dataflow graph
  llvm::SmallVector<std::pair<unsigned, unsigned>> extractBackedges(llvm::SmallVector<experimental::ArchBB> archs);

  // This function extract all CFDFCs from the bbList attribute and the corresponding II from CFDFCThroughputAttr
  LogicalResult extractAllCFDFCs(mlir::ModuleOp& topModule);

  // Extract all op names of the alus in order
  void extractHandshakeOpNames(handshake::FuncOp& topFunc);


};
} // namespace 

void SwitchingEstimationPass::runDynamaticPass() {
  // Read Component Latencies from the provided database.
  TimingDatabase timingDB(&getContext());
  if (failed(TimingDatabase::readFromJSON(timingModels, timingDB)))
    signalPassFailure();

  // Step 0 : Build the top-level CFDFC storing structure (abstract)
  // Get MLIR funcop, for each example, only 1 funcop can exist
  mlir::ModuleOp topModule = getOperation();
  if (failed(extractAllCFDFCs(topModule))) {
    llvm::errs() << "[ERROR] Extraction of CFDFCs failed\n";
  }

  // Step 1: Parse the SCF level profiling results
  llvm::dbgs() << "[DEBUG] [Step 1] Parsing Profiling Results\n";
  llvm::dbgs() << "[DEBUG] \tBBList Log file: " << bbList << "\n";
  llvm::dbgs() << "[DEBUG] \tData Profiling Log file: " << dataTrace << "\n";
  SCFProfilingResult profilingResults(dataTrace, bbList, switchInfo);

  // Step 2: Build Adjacency graph for each CFDFC
  llvm::dbgs() << "[DEBUG] [Step 2] Build Adjacency Graph for Each Segment\n";
  
  for (const auto& [mgIndex, mgInstance]: switchInfo.cfdfcs) {
    llvm::dbgs() << "[DEBUG] \tMG : " << mgIndex << "\n";
    
    AdjGraph tmpAdjGraph(mgInstance, timingDB, switchInfo.cfdfcIIs[mgIndex]);
    switchInfo.segToAdjGraphMap.insert_or_assign(std::to_string(mgIndex), &tmpAdjGraph);
  }
}


//===----------------------------------------------------------------------===//
//
// Internal Function Definitions
//
//===----------------------------------------------------------------------===//

void SwitchingEstimationPass::extractHandshakeOpNames(handshake::FuncOp& topFunc) {
  for (Operation& op : topFunc.getOps()) {
    // Get the handshake.name attribute
    // TODO: Make the retrieving of name attribute more natural
    std::string opName = op.getAttrOfType<StringAttr>("handshake.name").str();
    std::string opType = removeDigits(opName);

    if (NAME_SENSE_LIST.find(opType) != NAME_SENSE_LIST.end()) {
      switchInfo.funcOpNames.push_back(opName);
    }
  }
}

llvm::SmallVector<std::pair<unsigned, unsigned>> SwitchingEstimationPass::extractBackedges(llvm::SmallVector<experimental::ArchBB> archs) {
  llvm::SmallVector<std::pair<unsigned, unsigned>> backEdgeList;

  // Iterate through the archs list
  for (auto &selArch: archs) {
    if (selArch.isBackEdge) {
      backEdgeList.push_back(std::make_pair(selArch.srcBB, selArch.dstBB));
    }
  }

  return backEdgeList;
}

LogicalResult SwitchingEstimationPass::extractAllCFDFCs(mlir::ModuleOp& topModule) {
  for (handshake::FuncOp funcOp : topModule.getOps<handshake::FuncOp>()) {
    llvm::dbgs() << "[DEBUG] Entering Func: " << funcOp.getName() << "\n";

    // Get all ALU names in the selected FuncOp
    extractHandshakeOpNames(funcOp);

    // Get the CFDFC throughput and bb list from attributes
    llvm::dbgs() << "[DEBUG] [Step 0] Extracting CFDFCs\n";
    llvm::dbgs() << "[DEBUG] \tParsing CSV file: " << frequencies << "\n";

    llvm::SmallVector<experimental::ArchBB> archs; // Store all archs from the csv file

    // Read the CSV containing arch information (number of transitions between
    // pairs of basic blocks) from disk.
    if (failed(StdProfiler::readCSV(frequencies, archs))) {
      llvm::errs() << "[ERROR] Failed to read frequency profiling information from CSV\n";
      exit(-1);
    }

    // Get all Backedges
    switchInfo.backEdges = extractBackedges(archs);

    // Extract all needed attributes
    DictionaryAttr throughputAttr = getUniqueAttr<handshake::CFDFCThroughputAttr>(funcOp).getThroughputMap();
    DictionaryAttr bbListAttr = getUniqueAttr<handshake::CFDFCToBBListAttr>(funcOp).getCfdfcMap();

    // CONSTRUCT CFDFCS FROM THE BBLIST ATTRIBUTES
    for (auto &cfdfcBBListPair: bbListAttr) {
      llvm::SmallVector<experimental::ArchBB> tmpArchs;
      std::string cfdfcIndex = cfdfcBBListPair.getName().str();

      // Get the corresponding II
      float_t cfdfcII = 0;
      auto IIValueAttr = throughputAttr.get(cfdfcBBListPair.getName());
      if (IIValueAttr) {
        // Attribtue was successfully retrieved
        if (auto IIValue = IIValueAttr.dyn_cast<mlir::FloatAttr>()) {
          cfdfcII = 1.0 / IIValue.getValueAsDouble();

          switchInfo.cfdfcIIs[std::stoul(cfdfcIndex)] = cfdfcII;
        }
      }

      llvm::dbgs() << "[DEBUG] \t[CFDFC] " << cfdfcIndex << "\n";

      mlir::ArrayAttr bbList = llvm::dyn_cast<mlir::ArrayAttr>(cfdfcBBListPair.getValue());
      auto iter = bbList.begin();
      unsigned prevBBId = (*iter++).cast<IntegerAttr>().getUInt();
      unsigned curBBId = prevBBId;
      unsigned startBBId = prevBBId;

      // If more than 1 BB in the CFDFC
      if (bbList.size() > 1) {
        for (; iter != bbList.end(); iter++) {
          curBBId = (*iter).cast<IntegerAttr>().getUInt();

          // Check whether this is a backedge
          if (std::find(switchInfo.backEdges.begin(), switchInfo.backEdges.end(), std::make_pair(prevBBId, curBBId)) != switchInfo.backEdges.end()) {
            tmpArchs.push_back(experimental::ArchBB(prevBBId, curBBId, 0, true));

            switchInfo.insertBE(prevBBId, curBBId, cfdfcIndex);

            // Debug
            llvm::dbgs() << "[DEBUG] \t[Backedge] Arch: (" << prevBBId << ", " << curBBId << ") Found in CFDFC: " << cfdfcIndex << "\n";
          } else {
            tmpArchs.push_back(experimental::ArchBB(prevBBId, curBBId, 0, false));

            // Debug
            llvm::dbgs() << "[DEBUG] \t[Edge] Arch: (" << prevBBId << ", " << curBBId << ") Found in CFDFC: " << cfdfcIndex << "\n";
          }

          prevBBId = curBBId;
        }

        // Check the loop back edge
        if (std::find(switchInfo.backEdges.begin(), switchInfo.backEdges.end(), std::make_pair(prevBBId, startBBId)) != switchInfo.backEdges.end()) {
          tmpArchs.push_back(experimental::ArchBB(prevBBId, startBBId, 0, true));

          switchInfo.insertBE(prevBBId, startBBId, cfdfcIndex);

          // Debug
          llvm::dbgs() << "[DEBUG] \t[Backedge] Arch: (" << prevBBId << ", " << startBBId << ") Found in CFDFC: " << cfdfcIndex << "\n";
        } else {
          tmpArchs.push_back(experimental::ArchBB(prevBBId, startBBId, 0, false));

          // Debug
          llvm::dbgs() << "[DEBUG] \t[Edge] Arch: (" << prevBBId << ", " << startBBId << ") Found in CFDFC: " << cfdfcIndex << "\n";
        }
      } else {
        if (std::find(switchInfo.backEdges.begin(), switchInfo.backEdges.end(), std::make_pair(prevBBId, curBBId)) != switchInfo.backEdges.end()) {
          tmpArchs.push_back(experimental::ArchBB(prevBBId, curBBId, 0, true));

          switchInfo.insertBE(prevBBId, curBBId, cfdfcIndex);

          // Debug
          llvm::dbgs() << "[DEBUG] \t[Backedge] Arch: (" << prevBBId << ", " << curBBId << ") Found in CFDFC: " << cfdfcIndex << "\n";
        } else {
          tmpArchs.push_back(experimental::ArchBB(prevBBId, curBBId, 0, false));

          // Debug
          llvm::dbgs() << "[DEBUG] \t[Edge] Arch: (" << prevBBId << ", " << curBBId << ") Found in CFDFC: " << cfdfcIndex << "\n";
        }
      }
      
      // Construct the archset
      buffer::ArchSet archSet;
      for (auto &arch: tmpArchs) {
        archSet.insert(&arch);
      }

      // Since we don't care about the execution number of each CFDFC, it's set to 0
      buffer::CFDFC tmpMG(funcOp, archSet, 0);
      switchInfo.cfdfcs.insert_or_assign(std::stoul(cfdfcIndex), tmpMG);

      // Insert to the segment map as well
      std::vector<unsigned> newVector(tmpMG.cycle.begin(), tmpMG.cycle.end());
      switchInfo.segToBBListMap[cfdfcIndex] = newVector;
    }
  }
  
  return success();
}


namespace dynamatic {
namespace experimental {
namespace switching {

// Return a unique pointer for the switching estimation pass
std::unique_ptr<dynamatic::DynamaticPass>
createSwitchingEstimation(StringRef dataTrace, StringRef bbList, StringRef frequencies, StringRef timingModels) {
  return std::make_unique<SwitchingEstimationPass>(dataTrace, bbList, frequencies, timingModels);
}

} // namespace switching
} // namespace experimental
} // namespace dynamatic
