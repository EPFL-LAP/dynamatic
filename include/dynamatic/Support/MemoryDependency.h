#ifndef DYNAMATIC_MEMORYDEPENDENCY_H
#define DYNAMATIC_MEMORYDEPENDENCY_H

#include "dynamatic/Dialect/Handshake/HandshakeAttributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Metadata.h"
#include <optional>
#include <stdlib.h>
#include <string>
#include <vector>

#include "dynamatic/Analysis/NameAnalysis.h"

const std::string METADATA_DEPENDENCY = "dep.op";

/// \brief: This is a helper construct that handles the translation between the
/// dependency analysis result from LLVM and MLIR IR. It implements
/// - Serialize to LLVM metadata nodes;
/// - Unseralize from metadatanodes to construct;
/// - Convert the memory dependency attributes.
struct LLVMMemDependency {

  // Handshake name of the source operation/instruction
  std::string name;

  // A list of dest (destination op of RAW and WAW dependencies) and the loop
  // depth of the dependency (TODO: why do we care about the loop depth?).
  std::vector<std::pair<std::string, unsigned>> destAndDepth;

  // Convert the stored memory dependency values into a list of memory
  // dependence attributes
  llvm::SmallVector<dynamatic::handshake::MemDependenceAttr>
  getMemoryDependenceAttrs(mlir::MLIRContext &ctx) {

    llvm::SmallVector<dynamatic::handshake::MemDependenceAttr> attrs;

    for (const auto &[dstName, depth] : destAndDepth) {

      auto dstNameAttr = mlir::StringAttr::get(&ctx, dstName);
      auto attr = dynamatic::handshake::MemDependenceAttr::get(
          &ctx, dstNameAttr, depth);
      attrs.push_back(attr);
    }

    return attrs;
  }

  /// \brief: Serializes the contained memory dependency data and store it in a
  /// given LLVM instruction. The memory dependency meta data is formatted in
  /// the following way:
  ///
  /// - A metadata node with key METADATA_DEPENDENCY.
  /// - Inside the metadata node above, there is a list of tuples, each has
  /// (memoryOpName, loopDepth)
  ///
  /// Using json notation, an example of such a meta data node would be:
  /// {
  ///   METADATA_DEPENDENCY : [
  ///     ("load1", 1),
  ///     ("store1", 1),
  ///     ("store2", 1)
  ///   ]
  /// }
  void toLLVMMetaDataNode(llvm::LLVMContext &ctx, llvm::Instruction *inst) {
    llvm::SmallVector<llvm::Metadata *, 10> mdVals;
    for (const auto &[dstName, depth] : this->destAndDepth) {
      mdVals.push_back(llvm::MDNode::get(
          ctx, {llvm::MDString::get(ctx, dstName),
                llvm::MDString::get(ctx, std::to_string(depth))}));
    }
    inst->setMetadata(METADATA_DEPENDENCY,
                      llvm::MDNode::get(ctx, llvm::ArrayRef(mdVals)));
  }

  /// \brief: This factory function attempts to construct a *this from the meta
  /// data available from a given instruction. This function assumes that:
  /// - The instruction has a metadata node called !handshake.name
  /// - The instruction has a metadata node called !METADATA_DEPENDENCY
  static std::optional<LLVMMemDependency>
  fromLLVMInstruction(llvm::Instruction *inst) {

    if (!llvm::isa<llvm::LoadInst, llvm::StoreInst>(inst)) {
      // We can not possibly get LLVMMemDependency from ops other than loads and
      // stores.
      return std::nullopt;
    }

    LLVMMemDependency depData;

    auto *nameMetaData = inst->getMetadata(dynamatic::NameAnalysis::ATTR_NAME);

    if (!nameMetaData /* the operation is not named in the MemDepAnalysis pass */) {
      return std::nullopt;
    }

    // This metadata node has name METADATA_NAME, it contains a list of tuples:
    // {!depTupleNode1, !depTupleNode2, ...}
    llvm::MDString *data =
        llvm::dyn_cast<llvm::MDString>(nameMetaData->getOperand(0));

    // This checks if the metadata node is indeed called METADATA_DEPENDENCY
    depData.name = data->getString().str();
    auto *depsMetaDataNode = inst->getMetadata(METADATA_DEPENDENCY);
    if (!depsMetaDataNode) {
      return depData;
    }

    for (unsigned i = 0; i < depsMetaDataNode->getNumOperands(); i++) {
      llvm::MDNode *dep =
          llvm::dyn_cast<llvm::MDNode>(depsMetaDataNode->getOperand(i));
      assert(dep->getNumOperands() == 2 &&
             "Malformed dependency metadata! It must be a destination name and "
             "a depth!");

      auto *mdDstName = llvm::dyn_cast<llvm::MDString>(dep->getOperand(0));
      assert(mdDstName &&
             "Malformed IR metadata! The first element must be a string.");
      auto *mdDepth = llvm::dyn_cast<llvm::MDString>(dep->getOperand(1));
      assert(mdDepth);
      assert(mdDstName &&
             "Malformed IR metadata! The second element must be a string.");

      depData.destAndDepth.emplace_back(mdDstName->getString().str(),
                                        std::stoi(mdDepth->getString().str()));
    }

    return depData;
  }
};

#endif // DYNAMATIC_MEMORYDEPENDENCY_H