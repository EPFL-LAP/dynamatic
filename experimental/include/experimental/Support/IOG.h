#pragma once

#include "dynamatic/Analysis/NameAnalysis.h"
#include "dynamatic/Support/LLVM.h"
#include "mlir/IR/Value.h"
#include <unordered_set>

// An In-order graph (IOG) of a dataflow circuit is a subgraph of the dataflow
// circuit. It contains one and only one entry channel (i.e. an argument of a
// FuncOp) which can reach all other operations using only channels part of the
// IOG. The IOG does not lose or gain any tokens: For any merge/mux, all the
// data inputs must be contained in the IOG, and for any fork, only a single
// output can be part of the IOG. Similarly, all branch outputs are part of the
// IOG, and for each join-operation, only a single input is part of the IOG.
// This way, there is a fixed number of tokens within the IOG.

namespace dynamatic {
struct IOG;
struct IOGPathSet;

struct IOGPathSet {
  IOGPathSet(const IOG &iog, Operation *start, Operation *end);

  std::unordered_set<Operation *> units;
  Operation *start;
  Operation *end;
};

struct IOG {
  IOG() = default;
  std::unordered_set<Operation *> units;
  llvm::DenseSet<mlir::Value> channels;
  mlir::Value entry;

  inline bool contains(Operation *op) const {
    auto iter = units.find(op);
    return iter != units.end();
  }

  inline bool contains(mlir::Value channel) const {
    auto iter = channels.find(channel);
    return iter != channels.end();
  }

  void debug() const {
    std::vector<mlir::Value> stack;
    llvm::DenseSet<mlir::Value> visited;
    stack.push_back(entry);
    visited.insert(entry);
    while (!stack.empty()) {
      mlir::Value channel = stack.back();
      stack.pop_back();
      Operation *prev = channel.getDefiningOp();
      if (prev == nullptr) {
        llvm::errs() << "entry";
      } else {
        llvm::errs() << getUniqueName(prev);
      }
      Operation *next = channel.getUses().begin()->getOwner();
      assert(next);
      assert(contains(next));
      llvm::errs() << " -> " << getUniqueName(next) << "\n";
      for (mlir::Value out : next->getResults()) {
        if (auto iter = visited.find(out); iter != visited.end())
          continue;
        if (contains(out)) {
          visited.insert(out);
          stack.push_back(out);
        }
      }
    }
  }
};

std::vector<IOG> findAllIOGs(ModuleOp modOp);

} // namespace dynamatic
