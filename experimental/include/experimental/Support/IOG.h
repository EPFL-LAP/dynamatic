#pragma once

#include "dynamatic/Analysis/NameAnalysis.h"
#include "dynamatic/Support/LLVM.h"
#include "mlir/IR/Value.h"

#include <unordered_set>

namespace dynamatic {
struct IOG;
struct IOGPath;

struct IOGPath {
  std::unordered_map<Operation *, mlir::Value> prevSet;
  std::unordered_map<Operation *, mlir::Value> forwardSet;
  Operation *from;
  Operation *to;
  IOGPath(const IOG &iog, Operation *from, Operation *to);

  void computeBackPath(const IOG &iog);
  void computeForwardPathFromBackPath();

  inline bool exists() const { return prevSet.find(to) != prevSet.end(); }
  mlir::Value stepBack(Operation *cur) const {
    auto iter = prevSet.find(cur);
    assert(iter != prevSet.end());
    return iter->second;
  }
  mlir::Value stepForward(Operation *cur) const {
    auto iter = forwardSet.find(cur);
    assert(iter != forwardSet.end());
    return iter->second;
  }
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
