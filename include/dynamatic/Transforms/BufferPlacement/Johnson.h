#include "dynamatic/Analysis/NameAnalysis.h"
#include "dynamatic/Dialect/Handshake/HandshakeAttributes.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Dialect/Handshake/HandshakeTypes.h"
#include "dynamatic/Support/Attribute.h"
#include "dynamatic/Support/Logging.h"
#include "mlir/IR/OperationSupport.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Casting.h"
#include <string>

#ifndef DYNAMATIC_GUROBI_NOT_INSTALLED
#include "gurobi_c++.h"

using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::handshake;
using Cycle = std::vector<Operation *>;
using CycleList = std::vector<Cycle>;
using ChannelCycle = std::vector<Value>;
using ChannelCycleList = std::vector<ChannelCycle>;

class GraphForJohnson {

  std::vector<Operation *> vertices;
  llvm::DenseMap<Operation *, llvm::DenseMap<Operation *, Value>> adjacency;
  std::vector<Cycle> cycles;
  llvm::StringSet<> seenCycleHashes;
  std::vector<Operation *> stack;
  llvm::DenseMap<Operation *, bool> blocked;
  llvm::DenseMap<Operation *, llvm::SmallPtrSet<Operation *, 4>> blockedMap;

public:
  GraphForJohnson(FuncOp funcOp);

  ChannelCycleList findAllChannelCycles();

private:
  void clearBlocked();
  void unblock(Operation *op);
  bool findCyclesFromStart(Operation *currentOp, Operation *start);
  CycleList findAllCycles();
  void addCycleToCycleList(const Cycle &cycle);

  static Cycle normalizeCycle(const Cycle &cycle);
  static std::string hashCycle(const Cycle &cycle);
};

#endif // DYNAMATIC_GUROBI_NOT_INSTALLED