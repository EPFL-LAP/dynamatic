//===- BackwardUpdate.cpp ---------*- C++ -*-===//
//
// This file declares functions for backward pass in --optimize-bits.
//
//===----------------------------------------------------------------------===//
#include "dynamatic/Transforms/BackwardUpdate.h"
#include "dynamatic/Transforms/PassDetails.h"
#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

namespace backward {

  void constructFuncMap(DenseMap<StringRef, 
                        std::function<unsigned (Operation::result_range vecResults)>> 
                        &mapOpNameWidth)
  {
    mapOpNameWidth[StringRef("arith.addi")] = [](Operation::result_range vecResults)
    {
      return std::min(cpp_max_width, 
                      vecResults[0].getType().getIntOrFloatBitWidth());
    };

    mapOpNameWidth[StringRef("arith.subi")] = mapOpNameWidth[StringRef("arith.addi")];

    mapOpNameWidth[StringRef("arith.muli")] = mapOpNameWidth[StringRef("arith.addi")];

    mapOpNameWidth[StringRef("arith.andi")] = mapOpNameWidth[StringRef("arith.addi")];

    mapOpNameWidth[StringRef("arith.ori")] = mapOpNameWidth[StringRef("arith.addi")];

    mapOpNameWidth[StringRef("arith.xori")] = mapOpNameWidth[StringRef("arith.addi")];


  }

}
