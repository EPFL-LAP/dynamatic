//===- SCC.h - Strongly Connected Components ------------------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//     This file contains an algorithm that takes a graph with n nodes
//            and returns the Strongly connected components in this graph
//     The algorithm works well with low edge to node ratio
//            If this is not the case one might consider usng an other algorithm
//     Implementation of Kosaraju's algorithm
//     Explanatory video: https://www.youtube.com/watch?v=Qdh6-a_2MxE&t=328s
//===----------------------------------------------------------------------===//
#ifndef EXPERIMENTAL_INCLUDE_DYNAMATIC_TRANSFORMS_RESOURCESHARING_SCC_H
#define EXPERIMENTAL_INCLUDE_DYNAMATIC_TRANSFORMS_RESOURCESHARING_SCC_H

#include <set>
#include <list>
#include <stack>
#include <vector>
#include <algorithm>

#include "mlir/IR/Operation.h"
#include "llvm/ADT/SmallVector.h"
#include "dynamatic/Support/CFG.h"
#include "experimental/Support/StdProfiler.h"

namespace dynamatic {
namespace experimental {
namespace sharing {

// kosarajus algorithm performed on basic block level
std::vector<int> Kosarajus_algorithm_BBL(SmallVector<ArchBB> archs);

// different implementation: performed on operation level
void Kosarajus_algorithm_OPL(mlir::Operation* startOp, handshake::FuncOp *funcOp, std::set<mlir::Operation*>& result);

} // namespace sharing
} // namespace experimental
} // namespace dynamatic

#endif // EXPERIMENTAL_INCLUDE_DYNAMATIC_TRANSFORMS_RESOURCESHARING_SCC_H
