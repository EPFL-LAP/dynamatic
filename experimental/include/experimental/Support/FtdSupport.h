//===- FtdSupport.h --- FTD conversion support -----------------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Declares some utility functions which are useful for both the fast token
// delivery algorithm and for the GSA anlaysis pass. All the functions are about
// anlayzing relationships between blocks and handshake operations.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_SUPPORT_FTD_SUPPORT_H
#define DYNAMATIC_SUPPORT_FTD_SUPPORT_H

#include "dynamatic/Support/LLVM.h"
#include "experimental/Support/BooleanLogic/BoolExpression.h"
#include "mlir/Analysis/CFGLoopInfo.h"

namespace dynamatic {
namespace experimental {
namespace ftd {

/// Gets all the paths from block "start" to block "end" using a dfs search. If
/// "blockToTraverse" is non null, then we want the paths having that block in
/// the path; if "blocksToAvoid" is non empty, then we want the paths which do
/// not cross those paths.
std::vector<std::vector<Block *>>
findAllPaths(Block *start, Block *end, Block *blockToTraverse = nullptr,
             ArrayRef<Block *> blocksToAvoid = std::vector<Block *>());

/// Get the boolean condition determining when a path is executed. While
/// covering each block in the path, add the cofactor of each block to the list
/// of cofactors if not already covered
boolean::BoolExpression *
getPathExpression(ArrayRef<Block *> path, DenseSet<unsigned> &blockIndexSet,
                  const DenseMap<Block *, unsigned> &mapBlockToIndex,
                  const DenseSet<Block *> &deps = DenseSet<Block *>(),
                  bool ignoreDeps = true);

}; // namespace ftd
}; // namespace experimental
}; // namespace dynamatic

#endif // DYNAMATIC_SUPPORT_FTD_SUPPORT_H
