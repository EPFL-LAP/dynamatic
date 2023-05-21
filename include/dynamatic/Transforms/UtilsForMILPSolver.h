//===- UtilsForPlaceBuffers.h - functions for placing buffer  ---*- C++ -*-===//
//
// This file declaresfunction supports for buffer placement.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_TRANSFORMS_UTILSFORMILPSOVLER_H
#define DYNAMATIC_TRANSFORMS_UTILSFORMILPSOVLER_H

#include "dynamatic/Transforms/UtilsForPlaceBuffers.h"
#include "mlir/IR/BuiltinTypes.h"

namespace dynamatic {
namespace buffer {
    // MILP description functions
  arch *findArcWithVarName(std::string varName,
                         std::vector<basicBlock *> &bbList);

  static bool toSameDstOp(std::string var1, std::string var2,
                        std::vector<basicBlock *> &bbList);

  static bool fromSameSrcOp(std::string var1, std::string var2,
                          std::vector<basicBlock *> &bbList);

  std::vector<std::string>
  findSameDstOpStrings(const std::string &inputString,
                      const std::vector<std::string> &stringList,
                      std::vector<basicBlock *> &bbList);

  std::vector<std::string>
  findSameSrcOpStrings(const std::string &inputString,
                     const std::vector<std::string> &stringList,
                     std::vector<basicBlock *> &bbList);


  void extractMarkedGraphBB(std::vector<basicBlock *> &bbList);
}
}

#endif // DYNAMATIC_TRANSFORMS_UTILSFORMILPSOVLER_H