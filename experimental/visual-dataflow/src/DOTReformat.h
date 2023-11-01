//===- DOTReformat.h - Reformats a .dot file -------------------*- C++ -*-===//
//
// The DOTReformat class is a temporary fix to the dot file being formatted wr-
// onfully so it can be parsed correctly.
//===----------------------------------------------------------------------===//

#include <string>
#include "mlir/Support/LogicalResult.h"


mlir::LogicalResult reformatDot(const std::string &inputFileName,
                 const std::string &outputFileName);
