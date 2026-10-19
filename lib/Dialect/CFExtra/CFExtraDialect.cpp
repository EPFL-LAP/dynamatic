#include "dynamatic/Dialect/CFExtra/CFExtraDialect.h"
#include "dynamatic/Dialect/CFExtra/CFExtraOps.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/Support/SMLoc.h"

using namespace dynamatic;

void cf_extra::CFExtraDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "dynamatic/Dialect/CFExtra/CFExtra.cpp.inc"
      >();
}

#include "dynamatic/Dialect/CFExtra/CFExtraDialect.cpp.inc"