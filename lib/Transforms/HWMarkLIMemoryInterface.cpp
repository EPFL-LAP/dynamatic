//===- HWMarkLIMemoryInterface.cpp - Mark Latency Insensitive Memory Interfaces
//--*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the --hw-mark-li-memory-interface pass, which marks
// handshake memory interfaces as latency-insensitive.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Transforms/HWMarkLIMemoryInterface.h"
#include "dynamatic/Dialect/HW/HWOps.h"
#include "dynamatic/Support/LLVM.h"
#include "dynamatic/Support/Utils/Utils.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace dynamatic;

bool isMemoryController(StringRef opName) {
  return opName == "handshake.mem_controller";
}

namespace {
class HWMarkLIMemoryInterfacePass
    : public dynamatic::impl::HWMarkLIMemoryInterfaceBase<
          HWMarkLIMemoryInterfacePass> {
  void runDynamaticPass() override {
    mlir::ModuleOp modOp = getOperation();

    for (auto externalModule : modOp.getOps<hw::HWModuleExternOp>()) {
      // Identify memory controller modules by their hw.name attribute
      auto hw_name =
          externalModule->getAttrOfType<StringAttr>(RTL_NAME_ATTR_NAME);
      if (!hw_name)
        continue;
      if (!isMemoryController(hw_name.getValue()))
        continue;
      // Get hw.parameters attribute
      auto hw_parameters =
          externalModule->template getAttrOfType<DictionaryAttr>(
              "hw.parameters");
      // Add to the hw.parameters attribute the LATENCY_INSENSITIVE=1 entry
      DictionaryAttr newHwParameters;
      SmallVector<NamedAttribute> params;
      bool updatedAttr = false;
      for (auto param : hw_parameters.getValue()) {
        if (param.getName() == "LATENCY_INSENSITIVE") {
          // Update existing entry
          params.push_back(NamedAttribute(
              param.getName(),
              IntegerAttr::get(IntegerType::get(modOp.getContext(), 32,
                                                IntegerType::Unsigned),
                               /*value=*/1)));
          updatedAttr = true;
        } else {
          params.push_back(param);
        }
      }
      if (!updatedAttr) {
        llvm::fatal_error_handler_t(
            "Expected LATENCY_INSENSITIVE entry in hw.parameters");
      }
      newHwParameters = DictionaryAttr::get(modOp.getContext(), params);
      externalModule->setAttr("hw.parameters", newHwParameters);
    }
    doNotNameOperations();
  }
};
} // namespace

std::unique_ptr<dynamatic::DynamaticPass>
dynamatic::createHWMarkLIMemoryInterface() {
  return std::make_unique<HWMarkLIMemoryInterfacePass>();
}
