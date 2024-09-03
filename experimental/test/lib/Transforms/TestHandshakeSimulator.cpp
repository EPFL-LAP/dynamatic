//===- TestHandshakeSimulator.cpp - Handshake simulator tests  ---- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Test pass for Handhake simulator. Run with --exp-test-handshake-simulator.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/LLVM.h"
#include "experimental/Support/HandshakeSimulator.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/Any.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"
#include <cmath>
#include <fstream>
#include <string>

using namespace mlir;
using namespace dynamatic;
using namespace experimental;
namespace ljson = llvm::json;

namespace {

struct TestHandshakeSimulatorOptions {
  std::string tests;
};

struct TestHandshakeSimulator
    : public PassWrapper<TestHandshakeSimulator,
                         OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestHandshakeSimulator)
  using Base =
      PassWrapper<TestHandshakeSimulator, OperationPass<mlir::ModuleOp>>;

  TestHandshakeSimulator() : Base() {};
  TestHandshakeSimulator(const TestHandshakeSimulator &other) : Base(other) {};

  StringRef getArgument() const final { return "exp-test-handshake-simulator"; }

  StringRef getDescription() const final {
    return "Test the Handshake simulator";
  }

  void runOnOperation() override {
    mlir::ModuleOp modOp = getOperation();

    // Retrieve the single Handshake function
    auto allFunctions = modOp.getOps<handshake::FuncOp>();
    if (std::distance(allFunctions.begin(), allFunctions.end()) != 1) {
      llvm::errs() << "Expected single Handshake function\n";
      return signalPassFailure();
    }

    handshake::FuncOp funcOp = *allFunctions.begin();
    experimental::Simulator sim(funcOp);
    std::ifstream testFile(tests);
    if (!testFile.is_open()) {
      llvm::errs() << "Failed to open JSON with tests @\"" << tests << "\"\n";
      return signalPassFailure();
    }

    std::string jsonString;
    std::string line;
    while (std::getline(testFile, line))
      jsonString += line;

    llvm::Expected<ljson::Value> value = ljson::parse(jsonString);

    if (!value) {
      llvm::errs() << "Failed to parse JSON with tests @ \"" << tests
                   << "\" as JSON.\n-> " << toString(value.takeError()) << "\n";
      return signalPassFailure();
    }

    ljson::Path::Root jsonRoot(tests);
    ljson::Path jsonPath(jsonRoot);

    ljson::Array *jsonComponents = value->getAsArray();
    if (!jsonComponents) {
      jsonRoot.printErrorContext(*value, llvm::errs());
      return signalPassFailure();
    }

    // Iterate through all the tests
    for (auto [idx, jsonComponent] : llvm::enumerate(*jsonComponents)) {
      std::vector<std::string> inputArgs;
      auto *obj = jsonComponent.getAsObject();

      auto *argumentsTrue = obj->find("arguments")->getSecond().getAsArray();

      // read test's arguments
      for (auto &argObj : *argumentsTrue) {
        auto arg = argObj.getAsString();
        if (arg.has_value()) {
          inputArgs.push_back(arg.value().str());
        }
      }

      // Run the simulation
      sim.simulate(inputArgs);

      auto *resultsTrue = obj->find("results")->getSecond().getAsArray();

      // true results
      auto resTrue = resultsTrue->front().getAsString();
      auto iterationsTrue =
          obj->find("iterations")->getSecond().getAsUINT64().value();

      // results being checked
      auto res = sim.getResData();
      SmallVector<char, 1000> resDataStr;
      auto iterations = sim.getIterNum();

      bool resEq = false;
      Type resType;
      bool hasIntOrFloatType = false;

      // find the result with int or float type
      for (auto &type : funcOp.getResultTypes()) {
        llvm::TypeSwitch<mlir::Type>(type)
            .Case<handshake::ChannelType>(
                [&](handshake::ChannelType channelType) {
                  resType = channelType.getDataType();
                  hasIntOrFloatType = true;
                })
            .Case<handshake::ControlType>(
                [&](handshake::ControlType controlType) {})
            .Default([&](auto) {
              llvm::errs() << "Unsupported type: only control or channel!\n";
              return signalPassFailure();
            });
      }

      // if the result actually exists
      if (resTrue.has_value() && hasIntOrFloatType) {
        llvm::TypeSwitch<mlir::Type>(resType)
            .Case<IntegerType>([&](IntegerType intType) {
              APInt resData = any_cast<APInt>(res);
              resData.toString(resDataStr, 10, true);

              int resDataTrue;
              resTrue->getAsInteger(10, resDataTrue);

              if (resDataTrue == resData.getSExtValue())
                resEq = true;
            })
            .Case<FloatType>([&](FloatType floatType) {
              APFloat resData = any_cast<APFloat>(res);
              resData.toString(resDataStr);

              double resDataTrue;
              resTrue->getAsDouble(resDataTrue);

              if (std::fabs(resData.convertToDouble() - resDataTrue) < 0.0001)
                resEq = true;
            })
            .Default([&](auto) {
              llvm::errs()
                  << "Unsupported data type: only int or float are allowed!\n";
              return signalPassFailure();
            });

        if (iterations != iterationsTrue || !resEq) {
          llvm::errs() << "Test failed!\"\n"
                       << "Number of iterations: " << iterations << " vs "
                       << iterationsTrue << "\n";
          llvm::errs() << "Value: " << resDataStr << " vs " << resTrue->data()
                       << "\n";
          signalPassFailure();
        }
      }
      sim.reset();
    }
  }

  TestHandshakeSimulator(const TestHandshakeSimulatorOptions &options)
      : TestHandshakeSimulator() {
    tests = options.tests;
  }

protected:
  Pass::Option<std::string> tests{
      *this, "path-to-tests",
      ::llvm::cl::desc("Pass to JSON-formatted set of tests"),
      ::llvm::cl::init("")};
};
} // namespace

namespace dynamatic {
namespace experimental {
namespace test {
void registerTestHandshakeSimulator() {
  PassRegistration<TestHandshakeSimulator>();
}
} // namespace test
} // namespace experimental
} // namespace dynamatic
