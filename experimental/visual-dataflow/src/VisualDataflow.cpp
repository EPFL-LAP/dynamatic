//===- VisualDataflow.cpp - Godot-visible types -----------------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines Godot-visible data types.
//
//===----------------------------------------------------------------------===//

#include "VisualDataflow.h"
#include "dynamatic/Support/DOTPrinter.h"
#include "dynamatic/Support/TimingModels.h"
#include "godot_cpp/core/class_db.hpp"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"

using namespace llvm;
using namespace mlir;
using namespace dynamatic;
using namespace godot;

void VisualDataflow::_bind_methods() {
  ClassDB::bind_method(D_METHOD("getNodePosX", "index"),
                       &VisualDataflow::getNodePosX);
  ClassDB::bind_method(D_METHOD("getNodePosY", "index"),
                       &VisualDataflow::getNodePosY);
  ClassDB::bind_method(D_METHOD("getNumberOfNodes"),
                       &VisualDataflow::getNumberOfNodes);
}

VisualDataflow::VisualDataflow() {

  Node n1;
  Node n2;
  Node n3;

  n1.x = 100.0;
  n1.y = 200.0;

  n2.x = 100.0;
  n2.y = 600.0;

  n3.x = 400.0;
  n3.y = 800.0;

  /*nodes[0] = n1;
  nodes[1] = n2;
  nodes[2] = n3;*/

  std::string filename = "test.mlir";
  auto fileOrErr = MemoryBuffer::getFileOrSTDIN(filename.c_str());
  if (std::error_code error = fileOrErr.getError()) {
    // llvm::errs() << "could not open input file '" << filename
    //              << "': " << error.message() << "\n";
    return;
  }

  MLIRContext context;
  context.loadDialect<func::FuncDialect, memref::MemRefDialect,
                      arith::ArithDialect, LLVM::LLVMDialect,
                      handshake::HandshakeDialect>();
  context.allowUnregisteredDialects();

  // Load the MLIR module
  SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), SMLoc());
  mlir::OwningOpRef<mlir::ModuleOp> module(
      mlir::parseSourceFile<ModuleOp>(sourceMgr, &context));
  if (!module)
    return;
}

double VisualDataflow::getNodePosX(int index) { return 100; }
double VisualDataflow::getNodePosY(int index) { return 100; }
int VisualDataflow::getNumberOfNodes() { return numberOfNodes; }

void VisualDataflow::myProcess(double delta) {}
