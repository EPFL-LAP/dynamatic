//===- Simulators.h ----------------------------------------------*- C++
//-*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef HLS_VERIFIER_SIMULATORS_H
#define HLS_VERIFIER_SIMULATORS_H

#include "Utilities.h"
#include "VerificationContext.h"
#include "dynamatic/Support/System.h"
#include "mlir/Support/LogicalResult.h"

enum SimulatorKind {
  MODELSIM,
  XSIM,
  VERILATOR,
  GHDL,
};

class Simulator {

protected:
  VerificationContext *ctx;

public:
  std::string simulationCommand;

  Simulator(VerificationContext *context) : ctx(context) {}

  virtual ~Simulator() {};

  virtual mlir::LogicalResult generateScripts() const = 0;

  virtual void execSimulation() const = 0;
};

class XSimSimulator : public Simulator {

public:
  XSimSimulator(VerificationContext *context) : Simulator(context) {}

  void execSimulation() const override {
    exec("xelab", "-prj", ctx->getXsimPrjFilePath(), "work.tb", "--timescale",
         "1ns/1ps", "-s", "tb", "-R");
  }

  mlir::LogicalResult generateScripts() const override {
    vector<string> filelistVhdl =
        getListOfFilesInDirectory(ctx->getHdlSrcDir(), ".vhd");
    vector<string> filelistVerilog =
        getListOfFilesInDirectory(ctx->getHdlSrcDir(), ".v");
    vector<string> fileListSystemVerilog =
        getListOfFilesInDirectory(ctx->getHdlSrcDir(), ".sv");

    std::error_code ec;
    llvm::raw_fd_ostream os(ctx->getXsimPrjFilePath(), ec);

    for (auto &it : filelistVhdl)
      os << "vhdl2008 work " << it << "\n";

    for (auto &it : filelistVerilog)
      os << "verilog work " << it << "\n";

    for (auto &it : fileListSystemVerilog)
      os << "sv work " << it << "\n";

    return mlir::success();
  }
};

class GHDLSimulator : public Simulator {

  const StringLiteral simulatorName = "ghdl";

public:
  GHDLSimulator(VerificationContext *context) : Simulator(context) {}

  void execSimulation() const override {
    exec("bash", ctx->getGhdlShFilePath());
  }

  mlir::LogicalResult generateScripts() const override {
    // [START Example of generated script]
    // # Imports all design files into the GHDL library. Uses the VHDL-2008
    // # standard and allows the use of synopsys non-standard packages
    // ghdl -i --std=08 -fsynopsys
    // /data/dynamatic/integration-test/fir/out/sim/HDL_SRC/addi.vhd
    // ...
    // ghdl -i --std=08 -fsynopsys
    // data/dynamatic/integration-test/fir/out/sim/HDL_SRC/oebh.vhd
    //
    // # Compiles the design in the correct compilation order, and relaxes "
    // # some rules to only cause warnings instead of errors
    // ghdl -m --std=08 -fsynopsys -frelaxed tb
    //
    // # Runs the Simulation with top-level unit tb
    // ghdl -r --std==08 -fsynopsys tb
    //
    // # Exits the script
    // exit 0
    // [End Example of generated script]

    vector<string> filelistVhdl =
        getListOfFilesInDirectory(ctx->getHdlSrcDir(), ".vhd");

    vector<string> filelistVerilog =
        getListOfFilesInDirectory(ctx->getHdlSrcDir(), ".v");

    if (!filelistVerilog.empty()) {
      return mlir::failure();
    }

    std::error_code ec;
    llvm::raw_fd_ostream os(ctx->getGhdlShFilePath(), ec);

    // [Start Flag explanation]
    //  - --std=08 : use the VHDL-2008 standard for compilation
    //  - -fsynopsys : allow the use of synopsys non-standard packages
    //    (e.g.std_logic_arith, std_logic_signed, etc.). These packages would
    //    otherwise produce an error. -frelaxed : generates warnings instead of
    //    errors
    // [End Flag explanation]

    // We only import VHDL files (.vhd and .vhdl) because GHDL does not work
    // with Verilog files (.v)
    os << "# Imports all design files into the GHDL library. Uses the "
          "VHDL-2008 standard and allows the use of synopsys non-standard "
          "packages\n";
    for (auto &it : filelistVhdl)
      os << simulatorName << " -i --std=08 -fsynopsys " << it << "\n";

    // -m compiles a design in the correct compilation order
    os << "# Compiles the design in the correct compilation order, and relaxes "
          "some rules to only cause warnings instead of errors\n";
    os << simulatorName << " -m --std=08 -fsynopsys -frelaxed tb\n";

    // -r runs the simulation
    os << "# Runs the Simulation with top-level unit tb\n";
    os << simulatorName << " -r --std=08 -fsynopsys -frelaxed tb\n";

    os << "# Exits the script\n";
    os << "exit 0";

    return mlir::success();
  }
};

class VSimSimulator : public Simulator {

public:
  VSimSimulator(VerificationContext *context) : Simulator(context) {}

  void execSimulation() const override {
    exec("vsim", "-c", "-do", ctx->getModelsimDoFilePath());
  }

  mlir::LogicalResult generateScripts() const override {
    vector<string> filelistVhdl =
        getListOfFilesInDirectory(ctx->getHdlSrcDir(), ".vhd");
    vector<string> filelistVerilog =
        getListOfFilesInDirectory(ctx->getHdlSrcDir(), ".v");
    vector<string> fileListSystemVerilog =
        getListOfFilesInDirectory(ctx->getHdlSrcDir(), ".sv");

    std::error_code ec;
    llvm::raw_fd_ostream os(ctx->getModelsimDoFilePath(), ec);
    // os << "vdel -all" << endl;
    os << "vlib work\n";
    os << "vmap work work\n";
    os << "project new . simulation work modelsim.ini 0\n";
    os << "project open simulation\n";

    // We use the same VHDL TB for simulating both the VHDL and Verilog designs
    // in ModelSim.
    for (auto &it : filelistVhdl)
      os << "project addfile " << it << "\n";

    for (auto &it : filelistVerilog)
      os << "project addfile " << it << "\n";

    for (auto &it : fileListSystemVerilog)
      os << "project addfile " << it << "\n";

    os << "project calculateorder\n";
    os << "project compileall\n";
    if (ctx->useVivadoFPU()) {
      os << "eval vsim tb work.glbl\n";
    } else {
      os << "eval vsim tb\n";
    }
    os << "log -r *\n";
    os << "run -all\n";
    os << "exit\n";

    return mlir::success();
  }
};

class Verilator : public Simulator {

public:
  Verilator(VerificationContext *context) : Simulator(context) {}

  void execSimulation() const override {
    exec("bash", ctx->getVerilatorShFilePath());
  }

  mlir::LogicalResult generateScripts() const override {

    vector<string> filelistVerilog =
        getListOfFilesInDirectory(ctx->getHdlSrcDir(), ".v");
    vector<string> fileListSystemVerilog =
        getListOfFilesInDirectory(ctx->getHdlSrcDir(), ".sv");

    if (filelistVerilog.empty()) {
      return mlir::failure();
    }

    std::error_code ec;
    llvm::raw_fd_ostream os(ctx->getVerilatorShFilePath(), ec);

    os << "# Verilating (translate the desing into an executable)\n";
    os << "# --trace: enables wavefrom cration\n";
    os << "# --Mdir: specifies the './verilator' directory as working "
          "directory\n";
    os << "# -cc: Verilator output is C++\n";
    os << "# --exe: Generates executable with 'verilator_main.cpp' as main "
          "function\n";
    os << "# --trace-underscore: Enable coverage of signals that start with "
          "and underscore\n";
    os << "# --Wno-UNOPTFLAT: ignores warnings about disabled signal "
          "optimization\n";
    os << "# --top-module: 'tb' is specified as top level module\n";
    os << "# --timing: Enables support for timing constructs (e.g. wait "
          "statements)\n";
    os << "# --Wno-REALCVT: Ignores a real number being rounded to an integer "
          "(during calculation of the kernel's latency)\n";
    os << "verilator --trace -Mdir ./verilator -cc ";

    for (auto &it : filelistVerilog)
      os << it << " ";
    for (auto &it : fileListSystemVerilog)
      os << it << " ";

    os << "--exe verilator_main.cpp --trace-underscore --Wno-UNOPTFLAT "
          "--top-module tb --timing -Wno-REALCVT\n";

    os << "make -j -C ./verilator/ -f Vtb.mk Vtb\n";

    os << "./verilator/Vtb\n";

    return mlir::success();
  }
};

#endif // HLS_VERIFIER_SIMULATORS_H
