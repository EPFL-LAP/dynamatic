//===- util.h - Integration testing helper functions -----------*- C++-*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implementation of integration test helper functions.
//
//===----------------------------------------------------------------------===//
#include "util.h"

#include <regex>
#include <set>
#include <sstream>

#include <nlohmann/json.hpp>

using json = nlohmann::json;

static bool runSubprocess(const std::vector<std::string> &args,
                          const fs::path &outputPath) {
  std::ostringstream command;
  command << args[0];
  for (size_t i = 1; i < args.size(); ++i) {
    command << " " << args[i];
  }
  command << " 1>" << outputPath.string();
  std::cout << "[INFO] Running command: " << command.str().c_str() << std::endl;
  return std::system(command.str().c_str()) == 0;
}

int runIntegrationTest(IntegrationTestData &config) {
  fs::path cSourcePath =
      config.benchmarkPath / config.name / (config.name + ".c");

  std::string tmpFilename = "tmp_" + config.name + ".dyn";
  std::ofstream scriptFile(tmpFilename);
  if (!scriptFile.is_open()) {
    std::cout << "[ERROR] Failed to create .dyn script file" << std::endl;
    return -1;
  }

  scriptFile << "set-dynamatic-path " << DYNAMATIC_ROOT << std::endl
             << "set-src " << cSourcePath.string() << std::endl
             << "set-clock-period 5" << std::endl;

  // clang-format off
  scriptFile << "compile"
             << " --buffer-algorithm " << config.bufferAlgorithm
             << (config.useSharing ? " --sharing" : "")
             << (config.useRigidification ? " --rigidification" : "")
             << " --milp-solver " << config.milpSolver << std::endl;
  // clang-format on

  // Assert testVHDL or testVerilog is true
  if (!config.testVHDL && !config.testVerilog) {
    std::cout << "[ERROR] Either testVHDL or testVerilog must be true"
              << std::endl;
    return -1;
  }

  if (config.verifyInvariants) {
    scriptFile << "verify-invariants" << std::endl;
  }

  // Verify Verilog works correctly
  if (config.testVerilog) {
    scriptFile << "write-hdl --hdl verilog" << std::endl
               << "simulate" << std::endl;
  }
  // Verify VHDL works correctly
  if (config.testVHDL) {
    // By default, the report containing the simulation time is re-written
    // during the second simulation (i.e., the VHDL simulation).
    scriptFile << "write-hdl --hdl vhdl" << std::endl
               << "simulate" << std::endl;
  }
  scriptFile << "exit" << std::endl;

  scriptFile.close();

  fs::path dynamaticPath = fs::path(DYNAMATIC_ROOT) / "bin" / "dynamatic";
  fs::path dynamaticOutPath =
      cSourcePath.parent_path() / "out" / "dynamatic_out.txt";
  fs::path dynamaticErrPath =
      cSourcePath.parent_path() / "out" / "dynamatic_err.txt";
  if (!fs::exists(dynamaticOutPath.parent_path())) {
    fs::create_directories(dynamaticOutPath.parent_path());
  }

  std::string cmd = dynamaticPath.string() + " --exit-on-failure --run ";
  cmd += tmpFilename;
  cmd += " 1> ";
  cmd += dynamaticOutPath;
  cmd += " 2> ";
  cmd += dynamaticErrPath;

  int status = system(cmd.c_str());
  if (status == 0) {
    fs::path logFilePath =
        cSourcePath.parent_path() / "out" / "sim" / "report.txt";
    config.simTime = getSimulationTime(logFilePath);
  }

  return status;
}

bool runSpecIntegrationTest(const std::string &name, int &outSimTime) {
  bool spec = true;

  const std::string DYNAMATIC_OPT_BIN =
      fs::path(DYNAMATIC_ROOT) / "bin" / "dynamatic-opt";

  const std::string EXPORT_DOT_BIN =
      fs::path(DYNAMATIC_ROOT) / "bin" / "export-dot";

  const std::string EXPORT_RTL_BIN =
      fs::path(DYNAMATIC_ROOT) / "bin" / "export-rtl";

  const std::string SIMULATE_SH = fs::path(DYNAMATIC_ROOT) / "tools" /
                                  "dynamatic" / "scripts" / "simulate.sh";

  const std::string RTL_CONFIG =
      fs::path(DYNAMATIC_ROOT) / "data" / "rtl-config-vhdl.json";

  const std::string SIMULATOR_NAME = "vsim"; // modelsim

  fs::path cFilePath =
      fs::path(DYNAMATIC_ROOT) / "integration-test" / name / (name + ".c");

  std::cout << "[INFO] Running " << name << " with speculation flow"
            << std::endl;

  fs::path cFileDir = cFilePath.parent_path();
  fs::path outDir = cFileDir / "out_spec";
  if (fs::exists(outDir)) {
    fs::remove_all(outDir);
  }
  fs::create_directories(outDir);

  fs::path compOutDir = outDir / "comp";
  fs::create_directories(compOutDir);

  // Regenerate cf.mlir from the .c kernel via the new clang-plugin +
  // translate-llvm-to-std + consume-producer-output-attr-marker flow. Mirrors
  // the pre-handshake steps in tools/dynamatic/scripts/compile.sh so the
  // baked-in cf.mlir fixture is no longer needed and the spec attribute
  // injected by the DYN speculate pragma reaches HandshakeSpeculation.
  const std::string CLANG_BIN = fs::path(DYNAMATIC_ROOT) / "bin" / "clang";
  const std::string LLVM_OPT_BIN = fs::path(DYNAMATIC_ROOT) / "bin" / "opt";
  const std::string TRANSLATE_BIN =
      fs::path(DYNAMATIC_ROOT) / "build" / "bin" / "translate-llvm-to-std";
  const std::string DYN_PRAGMAS_PLUGIN =
      fs::path(DYNAMATIC_ROOT) / "build" / "lib" / "DynPragmasPlugin.so";
  const std::string SOURCE_REWRITER_BIN =
      fs::path(DYNAMATIC_ROOT) / "build" / "bin" / "source-rewriter";
  const std::string MEM_DEP_PLUGIN =
      fs::path(DYNAMATIC_ROOT) / "build" / "lib" / "MemDepAnalysis.so";
  const std::string CLANG_HEADERS =
      fs::path(DYNAMATIC_ROOT) / "build" / "include" / "clang_headers";
  const std::string DYN_INCLUDE = fs::path(DYNAMATIC_ROOT) / "include";

  fs::path fClang = compOutDir / "clang.ll";
  fs::path fClangOpt = compOutDir / "clang.opt.ll";
  fs::path fClangDep = compOutDir / "clang.opt.dep.ll";
  fs::path cfFile = compOutDir / "cf.mlir";

  // 0. Rewrite `&&`/`||` into bitwise `&`/`|` (source-rewriter) so the
  //    spec'd boolean isn't a phi merge across a short-circuit CFG diamond.
  //    Without this, the producer of `loop_again = i < N && !cond` becomes a
  //    block-arg after mem2reg, which ConsumeProducerOutputAttrMarker cannot
  //    attach attributes to. source-rewriter edits the file in place, so we
  //    work on a copy in compOutDir.
  fs::path fCNoSc = compOutDir / (name + ".no_sc.c");
  fs::copy_file(cFilePath, fCNoSc, fs::copy_options::overwrite_existing);
  if (!runSubprocess({SOURCE_REWRITER_BIN, fCNoSc.string(), "--", "-I",
                      DYN_INCLUDE, "-I", cFileDir.string(), "-I",
                      CLANG_HEADERS},
                     compOutDir / "no_sc.stdout.txt")) {
    std::cerr << "Failed to rewrite short-circuit ops for " << name << "\n";
    return false;
  }

  // 1. clang -O0 -emit-llvm with the DynPragmasPlugin loaded so the
  //    `#pragma DYN speculate` is rewritten to a `__dyn_speculate` call.
  if (!runSubprocess({CLANG_BIN, "-O0", "-funroll-loops", "-S", "-emit-llvm",
                      fCNoSc.string(), "-I", DYN_INCLUDE, "-I",
                      cFileDir.string(), "-I", CLANG_HEADERS,
                      "-fplugin=" + DYN_PRAGMAS_PLUGIN, "-Xclang",
                      "-ffp-contract=off", "-o", fClang.string()},
                     compOutDir / "clang.stdout.txt")) {
    std::cerr << "Failed to clang-compile " << name << "\n";
    return false;
  }

  // 2. Strip attributes the LLVM optimizer / mlir-translate cannot consume.
  std::system(("sed -i 's/optnone//g; s/noinline//g; "
               "s/^target datalayout = .*$//g; s/^target triple = .*$//g' " +
               fClang.string())
                  .c_str());

  // 3. opt with the standard dynamatic LLVM pass set.
  if (!runSubprocess(
          {LLVM_OPT_BIN, "-S",
           "'-passes=inline,mem2reg,consthoist,instcombine<max-iterations=1000;"
           "no-use-loop-info>,function(loop-mssa(licm<no-allowspeculation>)),"
           "function(loop(loop-idiom,indvars,loop-deletion)),simplifycfg,loop-"
           "rotate,simplifycfg,sink,lowerswitch,simplifycfg,dce'",
           fClang.string()},
          fClangOpt)) {
    std::cerr << "Failed to apply LLVM optimization to " << name << "\n";
    return false;
  }

  // 4. Memory dependency analysis (polly-backed) annotates loads/stores.
  if (!runSubprocess({LLVM_OPT_BIN, "-S", "-load-pass-plugin", MEM_DEP_PLUGIN,
                      "-passes=mem-dep-analysis", "-polly-process-unprofitable",
                      fClangOpt.string()},
                     fClangDep)) {
    std::cerr << "Failed to apply mem-dep-analysis to " << name << "\n";
    return false;
  }

  // 5. LLVM IR -> std MLIR. This is where the `__dyn_speculate` call gets
  //    rewritten into a `dynamatic.producer_output_attr_marker` op carrying the
  //    `dynamatic.speculate` attribute.
  if (!runSubprocess({TRANSLATE_BIN, fClangDep.string(), "-function-name", name,
                      "-csource", cFilePath.string(), "-dynamatic-path",
                      DYNAMATIC_ROOT, "-o", cfFile.string()},
                     compOutDir / "translate.stdout.txt")) {
    std::cerr << "Failed to translate LLVM IR to std for " << name << "\n";
    return false;
  }

  // 6. CF-level dynamatic transforms, including
  //    --consume-producer-output-attr-marker which migrates the speculate
  //    attribute from the marker onto its producer op.
  //    --allow-unregistered-dialect is required because the marker op is
  //    the unregistered `dynamatic.producer_output_attr_marker`.
  fs::path cfDynTransformed = compOutDir / "cfDynTransformed.mlir";
  if (!runSubprocess(
          {DYNAMATIC_OPT_BIN, "--allow-unregistered-dialect", cfFile.string(),
           "--drop-unlisted-functions=function-names=" + name,
           "--func-set-arg-names=source=" + cFilePath.string(),
           "--flatten-memref-row-major", "--canonicalize",
           "--arith-reduce-strength=max-adder-depth-mul=1", "--push-constants",
           "--consume-producer-output-attr-marker", "--mark-memory-interfaces"},
          cfDynTransformed)) {
    std::cerr << "Failed to apply Dynamatic CF transforms to " << name << "\n";
    return false;
  }

  fs::path handshake = compOutDir / "handshake.mlir";
  if (!runSubprocess({DYNAMATIC_OPT_BIN, cfDynTransformed.string(),
                      "--lower-cf-to-handshake"},
                     handshake)) {
    std::cerr << "Failed to compile cf to handshake\n";
    return false;
  }

  fs::path handshakeTransformed = compOutDir / "handshakeTransformed.mlir";
  if (!runSubprocess(
          {DYNAMATIC_OPT_BIN, handshake.string(),
           "--handshake-analyze-lsq-usage",
           "--handshake-replace-memory-interfaces",
           "--handshake-minimize-cst-width", "--handshake-optimize-bitwidths",
           "--handshake-materialize", "--handshake-infer-basic-blocks"},
          handshakeTransformed)) {
    std::cerr << "Failed to apply transformations to handshake\n";
    return false;
  }

  fs::path handshakeBuffered = compOutDir / "handshakeBuffered.mlir";
  std::string timingModel =
      (fs::path(DYNAMATIC_ROOT) / "data" / "components.json").string();
  if (!runSubprocess(
          {DYNAMATIC_OPT_BIN, handshakeTransformed.string(),
           "\"--handshake-set-unit-impl-attr=target-period=4 impl=flopoco "
           "timing-models=" +
               timingModel + "\"",
           "--handshake-set-buffering-properties=version=fpga20",
           "\"--handshake-place-buffers=algorithm=on-merges timing-models=" +
               timingModel + "\""},
          handshakeBuffered)) {
    std::cerr << "Failed to place simple buffers\n";
    return false;
  }

  fs::path handshakeCanonicalized = compOutDir / "handshakeCanonicalized.mlir";
  if (!runSubprocess({DYNAMATIC_OPT_BIN, handshakeBuffered.string(),
                      "--handshake-canonicalize",
                      "--handshake-hoist-ext-instances"},
                     handshakeCanonicalized)) {
    std::cerr << "Failed to canonicalize Handshake\n";
    return false;
  }

  fs::path handshakeExport;
  if (spec) {
    fs::path handshakeSpeculation = compOutDir / "handshakeSpeculation.mlir";
    if (!runSubprocess({DYNAMATIC_OPT_BIN, handshakeCanonicalized.string(),
                        "--handshake-speculation", "--handshake-materialize",
                        "--handshake-canonicalize"},
                       handshakeSpeculation)) {
      std::cerr << "Failed to add speculative units\n";
      return false;
    }

    fs::path bufferJsonPath = cFileDir / "buffer.json";
    std::ifstream bufferFile(bufferJsonPath);
    json buffers;
    bufferFile >> buffers;

    std::vector<std::string> bufferArgs = {DYNAMATIC_OPT_BIN,
                                           handshakeSpeculation.string()};
    for (const auto &buffer : buffers) {
      bufferArgs.push_back(
          "\"--handshake-placebuffers-custom=pred=" +
          buffer["pred"].get<std::string>() +
          " outid=" + std::to_string(buffer["outid"].get<int>()) +
          " slots=" + std::to_string(buffer["slots"].get<int>()) +
          " type=" + buffer["type"].get<std::string>() + "\"");
    }

    handshakeExport = compOutDir / "handshake_export.mlir";
    if (!runSubprocess(bufferArgs, handshakeExport)) {
      std::cerr << "Failed to export Handshake\n";
      return false;
    }
  } else {
    handshakeExport = compOutDir / "handshake_export.mlir";
    fs::copy_file(handshakeCanonicalized, handshakeExport,
                  fs::copy_options::overwrite_existing);
  }

  fs::path dotFile = compOutDir / (name + ".dot");
  if (!runSubprocess({EXPORT_DOT_BIN, handshakeExport.string(),
                      "--edge-style=spline", "--label-type=uname"},
                     dotFile)) {
    std::cerr << "Failed to export dot file\n";
    return false;
  }

  fs::path pngFile = compOutDir / (name + ".png");
  if (!runSubprocess({"dot", "-Tpng", dotFile.string()}, pngFile)) {
    std::cerr << "Failed to create PNG file\n";
    return false;
  }

  fs::path hw = compOutDir / "hw.mlir";
  if (!runSubprocess({DYNAMATIC_OPT_BIN, handshakeExport.string(),
                      "--lower-handshake-to-hw"},
                     hw)) {
    std::cerr << "Failed to lower handshake to hw\n";
    return false;
  }

  fs::path hdlDir = outDir / "hdl";
  if (std::system((EXPORT_RTL_BIN + " " + hw.string() + " " + hdlDir.string() +
                   " " + RTL_CONFIG + " --dynamatic-path " + DYNAMATIC_ROOT +
                   " --hdl vhdl")
                      .c_str()) != 0) {
    std::cerr << "Failed to export hdl\n";
    return false;
  }

  std::cout << "Simulator launching\n";
  if (std::system((SIMULATE_SH + " " + DYNAMATIC_ROOT + " " +
                   cFileDir.string() + " " + outDir.string() + " " + name +
                   " \"\" " + "false" + " " + SIMULATOR_NAME)
                      .c_str()) != 0) {
    std::cerr << "Failed to simulate\n";
    return false;
  }

  std::cout << "Simulation succeeded\n";

  fs::path logFilePath = cFileDir / "out_spec" / "sim" / "report.txt";
  outSimTime = getSimulationTime(logFilePath);

  return true;
}

int getSimulationTime(const fs::path &logFile) {
  std::ifstream file(logFile);
  if (!file.is_open()) {
    std::cout << "[WARNING] Failed to open " << logFile << std::endl;
    return -1;
  }

  std::vector<std::string> lines;
  std::string line;

  // Read all lines into a vector
  while (std::getline(file, line)) {
    lines.push_back(line);
  }

  std::regex pattern("Simulation done! Latency = (\\d+) cycles");
  std::smatch match;

  // Search lines in reverse order
  for (auto it = lines.rbegin(); it != lines.rend(); ++it) {
    if (std::regex_search(*it, match, pattern)) {
      return std::stoi(match[1]);
    }
  }

  std::cout << "[WARNING] Log file does not contain simulation time!"
            << std::endl;
  return -1;
}
