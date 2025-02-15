#include "mlir/IR/Attributes.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <llvm/ADT/StringSet.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/JSON.h>
#include <llvm/Support/raw_ostream.h>
#include <sstream>
#include <string>

#include "../experimental/tools/elastic-miter-generator/ElasticMiterFabricGeneration.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Transforms/HandshakeMaterialize.h"
#include "mlir/Support/LogicalResult.h"

using namespace mlir;

namespace dynamatic::experimental {

std::string createMiterCall(const SmallVector<std::string> &args,
                            const SmallVector<std::string> &res) {
  std::ostringstream miter;
  miter << "VAR miter : elastic_miter(";

  for (size_t i = 0; i < args.size(); ++i) {
    if (i > 0)
      miter << ", ";
    miter << "in_ndw" << i << ".dataOut0, in_ndw" << i << ".valid0";
  }

  miter << ", ";

  for (size_t i = 0; i < res.size(); ++i) {
    if (i > 0)
      miter << ", ";
    miter << "out_ndw" << i << ".ready0";
  }

  miter << ");\n";
  return miter.str();
}

FailureOr<std::string> createReachableStateWrapper(ModuleOp mlir, int n = 0,
                                                   bool inf = false) {

  auto failOrFuncOp = dynamatic::experimental::getModuleFuncOpAndCheck(mlir);
  if (failed(failOrFuncOp))
    return failure();

  FuncOp funcOp = failOrFuncOp.value();

  std::string funcName = funcOp.getNameAttr().str();
  // TODO maybe do this properly
  funcName = "model";

  SmallVector<std::string> argNames;
  for (Attribute attr : funcOp.getArgNames()) {
    auto strAttr = attr.dyn_cast<StringAttr>();
    argNames.push_back(strAttr.getValue().str());
  }

  SmallVector<std::string> resNames;
  for (Attribute attr : funcOp.getResNames()) {
    auto strAttr = attr.dyn_cast<StringAttr>();
    resNames.push_back(strAttr.getValue().str());
  }

  std::ostringstream wrapper;
  wrapper << "#include \"" << funcName << ".smv\"\n";
  wrapper << "\n#ifndef BOOL_INPUT\n#define BOOL_INPUT\n";
  wrapper << "MODULE bool_input(nReady0, max_tokens)\n"
             "    VAR dataOut0 : boolean;\n"
             "    VAR counter : 0..31;\n"
             "    ASSIGN\n"
             "    init(counter) := 0;\n"
             "    next(counter) := case\n"
             "      nReady0 & counter < max_tokens : counter + 1;\n"
             "      TRUE : counter;\n"
             "    esac;\n"
             "    \n"
             "    -- bool_input persistent\n"
             "    ASSIGN\n"
             "    next(dataOut0) := case \n"
             "      valid0 & !nReady0 : dataOut0;\n"
             "      TRUE : {TRUE, FALSE};\n"
             "    esac;\n"
             "    DEFINE valid0 := counter < max_tokens;\n"
             "\n"
             "MODULE bool_input_inf(nReady0)\n"
             "    VAR dataOut0 : boolean;\n"
             "    \n"
             "    -- bool_input persistent\n"
             "    ASSIGN\n"
             "    next(dataOut0) := case \n"
             "      valid0 & !nReady0 : dataOut0;\n"
             "      TRUE : {TRUE, FALSE};\n"
             "    esac;\n"
             "    DEFINE valid0 := TRUE;\n"
             "#endif // BOOL_INPUT\n"
             "\n"
             "MODULE main\n";

  for (size_t i = 0; i < argNames.size(); ++i) {
    if (inf) {
      wrapper << "  VAR seq_generator" << i << " : bool_input_inf(in_ndw" << i
              << ".ready0);\n";
    } else {
      wrapper << "  VAR seq_generator" << i << " : bool_input(in_ndw" << i
              << ".ready0, " << n << ");\n";
    }
    wrapper << "  VAR in_ndw" << i << " : ndw_1_1(seq_generator" << i
            << ".dataOut0, seq_generator" << i << ".valid0, miter."
            << argNames[i] << "_ready);\n";
  }

  wrapper << "\n  " << createMiterCall(argNames, resNames) << "\n";
  wrapper << "  -- TODO make sure we have sink_1_0\n";

  for (size_t i = 0; i < resNames.size(); ++i) {
    wrapper << "  VAR out_ndw" << i << " : ndw_1_1(miter." << resNames[i]
            << "_out, miter." << resNames[i] << "_valid, sink" << i
            << ".ready0);\n";
    wrapper << "  VAR sink" << i << " : sink_1_0(out_ndw" << i
            << ".dataOut0, out_ndw" << i << ".valid0);\n";
  }

  return wrapper.str();
}

LogicalResult createMiterWrapper(const std::filesystem::path &wrapperPath,
                                 const std::filesystem::path &jsonPath,
                                 const std::string &modelSmvName,
                                 size_t nrOfTokens) {
  std::ifstream file(jsonPath);
  if (!file) {
    llvm::errs() << "Config JSON file not found\n";
    return failure();
  }

  std::stringstream buffer;
  buffer << file.rdbuf();
  file.close();

  llvm::Expected<llvm::json::Value> jsonValue = llvm::json::parse(buffer.str());
  if (!jsonValue) {
    llvm::errs() << "Failed parsing JSON\n";
    return failure();
  }

  llvm::json::Object *config = jsonValue->getAsObject();
  if (!config) {
    llvm::errs() << "Failed parsing JSON\n";
    return failure();
  }

  std::string output;
  output += "#include \"" + modelSmvName + "\"\n";
  output += "#ifndef BOOL_INPUT\n"
            "#define BOOL_INPUT\n"
            "MODULE bool_input(nReady0, max_tokens)\n"
            "  VAR dataOut0 : boolean;\n"
            "  VAR counter : 0..31;\n"
            "  ASSIGN\n"
            "    init(counter) := 0;\n"
            "    next(counter) := case\n"
            "      nReady0 & counter < max_tokens : counter + 1;\n"
            "      TRUE : counter;\n"
            "    esac;\n"
            "\n"
            "  -- bool_input persistent\n"
            "  ASSIGN\n"
            "    next(dataOut0) := case\n"
            "      valid0 & !nReady0 : dataOut0;\n"
            "      TRUE : {TRUE, FALSE};\n"
            "    esac;\n"
            "\n"
            "  DEFINE valid0 := counter < max_tokens;\n"
            "#endif // BOOL_INPUT\n"
            "\n"
            "MODULE main\n";

  if (auto *args = config->getArray("arguments")) {
    for (size_t i = 0; i < args->size(); ++i) {
      output += "VAR seq_generator" + std::to_string(i) +
                " : bool_input(miter." +
                (*args)[i].getAsString().value_or("").str() + "_ready, " +
                std::to_string(nrOfTokens) + ");\n";
    }
  } else {
    llvm::errs() << "No arguments in JSON\n";
    return failure();
  }
  output += "\n";

  output += "VAR miter : elastic_miter(";
  if (auto *args = config->getArray("arguments")) {
    for (size_t i = 0; i < args->size(); ++i) {
      if (i > 0)
        output += ", ";
      output += "seq_generator" + std::to_string(i) +
                ".dataOut0, seq_generator" + std::to_string(i) + ".valid0";
    }
  }
  output += ", ";
  if (auto *res = config->getArray("results")) {
    for (size_t i = 0; i < res->size(); ++i) {
      if (i > 0)
        output += ", ";
      output += "sink" + std::to_string(i) + ".ready0";
    }
  } else {
    llvm::errs() << "No results in JSON\n";
    return failure();
  }
  output += ");\n\n";

  output += "-- TODO make sure we have sink_1_0\n";
  if (auto *res = config->getArray("results")) {
    for (size_t i = 0; i < res->size(); ++i) {
      output += "VAR sink" + std::to_string(i) + " : sink_1_0(miter." +
                (*res)[i].getAsString().value_or("").str() + "_out, miter." +
                (*res)[i].getAsString().value_or("").str() + "_valid);\n";
    }
  }
  output += "\n";

  if (auto *res = config->getArray("results")) {
    for (const auto &result : *res) {
      std::string resStr = result.getAsString().value_or("").str();
      output += "CTLSPEC AG (miter." + resStr + "_valid -> miter." + resStr +
                "_out)\n";
    }
  }
  output += "\n";

  std::string outputProp;
  if (auto *outBufs = config->getArray("output_buffers")) {
    for (const auto &bufArray : *outBufs) {
      if (auto *bufList = bufArray.getAsArray()) {
        for (const auto &buf : *bufList) {
          outputProp +=
              "miter." + buf.getAsString().value_or("").str() + ".num = 0 & ";
        }
      }
    }
  }
  if (!outputProp.empty())
    outputProp = outputProp.substr(0, outputProp.size() - 3);

  std::string inputProp;
  if (auto *inBufs = config->getArray("input_buffers")) {
    for (const auto &bufPair : *inBufs) {
      if (auto *pair = bufPair.getAsArray()) {
        inputProp += "miter." + (*pair)[0].getAsString().value_or("").str() +
                     ".num = miter." +
                     (*pair)[1].getAsString().value_or("").str() + ".num & ";
      }
    }
  }
  if (!inputProp.empty())
    inputProp = inputProp.substr(0, inputProp.size() - 3);

  std::string finalBufferProp =
      "AF (AG (" + inputProp + " & " + outputProp + "))";
  output += "CTLSPEC " + finalBufferProp + "\n";

  std::ofstream mainFile(wrapperPath);
  mainFile << output;
  mainFile.close();

  return success();
}
} // namespace dynamatic::experimental