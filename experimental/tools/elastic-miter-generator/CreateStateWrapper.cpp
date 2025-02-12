#include "mlir/IR/Attributes.h"
#include <iostream>
#include <llvm/ADT/StringSet.h>
#include <sstream>
#include <string>

#include "../experimental/tools/elastic-miter-generator/ElasticMiterFabricGeneration.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Transforms/HandshakeMaterialize.h"

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

FailureOr<std::string> createStateWrapper(const std::string &smv, ModuleOp mlir,
                                          int n = 0, bool inf = false) {

  auto failOrFuncOp = dynamatic::experimental::getModuleFuncOpAndCheck(mlir);
  if (failed(failOrFuncOp))
    return failure();

  FuncOp funcOp = failOrFuncOp.value();

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
  wrapper << "#include \"" << smv << "\"\n";
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
} // namespace dynamatic::experimental