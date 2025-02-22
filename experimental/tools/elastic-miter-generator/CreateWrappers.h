#ifndef DYNAMATIC_EXPERIMENTAL_ELASTIC_MITER_CREATE_WRAPPERS_H
#define DYNAMATIC_EXPERIMENTAL_ELASTIC_MITER_CREATE_WRAPPERS_H

#include "ElasticMiterFabricGeneration.h"
#include "dynamatic/Support/LLVM.h"
#include <filesystem>
#include <string>

using namespace mlir;

namespace dynamatic::experimental {

struct SequenceConstraints {
  std::string seqLengthRelationConstraint;
  SmallVector<size_t> loopSeqConstraint;
  SmallVector<size_t> loopStrictSeqConstraint;
  struct {
    size_t inputSequence;
    size_t outputSequence;
    size_t length = 0;
  } tokenLimitConstraint;
};

std::string
createModuleCall(const std::string &moduleName,
                 const SmallVector<std::string> &argNames,
                 const SmallVector<std::pair<std::string, Type>> &results);

FailureOr<std::string> createReachableStateWrapper(ModuleOp mlir, int n = 0,
                                                   bool inf = false);

LogicalResult createWrapper(const std::filesystem::path &wrapperPath,
                            const ElasticMiterConfig &config,
                            const std::string &modelSmvName, size_t nrOfTokens,
                            bool includeProperties,
                            const SequenceConstraints &sequenceConstraints);

const std::string BOOL_INPUT =
    "#ifndef BOOL_INPUT\n"
    "#define BOOL_INPUT\n"
    "MODULE bool_input(nReady0, max_tokens)\n"
    "  VAR dataOut0 : boolean;\n"
    "  VAR counter : 0..31;\n"
    "  FROZENVAR exact_tokens : 0..max_tokens;\n"
    "  ASSIGN\n"
    "    init(counter) := 0;\n"
    "    next(counter) := case\n"
    "      nReady0 & counter < exact_tokens : counter + 1;\n"
    "      TRUE : counter;\n"
    "    esac;\n"
    "\n"
    "  -- make sure dataOut0 is persistent\n"
    "  ASSIGN\n"
    "    next(dataOut0) := case\n"
    "      valid0 & !nReady0 : dataOut0;\n"
    "      TRUE : {TRUE, FALSE};\n"
    "    esac;\n"
    "\n"
    "  DEFINE valid0 := counter < exact_tokens;\n"
    "#endif // BOOL_INPUT\n\n";

const std::string BOOL_INPUT_INF = "MODULE bool_input_inf(nReady0)\n"
                                   "    VAR dataOut0 : boolean;\n"
                                   "    \n"
                                   "    -- make sure dataOut0 is persistent\n"
                                   "    ASSIGN\n"
                                   "    next(dataOut0) := case \n"
                                   "      valid0 & !nReady0 : dataOut0;\n"
                                   "      TRUE : {TRUE, FALSE};\n"
                                   "    esac;\n"
                                   "    DEFINE valid0 := TRUE;\n\n";
} // namespace dynamatic::experimental
#endif