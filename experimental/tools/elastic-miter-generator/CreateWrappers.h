
#include "mlir/IR/BuiltinOps.h"
#include <any>
#include <filesystem>
#include <string>

using namespace mlir;

namespace dynamatic::experimental {

std::string createModuleCall(const std::string &moduleName,
                             const SmallVector<std::string> &argNames,
                             const SmallVector<std::string> &resNames);

FailureOr<std::string> createReachableStateWrapper(ModuleOp mlir, int n = 0,
                                                   bool inf = false);

LogicalResult createWrapper(const std::filesystem::path &wrapperPath,
                            llvm::StringMap<std::any> config,
                            const std::string &modelSmvName, size_t nrOfTokens,
                            bool includeProperties = false);

const std::string BOOL_INPUT =
    "#ifndef BOOL_INPUT\n"
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
    "#endif // BOOL_INPUT\n\n";

const std::string BOOL_INPUT_INF = "MODULE bool_input_inf(nReady0)\n"
                                   "    VAR dataOut0 : boolean;\n"
                                   "    \n"
                                   "    -- bool_input persistent\n"
                                   "    ASSIGN\n"
                                   "    next(dataOut0) := case \n"
                                   "      valid0 & !nReady0 : dataOut0;\n"
                                   "      TRUE : {TRUE, FALSE};\n"
                                   "    esac;\n"
                                   "    DEFINE valid0 := TRUE;\n";
} // namespace dynamatic::experimental