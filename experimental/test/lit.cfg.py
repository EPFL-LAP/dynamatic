# -*- Python -*-

import os

import lit.formats
import lit.util

from lit.llvm import llvm_config

# Configuration file for the 'lit' test runner.

# name: The name of this test suite.
config.name = "DynamaticExperimental"

# Run the RUN lines with lit's internal shell rather than an external one (e.g.,
# bash): lit deprecates the external shell as of LLVM 23 and removes it in
# LLVM 24
config.test_format = lit.formats.ShTest(execute_external=False)

# suffixes: A list of file extensions to treat as test files.
config.suffixes = [".mlir"]

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.dynamatic_obj_root, "test")

config.substitutions.append(("%PATH%", config.environment["PATH"]))
config.substitutions.append(("%shlibext", config.llvm_shlib_ext))
config.substitutions.append(("%shlibdir", config.dynamatic_shlib_dir))

llvm_config.with_system_environment(["HOME", "INCLUDE", "LIB", "TMP", "TEMP"])

llvm_config.use_default_substitutions()

# excludes: A list of directories to exclude from the testsuite.
config.excludes = ["CMakeLists.txt", "README.md"]

# Tweak the PATH to include the tools dir.
llvm_config.with_environment("PATH", config.llvm_tools_dir, append_path=True)

tool_dirs = [config.dynamatic_tools_dir,
             config.mlir_tools_dir, config.llvm_tools_dir]
tools = ["dynamatic-opt"]

llvm_config.add_tool_substitutions(tools, tool_dirs)
