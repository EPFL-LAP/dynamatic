# -*- Python -*-

import os

import lit.formats
import lit.util

from lit.llvm import llvm_config
from lit.llvm.subst import ToolSubst

# Configuration file for the 'lit' test runner.

# name: The name of this test suite.
config.name = "Dynamatic"

config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)

# suffixes: A list of file extensions to treat as test files.
config.suffixes = [".mlir", ".ll"]

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
if config.cmake_build_type == "Release":
    print("[WARNING] Skipping `invalid.mlir` in Release mode")
    config.excludes.append("invalid.mlir")

# Tweak the PATH to include the tools dir.
llvm_config.with_environment("PATH", config.llvm_tools_dir, append_path=True)

tool_dirs = [config.dynamatic_tools_dir,
             config.mlir_tools_dir, config.llvm_tools_dir]
tools = ["dynamatic-opt", "hls-fuzzer-check-bitwidth",
         ToolSubst("%source-rewriter",
                   command=f"cp %s %t.c && {config.dynamatic_tools_dir}/source-rewriter %t.c --"),
         ToolSubst("%export-vhdl",
                   command=f"rm -rf %t; mkdir %t; {config.dynamatic_tools_dir}/export-rtl %s %t {config.dynamatic_src_root}/data/rtl-config-vhdl.json --dynamatic-path {config.dynamatic_src_root} --hdl vhdl"),
         ToolSubst("%translate-llvm-to-std",
                   command=f"{config.llvm_tools_dir}/split-file %s %t; {config.dynamatic_tools_dir}/translate-llvm-to-std %t/test.ll -csource %t/test.c -function-name=test --dynamatic-path {config.dynamatic_src_root}"),
         ToolSubst("%dyn-clang-pragmas",
                   command=f"{config.llvm_tools_dir}/clang "
                   f"-fplugin={config.dynamatic_shlib_dir}/DynPragmasPlugin{config.llvm_shlib_ext} "
                   f"-O2 -emit-llvm -S -o - 2>&1")]

llvm_config.add_tool_substitutions(tools, tool_dirs)
