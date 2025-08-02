#!/bin/bash
set -e

DYNAMATIC_PATH=$1

DYNAMATIC_BINS=$DYNAMATIC_PATH/build/bin

LLVM=$DYNAMATIC_PATH/polygeist/llvm-project

LLVM_BINS=$LLVM/build/bin

# Example: "dynamatic/integration-test/fir/fir.c"
F_SRC=$2

# Example: "fir"
FUNC_NAME=$3

[ -f "$F_SRC" ] || { echo "$F_SRC is not a file!"; exit 1;}

# Will be change to standard path in the future (i.e., out/comp).
OUT="./out-$FUNC_NAME"
rm -rf $OUT
mkdir -p $OUT

# NOTE:
# - ffp-contract will prevent clang from adding "fused add mul" into the IR
# We need to check out the clang language extensions carefully for more
# optimizations, e.g., loop unrolling:
# https://clang.llvm.org/docs/LanguageExtensions.html#loop-unrolling
$LLVM_BINS/clang -O0 -S -emit-llvm $F_SRC \
  -I "$DYNAMATIC_PATH/include"  \
  -Xclang \
  -ffp-contract=off \
  -o $OUT/clang.ll

# NOTE:
# - When calling clang with "-ffp-contract=off", clang will bypass the
# "-disable-O0-optnone" flag and still adds "optnone" to the IR. This is a hacky
# way to ignore it
# - Please aware that there might be other attributes that need to be stripped:
# for example, noinline, nounwind, and uwtable ("noinline" seems like something
# we also need to strip).
sed -i "s/optnone//g" $OUT/clang.ll

# Strip information that we don't care (and mlir-translate also doesn't know how
# to handle it).
sed -i "s/^target datalayout = .*$//g" $OUT/clang.ll
sed -i "s/^target triple = .*$//g" $OUT/clang.ll

# NOTE:
# Here is a brief summary of what each llvm pass does:
# - mem2reg: Suppresses allocas (allocate memory on the heap) into regs
# - instcombine: combine operations. Needed to canonicalize a chain of GEPs
# - loop-rotate: canonicalize loops to do-while loops
# - consthoist: moving constants around
# - simplifycfg: merge BBs
#
# NOTE: the optnone attribute sliently disables all the optimization in the
# passes; Check out the complete list: https://llvm.org/docs/Passes.html
$LLVM_BINS/opt -S \
  -passes="mem2reg,instcombine,loop-rotate,consthoist,simplifycfg" \
  -strip-debug \
  $OUT/clang.ll \
  > $OUT/clang_optimized.ll

# This pass uses polyhedral and alias analysis to determine the dependency
# between memory operations.
#
# Example:
# ======== histogram.ll =========
#  %2 = load float, ptr %arrayidx4, align 4, !handshake.name !5
#  ...
#  store float %add, ptr %arrayidx6, align 4, !handshake.name !6 !dest.ops !7 
#  ...
# !5 = !{!"load1"}
# !6 = !{!"store!"}
# !7 = !{!5, !"1"} ; this means that the store must happen before the load, with
# a loop depth of 1
# ===============================
$LLVM_BINS/opt $OUT/clang_optimized.ll -S \
  -polly-process-unprofitable \
  -load-pass-plugin "$DYNAMATIC_PATH/build/tools/mem-dep-analysis/libMemDepAnalysis.so" \
  -passes="mem-dep-analysis" \
  > $OUT/clang_optimized_dep_marked.ll

$LLVM_BINS/mlir-translate \
  --import-llvm $OUT/clang_optimized_dep_marked.ll \
  > $OUT/clang_optimized_translated.mlir

# The llvm -> mlir translation does not carry the dependency information (and
# any meta data in general), therefore, the "--llvm-mark-memory-dependencies"
# post-processes the converted mlir file and put the dependency information
# there 
$DYNAMATIC_BINS/dynamatic-opt \
  $OUT/clang_optimized_translated.mlir \
  --remove-polygeist-attributes \
  --llvm-mark-memory-dependencies="llvmir=$OUT/clang_optimized_dep_marked.ll" \
  --allow-unregistered-dialect \
  > $OUT/clang_optimized_translated_dep_marked.mlir

# - drop-unlist-functions: Dropping the functions that are not needed in HLS
# compilation
$DYNAMATIC_BINS/dynamatic-opt \
  $OUT/clang_optimized_translated_dep_marked.mlir \
  --remove-polygeist-attributes \
  --drop-unlisted-functions="function-names=$FUNC_NAME" \
  > $OUT/clang_optimized_translated_droped_main_removed_attributes.mlir \

$DYNAMATIC_BINS/dynamatic-opt \
  $OUT/clang_optimized_translated_droped_main_removed_attributes.mlir \
  --convert-llvm-to-cf="source=$F_SRC dynamatic-path=$DYNAMATIC_PATH" \
  > $OUT/cf.mlir

$DYNAMATIC_BINS/dynamatic-opt \
  $OUT/cf.mlir \
  --func-set-arg-names="source=$F_SRC" \
  --flatten-memref-row-major \
  --canonicalize \
  --push-constants \
  --mark-memory-interfaces \
  > $OUT/cf_transformed.mlir

$DYNAMATIC_BINS/dynamatic-opt \
  $OUT/cf_transformed.mlir \
  --lower-cf-to-handshake \
  > $OUT/handshake.mlir
