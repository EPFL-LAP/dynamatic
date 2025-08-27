#!/bin/bash
set -e

DYNAMATIC_PATH=$1

# Example: "dynamatic/integration-test/fir/fir.c"
F_SRC=$2

# Example: "fir"
FUNC_NAME=$3

# Example: "dynamatic/integration-test/fir"
OUTPUT_DIR=$4

DYNAMATIC_BINS=$DYNAMATIC_PATH/build/bin

LLVM=$DYNAMATIC_PATH/polygeist/llvm-project

LLVM_BINS=$LLVM/build/bin

export PATH=$PATH:$LLVM_BINS

[ -f "$F_SRC" ] || { echo "$F_SRC is not a file!"; exit 1;}

mkdir -p $OUTPUT_DIR

COMP_DIR="$OUTPUT_DIR/comp"

rm -rf "$COMP_DIR"
mkdir -p "$COMP_DIR"

HDL_DIR="$OUTPUT_DIR/hdl"

rm -rf $HDL_DIR/*

# ------------------------------------------------------------------------------
# NOTE:
# - ffp-contract will prevent clang from adding "fused add mul" into the IR
# We need to check out the clang language extensions carefully for more
# optimizations, e.g., loop unrolling:
# https://clang.llvm.org/docs/LanguageExtensions.html#loop-unrolling
# ------------------------------------------------------------------------------
$LLVM_BINS/clang -O0 -funroll-loops -S -emit-llvm $F_SRC \
  -I "$DYNAMATIC_PATH/include"  \
  -Xclang \
  -ffp-contract=off \
  -o "$COMP_DIR/clang.ll"

# ------------------------------------------------------------------------------
# NOTE:
# - When calling clang with "-ffp-contract=off", clang will bypass the
# "-disable-O0-optnone" flag and still adds "optnone" to the IR. This is a hacky
# way to ignore it
# - Clang always adds "noinline" to the IR.
# ------------------------------------------------------------------------------
sed -i "s/optnone//g" "$COMP_DIR/clang.ll"
sed -i "s/noinline//g" "$COMP_DIR/clang.ll"

# Strip information that we don't care (and mlir-translate also doesn't know how
# to handle it).
sed -i "s/^target datalayout = .*$//g" "$COMP_DIR/clang.ll"
sed -i "s/^target triple = .*$//g" "$COMP_DIR/clang.ll"

# ------------------------------------------------------------------------------
# NOTE:
# Here is a brief summary of what each llvm pass does:
# - inline: Inlines the function calls.
# - mem2reg: Promote allocas (allocate memory on the heap) into regs.
# - lowerswitch: Convert switch case into branches.
# - instcombine: combine operations. Needed to canonicalize a chain of GEPs.
# - loop-rotate: canonicalize loops to do-while loops
# - consthoist: moving constants around
# - simplifycfg: merge BBs
#
# NOTE: the optnone attribute sliently disables all the optimization in the
# passes; Check out the complete list: https://llvm.org/docs/Passes.html
# ------------------------------------------------------------------------------

$LLVM_BINS/opt -S \
 -passes="inline,mem2reg,consthoist,instcombine,simplifycfg,loop-rotate,simplifycfg,lowerswitch,simplifycfg" \
  "$COMP_DIR/clang.ll" \
  > "$COMP_DIR/clang_loop_canonicalized.ll"

# ------------------------------------------------------------------------------
# Example (how to unroll loops):
#
# void test_unrolling(const int A[N], const int B[N], const int C[N], int result[N]) {
#   // NOTE: The "array-partition" pass replicate this array to allow 2 concurrent accesses
#   int intermediate[N];
# #pragma clang loop unroll_count(2)
#   for (int i = 0; i < N; i++) {
#     intermediate[i] = A[i] * B[i];
#   }
# #pragma clang loop unroll_count(2)
#   for (int i = 0; i < N; i++) {
#     result[i] = intermediate[i] * C[i];
#   }
# }
#
# We use the clang pragma to tell the compiler to unroll the loop.
# 
# TODO: maybe some llvm passes can prove that some array locations are allocated
# by never used, so it automatically reshapes the array (and the accesses)?
# ------------------------------------------------------------------------------

## # TODO: Loop unrolling fully unrolls test_memory_12 and creates a need for
## # an LSQ (the LSQ doesn't have a load port, which is a unsupported situation)
## $LLVM_BINS/opt -S \
##   -passes="loop-unroll" \
##   "$COMP_DIR/clang_loop_canonicalized.ll" \
##   > "$COMP_DIR/clang_unrolled.ll"
## 
## $LLVM_BINS/opt -S \
##   -passes="loop-simplify,simplifycfg" \
##   -strip-debug \
##   "$COMP_DIR/clang_unrolled.ll" \
##   > "$COMP_DIR/clang_optimized.ll"

# ------------------------------------------------------------------------------
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
# ------------------------------------------------------------------------------

# NOTE:
# - without "--polly-process-unprofitable", polly ignores certain small loops
# - ArrayParititon pass currently breaks the SCoP analysis in Polly. Therefore,
# we need to first attach analysis results to memory ops and then apply memory
# bank partition.
$LLVM_BINS/opt -S \
  -load-pass-plugin "$DYNAMATIC_PATH/build/tools/mem-dep-analysis/libMemDepAnalysis.so" \
  -passes="mem-dep-analysis" \
  -polly-process-unprofitable \
  "$COMP_DIR/clang_loop_canonicalized.ll" \
  > "$COMP_DIR/clang_optimized_dep_marked.ll"

## # ------------------------------------------------------------------------------
## # This pass computes the set of disjoint accesses to the same baseptr, and
## # replicate the arrays if disjoint sets can be found.
## # Example: consider A[10] and loadA, loadB, loadC, loadD interact with it.
## #
## # - loadA:  accesses 0, 2, 4, 6, 8
## # - loadB:  accesses 1, 3, 5, 7, 9
## # - storeA: accesses 0, 2, 4, 6, 8
## # - storeB: accesses 1, 3, 5, 7, 9
## #
## # We can parition it into {loadA, storeA} and {loadB, storeB}, such that you
## # cannot find two insts in these two sets that access the same array element.
## # ------------------------------------------------------------------------------
## 
## # NOTE: without "--polly-process-unprofitable", polly ignores certain small loops
## $LLVM_BINS/opt -S \
##   -load-pass-plugin "$DYNAMATIC_PATH/build/tools/array-partition/libArrayPartition.so" \
##   -polly-process-unprofitable \
##   -passes="array-partition" \
##   -debug -debug-only="array-partition" \
##   $COMP_DIR/clang_optimized_dep_marked.ll \
##   > $COMP_DIR/clang_array_partitioned.ll
## 
## # Clean up the index calculation logic inserted by the array-partition pass
## $LLVM_BINS/opt -S \
##   -passes="instcombine" \
##   $COMP_DIR/clang_array_partitioned.ll \
##   > $COMP_DIR/clang_array_partitioned_cleaned.ll

$DYNAMATIC_BINS/translate-llvm-to-std \
  "$COMP_DIR/clang_optimized_dep_marked.ll" \
  -function-name "$FUNC_NAME" \
  -csource "$F_SRC" \
  -dynamatic-path "$DYNAMATIC_PATH" \
  -debug -debug-only="translate-llvm-ir-to-std" \
   -o "$COMP_DIR/cf.mlir"

# - drop-unlist-functions: Dropping the functions that are not needed in HLS
# compilation
$DYNAMATIC_BINS/dynamatic-opt \
  "$COMP_DIR/cf.mlir" \
  --drop-unlisted-functions="function-names=$FUNC_NAME" \
  > "$COMP_DIR/cf_drop_unlisted_functions.mlir" \

$DYNAMATIC_BINS/dynamatic-opt \
  "$COMP_DIR/cf_drop_unlisted_functions.mlir" \
  --func-set-arg-names="source=$F_SRC" \
  --flatten-memref-row-major \
  --canonicalize \
  --push-constants \
  --mark-memory-interfaces \
  > "$COMP_DIR/cf_transformed.mlir"

$DYNAMATIC_BINS/dynamatic-opt \
  "$COMP_DIR/cf_transformed.mlir" \
  --lower-cf-to-handshake \
  > "$COMP_DIR/handshake.mlir"

$DYNAMATIC_BINS/dynamatic-opt \
  "$COMP_DIR/handshake.mlir" \
  --handshake-analyze-lsq-usage --handshake-replace-memory-interfaces \
  --handshake-minimize-cst-width --handshake-optimize-bitwidths \
  --handshake-materialize --handshake-infer-basic-blocks \
  > "$COMP_DIR/handshake_transformed.mlir"

# ------------------------------------------------------------------------------
# Run simple buffer placement
# ------------------------------------------------------------------------------
$DYNAMATIC_BINS/dynamatic-opt \
  $COMP_DIR/handshake_transformed.mlir \
  --handshake-mark-fpu-impl="impl=flopoco" \
  --handshake-set-buffering-properties="version=fpga20" \
  --handshake-place-buffers="algorithm=on-merges timing-models=$DYNAMATIC_PATH/data/components.json" \
  > $COMP_DIR/handshake_buffered.mlir

# ------------------------------------------------------------------------------
# Run throughput-driven buffer placement (needs a valid Gurobi license)
# ------------------------------------------------------------------------------

## "$LLVM_BINS/clang++" "$F_SRC" \
##   -D PRINT_PROFILING_INFO -I "$DYNAMATIC_PATH/include" \
##   -Wno-deprecated -o "$COMP_DIR/profiler_bin.exe"
## 
## "$COMP_DIR/profiler_bin.exe" \
##   > "$COMP_DIR/profiler.txt"
## 
## "$DYNAMATIC_BINS/exp-frequency-profiler" \
##   "$COMP_DIR/cf_transformed.mlir" \
##   --top-level-function="$FUNC_NAME" \
##   --input-args-file="$COMP_DIR/profiler.txt" \
##   > "$COMP_DIR/frequencies.csv"
## 
## $DYNAMATIC_BINS/dynamatic-opt \
##   "$COMP_DIR/handshake_transformed.mlir" \
##   --handshake-mark-fpu-impl="impl=flopoco" \
##   --handshake-set-buffering-properties="version=fpga20" \
##   --handshake-place-buffers="algorithm=fpga20 frequencies=$COMP_DIR/frequencies.csv timing-models=$DYNAMATIC_PATH/data/components.json target-period=8 timeout=30" \
##   > "$COMP_DIR/handshake_buffered.mlir"

$DYNAMATIC_BINS/dynamatic-opt \
  "$COMP_DIR/handshake_buffered.mlir" \
  --handshake-canonicalize \
  --handshake-hoist-ext-instances \
  > "$COMP_DIR/handshake_export.mlir"

$DYNAMATIC_BINS/dynamatic-opt \
  "$COMP_DIR/handshake_export.mlir" \
  --lower-handshake-to-hw \
  > "$COMP_DIR/hw.mlir"

"$DYNAMATIC_BINS/export-rtl" \
  "$COMP_DIR/hw.mlir" "$HDL_DIR" "$DYNAMATIC_PATH/data/rtl-config-vhdl.json" \
  --dynamatic-path "$DYNAMATIC_PATH" --hdl vhdl

bash "$DYNAMATIC_PATH/tools/frontend/cosim.sh" \
  "$DYNAMATIC_PATH" \
  "$F_SRC" \
  "$FUNC_NAME" \
  "$OUTPUT_DIR" \
  "$OUTPUT_DIR/sim"
