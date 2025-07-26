#!/bin/bash
set -e

DYNAMATIC_PATH=$1

# Example: "dynamatic/integration-test/fir/fir.c"
F_SRC=$2

# Example: "fir"
FUNC_NAME=$3

DYNAMATIC_BINS=$DYNAMATIC_PATH/build/bin

LLVM=$DYNAMATIC_PATH/polygeist/llvm-project

LLVM_BINS=$LLVM/build/bin

[ -f "$F_SRC" ] || { echo "$F_SRC is not a file!"; exit 1;}

# Will be change to standard path in the future (i.e., out/comp).
OUT="./out-$FUNC_NAME"
rm -rf $OUT
mkdir -p $OUT

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
  -o $OUT/clang.ll

# ------------------------------------------------------------------------------
# NOTE:
# - When calling clang with "-ffp-contract=off", clang will bypass the
# "-disable-O0-optnone" flag and still adds "optnone" to the IR. This is a hacky
# way to ignore it
# - Please aware that there might be other attributes that need to be stripped:
# for example, noinline, nounwind, and uwtable ("noinline" seems like something
# we also need to strip).
# ------------------------------------------------------------------------------
sed -i "s/optnone//g" $OUT/clang.ll

# Strip information that we don't care (and mlir-translate also doesn't know how
# to handle it).
sed -i "s/^target datalayout = .*$//g" $OUT/clang.ll
sed -i "s/^target triple = .*$//g" $OUT/clang.ll

# ------------------------------------------------------------------------------
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
# ------------------------------------------------------------------------------

$LLVM_BINS/opt -S \
 -passes="mem2reg,consthoist,instcombine,simplifycfg,loop-rotate,simplifycfg" \
  $OUT/clang.ll \
  > $OUT/clang_loop_canonicalized.ll

# ------------------------------------------------------------------------------
# Example (how to unroll loops):
#
#// void test_unrolling(const int A[N], const int B[N], const int C[N], int result[N]) {
#//   // NOTE: The "array-partition" pass replicate this array to allow 2 concurrent accesses
#//   int intermediate[N];
#// #pragma clang loop unroll_count(2)
#//   for (int i = 0; i < N; i++) {
#//     intermediate[i] = A[i] * B[i];
#//   }
#// #pragma clang loop unroll_count(2)
#//   for (int i = 0; i < N; i++) {
#//     result[i] = intermediate[i] * C[i];
#//   }
#// }
#
# We use the clang pragma to tell the compiler to unroll the loop.
# 
# TODO: maybe some llvm passes can prove that some array locations are allocated
# by never used, so it automatically reshapes the array (and the accesses)?
# ------------------------------------------------------------------------------

$LLVM_BINS/opt -S \
  -passes="loop-unroll" \
  $OUT/clang_loop_canonicalized.ll \
  > $OUT/clang_unrolled.ll

$LLVM_BINS/opt -S \
  -passes="loop-simplify,simplifycfg" \
  -strip-debug \
  $OUT/clang_unrolled.ll \
  > $OUT/clang_optimized.ll

# ------------------------------------------------------------------------------
# This pass computes the set of disjoint accesses to the same baseptr, and
# replicate the arrays if disjoint sets can be found.
# Example: consider A[10] and loadA, loadB, loadC, loadD interact with it.
#
# - loadA:  accesses 0, 2, 4, 6, 8
# - loadB:  accesses 1, 3, 5, 7, 9
# - storeA: accesses 0, 2, 4, 6, 8
# - storeB: accesses 1, 3, 5, 7, 9
#
# We can parition it into {loadA, storeA} and {loadB, storeB}, such that you
# cannot find two insts in these two sets that access the same array element.
# ------------------------------------------------------------------------------

# Somehow here we need to canonicalized again, otherwise, Polly cannot recognize
# some Scops in some cases
$LLVM_BINS/opt -S \
  -mem2reg \
  -loop-simplify \
  -simplifycfg \
  -polly-canonicalize \
  $OUT/clang_optimized.ll \
  > $OUT/clang_optimized_polly_canonicalized.ll

$LLVM_BINS/opt -S \
  -load-pass-plugin "$DYNAMATIC_PATH/build/tools/array-partition/libArrayPartition.so" \
  -passes="array-partition" \
  $OUT/clang_optimized_polly_canonicalized.ll \
  > $OUT/clang_array_partitioned.ll

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
$LLVM_BINS/opt -S \
  -load-pass-plugin "$DYNAMATIC_PATH/build/tools/mem-dep-analysis/libMemDepAnalysis.so" \
  -passes="mem-dep-analysis" \
  $OUT/clang_array_partitioned.ll \
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

$DYNAMATIC_BINS/dynamatic-opt \
  $OUT/handshake.mlir \
  --handshake-analyze-lsq-usage --handshake-replace-memory-interfaces \
  --handshake-minimize-cst-width --handshake-optimize-bitwidths \
  --handshake-materialize --handshake-infer-basic-blocks \
  > $OUT/handshake_transformed.mlir

# ------------------------------------------------------------------------------
# Run simple buffer placement
# ------------------------------------------------------------------------------
$DYNAMATIC_BINS/dynamatic-opt \
  $OUT/handshake_transformed.mlir \
  --handshake-mark-fpu-impl="impl=flopoco" \
  --handshake-set-buffering-properties="version=fpga20" \
  --handshake-place-buffers="algorithm=on-merges timing-models=$DYNAMATIC_PATH/data/components.json" \
  > $OUT/handshake_buffered.mlir

# ------------------------------------------------------------------------------
# Run throughput-driven buffer placement (needs a valid Gurobi license)
# ------------------------------------------------------------------------------

# "$LLVM_BINS/clang++" "$F_SRC" \
#   -D PRINT_PROFILING_INFO -I "$DYNAMATIC_PATH/include" \
#   -Wno-deprecated -o "$OUT/profiler_bin.exe"

# "$OUT/profiler_bin.exe" \
#   > "$OUT/profiler.txt"

# "$DYNAMATIC_BINS/exp-frequency-profiler" \
#   "$OUT/cf_transformed.mlir" \
#   --top-level-function="$FUNC_NAME" \
#   --input-args-file="$OUT/profiler.txt" \
#   > "$OUT/frequencies.csv"

# $DYNAMATIC_BINS/dynamatic-opt \
#   $OUT/handshake_transformed.mlir \
#   --handshake-mark-fpu-impl="impl=flopoco" \
#   --handshake-set-buffering-properties="version=fpga20" \
#   --handshake-place-buffers="algorithm=fpga20 frequencies=$OUT/frequencies.csv timing-models=$DYNAMATIC_PATH/data/components.json target-period=8 timeout=300" \
#   > $OUT/handshake_buffered.mlir

$DYNAMATIC_BINS/dynamatic-opt \
  $OUT/handshake_buffered.mlir \
  --handshake-canonicalize \
  --handshake-hoist-ext-instances \
  > $OUT/handshake_export.mlir

$DYNAMATIC_BINS/dynamatic-opt \
  $OUT/handshake_export.mlir \
  --lower-handshake-to-hw \
  > $OUT/hw.mlir

"$DYNAMATIC_BINS/export-rtl" \
  "$OUT/hw.mlir" "$OUT/hdl" "$DYNAMATIC_PATH/data/rtl-config-vhdl.json" \
  --dynamatic-path "$DYNAMATIC_PATH" --hdl vhdl

bash "$DYNAMATIC_PATH/tools/frontend/cosim.sh" \
  "$DYNAMATIC_PATH" \
  "$F_SRC" \
  "$FUNC_NAME" \
  "$OUT" \
  "$OUT/sim"
