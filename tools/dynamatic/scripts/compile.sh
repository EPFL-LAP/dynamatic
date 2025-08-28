#!/bin/bash

source "$1"/tools/dynamatic/scripts/utils.sh

# ============================================================================ #
# Variable definitions
# ============================================================================ #

# Script arguments
DYNAMATIC_DIR=$1
SRC_DIR=$2
OUTPUT_DIR=$3
KERNEL_NAME=$4
BUFFER_ALGORITHM=$5
TARGET_CP=$6
USE_SHARING=$7
FPUNITS_GEN=$8
USE_RIGIDIFICATION=${9}
DISABLE_LSQ=${10}
FAST_TOKEN_DELIVERY=${11}

LLVM=$DYNAMATIC_DIR/polygeist/llvm-project
LLVM_BINS=$LLVM/build/bin
export PATH=$PATH:$LLVM_BINS

POLYGEIST_CLANG_BIN="$DYNAMATIC_DIR/bin/cgeist"
CLANGXX_BIN="$DYNAMATIC_DIR/bin/clang++"
LLVM_OPT="$LLVM_BINS/opt"
LLVM_TO_STD_TRANSLATION_BIN="$DYNAMATIC_DIR/build/bin/translate-llvm-to-std"
DYNAMATIC_OPT_BIN="$DYNAMATIC_DIR/bin/dynamatic-opt"
DYNAMATIC_PROFILER_BIN="$DYNAMATIC_DIR/bin/exp-frequency-profiler"
DYNAMATIC_EXPORT_DOT_BIN="$DYNAMATIC_DIR/bin/export-dot"
DYNAMATIC_EXPORT_CFG_BIN="$DYNAMATIC_DIR/bin/export-cfg"

RIGIDIFICATION_SH="$DYNAMATIC_DIR/experimental/tools/rigidification/rigidification.sh"

# Generated directories/files
COMP_DIR="$OUTPUT_DIR/comp"

F_C_SOURCE="$SRC_DIR/$KERNEL_NAME.c" 

F_CLANG="$COMP_DIR/clang.ll"
F_CLANG_OPTIMIZED="$COMP_DIR/clang.opt.ll"
F_CLANG_OPTIMIZED_DEPENDENCY="$COMP_DIR/clang.opt.dep.ll"

F_CF="$COMP_DIR/cf.mlir"
F_CF_TRANSFORMED="$COMP_DIR/cf_transformed.mlir"
F_CF_DYN_TRANSFORMED_MEM_DEP_MARKED="$COMP_DIR/cf_transformed_mem_interface_marked.mlir"
F_PROFILER_BIN="$COMP_DIR/$KERNEL_NAME-profile"
F_PROFILER_INPUTS="$COMP_DIR/profiler-inputs.txt"
F_HANDSHAKE="$COMP_DIR/handshake.mlir"
F_HANDSHAKE_TRANSFORMED="$COMP_DIR/handshake_transformed.mlir"
F_HANDSHAKE_BUFFERED="$COMP_DIR/handshake_buffered.mlir"
F_HANDSHAKE_EXPORT="$COMP_DIR/handshake_export.mlir"
F_HANDSHAKE_RIGIDIFIED="$COMP_DIR/handshake_rigidified.mlir"
F_HW="$COMP_DIR/hw.mlir"
F_FREQUENCIES="$COMP_DIR/frequencies.csv"

# ============================================================================ #
# Helper funtions
# ============================================================================ #

# Exports Handshake-level IR to DOT using Dynamatic, then converts the DOT to
# a PNG using dot.
#   $1: input handshake-level IR filename
#   $1: output filename, without extension (will use .dot and .png)
export_dot() {
  local f_handshake="$1"
  local f_dot="$COMP_DIR/$2.dot"
  local f_png="$COMP_DIR/$2.png"

  # Export to DOT
  "$DYNAMATIC_EXPORT_DOT_BIN" "$f_handshake" "--edge-style=spline" \
    > "$f_dot"
  exit_on_fail "Failed to create $2 DOT" "Created $2 DOT"

  # Convert DOT graph to PNG
  dot -Tpng "$f_dot" > "$f_png"
  exit_on_fail "Failed to convert $2 DOT to PNG" "Converted $2 DOT to PNG"
  return 0
}

export_cfg() {
  local f_cf="$1"
  local f_dot="$COMP_DIR/$2.dot"
  local f_png="$COMP_DIR/$2.png"


  # Export to DOT
  "$DYNAMATIC_EXPORT_CFG_BIN" "$f_cf" \
    > "$f_dot"
  exit_on_fail "Failed to create $2 DOT" "Created $2 DOT"

  # Convert DOT graph to PNG
  dot -Tpng "$f_dot" > "$f_png"
  exit_on_fail "Failed to convert $2 DOT to PNG" "Converted $2 DOT to PNG"
  return 0
}

# ============================================================================ #
# Compilation flow
# ============================================================================ #

# Reset output directory
rm -rf "$COMP_DIR" && mkdir -p "$COMP_DIR"

# ------------------------------------------------------------------------------
# NOTE:
# - ffp-contract will prevent clang from adding "fused add mul" into the IR
# We need to check out the clang language extensions carefully for more
# optimizations, e.g., loop unrolling:
# https://clang.llvm.org/docs/LanguageExtensions.html#loop-unrolling
# ------------------------------------------------------------------------------
$LLVM_BINS/clang -O0 -funroll-loops -S -emit-llvm "$F_C_SOURCE" \
  -I "$DYNAMATIC_DIR/include"  \
  -Xclang \
  -ffp-contract=off \
  -o "$F_CLANG"

exit_on_fail "Failed to compile to LLVM IR" \
  "Compiled to LLVM IR"

# ------------------------------------------------------------------------------
# NOTE:
# - When calling clang with "-ffp-contract=off", clang will bypass the
# "-disable-O0-optnone" flag and still adds "optnone" to the IR. This is a hacky
# way to ignore it
# - Clang always adds "noinline" to the IR.
# ------------------------------------------------------------------------------
sed -i "s/optnone//g" "$F_CLANG"
sed -i "s/noinline//g" "$F_CLANG"

# Strip information that we don't care (and mlir-translate also doesn't know how
# to handle it).
sed -i "s/^target datalayout = .*$//g" "$F_CLANG"
sed -i "s/^target triple = .*$//g" "$F_CLANG"

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
  "$F_CLANG" \
  > "$F_CLANG_OPTIMIZED"
exit_on_fail "Failed to apply optimization to LLVM IR" \
  "Optimized LLVM IR"

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
#
# ------------------------------------------------------------------------------
# NOTE:
# - without "--polly-process-unprofitable", polly ignores certain small loops
# - ArrayParititon pass currently breaks the SCoP analysis in Polly. Therefore,
# we need to first attach analysis results to memory ops and then apply memory
# bank partition.
$LLVM_BINS/opt -S \
  -load-pass-plugin "$DYNAMATIC_DIR/build/lib/MemDepAnalysis.so" \
  -passes="mem-dep-analysis" \
  -polly-process-unprofitable \
  "$F_CLANG_OPTIMIZED" \
  > "$F_CLANG_OPTIMIZED_DEPENDENCY"
exit_on_fail "Failed to apply memory dependency analysis to LLVM IR" \
  "Applied memory dependency analysis to LLVM IR"

$LLVM_TO_STD_TRANSLATION_BIN \
  "$F_CLANG_OPTIMIZED_DEPENDENCY" \
  -function-name "$KERNEL_NAME" \
  -csource "$F_C_SOURCE" \
  -dynamatic-path "$DYNAMATIC_DIR" \
   -o "$F_CF"
exit_on_fail "Failed to convert to std dialect" \
  "Converted to std dialect"

# cf transformations (dynamatic)
# - drop-unlist-functions: Dropping the functions that are not needed in HLS
# compilation
$DYNAMATIC_OPT_BIN \
  "$F_CF" \
  --drop-unlisted-functions="function-names=$KERNEL_NAME" \
  --func-set-arg-names="source=$F_C_SOURCE" \
  --flatten-memref-row-major \
  --canonicalize \
  --push-constants \
  --mark-memory-interfaces \
  > "$F_CF_TRANSFORMED"
exit_on_fail "Failed to apply CF transformations" \
  "Applied CF transformations"

if [[ $DISABLE_LSQ -ne 0 ]]; then
  "$DYNAMATIC_OPT_BIN" "$F_CF_TRANSFORMED" \
    --force-memory-interface="force-mc=true" \
    > "$F_CF_DYN_TRANSFORMED_MEM_DEP_MARKED"
  exit_on_fail "Failed to force usage of MC interface" \
    "Forced usage of MC interface in cf"
else
  "$DYNAMATIC_OPT_BIN" "$F_CF_TRANSFORMED" \
    --mark-memory-interfaces \
    > "$F_CF_DYN_TRANSFORMED_MEM_DEP_MARKED"
  exit_on_fail "Failed to mark memory interfaces in cf" \
    "Marked memory accesses with the corresponding interfaces in cf"
fi

# cf level -> handshake level
if [[ $FAST_TOKEN_DELIVERY -ne 0 ]]; then
  echo_info "Running FTD algorithm for handshake conversion"
  "$DYNAMATIC_OPT_BIN" "$F_CF_DYN_TRANSFORMED_MEM_DEP_MARKED" \
    --ftd-lower-cf-to-handshake \
    --handshake-combine-steering-logic \
    > "$F_HANDSHAKE"
  exit_on_fail "Failed to compile cf to handshake with FTD" "Compiled cf to handshake with FTD"
else
  "$DYNAMATIC_OPT_BIN" "$F_CF_DYN_TRANSFORMED_MEM_DEP_MARKED" --lower-cf-to-handshake \
    > "$F_HANDSHAKE"
  exit_on_fail "Failed to compile cf to handshake" "Compiled cf to handshake"
fi

# handshake transformations
"$DYNAMATIC_OPT_BIN" "$F_HANDSHAKE" \
  --handshake-analyze-lsq-usage --handshake-replace-memory-interfaces \
  --handshake-minimize-cst-width --handshake-optimize-bitwidths \
  --handshake-materialize --handshake-infer-basic-blocks \
  > "$F_HANDSHAKE_TRANSFORMED"
exit_on_fail "Failed to apply transformations to handshake" \
  "Applied transformations to handshake"

# Credit-based sharing
if [[ $USE_SHARING -ne 0 ]]; then
  # NOTE: to use this in dynamatic-opt, do ${SHARING_PASS:+"$SHARING_PASS"} to
  # conditionally pass the string as an argument if not empty.
  SHARING_PASS="--credit-based-sharing=timing-models=$DYNAMATIC_DIR/data/components.json target-period=$TARGET_CP"
  echo_info "Set to apply credit-based sharing after buffer placement."
fi

# Buffer placement
if [[ "$BUFFER_ALGORITHM" == "on-merges" ]]; then
  # Simple buffer placement
  echo_info "Running simple buffer placement (on-merges)."
  "$DYNAMATIC_OPT_BIN" "$F_HANDSHAKE_TRANSFORMED" \
    --handshake-mark-fpu-impl="impl=$FPUNITS_GEN" \
    --handshake-set-buffering-properties="version=fpga20" \
    --handshake-place-buffers="algorithm=$BUFFER_ALGORITHM timing-models=$DYNAMATIC_DIR/data/components.json" \
    ${SHARING_PASS:+"$SHARING_PASS"} \
    > "$F_HANDSHAKE_BUFFERED"
  exit_on_fail "Failed to place simple buffers" "Placed simple buffers"
else
  # Compile kernel's main function to extract profiling information
  "$CLANGXX_BIN" "$SRC_DIR/$KERNEL_NAME.c" -D PRINT_PROFILING_INFO -I \
    "$DYNAMATIC_DIR/include" -Wno-deprecated -o "$F_PROFILER_BIN"
  exit_on_fail "Failed to build kernel for profiling" "Built kernel for profiling"

  "$F_PROFILER_BIN" > "$F_PROFILER_INPUTS"
  exit_on_fail "Failed to kernel for profiling" "Ran kernel for profiling"

  # cf-level profiler
  "$DYNAMATIC_PROFILER_BIN" "$F_CF_DYN_TRANSFORMED_MEM_DEP_MARKED" \
    --top-level-function="$KERNEL_NAME" --input-args-file="$F_PROFILER_INPUTS" \
    > $F_FREQUENCIES
  exit_on_fail "Failed to profile cf-level" "Profiled cf-level"

  # Smart buffer placement
  echo_info "Running smart buffer placement with CP = $TARGET_CP and algorithm = '$BUFFER_ALGORITHM'"
  cd "$COMP_DIR"
  "$DYNAMATIC_OPT_BIN" "$F_HANDSHAKE_TRANSFORMED" \
    --handshake-mark-fpu-impl="impl=$FPUNITS_GEN" \
    --handshake-set-buffering-properties="version=fpga20" \
    --handshake-place-buffers="algorithm=$BUFFER_ALGORITHM frequencies=$F_FREQUENCIES timing-models=$DYNAMATIC_DIR/data/components.json target-period=$TARGET_CP timeout=300 dump-logs \
    blif-files=$DYNAMATIC_DIR/data/aig/ lut-delay=0.55 lut-size=6 acyclic-type" \
    ${SHARING_PASS:+"$SHARING_PASS"} \
    > "$F_HANDSHAKE_BUFFERED"
  exit_on_fail "Failed to place smart buffers" "Placed smart buffers"
  cd - > /dev/null
fi

# handshake canonicalization
"$DYNAMATIC_OPT_BIN" "$F_HANDSHAKE_BUFFERED" \
  --handshake-canonicalize \
  --handshake-hoist-ext-instances \
  > "$F_HANDSHAKE_EXPORT"
exit_on_fail "Failed to canonicalize Handshake" "Canonicalized handshake"

# Export to DOT
export_dot "$F_HANDSHAKE_EXPORT" "$KERNEL_NAME"
export_cfg "$F_CF_TRANSFORMED" "${KERNEL_NAME}_CFG"

if [[ $USE_RIGIDIFICATION -ne 0 ]]; then
  # rigidification
  bash $RIGIDIFICATION_SH $DYNAMATIC_DIR $OUTPUT_DIR $KERNEL_NAME $F_HANDSHAKE_EXPORT \
    > "$F_HANDSHAKE_RIGIDIFIED"
  exit_on_fail "Failed to rigidify" "Rigidification completed"

  # handshake level -> hw level
  "$DYNAMATIC_OPT_BIN" "$F_HANDSHAKE_RIGIDIFIED" --lower-handshake-to-hw \
    > "$F_HW"
  exit_on_fail "Failed to lower to HW" "Lowered to HW"
else
  # handshake level -> hw level
  "$DYNAMATIC_OPT_BIN" "$F_HANDSHAKE_EXPORT" --lower-handshake-to-hw \
    > "$F_HW"
  exit_on_fail "Failed to lower to HW" "Lowered to HW"
fi

echo_info "Compilation succeeded"
