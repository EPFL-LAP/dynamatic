#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DYNAMATIC_DIR="$(cd "$SCRIPT_DIR" && pwd)"
cd "$DYNAMATIC_DIR"


# =====================================================================
# EXPERIMENT SET DEFINITIONS
# =====================================================================

# Default for kernels not overridden in your Python dict
default_experiments=(baseline_7ns v1_7ns auto_7ns)

# bisection, collision_donut → no token prediction (no v1)
no_predict_experiments=(baseline_7ns auto_7ns)


# sparse transformed → eager experiment is "8_7ns"
sparse_experiments=(baseline_7ns v1_7ns 8_7ns)

# if_float / if_float2
if_float_experiments=(baseline_7ns v1_7ns 0_two_sided_7ns 0_one_sided_7ns)

# unrolled kernels
single_loop_unrolled_160_experiments=(baseline_7ns auto_7ns)
bisection_unrolled_16_experiments=(baseline_7ns auto_7ns)
kmp_experiments=(baseline_7ns auto_7ns)


# =====================================================================
# KERNEL LIST
# =====================================================================

kernels=(
  single_loop
  nested_loop
  fixed_log
  newton
  subdiag_fast
  golden_ratio
  collision_donut
  bisection
  sparse_dataspec
  sparse_dataspec_transformed
  if_float
  if_float2
  single_loop_unrolled_160
  bisection_unrolled_16
  kmp
)


# =====================================================================
# MAP: kernel → experiment-array-variable
# =====================================================================

declare -A exps

exps["single_loop"]="default_experiments[@]"
exps["nested_loop"]="default_experiments[@]"
exps["fixed_log"]="default_experiments[@]"
exps["newton"]="default_experiments[@]"
exps["subdiag_fast"]="default_experiments[@]"
exps["golden_ratio"]="default_experiments[@]"

exps["collision_donut"]="no_predict_experiments[@]"
exps["bisection"]="no_predict_experiments[@]"

exps["sparse_dataspec"]="sparse_experiments[@]"
exps["sparse_dataspec_transformed"]="default_experiments[@]"

exps["if_float"]="if_float_experiments[@]"
exps["if_float2"]="if_float_experiments[@]"

exps["single_loop_unrolled_160"]="single_loop_unrolled_160_experiments[@]"
exps["bisection_unrolled_16"]="bisection_unrolled_16_experiments[@]"
exps["kmp"]="kmp_experiments[@]"


# =====================================================================
# MAIN LOOP
# =====================================================================

for k in "${kernels[@]}"; do
  for v in "${!exps[$k]}"; do

    folder="$DYNAMATIC_DIR/integration-test/$k/out_$v"

    echo "[run] $k / $v"

    python tools/dynamatic/estimate_power/estimate_power.py \
      --output_dir "$folder" \
      --kernel_name "$k" \
      --cp 7 \
      --synth "pre" \
      --license "without" \
  done
done