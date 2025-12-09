SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DYNAMATIC_DIR="$(cd "$SCRIPT_DIR" && pwd)"

cd $DYNAMATIC_DIR

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
)

variants=(
  out_baseline_7ns
  out_v1_7ns
  out_auto_7ns
)

for k in "${kernels[@]}"; do
  for v in "${variants[@]}"; do

    # skip v1 for kernels that don't have it
    if [[ ( "$v" == "out_v1_7ns" && "$k" == "bisection" )       || \
          ( "$v" == "out_v1_7ns" && "$k" == "collision_donut" ) || \
          ( "$v" == "out_v1_7ns" && "$k" == "if_float" )        || \
          ( "$v" == "out_auto_7ns" && "$k" == "if_float" )        || \
          ( "$v" == "out_auto_7ns" && "$k" == "if_float2" )        || \
          ( "$v" == "out_v1_7ns" && "$k" == "if_float2" ) ]]; then
      continue
    fi

    python tools/dynamatic/estimate_power/estimate_power.py \
      --output_dir "$DYNAMATIC_DIR/integration-test/$k/$v" \
      --kernel_name "$k" \
      --cp 7
  done
done