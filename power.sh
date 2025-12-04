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
    python tools/dynamatic/estimate_power/estimate_power.py \
      --output_dir "$DYNAMATIC_DIR/integration-test/$k/$v" \
      --kernel_name "$k" \
      --cp 7
  done
done