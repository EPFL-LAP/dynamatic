SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DYNAMATIC_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd $DYNAMATIC_DIR

# # Baseline
# python3 experimental/tools/integration/run_specv2_integration.py sparse_dataspec --baseline --out out_baseline --cp 8.00

# # Spec v1
# python3 experimental/tools/integration/run_adapted_spec_integration.py sparse_dataspec --out out_v1 --cp 8.00 --default-value 0 --disable-constant-predictor

# python3 experimental/tools/integration/run_specv2_integration.py sparse_dataspec --n 0 --out out_0 --cp 8.00 --resolver
# python3 experimental/tools/integration/run_specv2_integration.py sparse_dataspec --n 8 --out out_8 --cp 8.00 --resolver
python3 experimental/tools/integration/run_specv2_integration.py sparse_dataspec --n 1 --out out_1 --cp 5.00 --resolver
python3 experimental/tools/integration/run_specv2_integration.py sparse_dataspec --n 2 --out out_2 --cp 5.00 --resolver --use-prof-cache
python3 experimental/tools/integration/run_specv2_integration.py sparse_dataspec --n 3 --out out_3 --cp 5.00 --resolver --use-prof-cache
python3 experimental/tools/integration/run_specv2_integration.py sparse_dataspec --n 4 --out out_4 --cp 5.00 --resolver --use-prof-cache
python3 experimental/tools/integration/run_specv2_integration.py sparse_dataspec --n 5 --out out_5 --cp 5.00 --resolver --use-prof-cache
python3 experimental/tools/integration/run_specv2_integration.py sparse_dataspec --n 6 --out out_6 --cp 5.00 --resolver --use-prof-cache
python3 experimental/tools/integration/run_specv2_integration.py sparse_dataspec --n 7 --out out_7 --cp 5.00 --resolver --use-prof-cache
python3 experimental/tools/integration/run_specv2_integration.py sparse_dataspec --n 8 --out out_8 --cp 5.00 --resolver --use-prof-cache
python3 experimental/tools/integration/run_specv2_integration.py sparse_dataspec --n 9 --out out_9 --cp 5.00 --resolver --use-prof-cache
python3 experimental/tools/integration/run_specv2_integration.py sparse_dataspec --n 10 --out out_10 --cp 5.00 --resolver --use-prof-cache
python3 experimental/tools/integration/run_specv2_integration.py sparse_dataspec --n 11 --out out_11 --cp 5.00 --resolver --use-prof-cache
python3 experimental/tools/integration/run_specv2_integration.py sparse_dataspec --n 12 --out out_12 --cp 5.00 --resolver --use-prof-cache
python3 experimental/tools/integration/run_specv2_integration.py sparse_dataspec --n 13 --out out_13 --cp 5.00 --resolver --use-prof-cache
python3 experimental/tools/integration/run_specv2_integration.py sparse_dataspec --n 14 --out out_14 --cp 5.00 --resolver --use-prof-cache
python3 experimental/tools/integration/run_specv2_integration.py sparse_dataspec --n 15 --out out_15 --cp 5.00 --resolver --use-prof-cache
python3 experimental/tools/integration/run_specv2_integration.py sparse_dataspec --n 16 --out out_16 --cp 5.00 --resolver --use-prof-cache
python3 experimental/tools/integration/run_specv2_integration.py sparse_dataspec --n 17 --out out_17 --cp 5.00 --resolver --use-prof-cache
python3 experimental/tools/integration/run_specv2_integration.py sparse_dataspec --n 18 --out out_18 --cp 5.00 --resolver --use-prof-cache
python3 experimental/tools/integration/run_specv2_integration.py sparse_dataspec --n 19 --out out_19 --cp 5.00 --resolver --use-prof-cache
python3 experimental/tools/integration/run_specv2_integration.py sparse_dataspec --n 20 --out out_20 --cp 5.00 --resolver --use-prof-cache

# Standard flow
# python3 experimental/tools/integration/run_specv2_integration.py sparse_dataspec --disable-spec --out out_standard

# Only code transformation
# python3 experimental/tools/integration/run_specv2_integration.py sparse_dataspec --disable-spec --transformed-code sparse_dataspec_transformed.c --out out_transformed

# Spec v1
# python3 tools/integration/run_spec_integration.py sparse_dataspec --out out_v1 --transformed-code sparse_dataspec_v1.c

# python3 experimental/tools/integration/run_specv2_integration.py sparse_dataspec --n 8 --out out_8 --transformed-code sparse_dataspec_transformed.c
# python3 experimental/tools/integration/run_specv2_integration.py sparse_dataspec --n 9 --out out_9 --transformed-code sparse_dataspec_transformed.c
# python3 experimental/tools/integration/run_specv2_integration.py sparse_dataspec --n 10 --out out_10 --transformed-code sparse_dataspec_transformed.c
# python3 experimental/tools/integration/run_specv2_integration.py sparse_dataspec --n 11 --out out_11 --transformed-code sparse_dataspec_transformed.c
# python3 experimental/tools/integration/run_specv2_integration.py sparse_dataspec --n 12 --out out_12 --transformed-code sparse_dataspec_transformed.c
# python3 experimental/tools/integration/run_specv2_integration.py sparse_dataspec --n 13 --out out_13 --transformed-code sparse_dataspec_transformed.c
# python3 experimental/tools/integration/run_specv2_integration.py sparse_dataspec --n 14 --out out_14 --transformed-code sparse_dataspec_transformed.c
# python3 experimental/tools/integration/run_specv2_integration.py sparse_dataspec --n 15 --out out_15 --transformed-code sparse_dataspec_transformed.c
# python3 experimental/tools/integration/run_specv2_integration.py sparse_dataspec --variable --out out_variable --transformed-code sparse_dataspec_transformed.c
