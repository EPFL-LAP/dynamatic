SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DYNAMATIC_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd $DYNAMATIC_DIR

# Baseline
# python3 experimental/tools/integration/run_specv2_integration.py sparse_dataspec_transformed --baseline --out out_baseline_5ns --cp 5.00 --transformed-code sparse_dataspec_modified.c

# Spec v1
# python3 experimental/tools/integration/run_adapted_spec_integration.py sparse_dataspec_transformed --out out_v1_5ns --cp 5.00 --transformed-code sparse_dataspec_modified.c

# python3 experimental/tools/integration/run_specv2_integration.py sparse_dataspec_transformed --n 0 --out out_0 --cp 10.00 --resolver --transformed-code sparse_dataspec_modified.c
# python3 experimental/tools/integration/run_specv2_integration.py sparse_dataspec_transformed --n 8 --out out_8_5ns --cp 5.00 --resolver --transformed-code sparse_dataspec_modified.c
# python3 experimental/tools/integration/run_specv2_integration.py sparse_dataspec_transformed --n 20 --out out_20_5ns --cp 5.00 --resolver --transformed-code sparse_dataspec_modified.c

# python3 experimental/tools/integration/run_specv2_integration.py sparse_dataspec_transformed --n 1 --out out_1_5ns --cp 5.00 --resolver --transformed-code sparse_dataspec_modified.c
# python3 experimental/tools/integration/run_specv2_integration.py sparse_dataspec_transformed --n 2 --out out_2_5ns --cp 5.00 --resolver --transformed-code sparse_dataspec_modified.c
# python3 experimental/tools/integration/run_specv2_integration.py sparse_dataspec_transformed --n 3 --out out_3_5ns --cp 5.00 --resolver --transformed-code sparse_dataspec_modified.c
# python3 experimental/tools/integration/run_specv2_integration.py sparse_dataspec_transformed --n 4 --out out_4_5ns --cp 5.00 --resolver --transformed-code sparse_dataspec_modified.c
# python3 experimental/tools/integration/run_specv2_integration.py sparse_dataspec_transformed --n 5 --out out_5_5ns --cp 5.00 --resolver --transformed-code sparse_dataspec_modified.c
# python3 experimental/tools/integration/run_specv2_integration.py sparse_dataspec_transformed --n 6 --out out_6_5ns --cp 5.00 --resolver --transformed-code sparse_dataspec_modified.c
# python3 experimental/tools/integration/run_specv2_integration.py sparse_dataspec_transformed --n 7 --out out_7_5ns --cp 5.00 --resolver --transformed-code sparse_dataspec_modified.c
# python3 experimental/tools/integration/run_specv2_integration.py sparse_dataspec_transformed --n 9 --out out_9_5ns --cp 5.00 --resolver --transformed-code sparse_dataspec_modified.c
# python3 experimental/tools/integration/run_specv2_integration.py sparse_dataspec_transformed --n 10 --out out_10_5ns --cp 5.00 --resolver --transformed-code sparse_dataspec_modified.c
# python3 experimental/tools/integration/run_specv2_integration.py sparse_dataspec_transformed --n 11 --out out_11_5ns --cp 5.00 --resolver --transformed-code sparse_dataspec_modified.c
# python3 experimental/tools/integration/run_specv2_integration.py sparse_dataspec_transformed --n 12 --out out_12_5ns --cp 5.00 --resolver --transformed-code sparse_dataspec_modified.c
# python3 experimental/tools/integration/run_specv2_integration.py sparse_dataspec_transformed --n 13 --out out_13_5ns --cp 5.00 --resolver --transformed-code sparse_dataspec_modified.c
# python3 experimental/tools/integration/run_specv2_integration.py sparse_dataspec_transformed --n 14 --out out_14_5ns --cp 5.00 --resolver --transformed-code sparse_dataspec_modified.c
# python3 experimental/tools/integration/run_specv2_integration.py sparse_dataspec_transformed --n 15 --out out_15_5ns --cp 5.00 --resolver --transformed-code sparse_dataspec_modified.c
# python3 experimental/tools/integration/run_specv2_integration.py sparse_dataspec_transformed --n 16 --out out_16_5ns --cp 5.00 --resolver --transformed-code sparse_dataspec_modified.c
# python3 experimental/tools/integration/run_specv2_integration.py sparse_dataspec_transformed --n 17 --out out_17_5ns --cp 5.00 --resolver --transformed-code sparse_dataspec_modified.c
# python3 experimental/tools/integration/run_specv2_integration.py sparse_dataspec_transformed --n 18 --out out_18_5ns --cp 5.00 --resolver --transformed-code sparse_dataspec_modified.c
# python3 experimental/tools/integration/run_specv2_integration.py sparse_dataspec_transformed --n 19 --out out_19_5ns --cp 5.00 --resolver --transformed-code sparse_dataspec_modified.c

# python3 experimental/tools/integration/run_specv2_integration.py sparse_dataspec_transformed --n 9 --out out_9 --cp 10.00 --resolver --transformed-code sparse_dataspec_modified.c

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

python3 experimental/tools/integration/run_specv2_integration.py sparse_dataspec_transformed --baseline --out out_baseline_5ns_om --cp 5.00 --transformed-code sparse_dataspec_modified.c --min-buffering
python3 experimental/tools/integration/run_specv2_integration.py sparse_dataspec_transformed --n 1 --out out_1_5ns_om --cp 5.00 --resolver --transformed-code sparse_dataspec_modified.c --min-buffering
python3 experimental/tools/integration/run_specv2_integration.py sparse_dataspec_transformed --n 2 --out out_2_5ns_om --cp 5.00 --resolver --transformed-code sparse_dataspec_modified.c --min-buffering --use-prof-cache
python3 experimental/tools/integration/run_specv2_integration.py sparse_dataspec_transformed --n 3 --out out_3_5ns_om --cp 5.00 --resolver --transformed-code sparse_dataspec_modified.c --min-buffering --use-prof-cache
python3 experimental/tools/integration/run_specv2_integration.py sparse_dataspec_transformed --n 4 --out out_4_5ns_om --cp 5.00 --resolver --transformed-code sparse_dataspec_modified.c --min-buffering --use-prof-cache
python3 experimental/tools/integration/run_specv2_integration.py sparse_dataspec_transformed --n 5 --out out_5_5ns_om --cp 5.00 --resolver --transformed-code sparse_dataspec_modified.c --min-buffering --use-prof-cache