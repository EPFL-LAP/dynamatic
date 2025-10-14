SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DYNAMATIC_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd $DYNAMATIC_DIR

# Baseline
python3 experimental/tools/integration/run_specv2_integration.py sparse_dataspec_transformed --baseline --out out_baseline_7ns --cp 7.00 --transformed-code sparse_dataspec_modified.c

# Spec v1
python3 experimental/tools/integration/run_adapted_spec_integration.py sparse_dataspec_transformed --out out_v1_7ns --cp 7.00 --transformed-code sparse_dataspec_modified.c


python3 experimental/tools/integration/run_specv2_integration.py sparse_dataspec_transformed --n 1 --out out_1_7ns --cp 7.00 --resolver --transformed-code sparse_dataspec_modified.c
python3 experimental/tools/integration/run_specv2_integration.py sparse_dataspec_transformed --n 2 --out out_2_7ns --cp 7.00 --resolver --transformed-code sparse_dataspec_modified.c --use-prof-cache
python3 experimental/tools/integration/run_specv2_integration.py sparse_dataspec_transformed --n 3 --out out_3_7ns --cp 7.00 --resolver --transformed-code sparse_dataspec_modified.c --use-prof-cache
python3 experimental/tools/integration/run_specv2_integration.py sparse_dataspec_transformed --n 4 --out out_4_7ns --cp 7.00 --resolver --transformed-code sparse_dataspec_modified.c --use-prof-cache
python3 experimental/tools/integration/run_specv2_integration.py sparse_dataspec_transformed --n 5 --out out_5_7ns --cp 7.00 --resolver --transformed-code sparse_dataspec_modified.c --use-prof-cache
python3 experimental/tools/integration/run_specv2_integration.py sparse_dataspec_transformed --n 6 --out out_6_7ns --cp 7.00 --resolver --transformed-code sparse_dataspec_modified.c --use-prof-cache
python3 experimental/tools/integration/run_specv2_integration.py sparse_dataspec_transformed --n 7 --out out_7_7ns --cp 7.00 --resolver --transformed-code sparse_dataspec_modified.c --use-prof-cache
python3 experimental/tools/integration/run_specv2_integration.py sparse_dataspec_transformed --n 8 --out out_8_7ns --cp 7.00 --resolver --transformed-code sparse_dataspec_modified.c --use-prof-cache
python3 experimental/tools/integration/run_specv2_integration.py sparse_dataspec_transformed --n 9 --out out_9_7ns --cp 7.00 --resolver --transformed-code sparse_dataspec_modified.c --use-prof-cache
python3 experimental/tools/integration/run_specv2_integration.py sparse_dataspec_transformed --n 10 --out out_10_7ns --cp 7.00 --resolver --transformed-code sparse_dataspec_modified.c --use-prof-cache
python3 experimental/tools/integration/run_specv2_integration.py sparse_dataspec_transformed --n 11 --out out_11_7ns --cp 7.00 --resolver --transformed-code sparse_dataspec_modified.c --use-prof-cache
python3 experimental/tools/integration/run_specv2_integration.py sparse_dataspec_transformed --n 12 --out out_12_7ns --cp 7.00 --resolver --transformed-code sparse_dataspec_modified.c --use-prof-cache
python3 experimental/tools/integration/run_specv2_integration.py sparse_dataspec_transformed --n 13 --out out_13_7ns --cp 7.00 --resolver --transformed-code sparse_dataspec_modified.c --use-prof-cache
python3 experimental/tools/integration/run_specv2_integration.py sparse_dataspec_transformed --n 14 --out out_14_7ns --cp 7.00 --resolver --transformed-code sparse_dataspec_modified.c --use-prof-cache
python3 experimental/tools/integration/run_specv2_integration.py sparse_dataspec_transformed --n 15 --out out_15_7ns --cp 7.00 --resolver --transformed-code sparse_dataspec_modified.c --use-prof-cache
python3 experimental/tools/integration/run_specv2_integration.py sparse_dataspec_transformed --n 16 --out out_16_7ns --cp 7.00 --resolver --transformed-code sparse_dataspec_modified.c --use-prof-cache
python3 experimental/tools/integration/run_specv2_integration.py sparse_dataspec_transformed --n 17 --out out_17_7ns --cp 7.00 --resolver --transformed-code sparse_dataspec_modified.c --use-prof-cache
python3 experimental/tools/integration/run_specv2_integration.py sparse_dataspec_transformed --n 18 --out out_18_7ns --cp 7.00 --resolver --transformed-code sparse_dataspec_modified.c --use-prof-cache
python3 experimental/tools/integration/run_specv2_integration.py sparse_dataspec_transformed --n 19 --out out_19_7ns --cp 7.00 --resolver --transformed-code sparse_dataspec_modified.c --use-prof-cache
python3 experimental/tools/integration/run_specv2_integration.py sparse_dataspec_transformed --n 20 --out out_20_7ns --cp 7.00 --resolver --transformed-code sparse_dataspec_modified.c --use-prof-cache
