SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DYNAMATIC_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd $DYNAMATIC_DIR

# # Baseline:
# # - No CMerge
# # - No Branch
# python3 experimental/tools/integration/run_specv2_integration.py subdiag_fast --baseline --out out_baseline_5ns_trans --transformed-code subdiag_fast_v1.c --cp 5.00

# # Spec v1
# python3 experimental/tools/integration/run_adapted_spec_integration.py subdiag_fast --transformed-code subdiag_fast_v1.c --cp 5.00 --out out_v1_5ns

# python3 experimental/tools/integration/run_specv2_integration.py subdiag_fast --n 13 --out out_13_5ns_trans --cp 5.00 --resolver --transformed-code subdiag_fast_v1.c

# python3 experimental/tools/integration/run_specv2_integration.py subdiag_fast --n 15 --out out_15_5ns --cp 5.00 --resolver --use-prof-cache
# python3 experimental/tools/integration/run_specv2_integration.py subdiag_fast --n 17 --out out_17_5ns --cp 5.00 --resolver --use-prof-cache
# python3 experimental/tools/integration/run_specv2_integration.py subdiag_fast --n 20 --out out_20_5ns --cp 5.00 --resolver --use-prof-cache

# python3 experimental/tools/integration/run_specv2_integration.py subdiag_fast --n 1 --out out_1_5ns_trans --cp 5.00 --resolver --transformed-code subdiag_fast_v1.c
# python3 experimental/tools/integration/run_specv2_integration.py subdiag_fast --n 2 --out out_2_5ns_trans --cp 5.00 --resolver --transformed-code subdiag_fast_v1.c --use-prof-cache
# python3 experimental/tools/integration/run_specv2_integration.py subdiag_fast --n 3 --out out_3_5ns_trans --cp 5.00 --resolver --transformed-code subdiag_fast_v1.c --use-prof-cache
# python3 experimental/tools/integration/run_specv2_integration.py subdiag_fast --n 4 --out out_4_5ns_trans --cp 5.00 --resolver --transformed-code subdiag_fast_v1.c --use-prof-cache
# python3 experimental/tools/integration/run_specv2_integration.py subdiag_fast --n 5 --out out_5_5ns_trans --cp 5.00 --resolver --transformed-code subdiag_fast_v1.c --use-prof-cache
# python3 experimental/tools/integration/run_specv2_integration.py subdiag_fast --n 6 --out out_6_5ns_trans --cp 5.00 --resolver --transformed-code subdiag_fast_v1.c --use-prof-cache
# python3 experimental/tools/integration/run_specv2_integration.py subdiag_fast --n 7 --out out_7_5ns_trans --cp 5.00 --resolver --transformed-code subdiag_fast_v1.c --use-prof-cache
# python3 experimental/tools/integration/run_specv2_integration.py subdiag_fast --n 8 --out out_8_5ns_trans --cp 5.00 --resolver --transformed-code subdiag_fast_v1.c --use-prof-cache
# python3 experimental/tools/integration/run_specv2_integration.py subdiag_fast --n 9 --out out_9_5ns_trans --cp 5.00 --resolver --transformed-code subdiag_fast_v1.c --use-prof-cache
# python3 experimental/tools/integration/run_specv2_integration.py subdiag_fast --n 10 --out out_10_5ns_trans --cp 5.00 --resolver --transformed-code subdiag_fast_v1.c --use-prof-cache
# python3 experimental/tools/integration/run_specv2_integration.py subdiag_fast --n 11 --out out_11_5ns_trans --cp 5.00 --resolver --transformed-code subdiag_fast_v1.c --use-prof-cache
# python3 experimental/tools/integration/run_specv2_integration.py subdiag_fast --n 12 --out out_12_5ns_trans --cp 5.00 --resolver --transformed-code subdiag_fast_v1.c --use-prof-cache
# # python3 experimental/tools/integration/run_specv2_integration.py subdiag_fast --n 13 --out out_13_5ns_trans --cp 5.00 --resolver --transformed-code subdiag_fast_v1.c --use-prof-cache
# python3 experimental/tools/integration/run_specv2_integration.py subdiag_fast --n 14 --out out_14_5ns_trans --cp 5.00 --resolver --transformed-code subdiag_fast_v1.c --use-prof-cache
# python3 experimental/tools/integration/run_specv2_integration.py subdiag_fast --n 15 --out out_15_5ns_trans --cp 5.00 --resolver --transformed-code subdiag_fast_v1.c --use-prof-cache
# python3 experimental/tools/integration/run_specv2_integration.py subdiag_fast --n 16 --out out_16_5ns_trans --cp 5.00 --resolver --transformed-code subdiag_fast_v1.c --use-prof-cache
# python3 experimental/tools/integration/run_specv2_integration.py subdiag_fast --n 17 --out out_17_5ns_trans --cp 5.00 --resolver --transformed-code subdiag_fast_v1.c --use-prof-cache
# python3 experimental/tools/integration/run_specv2_integration.py subdiag_fast --n 18 --out out_18_5ns_trans --cp 5.00 --resolver --transformed-code subdiag_fast_v1.c --use-prof-cache
# python3 experimental/tools/integration/run_specv2_integration.py subdiag_fast --n 19 --out out_19_5ns_trans --cp 5.00 --resolver --transformed-code subdiag_fast_v1.c --use-prof-cache
# python3 experimental/tools/integration/run_specv2_integration.py subdiag_fast --n 20 --out out_20_5ns_trans --cp 5.00 --resolver --transformed-code subdiag_fast_v1.c --use-prof-cache

python3 experimental/tools/integration/run_specv2_integration.py subdiag_fast --baseline --out out_baseline_5ns_om --transformed-code subdiag_fast_v1.c --cp 5.00 --min-buffering
python3 experimental/tools/integration/run_specv2_integration.py subdiag_fast --n 1 --out out_1_5ns_om --cp 5.00 --resolver --transformed-code subdiag_fast_v1.c --min-buffering
python3 experimental/tools/integration/run_specv2_integration.py subdiag_fast --n 2 --out out_2_5ns_om --cp 5.00 --resolver --transformed-code subdiag_fast_v1.c --use-prof-cache --min-buffering
python3 experimental/tools/integration/run_specv2_integration.py subdiag_fast --n 3 --out out_3_5ns_om --cp 5.00 --resolver --transformed-code subdiag_fast_v1.c --use-prof-cache --min-buffering
python3 experimental/tools/integration/run_specv2_integration.py subdiag_fast --n 4 --out out_4_5ns_om --cp 5.00 --resolver --transformed-code subdiag_fast_v1.c --use-prof-cache --min-buffering
python3 experimental/tools/integration/run_specv2_integration.py subdiag_fast --n 5 --out out_5_5ns_om --cp 5.00 --resolver --transformed-code subdiag_fast_v1.c --use-prof-cache --min-buffering
python3 experimental/tools/integration/run_specv2_integration.py subdiag_fast --n 6 --out out_6_5ns_om --cp 5.00 --resolver --transformed-code subdiag_fast_v1.c --use-prof-cache --min-buffering
python3 experimental/tools/integration/run_specv2_integration.py subdiag_fast --n 7 --out out_7_5ns_om --cp 5.00 --resolver --transformed-code subdiag_fast_v1.c --use-prof-cache --min-buffering
python3 experimental/tools/integration/run_specv2_integration.py subdiag_fast --n 8 --out out_8_5ns_om --cp 5.00 --resolver --transformed-code subdiag_fast_v1.c --use-prof-cache --min-buffering
python3 experimental/tools/integration/run_specv2_integration.py subdiag_fast --n 9 --out out_9_5ns_om --cp 5.00 --resolver --transformed-code subdiag_fast_v1.c --use-prof-cache --min-buffering
python3 experimental/tools/integration/run_specv2_integration.py subdiag_fast --n 10 --out out_10_5ns_om --cp 5.00 --resolver --transformed-code subdiag_fast_v1.c --use-prof-cache --min-buffering
python3 experimental/tools/integration/run_specv2_integration.py subdiag_fast --n 11 --out out_11_5ns_om --cp 5.00 --resolver --transformed-code subdiag_fast_v1.c --use-prof-cache --min-buffering
python3 experimental/tools/integration/run_specv2_integration.py subdiag_fast --n 12 --out out_12_5ns_om --cp 5.00 --resolver --transformed-code subdiag_fast_v1.c --use-prof-cache --min-buffering
python3 experimental/tools/integration/run_specv2_integration.py subdiag_fast --n 13 --out out_13_5ns_om --cp 5.00 --resolver --transformed-code subdiag_fast_v1.c --use-prof-cache --min-buffering
python3 experimental/tools/integration/run_specv2_integration.py subdiag_fast --n 14 --out out_14_5ns_om --cp 5.00 --resolver --transformed-code subdiag_fast_v1.c --use-prof-cache --min-buffering