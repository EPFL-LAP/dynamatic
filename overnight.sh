SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DYNAMATIC_DIR="$(cd "$SCRIPT_DIR" && pwd)"

cd $DYNAMATIC_DIR

date > start_time.txt

# Simulation

python3 experimental/tools/integration/run_specv2_integration.py single_loop --baseline --out out_baseline_7ns --cp 7.00
python3 experimental/tools/integration/run_adapted_spec_integration.py single_loop --out out_v1_7ns --cp 7.00
python3 experimental/tools/integration/run_specv2_integration.py single_loop --n 1 --out out_1_7ns --cp 7.00 --resolver
python3 experimental/tools/integration/run_specv2_integration.py single_loop --n 2 --out out_2_7ns --cp 7.00 --resolver --use-prof-cache
python3 experimental/tools/integration/run_specv2_integration.py single_loop --n 3 --out out_3_7ns --cp 7.00 --resolver --use-prof-cache
python3 experimental/tools/integration/run_specv2_integration.py single_loop --n 4 --out out_4_7ns --cp 7.00 --resolver --use-prof-cache
python3 experimental/tools/integration/run_specv2_integration.py single_loop --n 5 --out out_5_7ns --cp 7.00 --resolver --use-prof-cache
python3 experimental/tools/integration/run_specv2_integration.py single_loop --n 6 --out out_6_7ns --cp 7.00 --resolver --use-prof-cache
python3 experimental/tools/integration/run_specv2_integration.py single_loop --n 7 --out out_7_7ns --cp 7.00 --resolver --use-prof-cache
python3 experimental/tools/integration/run_specv2_integration.py single_loop --n 8 --out out_8_7ns --cp 7.00 --resolver --use-prof-cache
python3 experimental/tools/integration/run_specv2_integration.py single_loop --n 9 --out out_9_7ns --cp 7.00 --resolver --use-prof-cache
python3 experimental/tools/integration/run_specv2_integration.py single_loop --n 10 --out out_10_7ns --cp 7.00 --resolver --use-prof-cache
python3 experimental/tools/integration/run_specv2_integration.py single_loop --n 11 --out out_11_7ns --cp 7.00 --resolver --use-prof-cache
python3 experimental/tools/integration/run_specv2_integration.py single_loop --n 12 --out out_12_7ns --cp 7.00 --resolver --use-prof-cache
python3 experimental/tools/integration/run_specv2_integration.py single_loop --n 13 --out out_13_7ns --cp 7.00 --resolver --use-prof-cache
python3 experimental/tools/integration/run_specv2_integration.py single_loop --n 14 --out out_14_7ns --cp 7.00 --resolver --use-prof-cache
python3 experimental/tools/integration/run_specv2_integration.py single_loop --n 15 --out out_15_7ns --cp 7.00 --resolver --use-prof-cache
python3 experimental/tools/integration/run_specv2_integration.py single_loop --n 16 --out out_16_7ns --cp 7.00 --resolver --use-prof-cache
python3 experimental/tools/integration/run_specv2_integration.py single_loop --n 17 --out out_17_7ns --cp 7.00 --resolver --use-prof-cache
python3 experimental/tools/integration/run_specv2_integration.py single_loop --n 18 --out out_18_7ns --cp 7.00 --resolver --use-prof-cache
python3 experimental/tools/integration/run_specv2_integration.py single_loop --n 19 --out out_19_7ns --cp 7.00 --resolver --use-prof-cache
python3 experimental/tools/integration/run_specv2_integration.py single_loop --n 20 --out out_20_7ns --cp 7.00 --resolver --use-prof-cache

python3 experimental/tools/integration/run_specv2_integration.py nested_loop --baseline --out out_baseline_7ns --cp 7.00
python3 experimental/tools/integration/run_adapted_spec_integration.py nested_loop --out out_v1_7ns --cp 7.00
python3 experimental/tools/integration/run_specv2_integration.py nested_loop --n 0 --out out_0_7ns --cp 7.00 --resolver
python3 experimental/tools/integration/run_specv2_integration.py nested_loop --n 1 --out out_1_7ns --cp 7.00 --use-prof-cache --resolver
python3 experimental/tools/integration/run_specv2_integration.py nested_loop --n 2 --out out_2_7ns --cp 7.00 --use-prof-cache --resolver
python3 experimental/tools/integration/run_specv2_integration.py nested_loop --n 3 --out out_3_7ns --cp 7.00 --use-prof-cache --resolver
python3 experimental/tools/integration/run_specv2_integration.py nested_loop --n 4 --out out_4_7ns --cp 7.00 --use-prof-cache --resolver
python3 experimental/tools/integration/run_specv2_integration.py nested_loop --n 5 --out out_5_7ns --cp 7.00 --use-prof-cache --resolver
python3 experimental/tools/integration/run_specv2_integration.py nested_loop --n 6 --out out_6_7ns --cp 7.00 --use-prof-cache --resolver
python3 experimental/tools/integration/run_specv2_integration.py nested_loop --n 7 --out out_7_7ns --cp 7.00 --use-prof-cache --resolver
python3 experimental/tools/integration/run_specv2_integration.py nested_loop --n 8 --out out_8_7ns --cp 7.00 --use-prof-cache --resolver
python3 experimental/tools/integration/run_specv2_integration.py nested_loop --n 9 --out out_9_7ns --cp 7.00 --use-prof-cache --resolver
python3 experimental/tools/integration/run_specv2_integration.py nested_loop --n 10 --out out_10_7ns --cp 7.00 --use-prof-cache --resolver
python3 experimental/tools/integration/run_specv2_integration.py nested_loop --n 11 --out out_11_7ns --cp 7.00 --use-prof-cache --resolver
python3 experimental/tools/integration/run_specv2_integration.py nested_loop --n 12 --out out_12_7ns --cp 7.00 --use-prof-cache --resolver
python3 experimental/tools/integration/run_specv2_integration.py nested_loop --n 13 --out out_13_7ns --cp 7.00 --use-prof-cache --resolver
python3 experimental/tools/integration/run_specv2_integration.py nested_loop --n 14 --out out_14_7ns --cp 7.00 --use-prof-cache --resolver
python3 experimental/tools/integration/run_specv2_integration.py nested_loop --n 15 --out out_15_7ns --cp 7.00 --use-prof-cache --resolver
python3 experimental/tools/integration/run_specv2_integration.py nested_loop --n 16 --out out_16_7ns --cp 7.00 --use-prof-cache --resolver
python3 experimental/tools/integration/run_specv2_integration.py nested_loop --n 17 --out out_17_7ns --cp 7.00 --use-prof-cache --resolver
python3 experimental/tools/integration/run_specv2_integration.py nested_loop --n 18 --out out_18_7ns --cp 7.00 --use-prof-cache --resolver
python3 experimental/tools/integration/run_specv2_integration.py nested_loop --n 19 --out out_19_7ns --cp 7.00 --use-prof-cache --resolver
python3 experimental/tools/integration/run_specv2_integration.py nested_loop --n 20 --out out_20_7ns --cp 7.00 --use-prof-cache --resolver

python3 experimental/tools/integration/run_specv2_integration.py fixed_log --baseline --out out_baseline_7ns --cp 7.00
python3 experimental/tools/integration/run_adapted_spec_integration.py fixed_log --default-value=0 --out out_v1_7ns --cp 7.00
python3 experimental/tools/integration/run_specv2_integration.py fixed_log --n 1 --out out_1_7ns --cp 7.00 --resolver
python3 experimental/tools/integration/run_specv2_integration.py fixed_log --n 2 --out out_2_7ns --cp 7.00 --resolver --use-prof-cache
python3 experimental/tools/integration/run_specv2_integration.py fixed_log --n 3 --out out_3_7ns --cp 7.00 --resolver --use-prof-cache
python3 experimental/tools/integration/run_specv2_integration.py fixed_log --n 4 --out out_4_7ns --cp 7.00 --resolver --use-prof-cache
python3 experimental/tools/integration/run_specv2_integration.py fixed_log --n 5 --out out_5_7ns --cp 7.00 --resolver --use-prof-cache
python3 experimental/tools/integration/run_specv2_integration.py fixed_log --n 6 --out out_6_7ns --cp 7.00 --resolver --use-prof-cache
python3 experimental/tools/integration/run_specv2_integration.py fixed_log --n 7 --out out_7_7ns --cp 7.00 --resolver --use-prof-cache
python3 experimental/tools/integration/run_specv2_integration.py fixed_log --n 8 --out out_8_7ns --cp 7.00 --resolver --use-prof-cache
python3 experimental/tools/integration/run_specv2_integration.py fixed_log --n 9 --out out_9_7ns --cp 7.00 --resolver --use-prof-cache
python3 experimental/tools/integration/run_specv2_integration.py fixed_log --n 10 --out out_10_7ns --cp 7.00 --resolver --use-prof-cache
python3 experimental/tools/integration/run_specv2_integration.py fixed_log --n 11 --out out_11_7ns --cp 7.00 --resolver --use-prof-cache
python3 experimental/tools/integration/run_specv2_integration.py fixed_log --n 12 --out out_12_7ns --cp 7.00 --resolver --use-prof-cache
python3 experimental/tools/integration/run_specv2_integration.py fixed_log --n 13 --out out_13_7ns --cp 7.00 --resolver --use-prof-cache
python3 experimental/tools/integration/run_specv2_integration.py fixed_log --n 14 --out out_14_7ns --cp 7.00 --resolver --use-prof-cache
python3 experimental/tools/integration/run_specv2_integration.py fixed_log --n 15 --out out_15_7ns --cp 7.00 --resolver --use-prof-cache
python3 experimental/tools/integration/run_specv2_integration.py fixed_log --n 16 --out out_16_7ns --cp 7.00 --resolver --use-prof-cache
python3 experimental/tools/integration/run_specv2_integration.py fixed_log --n 17 --out out_17_7ns --cp 7.00 --resolver --use-prof-cache
python3 experimental/tools/integration/run_specv2_integration.py fixed_log --n 18 --out out_18_7ns --cp 7.00 --resolver --use-prof-cache
python3 experimental/tools/integration/run_specv2_integration.py fixed_log --n 19 --out out_19_7ns --cp 7.00 --resolver --use-prof-cache
python3 experimental/tools/integration/run_specv2_integration.py fixed_log --n 20 --out out_20_7ns --cp 7.00 --resolver --use-prof-cache

# python3 experimental/tools/integration/run_specv2_integration.py newton --baseline --out out_baseline_7ns --cp 7.00
# python3 experimental/tools/integration/run_adapted_spec_integration.py newton --out out_v1_7ns --cp 7.00 --default-value 0
python3 experimental/tools/integration/run_specv2_integration.py newton --n 1 --out out_1_7ns --cp 7.00 --resolver
python3 experimental/tools/integration/run_specv2_integration.py newton --n 2 --out out_2_7ns --cp 7.00 --resolver --use-prof-cache
python3 experimental/tools/integration/run_specv2_integration.py newton --n 3 --out out_3_7ns --cp 7.00 --resolver --use-prof-cache
python3 experimental/tools/integration/run_specv2_integration.py newton --n 4 --out out_4_7ns --cp 7.00 --resolver --use-prof-cache
python3 experimental/tools/integration/run_specv2_integration.py newton --n 5 --out out_5_7ns --cp 7.00 --resolver --use-prof-cache
python3 experimental/tools/integration/run_specv2_integration.py newton --n 6 --out out_6_7ns --cp 7.00 --resolver --use-prof-cache
python3 experimental/tools/integration/run_specv2_integration.py newton --n 7 --out out_7_7ns --cp 7.00 --resolver --use-prof-cache
python3 experimental/tools/integration/run_specv2_integration.py newton --n 8 --out out_8_7ns --cp 7.00 --resolver --use-prof-cache
python3 experimental/tools/integration/run_specv2_integration.py newton --n 9 --out out_9_7ns --cp 7.00 --resolver --use-prof-cache
python3 experimental/tools/integration/run_specv2_integration.py newton --n 10 --out out_10_7ns --cp 7.00 --resolver --use-prof-cache
python3 experimental/tools/integration/run_specv2_integration.py newton --n 11 --out out_11_7ns --cp 7.00 --resolver --use-prof-cache
python3 experimental/tools/integration/run_specv2_integration.py newton --n 12 --out out_12_7ns --cp 7.00 --resolver --use-prof-cache
python3 experimental/tools/integration/run_specv2_integration.py newton --n 13 --out out_13_7ns --cp 7.00 --resolver --use-prof-cache
python3 experimental/tools/integration/run_specv2_integration.py newton --n 14 --out out_14_7ns --cp 7.00 --resolver --use-prof-cache
python3 experimental/tools/integration/run_specv2_integration.py newton --n 15 --out out_15_7ns --cp 7.00 --resolver --use-prof-cache
python3 experimental/tools/integration/run_specv2_integration.py newton --n 16 --out out_16_7ns --cp 7.00 --resolver --use-prof-cache
python3 experimental/tools/integration/run_specv2_integration.py newton --n 17 --out out_17_7ns --cp 7.00 --resolver --use-prof-cache
python3 experimental/tools/integration/run_specv2_integration.py newton --n 18 --out out_18_7ns --cp 7.00 --resolver --use-prof-cache
python3 experimental/tools/integration/run_specv2_integration.py newton --n 19 --out out_19_7ns --cp 7.00 --resolver --use-prof-cache
python3 experimental/tools/integration/run_specv2_integration.py newton --n 20 --out out_20_7ns --cp 7.00 --resolver --use-prof-cache


python3 experimental/tools/integration/run_specv2_integration.py subdiag_fast --n 1 --out out_1_7ns_trans --cp 7.00 --resolver --transformed-code subdiag_fast_v1.c
python3 experimental/tools/integration/run_specv2_integration.py subdiag_fast --n 2 --out out_2_7ns_trans --cp 7.00 --resolver --transformed-code subdiag_fast_v1.c --use-prof-cache
python3 experimental/tools/integration/run_specv2_integration.py subdiag_fast --n 3 --out out_3_7ns_trans --cp 7.00 --resolver --transformed-code subdiag_fast_v1.c --use-prof-cache
python3 experimental/tools/integration/run_specv2_integration.py subdiag_fast --n 4 --out out_4_7ns_trans --cp 7.00 --resolver --transformed-code subdiag_fast_v1.c --use-prof-cache
python3 experimental/tools/integration/run_specv2_integration.py subdiag_fast --n 5 --out out_5_7ns_trans --cp 7.00 --resolver --transformed-code subdiag_fast_v1.c --use-prof-cache
python3 experimental/tools/integration/run_specv2_integration.py subdiag_fast --n 6 --out out_6_7ns_trans --cp 7.00 --resolver --transformed-code subdiag_fast_v1.c --use-prof-cache
python3 experimental/tools/integration/run_specv2_integration.py subdiag_fast --n 7 --out out_7_7ns_trans --cp 7.00 --resolver --transformed-code subdiag_fast_v1.c --use-prof-cache
python3 experimental/tools/integration/run_specv2_integration.py subdiag_fast --n 8 --out out_8_7ns_trans --cp 7.00 --resolver --transformed-code subdiag_fast_v1.c --use-prof-cache
python3 experimental/tools/integration/run_specv2_integration.py subdiag_fast --n 9 --out out_9_7ns_trans --cp 7.00 --resolver --transformed-code subdiag_fast_v1.c --use-prof-cache
python3 experimental/tools/integration/run_specv2_integration.py subdiag_fast --n 10 --out out_10_7ns_trans --cp 7.00 --resolver --transformed-code subdiag_fast_v1.c --use-prof-cache
python3 experimental/tools/integration/run_specv2_integration.py subdiag_fast --n 11 --out out_11_7ns_trans --cp 7.00 --resolver --transformed-code subdiag_fast_v1.c --use-prof-cache
python3 experimental/tools/integration/run_specv2_integration.py subdiag_fast --n 12 --out out_12_7ns_trans --cp 7.00 --resolver --transformed-code subdiag_fast_v1.c --use-prof-cache
python3 experimental/tools/integration/run_specv2_integration.py subdiag_fast --n 13 --out out_13_7ns_trans --cp 7.00 --resolver --transformed-code subdiag_fast_v1.c --use-prof-cache
python3 experimental/tools/integration/run_specv2_integration.py subdiag_fast --n 14 --out out_14_7ns_trans --cp 7.00 --resolver --transformed-code subdiag_fast_v1.c --use-prof-cache
python3 experimental/tools/integration/run_specv2_integration.py subdiag_fast --n 15 --out out_15_7ns_trans --cp 7.00 --resolver --transformed-code subdiag_fast_v1.c --use-prof-cache
python3 experimental/tools/integration/run_specv2_integration.py subdiag_fast --n 16 --out out_16_7ns_trans --cp 7.00 --resolver --transformed-code subdiag_fast_v1.c --use-prof-cache
python3 experimental/tools/integration/run_specv2_integration.py subdiag_fast --n 17 --out out_17_7ns_trans --cp 7.00 --resolver --transformed-code subdiag_fast_v1.c --use-prof-cache
python3 experimental/tools/integration/run_specv2_integration.py subdiag_fast --n 18 --out out_18_7ns_trans --cp 7.00 --resolver --transformed-code subdiag_fast_v1.c --use-prof-cache
python3 experimental/tools/integration/run_specv2_integration.py subdiag_fast --n 19 --out out_19_7ns_trans --cp 7.00 --resolver --transformed-code subdiag_fast_v1.c --use-prof-cache
python3 experimental/tools/integration/run_specv2_integration.py subdiag_fast --n 20 --out out_20_7ns_trans --cp 7.00 --resolver --transformed-code subdiag_fast_v1.c --use-prof-cache

python3 experimental/tools/integration/run_specv2_integration.py golden_ratio --baseline --out out_baseline_7ns --cp 7.00
python3 experimental/tools/integration/run_adapted_spec_integration.py golden_ratio --out out_v1_7ns --cp 7.00 --default-value 0
python3 experimental/tools/integration/run_specv2_integration.py golden_ratio --n 1 --out out_1_exit_7ns --cp 7.00 --exit-eager-eval --resolver
python3 experimental/tools/integration/run_specv2_integration.py golden_ratio --n 2 --out out_2_exit_7ns --cp 7.00 --exit-eager-eval --resolver --use-prof-cache
python3 experimental/tools/integration/run_specv2_integration.py golden_ratio --n 3 --out out_3_exit_7ns --cp 7.00 --exit-eager-eval --resolver --use-prof-cache
python3 experimental/tools/integration/run_specv2_integration.py golden_ratio --n 4 --out out_4_exit_7ns --cp 7.00 --exit-eager-eval --resolver --use-prof-cache
python3 experimental/tools/integration/run_specv2_integration.py golden_ratio --n 5 --out out_5_exit_7ns --cp 7.00 --exit-eager-eval --resolver --use-prof-cache
python3 experimental/tools/integration/run_specv2_integration.py golden_ratio --n 6 --out out_6_exit_7ns --cp 7.00 --exit-eager-eval --resolver --use-prof-cache
python3 experimental/tools/integration/run_specv2_integration.py golden_ratio --n 7 --out out_7_exit_7ns --cp 7.00 --exit-eager-eval --resolver --use-prof-cache
python3 experimental/tools/integration/run_specv2_integration.py golden_ratio --n 8 --out out_8_exit_7ns --cp 7.00 --exit-eager-eval --resolver --use-prof-cache
python3 experimental/tools/integration/run_specv2_integration.py golden_ratio --n 9 --out out_9_exit_7ns --cp 7.00 --exit-eager-eval --resolver --use-prof-cache
python3 experimental/tools/integration/run_specv2_integration.py golden_ratio --n 10 --out out_10_exit_7ns --cp 7.00 --exit-eager-eval --resolver --use-prof-cache

python3 experimental/tools/integration/run_specv2_integration.py collision_donut --baseline --out out_baseline_7ns --cp 7.00
python3 experimental/tools/integration/run_specv2_integration.py collision_donut --n 0 --out out_0_7ns --cp 7.00 --resolver
python3 experimental/tools/integration/run_specv2_integration.py collision_donut --n 1 --out out_1_7ns --cp 7.00 --use-prof-cache --resolver
python3 experimental/tools/integration/run_specv2_integration.py collision_donut --n 2 --out out_2_7ns --cp 7.00 --use-prof-cache --resolver
python3 experimental/tools/integration/run_specv2_integration.py collision_donut --n 3 --out out_3_7ns --cp 7.00 --use-prof-cache --resolver
python3 experimental/tools/integration/run_specv2_integration.py collision_donut --n 4 --out out_4_7ns --cp 7.00 --use-prof-cache --resolver
python3 experimental/tools/integration/run_specv2_integration.py collision_donut --n 5 --out out_5_7ns --cp 7.00 --use-prof-cache --resolver
python3 experimental/tools/integration/run_specv2_integration.py collision_donut --n 6 --out out_6_7ns --cp 7.00 --use-prof-cache --resolver
python3 experimental/tools/integration/run_specv2_integration.py collision_donut --n 7 --out out_7_7ns --cp 7.00 --use-prof-cache --resolver
python3 experimental/tools/integration/run_specv2_integration.py collision_donut --n 8 --out out_8_7ns --cp 7.00 --use-prof-cache --resolver
python3 experimental/tools/integration/run_specv2_integration.py collision_donut --n 9 --out out_9_7ns --cp 7.00 --use-prof-cache --resolver
python3 experimental/tools/integration/run_specv2_integration.py collision_donut --n 10 --out out_10_7ns --cp 7.00 --use-prof-cache --resolver
python3 experimental/tools/integration/run_specv2_integration.py collision_donut --n 11 --out out_11_7ns --cp 7.00 --use-prof-cache --resolver
python3 experimental/tools/integration/run_specv2_integration.py collision_donut --n 12 --out out_12_7ns --cp 7.00 --use-prof-cache --resolver
python3 experimental/tools/integration/run_specv2_integration.py collision_donut --n 13 --out out_13_7ns --cp 7.00 --use-prof-cache --resolver
python3 experimental/tools/integration/run_specv2_integration.py collision_donut --n 14 --out out_14_7ns --cp 7.00 --use-prof-cache --resolver
python3 experimental/tools/integration/run_specv2_integration.py collision_donut --n 15 --out out_15_7ns --cp 7.00 --use-prof-cache --resolver
python3 experimental/tools/integration/run_specv2_integration.py collision_donut --n 16 --out out_16_7ns --cp 7.00 --use-prof-cache --resolver
python3 experimental/tools/integration/run_specv2_integration.py collision_donut --n 17 --out out_17_7ns --cp 7.00 --use-prof-cache --resolver
python3 experimental/tools/integration/run_specv2_integration.py collision_donut --n 18 --out out_18_7ns --cp 7.00 --use-prof-cache --resolver
python3 experimental/tools/integration/run_specv2_integration.py collision_donut --n 19 --out out_19_7ns --cp 7.00 --use-prof-cache --resolver
python3 experimental/tools/integration/run_specv2_integration.py collision_donut --n 20 --out out_20_7ns --cp 7.00 --use-prof-cache --resolver


python3 experimental/tools/integration/run_specv2_integration.py bisection --baseline --out out_baseline_7ns --cp 7.00 --transformed-code bisection_transformed.c
python3 experimental/tools/integration/run_specv2_integration.py bisection --n 0 --out out_0_7ns --cp 7.00 --resolver --transformed-code bisection_transformed.c
python3 experimental/tools/integration/run_specv2_integration.py bisection --n 1 --out out_1_7ns --cp 7.00 --resolver --transformed-code bisection_transformed.c
python3 experimental/tools/integration/run_specv2_integration.py bisection --n 2 --out out_2_7ns --cp 7.00 --resolver --transformed-code bisection_transformed.c
python3 experimental/tools/integration/run_specv2_integration.py bisection --n 3 --out out_3_7ns --cp 7.00 --resolver --use-prof-cache --transformed-code bisection_transformed.c
python3 experimental/tools/integration/run_specv2_integration.py bisection --n 4 --out out_4_7ns --cp 7.00 --resolver --use-prof-cache --transformed-code bisection_transformed.c
python3 experimental/tools/integration/run_specv2_integration.py bisection --n 5 --out out_5_7ns --cp 7.00 --resolver --use-prof-cache --transformed-code bisection_transformed.c
python3 experimental/tools/integration/run_specv2_integration.py bisection --n 6 --out out_6_7ns --cp 7.00 --resolver --use-prof-cache --transformed-code bisection_transformed.c
python3 experimental/tools/integration/run_specv2_integration.py bisection --n 7 --out out_7_7ns --cp 7.00 --resolver --use-prof-cache --transformed-code bisection_transformed.c
python3 experimental/tools/integration/run_specv2_integration.py bisection --n 8 --out out_8_7ns --cp 7.00 --resolver --use-prof-cache --transformed-code bisection_transformed.c
python3 experimental/tools/integration/run_specv2_integration.py bisection --n 9 --out out_9_7ns --cp 7.00 --resolver --use-prof-cache --transformed-code bisection_transformed.c
python3 experimental/tools/integration/run_specv2_integration.py bisection --n 10 --out out_10_7ns --cp 7.00 --resolver --use-prof-cache --transformed-code bisection_transformed.c
python3 experimental/tools/integration/run_specv2_integration.py bisection --n 11 --out out_11_7ns --cp 7.00 --resolver --use-prof-cache --transformed-code bisection_transformed.c
python3 experimental/tools/integration/run_specv2_integration.py bisection --n 12 --out out_12_7ns --cp 7.00 --resolver --use-prof-cache --transformed-code bisection_transformed.c
python3 experimental/tools/integration/run_specv2_integration.py bisection --n 13 --out out_13_7ns --cp 7.00 --resolver --use-prof-cache --transformed-code bisection_transformed.c
python3 experimental/tools/integration/run_specv2_integration.py bisection --n 14 --out out_14_7ns --cp 7.00 --resolver --use-prof-cache --transformed-code bisection_transformed.c
python3 experimental/tools/integration/run_specv2_integration.py bisection --n 15 --out out_15_7ns --cp 7.00 --resolver --use-prof-cache --transformed-code bisection_transformed.c
python3 experimental/tools/integration/run_specv2_integration.py bisection --n 16 --out out_16_7ns --cp 7.00 --resolver --use-prof-cache --transformed-code bisection_transformed.c
python3 experimental/tools/integration/run_specv2_integration.py bisection --n 17 --out out_17_7ns --cp 7.00 --resolver --use-prof-cache --transformed-code bisection_transformed.c
python3 experimental/tools/integration/run_specv2_integration.py bisection --n 18 --out out_18_7ns --cp 7.00 --resolver --use-prof-cache --transformed-code bisection_transformed.c
python3 experimental/tools/integration/run_specv2_integration.py bisection --n 19 --out out_19_7ns --cp 7.00 --resolver --use-prof-cache --transformed-code bisection_transformed.c
python3 experimental/tools/integration/run_specv2_integration.py bisection --n 20 --out out_20_7ns --cp 7.00 --resolver --use-prof-cache --transformed-code bisection_transformed.c

python3 experimental/tools/integration/run_specv2_integration.py sparse_dataspec --baseline --out out_baseline_7ns --cp 7.00
python3 experimental/tools/integration/run_adapted_spec_integration.py sparse_dataspec --out out_v1_7ns --cp 7.00 --default-value 0 --disable-constant-predictor
python3 experimental/tools/integration/run_specv2_integration.py sparse_dataspec --n 0 --out out_0_7ns --cp 7.00 --resolver
python3 experimental/tools/integration/run_specv2_integration.py sparse_dataspec --n 1 --out out_1_7ns --cp 7.00 --resolver --use-prof-cache
python3 experimental/tools/integration/run_specv2_integration.py sparse_dataspec --n 2 --out out_2_7ns --cp 7.00 --resolver --use-prof-cache
python3 experimental/tools/integration/run_specv2_integration.py sparse_dataspec --n 3 --out out_3_7ns --cp 7.00 --resolver --use-prof-cache
python3 experimental/tools/integration/run_specv2_integration.py sparse_dataspec --n 4 --out out_4_7ns --cp 7.00 --resolver --use-prof-cache
python3 experimental/tools/integration/run_specv2_integration.py sparse_dataspec --n 5 --out out_5_7ns --cp 7.00 --resolver --use-prof-cache
python3 experimental/tools/integration/run_specv2_integration.py sparse_dataspec --n 6 --out out_6_7ns --cp 7.00 --resolver --use-prof-cache
python3 experimental/tools/integration/run_specv2_integration.py sparse_dataspec --n 7 --out out_7_7ns --cp 7.00 --resolver --use-prof-cache
python3 experimental/tools/integration/run_specv2_integration.py sparse_dataspec --n 8 --out out_8_7ns --cp 7.00 --resolver --use-prof-cache
python3 experimental/tools/integration/run_specv2_integration.py sparse_dataspec --n 9 --out out_9_7ns --cp 7.00 --resolver --use-prof-cache
python3 experimental/tools/integration/run_specv2_integration.py sparse_dataspec --n 10 --out out_10_7ns --cp 7.00 --resolver --use-prof-cache
python3 experimental/tools/integration/run_specv2_integration.py sparse_dataspec --n 11 --out out_11_7ns --cp 7.00 --resolver --use-prof-cache
python3 experimental/tools/integration/run_specv2_integration.py sparse_dataspec --n 12 --out out_12_7ns --cp 7.00 --resolver --use-prof-cache
python3 experimental/tools/integration/run_specv2_integration.py sparse_dataspec --n 13 --out out_13_7ns --cp 7.00 --resolver --use-prof-cache
python3 experimental/tools/integration/run_specv2_integration.py sparse_dataspec --n 14 --out out_14_7ns --cp 7.00 --resolver --use-prof-cache
python3 experimental/tools/integration/run_specv2_integration.py sparse_dataspec --n 15 --out out_15_7ns --cp 7.00 --resolver --use-prof-cache
python3 experimental/tools/integration/run_specv2_integration.py sparse_dataspec --n 16 --out out_16_7ns --cp 7.00 --resolver --use-prof-cache
python3 experimental/tools/integration/run_specv2_integration.py sparse_dataspec --n 17 --out out_17_7ns --cp 7.00 --resolver --use-prof-cache
python3 experimental/tools/integration/run_specv2_integration.py sparse_dataspec --n 18 --out out_18_7ns --cp 7.00 --resolver --use-prof-cache
python3 experimental/tools/integration/run_specv2_integration.py sparse_dataspec --n 19 --out out_19_7ns --cp 7.00 --resolver --use-prof-cache
python3 experimental/tools/integration/run_specv2_integration.py sparse_dataspec --n 20 --out out_20_7ns --cp 7.00 --resolver --use-prof-cache

# On-merges
python3 experimental/tools/integration/run_specv2_integration.py single_loop --baseline --out out_baseline_7ns_om --cp 7.00 --resolver --min-buffering
python3 experimental/tools/integration/run_specv2_integration.py single_loop --n 1 --out out_1_7ns_om --cp 7.00 --resolver --use-prof-cache --min-buffering
python3 experimental/tools/integration/run_specv2_integration.py single_loop --n 2 --out out_2_7ns_om --cp 7.00 --resolver --use-prof-cache --min-buffering
python3 experimental/tools/integration/run_specv2_integration.py single_loop --n 3 --out out_3_7ns_om --cp 7.00 --resolver --use-prof-cache --min-buffering
python3 experimental/tools/integration/run_specv2_integration.py single_loop --n 4 --out out_4_7ns_om --cp 7.00 --resolver --use-prof-cache --min-buffering
python3 experimental/tools/integration/run_specv2_integration.py single_loop --n 5 --out out_5_7ns_om --cp 7.00 --resolver --use-prof-cache --min-buffering

python3 experimental/tools/integration/run_specv2_integration.py collision_donut --baseline --out out_baseline_7ns_om --cp 7.00 --min-buffering
python3 experimental/tools/integration/run_specv2_integration.py collision_donut --n 0 --out out_0_7ns_om --cp 7.00 --resolver --min-buffering
python3 experimental/tools/integration/run_specv2_integration.py collision_donut --n 1 --out out_1_7ns_om --cp 7.00 --use-prof-cache --resolver --min-buffering
python3 experimental/tools/integration/run_specv2_integration.py collision_donut --n 2 --out out_2_7ns_om --cp 7.00 --use-prof-cache --resolver --min-buffering
python3 experimental/tools/integration/run_specv2_integration.py collision_donut --n 3 --out out_3_7ns_om --cp 7.00 --use-prof-cache --resolver --min-buffering
python3 experimental/tools/integration/run_specv2_integration.py collision_donut --n 4 --out out_4_7ns_om --cp 7.00 --use-prof-cache --resolver --min-buffering
python3 experimental/tools/integration/run_specv2_integration.py collision_donut --n 5 --out out_5_7ns_om --cp 7.00 --use-prof-cache --resolver --min-buffering

python3 experimental/tools/integration/run_specv2_integration.py subdiag_fast --baseline --out out_baseline_7ns_om --transformed-code subdiag_fast_v1.c --cp 7.00 --min-buffering
python3 experimental/tools/integration/run_specv2_integration.py subdiag_fast --n 1 --out out_1_7ns_om --cp 7.00 --resolver --transformed-code subdiag_fast_v1.c --min-buffering
python3 experimental/tools/integration/run_specv2_integration.py subdiag_fast --n 2 --out out_2_7ns_om --cp 7.00 --resolver --transformed-code subdiag_fast_v1.c --use-prof-cache --min-buffering
python3 experimental/tools/integration/run_specv2_integration.py subdiag_fast --n 3 --out out_3_7ns_om --cp 7.00 --resolver --transformed-code subdiag_fast_v1.c --use-prof-cache --min-buffering
python3 experimental/tools/integration/run_specv2_integration.py subdiag_fast --n 4 --out out_4_7ns_om --cp 7.00 --resolver --transformed-code subdiag_fast_v1.c --use-prof-cache --min-buffering
python3 experimental/tools/integration/run_specv2_integration.py subdiag_fast --n 5 --out out_5_7ns_om --cp 7.00 --resolver --transformed-code subdiag_fast_v1.c --use-prof-cache --min-buffering
python3 experimental/tools/integration/run_specv2_integration.py subdiag_fast --n 6 --out out_6_7ns_om --cp 7.00 --resolver --transformed-code subdiag_fast_v1.c --use-prof-cache --min-buffering
python3 experimental/tools/integration/run_specv2_integration.py subdiag_fast --n 7 --out out_7_7ns_om --cp 7.00 --resolver --transformed-code subdiag_fast_v1.c --use-prof-cache --min-buffering
python3 experimental/tools/integration/run_specv2_integration.py subdiag_fast --n 8 --out out_8_7ns_om --cp 7.00 --resolver --transformed-code subdiag_fast_v1.c --use-prof-cache --min-buffering
python3 experimental/tools/integration/run_specv2_integration.py subdiag_fast --n 9 --out out_9_7ns_om --cp 7.00 --resolver --transformed-code subdiag_fast_v1.c --use-prof-cache --min-buffering
python3 experimental/tools/integration/run_specv2_integration.py subdiag_fast --n 10 --out out_10_7ns_om --cp 7.00 --resolver --transformed-code subdiag_fast_v1.c --use-prof-cache --min-buffering
python3 experimental/tools/integration/run_specv2_integration.py subdiag_fast --n 11 --out out_11_7ns_om --cp 7.00 --resolver --transformed-code subdiag_fast_v1.c --use-prof-cache --min-buffering
python3 experimental/tools/integration/run_specv2_integration.py subdiag_fast --n 12 --out out_12_7ns_om --cp 7.00 --resolver --transformed-code subdiag_fast_v1.c --use-prof-cache --min-buffering
python3 experimental/tools/integration/run_specv2_integration.py subdiag_fast --n 13 --out out_13_7ns_om --cp 7.00 --resolver --transformed-code subdiag_fast_v1.c --use-prof-cache --min-buffering
python3 experimental/tools/integration/run_specv2_integration.py subdiag_fast --n 14 --out out_14_7ns_om --cp 7.00 --resolver --transformed-code subdiag_fast_v1.c --use-prof-cache --min-buffering

python3 experimental/tools/integration/run_specv2_integration.py sparse_dataspec_transformed --baseline --out out_baseline_7ns_om --cp 7.00 --transformed-code sparse_dataspec_modified.c --min-buffering
python3 experimental/tools/integration/run_specv2_integration.py sparse_dataspec_transformed --n 1 --out out_1_7ns_om --cp 7.00 --resolver --transformed-code sparse_dataspec_modified.c --min-buffering
python3 experimental/tools/integration/run_specv2_integration.py sparse_dataspec_transformed --n 2 --out out_2_7ns_om --cp 7.00 --resolver --transformed-code sparse_dataspec_modified.c --min-buffering --use-prof-cache
python3 experimental/tools/integration/run_specv2_integration.py sparse_dataspec_transformed --n 3 --out out_3_7ns_om --cp 7.00 --resolver --transformed-code sparse_dataspec_modified.c --min-buffering --use-prof-cache
python3 experimental/tools/integration/run_specv2_integration.py sparse_dataspec_transformed --n 4 --out out_4_7ns_om --cp 7.00 --resolver --transformed-code sparse_dataspec_modified.c --min-buffering --use-prof-cache
python3 experimental/tools/integration/run_specv2_integration.py sparse_dataspec_transformed --n 5 --out out_5_7ns_om --cp 7.00 --resolver --transformed-code sparse_dataspec_modified.c --min-buffering --use-prof-cache

# Synthesis

timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/single_loop/out_baseline_7ns single_loop 7.000 3.500
timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/single_loop/out_v1_7ns single_loop 7.000 3.500
timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/single_loop/out_4_7ns single_loop 7.000 3.500

timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/nested_loop/out_v1_7ns nested_loop 7.000 3.500
timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/nested_loop/out_baseline_7ns nested_loop 7.000 3.500
timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/nested_loop/out_4_7ns nested_loop 7.000 3.500

timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/fixed_log/out_baseline_7ns fixed_log 7.000 3.500
timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/fixed_log/out_v1_7ns fixed_log 7.000 3.500
timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/fixed_log/out_3_7ns fixed_log 7.000 3.500

# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/newton/out_baseline_7ns newton 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/newton/out_v1_7ns newton 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/newton/out_1_7ns newton 7.000 3.500

# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/subdiag_fast/out_baseline_7ns subdiag_fast 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/subdiag_fast/out_v1_7ns subdiag_fast 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/subdiag_fast/out_13_7ns subdiag_fast 7.000 3.500

timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/golden_ratio/out_baseline_7ns golden_ratio 7.000 3.500
timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/golden_ratio/out_v1_7ns golden_ratio 7.000 3.500
timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/golden_ratio/out_1_exit_7ns golden_ratio 7.000 3.500

timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/collision_donut/out_baseline_7ns collision_donut 7.000 3.500
timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/collision_donut/out_5_7ns collision_donut 7.000 3.500

timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/bisection/out_baseline_7ns bisection 7.000 3.500
timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/bisection/out_2_7ns bisection 7.000 3.500

timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/sparse_dataspec/out_baseline_7ns sparse_dataspec 7.000 3.500
timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/sparse_dataspec/out_v1_7ns sparse_dataspec 7.000 3.500
timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/sparse_dataspec/out_8_7ns sparse_dataspec 7.000 3.500

timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/sparse_dataspec_transformed/out_baseline_7ns sparse_dataspec_transformed 7.000 3.500
timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/sparse_dataspec_transformed/out_v1_7ns sparse_dataspec_transformed 7.000 3.500
timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/sparse_dataspec_transformed/out_4_7ns sparse_dataspec_transformed 7.000 3.500

timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/if_float/out_v1_7ns if_float 7.000 3.500
timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/if_float/out_baseline_7ns if_float 7.000 3.500
timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/if_float/out_0_one_sided_7ns if_float 7.000 3.500
timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/if_float/out_1_one_sided_7ns if_float 7.000 3.500
timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/if_float/out_0_two_sided_7ns if_float 7.000 3.500
timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/if_float/out_1_two_sided_7ns if_float 7.000 3.500

timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/if_float2/out_baseline_7ns if_float2 7.000 3.500
timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/if_float2/out_v1_7ns if_float2 7.000 3.500
timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/if_float2/out_0_one_sided_7ns if_float2 7.000 3.500
timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/if_float2/out_1_one_sided_7ns if_float2 7.000 3.500
timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/if_float2/out_0_two_sided_7ns if_float2 7.000 3.500
timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/if_float2/out_1_two_sided_7ns if_float2 7.000 3.500

# On-merge synthesis
timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/single_loop/out_baseline_7ns_om single_loop 7.000 3.500
timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/single_loop/out_1_7ns_om single_loop 7.000 3.500
timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/single_loop/out_2_7ns_om single_loop 7.000 3.500
timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/single_loop/out_3_7ns_om single_loop 7.000 3.500
timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/single_loop/out_4_7ns_om single_loop 7.000 3.500
timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/single_loop/out_5_7ns_om single_loop 7.000 3.500

timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/collision_donut/out_baseline_7ns_om collision_donut 7.000 3.500
timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/collision_donut/out_0_7ns_om collision_donut 7.000 3.500
timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/collision_donut/out_1_7ns_om collision_donut 7.000 3.500
timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/collision_donut/out_2_7ns_om collision_donut 7.000 3.500
timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/collision_donut/out_3_7ns_om collision_donut 7.000 3.500
timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/collision_donut/out_4_7ns_om collision_donut 7.000 3.500
timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/collision_donut/out_5_7ns_om collision_donut 7.000 3.500

timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/subdiag_fast/out_baseline_7ns_om subdiag_fast 7.000 3.500
timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/subdiag_fast/out_1_7ns_om subdiag_fast 7.000 3.500
timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/subdiag_fast/out_2_7ns_om subdiag_fast 7.000 3.500
timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/subdiag_fast/out_3_7ns_om subdiag_fast 7.000 3.500
timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/subdiag_fast/out_4_7ns_om subdiag_fast 7.000 3.500
timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/subdiag_fast/out_5_7ns_om subdiag_fast 7.000 3.500
timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/subdiag_fast/out_6_7ns_om subdiag_fast 7.000 3.500
timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/subdiag_fast/out_7_7ns_om subdiag_fast 7.000 3.500
timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/subdiag_fast/out_8_7ns_om subdiag_fast 7.000 3.500
timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/subdiag_fast/out_9_7ns_om subdiag_fast 7.000 3.500
timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/subdiag_fast/out_10_7ns_om subdiag_fast 7.000 3.500
timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/subdiag_fast/out_11_7ns_om subdiag_fast 7.000 3.500
timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/subdiag_fast/out_12_7ns_om subdiag_fast 7.000 3.500
timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/subdiag_fast/out_13_7ns_om subdiag_fast 7.000 3.500
timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/subdiag_fast/out_14_7ns_om subdiag_fast 7.000 3.500

timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/sparse_dataspec_transformed/out_baseline_7ns_om sparse_dataspec_transformed 7.000 3.500
timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/sparse_dataspec_transformed/out_1_7ns_om sparse_dataspec_transformed 7.000 3.500
timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/sparse_dataspec_transformed/out_2_7ns_om sparse_dataspec_transformed 7.000 3.500
timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/sparse_dataspec_transformed/out_3_7ns_om sparse_dataspec_transformed 7.000 3.500
timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/sparse_dataspec_transformed/out_4_7ns_om sparse_dataspec_transformed 7.000 3.500

date > end_time.txt

systemctl suspend
