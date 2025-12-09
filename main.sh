SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DYNAMATIC_DIR="$(cd "$SCRIPT_DIR" && pwd)"

cd $DYNAMATIC_DIR

date > start_time.txt

# Simulation (Table 2, Fig. 7)

# python3 experimental/tools/integration/run_specv2_integration.py single_loop --baseline --out out_baseline_7ns --cp 7.00
# python3 experimental/tools/integration/run_adapted_spec_integration.py single_loop --out out_v1_7ns --cp 7.00
# python3 experimental/tools/integration/run_specv2_integration.py single_loop --out out_auto_7ns --cp 7.00 --resolver --decide-n 0
# python3 experimental/tools/integration/run_specv2_integration.py single_loop --n 0 --out out_0_7ns --cp 7.00 --resolver
# python3 experimental/tools/integration/run_specv2_integration.py single_loop --n 1 --out out_1_7ns --cp 7.00 --resolver
# python3 experimental/tools/integration/run_specv2_integration.py single_loop --n 2 --out out_2_7ns --cp 7.00 --resolver --use-prof-cache
# python3 experimental/tools/integration/run_specv2_integration.py single_loop --n 3 --out out_3_7ns --cp 7.00 --resolver --use-prof-cache
# python3 experimental/tools/integration/run_specv2_integration.py single_loop --n 4 --out out_4_7ns --cp 7.00 --resolver --use-prof-cache
# python3 experimental/tools/integration/run_specv2_integration.py single_loop --n 5 --out out_5_7ns --cp 7.00 --resolver --use-prof-cache
# python3 experimental/tools/integration/run_specv2_integration.py single_loop --n 6 --out out_6_7ns --cp 7.00 --resolver --use-prof-cache
# python3 experimental/tools/integration/run_specv2_integration.py single_loop --n 7 --out out_7_7ns --cp 7.00 --resolver --use-prof-cache
# python3 experimental/tools/integration/run_specv2_integration.py single_loop --n 8 --out out_8_7ns --cp 7.00 --resolver --use-prof-cache
# python3 experimental/tools/integration/run_specv2_integration.py single_loop --n 9 --out out_9_7ns --cp 7.00 --resolver --use-prof-cache
# python3 experimental/tools/integration/run_specv2_integration.py single_loop --n 10 --out out_10_7ns --cp 7.00 --resolver --use-prof-cache
# python3 experimental/tools/integration/run_specv2_integration.py single_loop --n 11 --out out_11_7ns --cp 7.00 --resolver --use-prof-cache
# python3 experimental/tools/integration/run_specv2_integration.py single_loop --n 12 --out out_12_7ns --cp 7.00 --resolver --use-prof-cache
# python3 experimental/tools/integration/run_specv2_integration.py single_loop --n 13 --out out_13_7ns --cp 7.00 --resolver --use-prof-cache
# python3 experimental/tools/integration/run_specv2_integration.py single_loop --n 14 --out out_14_7ns --cp 7.00 --resolver --use-prof-cache
# python3 experimental/tools/integration/run_specv2_integration.py single_loop --n 15 --out out_15_7ns --cp 7.00 --resolver --use-prof-cache
# python3 experimental/tools/integration/run_specv2_integration.py single_loop --n 16 --out out_16_7ns --cp 7.00 --resolver --use-prof-cache
# python3 experimental/tools/integration/run_specv2_integration.py single_loop --n 17 --out out_17_7ns --cp 7.00 --resolver --use-prof-cache
# python3 experimental/tools/integration/run_specv2_integration.py single_loop --n 18 --out out_18_7ns --cp 7.00 --resolver --use-prof-cache
# python3 experimental/tools/integration/run_specv2_integration.py single_loop --n 19 --out out_19_7ns --cp 7.00 --resolver --use-prof-cache
# python3 experimental/tools/integration/run_specv2_integration.py single_loop --n 20 --out out_20_7ns --cp 7.00 --resolver --use-prof-cache

# python3 experimental/tools/integration/run_specv2_integration.py nested_loop --baseline --out out_baseline_7ns --cp 7.00
# python3 experimental/tools/integration/run_adapted_spec_integration.py nested_loop --out out_v1_7ns --cp 7.00
# python3 experimental/tools/integration/run_specv2_integration.py nested_loop --out out_auto_7ns --cp 7.00 --resolver --decide-n 0
# python3 experimental/tools/integration/run_specv2_integration.py nested_loop --n 0 --out out_0_7ns --cp 7.00 --resolver
# python3 experimental/tools/integration/run_specv2_integration.py nested_loop --n 1 --out out_1_7ns --cp 7.00 --use-prof-cache --resolver
# python3 experimental/tools/integration/run_specv2_integration.py nested_loop --n 2 --out out_2_7ns --cp 7.00 --use-prof-cache --resolver
# python3 experimental/tools/integration/run_specv2_integration.py nested_loop --n 3 --out out_3_7ns --cp 7.00 --use-prof-cache --resolver
# python3 experimental/tools/integration/run_specv2_integration.py nested_loop --n 4 --out out_4_7ns --cp 7.00 --use-prof-cache --resolver
# python3 experimental/tools/integration/run_specv2_integration.py nested_loop --n 5 --out out_5_7ns --cp 7.00 --use-prof-cache --resolver
# python3 experimental/tools/integration/run_specv2_integration.py nested_loop --n 6 --out out_6_7ns --cp 7.00 --use-prof-cache --resolver
# python3 experimental/tools/integration/run_specv2_integration.py nested_loop --n 7 --out out_7_7ns --cp 7.00 --use-prof-cache --resolver
# python3 experimental/tools/integration/run_specv2_integration.py nested_loop --n 8 --out out_8_7ns --cp 7.00 --use-prof-cache --resolver
# python3 experimental/tools/integration/run_specv2_integration.py nested_loop --n 9 --out out_9_7ns --cp 7.00 --use-prof-cache --resolver
# python3 experimental/tools/integration/run_specv2_integration.py nested_loop --n 10 --out out_10_7ns --cp 7.00 --use-prof-cache --resolver
# python3 experimental/tools/integration/run_specv2_integration.py nested_loop --n 11 --out out_11_7ns --cp 7.00 --use-prof-cache --resolver
# python3 experimental/tools/integration/run_specv2_integration.py nested_loop --n 12 --out out_12_7ns --cp 7.00 --use-prof-cache --resolver
# python3 experimental/tools/integration/run_specv2_integration.py nested_loop --n 13 --out out_13_7ns --cp 7.00 --use-prof-cache --resolver
# python3 experimental/tools/integration/run_specv2_integration.py nested_loop --n 14 --out out_14_7ns --cp 7.00 --use-prof-cache --resolver
# python3 experimental/tools/integration/run_specv2_integration.py nested_loop --n 15 --out out_15_7ns --cp 7.00 --use-prof-cache --resolver
# python3 experimental/tools/integration/run_specv2_integration.py nested_loop --n 16 --out out_16_7ns --cp 7.00 --use-prof-cache --resolver
# python3 experimental/tools/integration/run_specv2_integration.py nested_loop --n 17 --out out_17_7ns --cp 7.00 --use-prof-cache --resolver
# python3 experimental/tools/integration/run_specv2_integration.py nested_loop --n 18 --out out_18_7ns --cp 7.00 --use-prof-cache --resolver
# python3 experimental/tools/integration/run_specv2_integration.py nested_loop --n 19 --out out_19_7ns --cp 7.00 --use-prof-cache --resolver
# python3 experimental/tools/integration/run_specv2_integration.py nested_loop --n 20 --out out_20_7ns --cp 7.00 --use-prof-cache --resolver

# python3 experimental/tools/integration/run_specv2_integration.py fixed_log --baseline --out out_baseline_7ns --cp 7.00
# python3 experimental/tools/integration/run_adapted_spec_integration.py fixed_log --default-value=0 --out out_v1_7ns --cp 7.00
# python3 experimental/tools/integration/run_specv2_integration.py fixed_log --out out_auto_7ns --cp 7.00 --resolver --decide-n 0
# python3 experimental/tools/integration/run_specv2_integration.py fixed_log --n 0 --out out_0_7ns --cp 7.00 --resolver
# python3 experimental/tools/integration/run_specv2_integration.py fixed_log --n 1 --out out_1_7ns --cp 7.00 --resolver
# python3 experimental/tools/integration/run_specv2_integration.py fixed_log --n 2 --out out_2_7ns --cp 7.00 --resolver --use-prof-cache
# python3 experimental/tools/integration/run_specv2_integration.py fixed_log --n 3 --out out_3_7ns --cp 7.00 --resolver --use-prof-cache
# python3 experimental/tools/integration/run_specv2_integration.py fixed_log --n 4 --out out_4_7ns --cp 7.00 --resolver --use-prof-cache
# python3 experimental/tools/integration/run_specv2_integration.py fixed_log --n 5 --out out_5_7ns --cp 7.00 --resolver --use-prof-cache
# python3 experimental/tools/integration/run_specv2_integration.py fixed_log --n 6 --out out_6_7ns --cp 7.00 --resolver --use-prof-cache
# python3 experimental/tools/integration/run_specv2_integration.py fixed_log --n 7 --out out_7_7ns --cp 7.00 --resolver --use-prof-cache
# python3 experimental/tools/integration/run_specv2_integration.py fixed_log --n 8 --out out_8_7ns --cp 7.00 --resolver --use-prof-cache
# python3 experimental/tools/integration/run_specv2_integration.py fixed_log --n 9 --out out_9_7ns --cp 7.00 --resolver --use-prof-cache
# python3 experimental/tools/integration/run_specv2_integration.py fixed_log --n 10 --out out_10_7ns --cp 7.00 --resolver --use-prof-cache
# python3 experimental/tools/integration/run_specv2_integration.py fixed_log --n 11 --out out_11_7ns --cp 7.00 --resolver --use-prof-cache
# python3 experimental/tools/integration/run_specv2_integration.py fixed_log --n 12 --out out_12_7ns --cp 7.00 --resolver --use-prof-cache
# python3 experimental/tools/integration/run_specv2_integration.py fixed_log --n 13 --out out_13_7ns --cp 7.00 --resolver --use-prof-cache
# python3 experimental/tools/integration/run_specv2_integration.py fixed_log --n 14 --out out_14_7ns --cp 7.00 --resolver --use-prof-cache
# python3 experimental/tools/integration/run_specv2_integration.py fixed_log --n 15 --out out_15_7ns --cp 7.00 --resolver --use-prof-cache
# python3 experimental/tools/integration/run_specv2_integration.py fixed_log --n 16 --out out_16_7ns --cp 7.00 --resolver --use-prof-cache
# python3 experimental/tools/integration/run_specv2_integration.py fixed_log --n 17 --out out_17_7ns --cp 7.00 --resolver --use-prof-cache
# python3 experimental/tools/integration/run_specv2_integration.py fixed_log --n 18 --out out_18_7ns --cp 7.00 --resolver --use-prof-cache
# python3 experimental/tools/integration/run_specv2_integration.py fixed_log --n 19 --out out_19_7ns --cp 7.00 --resolver --use-prof-cache
# python3 experimental/tools/integration/run_specv2_integration.py fixed_log --n 20 --out out_20_7ns --cp 7.00 --resolver --use-prof-cache


# python3 experimental/tools/integration/run_specv2_integration.py newton --baseline --out out_baseline_7ns --cp 7.00
# python3 experimental/tools/integration/run_adapted_spec_integration.py newton --out out_v1_7ns --cp 7.00 --default-value 0
# python3 experimental/tools/integration/run_specv2_integration.py newton --out out_auto_7ns --cp 7.00 --resolver --decide-n 0
# python3 experimental/tools/integration/run_specv2_integration.py newton --n 0 --out out_0_7ns --cp 7.00 --resolver
# python3 experimental/tools/integration/run_specv2_integration.py newton --n 1 --out out_1_7ns --cp 7.00 --resolver
# python3 experimental/tools/integration/run_specv2_integration.py newton --n 2 --out out_2_7ns --cp 7.00 --resolver --use-prof-cache
# python3 experimental/tools/integration/run_specv2_integration.py newton --n 3 --out out_3_7ns --cp 7.00 --resolver --use-prof-cache
# python3 experimental/tools/integration/run_specv2_integration.py newton --n 4 --out out_4_7ns --cp 7.00 --resolver --use-prof-cache
# python3 experimental/tools/integration/run_specv2_integration.py newton --n 5 --out out_5_7ns --cp 7.00 --resolver --use-prof-cache
# python3 experimental/tools/integration/run_specv2_integration.py newton --n 6 --out out_6_7ns --cp 7.00 --resolver --use-prof-cache
# python3 experimental/tools/integration/run_specv2_integration.py newton --n 7 --out out_7_7ns --cp 7.00 --resolver --use-prof-cache
# python3 experimental/tools/integration/run_specv2_integration.py newton --n 8 --out out_8_7ns --cp 7.00 --resolver --use-prof-cache
# python3 experimental/tools/integration/run_specv2_integration.py newton --n 9 --out out_9_7ns --cp 7.00 --resolver --use-prof-cache
# python3 experimental/tools/integration/run_specv2_integration.py newton --n 10 --out out_10_7ns --cp 7.00 --resolver --use-prof-cache
# python3 experimental/tools/integration/run_specv2_integration.py newton --n 11 --out out_11_7ns --cp 7.00 --resolver --use-prof-cache
# python3 experimental/tools/integration/run_specv2_integration.py newton --n 12 --out out_12_7ns --cp 7.00 --resolver --use-prof-cache
# python3 experimental/tools/integration/run_specv2_integration.py newton --n 13 --out out_13_7ns --cp 7.00 --resolver --use-prof-cache
# python3 experimental/tools/integration/run_specv2_integration.py newton --n 14 --out out_14_7ns --cp 7.00 --resolver --use-prof-cache
# python3 experimental/tools/integration/run_specv2_integration.py newton --n 15 --out out_15_7ns --cp 7.00 --resolver --use-prof-cache
# python3 experimental/tools/integration/run_specv2_integration.py newton --n 16 --out out_16_7ns --cp 7.00 --resolver --use-prof-cache
# python3 experimental/tools/integration/run_specv2_integration.py newton --n 17 --out out_17_7ns --cp 7.00 --resolver --use-prof-cache
# python3 experimental/tools/integration/run_specv2_integration.py newton --n 18 --out out_18_7ns --cp 7.00 --resolver --use-prof-cache
# python3 experimental/tools/integration/run_specv2_integration.py newton --n 19 --out out_19_7ns --cp 7.00 --resolver --use-prof-cache
# python3 experimental/tools/integration/run_specv2_integration.py newton --n 20 --out out_20_7ns --cp 7.00 --resolver --use-prof-cache

# python3 experimental/tools/integration/run_specv2_integration.py subdiag_fast --baseline --out out_baseline_7ns --cp 7.00 --transformed-code subdiag_fast_v1.c
# python3 experimental/tools/integration/run_adapted_spec_integration.py subdiag_fast --out out_v1_7ns --cp 7.00 --transformed-code subdiag_fast_v1.c
# python3 experimental/tools/integration/run_specv2_integration.py subdiag_fast --out out_auto_7ns --cp 7.00 --resolver --decide-n 0 --transformed-code subdiag_fast_v1.c
# python3 experimental/tools/integration/run_specv2_integration.py subdiag_fast --n 0 --out out_0_7ns --cp 7.00 --resolver --transformed-code subdiag_fast_v1.c
# python3 experimental/tools/integration/run_specv2_integration.py subdiag_fast --n 1 --out out_1_7ns --cp 7.00 --resolver --transformed-code subdiag_fast_v1.c
# python3 experimental/tools/integration/run_specv2_integration.py subdiag_fast --n 2 --out out_2_7ns --cp 7.00 --resolver --transformed-code subdiag_fast_v1.c --use-prof-cache
# python3 experimental/tools/integration/run_specv2_integration.py subdiag_fast --n 3 --out out_3_7ns --cp 7.00 --resolver --transformed-code subdiag_fast_v1.c --use-prof-cache
# python3 experimental/tools/integration/run_specv2_integration.py subdiag_fast --n 4 --out out_4_7ns --cp 7.00 --resolver --transformed-code subdiag_fast_v1.c --use-prof-cache
# python3 experimental/tools/integration/run_specv2_integration.py subdiag_fast --n 5 --out out_5_7ns --cp 7.00 --resolver --transformed-code subdiag_fast_v1.c --use-prof-cache
# python3 experimental/tools/integration/run_specv2_integration.py subdiag_fast --n 6 --out out_6_7ns --cp 7.00 --resolver --transformed-code subdiag_fast_v1.c --use-prof-cache
# python3 experimental/tools/integration/run_specv2_integration.py subdiag_fast --n 7 --out out_7_7ns --cp 7.00 --resolver --transformed-code subdiag_fast_v1.c --use-prof-cache
# python3 experimental/tools/integration/run_specv2_integration.py subdiag_fast --n 8 --out out_8_7ns --cp 7.00 --resolver --transformed-code subdiag_fast_v1.c --use-prof-cache
# python3 experimental/tools/integration/run_specv2_integration.py subdiag_fast --n 9 --out out_9_7ns --cp 7.00 --resolver --transformed-code subdiag_fast_v1.c --use-prof-cache
# python3 experimental/tools/integration/run_specv2_integration.py subdiag_fast --n 10 --out out_10_7ns --cp 7.00 --resolver --transformed-code subdiag_fast_v1.c --use-prof-cache
# python3 experimental/tools/integration/run_specv2_integration.py subdiag_fast --n 11 --out out_11_7ns --cp 7.00 --resolver --transformed-code subdiag_fast_v1.c --use-prof-cache
# python3 experimental/tools/integration/run_specv2_integration.py subdiag_fast --n 12 --out out_12_7ns --cp 7.00 --resolver --transformed-code subdiag_fast_v1.c --use-prof-cache
# python3 experimental/tools/integration/run_specv2_integration.py subdiag_fast --n 13 --out out_13_7ns --cp 7.00 --resolver --transformed-code subdiag_fast_v1.c --use-prof-cache
# python3 experimental/tools/integration/run_specv2_integration.py subdiag_fast --n 14 --out out_14_7ns --cp 7.00 --resolver --transformed-code subdiag_fast_v1.c --use-prof-cache
# python3 experimental/tools/integration/run_specv2_integration.py subdiag_fast --n 15 --out out_15_7ns --cp 7.00 --resolver --transformed-code subdiag_fast_v1.c --use-prof-cache
# python3 experimental/tools/integration/run_specv2_integration.py subdiag_fast --n 16 --out out_16_7ns --cp 7.00 --resolver --transformed-code subdiag_fast_v1.c --use-prof-cache
# python3 experimental/tools/integration/run_specv2_integration.py subdiag_fast --n 17 --out out_17_7ns --cp 7.00 --resolver --transformed-code subdiag_fast_v1.c --use-prof-cache
# python3 experimental/tools/integration/run_specv2_integration.py subdiag_fast --n 18 --out out_18_7ns --cp 7.00 --resolver --transformed-code subdiag_fast_v1.c --use-prof-cache
# python3 experimental/tools/integration/run_specv2_integration.py subdiag_fast --n 19 --out out_19_7ns --cp 7.00 --resolver --transformed-code subdiag_fast_v1.c --use-prof-cache
# python3 experimental/tools/integration/run_specv2_integration.py subdiag_fast --n 20 --out out_20_7ns --cp 7.00 --resolver --transformed-code subdiag_fast_v1.c --use-prof-cache

# python3 experimental/tools/integration/run_specv2_integration.py golden_ratio --baseline --out out_baseline_7ns --cp 7.00
# python3 experimental/tools/integration/run_adapted_spec_integration.py golden_ratio --out out_v1_7ns --cp 7.00 --default-value 0
# python3 experimental/tools/integration/run_specv2_integration.py golden_ratio --out out_auto_7ns --cp 7.00 --resolver --exit-eager-eval --decide-n 0
# python3 experimental/tools/integration/run_specv2_integration.py golden_ratio --n 0 --out out_0_7ns --cp 7.00 --exit-eager-eval --resolver
# python3 experimental/tools/integration/run_specv2_integration.py golden_ratio --n 1 --out out_1_7ns --cp 7.00 --exit-eager-eval --resolver
# python3 experimental/tools/integration/run_specv2_integration.py golden_ratio --n 2 --out out_2_7ns --cp 7.00 --exit-eager-eval --resolver --use-prof-cache
# python3 experimental/tools/integration/run_specv2_integration.py golden_ratio --n 3 --out out_3_7ns --cp 7.00 --exit-eager-eval --resolver --use-prof-cache
# python3 experimental/tools/integration/run_specv2_integration.py golden_ratio --n 4 --out out_4_7ns --cp 7.00 --exit-eager-eval --resolver --use-prof-cache
# python3 experimental/tools/integration/run_specv2_integration.py golden_ratio --n 5 --out out_5_7ns --cp 7.00 --exit-eager-eval --resolver --use-prof-cache
# python3 experimental/tools/integration/run_specv2_integration.py golden_ratio --n 6 --out out_6_7ns --cp 7.00 --exit-eager-eval --resolver --use-prof-cache
# python3 experimental/tools/integration/run_specv2_integration.py golden_ratio --n 7 --out out_7_7ns --cp 7.00 --exit-eager-eval --resolver --use-prof-cache
# python3 experimental/tools/integration/run_specv2_integration.py golden_ratio --n 8 --out out_8_7ns --cp 7.00 --exit-eager-eval --resolver --use-prof-cache
# python3 experimental/tools/integration/run_specv2_integration.py golden_ratio --n 9 --out out_9_7ns --cp 7.00 --exit-eager-eval --resolver --use-prof-cache
# python3 experimental/tools/integration/run_specv2_integration.py golden_ratio --n 10 --out out_10_7ns --cp 7.00 --exit-eager-eval --resolver --use-prof-cache
# python3 experimental/tools/integration/run_specv2_integration.py golden_ratio --n 11 --out out_11_7ns --cp 7.00 --exit-eager-eval --resolver
# python3 experimental/tools/integration/run_specv2_integration.py golden_ratio --n 12 --out out_12_7ns --cp 7.00 --exit-eager-eval --resolver
# python3 experimental/tools/integration/run_specv2_integration.py golden_ratio --n 13 --out out_13_7ns --cp 7.00 --exit-eager-eval --resolver --use-prof-cache
# python3 experimental/tools/integration/run_specv2_integration.py golden_ratio --n 14 --out out_14_7ns --cp 7.00 --exit-eager-eval --resolver --use-prof-cache
# python3 experimental/tools/integration/run_specv2_integration.py golden_ratio --n 15 --out out_15_7ns --cp 7.00 --exit-eager-eval --resolver --use-prof-cache
# python3 experimental/tools/integration/run_specv2_integration.py golden_ratio --n 16 --out out_16_7ns --cp 7.00 --exit-eager-eval --resolver --use-prof-cache
# python3 experimental/tools/integration/run_specv2_integration.py golden_ratio --n 17 --out out_17_7ns --cp 7.00 --exit-eager-eval --resolver --use-prof-cache
# python3 experimental/tools/integration/run_specv2_integration.py golden_ratio --n 18 --out out_18_7ns --cp 7.00 --exit-eager-eval --resolver --use-prof-cache
# python3 experimental/tools/integration/run_specv2_integration.py golden_ratio --n 19 --out out_19_7ns --cp 7.00 --exit-eager-eval --resolver --use-prof-cache
# python3 experimental/tools/integration/run_specv2_integration.py golden_ratio --n 20 --out out_20_7ns --cp 7.00 --exit-eager-eval --resolver --use-prof-cache

# python3 experimental/tools/integration/run_specv2_integration.py collision_donut --baseline --out out_baseline_7ns --cp 7.00
# python3 experimental/tools/integration/run_specv2_integration.py collision_donut --out out_auto_7ns --cp 7.00 --resolver --decide-n 0
# python3 experimental/tools/integration/run_specv2_integration.py collision_donut --n 0 --out out_0_7ns --cp 7.00 --resolver
# python3 experimental/tools/integration/run_specv2_integration.py collision_donut --n 1 --out out_1_7ns --cp 7.00 --use-prof-cache --resolver
# python3 experimental/tools/integration/run_specv2_integration.py collision_donut --n 2 --out out_2_7ns --cp 7.00 --use-prof-cache --resolver
# python3 experimental/tools/integration/run_specv2_integration.py collision_donut --n 3 --out out_3_7ns --cp 7.00 --use-prof-cache --resolver
# python3 experimental/tools/integration/run_specv2_integration.py collision_donut --n 4 --out out_4_7ns --cp 7.00 --use-prof-cache --resolver
# python3 experimental/tools/integration/run_specv2_integration.py collision_donut --n 5 --out out_5_7ns --cp 7.00 --use-prof-cache --resolver
# python3 experimental/tools/integration/run_specv2_integration.py collision_donut --n 6 --out out_6_7ns --cp 7.00 --use-prof-cache --resolver
# python3 experimental/tools/integration/run_specv2_integration.py collision_donut --n 7 --out out_7_7ns --cp 7.00 --use-prof-cache --resolver
# python3 experimental/tools/integration/run_specv2_integration.py collision_donut --n 8 --out out_8_7ns --cp 7.00 --use-prof-cache --resolver
# python3 experimental/tools/integration/run_specv2_integration.py collision_donut --n 9 --out out_9_7ns --cp 7.00 --use-prof-cache --resolver
# python3 experimental/tools/integration/run_specv2_integration.py collision_donut --n 10 --out out_10_7ns --cp 7.00 --use-prof-cache --resolver
# python3 experimental/tools/integration/run_specv2_integration.py collision_donut --n 11 --out out_11_7ns --cp 7.00 --use-prof-cache --resolver
# python3 experimental/tools/integration/run_specv2_integration.py collision_donut --n 12 --out out_12_7ns --cp 7.00 --use-prof-cache --resolver
# python3 experimental/tools/integration/run_specv2_integration.py collision_donut --n 13 --out out_13_7ns --cp 7.00 --use-prof-cache --resolver
# python3 experimental/tools/integration/run_specv2_integration.py collision_donut --n 14 --out out_14_7ns --cp 7.00 --use-prof-cache --resolver
# python3 experimental/tools/integration/run_specv2_integration.py collision_donut --n 15 --out out_15_7ns --cp 7.00 --use-prof-cache --resolver
# python3 experimental/tools/integration/run_specv2_integration.py collision_donut --n 16 --out out_16_7ns --cp 7.00 --use-prof-cache --resolver
# python3 experimental/tools/integration/run_specv2_integration.py collision_donut --n 17 --out out_17_7ns --cp 7.00 --use-prof-cache --resolver
# python3 experimental/tools/integration/run_specv2_integration.py collision_donut --n 18 --out out_18_7ns --cp 7.00 --use-prof-cache --resolver
# python3 experimental/tools/integration/run_specv2_integration.py collision_donut --n 19 --out out_19_7ns --cp 7.00 --use-prof-cache --resolver
# python3 experimental/tools/integration/run_specv2_integration.py collision_donut --n 20 --out out_20_7ns --cp 7.00 --use-prof-cache --resolver


# python3 experimental/tools/integration/run_specv2_integration.py bisection --baseline --out out_baseline_7ns --cp 7.00 --transformed-code bisection_transformed.c
# python3 experimental/tools/integration/run_specv2_integration.py bisection --out out_auto_7ns --cp 7.00 --resolver --decide-n 0 --transformed-code bisection_transformed.c
# python3 experimental/tools/integration/run_specv2_integration.py bisection --n 0 --out out_0_7ns --cp 7.00 --resolver --transformed-code bisection_transformed.c
# python3 experimental/tools/integration/run_specv2_integration.py bisection --n 1 --out out_1_7ns --cp 7.00 --resolver --transformed-code bisection_transformed.c
# python3 experimental/tools/integration/run_specv2_integration.py bisection --n 2 --out out_2_7ns --cp 7.00 --resolver --transformed-code bisection_transformed.c
# python3 experimental/tools/integration/run_specv2_integration.py bisection --n 3 --out out_3_7ns --cp 7.00 --resolver --use-prof-cache --transformed-code bisection_transformed.c
# python3 experimental/tools/integration/run_specv2_integration.py bisection --n 4 --out out_4_7ns --cp 7.00 --resolver --use-prof-cache --transformed-code bisection_transformed.c
# python3 experimental/tools/integration/run_specv2_integration.py bisection --n 5 --out out_5_7ns --cp 7.00 --resolver --use-prof-cache --transformed-code bisection_transformed.c
# python3 experimental/tools/integration/run_specv2_integration.py bisection --n 6 --out out_6_7ns --cp 7.00 --resolver --use-prof-cache --transformed-code bisection_transformed.c
# python3 experimental/tools/integration/run_specv2_integration.py bisection --n 7 --out out_7_7ns --cp 7.00 --resolver --use-prof-cache --transformed-code bisection_transformed.c
# python3 experimental/tools/integration/run_specv2_integration.py bisection --n 8 --out out_8_7ns --cp 7.00 --resolver --use-prof-cache --transformed-code bisection_transformed.c
# python3 experimental/tools/integration/run_specv2_integration.py bisection --n 9 --out out_9_7ns --cp 7.00 --resolver --use-prof-cache --transformed-code bisection_transformed.c
# python3 experimental/tools/integration/run_specv2_integration.py bisection --n 10 --out out_10_7ns --cp 7.00 --resolver --use-prof-cache --transformed-code bisection_transformed.c
# python3 experimental/tools/integration/run_specv2_integration.py bisection --n 11 --out out_11_7ns --cp 7.00 --resolver --use-prof-cache --transformed-code bisection_transformed.c
# python3 experimental/tools/integration/run_specv2_integration.py bisection --n 12 --out out_12_7ns --cp 7.00 --resolver --use-prof-cache --transformed-code bisection_transformed.c
# python3 experimental/tools/integration/run_specv2_integration.py bisection --n 13 --out out_13_7ns --cp 7.00 --resolver --use-prof-cache --transformed-code bisection_transformed.c
# python3 experimental/tools/integration/run_specv2_integration.py bisection --n 14 --out out_14_7ns --cp 7.00 --resolver --use-prof-cache --transformed-code bisection_transformed.c
# python3 experimental/tools/integration/run_specv2_integration.py bisection --n 15 --out out_15_7ns --cp 7.00 --resolver --use-prof-cache --transformed-code bisection_transformed.c
# python3 experimental/tools/integration/run_specv2_integration.py bisection --n 16 --out out_16_7ns --cp 7.00 --resolver --use-prof-cache --transformed-code bisection_transformed.c
# python3 experimental/tools/integration/run_specv2_integration.py bisection --n 17 --out out_17_7ns --cp 7.00 --resolver --use-prof-cache --transformed-code bisection_transformed.c
# python3 experimental/tools/integration/run_specv2_integration.py bisection --n 18 --out out_18_7ns --cp 7.00 --resolver --use-prof-cache --transformed-code bisection_transformed.c
# python3 experimental/tools/integration/run_specv2_integration.py bisection --n 19 --out out_19_7ns --cp 7.00 --resolver --use-prof-cache --transformed-code bisection_transformed.c
# python3 experimental/tools/integration/run_specv2_integration.py bisection --n 20 --out out_20_7ns --cp 7.00 --resolver --use-prof-cache --transformed-code bisection_transformed.c

# python3 experimental/tools/integration/run_specv2_integration.py sparse_dataspec --baseline --out out_baseline_7ns --cp 7.00
# python3 experimental/tools/integration/run_adapted_spec_integration.py sparse_dataspec --out out_v1_7ns --cp 7.00 --default-value 0 --disable-constant-predictor
# python3 experimental/tools/integration/run_specv2_integration.py sparse_dataspec --out out_auto_7ns --cp 7.00 --resolver --decide-n 0
# python3 experimental/tools/integration/run_specv2_integration.py sparse_dataspec --n 0 --out out_0_7ns --cp 7.00 --resolver
# python3 experimental/tools/integration/run_specv2_integration.py sparse_dataspec --n 1 --out out_1_7ns --cp 7.00 --resolver --use-prof-cache
# python3 experimental/tools/integration/run_specv2_integration.py sparse_dataspec --n 2 --out out_2_7ns --cp 7.00 --resolver --use-prof-cache
# python3 experimental/tools/integration/run_specv2_integration.py sparse_dataspec --n 3 --out out_3_7ns --cp 7.00 --resolver --use-prof-cache
# python3 experimental/tools/integration/run_specv2_integration.py sparse_dataspec --n 4 --out out_4_7ns --cp 7.00 --resolver --use-prof-cache
# python3 experimental/tools/integration/run_specv2_integration.py sparse_dataspec --n 5 --out out_5_7ns --cp 7.00 --resolver --use-prof-cache
# python3 experimental/tools/integration/run_specv2_integration.py sparse_dataspec --n 6 --out out_6_7ns --cp 7.00 --resolver --use-prof-cache
# python3 experimental/tools/integration/run_specv2_integration.py sparse_dataspec --n 7 --out out_7_7ns --cp 7.00 --resolver --use-prof-cache
# python3 experimental/tools/integration/run_specv2_integration.py sparse_dataspec --n 8 --out out_8_7ns --cp 7.00 --resolver --use-prof-cache
# python3 experimental/tools/integration/run_specv2_integration.py sparse_dataspec --n 9 --out out_9_7ns --cp 7.00 --resolver --use-prof-cache
# python3 experimental/tools/integration/run_specv2_integration.py sparse_dataspec --n 10 --out out_10_7ns --cp 7.00 --resolver --use-prof-cache
# python3 experimental/tools/integration/run_specv2_integration.py sparse_dataspec --n 11 --out out_11_7ns --cp 7.00 --resolver --use-prof-cache
# python3 experimental/tools/integration/run_specv2_integration.py sparse_dataspec --n 12 --out out_12_7ns --cp 7.00 --resolver --use-prof-cache
# python3 experimental/tools/integration/run_specv2_integration.py sparse_dataspec --n 13 --out out_13_7ns --cp 7.00 --resolver --use-prof-cache
# python3 experimental/tools/integration/run_specv2_integration.py sparse_dataspec --n 14 --out out_14_7ns --cp 7.00 --resolver --use-prof-cache
# python3 experimental/tools/integration/run_specv2_integration.py sparse_dataspec --n 15 --out out_15_7ns --cp 7.00 --resolver --use-prof-cache
# python3 experimental/tools/integration/run_specv2_integration.py sparse_dataspec --n 16 --out out_16_7ns --cp 7.00 --resolver --use-prof-cache
# python3 experimental/tools/integration/run_specv2_integration.py sparse_dataspec --n 17 --out out_17_7ns --cp 7.00 --resolver --use-prof-cache
# python3 experimental/tools/integration/run_specv2_integration.py sparse_dataspec --n 18 --out out_18_7ns --cp 7.00 --resolver --use-prof-cache
# python3 experimental/tools/integration/run_specv2_integration.py sparse_dataspec --n 19 --out out_19_7ns --cp 7.00 --resolver --use-prof-cache
# python3 experimental/tools/integration/run_specv2_integration.py sparse_dataspec --n 20 --out out_20_7ns --cp 7.00 --resolver --use-prof-cache

# python3 experimental/tools/integration/run_specv2_integration.py sparse_dataspec_transformed --baseline --out out_baseline_7ns --cp 7.00 --transformed-code sparse_dataspec_modified.c
# python3 experimental/tools/integration/run_adapted_spec_integration.py sparse_dataspec_transformed --out out_v1_7ns --cp 7.00 --transformed-code sparse_dataspec_modified.c
# python3 experimental/tools/integration/run_specv2_integration.py sparse_dataspec_transformed --out out_auto_7ns --cp 7.00 --resolver --decide-n 0 --transformed-code sparse_dataspec_modified.c
# python3 experimental/tools/integration/run_specv2_integration.py sparse_dataspec_transformed --n 0 --out out_0_7ns --cp 7.00 --resolver --transformed-code sparse_dataspec_modified.c
# python3 experimental/tools/integration/run_specv2_integration.py sparse_dataspec_transformed --n 1 --out out_1_7ns --cp 7.00 --resolver --transformed-code sparse_dataspec_modified.c
# python3 experimental/tools/integration/run_specv2_integration.py sparse_dataspec_transformed --n 2 --out out_2_7ns --cp 7.00 --resolver --transformed-code sparse_dataspec_modified.c --use-prof-cache
# python3 experimental/tools/integration/run_specv2_integration.py sparse_dataspec_transformed --n 3 --out out_3_7ns --cp 7.00 --resolver --transformed-code sparse_dataspec_modified.c --use-prof-cache
# python3 experimental/tools/integration/run_specv2_integration.py sparse_dataspec_transformed --n 4 --out out_4_7ns --cp 7.00 --resolver --transformed-code sparse_dataspec_modified.c --use-prof-cache
# python3 experimental/tools/integration/run_specv2_integration.py sparse_dataspec_transformed --n 5 --out out_5_7ns --cp 7.00 --resolver --transformed-code sparse_dataspec_modified.c --use-prof-cache
# python3 experimental/tools/integration/run_specv2_integration.py sparse_dataspec_transformed --n 6 --out out_6_7ns --cp 7.00 --resolver --transformed-code sparse_dataspec_modified.c --use-prof-cache
# python3 experimental/tools/integration/run_specv2_integration.py sparse_dataspec_transformed --n 7 --out out_7_7ns --cp 7.00 --resolver --transformed-code sparse_dataspec_modified.c --use-prof-cache
# python3 experimental/tools/integration/run_specv2_integration.py sparse_dataspec_transformed --n 8 --out out_8_7ns --cp 7.00 --resolver --transformed-code sparse_dataspec_modified.c --use-prof-cache
# python3 experimental/tools/integration/run_specv2_integration.py sparse_dataspec_transformed --n 9 --out out_9_7ns --cp 7.00 --resolver --transformed-code sparse_dataspec_modified.c --use-prof-cache
# python3 experimental/tools/integration/run_specv2_integration.py sparse_dataspec_transformed --n 10 --out out_10_7ns --cp 7.00 --resolver --transformed-code sparse_dataspec_modified.c --use-prof-cache
# python3 experimental/tools/integration/run_specv2_integration.py sparse_dataspec_transformed --n 11 --out out_11_7ns --cp 7.00 --resolver --transformed-code sparse_dataspec_modified.c --use-prof-cache
# python3 experimental/tools/integration/run_specv2_integration.py sparse_dataspec_transformed --n 12 --out out_12_7ns --cp 7.00 --resolver --transformed-code sparse_dataspec_modified.c --use-prof-cache
# python3 experimental/tools/integration/run_specv2_integration.py sparse_dataspec_transformed --n 13 --out out_13_7ns --cp 7.00 --resolver --transformed-code sparse_dataspec_modified.c --use-prof-cache
# python3 experimental/tools/integration/run_specv2_integration.py sparse_dataspec_transformed --n 14 --out out_14_7ns --cp 7.00 --resolver --transformed-code sparse_dataspec_modified.c --use-prof-cache
# python3 experimental/tools/integration/run_specv2_integration.py sparse_dataspec_transformed --n 15 --out out_15_7ns --cp 7.00 --resolver --transformed-code sparse_dataspec_modified.c --use-prof-cache
# python3 experimental/tools/integration/run_specv2_integration.py sparse_dataspec_transformed --n 16 --out out_16_7ns --cp 7.00 --resolver --transformed-code sparse_dataspec_modified.c --use-prof-cache
# python3 experimental/tools/integration/run_specv2_integration.py sparse_dataspec_transformed --n 17 --out out_17_7ns --cp 7.00 --resolver --transformed-code sparse_dataspec_modified.c --use-prof-cache
# python3 experimental/tools/integration/run_specv2_integration.py sparse_dataspec_transformed --n 18 --out out_18_7ns --cp 7.00 --resolver --transformed-code sparse_dataspec_modified.c --use-prof-cache
# python3 experimental/tools/integration/run_specv2_integration.py sparse_dataspec_transformed --n 19 --out out_19_7ns --cp 7.00 --resolver --transformed-code sparse_dataspec_modified.c --use-prof-cache
# python3 experimental/tools/integration/run_specv2_integration.py sparse_dataspec_transformed --n 20 --out out_20_7ns --cp 7.00 --resolver --transformed-code sparse_dataspec_modified.c --use-prof-cache

# python3 experimental/tools/integration/run_gamma_integration.py if_float --branch-bb=1 --merge-bb=4 --steps-until 0 --prioritized-side 0 --out out_baseline_7ns --cp 7.00
# python3 experimental/tools/integration/run_adapted_spec_integration.py if_float --default-value=0 --out out_v1_7ns --cp 7.00 --loop-bottom-passer-disabled
# python3 experimental/tools/integration/run_gamma_integration.py if_float --branch-bb=1 --merge-bb=4 --prioritized-side 0 --one-sided --out out_0_one_sided_7ns --cp 7.00
# python3 experimental/tools/integration/run_gamma_integration.py if_float --branch-bb=1 --merge-bb=4 --prioritized-side 0 --out out_0_two_sided_7ns --use-prof-cache --cp 7.00
# python3 experimental/tools/integration/run_gamma_integration.py if_float --branch-bb=1 --merge-bb=4 --prioritized-side 1 --out out_1_two_sided_7ns --use-prof-cache --cp 7.00
# python3 experimental/tools/integration/run_gamma_integration.py if_float --branch-bb=1 --merge-bb=4 --prioritized-side 1 --one-sided --out out_1_one_sided_7ns --use-prof-cache --cp 7.00

# python3 experimental/tools/integration/run_gamma_integration.py if_float2 --branch-bb=1 --merge-bb=4 --steps-until 0 --prioritized-side 0 --out out_baseline_7ns --cp 7.00
# python3 experimental/tools/integration/run_adapted_spec_integration.py if_float2 --default-value=0 --out out_v1_7ns --cp 7.00 --loop-bottom-passer-disabled
# python3 experimental/tools/integration/run_gamma_integration.py if_float2 --branch-bb=1 --merge-bb=4 --prioritized-side 0 --one-sided --out out_0_one_sided_7ns --cp 7.00
# python3 experimental/tools/integration/run_gamma_integration.py if_float2 --branch-bb=1 --merge-bb=4 --prioritized-side 0 --out out_0_two_sided_7ns --use-prof-cache --cp 7.00
# python3 experimental/tools/integration/run_gamma_integration.py if_float2 --branch-bb=1 --merge-bb=4 --prioritized-side 1 --one-sided --out out_1_one_sided_7ns --use-prof-cache --cp 7.00
# python3 experimental/tools/integration/run_gamma_integration.py if_float2 --branch-bb=1 --merge-bb=4 --prioritized-side 1 --out out_1_two_sided_7ns --use-prof-cache --cp 7.00

# # On-merges (Fig. 6)
# python3 experimental/tools/integration/run_specv2_integration.py single_loop --baseline --out out_baseline_7ns_om --cp 7.00 --resolver --min-buffering
# python3 experimental/tools/integration/run_specv2_integration.py single_loop --n 1 --out out_1_7ns_om --cp 7.00 --resolver --use-prof-cache --min-buffering
# python3 experimental/tools/integration/run_specv2_integration.py single_loop --n 2 --out out_2_7ns_om --cp 7.00 --resolver --use-prof-cache --min-buffering
# python3 experimental/tools/integration/run_specv2_integration.py single_loop --n 3 --out out_3_7ns_om --cp 7.00 --resolver --use-prof-cache --min-buffering
# python3 experimental/tools/integration/run_specv2_integration.py single_loop --n 4 --out out_4_7ns_om --cp 7.00 --resolver --use-prof-cache --min-buffering
# python3 experimental/tools/integration/run_specv2_integration.py single_loop --n 5 --out out_5_7ns_om --cp 7.00 --resolver --use-prof-cache --min-buffering

# python3 experimental/tools/integration/run_specv2_integration.py collision_donut --baseline --out out_baseline_7ns_om --cp 7.00 --min-buffering
# python3 experimental/tools/integration/run_specv2_integration.py collision_donut --n 0 --out out_0_7ns_om --cp 7.00 --resolver --min-buffering
# python3 experimental/tools/integration/run_specv2_integration.py collision_donut --n 1 --out out_1_7ns_om --cp 7.00 --use-prof-cache --resolver --min-buffering
# python3 experimental/tools/integration/run_specv2_integration.py collision_donut --n 2 --out out_2_7ns_om --cp 7.00 --use-prof-cache --resolver --min-buffering
# python3 experimental/tools/integration/run_specv2_integration.py collision_donut --n 3 --out out_3_7ns_om --cp 7.00 --use-prof-cache --resolver --min-buffering
# python3 experimental/tools/integration/run_specv2_integration.py collision_donut --n 4 --out out_4_7ns_om --cp 7.00 --use-prof-cache --resolver --min-buffering
# python3 experimental/tools/integration/run_specv2_integration.py collision_donut --n 5 --out out_5_7ns_om --cp 7.00 --use-prof-cache --resolver --min-buffering

# python3 experimental/tools/integration/run_specv2_integration.py subdiag --baseline --out out_baseline_7ns_om --transformed-code subdiag_v1.c --cp 7.00 --min-buffering
# python3 experimental/tools/integration/run_specv2_integration.py subdiag --n 1 --out out_1_7ns_om --cp 7.00 --resolver --transformed-code subdiag_v1.c --min-buffering
# python3 experimental/tools/integration/run_specv2_integration.py subdiag --n 2 --out out_2_7ns_om --cp 7.00 --resolver --transformed-code subdiag_v1.c --use-prof-cache --min-buffering
# python3 experimental/tools/integration/run_specv2_integration.py subdiag --n 3 --out out_3_7ns_om --cp 7.00 --resolver --transformed-code subdiag_v1.c --use-prof-cache --min-buffering
# python3 experimental/tools/integration/run_specv2_integration.py subdiag --n 4 --out out_4_7ns_om --cp 7.00 --resolver --transformed-code subdiag_v1.c --use-prof-cache --min-buffering
# python3 experimental/tools/integration/run_specv2_integration.py subdiag --n 5 --out out_5_7ns_om --cp 7.00 --resolver --transformed-code subdiag_v1.c --use-prof-cache --min-buffering
# python3 experimental/tools/integration/run_specv2_integration.py subdiag --n 6 --out out_6_7ns_om --cp 7.00 --resolver --transformed-code subdiag_v1.c --use-prof-cache --min-buffering
# python3 experimental/tools/integration/run_specv2_integration.py subdiag --n 7 --out out_7_7ns_om --cp 7.00 --resolver --transformed-code subdiag_v1.c --use-prof-cache --min-buffering
# python3 experimental/tools/integration/run_specv2_integration.py subdiag --n 8 --out out_8_7ns_om --cp 7.00 --resolver --transformed-code subdiag_v1.c --use-prof-cache --min-buffering
# python3 experimental/tools/integration/run_specv2_integration.py subdiag --n 9 --out out_9_7ns_om --cp 7.00 --resolver --transformed-code subdiag_v1.c --use-prof-cache --min-buffering
# python3 experimental/tools/integration/run_specv2_integration.py subdiag --n 10 --out out_10_7ns_om --cp 7.00 --resolver --transformed-code subdiag_v1.c --use-prof-cache --min-buffering
# python3 experimental/tools/integration/run_specv2_integration.py subdiag --n 11 --out out_11_7ns_om --cp 7.00 --resolver --transformed-code subdiag_v1.c --use-prof-cache --min-buffering
# python3 experimental/tools/integration/run_specv2_integration.py subdiag --n 12 --out out_12_7ns_om --cp 7.00 --resolver --transformed-code subdiag_v1.c --use-prof-cache --min-buffering
# python3 experimental/tools/integration/run_specv2_integration.py subdiag --n 13 --out out_13_7ns_om --cp 7.00 --resolver --transformed-code subdiag_v1.c --use-prof-cache --min-buffering
# python3 experimental/tools/integration/run_specv2_integration.py subdiag --n 14 --out out_14_7ns_om --cp 7.00 --resolver --transformed-code subdiag_v1.c --use-prof-cache --min-buffering

# python3 experimental/tools/integration/run_specv2_integration.py sparse_dataspec_transformed --baseline --out out_baseline_7ns_om --cp 7.00 --transformed-code sparse_dataspec_modified.c --min-buffering
# python3 experimental/tools/integration/run_specv2_integration.py sparse_dataspec_transformed --n 1 --out out_1_7ns_om --cp 7.00 --resolver --transformed-code sparse_dataspec_modified.c --min-buffering
# python3 experimental/tools/integration/run_specv2_integration.py sparse_dataspec_transformed --n 2 --out out_2_7ns_om --cp 7.00 --resolver --transformed-code sparse_dataspec_modified.c --min-buffering --use-prof-cache
# python3 experimental/tools/integration/run_specv2_integration.py sparse_dataspec_transformed --n 3 --out out_3_7ns_om --cp 7.00 --resolver --transformed-code sparse_dataspec_modified.c --min-buffering --use-prof-cache
# python3 experimental/tools/integration/run_specv2_integration.py sparse_dataspec_transformed --n 4 --out out_4_7ns_om --cp 7.00 --resolver --transformed-code sparse_dataspec_modified.c --min-buffering --use-prof-cache
# python3 experimental/tools/integration/run_specv2_integration.py sparse_dataspec_transformed --n 5 --out out_5_7ns_om --cp 7.00 --resolver --transformed-code sparse_dataspec_modified.c --min-buffering --use-prof-cache

# # Synthesis (Table 2, Fig. 7)

# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/single_loop/out_v1_7ns single_loop 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/single_loop/out_baseline_7ns single_loop 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/single_loop/out_auto_7ns single_loop 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/single_loop/out_0_7ns single_loop 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/single_loop/out_1_7ns single_loop 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/single_loop/out_2_7ns single_loop 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/single_loop/out_3_7ns single_loop 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/single_loop/out_4_7ns single_loop 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/single_loop/out_5_7ns single_loop 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/single_loop/out_6_7ns single_loop 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/single_loop/out_7_7ns single_loop 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/single_loop/out_8_7ns single_loop 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/single_loop/out_9_7ns single_loop 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/single_loop/out_10_7ns single_loop 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/single_loop/out_11_7ns single_loop 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/single_loop/out_12_7ns single_loop 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/single_loop/out_13_7ns single_loop 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/single_loop/out_14_7ns single_loop 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/single_loop/out_15_7ns single_loop 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/single_loop/out_16_7ns single_loop 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/single_loop/out_17_7ns single_loop 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/single_loop/out_18_7ns single_loop 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/single_loop/out_19_7ns single_loop 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/single_loop/out_20_7ns single_loop 7.000 3.500


# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/nested_loop/out_v1_7ns nested_loop 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/nested_loop/out_baseline_7ns nested_loop 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/nested_loop/out_auto_7ns nested_loop 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/nested_loop/out_0_7ns nested_loop 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/nested_loop/out_1_7ns nested_loop 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/nested_loop/out_2_7ns nested_loop 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/nested_loop/out_3_7ns nested_loop 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/nested_loop/out_4_7ns nested_loop 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/nested_loop/out_5_7ns nested_loop 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/nested_loop/out_6_7ns nested_loop 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/nested_loop/out_7_7ns nested_loop 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/nested_loop/out_8_7ns nested_loop 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/nested_loop/out_9_7ns nested_loop 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/nested_loop/out_10_7ns nested_loop 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/nested_loop/out_11_7ns nested_loop 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/nested_loop/out_12_7ns nested_loop 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/nested_loop/out_13_7ns nested_loop 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/nested_loop/out_14_7ns nested_loop 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/nested_loop/out_15_7ns nested_loop 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/nested_loop/out_16_7ns nested_loop 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/nested_loop/out_17_7ns nested_loop 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/nested_loop/out_18_7ns nested_loop 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/nested_loop/out_19_7ns nested_loop 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/nested_loop/out_20_7ns nested_loop 7.000 3.500

# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/fixed_log/out_v1_7ns fixed_log 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/fixed_log/out_baseline_7ns fixed_log 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/fixed_log/out_auto_7ns fixed_log 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/fixed_log/out_0_7ns fixed_log 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/fixed_log/out_1_7ns fixed_log 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/fixed_log/out_2_7ns fixed_log 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/fixed_log/out_3_7ns fixed_log 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/fixed_log/out_4_7ns fixed_log 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/fixed_log/out_5_7ns fixed_log 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/fixed_log/out_6_7ns fixed_log 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/fixed_log/out_7_7ns fixed_log 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/fixed_log/out_8_7ns fixed_log 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/fixed_log/out_9_7ns fixed_log 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/fixed_log/out_10_7ns fixed_log 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/fixed_log/out_11_7ns fixed_log 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/fixed_log/out_12_7ns fixed_log 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/fixed_log/out_13_7ns fixed_log 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/fixed_log/out_14_7ns fixed_log 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/fixed_log/out_15_7ns fixed_log 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/fixed_log/out_16_7ns fixed_log 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/fixed_log/out_17_7ns fixed_log 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/fixed_log/out_18_7ns fixed_log 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/fixed_log/out_19_7ns fixed_log 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/fixed_log/out_20_7ns fixed_log 7.000 3.500

# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/newton/out_v1_7ns newton 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/newton/out_baseline_7ns newton 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/newton/out_auto_7ns newton 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/newton/out_0_7ns newton 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/newton/out_1_7ns newton 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/newton/out_2_7ns newton 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/newton/out_3_7ns newton 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/newton/out_4_7ns newton 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/newton/out_5_7ns newton 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/newton/out_6_7ns newton 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/newton/out_7_7ns newton 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/newton/out_8_7ns newton 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/newton/out_9_7ns newton 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/newton/out_10_7ns newton 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/newton/out_11_7ns newton 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/newton/out_12_7ns newton 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/newton/out_13_7ns newton 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/newton/out_14_7ns newton 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/newton/out_15_7ns newton 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/newton/out_16_7ns newton 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/newton/out_17_7ns newton 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/newton/out_18_7ns newton 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/newton/out_19_7ns newton 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/newton/out_20_7ns newton 7.000 3.500

# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/subdiag_fast/out_v1_7ns subdiag_fast 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/subdiag_fast/out_baseline_7ns subdiag_fast 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/subdiag_fast/out_auto_7ns subdiag_fast 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/subdiag_fast/out_0_7ns subdiag_fast 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/subdiag_fast/out_1_7ns subdiag_fast 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/subdiag_fast/out_2_7ns subdiag_fast 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/subdiag_fast/out_3_7ns subdiag_fast 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/subdiag_fast/out_4_7ns subdiag_fast 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/subdiag_fast/out_5_7ns subdiag_fast 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/subdiag_fast/out_6_7ns subdiag_fast 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/subdiag_fast/out_7_7ns subdiag_fast 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/subdiag_fast/out_8_7ns subdiag_fast 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/subdiag_fast/out_9_7ns subdiag_fast 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/subdiag_fast/out_10_7ns subdiag_fast 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/subdiag_fast/out_11_7ns subdiag_fast 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/subdiag_fast/out_12_7ns subdiag_fast 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/subdiag_fast/out_13_7ns subdiag_fast 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/subdiag_fast/out_14_7ns subdiag_fast 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/subdiag_fast/out_15_7ns subdiag_fast 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/subdiag_fast/out_16_7ns subdiag_fast 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/subdiag_fast/out_17_7ns subdiag_fast 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/subdiag_fast/out_18_7ns subdiag_fast 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/subdiag_fast/out_19_7ns subdiag_fast 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/subdiag_fast/out_20_7ns subdiag_fast 7.000 3.500

# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/golden_ratio/out_v1_7ns golden_ratio 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/golden_ratio/out_baseline_7ns golden_ratio 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/golden_ratio/out_auto_7ns golden_ratio 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/golden_ratio/out_0_7ns golden_ratio 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/golden_ratio/out_1_7ns golden_ratio 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/golden_ratio/out_2_7ns golden_ratio 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/golden_ratio/out_3_7ns golden_ratio 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/golden_ratio/out_4_7ns golden_ratio 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/golden_ratio/out_5_7ns golden_ratio 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/golden_ratio/out_6_7ns golden_ratio 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/golden_ratio/out_7_7ns golden_ratio 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/golden_ratio/out_8_7ns golden_ratio 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/golden_ratio/out_9_7ns golden_ratio 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/golden_ratio/out_10_7ns golden_ratio 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/golden_ratio/out_11_7ns golden_ratio 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/golden_ratio/out_12_7ns golden_ratio 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/golden_ratio/out_13_7ns golden_ratio 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/golden_ratio/out_14_7ns golden_ratio 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/golden_ratio/out_15_7ns golden_ratio 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/golden_ratio/out_16_7ns golden_ratio 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/golden_ratio/out_17_7ns golden_ratio 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/golden_ratio/out_18_7ns golden_ratio 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/golden_ratio/out_19_7ns golden_ratio 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/golden_ratio/out_20_7ns golden_ratio 7.000 3.500

# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/collision_donut/out_v1_7ns collision_donut 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/collision_donut/out_baseline_7ns collision_donut 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/collision_donut/out_auto_7ns collision_donut 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/collision_donut/out_0_7ns collision_donut 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/collision_donut/out_1_7ns collision_donut 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/collision_donut/out_2_7ns collision_donut 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/collision_donut/out_3_7ns collision_donut 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/collision_donut/out_4_7ns collision_donut 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/collision_donut/out_5_7ns collision_donut 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/collision_donut/out_6_7ns collision_donut 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/collision_donut/out_7_7ns collision_donut 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/collision_donut/out_8_7ns collision_donut 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/collision_donut/out_9_7ns collision_donut 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/collision_donut/out_10_7ns collision_donut 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/collision_donut/out_11_7ns collision_donut 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/collision_donut/out_12_7ns collision_donut 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/collision_donut/out_13_7ns collision_donut 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/collision_donut/out_14_7ns collision_donut 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/collision_donut/out_15_7ns collision_donut 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/collision_donut/out_16_7ns collision_donut 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/collision_donut/out_17_7ns collision_donut 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/collision_donut/out_18_7ns collision_donut 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/collision_donut/out_19_7ns collision_donut 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/collision_donut/out_20_7ns collision_donut 7.000 3.500

# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/bisection/out_v1_7ns bisection 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/bisection/out_baseline_7ns bisection 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/bisection/out_auto_7ns bisection 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/bisection/out_0_7ns bisection 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/bisection/out_1_7ns bisection 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/bisection/out_2_7ns bisection 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/bisection/out_3_7ns bisection 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/bisection/out_4_7ns bisection 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/bisection/out_5_7ns bisection 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/bisection/out_6_7ns bisection 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/bisection/out_7_7ns bisection 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/bisection/out_8_7ns bisection 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/bisection/out_9_7ns bisection 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/bisection/out_10_7ns bisection 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/bisection/out_11_7ns bisection 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/bisection/out_12_7ns bisection 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/bisection/out_13_7ns bisection 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/bisection/out_14_7ns bisection 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/bisection/out_15_7ns bisection 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/bisection/out_16_7ns bisection 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/bisection/out_17_7ns bisection 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/bisection/out_18_7ns bisection 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/bisection/out_19_7ns bisection 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/bisection/out_20_7ns bisection 7.000 3.500

# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/sparse_dataspec/out_v1_7ns sparse_dataspec 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/sparse_dataspec/out_baseline_7ns sparse_dataspec 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/sparse_dataspec/out_auto_7ns sparse_dataspec 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/sparse_dataspec/out_0_7ns sparse_dataspec 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/sparse_dataspec/out_1_7ns sparse_dataspec 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/sparse_dataspec/out_2_7ns sparse_dataspec 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/sparse_dataspec/out_3_7ns sparse_dataspec 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/sparse_dataspec/out_4_7ns sparse_dataspec 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/sparse_dataspec/out_5_7ns sparse_dataspec 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/sparse_dataspec/out_6_7ns sparse_dataspec 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/sparse_dataspec/out_7_7ns sparse_dataspec 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/sparse_dataspec/out_8_7ns sparse_dataspec 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/sparse_dataspec/out_9_7ns sparse_dataspec 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/sparse_dataspec/out_10_7ns sparse_dataspec 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/sparse_dataspec/out_11_7ns sparse_dataspec 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/sparse_dataspec/out_12_7ns sparse_dataspec 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/sparse_dataspec/out_13_7ns sparse_dataspec 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/sparse_dataspec/out_14_7ns sparse_dataspec 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/sparse_dataspec/out_15_7ns sparse_dataspec 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/sparse_dataspec/out_16_7ns sparse_dataspec 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/sparse_dataspec/out_17_7ns sparse_dataspec 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/sparse_dataspec/out_18_7ns sparse_dataspec 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/sparse_dataspec/out_19_7ns sparse_dataspec 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/sparse_dataspec/out_20_7ns sparse_dataspec 7.000 3.500

# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/sparse_dataspec_transformed/out_v1_7ns sparse_dataspec_transformed 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/sparse_dataspec_transformed/out_baseline_7ns sparse_dataspec_transformed 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/sparse_dataspec_transformed/out_auto_7ns sparse_dataspec_transformed 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/sparse_dataspec_transformed/out_0_7ns sparse_dataspec_transformed 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/sparse_dataspec_transformed/out_1_7ns sparse_dataspec_transformed 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/sparse_dataspec_transformed/out_2_7ns sparse_dataspec_transformed 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/sparse_dataspec_transformed/out_3_7ns sparse_dataspec_transformed 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/sparse_dataspec_transformed/out_4_7ns sparse_dataspec_transformed 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/sparse_dataspec_transformed/out_5_7ns sparse_dataspec_transformed 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/sparse_dataspec_transformed/out_6_7ns sparse_dataspec_transformed 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/sparse_dataspec_transformed/out_7_7ns sparse_dataspec_transformed 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/sparse_dataspec_transformed/out_8_7ns sparse_dataspec_transformed 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/sparse_dataspec_transformed/out_9_7ns sparse_dataspec_transformed 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/sparse_dataspec_transformed/out_10_7ns sparse_dataspec_transformed 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/sparse_dataspec_transformed/out_11_7ns sparse_dataspec_transformed 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/sparse_dataspec_transformed/out_12_7ns sparse_dataspec_transformed 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/sparse_dataspec_transformed/out_13_7ns sparse_dataspec_transformed 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/sparse_dataspec_transformed/out_14_7ns sparse_dataspec_transformed 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/sparse_dataspec_transformed/out_15_7ns sparse_dataspec_transformed 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/sparse_dataspec_transformed/out_16_7ns sparse_dataspec_transformed 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/sparse_dataspec_transformed/out_17_7ns sparse_dataspec_transformed 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/sparse_dataspec_transformed/out_18_7ns sparse_dataspec_transformed 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/sparse_dataspec_transformed/out_19_7ns sparse_dataspec_transformed 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/sparse_dataspec_transformed/out_20_7ns sparse_dataspec_transformed 7.000 3.500

# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/if_float/out_v1_7ns if_float 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/if_float/out_baseline_7ns if_float 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/if_float/out_0_one_sided_7ns if_float 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/if_float/out_1_one_sided_7ns if_float 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/if_float/out_0_two_sided_7ns if_float 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/if_float/out_1_two_sided_7ns if_float 7.000 3.500

# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/if_float2/out_baseline_7ns if_float2 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/if_float2/out_v1_7ns if_float2 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/if_float2/out_0_one_sided_7ns if_float2 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/if_float2/out_1_one_sided_7ns if_float2 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/if_float2/out_0_two_sided_7ns if_float2 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/if_float2/out_1_two_sided_7ns if_float2 7.000 3.500

# # On-merge synthesis (Fig. 6)
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/single_loop/out_baseline_7ns_om single_loop 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/single_loop/out_1_7ns_om single_loop 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/single_loop/out_2_7ns_om single_loop 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/single_loop/out_3_7ns_om single_loop 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/single_loop/out_4_7ns_om single_loop 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/single_loop/out_5_7ns_om single_loop 7.000 3.500


# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/collision_donut/out_baseline_7ns_om collision_donut 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/collision_donut/out_0_7ns_om collision_donut 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/collision_donut/out_1_7ns_om collision_donut 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/collision_donut/out_2_7ns_om collision_donut 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/collision_donut/out_3_7ns_om collision_donut 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/collision_donut/out_4_7ns_om collision_donut 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/collision_donut/out_5_7ns_om collision_donut 7.000 3.500

# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/subdiag/out_baseline_7ns_om subdiag 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/subdiag/out_1_7ns_om subdiag 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/subdiag/out_2_7ns_om subdiag 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/subdiag/out_3_7ns_om subdiag 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/subdiag/out_4_7ns_om subdiag 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/subdiag/out_5_7ns_om subdiag 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/subdiag/out_6_7ns_om subdiag 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/subdiag/out_7_7ns_om subdiag 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/subdiag/out_8_7ns_om subdiag 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/subdiag/out_9_7ns_om subdiag 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/subdiag/out_10_7ns_om subdiag 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/subdiag/out_11_7ns_om subdiag 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/subdiag/out_12_7ns_om subdiag 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/subdiag/out_13_7ns_om subdiag 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/subdiag/out_14_7ns_om subdiag 7.000 3.500

# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/sparse_dataspec_transformed/out_baseline_7ns_om sparse_dataspec_transformed 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/sparse_dataspec_transformed/out_1_7ns_om sparse_dataspec_transformed 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/sparse_dataspec_transformed/out_2_7ns_om sparse_dataspec_transformed 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/sparse_dataspec_transformed/out_3_7ns_om sparse_dataspec_transformed 7.000 3.500
# timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/sparse_dataspec_transformed/out_4_7ns_om sparse_dataspec_transformed 7.000 3.500

# Large Benchmarks

# prof-cache is not used
python3 experimental/tools/integration/run_specv2_large_integration.py single_loop_unrolled_160 --min-buffering --baseline --use-prof-cache --out out_baseline_7ns --factor 160 --pre_unrolling $DYNAMATIC_DIR/integration-test/single_loop_unrolled_160/copy_src_baseline.mlir
python3 experimental/tools/integration/run_specv2_large_integration.py single_loop_unrolled_160 --min-buffering --decide-n 0 --resolver --use-prof-cache --out out_auto_7ns --factor 160 --pre_unrolling $DYNAMATIC_DIR/integration-test/single_loop_unrolled_160/copy_src_eager.mlir
# rm integration-test/single_loop_unrolled_160/specv2_*

# Longer timeout
# timeout --kill-after=10s 9000s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/single_loop_unrolled_160/out_baseline_7ns single_loop_unrolled_160 7.000 3.500
# timeout --kill-after=10s 9000s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/single_loop_unrolled_160/out_auto_7ns single_loop_unrolled_160 7.000 3.500

# prof-cache is not used
# Buffer copy (TODO)
# python3 experimental/tools/integration/run_specv2_large_integration.py bisection_unrolled_16 --min-buffering --decide-n 0 --resolver --use-prof-cache --out out_auto_7ns --factor 16 --transformed-code bisection_transformed_unrolled_16.c
# python3 experimental/tools/integration/run_specv2_large_integration.py bisection_unrolled_16 --min-buffering --baseline --use-prof-cache --out out_baseline_7ns --factor 16 --transformed-code bisection_transformed_unrolled_16.c

# On-merges
python3 experimental/tools/integration/run_specv2_large_integration.py bisection_unrolled_16 --on-merges --decide-n 0 --resolver --use-prof-cache --out out_auto_7ns --factor 16 --transformed-code bisection_transformed_unrolled_16.c
python3 experimental/tools/integration/run_specv2_large_integration.py bisection_unrolled_16 --on-merges --baseline --use-prof-cache --out out_baseline_7ns --factor 16 --transformed-code bisection_transformed_unrolled_16.c

# Longer timeout
timeout --kill-after=10s 9000s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/bisection_unrolled_16/out_baseline_7ns bisection_unrolled_16 7.000 3.500
timeout --kill-after=10s 9000s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/bisection_unrolled_16/out_auto_7ns bisection_unrolled_16 7.000 3.500

python3 experimental/tools/integration/run_specv2_large_integration.py kmp --on-merges --out out_baseline_7ns --factor 10 --disable-spec
python3 experimental/tools/integration/run_gamma.py kmp --on-merges --out out_eager_7ns --factor 10


date > end_time.txt
