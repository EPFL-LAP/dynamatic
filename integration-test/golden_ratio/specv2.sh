SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DYNAMATIC_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd $DYNAMATIC_DIR

python3 tools/integration/run_spec_integration.py golden_ratio --out out_v1 --cp 7.00
python3 experimental/tools/integration/run_specv2_integration.py golden_ratio --n 0 --out out_0 --cp 7.00 --resolver
python3 experimental/tools/integration/run_specv2_integration.py golden_ratio --n 1 --out out_1 --cp 7.00 --resolver
python3 experimental/tools/integration/run_specv2_integration.py golden_ratio --n 1 --out out_1_exit --cp 7.00 --exit-eager-eval --resolver
python3 experimental/tools/integration/run_specv2_integration.py golden_ratio --n 2 --out out_2_exit --cp 7.00 --exit-eager-eval --resolver --use-prof-cache
python3 experimental/tools/integration/run_specv2_integration.py golden_ratio --n 3 --out out_3_exit --cp 7.00 --exit-eager-eval --resolver --use-prof-cache
python3 experimental/tools/integration/run_specv2_integration.py golden_ratio --n 4 --out out_4_exit --cp 7.00 --exit-eager-eval --resolver --use-prof-cache
python3 experimental/tools/integration/run_specv2_integration.py golden_ratio --n 5 --out out_5_exit --cp 7.00 --exit-eager-eval --resolver --use-prof-cache
python3 experimental/tools/integration/run_specv2_integration.py golden_ratio --n 6 --out out_6_exit --cp 7.00 --exit-eager-eval --resolver --use-prof-cache
python3 experimental/tools/integration/run_specv2_integration.py golden_ratio --n 7 --out out_7_exit --cp 7.00 --exit-eager-eval --resolver --use-prof-cache
python3 experimental/tools/integration/run_specv2_integration.py golden_ratio --n 8 --out out_8_exit --cp 7.00 --exit-eager-eval --resolver --use-prof-cache
python3 experimental/tools/integration/run_specv2_integration.py golden_ratio --n 9 --out out_9_exit --cp 7.00 --exit-eager-eval --resolver --use-prof-cache
python3 experimental/tools/integration/run_specv2_integration.py golden_ratio --n 10 --out out_10_exit --cp 7.00 --exit-eager-eval --resolver --use-prof-cache
# python3 experimental/tools/integration/run_specv2_integration.py golden_ratio --n 11 --out out_11_exit --cp 7.00 --exit-eager-eval --resolver --use-prof-cache
# python3 experimental/tools/integration/run_specv2_integration.py golden_ratio --n 12 --out out_12_exit --cp 7.00 --exit-eager-eval --resolver --use-prof-cache
# python3 experimental/tools/integration/run_specv2_integration.py golden_ratio --n 13 --out out_13_exit --cp 7.00 --exit-eager-eval --resolver --use-prof-cache
# python3 experimental/tools/integration/run_specv2_integration.py golden_ratio --n 14 --out out_14_exit --cp 7.00 --exit-eager-eval --resolver --use-prof-cache
# python3 experimental/tools/integration/run_specv2_integration.py golden_ratio --n 15 --out out_15_exit --cp 7.00 --exit-eager-eval --resolver --use-prof-cache
# python3 experimental/tools/integration/run_specv2_integration.py golden_ratio --n 16 --out out_16_exit --cp 7.00 --exit-eager-eval --resolver --use-prof-cache
# python3 experimental/tools/integration/run_specv2_integration.py golden_ratio --n 17 --out out_17_exit --cp 7.00 --exit-eager-eval --resolver --use-prof-cache
# python3 experimental/tools/integration/run_specv2_integration.py golden_ratio --n 18 --out out_18_exit --cp 7.00 --exit-eager-eval --resolver --use-prof-cache
# python3 experimental/tools/integration/run_specv2_integration.py golden_ratio --n 19 --out out_19_exit --cp 7.00 --exit-eager-eval --resolver --use-prof-cache
# python3 experimental/tools/integration/run_specv2_integration.py golden_ratio --n 20 --out out_20_exit --cp 7.00 --exit-eager-eval --resolver --use-prof-cache
