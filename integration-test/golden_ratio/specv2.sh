SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DYNAMATIC_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd $DYNAMATIC_DIR

python3 experimental/tools/integration/run_specv2_integration.py golden_ratio --n 0 --out out_0 --cp 10.00
python3 experimental/tools/integration/run_specv2_integration.py golden_ratio --n 1 --out out_1 --cp 10.00
python3 experimental/tools/integration/run_specv2_integration.py golden_ratio --n 1 --out out_1_exit --cp 10.00 --exit-eager-eval
# python3 experimental/tools/integration/run_specv2_integration.py golden_ratio --n 2 --out out_2 --cp 10.00

# # Spec v1
# python3 tools/integration/run_spec_integration.py subdiag_fast --out out_v1 --cp 15.00 --transformed-code subdiag_fast_v1.c

# python3 experimental/tools/integration/run_specv2_integration.py subdiag_fast --n 12 --out out_12 --cp 15.00
# python3 experimental/tools/integration/run_specv2_integration.py subdiag_fast --n 13 --out out_13 --cp 15.00
# python3 experimental/tools/integration/run_specv2_integration.py subdiag_fast --n 14 --out out_14 --cp 15.00
