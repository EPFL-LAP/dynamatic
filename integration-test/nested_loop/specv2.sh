SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DYNAMATIC_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd $DYNAMATIC_DIR

# Spec v1
python3 tools/integration/run_spec_integration.py nested_loop --out out_v1

python3 experimental/tools/integration/run_specv2_integration.py nested_loop --n 0 --out out_0
python3 experimental/tools/integration/run_specv2_integration.py nested_loop --n 1 --out out_1
python3 experimental/tools/integration/run_specv2_integration.py nested_loop --n 2 --out out_2
python3 experimental/tools/integration/run_specv2_integration.py nested_loop --n 3 --out out_3
python3 experimental/tools/integration/run_specv2_integration.py nested_loop --n 4 --out out_4
python3 experimental/tools/integration/run_specv2_integration.py nested_loop --n 5 --out out_5
python3 experimental/tools/integration/run_specv2_integration.py nested_loop --n 6 --out out_6
python3 experimental/tools/integration/run_specv2_integration.py nested_loop --variable --out out_variable
