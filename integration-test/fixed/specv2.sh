SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DYNAMATIC_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd $DYNAMATIC_DIR

# Spec v1
python3 tools/integration/run_spec_integration.py fixed --out out_v1

python3 experimental/tools/integration/run_specv2_integration.py fixed --n 0 --out out_0
python3 experimental/tools/integration/run_specv2_integration.py fixed --n 1 --out out_1
python3 experimental/tools/integration/run_specv2_integration.py fixed --n 2 --out out_2
python3 experimental/tools/integration/run_specv2_integration.py fixed --n 3 --out out_3
python3 experimental/tools/integration/run_specv2_integration.py fixed --n 4 --out out_4
python3 experimental/tools/integration/run_specv2_integration.py fixed --variable --out out_variable
