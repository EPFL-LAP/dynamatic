SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DYNAMATIC_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd $DYNAMATIC_DIR

python3 experimental/tools/integration/run_specv2_integration.py nested_loop --n 7 --out out_7 --cp 5.00
python3 experimental/tools/integration/run_specv2_integration.py nested_loop --n 8 --out out_8 --cp 5.00
python3 experimental/tools/integration/run_specv2_integration.py nested_loop --n 9 --out out_9 --cp 5.00
python3 experimental/tools/integration/run_specv2_integration.py nested_loop --n 10 --out out_10 --cp 5.00
python3 experimental/tools/integration/run_specv2_integration.py nested_loop --n 11 --out out_11 --cp 5.00
