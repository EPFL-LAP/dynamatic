SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DYNAMATIC_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd $DYNAMATIC_DIR

python3 experimental/tools/integration/run_specv2_integration.py fixed_log --n 0 --out out_0 --cp 5.00
python3 experimental/tools/integration/run_specv2_integration.py fixed_log --n 3 --out out_3 --cp 5.00
