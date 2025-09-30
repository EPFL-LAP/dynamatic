SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DYNAMATIC_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd $DYNAMATIC_DIR

# Baseline
python3 experimental/tools/integration/run_specv2_integration.py single_loop --baseline --out out_baseline --cp 5.00

# Spec v1
python3 experimental/tools/integration/run_adapted_spec_integration.py single_loop --out out_v1 --cp 5.00

python3 experimental/tools/integration/run_specv2_integration.py single_loop --n 0 --out out_0 --cp 5.00 --resolver
python3 experimental/tools/integration/run_specv2_integration.py single_loop --n 4 --out out_4 --cp 5.00 --resolver
