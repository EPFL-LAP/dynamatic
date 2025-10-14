SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DYNAMATIC_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd $DYNAMATIC_DIR

# Baseline
python3 experimental/tools/integration/run_specv2_integration.py single_loop --baseline --out out_baseline_5ns --cp 5.00

# Spec v1
python3 experimental/tools/integration/run_adapted_spec_integration.py single_loop --out out_v1_5ns --cp 5.00

python3 experimental/tools/integration/run_specv2_integration.py single_loop --n 4 --out out_4_5ns --cp 5.00 --resolver

python3 experimental/tools/integration/run_specv2_integration.py single_loop --n 0 --out out_0_om --cp 5.00 --resolver --use-prof-cache --min-buffering
python3 experimental/tools/integration/run_specv2_integration.py single_loop --n 1 --out out_1_om --cp 5.00 --resolver --use-prof-cache --min-buffering
python3 experimental/tools/integration/run_specv2_integration.py single_loop --n 2 --out out_2_om --cp 5.00 --resolver --use-prof-cache --min-buffering
python3 experimental/tools/integration/run_specv2_integration.py single_loop --n 3 --out out_3_om --cp 5.00 --resolver --use-prof-cache --min-buffering
python3 experimental/tools/integration/run_specv2_integration.py single_loop --n 4 --out out_4_om --cp 5.00 --resolver --use-prof-cache --min-buffering
python3 experimental/tools/integration/run_specv2_integration.py single_loop --n 5 --out out_5_om --cp 5.00 --resolver --use-prof-cache --min-buffering
