SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DYNAMATIC_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd $DYNAMATIC_DIR

# Baseline
python3 experimental/tools/integration/run_specv2_integration.py fixed_log --baseline --out out_baseline_5ns --cp 5.00

# Spec v1
python3 experimental/tools/integration/run_adapted_spec_integration.py fixed_log --default-value=0 --out out_v1_5ns --cp 5.00

python3 experimental/tools/integration/run_specv2_integration.py fixed_log --n 3 --out out_3_5ns --cp 5.00 --resolver --use-prof-cache
python3 experimental/tools/integration/run_specv2_integration.py fixed_log --n 10 --out out_10_5ns --cp 5.00 --resolver --use-prof-cache
python3 experimental/tools/integration/run_specv2_integration.py fixed_log --n 20 --out out_20_5ns --cp 5.00 --resolver --use-prof-cache
