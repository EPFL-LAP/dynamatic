SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DYNAMATIC_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd $DYNAMATIC_DIR

# Estimation
# python3 experimental/tools/integration/run_specv2_integration.py newton --decide-n > integration-test/newton/decide-n.txt

# Baseline
python3 experimental/tools/integration/run_specv2_integration.py newton --baseline --out out_baseline_7ns --cp 7.00

# Spec v1
python3 experimental/tools/integration/run_adapted_spec_integration.py newton --out out_v1_7ns --cp 7.00 --default-value 0

python3 experimental/tools/integration/run_specv2_integration.py newton --n 1 --out out_1_7ns --cp 7.00 --resolver
python3 experimental/tools/integration/run_specv2_integration.py newton --n 10 --out out_10_7ns --cp 7.00 --resolver --use-prof-cache
python3 experimental/tools/integration/run_specv2_integration.py newton --n 20 --out out_20_7ns --cp 7.00 --resolver --use-prof-cache
