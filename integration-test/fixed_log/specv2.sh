SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DYNAMATIC_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd $DYNAMATIC_DIR

# Baseline
python3 experimental/tools/integration/run_specv2_integration.py fixed_log --baseline --out out_baseline --cp 8.00

# Spec v1
python3 experimental/tools/integration/run_adapted_spec_integration.py fixed_log --default-value=0 --out out_v1 --cp 8.00

python3 experimental/tools/integration/run_specv2_integration.py fixed_log --disable-initial-motion --n 0 --out out_default --cp 8.00 --resolver
python3 experimental/tools/integration/run_specv2_integration.py fixed_log --n 0 --out out_0 --cp 8.00 --resolver
python3 experimental/tools/integration/run_specv2_integration.py fixed_log --n 1 --out out_1 --cp 8.00 --resolver --use-prof-cache
python3 experimental/tools/integration/run_specv2_integration.py fixed_log --n 2 --out out_2 --cp 8.00 --resolver --use-prof-cache
python3 experimental/tools/integration/run_specv2_integration.py fixed_log --n 3 --out out_3 --cp 8.00 --resolver --use-prof-cache
python3 experimental/tools/integration/run_specv2_integration.py fixed_log --n 4 --out out_4 --cp 8.00 --resolver --use-prof-cache
python3 experimental/tools/integration/run_specv2_integration.py fixed_log --n 5 --out out_5 --cp 8.00 --resolver --use-prof-cache
python3 experimental/tools/integration/run_specv2_integration.py fixed_log --n 6 --out out_6 --cp 8.00 --resolver --use-prof-cache
python3 experimental/tools/integration/run_specv2_integration.py fixed_log --n 7 --out out_7 --cp 8.00 --resolver --use-prof-cache
python3 experimental/tools/integration/run_specv2_integration.py fixed_log --n 8 --out out_8 --cp 8.00 --resolver --use-prof-cache
python3 experimental/tools/integration/run_specv2_integration.py fixed_log --n 9 --out out_9 --cp 8.00 --resolver --use-prof-cache
python3 experimental/tools/integration/run_specv2_integration.py fixed_log --n 10 --out out_10 --cp 8.00 --resolver --use-prof-cache
python3 experimental/tools/integration/run_specv2_integration.py fixed_log --n 11 --out out_11 --cp 8.00 --resolver --use-prof-cache
python3 experimental/tools/integration/run_specv2_integration.py fixed_log --n 12 --out out_12 --cp 8.00 --resolver --use-prof-cache
python3 experimental/tools/integration/run_specv2_integration.py fixed_log --n 13 --out out_13 --cp 8.00 --resolver --use-prof-cache
python3 experimental/tools/integration/run_specv2_integration.py fixed_log --n 14 --out out_14 --cp 8.00 --resolver --use-prof-cache
python3 experimental/tools/integration/run_specv2_integration.py fixed_log --n 15 --out out_15 --cp 8.00 --resolver --use-prof-cache
python3 experimental/tools/integration/run_specv2_integration.py fixed_log --n 16 --out out_16 --cp 8.00 --resolver --use-prof-cache
python3 experimental/tools/integration/run_specv2_integration.py fixed_log --n 17 --out out_17 --cp 8.00 --resolver --use-prof-cache
python3 experimental/tools/integration/run_specv2_integration.py fixed_log --n 18 --out out_18 --cp 8.00 --resolver --use-prof-cache
python3 experimental/tools/integration/run_specv2_integration.py fixed_log --n 19 --out out_19 --cp 8.00 --resolver --use-prof-cache
python3 experimental/tools/integration/run_specv2_integration.py fixed_log --n 20 --out out_20 --cp 8.00 --resolver --use-prof-cache
