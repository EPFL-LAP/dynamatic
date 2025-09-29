SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DYNAMATIC_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd $DYNAMATIC_DIR

# Spec v1
python3 tools/integration/run_spec_integration.py fixed --out out_v1 --cp 5.00 --default-value 0

python3 experimental/tools/integration/run_specv2_integration.py fixed --n 0 --out out_0 --cp 5.00 --resolver
python3 experimental/tools/integration/run_specv2_integration.py fixed --n 1 --out out_1 --cp 5.00 --use-prof-cache --resolver
python3 experimental/tools/integration/run_specv2_integration.py fixed --n 2 --out out_2 --cp 5.00 --use-prof-cache --resolver
python3 experimental/tools/integration/run_specv2_integration.py fixed --n 3 --out out_3 --cp 5.00 --use-prof-cache --resolver
python3 experimental/tools/integration/run_specv2_integration.py fixed --n 4 --out out_4 --cp 5.00 --use-prof-cache --resolver
python3 experimental/tools/integration/run_specv2_integration.py fixed --n 5 --out out_5 --cp 5.00 --use-prof-cache --resolver
python3 experimental/tools/integration/run_specv2_integration.py fixed --n 6 --out out_6 --cp 5.00 --use-prof-cache --resolver
python3 experimental/tools/integration/run_specv2_integration.py fixed --n 7 --out out_7 --cp 5.00 --use-prof-cache --resolver
python3 experimental/tools/integration/run_specv2_integration.py fixed --n 8 --out out_8 --cp 5.00 --use-prof-cache --resolver
python3 experimental/tools/integration/run_specv2_integration.py fixed --n 9 --out out_9 --cp 5.00 --use-prof-cache --resolver
python3 experimental/tools/integration/run_specv2_integration.py fixed --n 10 --out out_10 --cp 5.00 --use-prof-cache --resolver
python3 experimental/tools/integration/run_specv2_integration.py fixed --n 11 --out out_11 --cp 5.00 --use-prof-cache --resolver
python3 experimental/tools/integration/run_specv2_integration.py fixed --n 12 --out out_12 --cp 5.00 --use-prof-cache --resolver
python3 experimental/tools/integration/run_specv2_integration.py fixed --n 13 --out out_13 --cp 5.00 --use-prof-cache --resolver
python3 experimental/tools/integration/run_specv2_integration.py fixed --n 14 --out out_14 --cp 5.00 --use-prof-cache --resolver
python3 experimental/tools/integration/run_specv2_integration.py fixed --n 15 --out out_15 --cp 5.00 --use-prof-cache --resolver
python3 experimental/tools/integration/run_specv2_integration.py fixed --n 16 --out out_16 --cp 5.00 --use-prof-cache --resolver
python3 experimental/tools/integration/run_specv2_integration.py fixed --n 17 --out out_17 --cp 5.00 --use-prof-cache --resolver
python3 experimental/tools/integration/run_specv2_integration.py fixed --n 18 --out out_18 --cp 5.00 --use-prof-cache --resolver
python3 experimental/tools/integration/run_specv2_integration.py fixed --n 19 --out out_19 --cp 5.00 --use-prof-cache --resolver
python3 experimental/tools/integration/run_specv2_integration.py fixed --n 20 --out out_20 --cp 5.00 --use-prof-cache --resolver
