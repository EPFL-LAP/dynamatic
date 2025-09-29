SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DYNAMATIC_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd $DYNAMATIC_DIR

# python3 experimental/tools/integration/run_specv2_integration.py subdiag_fast --n 1 --out out_1 --cp 8.00 --resolver
# python3 experimental/tools/integration/run_specv2_integration.py subdiag_fast --n 2 --out out_2 --cp 8.00 --use-prof-cache --resolver
# python3 experimental/tools/integration/run_specv2_integration.py subdiag_fast --n 3 --out out_3 --cp 8.00 --use-prof-cache --resolver
# python3 experimental/tools/integration/run_specv2_integration.py subdiag_fast --n 4 --out out_4 --cp 8.00 --use-prof-cache --resolver
# python3 experimental/tools/integration/run_specv2_integration.py subdiag_fast --n 5 --out out_5 --cp 8.00 --use-prof-cache --resolver
# python3 experimental/tools/integration/run_specv2_integration.py subdiag_fast --n 6 --out out_6 --cp 8.00 --use-prof-cache --resolver
# python3 experimental/tools/integration/run_specv2_integration.py subdiag_fast --n 7 --out out_7 --cp 8.00 --use-prof-cache --resolver
# python3 experimental/tools/integration/run_specv2_integration.py subdiag_fast --n 8 --out out_8 --cp 8.00 --use-prof-cache --resolver
# python3 experimental/tools/integration/run_specv2_integration.py subdiag_fast --n 9 --out out_9 --cp 8.00 --use-prof-cache --resolver
# python3 experimental/tools/integration/run_specv2_integration.py subdiag_fast --n 10 --out out_10 --cp 8.00 --use-prof-cache --resolver
# python3 experimental/tools/integration/run_specv2_integration.py subdiag_fast --n 11 --out out_11 --cp 8.00 --use-prof-cache --resolver
# python3 experimental/tools/integration/run_specv2_integration.py subdiag_fast --n 12 --out out_12 --cp 8.00 --use-prof-cache --resolver
# python3 experimental/tools/integration/run_specv2_integration.py subdiag_fast --n 13 --out out_13 --cp 8.00 --use-prof-cache --resolver
# python3 experimental/tools/integration/run_specv2_integration.py subdiag_fast --n 14 --out out_14 --cp 8.00 --use-prof-cache --resolver
# python3 experimental/tools/integration/run_specv2_integration.py subdiag_fast --n 15 --out out_15 --cp 8.00 --use-prof-cache --resolver
# python3 experimental/tools/integration/run_specv2_integration.py subdiag_fast --n 16 --out out_16 --cp 8.00 --use-prof-cache --resolver
# python3 experimental/tools/integration/run_specv2_integration.py subdiag_fast --n 17 --out out_17 --cp 8.00 --use-prof-cache --resolver
# python3 experimental/tools/integration/run_specv2_integration.py subdiag_fast --n 18 --out out_18 --cp 8.00 --use-prof-cache --resolver
# python3 experimental/tools/integration/run_specv2_integration.py subdiag_fast --n 19 --out out_19 --cp 8.00 --use-prof-cache --resolver
# python3 experimental/tools/integration/run_specv2_integration.py subdiag_fast --n 20 --out out_20 --cp 8.00 --use-prof-cache --resolver
# python3 experimental/tools/integration/run_specv2_integration.py subdiag_fast --n 25 --out out_25 --cp 8.00 --use-prof-cache --resolver
# python3 experimental/tools/integration/run_specv2_integration.py subdiag_fast --n 30 --out out_30 --cp 8.00 --use-prof-cache --resolver
# python3 experimental/tools/integration/run_specv2_integration.py subdiag_fast --n 35 --out out_35 --cp 8.00 --use-prof-cache --resolver
# python3 experimental/tools/integration/run_specv2_integration.py subdiag_fast --n 40 --out out_40 --cp 8.00 --use-prof-cache --resolver
# python3 experimental/tools/integration/run_specv2_integration.py subdiag_fast --n 45 --out out_45 --cp 8.00 --use-prof-cache --resolver
# python3 experimental/tools/integration/run_specv2_integration.py subdiag_fast --n 50 --out out_50 --cp 8.00 --use-prof-cache --resolver

python3 experimental/tools/integration/run_specv2_integration.py subdiag_fast --n 60 --out out_60 --cp 8.00 --use-prof-cache --resolver
python3 experimental/tools/integration/run_specv2_integration.py subdiag_fast --n 70 --out out_70 --cp 8.00 --use-prof-cache --resolver
python3 experimental/tools/integration/run_specv2_integration.py subdiag_fast --n 80 --out out_80 --cp 8.00 --use-prof-cache --resolver
python3 experimental/tools/integration/run_specv2_integration.py subdiag_fast --n 90 --out out_90 --cp 8.00 --use-prof-cache --resolver
python3 experimental/tools/integration/run_specv2_integration.py subdiag_fast --n 100 --out out_100 --cp 8.00 --use-prof-cache --resolver

