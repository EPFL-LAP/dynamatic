SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DYNAMATIC_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd $DYNAMATIC_DIR

# python3 experimental/tools/integration/run_specv2_integration.py single_loop --n 0 --out out_0 --cp 25.00
# python3 experimental/tools/integration/run_specv2_integration.py single_loop --n 1 --out out_1 --cp 25.00 --use-prof-cache
# python3 experimental/tools/integration/run_specv2_integration.py single_loop --n 2 --out out_2 --cp 25.00 --use-prof-cache
# python3 experimental/tools/integration/run_specv2_integration.py single_loop --n 3 --out out_3 --cp 25.00 --use-prof-cache
# python3 experimental/tools/integration/run_specv2_integration.py single_loop --n 4 --out out_4 --cp 25.00 --use-prof-cache
# python3 experimental/tools/integration/run_specv2_integration.py single_loop --n 5 --out out_5 --cp 25.00 --use-prof-cache
# python3 experimental/tools/integration/run_specv2_integration.py single_loop --n 6 --out out_6 --cp 25.00 --use-prof-cache
# python3 experimental/tools/integration/run_specv2_integration.py single_loop --n 7 --out out_7 --cp 25.00 --use-prof-cache
# python3 experimental/tools/integration/run_specv2_integration.py single_loop --n 8 --out out_8 --cp 25.00 --use-prof-cache
# python3 experimental/tools/integration/run_specv2_integration.py single_loop --n 9 --out out_9 --cp 25.00 --use-prof-cache
# python3 experimental/tools/integration/run_specv2_integration.py single_loop --n 10 --out out_10 --cp 25.00 --use-prof-cache
# python3 experimental/tools/integration/run_specv2_integration.py single_loop --n 11 --out out_11 --cp 25.00 --use-prof-cache
# python3 experimental/tools/integration/run_specv2_integration.py single_loop --n 12 --out out_12 --cp 25.00 --use-prof-cache
# python3 experimental/tools/integration/run_specv2_integration.py single_loop --n 13 --out out_13 --cp 25.00 --use-prof-cache
# python3 experimental/tools/integration/run_specv2_integration.py single_loop --n 14 --out out_14 --cp 25.00 --use-prof-cache
# python3 experimental/tools/integration/run_specv2_integration.py single_loop --n 15 --out out_15 --cp 25.00 --use-prof-cache
# python3 experimental/tools/integration/run_specv2_integration.py single_loop --n 16 --out out_16 --cp 25.00 --use-prof-cache
# python3 experimental/tools/integration/run_specv2_integration.py single_loop --n 17 --out out_17 --cp 25.00 --use-prof-cache
# python3 experimental/tools/integration/run_specv2_integration.py single_loop --n 18 --out out_18 --cp 25.00 --use-prof-cache
# python3 experimental/tools/integration/run_specv2_integration.py single_loop --n 19 --out out_19 --cp 25.00 --use-prof-cache
# python3 experimental/tools/integration/run_specv2_integration.py single_loop --n 20 --out out_20 --cp 25.00 --use-prof-cache

python3 experimental/tools/integration/run_specv2_integration.py single_loop --n 50 --out out_50 --cp 200.00 --use-prof-cache
python3 experimental/tools/integration/run_specv2_integration.py single_loop --n 100 --out out_100 --cp 200.00 --use-prof-cache