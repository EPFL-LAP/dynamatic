SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DYNAMATIC_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd $DYNAMATIC_DIR

# python3 experimental/tools/integration/run_specv2_integration.py single_loop --n 0 --out out_resolver_0 --cp 5.00 --resolver
# python3 experimental/tools/integration/run_specv2_integration.py single_loop --n 1 --out out_resolver_1 --cp 5.00 --use-prof-cache --resolver
# python3 experimental/tools/integration/run_specv2_integration.py single_loop --n 2 --out out_resolver_2 --cp 5.00 --use-prof-cache --resolver
# python3 experimental/tools/integration/run_specv2_integration.py single_loop --n 3 --out out_resolver_3 --cp 5.00 --use-prof-cache --resolver
# python3 experimental/tools/integration/run_specv2_integration.py single_loop --n 4 --out out_resolver_4 --cp 5.00 --use-prof-cache --resolver
# python3 experimental/tools/integration/run_specv2_integration.py single_loop --n 5 --out out_resolver_5 --cp 5.00 --use-prof-cache --resolver
# python3 experimental/tools/integration/run_specv2_integration.py single_loop --n 6 --out out_resolver_6 --cp 5.00 --use-prof-cache --resolver
# python3 experimental/tools/integration/run_specv2_integration.py single_loop --n 7 --out out_resolver_7 --cp 5.00 --use-prof-cache --resolver
# python3 experimental/tools/integration/run_specv2_integration.py single_loop --n 8 --out out_resolver_8 --cp 5.00 --use-prof-cache --resolver
# python3 experimental/tools/integration/run_specv2_integration.py single_loop --n 9 --out out_resolver_9 --cp 5.00 --use-prof-cache --resolver
# python3 experimental/tools/integration/run_specv2_integration.py single_loop --n 10 --out out_resolver_10 --cp 5.00 --use-prof-cache --resolver
# python3 experimental/tools/integration/run_specv2_integration.py single_loop --n 11 --out out_resolver_11 --cp 5.00 --use-prof-cache --resolver
# python3 experimental/tools/integration/run_specv2_integration.py single_loop --n 12 --out out_resolver_12 --cp 5.00 --use-prof-cache --resolver
# python3 experimental/tools/integration/run_specv2_integration.py single_loop --n 13 --out out_resolver_13 --cp 5.00 --use-prof-cache --resolver
# python3 experimental/tools/integration/run_specv2_integration.py single_loop --n 14 --out out_resolver_14 --cp 5.00 --use-prof-cache --resolver
# python3 experimental/tools/integration/run_specv2_integration.py single_loop --n 15 --out out_resolver_15 --cp 5.00 --use-prof-cache --resolver
# python3 experimental/tools/integration/run_specv2_integration.py single_loop --n 16 --out out_resolver_16 --cp 5.00 --use-prof-cache --resolver
# python3 experimental/tools/integration/run_specv2_integration.py single_loop --n 17 --out out_resolver_17 --cp 5.00 --use-prof-cache --resolver
# python3 experimental/tools/integration/run_specv2_integration.py single_loop --n 18 --out out_resolver_18 --cp 5.00 --use-prof-cache --resolver
# python3 experimental/tools/integration/run_specv2_integration.py single_loop --n 19 --out out_resolver_19 --cp 5.00 --use-prof-cache --resolver
# python3 experimental/tools/integration/run_specv2_integration.py single_loop --n 20 --out out_resolver_20 --cp 5.00 --use-prof-cache --resolver
python3 experimental/tools/integration/run_specv2_integration.py single_loop --n 50 --out out_resolver_50 --cp 5.00 --use-prof-cache --resolver
python3 experimental/tools/integration/run_specv2_integration.py single_loop --n 100 --out out_resolver_100 --cp 5.00 --use-prof-cache --resolver
