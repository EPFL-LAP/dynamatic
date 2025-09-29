SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DYNAMATIC_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd $DYNAMATIC_DIR

python3 experimental/tools/integration/run_specv2_integration.py collision_donut --disable-spec --out out_standard --cp 5.00
python3 experimental/tools/integration/run_specv2_integration.py collision_donut --n 0 --out out_0 --cp 5.00 --resolver
python3 experimental/tools/integration/run_specv2_integration.py collision_donut --n 1 --out out_1 --cp 5.00 --use-prof-cache --resolver
python3 experimental/tools/integration/run_specv2_integration.py collision_donut --n 2 --out out_2 --cp 5.00 --use-prof-cache --resolver
python3 experimental/tools/integration/run_specv2_integration.py collision_donut --n 3 --out out_3 --cp 5.00 --use-prof-cache --resolver
python3 experimental/tools/integration/run_specv2_integration.py collision_donut --n 4 --out out_4 --cp 5.00 --use-prof-cache --resolver
python3 experimental/tools/integration/run_specv2_integration.py collision_donut --n 5 --out out_5 --cp 5.00 --use-prof-cache --resolver
python3 experimental/tools/integration/run_specv2_integration.py collision_donut --n 6 --out out_6 --cp 5.00 --use-prof-cache --resolver
python3 experimental/tools/integration/run_specv2_integration.py collision_donut --n 7 --out out_7 --cp 5.00 --use-prof-cache --resolver
python3 experimental/tools/integration/run_specv2_integration.py collision_donut --n 8 --out out_8 --cp 5.00 --use-prof-cache --resolver
python3 experimental/tools/integration/run_specv2_integration.py collision_donut --n 9 --out out_9 --cp 5.00 --use-prof-cache --resolver
python3 experimental/tools/integration/run_specv2_integration.py collision_donut --n 10 --out out_10 --cp 5.00 --use-prof-cache --resolver
python3 experimental/tools/integration/run_specv2_integration.py collision_donut --n 11 --out out_11 --cp 5.00 --use-prof-cache --resolver
python3 experimental/tools/integration/run_specv2_integration.py collision_donut --n 12 --out out_12 --cp 5.00 --use-prof-cache --resolver
python3 experimental/tools/integration/run_specv2_integration.py collision_donut --n 13 --out out_13 --cp 5.00 --use-prof-cache --resolver
python3 experimental/tools/integration/run_specv2_integration.py collision_donut --n 14 --out out_14 --cp 5.00 --use-prof-cache --resolver
python3 experimental/tools/integration/run_specv2_integration.py collision_donut --n 15 --out out_15 --cp 5.00 --use-prof-cache --resolver
python3 experimental/tools/integration/run_specv2_integration.py collision_donut --n 16 --out out_16 --cp 5.00 --use-prof-cache --resolver
python3 experimental/tools/integration/run_specv2_integration.py collision_donut --n 17 --out out_17 --cp 5.00 --use-prof-cache --resolver
python3 experimental/tools/integration/run_specv2_integration.py collision_donut --n 18 --out out_18 --cp 5.00 --use-prof-cache --resolver
python3 experimental/tools/integration/run_specv2_integration.py collision_donut --n 19 --out out_19 --cp 5.00 --use-prof-cache --resolver
python3 experimental/tools/integration/run_specv2_integration.py collision_donut --n 20 --out out_20 --cp 5.00 --use-prof-cache --resolver
