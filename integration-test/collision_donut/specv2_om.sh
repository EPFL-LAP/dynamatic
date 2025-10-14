SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DYNAMATIC_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd $DYNAMATIC_DIR

# Baseline
python3 experimental/tools/integration/run_specv2_integration.py collision_donut --baseline --out out_baseline_om --cp 10.00 --min-buffering
python3 experimental/tools/integration/run_specv2_integration.py collision_donut --n 0 --out out_0_om --cp 5.00 --resolver --min-buffering
python3 experimental/tools/integration/run_specv2_integration.py collision_donut --n 1 --out out_1_om --cp 5.00 --use-prof-cache --resolver --min-buffering
python3 experimental/tools/integration/run_specv2_integration.py collision_donut --n 2 --out out_2_om --cp 5.00 --use-prof-cache --resolver --min-buffering
python3 experimental/tools/integration/run_specv2_integration.py collision_donut --n 3 --out out_3_om --cp 5.00 --use-prof-cache --resolver --min-buffering
python3 experimental/tools/integration/run_specv2_integration.py collision_donut --n 4 --out out_4_om --cp 5.00 --use-prof-cache --resolver --min-buffering
python3 experimental/tools/integration/run_specv2_integration.py collision_donut --n 5 --out out_5_om --cp 5.00 --use-prof-cache --resolver --min-buffering