SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DYNAMATIC_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd $DYNAMATIC_DIR

python3 experimental/tools/integration/run_specv2_integration.py collision_donut --disable-spec --out out_standard --cp 5.00
python3 experimental/tools/integration/run_specv2_integration.py collision_donut --n 5 --out out_5 --cp 5.00
