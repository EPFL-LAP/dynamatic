SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DYNAMATIC_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd $DYNAMATIC_DIR

# Spec v1
python3 tools/integration/run_spec_integration.py if_convert --out out_v1 --cp 10.00 --default-value 0

python3 experimental/tools/integration/run_specv2_integration.py if_convert --n 0 --out out_0 --cp 10.00 --resolver
python3 experimental/tools/integration/run_specv2_integration.py if_convert --n 5 --out out_5 --cp 10.00 --use-prof-cache --resolver

python3 experimental/tools/integration/run_specv2_integration.py if_convert --decide-n > integration-test/if_convert/decide-n.txt
