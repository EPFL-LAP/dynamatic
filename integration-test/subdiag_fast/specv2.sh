SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DYNAMATIC_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd $DYNAMATIC_DIR

# Baseline:
# - No CMerge
# - No Branch
python3 experimental/tools/integration/run_specv2_integration.py subdiag_fast --baseline --out out_baseline --cp 8.00

# Spec v1
# python3 tools/integration/run_spec_integration.py subdiag_fast --out out_v1 --cp 8.00 --transformed-code subdiag_fast_v1.c
python3 experimental/tools/integration/run_adapted_spec_integration.py subdiag_fast --transformed-code subdiag_fast_v1.c --cp 8.00 --out out_v1

# python3 experimental/tools/integration/run_specv2_integration.py subdiag_fast --disable-spec --out out_standard --cp 8.00

# python3 experimental/tools/integration/run_specv2_integration.py subdiag_fast --n 12 --out out_12 --cp 15.00
# python3 experimental/tools/integration/run_specv2_integration.py subdiag_fast --n 13 --out out_13 --cp 8.00 --resolver
# python3 experimental/tools/integration/run_specv2_integration.py subdiag_fast --n 14 --out out_14 --cp 15.00
# python3 experimental/tools/integration/run_specv2_integration.py subdiag_fast --variable --out out_variable --cp 5.00
