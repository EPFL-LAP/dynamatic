SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DYNAMATIC_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd $DYNAMATIC_DIR

# Standard flow
python3 experimental/tools/integration/run_specv2_integration.py if_convert --disable-spec --out out_standard

# Only code transformation
python3 experimental/tools/integration/run_specv2_integration.py if_convert --disable-spec --transformed-code if_convert_transformed.c --out out_transformed

# Spec v1
python3 tools/integration/run_spec_integration.py if_convert --out out_v1

python3 experimental/tools/integration/run_specv2_integration.py if_convert --n 0 --out out_0 --transformed-code if_convert_transformed.c
python3 experimental/tools/integration/run_specv2_integration.py if_convert --n 1 --out out_1 --transformed-code if_convert_transformed.c
python3 experimental/tools/integration/run_specv2_integration.py if_convert --n 2 --out out_2 --transformed-code if_convert_transformed.c
python3 experimental/tools/integration/run_specv2_integration.py if_convert --n 3 --out out_3 --transformed-code if_convert_transformed.c
python3 experimental/tools/integration/run_specv2_integration.py if_convert --n 4 --out out_4 --transformed-code if_convert_transformed.c
python3 experimental/tools/integration/run_specv2_integration.py if_convert --n 5 --out out_5 --transformed-code if_convert_transformed.c
python3 experimental/tools/integration/run_specv2_integration.py if_convert --variable --out out_variable --transformed-code if_convert_transformed.c
