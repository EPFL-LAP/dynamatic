SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DYNAMATIC_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd $DYNAMATIC_DIR

# python3 experimental/tools/integration/run_specv2_integration.py bisection --disable-spec --cp 10.00 --out out_standard
# python3 experimental/tools/integration/run_specv2_integration.py bisection --gate-binarized cf_transformed.mlir --n 0 --out out_pre --cp 10.00 --disable-initial-motion --resolver
python3 experimental/tools/integration/run_specv2_integration.py bisection --gate-binarized cf_transformed.mlir --n 1 --out out_1 --cp 10.00 --use-prof-cache --resolver
python3 experimental/tools/integration/run_specv2_integration.py bisection --gate-binarized cf_transformed.mlir --n 2 --out out_2 --cp 10.00 --use-prof-cache --resolver
python3 experimental/tools/integration/run_specv2_integration.py bisection --gate-binarized cf_transformed.mlir --n 3 --out out_3 --cp 10.00 --use-prof-cache --resolver
python3 experimental/tools/integration/run_specv2_integration.py bisection --gate-binarized cf_transformed.mlir --n 4 --out out_4 --cp 10.00 --use-prof-cache --resolver
python3 experimental/tools/integration/run_specv2_integration.py bisection --gate-binarized cf_transformed.mlir --n 5 --out out_5 --cp 10.00 --use-prof-cache --resolver
python3 experimental/tools/integration/run_specv2_integration.py bisection --gate-binarized cf_transformed.mlir --n 6 --out out_6 --cp 10.00 --use-prof-cache --resolver
python3 experimental/tools/integration/run_specv2_integration.py bisection --gate-binarized cf_transformed.mlir --n 7 --out out_7 --cp 10.00 --use-prof-cache --resolver
python3 experimental/tools/integration/run_specv2_integration.py bisection --gate-binarized cf_transformed.mlir --n 8 --out out_8 --cp 10.00 --use-prof-cache --resolver
python3 experimental/tools/integration/run_specv2_integration.py bisection --gate-binarized cf_transformed.mlir --n 9 --out out_9 --cp 10.00 --use-prof-cache --resolver
python3 experimental/tools/integration/run_specv2_integration.py bisection --gate-binarized cf_transformed.mlir --n 10 --out out_10 --cp 10.00 --use-prof-cache --resolver
python3 experimental/tools/integration/run_specv2_integration.py bisection --gate-binarized cf_transformed.mlir --n 11 --out out_11 --cp 10.00 --use-prof-cache --resolver
python3 experimental/tools/integration/run_specv2_integration.py bisection --gate-binarized cf_transformed.mlir --n 12 --out out_12 --cp 10.00 --use-prof-cache --resolver
python3 experimental/tools/integration/run_specv2_integration.py bisection --gate-binarized cf_transformed.mlir --n 13 --out out_13 --cp 10.00 --use-prof-cache --resolver
python3 experimental/tools/integration/run_specv2_integration.py bisection --gate-binarized cf_transformed.mlir --n 14 --out out_14 --cp 10.00 --use-prof-cache --resolver
python3 experimental/tools/integration/run_specv2_integration.py bisection --gate-binarized cf_transformed.mlir --n 15 --out out_15 --cp 10.00 --use-prof-cache --resolver
python3 experimental/tools/integration/run_specv2_integration.py bisection --gate-binarized cf_transformed.mlir --n 16 --out out_16 --cp 10.00 --use-prof-cache --resolver
python3 experimental/tools/integration/run_specv2_integration.py bisection --gate-binarized cf_transformed.mlir --n 17 --out out_17 --cp 10.00 --use-prof-cache --resolver
python3 experimental/tools/integration/run_specv2_integration.py bisection --gate-binarized cf_transformed.mlir --n 18 --out out_18 --cp 10.00 --use-prof-cache --resolver
python3 experimental/tools/integration/run_specv2_integration.py bisection --gate-binarized cf_transformed.mlir --n 19 --out out_19 --cp 10.00 --use-prof-cache --resolver
python3 experimental/tools/integration/run_specv2_integration.py bisection --gate-binarized cf_transformed.mlir --n 20 --out out_20 --cp 10.00 --use-prof-cache --resolver

# python3 experimental/tools/integration/run_specv2_integration.py single_loop --variable --out out_variable --cp 5.00
