SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DYNAMATIC_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd $DYNAMATIC_DIR

# Baseline
# python3 experimental/tools/integration/run_specv2_integration.py bisection --baseline --out out_baseline_5ns --cp 5.00 --transformed-code bisection_transformed.c

# python3 experimental/tools/integration/run_specv2_integration.py bisection --n 2 --out out_2_5ns --cp 5.00 --resolver --transformed-code bisection_transformed.c
# python3 experimental/tools/integration/run_specv2_integration.py bisection --n 10 --out out_10_5ns --cp 5.00 --use-prof-cache --resolver --transformed-code bisection_transformed.c
# python3 experimental/tools/integration/run_specv2_integration.py bisection --n 20 --out out_20_5ns --cp 5.00 --use-prof-cache --resolver --transformed-code bisection_transformed.c


python3 experimental/tools/integration/run_specv2_integration.py bisection --n 0 --out out_0_5ns --cp 5.00 --resolver --transformed-code bisection_transformed.c
python3 experimental/tools/integration/run_specv2_integration.py bisection --n 1 --out out_1_5ns --cp 5.00 --resolver --transformed-code bisection_transformed.c
python3 experimental/tools/integration/run_specv2_integration.py bisection --n 3 --out out_3_5ns --cp 5.00 --resolver --use-prof-cache --transformed-code bisection_transformed.c
python3 experimental/tools/integration/run_specv2_integration.py bisection --n 4 --out out_4_5ns --cp 5.00 --resolver --use-prof-cache --transformed-code bisection_transformed.c
python3 experimental/tools/integration/run_specv2_integration.py bisection --n 5 --out out_5_5ns --cp 5.00 --resolver --use-prof-cache --transformed-code bisection_transformed.c
python3 experimental/tools/integration/run_specv2_integration.py bisection --n 6 --out out_6_5ns --cp 5.00 --resolver --use-prof-cache --transformed-code bisection_transformed.c
python3 experimental/tools/integration/run_specv2_integration.py bisection --n 7 --out out_7_5ns --cp 5.00 --resolver --use-prof-cache --transformed-code bisection_transformed.c
python3 experimental/tools/integration/run_specv2_integration.py bisection --n 8 --out out_8_5ns --cp 5.00 --resolver --use-prof-cache --transformed-code bisection_transformed.c
python3 experimental/tools/integration/run_specv2_integration.py bisection --n 9 --out out_9_5ns --cp 5.00 --resolver --use-prof-cache --transformed-code bisection_transformed.c
python3 experimental/tools/integration/run_specv2_integration.py bisection --n 11 --out out_11_5ns --cp 5.00 --resolver --use-prof-cache --transformed-code bisection_transformed.c
python3 experimental/tools/integration/run_specv2_integration.py bisection --n 12 --out out_12_5ns --cp 5.00 --resolver --use-prof-cache --transformed-code bisection_transformed.c
python3 experimental/tools/integration/run_specv2_integration.py bisection --n 13 --out out_13_5ns --cp 5.00 --resolver --use-prof-cache --transformed-code bisection_transformed.c
python3 experimental/tools/integration/run_specv2_integration.py bisection --n 14 --out out_14_5ns --cp 5.00 --resolver --use-prof-cache --transformed-code bisection_transformed.c
python3 experimental/tools/integration/run_specv2_integration.py bisection --n 15 --out out_15_5ns --cp 5.00 --resolver --use-prof-cache --transformed-code bisection_transformed.c
python3 experimental/tools/integration/run_specv2_integration.py bisection --n 16 --out out_16_5ns --cp 5.00 --resolver --use-prof-cache --transformed-code bisection_transformed.c
python3 experimental/tools/integration/run_specv2_integration.py bisection --n 17 --out out_17_5ns --cp 5.00 --resolver --use-prof-cache --transformed-code bisection_transformed.c
python3 experimental/tools/integration/run_specv2_integration.py bisection --n 18 --out out_18_5ns --cp 5.00 --resolver --use-prof-cache --transformed-code bisection_transformed.c
python3 experimental/tools/integration/run_specv2_integration.py bisection --n 19 --out out_19_5ns --cp 5.00 --resolver --use-prof-cache --transformed-code bisection_transformed.c