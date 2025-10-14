SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DYNAMATIC_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd $DYNAMATIC_DIR

# Baseline
python3 experimental/tools/integration/run_gamma_integration.py if_float --branch-bb=1 --merge-bb=4 --steps-until 0 --prioritized-side 0 --out out_baseline --cp 5.00

# spec v1
python3 experimental/tools/integration/run_adapted_spec_integration.py if_float --default-value=0 --out out_v1

# python3 experimental/tools/integration/run_gamma_integration.py if_float --disable-spec --out out_standard
python3 experimental/tools/integration/run_gamma_integration.py if_float --branch-bb=1 --merge-bb=4 --prioritized-side 0 --one-sided --out out_0_one_sided
python3 experimental/tools/integration/run_gamma_integration.py if_float --branch-bb=1 --merge-bb=4 --prioritized-side 0 --out out_0_two_sided --use-prof-cache
python3 experimental/tools/integration/run_gamma_integration.py if_float --branch-bb=1 --merge-bb=4 --prioritized-side 1 --one-sided --out out_1_one_sided --use-prof-cache
python3 experimental/tools/integration/run_gamma_integration.py if_float --branch-bb=1 --merge-bb=4 --prioritized-side 1 --out out_1_two_sided --use-prof-cache
# python3 experimental/tools/integration/run_gamma_integration.py if_float --branch-bb=1 --merge-bb=4 --prioritized-side 1 --one-sided --steps-until 0 --out out_straight