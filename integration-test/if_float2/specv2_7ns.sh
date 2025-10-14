SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DYNAMATIC_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd $DYNAMATIC_DIR

# Baseline
python3 experimental/tools/integration/run_gamma_integration.py if_float2 --branch-bb=1 --merge-bb=4 --steps-until 0 --prioritized-side 0 --out out_baseline_7ns --cp 7.00

# spec v1
python3 experimental/tools/integration/run_adapted_spec_integration.py if_float2 --default-value=0 --out out_v1_7ns --cp 7.00 --loop-bottom-passer-disabled

python3 experimental/tools/integration/run_gamma_integration.py if_float2 --branch-bb=1 --merge-bb=4 --prioritized-side 0 --one-sided --out out_0_one_sided_7ns --cp 7.00
python3 experimental/tools/integration/run_gamma_integration.py if_float2 --branch-bb=1 --merge-bb=4 --prioritized-side 0 --out out_0_two_sided_7ns --use-prof-cache --cp 7.00
python3 experimental/tools/integration/run_gamma_integration.py if_float2 --branch-bb=1 --merge-bb=4 --prioritized-side 1 --one-sided --out out_1_one_sided_7ns --use-prof-cache --cp 7.00
python3 experimental/tools/integration/run_gamma_integration.py if_float2 --branch-bb=1 --merge-bb=4 --prioritized-side 1 --out out_1_two_sided_7ns --use-prof-cache --cp 7.00
