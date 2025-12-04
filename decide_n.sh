mkdir -p decide_n
echo "single_loop"
python3 experimental/tools/integration/run_specv2_integration.py single_loop --decide-n 0 --cp 7.00 > decide_n/single_loop.txt
echo "nested_loop"
python3 experimental/tools/integration/run_specv2_integration.py nested_loop --decide-n 0 --cp 7.00 > decide_n/nested_loop.txt
echo "fixed_log"
python3 experimental/tools/integration/run_specv2_integration.py fixed_log --decide-n 0 --cp 7.00 > decide_n/fixed_log.txt
echo "newton"
python3 experimental/tools/integration/run_specv2_integration.py newton --decide-n 0 --cp 7.00 > decide_n/newton.txt
echo "subdiag_fast"
python3 experimental/tools/integration/run_specv2_integration.py subdiag_fast --decide-n 0 --cp 7.00 --transformed-code subdiag_fast_v1.c > decide_n/subdiag_fast.txt
echo "golden_ratio"
python3 experimental/tools/integration/run_specv2_integration.py golden_ratio --decide-n 0 --cp 7.00 > decide_n/golden_ratio.txt
echo "collision_donut"
python3 experimental/tools/integration/run_specv2_integration.py collision_donut --decide-n 0 --cp 7.00 > decide_n/collision_donut.txt
echo "bisection"
python3 experimental/tools/integration/run_specv2_integration.py bisection --decide-n 0 --cp 7.00 --transformed-code bisection_transformed.c > decide_n/bisection.txt
echo "sparse_dataspec"
python3 experimental/tools/integration/run_specv2_integration.py sparse_dataspec --decide-n 0 --cp 7.00 > decide_n/sparse_dataspec.txt
echo "sparse_dataspec_transformed"
python3 experimental/tools/integration/run_specv2_integration.py sparse_dataspec_transformed --decide-n 0 --cp 7.00 --transformed-code sparse_dataspec_modified.c > decide_n/sparse_dataspec_transformed.txt

# Validate results
python3 validate_n.py
