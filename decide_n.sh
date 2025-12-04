mkdir -p decide_n
python3 experimental/tools/integration/run_specv2_integration.py single_loop --decide-n 0 --cp 7.00 > decide_n/single_loop.txt
python3 experimental/tools/integration/run_specv2_integration.py nested_loop --decide-n 0 --cp 7.00 > decide_n/nested_loop.txt
python3 experimental/tools/integration/run_specv2_integration.py fixed_log --decide-n 0 --cp 7.00 > decide_n/fixed_log.txt
python3 experimental/tools/integration/run_specv2_integration.py newton --decide-n 0 --cp 7.00 > decide_n/newton.txt
python3 experimental/tools/integration/run_specv2_integration.py subdiag_fast --decide-n 0 --cp 7.00 --transformed-code subdiag_fast_v1.c > decide_n/subdiag_fast.txt
python3 experimental/tools/integration/run_specv2_integration.py golden_ratio --decide-n 0 --cp 7.00 > decide_n/golden_ratio.txt
python3 experimental/tools/integration/run_specv2_integration.py collision_donut --decide-n 0 --cp 7.00 > decide_n/collision_donut.txt
python3 experimental/tools/integration/run_specv2_integration.py bisection --decide-n 0 --cp 7.00 --transformed-code bisection_transformed.c > decide_n/bisection.txt
python3 experimental/tools/integration/run_specv2_integration.py sparse_dataspec --decide-n 0 --cp 7.00 > decide_n/sparse_dataspec.txt
python3 experimental/tools/integration/run_specv2_integration.py sparse_dataspec_transformed --decide-n 0 --cp 7.00 --transformed-code sparse_dataspec_modified.c > decide_n/sparse_dataspec_transformed.txt

# Validate results
python3 validate_n.py
