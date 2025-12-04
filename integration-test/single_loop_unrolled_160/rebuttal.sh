# prof-cache is not used
python3 experimental/tools/integration/run_specv2_large_integration.py single_loop_unrolled_160 --min-buffering --n=4 --resolver --use-prof-cache --out out_unroll_160_eager --factor 160
python3 experimental/tools/integration/run_specv2_large_integration.py single_loop_unrolled_160 --min-buffering --baseline --use-prof-cache --out out_unroll_160_baseline --factor 160
rm integration-test/single_loop_unrolled_160/specv2_*