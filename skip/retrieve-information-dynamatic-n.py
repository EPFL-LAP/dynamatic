from recopilation_aux import *
import argparse

# REFERENCES TO FILES THAT ARE NEEDED
ROOT_DIR = "../integration-test"
# This is just the name, not the full path. Used to determine if the simulation was completed
LAST_REPORT = "timing_post_pr.rpt"
LAST_REPORT_PATH = "out/vivado/timing_post_pr.rpt"

OUTPUT_NAME = "30-late.csv"

# === LIST OF REGULAR EXPRESSIONS FOR INFORMATION EXTRACTION ===
compilation_time_rpt = "out/vivado/compilation-time.txt"

cpu_user_time_pattern = r"user\s*([\d]+m[\d\.]+)s"
cpu_user_time_key = "Compilation CPU user time"

cpu_system_time_pattern = r"sys\s*([\d]+m[\d\.]+)s"
cpu_system_time_key = "Compilation CPU system time"

modelsim_rpt = "out/sim/report.txt"
latency_pattern = r"# \*\* Note: simulation done!\n#    Time: ([\d\.]+)"
latency_key = "Latency (ns)"

timing_rpt = "out/vivado/timing_post_pr.rpt"
timing_pattern = r"Data Path Delay:\s*([\d\.]+)"
timing_key = "Post route timing"

slice_pattern = r"Slice\s*\|\s*([\d]+)"
slice_key = "SLICE"

cp_rpt = "out/vivado/cp.txt"

cp_rpt = "out/vivado/cp.txt"

buffer_removed_rpt = "out/comp/removed_buffers.txt"


clock_cycle_pattern = r"Clock cycles:\s*([\d\.]+)"
clock_cycle_key = "Clock cycles"

real_cp_pattern = r"Real CP:\s*([0-9]+(?:\.[0-9]+)?)"
real_cp_key = "Real CP"

real_timing_pattern = r"Timing:\s*([0-9]+(?:\.[0-9]+)?)"
real_timing_key = "Timing"

buffer_removed_pattern = r"Buffers removed:\s*([\d]+)"
buffer_removed_key = "Buffers removed"

buffer_total_pattern = r"Total buffers:\s*([\d]+)"
buffer_total_key = "Total buffers"


utilization_rpt = "out/vivado/utilization_post_pr.rpt"

lut_pattern = r"Slice LUTs\s*\|\s*([\d]+)"
lut_key = "LUT"

ff_pattern = r"Register as Flip Flop\s*\|\s*([\d]+)"
ff_key = "FF"

dsp_pattern = r"DSPs\s*\|\s*([\d]+)"
dsp_key = "DSP"

bram_pattern = r"Block RAM Tile\s*\|\s*([\d]+)"
bram_key = "BRAM"

# uram_pattern = r"URAM:[\s]*([\d]+)"
# uram_key = "URAM"

srl_pattern = r"LUT as Shift Register\s*\|\s*([\d]+)"
srl_key = "SRL"

passed_test_pattern = r"Comparison of \[.*\] : ([a-zA-Z0-9_]+)"
passed_key = "cosim"

# === END LIST ====


def main(test_name, n, cp):
    # Use the LAST_REPORT file to identify each test file that has succesfully completed execution
    # (assumes all other necessary files are there if this one is present)

    if "memory" in test_name:
        integration_directory = "../integration-test/memory/" + test_name + "/"
    else:
        integration_directory = "../integration-test/" + test_name + "/"
    data = []
    data.append(
        extract_info_from_files_n(
            test_name,
            n, cp,
            [
                clock_cycle_pattern, 
                real_cp_pattern, 
                buffer_removed_pattern,
                buffer_total_pattern,
                slice_pattern,
                lut_pattern,
                ff_pattern,
                dsp_pattern,
                bram_pattern,
                srl_pattern,
                passed_test_pattern,
            ],
            [                   
                clock_cycle_key, 
                real_cp_key, 
                buffer_removed_key,
                buffer_total_key,
                slice_key,
                lut_key,
                ff_key,
                dsp_key,
                bram_key,
                srl_key,
                passed_key,
            ],
            [
                integration_directory + cp_rpt,
                integration_directory + cp_rpt,
                integration_directory + buffer_removed_rpt,
                integration_directory + buffer_removed_rpt,
                integration_directory + utilization_rpt,
                integration_directory + utilization_rpt,
                integration_directory + utilization_rpt,
                integration_directory + utilization_rpt,
                integration_directory + utilization_rpt,
                integration_directory + utilization_rpt,
                integration_directory + modelsim_rpt,
            ],
        )
    )

    write_to_csv(data, OUTPUT_NAME)
    print(f"Wrote a csv file with the dynamatic compilation stats to {OUTPUT_NAME}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process ModelSim report for a specific test.")
    parser.add_argument("test_name", help="The name of the test to process", type=str)
    parser.add_argument("n", help="The name of the test to process", type=int)
    parser.add_argument("cp", help="", type=float)
    args = parser.parse_args()

    main(args.test_name, args.n, args.cp)
