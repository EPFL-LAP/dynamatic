import re
import csv

def write_to_csv(input_file, output_csv, benchmark, N, target_cp, real_cp, cycles):
    patterns = {
        'Slices': r"\| *Slice *\| *(\d+)",
        'FF': r"\| *Slice Registers *\| *(\d+)",
        'DSPs': r"\| *DSPs *\| *(\d+)",
        'LUTs': r"\| *Slice LUTs *\| *(\d+)"
    }

    # Initialize a dictionary to store the extracted values
    extracted_data = {
        'Slices': None,
        'FF': None,
        'DSPs': None,
        'LUTs': None
    }

    # Open and read the input file
    with open(input_file, 'r') as file:
        content = file.read()
        
        # Search for each pattern and extract the values
        for key, pattern in patterns.items():
            match = re.search(pattern, content)
            if match:
                extracted_data[key] = match.group(1)
    
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        
        writer.writerow(['Benchmark', 'N', 'Target CP', 'Real CP', 'Cycles'] + list(extracted_data.keys()))
        
        writer.writerow([benchmark, N, target_cp, real_cp, cycles] + list(extracted_data.values()))



# input_file = 'skip/runs/run_2025-12-03_10-50-54/histogram-0/out/vivado/pr_3.00_cp_3.73/utilization_post_pr.rpt' 
# output_csv = 'extracted_data.csv'  # The path to save the output CSV