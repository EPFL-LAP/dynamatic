import os
real_n = {
    "single_loop": 4,
    "nested_loop": 4,
    "fixed_log": 3,
    "newton": 1,
    "subdiag_fast": 13,
    "golden_ratio": 1,
    "collision_donut": 4,
    "bisection": 1,
    "sparse_dataspec": 0,  # In the paper n=8 is used
    "sparse_dataspec_transformed": 4
}

# Get all files in the `decide_n` directory
decide_n_dir = "decide_n"
err = False
files = os.listdir(decide_n_dir)
if len(files) != len(real_n):
    print(
        f"Mismatch in number of files: expected {len(real_n)}, found {len(files)}"
    )
    err = True

for filename in files:
    if not filename.endswith(".txt"):
        print("Unknown file in decide_n directory:", filename)
        err = True
        continue

    kernel_name = filename.split(".txt")[0]
    if kernel_name not in real_n:
        print("Unknown kernel in decide_n directory:", kernel_name)
        err = True
        continue

    expected_n = real_n[kernel_name]
    with open(os.path.join(decide_n_dir, filename), "r") as f:
        last_line = f.readlines()[-1]
        reported_n = int(last_line.split(":")[-1].strip())

        if reported_n != expected_n:
            print(
                f"Mismatch in {kernel_name}: expected n={expected_n}, reported n={reported_n}"
            )
            err = True

if err:
    print("Validation result: FAILED")
    exit(1)
else:
    print("Validation result: PASSED")
    exit(0)
