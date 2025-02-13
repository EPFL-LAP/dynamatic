import json

import matplotlib.pyplot as plt
from scipy.stats import gmean

json_data_slices = """
[
    {"kernel_name": "binary_searcy", "slice_ratio": 0.58, "time_ratio": 0.46, "va":"", "ha":"left"},
    {"kernel_name": "fir", "slice_ratio": 0.91, "time_ratio": 0.99, "va":"", "ha":""},
    {"kernel_name": "gcd", "slice_ratio": 0.46, "time_ratio": 0.61, "va":"", "ha":""},
    {"kernel_name": "get_tanh", "slice_ratio": 0.97, "time_ratio": 0.60, "va":"", "ha":""},
    {"kernel_name": "jacobi_1d_imper", "slice_ratio": 1.03, "time_ratio": 0.34, "va":"", "ha":"left"},
    {"kernel_name": "kernel_2mm", "slice_ratio": 1.04, "time_ratio": 0.89, "va":"", "ha":"left"},
    {"kernel_name": "kernel_3mm", "slice_ratio": 1.10, "time_ratio": 0.73, "va":"", "ha":""},
    {"kernel_name": "matvec", "slice_ratio": 1.12, "time_ratio": 1.11, "va":"", "ha":""},
    {"kernel_name": "sobel", "slice_ratio": 0.54, "time_ratio": 0.48, "va":"bottom", "ha":"left"},
    {"kernel_name": "spmv", "slice_ratio": 0.65, "time_ratio": 0.35, "va":"", "ha":""},
    {"kernel_name": "atax", "slice_ratio": 1.01, "time_ratio": 0.61, "va":"", "ha":"left"},
    {"kernel_name": "bigc", "slice_ratio": 1.01, "time_ratio": 0.92, "va":"bottom", "ha":"left"},
    {"kernel_name": "stencil_2d", "slice_ratio": 0.96, "time_ratio": 0.9, "va":"", "ha":""}
]
"""

# Load JSON data
data = json.loads(json_data_slices)

# Extract values
kernel_names = [item["kernel_name"] for item in data]
x_values = [item["slice_ratio"] for item in data]
y_values = [item["time_ratio"] for item in data]
vas = [item["va"] for item in data]
has = [item["ha"] for item in data]

# Calculate geometric mean of x and y values
geo_mean_x = gmean(x_values)
geo_mean_y = gmean(y_values)

# Create the plot
plt.figure(figsize=(12, 6), dpi=400)
plt.grid(color="lightgray", linestyle="--", linewidth=0.5, alpha=0.7)
plt.scatter(
    x_values,
    y_values,
    color="red",
    label="Fast Token Delviery + Straight To The Queue",
)

# Add labels
for i, name in enumerate(kernel_names):
    plt.text(
        x_values[i],
        y_values[i],
        name,
        fontsize=14,
        ha=has[i] if has[i] != "" else "right",
        va=vas[i] if vas[i] != "" else "top",
    )

# Plot reference FPGA point
plt.scatter([1.0], [1.0], color="blue", label="Main Dynamtic")
plt.axvline(x=1.0, linestyle="dashed", color="black", linewidth=0.8)
plt.axhline(y=1.0, linestyle="dashed", color="black", linewidth=0.8)

# Plot geometric mean point
plt.scatter(
    [geo_mean_x],
    [geo_mean_y],
    color="yellow",
    edgecolor="black",
)

plt.text(
    geo_mean_x,
    geo_mean_y,
    "Geometric Mean",
    fontsize=14,
)


# Labels and title
obj = "SLICES"
text_pic = f"Area ({obj}), Normalized"
plt.xlabel(text_pic, fontsize=15)
plt.ylabel("Execution time (CCs * Tck), Normalized", fontsize=15)
plt.title(
    f"Area ({obj}) Performance Comparison of Benchmarks With Respect To Baseline",
    fontsize=18,
)
plt.legend(fontsize=12)

# Save as high-quality JPEG
plt.savefig("ftdscripting/performance_comparison.jpg", format="jpeg", dpi=600)
