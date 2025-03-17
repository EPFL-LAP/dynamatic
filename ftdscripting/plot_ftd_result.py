import json

import matplotlib.pyplot as plt
from scipy.stats import gmean

# Sample JSON data
json_data_slices = """
[
    {"kernel_name": "binary_search",    "slice_ratio": 0.61, "time_ratio": 0.46, "va":"", "ha":"left"},
    {"kernel_name": "fir",              "slice_ratio": 0.92, "time_ratio": 0.99, "va":"", "ha":""},
    {"kernel_name": "gcd",              "slice_ratio": 0.59, "time_ratio": 0.61, "va":"", "ha":""},
    {"kernel_name": "get_tanh",         "slice_ratio": 0.99, "time_ratio": 0.60, "va":"", "ha":""},
    {"kernel_name": "jacobi_1d_imper",  "slice_ratio": 1.03, "time_ratio": 0.34, "va":"", "ha":"left"},
    {"kernel_name": "kernel_2mm",       "slice_ratio": 1.04, "time_ratio": 0.89, "va":"", "ha":"left"},
    {"kernel_name": "kernel_3mm",       "slice_ratio": 1.09, "time_ratio": 0.73, "va":"", "ha":""},
    {"kernel_name": "matvec",           "slice_ratio": 1.18, "time_ratio": 1.11, "va":"", "ha":""},
    {"kernel_name": "sobel",            "slice_ratio": 0.63, "time_ratio": 0.48, "va":"bottom", "ha":"left"},
    {"kernel_name": "spmv",             "slice_ratio": 0.83, "time_ratio": 0.35, "va":"", "ha":""},
    {"kernel_name": "atax",             "slice_ratio": 1.00, "time_ratio": 0.61, "va":"", "ha":"left"},
    {"kernel_name": "bigc",             "slice_ratio": 1.00, "time_ratio": 0.94, "va":"bottom", "ha":"left"},
    {"kernel_name": "stencil_2d",       "slice_ratio": 0.94, "time_ratio": 0.91, "va":"", "ha":""}
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

# Set a professional style
plt.style.use("seaborn-whitegrid")
plt.rcParams.update({"font.size": 12, "font.family": "serif"})

# Create the plot
fig, ax = plt.subplots(figsize=(12, 8), dpi=800)
ax.grid(color="lightgray", linestyle="--", linewidth=0.5, alpha=0.7)

# Scatter plot with improved colors
scatter = ax.scatter(
    x_values,
    y_values,
    color="#1f77b4",  # A professional blue color
    s=100,  # Larger markers
    edgecolor="black",  # Add edge color for clarity
    linewidth=0.5,
    label="Fast Token Delivery + Straight To The Queue",
)

# Add labels with improved positioning
for i, name in enumerate(kernel_names):
    ax.text(
        x_values[i],
        y_values[i],
        name,
        fontsize=12,
        ha=has[i] if has[i] != "" else "right",
        va=vas[i] if vas[i] != "" else "top",
        bbox=dict(
            facecolor="white", alpha=0, edgecolor="none", pad=2
        ),  # Add background to text
    )

# Plot reference FPGA point
ax.scatter(
    [1.0],
    [1.0],
    color="#d62728",
    s=100,
    edgecolor="black",
    linewidth=0.5,
    label="Main Dynamic",
)
ax.axvline(x=1.0, linestyle="dashed", color="black", linewidth=0.8, alpha=0)
ax.axhline(y=1.0, linestyle="dashed", color="black", linewidth=0.8, alpha=0)

# Plot geometric mean point
ax.scatter(
    [geo_mean_x],
    [geo_mean_y],
    color="#ff7f0e",  # A professional orange color
    s=150,
    edgecolor="black",
    linewidth=0.5,
    label="Geometric Mean",
)

ax.text(
    geo_mean_x,
    geo_mean_y,
    "Geometric Mean",
    fontsize=12,
    ha="right",
    va="bottom",
    bbox=dict(facecolor="white", alpha=0, edgecolor="none", pad=2),
)

# Labels and title
obj = "LUT"
ax.set_xlabel(f"Area ({obj}), Normalized", fontsize=14)
ax.set_ylabel("Execution Time (CCs * Tck), Normalized", fontsize=14)
ax.set_title(
    f"Area ({obj}) Performance Comparison of Benchmarks With Respect To Baseline",
    fontsize=16,
    pad=20,
)

# Improve legend
ax.legend(fontsize=12, frameon=True, shadow=True, borderpad=1)

# Save as high-quality JPEG
plt.savefig(
    "ftdscripting/performance_comparison.jpg",
    format="jpeg",
    dpi=800,
    bbox_inches="tight",
)
