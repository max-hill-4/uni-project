import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline

# Data preparation
biomarkers = ['TAC mmol/L', 'ADA U/L', 'ADA2 U/L', '%ADA2', 'GLU mg/Dl', 'PHOS mg/Dl', 
              'CA mg/Dl', 'CHOL mg/Dl', 'TRI mg/Dl', 'HDL mg/dL', 'LDL-C mg/Dl', 'CPK U/L']

# Maximum accuracy data and corresponding sleep stages
max_data = [
    0.7828947368421053, 0.7039473684210527, 0.881578947368421, 0.5888888888888889,
    0.7666666666666667, 0.5, 0.580952380952381, 0.6190476190476191,
    0.6333333333333333, 0.7254901960784313, 0.42105263157894735, 0.5686274509803921
]

sleep_stages = ['N1', 'N1', 'N1', 'N3', 'REM', 'REM', 'N3', 'REM', 
                'REM', 'N1', 'N1', 'N1']

# Bright color mapping for sleep stages
color_map = {
    'N1': 'green',
    'N2': 'red',
    'N3': 'lime',
    'REM': 'hotpink'
}

# Prepare x-axis for interpolation
x = np.arange(len(biomarkers))
x_smooth = np.linspace(0, len(biomarkers)-1, 300)  # More points for smoother curve

# Spline interpolation for the entire dataset
spline_max = make_interp_spline(x, max_data, k=3)
max_smooth = spline_max(x_smooth)

# Clip smoothed values to stay above the minimum of nearest points
max_smooth_clipped = np.copy(max_smooth)
for i, x_val in enumerate(x_smooth):
    # Find the two nearest original points
    idx_left = int(np.floor(x_val))
    idx_right = min(idx_left + 1, len(max_data) - 1)
    # Get the minimum y-value of the nearest points
    min_y = min(max_data[idx_left], max_data[idx_right])
    # Clip the smoothed value to be at least the minimum
    max_smooth_clipped[i] = max(max_smooth[i], min_y)

# Plotting
plt.figure(figsize=(12, 6))

# Plot colored line segments
for i in range(len(biomarkers) - 1):
    # Define segment range in smoothed coordinates
    start_idx = int(i * 300 / (len(biomarkers) - 1))
    end_idx = int((i + 1) * 300 / (len(biomarkers) - 1))
    
    # Extract segment of smoothed curve
    x_segment = x_smooth[start_idx:end_idx + 1]
    y_segment = max_smooth_clipped[start_idx:end_idx + 1]
    
    # Plot segment with color based on starting point's sleep stage
    plt.plot(x_segment, y_segment, color=color_map[sleep_stages[i]], linewidth=2)

# Plot colored scatter points and add labels
for i, (acc, stage) in enumerate(zip(max_data, sleep_stages)):
    plt.scatter(i, acc, color=color_map[stage], marker='o', s=50)
    plt.text(i, acc + 0.02, stage, ha='center', va='bottom', fontsize=8)

# Create legend for sleep stages
legend_elements = [
    plt.Line2D([0], [0], color='cyan', label='N1', linewidth=2),
    plt.Line2D([0], [0], color='red', label='N2', linewidth=2),
    plt.Line2D([0], [0], color='lime', label='N3', linewidth=2),
    plt.Line2D([0], [0], color='hotpink', label='REM', linewidth=2)
]
plt.legend(handles=legend_elements, title='Sleep Stage')

# Customize the plot
plt.title('Maximum Accuracy Across Hormone Coherence Types')
plt.xlabel('Biomarkers')
plt.ylabel('Accuracy')
plt.grid(True)
plt.xticks(x, biomarkers, rotation=45)

# Save the plot
plt.tight_layout()
plt.savefig('hormone_coherence_plot.png')