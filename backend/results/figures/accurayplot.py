import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline

# Data preparation
biomarkers = ['TAC mmol/L', 'ADA U/L', 'ADA2 U/L', '%ADA2', 'GLU mg/Dl', 'PHOS mg/Dl', 
              'CA mg/Dl', 'CHOL mg/Dl', 'TRI mg/Dl', 'HDL mg/dL', 'LDL-C mg/Dl', 'CPK U/L']

# R² data for each BDC
r2_data = [
    0.01650065742433071, 0.17535275220870972, 0.08211833238601685, 0.04927951097488403,
    0.014501635916531086, 0.020344793796539307, 0.019057273864746094, 0.019057273864746094,
    0.1247519850730896, 0.0803578794002533, 0.031250715255737305, 0.04927951097488403
]

# Prepare x-axis for interpolation
x = np.arange(len(biomarkers))
x_smooth = np.linspace(0, len(biomarkers)-1, 300)  # More points for smoother curve

# Spline interpolation
spline_r2 = make_interp_spline(x, r2_data, k=3)
r2_smooth = spline_r2(x_smooth)

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(x_smooth, r2_smooth, 'b-', label='R²')
plt.scatter(x, r2_data, color='blue', marker='o', s=50)

# Customize the plot
plt.title('R² Across Hormones')
plt.xlabel('Biomarkers')
plt.ylabel('R²')
plt.grid(True)
plt.legend()
plt.xticks(x, biomarkers, rotation=45)

# Save the plot
plt.tight_layout()
plt.savefig('hormone_coherence_plot.png')