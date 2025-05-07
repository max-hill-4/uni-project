import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline

# Data preparation
biomarkers = ['TAC mmol/L', 'ADA U/L', 'ADA2 U/L', '%ADA2', 'GLU mg/Dl', 'PHOS mg/Dl', 
              'CA mg/Dl', 'CHOL mg/Dl', 'TRI mg/Dl', 'HDL mg/dL', 'LDL-C mg/Dl', 'CPK U/L']

# Accuracy data for each coherence type
alpha_data = [
    0.7828947368421053, 0.5686274509803921, 0.881578947368421, 0.5053763440860215,
    0.7450980392156863, 0.3815789473684211, 0.34408602150537637, 0.4342105263157895,
    0.3881578947368421, 0.43010752688172044, 0.42105263157894735, 0.5686274509803921
]

delta_data = [
    0.4605263157894737, 0.4117647058823529, 0.8421052631578947, 0.5098039215686274,
    0.6078431372549019, 0.45161290322580644, 0.3870967741935484, 0.3618421052631579,
    0.43137254901960786, 0.6862745098039216, 0.42105263157894735, 0.5686274509803921
]

delta_beta_data = [
    0.6118421052631579, 0.6118421052631579, 0.8157894736842105, 0.5294117647058824,
    0.6666666666666666, 0.3815789473684211, 0.45098039215686275, 0.3548387096774194,
    0.43137254901960786, 0.7254901960784313, 0.42105263157894735, 0.43137254901960786
]

alpha_theta_data = [
    0.6447368421052632, 0.7039473684210527, 0.8289473684210527, 0.5053763440860215,
    0.6666666666666666, 0.3815789473684211, 0.39215686274509803, 0.40131578947368424,
    0.4117647058823529, 0.6862745098039216, 0.42105263157894735, 0.21710526315789475
]

# Prepare x-axis for interpolation
x = np.arange(len(biomarkers))
x_smooth = np.linspace(0, len(biomarkers)-1, 300)  # More points for smoother curve

# Spline interpolation for each coherence type
spline_alpha = make_interp_spline(x, alpha_data, k=3)
spline_delta = make_interp_spline(x, delta_data, k=3)
spline_delta_beta = make_interp_spline(x, delta_beta_data, k=3)
spline_alpha_theta = make_interp_spline(x, alpha_theta_data, k=3)

# Interpolated data
alpha_smooth = spline_alpha(x_smooth)
delta_smooth = spline_delta(x_smooth)
delta_beta_smooth = spline_delta_beta(x_smooth)
alpha_theta_smooth = spline_alpha_theta(x_smooth)

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(x_smooth, alpha_smooth, 'b-', label='coh alpha')
plt.plot(x_smooth, delta_smooth, 'r-', label='coh delta')
plt.plot(x_smooth, delta_beta_smooth, 'g-', label='coh delta + beta')
plt.plot(x_smooth, alpha_theta_smooth, 'm-', label='coh alpha + theta')

# Add original data points
plt.scatter(x, alpha_data, color='blue', marker='o', s=50)
plt.scatter(x, delta_data, color='red', marker='s', s=50)
plt.scatter(x, delta_beta_data, color='green', marker='^', s=50)
plt.scatter(x, alpha_theta_data, color='magenta', marker='d', s=50)

# Customize the plot
plt.title('Accuracy Across Hormone in N1 Sleep')
plt.xlabel('Biomarkers')
plt.ylabel('Accuracy')
plt.grid(True)
plt.legend()
plt.xticks(x, biomarkers, rotation=45)

# Save the plot
plt.tight_layout()
plt.savefig('hormone_coherence_plot.png')