import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator

# Assuming time is the x-axis and x3 is the y-axis data for the green line

TT = int(5e2)          #discrete time samples
tt_hor = range(TT)

time = tt_hor  # Replace with your actual time data
x3 =     # Replace with your actual x3 data for the green line

# Find the indices where the blue line is constant
# You would replace the following arrays with the actual constant parts you have identified
constant_sections_indices = np.array([...])  # Indices of the start of the constant sections
constant_values = np.array([...])            # The constant values for each section

# Create the PCHIP Interpolator
pchip_interpolator = PchipInterpolator(time, x3)

# Generate new, smoother time values (denser for plotting)
time_new = np.linspace(time.min(), time.max(), 500)

# Compute the smoothed x3 values
x3_smooth = pchip_interpolator(time_new)

# Overwrite the smoothed values with the original constant values in the constant sections
for idx, const_value in zip(constant_sections_indices, constant_values):
    x3_smooth[time_new == idx] = const_value

# Plotting the original and smoothed trajectories
plt.figure(figsize=(10, 6))
plt.plot(time, x3, 'g--', label='Original traj_ref[0]')
plt.plot(time_new, x3_smooth, 'b-', label='Smoothed traj_ref[0]')
plt.title('Smoothed Trajectory with Constant Sections')
plt.xlabel('Time')
plt.ylabel('x3')
plt.legend()
plt.show()
