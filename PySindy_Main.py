# Import necessary libraries
import pandas as pd
import numpy as np
import pysindy as ps
import matplotlib.pyplot as plt

# HYPERPARAMETERS
# Define the threshold for coefficient selection -> Higher threshold for more sparsity
threshold = 0.1
# Define the degree of the polynomial library -> Lower degree to avoid overfitting
degree = 1


# Load the data from the Excel file
data_raw = pd.read_excel('Data_Prova.xls', sheet_name='Raw Data')

# Extract the time and linear acceleration data
time = data_raw['Time (s)']
accel_x = data_raw['Linear Acceleration x (m/s^2)']
accel_y = data_raw['Linear Acceleration y (m/s^2)']
accel_z = data_raw['Linear Acceleration z (m/s^2)']

# Combine the acceleration data into a single array for modeling
acceleration_data = np.vstack((accel_x, accel_y, accel_z)).T

# Remove any NaN or infinite values
acceleration_data = acceleration_data[np.isfinite(acceleration_data).all(axis=1)]

# Define the time step (average difference between consecutive time points)
dt = time.diff().mean()

# Create a PySINDy model with more conservative settings
model = ps.SINDy(
    optimizer=ps.STLSQ(threshold=threshold),  
    feature_library=ps.PolynomialLibrary(degree=degree)  
)

# Fit the model to the data
model.fit(acceleration_data, t=dt)

# Print the discovered equations
print("Discovered system dynamics:")
model.print()

# Try a shorter simulation to avoid numerical issues
t_sim = np.arange(0, min(4.0, time.max()), dt)
x0 = acceleration_data[0]  # Initial condition

try:
    # Use a try-except block to catch any simulation errors
    x_sim = model.simulate(x0, t_sim)
    
    # Plot comparison between original data and simulation
    plt.figure(figsize=(12, 8))
    
    # Plot X acceleration
    plt.subplot(3, 1, 1)
    plt.plot(time[:len(t_sim)], accel_x[:len(t_sim)], 'r-', label='Original X')
    plt.plot(t_sim, x_sim[:, 0], 'r--', label='Model X')
    plt.legend()
    plt.ylabel('X Accel (m/s²)')
    plt.grid(True, linestyle='-', linewidth=0.5, alpha=0.7)

    
    # Plot Y acceleration
    plt.subplot(3, 1, 2)
    plt.plot(time[:len(t_sim)], accel_y[:len(t_sim)], 'g-', label='Original Y')
    plt.plot(t_sim, x_sim[:, 1], 'g--', label='Model Y')
    plt.legend()
    plt.ylabel('Y Accel (m/s²)')
    plt.grid(True, linestyle='-', linewidth=0.5, alpha=0.7)

    
    # Plot Z acceleration
    plt.subplot(3, 1, 3)
    plt.plot(time[:len(t_sim)], accel_z[:len(t_sim)], 'b-', label='Original Z')
    plt.plot(t_sim, x_sim[:, 2], 'b--', label='Model Z')
    plt.legend()
    plt.xlabel('Time (s)')
    plt.ylabel('Z Accel (m/s²)')
    plt.grid(True, linestyle='-', linewidth=0.5, alpha=0.7)

    
    plt.tight_layout()
    plt.show()
    
except Exception as e:
    print(f"Simulation failed with error: {e}")
    print("Try adjusting model parameters or using a different integration method")
