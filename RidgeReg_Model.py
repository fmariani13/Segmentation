import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pysindy as ps
from scipy.signal import savgol_filter
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
import joblib
import os

# Turn on interactive mode
plt.ion()

#HYPERPARAMETERS:
alpha = 1.0  # Regularization strength for Ridge regression

# Load the data from the Excel file
data_raw = pd.read_excel('Data_ArmBend1.0.xls', sheet_name='Raw Data')

# Extract the time and linear acceleration data
time = data_raw['Time (s)']
accel_x = data_raw['Linear Acceleration x (m/s^2)']
accel_y = data_raw['Linear Acceleration y (m/s^2)']
accel_z = data_raw['Linear Acceleration z (m/s^2)']

# Apply stronger Savitzky-Golay filter to smooth the acceleration data
window_length = min(101, len(time) - (len(time) % 2) - 1)  # Ensure window length is valid
polyorder = 2  # Lower polynomial order for smoother results

accel_x_smooth = savgol_filter(accel_x, window_length=window_length, polyorder=polyorder)
accel_y_smooth = savgol_filter(accel_y, window_length=window_length, polyorder=polyorder)
accel_z_smooth = savgol_filter(accel_z, window_length=window_length, polyorder=polyorder)

# Plot to show the filtering effect
fig1 = plt.figure(figsize=(15, 10))

# X acceleration
plt.subplot(3, 1, 1)
plt.plot(time, accel_x, 'r-', alpha=0.4, label='Raw X')
plt.plot(time, accel_x_smooth, 'b-', linewidth=2, label='Filtered X')
plt.ylabel('X Accel (m/s²)')
plt.legend()
plt.grid(True)
plt.title('Raw vs Filtered Acceleration Data')

# Y acceleration
plt.subplot(3, 1, 2)
plt.plot(time, accel_y, 'r-', alpha=0.4, label='Raw Y')
plt.plot(time, accel_y_smooth, 'b-', linewidth=2, label='Filtered Y')
plt.ylabel('Y Accel (m/s²)')
plt.legend()
plt.grid(True)

# Z acceleration
plt.subplot(3, 1, 3)
plt.plot(time, accel_z, 'r-', alpha=0.4, label='Raw Z')
plt.plot(time, accel_z_smooth, 'b-', linewidth=2, label='Filtered Z')
plt.xlabel('Time (s)')
plt.ylabel('Z Accel (m/s²)')
plt.legend()
plt.grid(True)

plt.tight_layout()
fig1.savefig('filtered_data.png')  # Save the figure to file
fig1.canvas.draw()  # Draw the figure

# Calculate additional features
accel_mag = np.sqrt(accel_x_smooth**2 + accel_y_smooth**2 + accel_z_smooth**2)
accel_mag_smooth = savgol_filter(accel_mag, window_length=window_length, polyorder=polyorder)

# Create a progress variable (this will be our target)
progress = np.linspace(0, 1, len(time))

# Create feature matrix with engineered features
features = np.column_stack((
    accel_x_smooth, accel_y_smooth, accel_z_smooth, 
    accel_mag_smooth,
    np.cumsum(np.abs(np.diff(accel_x_smooth, prepend=accel_x_smooth[0]))),
    np.cumsum(np.abs(np.diff(accel_y_smooth, prepend=accel_y_smooth[0]))),
    np.cumsum(np.abs(np.diff(accel_z_smooth, prepend=accel_z_smooth[0]))),
    np.cumsum(np.abs(np.diff(accel_mag_smooth, prepend=accel_mag_smooth[0])))
))

# Normalize features
feature_mins = []
feature_maxs = []
for i in range(features.shape[1]):
    feature_min = features[:, i].min()
    feature_max = features[:, i].max()
    feature_mins.append(feature_min)
    feature_maxs.append(feature_max)
    features[:, i] = (features[:, i] - feature_min) / (feature_max - feature_min)

# Create polynomial features
poly = PolynomialFeatures(degree=2, include_bias=True)
features_poly = poly.fit_transform(features)

# Fit Ridge regression model
ridge = Ridge(alpha=alpha)
ridge.fit(features_poly, progress)

# Test the model by predicting progress from features
predicted_progress_ridge = ridge.predict(features_poly)

# Apply final smoothing to the predictions
smooth_window = min(101, len(predicted_progress_ridge) - (len(predicted_progress_ridge) % 2) - 1)
predicted_progress_ridge_smooth = savgol_filter(predicted_progress_ridge, window_length=smooth_window, polyorder=2)

# Add normalization to ensure 0 to 1 range:
predicted_progress_ridge_smooth = (predicted_progress_ridge_smooth - predicted_progress_ridge_smooth[0]) / (predicted_progress_ridge_smooth[-1] - predicted_progress_ridge_smooth[0])

# Plot the results
fig2 = plt.figure(figsize=(12, 8))
plt.plot(time, progress, 'b-', linewidth=2, label='Actual Progress (0-1)')
plt.plot(time, predicted_progress_ridge_smooth, 'g-', linewidth=2, label='Ridge Regression (Smoothed)')
plt.xlabel('Time (s)')
plt.ylabel('Movement Progress (0-1)')
plt.title('Arm Bending Movement Progress')
plt.grid(True)
plt.legend()
fig2.savefig('progress_prediction.png')  # Save the figure to file
fig2.canvas.draw()  # Draw the figure

# Save the model and all necessary components to the script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))  # Get the directory of the current script

# Define file paths
ridge_model_path = os.path.join(script_dir, 'ridge_model.pkl')
poly_features_path = os.path.join(script_dir, 'poly_features.pkl')
normalization_params_path = os.path.join(script_dir, 'normalization_params.pkl')
filter_params_path = os.path.join(script_dir, 'filter_params.pkl')

# Delete previous files if they exist
for file_path in [ridge_model_path, poly_features_path, normalization_params_path, filter_params_path]:
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
            print(f"Deleted previous file: {file_path}")
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")

# Save new files
joblib.dump(ridge, ridge_model_path)
joblib.dump(poly, poly_features_path)
joblib.dump((feature_mins, feature_maxs), normalization_params_path)
joblib.dump((window_length, polyorder), filter_params_path)

print("Model and parameters saved successfully in:", script_dir)
print("Script execution completed.")

# Keep figures open without blocking
plt.ioff()  # Turn off interactive mode
plt.draw()  # Update all figures
plt.pause(0.001)  # Small pause to allow GUI events

# Optional: Add this to keep the script running until manually terminated
input("Press Enter to close all figures and exit...")
plt.close('all')
