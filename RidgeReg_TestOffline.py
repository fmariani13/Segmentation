import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
import joblib

# Function to load or train a new Ridge model if saved model is not found
def get_ridge_model(training_data_path=None):
    try:
        # Try to load the saved model
        ridge_model = joblib.load('ridge_model.pkl')
        poly_features = joblib.load('poly_features.pkl')
        print("Loaded existing model successfully")
        return ridge_model, poly_features
    except FileNotFoundError:
        print("Saved model not found. Training a new model...")
        if training_data_path is None:
            raise ValueError("Training data path must be provided to train a new model")
        
        # Load the training data
        data_raw = pd.read_excel(training_data_path, sheet_name='Raw Data')
        
        # Extract the time and linear acceleration data
        time = data_raw['Time (s)']
        accel_x = data_raw['Linear Acceleration x (m/s^2)']
        accel_y = data_raw['Linear Acceleration y (m/s^2)']
        accel_z = data_raw['Linear Acceleration z (m/s^2)']
        
        # Apply filtering
        window_length = 101  # Must be odd and less than data length
        polyorder = 2
        
        accel_x_smooth = savgol_filter(accel_x, window_length=window_length, polyorder=polyorder)
        accel_y_smooth = savgol_filter(accel_y, window_length=window_length, polyorder=polyorder)
        accel_z_smooth = savgol_filter(accel_z, window_length=window_length, polyorder=polyorder)
        
        # Calculate additional features
        accel_mag = np.sqrt(accel_x_smooth**2 + accel_y_smooth**2 + accel_z_smooth**2)
        accel_mag_smooth = savgol_filter(accel_mag, window_length=window_length, polyorder=polyorder)
        
        # Create a progress variable (this will be our target)
        progress = np.linspace(0, 1, len(time))
        
        # Create feature matrix
        features = np.column_stack((
            accel_x_smooth, accel_y_smooth, accel_z_smooth, 
            accel_mag_smooth,
            np.cumsum(np.abs(np.diff(accel_x_smooth, prepend=accel_x_smooth[0]))),
            np.cumsum(np.abs(np.diff(accel_y_smooth, prepend=accel_y_smooth[0]))),
            np.cumsum(np.abs(np.diff(accel_z_smooth, prepend=accel_z_smooth[0]))),
            np.cumsum(np.abs(np.diff(accel_mag_smooth, prepend=accel_mag_smooth[0])))
        ))
        
        # Normalize features
        for i in range(features.shape[1]):
            features[:, i] = (features[:, i] - features[:, i].min()) / (features[:, i].max() - features[:, i].min())
        
        # Create polynomial features
        poly = PolynomialFeatures(degree=2, include_bias=True)
        features_poly = poly.fit_transform(features)
        
        # Fit Ridge regression model
        ridge = Ridge(alpha=1.0)
        ridge.fit(features_poly, progress)
        
        # Save the model
        joblib.dump(ridge, 'ridge_model.pkl')
        joblib.dump(poly, 'poly_features.pkl')
        
        print("New model trained and saved")
        return ridge, poly

# Function to process new acceleration data and predict movement progress
def predict_movement_progress(new_data_path, ridge_model, poly_features):
    # Load the new data
    new_data = pd.read_excel(new_data_path)
    
    # Extract acceleration data
    time = new_data['Time (s)']
    accel_x = new_data['Linear Acceleration x (m/s^2)']
    accel_y = new_data['Linear Acceleration y (m/s^2)']
    accel_z = new_data['Linear Acceleration z (m/s^2)']
    
    # Apply the same filtering as in training
    window_length = 101
    polyorder = 2
    
    accel_x_smooth = savgol_filter(accel_x, window_length=window_length, polyorder=polyorder)
    accel_y_smooth = savgol_filter(accel_y, window_length=window_length, polyorder=polyorder)
    accel_z_smooth = savgol_filter(accel_z, window_length=window_length, polyorder=polyorder)
    
    # Calculate additional features
    accel_mag = np.sqrt(accel_x_smooth**2 + accel_y_smooth**2 + accel_z_smooth**2)
    accel_mag_smooth = savgol_filter(accel_mag, window_length=window_length, polyorder=polyorder)
    
    # Create the same features as used in training
    features = np.column_stack((
        accel_x_smooth, accel_y_smooth, accel_z_smooth, 
        accel_mag_smooth,
        np.cumsum(np.abs(np.diff(accel_x_smooth, prepend=accel_x_smooth[0]))),
        np.cumsum(np.abs(np.diff(accel_y_smooth, prepend=accel_y_smooth[0]))),
        np.cumsum(np.abs(np.diff(accel_z_smooth, prepend=accel_z_smooth[0]))),
        np.cumsum(np.abs(np.diff(accel_mag_smooth, prepend=accel_mag_smooth[0])))
    ))
    
    # Normalize features
    for i in range(features.shape[1]):
        features[:, i] = (features[:, i] - features[:, i].min()) / (features[:, i].max() - features[:, i].min())
    
    # Transform for Ridge regression
    features_poly = poly_features.transform(features)
    
    # Predict movement progress
    predictions = ridge_model.predict(features_poly)
    
    # Apply final smoothing
    predictions_smooth = savgol_filter(predictions, window_length=min(101, len(predictions)), polyorder=2)
    
    # Force predictions to start at 0 and end at 1
    predictions_smooth = (predictions_smooth - predictions_smooth[0]) / (predictions_smooth[-1] - predictions_smooth[0])
    
    # Ensure predictions are within 0-1 range (for safety)
    predictions_smooth = np.clip(predictions_smooth, 0, 1)
    
    return time, predictions_smooth


# Main execution
if __name__ == "__main__":
    # Path to training data (only used if model needs to be trained)
    training_data_path = 'Data_ArmbBend1.0.xls'
    
    # Path to new data for prediction
    new_data_path = 'Data_ArmBend1.0.xls'  # Replace with your new data file
    
    # Get the model (either load existing or train new)
    ridge_model, poly_features = get_ridge_model(training_data_path)
    
    # Process new data and get predictions
    time, movement_progress = predict_movement_progress(new_data_path, ridge_model, poly_features)
    
    # Plot the results
    plt.figure(figsize=(12, 6))
    plt.plot(time, movement_progress * 100, 'g-', linewidth=2)
    plt.xlabel('Time (s)')
    plt.ylabel('Movement Progress (%)')
    plt.title('Arm Bending Movement Progress')
    plt.grid(True)
    plt.ylim(0, 100)
    plt.show()
    
    # Optionally save the results to a new Excel file
    results_df = pd.DataFrame({
        'Time (s)': time,
        'Movement Progress (%)': movement_progress * 100
    })
    results_df.to_excel('Movement_Progress_Results.xlsx', index=False)
    
    print("Analysis complete. Results saved to 'Movement_Progress_Results.xlsx'")
