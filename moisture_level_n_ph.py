import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score

# Load dataset
file_path = "C:/Users/vasav/OneDrive/Desktop/CSI_2025/soildataset (1).xlsx"
df = pd.read_excel(file_path)

df = df.dropna()

df['Moisture_Level'] = df['Records'].str.extract(r'_(\d+)ml-').astype(float)
df_filtered = df[df['Moisture_Level'].isin([0, 25, 50])]

# Define soil parameters and wavelength columns
soil_params = ['Ph', 'Nitro (mg/10 g)', 'Posh Nitro (mg/10 g)', 'Pota Nitro (mg/10 g)']
soil_params = [param for param in soil_params if df_filtered[param].sum() != 0]  # Exclude parameters with all zero values
wavelength_cols = [col for col in df.columns if isinstance(col, (int, float))]

feature_importance_matrix = pd.DataFrame(columns=['Parameter', 'Wavelength', 'Importance'])

# Find important wavelengths for each soil parameter
for param in soil_params:
    X_wavelengths = df_filtered[wavelength_cols]
    y_param = df_filtered[param]
    
    gb_regressor = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42)
    gb_regressor.fit(X_wavelengths, y_param)
    
    feature_importances = gb_regressor.feature_importances_
    importance_df = pd.DataFrame({'Wavelength': wavelength_cols, 'Importance': feature_importances})
    importance_df['Parameter'] = param
    feature_importance_matrix = pd.concat([feature_importance_matrix, importance_df], ignore_index=True)

# Select significant wavelengths dynamically
def select_best_wavelengths(group):
    threshold = group['Importance'].mean() + group['Importance'].std()  # Select wavelengths above mean + std dev
    return group[group['Importance'] >= threshold]

top_wavelengths = feature_importance_matrix.groupby('Parameter', group_keys=False).apply(select_best_wavelengths)

top_wavelengths = top_wavelengths.reset_index(drop=True)  # Prevent index conflicts

print("The moisture levels are:", df_filtered['Moisture_Level'].value_counts())

# Train models using selected wavelengths and evaluate accuracy per moisture level
accuracy_results = []
models = {
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=200, max_depth=4, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42)
}

for param, group in top_wavelengths.groupby('Parameter'):
    selected_wavelengths = group['Wavelength'].tolist()
    for moisture in [0, 25, 50]:
        df_moisture = df_filtered[df_filtered['Moisture_Level'] == moisture]
        X_selected = df_moisture[selected_wavelengths]
        y_param = df_moisture[param]
        
        if len(df_moisture) < 2:
            continue  # Skip if insufficient data
        
        X_train, X_test, y_train, y_test = train_test_split(X_selected, y_param, test_size=0.1, random_state=42)
        
        for model_name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = r2_score(y_test, y_pred)  # Keep r2_score as it is
            accuracy_results.append({'Parameter': param, 'Moisture Level': moisture, 'Model': model_name, 'Used Wavelengths': selected_wavelengths, 'R2 Score': accuracy})

accuracy_results_df = pd.DataFrame(accuracy_results)

# Print results in table format
print("\nModel Accuracy Based on Selected Wavelengths and Moisture Levels:")
print(accuracy_results_df.pivot(index=['Parameter', 'Model'], columns='Moisture Level', values='R2 Score').to_string())

# Plot predictions for visualization
for param, group in top_wavelengths.groupby('Parameter'):
    if df_filtered[param].sum() == 0:
        continue  # Skip plotting if the parameter has all zero values
    
    selected_wavelengths = group['Wavelength'].tolist()
    for moisture in [0, 25, 50]:
        df_moisture = df_filtered[df_filtered['Moisture_Level'] == moisture]
        X_selected = df_moisture[selected_wavelengths]
        y_param = df_moisture[param]
        
        if len(df_moisture) < 2:
            continue  # Skip if insufficient data
        
        X_train, X_test, y_train, y_test = train_test_split(X_selected, y_param, test_size=0.1, random_state=42)
        best_model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42)
        best_model.fit(X_train, y_train)
        y_pred = best_model.predict(X_test)
        
        plt.figure(figsize=(8, 5))
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.title(f"Prediction for {param} at {moisture}ml Moisture Level")
        plt.grid(True)
        plt.show()
