import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
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
    "XGBoost": XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42),
    "Linear Regression": LinearRegression()
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

# Print the selected wavelengths for each parameter
print("\nSelected Wavelengths for Each Parameter:")
for param, group in top_wavelengths.groupby('Parameter'):
    print(f"\nParameter: {param}")
    print(f"Wavelengths: {group['Wavelength'].tolist()}")


# Calculate the best moisture level for each parameter based on the highest R2 score
best_moisture_suggestion = {}

# Iterate through the accuracy results and find the best moisture level for each parameter
for param in accuracy_results_df['Parameter'].unique():
    best_r2 = -np.inf
    best_moisture = None
    
    for moisture in [0, 25, 50]:
        r2_score_for_moisture = accuracy_results_df.loc[
            (accuracy_results_df['Parameter'] == param) & 
            (accuracy_results_df['Moisture Level'] == moisture), 
            'R2 Score'
        ].max()
        
        if r2_score_for_moisture > best_r2:
            best_r2 = r2_score_for_moisture
            best_moisture = moisture
    
    best_moisture_suggestion[param] = best_moisture

# Output the suggested best moisture level for each parameter
print("\nSuggested Best Moisture Level for Each Parameter:")
for param, moisture in best_moisture_suggestion.items():
    print(f"Parameter: {param} -> Best Moisture Level: {moisture}ml")
