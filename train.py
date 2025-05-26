#!/usr/bin/env python
# coding: utf-8

# **1 - Data Understanding and Preprocessing**

# Importing necessary libraries
import pandas as pd  # Data manipulation and analysis
import numpy as np  # Numerical operations and array handling
import seaborn as sns  # Statistical data visualization
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving plots
import matplotlib.pyplot as plt  # Plotting library
from sklearn.preprocessing import MinMaxScaler  # Feature scaling
from sklearn.model_selection import train_test_split, KFold, cross_validate, GridSearchCV  # Model selection and validation
from sklearn.linear_model import LinearRegression, Ridge, Lasso  # Linear regression models
from sklearn.metrics import mean_squared_error, r2_score  # Evaluation metrics
from mlxtend.frequent_patterns import apriori, association_rules  # Association rule mining
from sklearn.tree import DecisionTreeRegressor  # Decision tree regression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor  # Ensemble methods
from sklearn.svm import SVR  # Support Vector Regression
import xgboost as xgb  # Gradient boosting framework
import lightgbm as lgb  # Light Gradient Boosting Machine
from catboost import CatBoostRegressor  # CatBoost gradient boosting
import time  # For timing operations
from sklearn.metrics import mean_absolute_error, explained_variance_score, max_error  # Additional evaluation metrics
from matplotlib.colors import Normalize  # Color normalization for plots
import collections  # Container datatypes
import os  # Operating system interface
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

# Set the path to your desired directory
# os.chdir("C:/Users/bedoo/Desktop/Bitirme/Project")

# Print to confirm
print("Current working directory:", os.getcwd())

# **1.1 Loading the Dataset**

# Load dataset
filepath = "C:\\Users\\Gokberk\\Desktop\\dataset.csv"
df = pd.read_csv(filepath)

# **1.2 Inspecting Dataset**

# Inspect the dataset
print("Dataset Info:")
print(df.info())
print("Dataset Statistics:")
print(df.describe())
print("\nFirst 5 Rows:")
print(df.head())

# **1.3 - Dropping the irrelevant columns** 

# Drop irrelevant columns
df = df.drop(columns=["ID","DOB","year","fireday","lon","lat"]) 
# ID: unique identifier
# DOB: Day of fire in the year
# year: year of the fire
# fireday: the day of the fire in the month
# lon: longitude of the fire
# lat: latitude of the fire

print("Columns after dropping irrelevant ones:", df.columns.tolist())

# **1.4 - Dropping the missing values** 

# Drop missing values
# Since there is a smaller value of null values (~1k << ~75k)
# we can erase them from the dataset
df = df.dropna(how='any', axis=0)
print("\nNull values after dropping:")
print(df.isnull().sum())

# **1.5 - Normalizing columns using Min-Max Scaling** 

# First select features based on correlation
target_column = 'sprdistm'
correlation_with_target = df.corr()[target_column].drop(target_column)
selected_features = correlation_with_target[correlation_with_target.abs() >= 0.1].index.tolist()
print("\nSelected features based on correlation threshold (|corr| >= 0.1):\n", selected_features)

# Drop redundant features (high inter-feature correlation)
def drop_highly_correlated_features(df, threshold=0.8):
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    return df.drop(columns=to_drop), to_drop

# Apply to selected features only
X_temp = df[selected_features]
X_temp, dropped_cols = drop_highly_correlated_features(X_temp)
print("\nDropped due to high inter-feature correlation:", dropped_cols)

# Get the final feature set that will be used for training
final_features = X_temp.columns.tolist()
print("\nFinal features used for training:", final_features)

# Now normalize only the final features
scaler = MinMaxScaler()
df[final_features] = scaler.fit_transform(df[final_features])

# **1.6 - Detecting and Handling Outliers** 

# Visualization: Boxplots BEFORE outlier treatment
plt.figure(figsize=(16, 8))
sns.boxplot(data=df[final_features])
plt.title("Boxplot Before Outlier Capping (Normalized Data)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("boxplot_before_capping.png", dpi=300)
print('Saved plot: boxplot_before_capping.png')

# Function to cap outliers using IQR method (corrected version)
def cap_outliers_iqr(df, columns):
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Count and display outliers
        num_outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
        print(f"{col}: {num_outliers} outliers capped")

        # Apply capping explicitly
        df.loc[df[col] < lower_bound, col] = lower_bound
        df.loc[df[col] > upper_bound, col] = upper_bound

    return df

# Apply capping only to final features
df = cap_outliers_iqr(df, final_features)

# Visualization: Boxplots AFTER outlier treatment
plt.figure(figsize=(16, 8))
sns.boxplot(data=df[final_features])
plt.title("Boxplot After Outlier Capping (Normalized Data)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("boxplot_after_capping.png", dpi=300)
print('Saved plot: boxplot_after_capping.png')

# **1.7 - Correlation Matrix** 

# Plot correlation matrix to explore feature relationships
plt.figure(figsize=(14, 10))
corr_matrix = df[final_features + [target_column]].corr()
sns.heatmap(corr_matrix, annot=False, fmt=".2f", cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Matrix")
plt.tight_layout()
plt.savefig("correlation_matrix.png", dpi=300)
print('Saved plot: correlation_matrix.png')

# Final feature set
X = df[final_features]
y = df[target_column]

print("Number of entries (rows) in X:", X.shape[0])
print("Number of features (columns) in X:", X.shape[1])

# **2 - Model Training and Evaluation**

# Define models and their parameter grids
models = {
    # 'Linear Regression': {
    #     'model': LinearRegression(),
    #     'params': {}
    # },
    # 'Polynomial Regression': {
    #     'model': Pipeline([
    #         ('poly', PolynomialFeatures()),
    #         ('linear', LinearRegression())
    #     ]),
    #     'params': {
    #         'poly__degree': [2, 3],
    #         'poly__include_bias': [False]
    #     }
    # },
    # 'Ridge': {
    #     'model': Ridge(),
    #     'params': {
    #         'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]
    #     }
    # },
    # 'Lasso': {
    #     'model': Lasso(),
    #     'params': {
    #         'alpha': [0.1, 1.0, 10.0, 100.0]  
    #     }
    # },
    # 'SVR': {
    #     'model': SVR(),
    #     'params': {
    #         'C': [0.1, 1, 10, 100],  
    #         'gamma': ['scale', 'auto', 0.1, 0.01, 0.001],  
    #         'kernel': ['rbf', 'linear','poly']
    #     }
    # },
    # 'Decision Tree': {
    #     'model': DecisionTreeRegressor(),
    #     'params': {
    #         'max_depth': [None, 5, 10, 15],
    #         'min_samples_split': [2, 5, 10],
    #         'min_samples_leaf': [1, 2, 4]
    #     }
    # },
    # 'Gradient Boosting': {
    #     'model': GradientBoostingRegressor(),
    #     'params': {
    #         'n_estimators': [100],
    #         'learning_rate': [0.1],
    #         'max_depth': [3],
    #         'min_samples_split': [2]
    #     }
    # },
    # 'XGBoost': {
    #     'model': xgb.XGBRegressor(n_jobs=-1),
    #     'params': {
    #         'n_estimators': [100],
    #         'max_depth': [3],
    #         'learning_rate': [0.1],
    #         'subsample': [0.8]
    #     }
    # },
    # 'LightGBM': {
    #     'model': lgb.LGBMRegressor(n_jobs=-1, verbose=-1),
    #     'params': {
    #         'n_estimators': [100],
    #         'max_depth': [3],
    #         'learning_rate': [0.1],
    #         'num_leaves': [31],
    #         'subsample': [0.8]
    #     }
    # },
    'Random Forest': {
        'model': RandomForestRegressor(n_jobs=-1),
        'params': {
            'n_estimators': [100],
            'max_depth': [10],
            'min_samples_split': [2],
            'min_samples_leaf': [1]
        }
    }
}

# Initialize results storage for cross-validation
cv_results = {
    'model_name': [],
    'mean_rmse': [],
    'std_rmse': [],
    'mean_r2': [],
    'std_r2': [],
    'mean_mae': [],
    'std_mae': [],
    'mean_explained_variance': [],
    'std_explained_variance': [],
    'mean_max_error': [],
    'std_max_error': [],
    'best_params': [],
    'total_training_time': []  
}

# Initialize results storage for test set evaluation
test_results = {
    'model_name': [],
    'rmse': [],
    'r2': [],
    'mae': [],
    'explained_variance': [],
    'max_error': [],
    'test_time': []  
}

# Perform cross-validation on training set
outer_cv = KFold(n_splits=10, shuffle=True, random_state=42)
inner_cv = KFold(n_splits=5, shuffle=True, random_state=42)

# Store best models for each type
best_models = {}

for model_name, model_info in models.items():
    print(f"\nTraining {model_name}...")
    model_start_time = time.time()
    
    # Initialize metrics storage for this model
    rmse_scores = []
    r2_scores = []
    mae_scores = []
    explained_variance_scores = []
    max_error_scores = []
    best_params_list = []
    
    # Cross-validation loop on training data
    for fold, (train_idx, val_idx) in enumerate(outer_cv.split(X), 1):
        fold_start_time = time.time()
        print(f"\nProcessing fold {fold}/10 for {model_name}...")
        
        X_train_cv, X_val_cv = X.iloc[train_idx], X.iloc[val_idx]
        y_train_cv, y_val_cv = y.iloc[train_idx], y.iloc[val_idx]
        
        # Inner cross-validation for hyperparameter tuning
        grid_search = GridSearchCV(
            estimator=model_info['model'],
            param_grid=model_info['params'],
            cv=inner_cv,
            scoring='neg_mean_squared_error',
            n_jobs=-1  # Enable parallel processing for grid search
        )
        
        # Fit the model
        grid_search.fit(X_train_cv, y_train_cv)
        best_model = grid_search.best_estimator_
        best_params_list.append(grid_search.best_params_)
        
        # Make predictions on validation set
        y_pred = best_model.predict(X_val_cv)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_val_cv, y_pred))
        r2 = r2_score(y_val_cv, y_pred)
        mae = mean_absolute_error(y_val_cv, y_pred)
        explained_variance = explained_variance_score(y_val_cv, y_pred)
        max_err = max_error(y_val_cv, y_pred)
        
        # Store metrics
        rmse_scores.append(rmse)
        r2_scores.append(r2)
        mae_scores.append(mae)
        explained_variance_scores.append(explained_variance)
        max_error_scores.append(max_err)
        
        fold_time = time.time() - fold_start_time
        print(f"Fold {fold} completed in {fold_time:.2f} seconds")
    
    # Calculate mean and std of metrics
    cv_results['model_name'].append(model_name)
    cv_results['mean_rmse'].append(np.mean(rmse_scores))
    cv_results['std_rmse'].append(np.std(rmse_scores))
    cv_results['mean_r2'].append(np.mean(r2_scores))
    cv_results['std_r2'].append(np.std(r2_scores))
    cv_results['mean_mae'].append(np.mean(mae_scores))
    cv_results['std_mae'].append(np.std(mae_scores))
    cv_results['mean_explained_variance'].append(np.mean(explained_variance_scores))
    cv_results['std_explained_variance'].append(np.std(explained_variance_scores))
    cv_results['mean_max_error'].append(np.mean(max_error_scores))
    cv_results['std_max_error'].append(np.std(max_error_scores))
    cv_results['best_params'].append(best_params_list)
    
    # Train final model on entire training set with best parameters
    best_params = max(best_params_list, key=best_params_list.count)
    final_model = model_info['model'].set_params(**best_params)
    final_model.fit(X, y)
    best_models[model_name] = final_model
    
    # Evaluate on test set
    test_start_time = time.time()
    y_pred_test = final_model.predict(X)
    test_time = time.time() - test_start_time
    
    # Calculate and store test metrics
    test_results['model_name'].append(model_name)
    test_results['rmse'].append(np.sqrt(mean_squared_error(y, y_pred_test)))
    test_results['r2'].append(r2_score(y, y_pred_test))
    test_results['mae'].append(mean_absolute_error(y, y_pred_test))
    test_results['explained_variance'].append(explained_variance_score(y, y_pred_test))
    test_results['max_error'].append(max_error(y, y_pred_test))
    test_results['test_time'].append(test_time)
    
    # Store total training time
    total_time = time.time() - model_start_time
    cv_results['total_training_time'].append(total_time)
    print(f"\nTotal training time for {model_name}: {total_time:.2f} seconds")
    print(f"Test set evaluation time: {test_time:.2f} seconds")

# Convert results to DataFrames
cv_results_df = pd.DataFrame(cv_results)
test_results_df = pd.DataFrame(test_results)

# # Save results to CSV
# cv_results_df.to_csv('cross_validation_results.csv', index=False)
# test_results_df.to_csv('test_set_results.csv', index=False)
# print("\nResults saved to 'cross_validation_results.csv' and 'test_set_results.csv'")

# # **3 - Visualization of Results**

# # Create a figure with multiple subplots for cross-validation results
# plt.figure(figsize=(20, 15))

# # 1. RMSE Comparison
# plt.subplot(2, 2, 1)
# plt.bar(cv_results_df['model_name'], cv_results_df['mean_rmse'], yerr=cv_results_df['std_rmse'])
# plt.title('RMSE Comparison Across Models (Cross-Validation)')
# plt.xticks(rotation=45)
# plt.ylabel('RMSE')

# # 2. R² Score Comparison
# plt.subplot(2, 2, 2)
# plt.bar(cv_results_df['model_name'], cv_results_df['mean_r2'], yerr=cv_results_df['std_r2'])
# plt.title('R² Score Comparison Across Models (Cross-Validation)')
# plt.xticks(rotation=45)
# plt.ylabel('R² Score')

# # 3. MAE Comparison
# plt.subplot(2, 2, 3)
# plt.bar(cv_results_df['model_name'], cv_results_df['mean_mae'], yerr=cv_results_df['std_mae'])
# plt.title('MAE Comparison Across Models (Cross-Validation)')
# plt.xticks(rotation=45)
# plt.ylabel('MAE')

# # 4. Explained Variance Comparison
# plt.subplot(2, 2, 4)
# plt.bar(cv_results_df['model_name'], cv_results_df['mean_explained_variance'], 
#         yerr=cv_results_df['std_explained_variance'])
# plt.title('Explained Variance Comparison Across Models (Cross-Validation)')
# plt.xticks(rotation=45)
# plt.ylabel('Explained Variance')

# plt.tight_layout()
# plt.savefig('cv_model_comparison_metrics.png', dpi=300, bbox_inches='tight')
# print('Saved plot: cv_model_comparison_metrics.png')

# # Create a figure for test set results
# plt.figure(figsize=(20, 15))

# # 1. RMSE Comparison
# plt.subplot(2, 2, 1)
# plt.bar(test_results_df['model_name'], test_results_df['rmse'])
# plt.title('RMSE Comparison Across Models (Test Set)')
# plt.xticks(rotation=45)
# plt.ylabel('RMSE')

# # 2. R² Score Comparison
# plt.subplot(2, 2, 2)
# plt.bar(test_results_df['model_name'], test_results_df['r2'])
# plt.title('R² Score Comparison Across Models (Test Set)')
# plt.xticks(rotation=45)
# plt.ylabel('R² Score')

# # 3. MAE Comparison
# plt.subplot(2, 2, 3)
# plt.bar(test_results_df['model_name'], test_results_df['mae'])
# plt.title('MAE Comparison Across Models (Test Set)')
# plt.xticks(rotation=45)
# plt.ylabel('MAE')

# # 4. Explained Variance Comparison
# plt.subplot(2, 2, 4)
# plt.bar(test_results_df['model_name'], test_results_df['explained_variance'])
# plt.title('Explained Variance Comparison Across Models (Test Set)')
# plt.xticks(rotation=45)
# plt.ylabel('Explained Variance')

# plt.tight_layout()
# plt.savefig('test_set_model_comparison_metrics.png', dpi=300, bbox_inches='tight')
# print('Saved plot: test_set_model_comparison_metrics.png')

# # Create a heatmap of best parameters
# plt.figure(figsize=(15, 10))

# # Convert parameters to a format suitable for heatmap
# param_data = []
# for params in cv_results_df['best_params']:
#     # Get the most common parameters
#     best_params = max(params, key=params.count)
#     # Convert numeric values to float, keep strings as is
#     param_row = {}
#     for key, value in best_params.items():
#         if isinstance(value, (int, float)):
#             param_row[key] = float(value)
#         else:
#             param_row[key] = 0  # Use 0 for non-numeric values
#     param_data.append(param_row)

# # Create DataFrame for heatmap
# param_heatmap = pd.DataFrame(param_data, index=cv_results_df['model_name'])

# # Create the heatmap
# sns.heatmap(param_heatmap, annot=True, cmap='YlOrRd', fmt='.2f')
# plt.title('Best Parameters for Each Model (Numeric Values Only)')
# plt.tight_layout()
# plt.savefig('best_parameters_heatmap.png', dpi=300, bbox_inches='tight')
# print('Saved plot: best_parameters_heatmap.png')

# # Create a separate text file for all parameters (including non-numeric)
# with open('best_parameters.txt', 'w') as f:
#     f.write("Best Parameters for Each Model:\n\n")
#     for model_name, params in zip(cv_results_df['model_name'], cv_results_df['best_params']):
#         best_params = max(params, key=params.count)
#         f.write(f"{model_name}:\n")
#         for key, value in best_params.items():
#             f.write(f"  {key}: {value}\n")
#         f.write("\n")
# print('Saved detailed parameters to best_parameters.txt')

# # Print final results
# print("\nCross-Validation Results:")
# print(cv_results_df.to_string(index=False))
# print("\nTest Set Results:")
# print(test_results_df.to_string(index=False))

# # === Predicted vs True Value Scatter Plots for Each Model ===
# for i, model_name in enumerate(test_results_df['model_name']):
#     # Get the corresponding model
#     model = best_models[model_name]
#     # Predict on test set
#     y_pred = model.predict(X)
#     # Create scatter plot
#     plt.figure(figsize=(8, 6))
#     plt.scatter(y, y_pred, alpha=0.6, color='dodgerblue')
#     plt.xlabel('True Values', fontsize=12, style='italic')
#     plt.ylabel('Predicted Values', fontsize=12, style='italic')
#     plt.title(f'Predicted vs True Values: {model_name}', fontsize=14)
#     # Plot 1:1 line
#     min_val = min(y.min(), y_pred.min())
#     max_val = max(y.max(), y_pred.max())
#     plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='1:1 Line')
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(f'pred_vs_true_{model_name.replace(" ", "_").lower()}.png', dpi=300)
#     plt.close()
#     print(f'Saved plot: pred_vs_true_{model_name.replace(" ", "_").lower()}.png')

# **4 - Select Best Model and Prepare for Deployment**

# Select best model based on test set RMSE
best_model_name = test_results_df.loc[test_results_df['rmse'].idxmin(), 'model_name']
print(f"\nBest performing model: {best_model_name}")
print(f"Test RMSE: {test_results_df.loc[test_results_df['rmse'].idxmin(), 'rmse']:.4f}")
print(f"Test R²: {test_results_df.loc[test_results_df['rmse'].idxmin(), 'r2']:.4f}")

# Get the best parameters for the selected model
best_model_params = max(cv_results_df.loc[cv_results_df['model_name'] == best_model_name, 'best_params'].iloc[0], 
                       key=cv_results_df.loc[cv_results_df['model_name'] == best_model_name, 'best_params'].iloc[0].count)

print("\nBest parameters for the selected model:")
for param, value in best_model_params.items():
    print(f"{param}: {value}")

# Train the best model on the full dataset
print("\nTraining best model on the full dataset...")
full_X = pd.concat([X, X])
full_y = pd.concat([y, y])

# Initialize the best model with its parameters
best_model = models[best_model_name]['model'].set_params(**best_model_params)

# Train on full dataset
start_time = time.time()
best_model.fit(full_X, full_y)
training_time = time.time() - start_time
print(f"Training completed in {training_time:.2f} seconds")

# Save the model and scaler for deployment
import joblib

# Create a deployment directory if it doesn't exist
deployment_dir = "deployment"
if not os.path.exists(deployment_dir):
    os.makedirs(deployment_dir)

# Save the model
model_path = os.path.join(deployment_dir, "best_model.joblib")
joblib.dump(best_model, model_path)

# Save the scaler
scaler_path = os.path.join(deployment_dir, "scaler.joblib")
joblib.dump(scaler, scaler_path)

# Save the feature names - use the final features that were actually used in training
feature_names_path = os.path.join(deployment_dir, "feature_names.joblib")
joblib.dump(final_features, feature_names_path)

print("\nSaved model files:")
print(f"- Model: {model_path}")
print(f"- Scaler: {scaler_path}")
print(f"- Feature names: {feature_names_path}")
print("\nFeatures saved:", final_features)

# %% 
# # **5 - Add Explainable AI to Best Model**

# # Install required packages for explainability
# import shap
# import matplotlib.pyplot as plt
# import json
# import numpy as np

# print("\nGenerating model explanations using SHAP values...")

# # Create a SHAP explainer
# if hasattr(best_model, 'predict_proba'):
#     explainer = shap.TreeExplainer(best_model)
# else:
#     # For non-tree models, use KernelExplainer with a background dataset
#     background_data = shap.kmeans(full_X, 10)  # Use k-means to create background dataset
#     explainer = shap.KernelExplainer(best_model.predict, background_data)

# # Calculate SHAP values for a sample of the data
# sample_size = min(1000, len(full_X))  # Use up to 1000 samples
# sample_indices = np.random.choice(len(full_X), sample_size, replace=False)
# sample_X = full_X.iloc[sample_indices]
# shap_values = explainer.shap_values(sample_X)

# # Create and save SHAP summary plot
# plt.figure(figsize=(10, 6))
# shap.summary_plot(shap_values, sample_X, plot_type="bar", show=False)
# plt.title("Feature Importance (SHAP Values)")
# plt.tight_layout()
# plt.savefig(os.path.join(deployment_dir, "shap_summary_plot.png"), dpi=300, bbox_inches='tight')
# plt.close()

# # Create and save SHAP dependence plots for top features
# top_features = np.abs(shap_values).mean(0).argsort()[-5:]  # Top 5 features
# for feature_idx in top_features:
#     feature_name = sample_X.columns[feature_idx]
#     plt.figure(figsize=(10, 6))
#     shap.dependence_plot(feature_idx, shap_values, sample_X, show=False)
#     plt.title(f"SHAP Dependence Plot for {feature_name}")
#     plt.tight_layout()
#     plt.savefig(os.path.join(deployment_dir, f"shap_dependence_{feature_name}.png"), dpi=300, bbox_inches='tight')
#     plt.close()

# # Save the explainer state instead of the object itself
# explainer_state = {
#     'expected_value': float(explainer.expected_value),  # Convert to float for JSON serialization
#     'feature_names': list(sample_X.columns),
#     'background_data': background_data.data.tolist() if not hasattr(best_model, 'predict_proba') else None,  # Convert to list for JSON serialization
#     'model_type': 'tree' if hasattr(best_model, 'predict_proba') else 'kernel'
# }

# # Save the explainer state
# explainer_path = os.path.join(deployment_dir, "shap_explainer_state.json")
# with open(explainer_path, 'w') as f:
#     json.dump(explainer_state, f)

# # Update the example code to include explainability
# example_code = '''
# import joblib
# import pandas as pd
# import numpy as np
# import shap
# import matplotlib.pyplot as plt
# import json

# def load_model():
#     # Load the model and scaler
#     model = joblib.load('deployment/best_model.joblib')
#     scaler = joblib.load('deployment/scaler.joblib')
#     feature_names = joblib.load('deployment/feature_names.joblib')
    
#     # Load the explainer state and recreate the explainer
#     with open('deployment/shap_explainer_state.json', 'r') as f:
#         explainer_state = json.load(f)
    
#     if explainer_state['model_type'] == 'tree':
#         explainer = shap.TreeExplainer(model)
#     else:
#         background_data = np.array(explainer_state['background_data'])
#         explainer = shap.KernelExplainer(model.predict, background_data)
    
#     # Set the expected value
#     explainer.expected_value = explainer_state['expected_value']
    
#     return model, scaler, feature_names, explainer

# def predict(input_data):
#     # Load model components
#     model, scaler, feature_names, _ = load_model()
    
#     # Ensure input data has the correct features
#     input_df = pd.DataFrame([input_data])
#     input_df = input_df[feature_names]
    
#     # Scale the input data
#     scaled_input = scaler.transform(input_df)
    
#     # Make prediction
#     prediction = model.predict(scaled_input)
    
#     return prediction[0]

# def explain_prediction(input_data, plot_type="force"):
#     """
#     Explain the model's prediction for the given input data.
    
#     Parameters:
#     -----------
#     input_data : dict
#         Dictionary containing feature names and their values
#     plot_type : str
#         Type of explanation plot to generate:
#         - "force": Force plot showing how each feature contributes to the prediction
#         - "waterfall": Waterfall plot showing the prediction breakdown
#         - "bar": Bar plot of feature importance
    
#     Returns:
#     --------
#     matplotlib.figure.Figure
#         The generated explanation plot
#     """
#     # Load model components
#     model, scaler, feature_names, explainer = load_model()
    
#     # Prepare input data
#     input_df = pd.DataFrame([input_data])
#     input_df = input_df[feature_names]
#     scaled_input = scaler.transform(input_df)
    
#     # Calculate SHAP values
#     shap_values = explainer.shap_values(scaled_input)
    
#     # Create the appropriate plot
#     plt.figure(figsize=(10, 6))
    
#     if plot_type == "force":
#         shap.force_plot(explainer.expected_value, shap_values, input_df, matplotlib=True, show=False)
#         plt.title("Force Plot - Feature Contributions to Prediction")
    
#     elif plot_type == "waterfall":
#         shap.waterfall_plot(explainer.expected_value, shap_values, input_df, show=False)
#         plt.title("Waterfall Plot - Prediction Breakdown")
    
#     elif plot_type == "bar":
#         shap.summary_plot(shap_values, input_df, plot_type="bar", show=False)
#         plt.title("Feature Importance for Prediction")
    
#     plt.tight_layout()
#     return plt.gcf()

# # Example usage:
# # input_data = {
# #     'feature1': value1,
# #     'feature2': value2,
# #     # ... add all required features
# # }
# # prediction = predict(input_data)
# # 
# # # Generate different types of explanations
# # force_plot = explain_prediction(input_data, plot_type="force")
# # waterfall_plot = explain_prediction(input_data, plot_type="waterfall")
# # bar_plot = explain_prediction(input_data, plot_type="bar")
# '''

# # Update the example code file
# with open(os.path.join(deployment_dir, "model_usage_example.py"), 'w') as f:
#     f.write(example_code)

# # Update requirements.txt to include SHAP
# requirements = '''
# numpy
# pandas
# scikit-learn
# joblib
# shap
# matplotlib
# '''

# with open(os.path.join(deployment_dir, "requirements.txt"), 'w') as f:
#     f.write(requirements)

# # Update README with explainability information
# readme_content = f'''
# # Model Deployment

# This directory contains the trained model and necessary files for deployment.

# ## Model Information
# - Best Model: {best_model_name}
# - Test RMSE: {test_results_df.loc[test_results_df['rmse'].idxmin(), 'rmse']:.4f}
# - Test R²: {test_results_df.loc[test_results_df['rmse'].idxmin(), 'r2']:.4f}

# ## Files
# 1. best_model.joblib - The trained model
# 2. scaler.joblib - The scaler used for feature normalization
# 3. feature_names.joblib - The list of features used by the model
# 4. shap_explainer_state.json - The SHAP explainer state for model interpretability
# 5. model_usage_example.py - Example code for using the model and generating explanations
# 6. requirements.txt - Required Python packages
# 7. shap_summary_plot.png - Overall feature importance plot
# 8. shap_dependence_*.png - Dependence plots for top features

# ## Usage
# 1. Install the required packages:
#    ```
#    pip install -r requirements.txt
#    ```

# 2. Use the model as shown in model_usage_example.py

# ## Model Parameters
# {chr(10).join([f"- {param}: {value}" for param, value in best_model_params.items()])}

# ## Model Explainability
# The model includes SHAP-based explainability, which provides:
# - Feature importance analysis
# - Individual prediction explanations
# - Feature dependence analysis

# Three types of explanation plots are available:
# 1. Force Plot: Shows how each feature contributes to the prediction
# 2. Waterfall Plot: Shows the prediction breakdown
# 3. Bar Plot: Shows feature importance for the prediction

# See model_usage_example.py for details on generating these explanations.
# '''

# with open(os.path.join(deployment_dir, "README.md"), 'w') as f:
#     f.write(readme_content)

# print("\nAdded explainability features to the deployment package:")
# print("1. SHAP explainer state saved as shap_explainer_state.json")
# print("2. SHAP summary plot saved as shap_summary_plot.png")
# print("3. SHAP dependence plots saved for top features")
# print("4. Updated model_usage_example.py with explanation functions")
# print("5. Updated requirements.txt with SHAP dependency")
# print("6. Updated README.md with explainability information")
