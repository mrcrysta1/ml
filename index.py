import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from scipy import stats

# Create a models folder if it doesn't exist
if not os.path.exists('models'):
    os.makedirs('models')

# Load data
data = pd.read_csv('StudentsPerformance_datasets.csv')

# Separate features and target
X = data.drop(columns=['math score', 'reading score', 'writing score'])
y_math = data['math score']
y_reading = data['reading score']
y_writing = data['writing score']

# Extract categorical columns
categorical_columns = ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']

# Initialize OneHotEncoder
encoder = OneHotEncoder(drop='first')

# Fit OneHotEncoder on categorical data
X_encoded = pd.DataFrame(encoder.fit_transform(X[categorical_columns]).toarray(),
                         columns=encoder.get_feature_names_out(categorical_columns))

# Concatenate the encoded DataFrame with the original data
X = pd.concat([X, X_encoded], axis=1).drop(columns=categorical_columns)

# Split data into training and testing sets
X_train, X_test, y_math_train, y_math_test, y_reading_train, y_reading_test, y_writing_train, y_writing_test = \
    train_test_split(X, y_math, y_reading, y_writing, test_size=0.2, random_state=42)

# Initialize and fit StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)

# Save the scaler
scaler_path = 'models/standard_scaler.pkl'
joblib.dump(scaler, scaler_path)
print(f"StandardScaler saved at {scaler_path}")

# Save the training data
X_train.to_csv('training_data.csv', index=False)
print("Training data saved at 'training_data.csv'")

# Define models and their hyperparameters
models = {
    'Support Vector Regression': (SVR(), {
        'regressor__C': [0.1, 1, 10],
        'regressor__kernel': ['linear', 'rbf']
    }),
    'Linear Regression': (LinearRegression(), {}),
    'Gradient Boosting': (GradientBoostingRegressor(), {
        'regressor__n_estimators': [50, 100, 150],
        'regressor__learning_rate': [0.01, 0.1, 0.5]
    }),
    'Decision Tree': (DecisionTreeRegressor(), {
        'regressor__max_depth': [None, 10, 20]
    })
}

# Function to train and save models and collect accuracies
def train_and_save_model(X_train, y_train, X_test, y_test, target_name, accuracies):
    for model_name, (model, param_grid) in models.items():
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', model)
        ])
        grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error')
        grid_search.fit(X_train, y_train)
        mse = -grid_search.best_score_
        r2 = grid_search.best_estimator_.score(X_test, y_test)  # Evaluate R^2 on the test set
        print(f"{model_name} ({target_name}) - Best parameters: {grid_search.best_params_}, Best MSE: {mse}")

        # Save the trained model
        model_path = f'models/{model_name.lower().replace(" ", "_")}_{target_name}_model.pkl'
        joblib.dump(grid_search.best_estimator_, model_path)
        print(f"Trained model saved at {model_path}\n")

        # Store the accuracy (R^2 score)
        accuracies[target_name][model_name] = r2

# Initialize dictionary to hold accuracies
accuracies = {'math': {}, 'reading': {}, 'writing': {}}

# Train and save models for each target
train_and_save_model(X_train, y_math_train, X_test, y_math_test, 'math', accuracies)
train_and_save_model(X_train, y_reading_train, X_test, y_reading_test, 'reading', accuracies)
train_and_save_model(X_train, y_writing_train, X_test, y_writing_test, 'writing', accuracies)

# Save the accuracies to a JSON file
accuracies_path = 'models/model_accuracies.json'
with open(accuracies_path, 'w') as f:
    json.dump(accuracies, f)
print(f"Model accuracies saved at {accuracies_path}")

# Function to calculate confidence intervals
def calculate_confidence_intervals(y_true, y_pred, alpha=0.05):
    residuals = y_true - y_pred
    mean_residuals = np.mean(residuals)
    std_residuals = np.std(residuals)
    n = len(residuals)
    t_value = stats.t.ppf(1 - alpha/2, n - 1)
    margin_of_error = t_value * (std_residuals / np.sqrt(n))
    confidence_interval_width = 2 * margin_of_error  # Calculate the width of the confidence interval
    confidence_level = (confidence_interval_width / np.abs(mean_residuals)) * 100  # Convert to percentage
    return confidence_level

# Function to test model performance and collect metrics
def test_model_performance(target_name, y_test):
    model_performance = []
    for model_name in models.keys():
        model_path = f'models/{model_name.lower().replace(" ", "_")}_{target_name}_model.pkl'
        trained_model = joblib.load(model_path)
        
        # Predict on test set
        y_pred = trained_model.predict(X_test)

        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        confidence_level = calculate_confidence_intervals(y_test, y_pred)
        
        # Calculate accuracy
        accuracy = 100 - np.abs(y_test - y_pred).mean() / y_test.mean() * 100

        model_performance.append((model_name, mse, r2, confidence_level, accuracy))
        print(f"{model_name} ({target_name}) - MSE: {mse}, R2: {r2}, Confidence Level: {confidence_level:.2f}%, Accuracy: {accuracy:.2f}%")

    # Compare model performance
    model_performance = sorted(model_performance, key=lambda x: x[1])
    print(f"\nModel performance comparison for {target_name.capitalize()} Scores:")
    for model_name, mse, r2, confidence_level, accuracy in model_performance:
        print(f"{model_name}: MSE = {mse}, R2 = {r2}, Confidence Level = {confidence_level:.2f}%, Accuracy = {accuracy:.2f}%")

    # Visualize model performance
    performance_df = pd.DataFrame(model_performance, columns=['Model', 'MSE', 'R2', 'Confidence Level', 'Accuracy'])
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Model', y='Accuracy', data=performance_df)
    plt.title(f'Accuracy Comparison for {target_name.capitalize()} Scores')
    plt.ylabel('Accuracy (%)')
    plt.xlabel('Model')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Test performance of each model for each target and visualize
test_model_performance('math', y_math_test)
test_model_performance('reading', y_reading_test)
test_model_performance('writing', y_writing_test)
