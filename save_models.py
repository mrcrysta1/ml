import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

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
encoder.fit(X[categorical_columns])

# Save the encoder
joblib.dump(encoder, 'models/label_encoder.pkl')

print("Label encoder saved successfully!")

# One-hot encode categorical variables
data_encoded = encoder.transform(X[categorical_columns])

# Extract column names after one-hot encoding
columns = encoder.get_feature_names_out(categorical_columns)

# Create DataFrame from encoded data and column names
data_encoded = pd.DataFrame(data_encoded.toarray(), columns=columns)

# Concatenate the encoded DataFrame with the original data
X = pd.concat([X, data_encoded], axis=1).drop(columns=categorical_columns)

# Split data into training and testing sets
X_train, X_test, y_math_train, y_math_test, y_reading_train, y_reading_test, y_writing_train, y_writing_test = train_test_split(X, y_math, y_reading, y_writing, test_size=0.2, random_state=42)

# Define additional models and their hyperparameters
models = {
    'Support Vector Regression': (SVR(), {
        'regressor__kernel': ['linear', 'rbf'],
        'regressor__C': [0.1, 1, 10]
    }),
    'Linear Regression': (LinearRegression(), {}),
    'Decision Tree': (DecisionTreeRegressor(random_state=42), {
        'regressor__max_depth': [None, 10, 20]
    }),
    'Gradient Boosting': (GradientBoostingRegressor(random_state=42), {
        'regressor__n_estimators': [50, 100, 150],
        'regressor__learning_rate': [0.01, 0.1, 0.5]
    })
}

# Train and tune models
for model_name, (model, param_grid) in models.items():
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', model)
    ])
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error')
    if 'math' in model_name.lower():
        y_train = y_math_train
    elif 'reading' in model_name.lower():
        y_train = y_reading_train
    else:
        y_train = y_writing_train
    grid_search.fit(X_train, y_train)
    print(f"Best parameters for {model_name}: {grid_search.best_params_}")
    print(f"Best MSE for {model_name}: {-grid_search.best_score_}\n")

    # Save the trained model
    model_path = f'models/{model_name.replace(" ", "_").lower()}_model.pkl'
    joblib.dump(grid_search.best_estimator_, model_path)
    print(f"Trained model saved at {model_path}\n")
