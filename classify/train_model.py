import numpy as np
import pandas as pd
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression

# Load the dataset
data_path = os.path.join(os.path.dirname(__file__), 'data', 'spam.csv')
data = pd.read_csv(data_path, encoding='latin-1') 
data = data.iloc[:, :2]
data.columns = ['label', 'text']

# Data Cleaning and EDA
data.drop_duplicates(inplace=True)  # Remove duplicate entries
data.dropna(inplace=True)  # Remove rows with missing values

# Convert labels to numerical values ("ham" to 0, "spam" to 1)
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# Split the dataset into features (X) and labels (y)
X = data['text']
y = data['label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create a CountVectorizer to convert text data into numerical features
vectorizer = CountVectorizer(stop_words='english', lowercase=True)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Define the parameter grid for GridSearchCV for Logistic Regression
param_grid_lr = {
    'C': [100, 200],  
    # Regularization strength; smaller values specify stronger regularization

    'penalty': ['l1', 'l2'],  
    # Regularization type; 'l1' is Lasso (L1) regularization, 'l2' is Ridge (L2) regularization
    
    'solver': ['liblinear', 'saga']  
    # Algorithm to use in the optimization problem; 
    # 'liblinear' is good for small datasets, 'saga' is suitable for larger datasets
}

# Initialize the Logistic Regression classifier
log_reg = LogisticRegression(random_state=42, max_iter=10000)


# Initialize GridSearchCV with cross-validation for Logistic Regression
grid_search_lr = GridSearchCV(estimator=log_reg, param_grid=param_grid_lr, cv=3, n_jobs=-1, verbose=2)


# Train the model using GridSearchCV for Logistic Regression
grid_search_lr.fit(X_train_vec, y_train)

# Print the best parameters for Logistic Regression
print("Best parameters found for Logistic Regression: ", grid_search_lr.best_params_)



# Use the best estimator to make predictions for Logistic Regression
best_log_reg = grid_search_lr.best_estimator_
y_pred_log_reg = best_log_reg.predict(X_test_vec)

# Calculate accuracy for Logistic Regression
accuracy_log_reg = accuracy_score(y_test, y_pred_log_reg)
print("Logistic Regression Accuracy:", accuracy_log_reg)

# Print classification report for Logistic Regression
print("Logistic Regression Classification Report:")
print(classification_report(y_test, y_pred_log_reg))

# Generate and print the confusion matrix for Logistic Regression
conf_matrix_log_reg = confusion_matrix(y_test, y_pred_log_reg)
print("Logistic Regression Confusion Matrix:")
print(conf_matrix_log_reg)

# Format the confusion matrix for better readability
conf_matrix_log_reg_df = pd.DataFrame(conf_matrix_log_reg, index=['Actual Not Spam', 'Actual Spam'], columns=['Predicted Not Spam', 'Predicted Spam'])
print(conf_matrix_log_reg_df)

# # Define the parameter grid for GridSearchCV
# param_grid = {
#     'n_estimators': [100,300,500],
#     'learning_rate': [0, 0.05, 0.1],
#     'max_depth': [3,4,5],
#     'min_child_weight': [1, 3, 5],
#     'gamma': [0, 0.05, 0.1, 0.2],
#     'subsample': [0.8],
#     'colsample_bytree': [0.8],
#     'objective':['binary:logistic'],                  
#     'random_state': [42]
# }

# # Initialize the XGBoost classifier
# xgb_classifier = XGBClassifier(objective='binary:logistic', random_state=42)

# # Initialize GridSearchCV with cross-validation
# grid_search = GridSearchCV(estimator=xgb_classifier, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)

# # Train the model using GridSearchCV
# grid_search.fit(X_train_vec, y_train)

# # Print the best parameters
# print("Best parameters found: ", grid_search.best_params_)

# # Use the best estimator to make predictions
# best_xgb_classifier = grid_search.best_estimator_
# y_pred = best_xgb_classifier.predict(X_test_vec)

# # Calculate accuracy
# accuracy = accuracy_score(y_test, y_pred)
# print("Accuracy:", accuracy)

# # Print classification report
# print(classification_report(y_test, y_pred))

# # Generate and print the confusion matrix as a table
# conf_matrix = confusion_matrix(y_test, y_pred)
# print("Confusion Matrix:")
# print(conf_matrix)

# # Optionally, format the confusion matrix for better readability
# conf_matrix_df = pd.DataFrame(conf_matrix, index=['Actual Not Spam', 'Actual Spam'], columns=['Predicted Not Spam', 'Predicted Spam'])
# print(conf_matrix_df)


# Save the model and vectorizer
model_path = os.path.join(os.path.dirname(__file__), 'spam_classifier_model.pkl')
vectorizer_path = os.path.join(os.path.dirname(__file__), 'vectorizer.pkl')

joblib.dump(best_log_reg, model_path)
joblib.dump(vectorizer, vectorizer_path)
