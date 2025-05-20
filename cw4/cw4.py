import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_curve, auc
import matplotlib.pyplot as plt
from custom_random_forest import CustomRandomForest

# Load and prepare the dataset
data = pd.read_csv("cw4/lab-4-dataset.csv").dropna()  # Load dataset and drop missing values
categorical_columns = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope', 'AgeGroup', 
                       'RestingBP_Category', 'Cholesterol_Category', 'MaxHR_Category', 'Oldpeak_Category']
data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)  # One-hot encode categorical variables

# Split features and target variable
X = data.drop(columns=['HeartDisease'])
y = data['HeartDisease']

# Split data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Define range for n_estimators (number of trees)
n_estimators_range = [10, 50, 100, 150, 200]

# -------------------------------
# Scikit-learn Random Forest
# -------------------------------
best_sklearn_params = None  # To store the best number of trees
best_sklearn_auc = 0  # To store the best validation AUC

# Tune n_estimators for scikit-learn Random Forest
for n_estimators in n_estimators_range:
    # Train Random Forest with the current number of trees
    clf_sklearn = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    clf_sklearn.fit(X_train, y_train)
    
    # Evaluate on the validation set
    y_pred_proba_val_sklearn = clf_sklearn.predict_proba(X_val)[:, 1]
    fpr_val_sklearn, tpr_val_sklearn, _ = roc_curve(y_val, y_pred_proba_val_sklearn)
    roc_auc_val_sklearn = auc(fpr_val_sklearn, tpr_val_sklearn)
    
    # Update the best parameters if the current AUC is better
    if roc_auc_val_sklearn > best_sklearn_auc:
        best_sklearn_auc = roc_auc_val_sklearn
        best_sklearn_params = n_estimators

# Print the best number of trees and validation AUC
print(f"\nBest Scikit-learn Random Forest Parameters:")
print(f"n_estimators: {best_sklearn_params}")
print(f"Best Validation AUC (Scikit-learn): {best_sklearn_auc:.2f}")

# Train the final scikit-learn Random Forest with the best number of trees
clf_sklearn = RandomForestClassifier(n_estimators=best_sklearn_params, random_state=42)
clf_sklearn.fit(X_train, y_train)

# Evaluate on the validation set
y_pred_val_sklearn = clf_sklearn.predict(X_val)
y_pred_proba_val_sklearn = clf_sklearn.predict_proba(X_val)[:, 1]
accuracy_val_sklearn = accuracy_score(y_val, y_pred_val_sklearn)
precision_val_sklearn = precision_score(y_val, y_pred_val_sklearn)
recall_val_sklearn = recall_score(y_val, y_pred_val_sklearn)
roc_auc_val_sklearn = auc(*roc_curve(y_val, y_pred_proba_val_sklearn)[:2])

# Print validation metrics
print("\nScikit-learn Random Forest (Validation):")
print(f"Validation Accuracy: {accuracy_val_sklearn:.2f}")
print(f"Validation Precision: {precision_val_sklearn:.2f}")
print(f"Validation Recall: {recall_val_sklearn:.2f}")
print(f"Validation AUC: {roc_auc_val_sklearn:.2f}")

# Evaluate on the test set
y_pred_test_sklearn = clf_sklearn.predict(X_test)
y_pred_proba_test_sklearn = clf_sklearn.predict_proba(X_test)[:, 1]
accuracy_test_sklearn = accuracy_score(y_test, y_pred_test_sklearn)
precision_test_sklearn = precision_score(y_test, y_pred_test_sklearn)
recall_test_sklearn = recall_score(y_test, y_pred_test_sklearn)
roc_auc_test_sklearn = auc(*roc_curve(y_test, y_pred_proba_test_sklearn)[:2])

# Print test metrics
print("\nScikit-learn Random Forest (Test):")
print(f"Test Accuracy: {accuracy_test_sklearn:.2f}")
print(f"Test Precision: {precision_test_sklearn:.2f}")
print(f"Test Recall: {recall_test_sklearn:.2f}")
print(f"Test AUC: {roc_auc_test_sklearn:.2f}")

# -------------------------------
# Custom Random Forest
# -------------------------------
best_custom_params = None  # To store the best number of trees
best_custom_auc = 0  # To store the best validation AUC

# Tune n_estimators for Custom Random Forest
for n_estimators in n_estimators_range:
    # Train Custom Random Forest with the current number of trees
    clf_custom = CustomRandomForest(n_estimators=n_estimators, random_state=42)
    clf_custom.fit(X_train.values, y_train.values)
    
    # Evaluate on the validation set
    y_pred_proba_val_custom = clf_custom.predict_proba(X_val.values)[:, 1]
    fpr_val_custom, tpr_val_custom, _ = roc_curve(y_val, y_pred_proba_val_custom)
    roc_auc_val_custom = auc(fpr_val_custom, tpr_val_custom)
    
    # Update the best parameters if the current AUC is better
    if roc_auc_val_custom > best_custom_auc:
        best_custom_auc = roc_auc_val_custom
        best_custom_params = n_estimators

# Print the best number of trees and validation AUC
print(f"\nBest Custom Random Forest Parameters:")
print(f"n_estimators: {best_custom_params}")
print(f"Best Validation AUC (Custom Random Forest): {best_custom_auc:.2f}")

# Train the final Custom Random Forest with the best number of trees
clf_custom = CustomRandomForest(n_estimators=best_custom_params, random_state=42)
clf_custom.fit(X_train.values, y_train.values)

# Evaluate on the validation set
y_pred_val_custom = clf_custom.predict(X_val.values)
y_pred_proba_val_custom = clf_custom.predict_proba(X_val.values)[:, 1]
accuracy_val_custom = accuracy_score(y_val, y_pred_val_custom)
precision_val_custom = precision_score(y_val, y_pred_val_custom)
recall_val_custom = recall_score(y_val, y_pred_val_custom)
roc_auc_val_custom = auc(*roc_curve(y_val, y_pred_proba_val_custom)[:2])

# Print validation metrics
print("\nCustom Random Forest (Validation):")
print(f"Validation Accuracy: {accuracy_val_custom:.2f}")
print(f"Validation Precision: {precision_val_custom:.2f}")
print(f"Validation Recall: {recall_val_custom:.2f}")
print(f"Validation AUC: {roc_auc_val_custom:.2f}")

# Evaluate on the test set
y_pred_test_custom = clf_custom.predict(X_test.values)
y_pred_proba_test_custom = clf_custom.predict_proba(X_test.values)[:, 1]
accuracy_test_custom = accuracy_score(y_test, y_pred_test_custom)
precision_test_custom = precision_score(y_test, y_pred_test_custom)
recall_test_custom = recall_score(y_test, y_pred_test_custom)
roc_auc_test_custom = auc(*roc_curve(y_test, y_pred_proba_test_custom)[:2])

# Print test metrics
print("\nCustom Random Forest (Test):")
print(f"Test Accuracy: {accuracy_test_custom:.2f}")
print(f"Test Precision: {precision_test_custom:.2f}")
print(f"Test Recall: {recall_test_custom:.2f}")
print(f"Test AUC: {roc_auc_test_custom:.2f}")

# -------------------------------
# Plot ROC Curves
# -------------------------------
plt.figure()
plt.plot(*roc_curve(y_test, y_pred_proba_test_sklearn)[:2], color='blue', lw=2, label=f'Scikit-learn Random Forest (AUC = {roc_auc_test_sklearn:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Scikit-learn Random Forest')
plt.legend(loc="lower right")
plt.show()

plt.figure()
plt.plot(*roc_curve(y_test, y_pred_proba_test_custom)[:2], color='green', lw=2, label=f'Custom Random Forest (AUC = {roc_auc_test_custom:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Custom Random Forest')
plt.legend(loc="lower right")
plt.show()