import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_curve, auc
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv("cw4/lab-4-dataset.csv")

# Data preparation
# Remove missing values
data = data.dropna()

# Convert categorical variables to numerical values
categorical_columns = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope', 'AgeGroup', 
                       'RestingBP_Category', 'Cholesterol_Category', 'MaxHR_Category', 'Oldpeak_Category']
data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)

# Split into features (X) and labels (y)
X = data.drop(columns=['HeartDisease'])
y = data['HeartDisease']

# Split into training set (60%), validation set (20%), and test set (20%)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.4, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)

# Train the model on the training set
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Prediction on the validation set
y_pred_val = clf.predict(X_val)
y_pred_proba_val = clf.predict_proba(X_val)[:, 1]

# Metrics on the validation set
accuracy_val = accuracy_score(y_val, y_pred_val)
precision_val = precision_score(y_val, y_pred_val)
recall_val = recall_score(y_val, y_pred_val)
fpr_val, tpr_val, thresholds_val = roc_curve(y_val, y_pred_proba_val)
roc_auc_val = auc(fpr_val, tpr_val)

print(f"Validation Accuracy: {accuracy_val:.2f}")
print(f"Validation Precision: {precision_val:.2f}")
print(f"Validation Recall: {recall_val:.2f}")
print(f"Validation AUC: {roc_auc_val:.2f}")

# Hyperparameter tuning on the training set using validation set
param_grid = {'n_estimators': [50, 100, 200, 300, 500]}
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring='roc_auc')
grid_search.fit(X_train, y_train)

# Best model
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

# Prediction on the test set
y_pred_test = best_model.predict(X_test)
y_pred_proba_test = best_model.predict_proba(X_test)[:, 1]

# Metrics on the test set
accuracy_test = accuracy_score(y_test, y_pred_test)
precision_test = precision_score(y_test, y_pred_test)
recall_test = recall_score(y_test, y_pred_test)
fpr_test, tpr_test, thresholds_test = roc_curve(y_test, y_pred_proba_test)
roc_auc_test = auc(fpr_test, tpr_test)

print(f"Best parameters: {best_params}")
print(f"Test Accuracy: {accuracy_test:.2f}")
print(f"Test Precision: {precision_test:.2f}")
print(f"Test Recall: {recall_test:.2f}")
print(f"Test AUC: {roc_auc_test:.2f}")

#  ROC curve for the test set
plt.figure()
plt.plot(fpr_test, tpr_test, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc_test:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) - Test Set')
plt.legend(loc="lower right")
plt.show()