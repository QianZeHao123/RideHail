# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from imblearn.ensemble import BalancedRandomForestClassifier, RUSBoostClassifier
import joblib

# Load training data
df = pd.read_csv('data/order_driver.csv')

# ✅ Filter only completed orders within the target region
df = df.loc[(df['status'] == 5) & (df['outside'] == 0)]
print(df.shape)
print(df['accept'].describe())

# ✅ Define features & target variable
X = df[['commission', 'driver_distance', 'hour', 'weather_code', 'work_time_minutes']]
y = df['accept']

# ✅ Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Apply SMOTE only to training data (prevents leakage)
# smote = SMOTE(random_state=42)
# X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# ✅ Train model (BalancedRandomForest handles imbalance natively)
model = RUSBoostClassifier(random_state=42)
model.fit(X_train, y_train)

# ✅ Make predictions
y_pred = model.predict(X_test)
y_probs = model.predict_proba(X_test)[:, 1]

# ✅ Evaluate model performance
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("ROC AUC Score:", roc_auc_score(y_test, y_probs))

# ✅ Save trained model (only the classifier, without SMOTE)
joblib.dump(model, 'models/acceptance_model_ensemble.pkl')
