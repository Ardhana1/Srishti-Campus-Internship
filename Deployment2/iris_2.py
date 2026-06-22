import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
iris = load_iris(as_frame=True)
X = iris.data
y = iris.target
df = pd.concat([X, pd.Series(y, name='target')], axis=1)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
# Train
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)
# Predict & evaluate
y_pred = rf_classifier.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f'RandomForest Test Accuracy: {acc:.4f}\n')
print('Classification report:')
print(classification_report(y_test, y_pred, target_names=iris.target_names))
# Confusion matrix (plot)
cm = confusion_matrix(y_test, y_pred)
print('Confusion matrix:\n', cm)
import matplotlib.pyplot as plt
plt.figure(figsize=(4,3))
plt.imshow(cm, interpolation='nearest')
plt.title('Confusion Matrix')
plt.colorbar()
plt.xticks(range(len(iris.target_names)), iris.target_names, rotation=45)
plt.yticks(range(len(iris.target_names)), iris.target_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.show()

import os
import joblib
import pickle
import json
from sklearn.metrics import accuracy_score

# Create models directory
os.makedirs('models', exist_ok=True)

# Save model using joblib
joblib.dump(rf_classifier, 'models/iris_model.joblib')

# Save model using pickle
with open('models/iris_model.pickle', 'wb') as f:
    pickle.dump(rf_classifier, f)

# Save metadata
try:
    acc_val = float(accuracy_score(y_test, y_pred))
except Exception:
    acc_val = None

model_info = {
    'model_type': 'RandomForestClassifier',
    'accuracy': acc_val,
    'feature_names': iris.feature_names,
    'target_names': iris.target_names.tolist()
}

with open('models/model_info.json', 'w') as f:
    json.dump(model_info, f, indent=2)

# Define slider ranges
feature_ranges = {
    'sepal_length': {
        'min': float(X['sepal length (cm)'].min()),
        'max': float(X['sepal length (cm)'].max()),
        'default': float(X['sepal length (cm)'].mean())
    },
    'sepal_width': {
        'min': float(X['sepal width (cm)'].min()),
        'max': float(X['sepal width (cm)'].max()),
        'default': float(X['sepal width (cm)'].mean())
    },
    'petal_length': {
        'min': float(X['petal length (cm)'].min()),
        'max': float(X['petal length (cm)'].max()),
        'default': float(X['petal length (cm)'].mean())
    },
    'petal_width': {
        'min': float(X['petal width (cm)'].min()),
        'max': float(X['petal width (cm)'].max()),
        'default': float(X['petal width (cm)'].mean())
    }
}

with open('models/feature_ranges.json', 'w') as f:
    json.dump(feature_ranges, f, indent=2)

print("Saved:")
print("models/iris_model.joblib")
print("models/iris_model.pickle")
print("models/model_info.json")
print("models/feature_ranges.json")