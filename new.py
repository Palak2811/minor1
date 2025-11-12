import pickle
import numpy as np
import pandas as pd
from custom_trees import KLDecisionTree, TsallisDecisionTree  # Import the classes

# Load the saved model
with open('boosted_tree_model.pkl', 'rb') as f:
    loaded_data = pickle.load(f)
    ensemble = loaded_data['ensemble']
    le = loaded_data['label_encoder']
    n_classes = loaded_data['n_classes']

# Function to make predictions
def boosting_predict(ensemble, X, n_classes):
    n_samples = X.shape[0]
    scores = np.zeros((n_samples, n_classes))
    for model, alpha in ensemble:
        preds = model.predict(X)
        for i, p in enumerate(preds):
            scores[i, int(p)] += alpha
    return np.argmax(scores, axis=1)

# Example 1: Predict on new single sample
new_sample = np.array([[25, 120, 80, 7.5, 98.6, 75]])
prediction = boosting_predict(ensemble, new_sample, n_classes)
predicted_label = le.inverse_transform(prediction)
print(f"Prediction for new sample: {predicted_label[0]}")

# Example 2: Predict on multiple samples
new_samples = np.array([
    [25, 120, 80, 7.5, 98.6, 75],
    [35, 140, 90, 8.5, 99.0, 85],
    [28, 130, 85, 7.0, 98.0, 70]
])
predictions = boosting_predict(ensemble, new_samples, n_classes)
predicted_labels = le.inverse_transform(predictions)

print("\nPredictions for multiple samples:")
for i, label in enumerate(predicted_labels):
    print(f"Sample {i+1}: {label}")

# Example 3: Load test data and predict
df_test = pd.read_csv("E:/fortransferee/mlproject6-p/Maternal Health Risk Data Set.csv")
X_test = df_test.drop(columns=['RiskLevel']).values[:5]

predictions = boosting_predict(ensemble, X_test, n_classes)
predicted_labels = le.inverse_transform(predictions)

print("\nPredictions for test data:")
for i, label in enumerate(predicted_labels):
    print(f"Sample {i+1}: {label}")