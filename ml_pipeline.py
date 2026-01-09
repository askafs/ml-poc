import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib

# Step 1: Load data
data = pd.read_csv("data.csv")
X = data[['cpu_percent']]

# Step 2: Train anomaly detection model
model = IsolationForest(contamination=0.2, random_state=42)
model.fit(X)

# Step 3: Predict anomalies
data['anomaly'] = model.predict(X)
data['anomaly'] = data['anomaly'].map({1: 0, -1: 1})  # 1 = anomaly

# Step 4: Save model and predictions
joblib.dump(model, "anomaly_model.joblib")
data.to_csv("predictions.csv", index=False)

print("ML pipeline executed successfully.")
print(data)

