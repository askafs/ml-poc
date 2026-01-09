import joblib
import pandas as pd

model = joblib.load("predictive_model.pkl")
anomaly_model = joblib.load("anomaly_model.pkl")

new_data = pd.DataFrame({
    "day": [10],
    "month": [1],
    "value": [350]
})

prediction = model.predict(new_data[["day", "month"]])
anomaly = anomaly_model.predict(new_data[["value"]])

print("Prediction:", prediction[0])
print("Anomaly:", "Yes" if anomaly[0] == -1 else "No")
o

