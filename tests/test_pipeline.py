import pandas as pd
from sklearn.ensemble import IsolationForest

def test_csv_loads():
    df = pd.read_csv("data.csv")
    assert not df.empty, "CSV file is empty"

def test_model_runs():
    df = pd.read_csv("data.csv")
    X = df[['cpu_percent']]
    model = IsolationForest()
    model.fit(X)
    preds = model.predict(X)
    assert len(preds) == len(X), "Predictions length mismatch"
