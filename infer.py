import os
import sys
import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.metrics import accuracy_score
from pprint import pprint

MLFLOW_URI = "http://127.0.0.1:8100"
REGISTERED_MODEL_NAME = "iris_dt"
TEST_DATA = "data/test.csv"  # data to run sanity checks

mlflow.set_tracking_uri(MLFLOW_URI)
client = MlflowClient(tracking_uri=MLFLOW_URI)

def get_latest_model_uri(model_name):
    # Try to get production model first
    # Search all model versions for this registered model
    versions = client.search_model_versions(f"name = '{model_name}'")
    if not versions:
        raise ValueError(f"No versions found for model '{model_name}'")

    # Each version object has version number as string, convert to int
    latest = max(versions, key=lambda v: int(v.version))
    latest_version = latest.version
    model_uri = f"models:/{model_name}/{latest_version}"
    return model_uri, latest_version

if __name__ == "__main__":
    if not os.path.exists(TEST_DATA):
        print("Test data not found:", TEST_DATA)
        sys.exit(2)

    try:
        model_uri, version = get_latest_model_uri(REGISTERED_MODEL_NAME)
        print(f"Selected the latest model version {version}")
    except Exception as e:
        print("Error finding model in registry:", str(e))
        sys.exit(2)


    model_uri = f"models:/{REGISTERED_MODEL_NAME}/{version}"
    print("Loading model_uri:", model_uri)

    model = mlflow.pyfunc.load_model(model_uri)
    df = pd.read_csv(TEST_DATA)
    X = df[['sepal_length','sepal_width','petal_length','petal_width']]
    y = df['species']

    preds = model.predict(X)
    acc = accuracy_score(y, preds)
    print("Accuracy on test set:", acc*100, "%")
    
    