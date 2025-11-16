import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score
)
import mlflow
from mlflow.tracking import MlflowClient
import mlflow.sklearn
from mlflow.models import infer_signature
import matplotlib.pyplot as plt

MLFLOW_URI = "http://127.0.0.1:8100"
EXPERIMENT_NAME = "IRIS_Classifier_Poisoned"
REGISTERED_MODEL_NAME = "iris_dt_poisoned"
ARTIFACT_PATH = "iris_model"
LOCAL_MODEL_DIR = "models"
TEST_DATA = "data/test.csv"

os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)

mlflow.set_tracking_uri(MLFLOW_URI)
client = MlflowClient()
mlflow.set_experiment(EXPERIMENT_NAME)

# =====================================================================
# POISONING
# =====================================================================

def poison_data(df, poison_level, random_state=42):
    np.random.seed(random_state)
    df_poisoned = df.copy()

    feature_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

    n_samples = len(df_poisoned)
    n_poison = int(n_samples * poison_level)

    poison_indices = np.random.choice(n_samples, size=n_poison, replace=False)

    for idx in poison_indices:
        df_poisoned.loc[idx, feature_cols] = np.random.uniform(1, 8, size=len(feature_cols))

    return df_poisoned, poison_indices


# =====================================================================
# TRAINING
# =====================================================================

def train_and_store_with_poisoning(data_path, poison_level=0.0, n_iter=10, random_state=42, run_name=None):

    df = pd.read_csv(data_path)
    print("\n" + "="*70)
    print(f"Training with poison level {poison_level*100:.0f}%")
    print("="*70)

    # Apply poisoning
    if poison_level > 0:
        df, poison_indices = poison_data(df, poison_level, random_state)
        print(f"Poisoned samples: {len(poison_indices)}")
    else:
        poison_indices = []
        print("Using clean training data")

    X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
    y = df['species']

    # No train/test split — training on full dataset
    X_train, y_train = X, y

    # Simple hyperparameters
    param_dist = {"max_depth": [3, 4, 5, None]}

    model = DecisionTreeClassifier(random_state=random_state)
    search = RandomizedSearchCV(
        model,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring="accuracy",
        cv=3,
        random_state=random_state
    )

    search.fit(X_train, y_train)
    best = search.best_estimator_

    # Only CV metrics (no test metrics here)
    cv_score = search.best_score_

    # Save model
    local_model_path = os.path.join(
        LOCAL_MODEL_DIR,
        f"model_poison_{poison_level*100:.0f}.joblib"
    )
    joblib.dump(best, local_model_path)

    # MLflow logging
    with mlflow.start_run(run_name=run_name or f"poison_{poison_level*100:.0f}%") as run:
        mlflow.log_params(search.best_params_)
        mlflow.log_param("poison_level", poison_level)
        mlflow.log_metric("cv_accuracy", cv_score)

        signature = infer_signature(X_train, best.predict(X_train))

        mlflow.sklearn.log_model(
            best,
            name=ARTIFACT_PATH,
            registered_model_name=f"{REGISTERED_MODEL_NAME}_{poison_level*100:.0f}pct",
            signature=signature
        )

        mlflow.log_artifact(local_model_path)

        mlflow.log_dict(
            {"poison_level": poison_level,
             "poisoned_samples": len(poison_indices)},
            "poison_info.json"
        )

        run_id = run.info.run_id

    return {
        "model": best,
        "cv_score": cv_score,
        "poison_level": poison_level,
        "run_id": run_id
    }


# =====================================================================
# INFERENCE
# =====================================================================

def perform_inference(models_dict, test_data_path):

    df_test = pd.read_csv(test_data_path)
    X_clean = df_test[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
    y_clean = df_test['species']

    results = {}

    for poison_level, model_info in models_dict.items():
        model = model_info["model"]

        # Prediction
        y_pred = model.predict(X_clean)
        y_proba = model.predict_proba(X_clean)
        max_proba = np.max(y_proba, axis=1)

        acc = accuracy_score(y_clean, y_pred)
        report = classification_report(y_clean, y_pred, output_dict=True)
        cm = confusion_matrix(y_clean, y_pred)

        results[poison_level] = {
            "accuracy_clean_test": acc,
            "mean_confidence": float(np.mean(max_proba)),
            "std_confidence": float(np.std(max_proba)),
            "report": report,
            "confusion_matrix": cm
        }

        print(f"\nPoison {poison_level*100:.0f}% → Accuracy: {acc:.4f}")

    return results


# =====================================================================
# MLflow logging for inference
# =====================================================================

def log_inference(results, baseline_accuracy):

    with mlflow.start_run(run_name="inference_results"):

        for poison_level, res in results.items():
            acc = res["accuracy_clean_test"]
            drop = baseline_accuracy - acc

            mlflow.log_metric(f"accuracy_poison_{poison_level*100:.0f}", acc)
            mlflow.log_metric(f"accuracy_drop_poison_{poison_level*100:.0f}", drop)

        mlflow.log_dict(results, "inference_results.json")


# =====================================================================
# MAIN
# =====================================================================

print("\n=== IRIS POISONING EXPERIMENT ===")

DATA_PATH = "data/data.csv"

models = {}

# Train 0%, 5%, 10%, 50%
models[0.0] = train_and_store_with_poisoning(DATA_PATH, 0.0, run_name="clean_baseline")
models[0.05] = train_and_store_with_poisoning(DATA_PATH, 0.05)
models[0.10] = train_and_store_with_poisoning(DATA_PATH, 0.10)
models[0.50] = train_and_store_with_poisoning(DATA_PATH, 0.50)

baseline_accuracy = None  # filled after inference

# Inference
results = perform_inference(models, TEST_DATA)

baseline_accuracy = results[0.0]["accuracy_clean_test"]

# Log inference
log_inference(results, baseline_accuracy)

