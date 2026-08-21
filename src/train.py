import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
import json
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

EVAL_THRESHOLD = 0.68
if os.getenv("MLFLOW_TRACKING_URI"):
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "wine-quality"))


def _build_model(params: dict):
    model_type = params.get("model_type", "random_forest")
    model_params = {key: value for key, value in params.items() if key != "model_type"}

    if model_type == "random_forest":
        model = RandomForestClassifier(**model_params, random_state=42)
    elif model_type == "gradient_boosting":
        model = GradientBoostingClassifier(**model_params, random_state=42)
    elif model_type == "logistic_regression":
        model = LogisticRegression(**model_params, random_state=42, max_iter=1000)
    else:
        raise ValueError(
            f"Unknown model_type {model_type!r}. "
            "Choose random_forest, gradient_boosting, or logistic_regression."
        )
    return model, model_type


def train(
    params: dict,
    data_path: str = "data/train_phase1.csv",
    eval_path: str = "data/eval.csv",
) -> float:
    """
    Huan luyen mo hinh va ghi nhan ket qua vao MLflow.

    Tham so:
        params     : dict chua cac sieu tham so cho RandomForestClassifier.
        data_path  : duong dan den file du lieu huan luyen.
        eval_path  : duong dan den file du lieu danh gia.

    Tra ve:
        accuracy (float): do chinh xac tren tap danh gia.
    """

    # Doc du lieu huan luyen va danh gia
    df_train = pd.read_csv(data_path)
    df_eval = pd.read_csv(eval_path)

    # Tach dac trung (X) va nhan (y)
    X_train = df_train.drop(columns=["target"])
    y_train = df_train["target"]
    X_eval = df_eval.drop(columns=["target"])
    y_eval = df_eval["target"]
    label_distribution = y_train.value_counts(normalize=True).reindex([0, 1, 2], fill_value=0)
    label_distribution = {
        str(label): float(ratio) for label, ratio in label_distribution.items()
    }
    for label, ratio in label_distribution.items():
        if ratio < 0.10:
            print(f"WARNING: label {label} represents only {ratio:.2%} of training samples")

    with mlflow.start_run():

        # Ghi nhan cac sieu tham so
        mlflow.log_params(params)

        model, model_type = _build_model(params)
        mlflow.log_param("model_type", model_type)
        model.fit(X_train, y_train)

        # Du doan tren tap danh gia va tinh chi so
        preds = model.predict(X_eval)
        acc = accuracy_score(y_eval, preds)
        f1 = f1_score(y_eval, preds, average="weighted")
        labels = [0, 1, 2]
        matrix = confusion_matrix(y_eval, preds, labels=labels)
        precision, recall, _, _ = precision_recall_fscore_support(
            y_eval, preds, labels=labels, zero_division=0
        )

        # Ghi nhan chi so vao MLflow
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.sklearn.log_model(model, "model")

        # In ket qua ra man hinh
        print(f"Accuracy: {acc:.4f} | F1: {f1:.4f}")

        # Luu metrics ra file outputs/metrics.json
        # File nay duoc doc boi GitHub Actions o Buoc 2
        os.makedirs("outputs", exist_ok=True)
        metrics = {
            "accuracy": acc,
            "f1_score": f1,
            "model_type": model_type,
            "label_distribution": label_distribution,
            "precision": {str(label): float(value) for label, value in zip(labels, precision)},
            "recall": {str(label): float(value) for label, value in zip(labels, recall)},
        }
        with open("outputs/metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        with open("outputs/report.txt", "w") as f:
            f.write(f"model_type: {model_type}\n")
            f.write(f"accuracy: {acc:.6f}\n")
            f.write(f"f1_score: {f1:.6f}\n\n")
            f.write("confusion_matrix (labels 0, 1, 2):\n")
            for row in matrix:
                f.write(" ".join(str(int(value)) for value in row) + "\n")
            f.write("\nper-class metrics:\n")
            for label, p_value, r_value in zip(labels, precision, recall):
                f.write(f"class {label}: precision={p_value:.6f}, recall={r_value:.6f}\n")
            f.write("\nlabel_distribution:\n")
            for label, ratio in label_distribution.items():
                f.write(f"class {label}: {ratio:.6%}\n")

        # Luu mo hinh ra file models/model.pkl
        # File nay duoc upload len GCS o Buoc 2
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/model.pkl")

    # Tra ve accuracy
    return acc


if __name__ == "__main__":
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    train(params)
