import mlflow
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

try:
    from src.train import _build_model
except ModuleNotFoundError:
    from train import _build_model


def main():
    train_df = pd.read_csv("data/train_phase1.csv")
    eval_df = pd.read_csv("data/eval.csv")
    X_train = train_df.drop(columns=["target"])
    y_train = train_df["target"]
    X_eval = eval_df.drop(columns=["target"])
    y_eval = eval_df["target"]
    candidates = [
        {"model_type": "gradient_boosting", "n_estimators": 100, "max_depth": 3},
        {"model_type": "logistic_regression", "C": 1.0},
    ]

    for params in candidates:
        with mlflow.start_run(run_name=f"comparison-{params['model_type']}"):
            model, model_type = _build_model(params)
            model.fit(X_train, y_train)
            predictions = model.predict(X_eval)
            accuracy = accuracy_score(y_eval, predictions)
            f1 = f1_score(y_eval, predictions, average="weighted")
            mlflow.log_params(params)
            mlflow.log_param("model_type", model_type)
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("f1_score", f1)
            print(f"{model_type}: accuracy={accuracy:.4f}, f1_score={f1:.4f}")


if __name__ == "__main__":
    main()