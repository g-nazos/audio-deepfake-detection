import os
import json
import joblib
import platform
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_curve,
)

from sklearn.model_selection import GridSearchCV, GroupKFold


def get_file_path(file, dataset_pathing, label):
    """
    This function gets the system path of the input audio file path.
    Input:
    path = Input audio file path (relative to the current working directory)
    returns:
    System path of the input audio file path if it exists, otherwise None
    """
    try:
        cwd = os.getcwd()
    except Exception as e:
        print(f"Error getting current working directory: {e}")
        return None
    if label == "fake":
        file_path = os.path.join(cwd, dataset_pathing, label, file)
    elif label == "real":
        file_path = os.path.join(cwd, dataset_pathing, label, file)
    else:
        raise FileNotFoundError(f"Label given doesn't match!")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} does not exist")
    return file_path


def train_and_evaluate_linear_svm(
    train_path: str,
    test_path: str,
    svc_params: dict | None = None,
):
    """
    Train a Linear SVM on extracted audio features and evaluate on a test set.

    Returns everything needed to save an experiment:
    - trained pipeline
    - evaluation metrics
    - model parameters
    - feature names
    - extra metadata (train/test size)
    """

    if svc_params is None:
        svc_params = {
            "C": 1.0,
            "class_weight": "balanced",
            "max_iter": 20000,
            "random_state": 42,
        }

    train_df = pd.read_parquet(train_path)
    train_df.dropna(inplace=True)
    test_df = pd.read_parquet(test_path)
    test_df.dropna(inplace=True)

    # Split features and labels
    def split_xy(df):
        X = df.drop(columns=["label", "filename"], errors="ignore")
        y = df["label"].map({"real": 0, "fake": 1}).values
        return X.values, y, X.columns.tolist()

    X_train, y_train, feature_names = split_xy(train_df)
    X_test, y_test, _ = split_xy(test_df)

    # Build pipeline
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("svm", LinearSVC(**svc_params)),
        ]
    )

    # Train
    pipeline.fit(X_train, y_train)

    # Predict
    y_pred = pipeline.predict(X_test)
    y_scores = pipeline.decision_function(X_test)
    # Compute metrics
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, average="macro")),
        "recall": float(recall_score(y_test, y_pred, average="macro")),
        "f1": float(f1_score(y_test, y_pred, average="macro")),
        "roc_auc": float(roc_auc_score(y_test, y_scores)),
    }

    # Extra metadata for saving
    metadata_extra = {
        "train_samples": X_train.shape[0],
        "test_samples": X_test.shape[0],
    }

    return pipeline, metrics, svc_params, feature_names, metadata_extra


def train_and_evaluate_non_linear_svm(
    train_path: str,
    test_path: str,
    svc_params: dict | None = None,
):
    """
    Train a Linear SVM on extracted audio features and evaluate on a test set.

    Returns everything needed to save an experiment:
    - trained pipeline
    - evaluation metrics
    - model parameters
    - feature names
    - extra metadata (train/test size)
    """

    if svc_params is None:
        svc_params = {
            "kernel": "rbf",
            "C": 1.0,
            "class_weight": "balanced",
            "max_iter": 20000,
            "random_state": 42,
        }

    train_df = pd.read_parquet(train_path)
    train_df.dropna(inplace=True)
    test_df = pd.read_parquet(test_path)
    test_df.dropna(inplace=True)

    # Split features and labels
    def split_xy(df):
        X = df.drop(columns=["label", "filename"], errors="ignore")
        y = df["label"].map({"real": 0, "fake": 1}).values
        return X.values, y, X.columns.tolist()

    X_train, y_train, feature_names = split_xy(train_df)
    X_test, y_test, _ = split_xy(test_df)

    # Build pipeline
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("svm", SVC(**svc_params)),
        ]
    )

    # Train
    pipeline.fit(X_train, y_train)

    # Predict
    y_pred = pipeline.predict(X_test)
    y_scores = pipeline.decision_function(X_test)

    # Compute metrics
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, average="macro")),
        "recall": float(recall_score(y_test, y_pred, average="macro")),
        "f1": float(f1_score(y_test, y_pred, average="macro")),
        "roc_auc": float(roc_auc_score(y_test, y_scores)),
    }

    # Extra metadata for saving
    metadata_extra = {
        "train_samples": X_train.shape[0],
        "test_samples": X_test.shape[0],
    }

    return pipeline, metrics, svc_params, feature_names, metadata_extra


def train_and_evaluate_logistic_regression(
    train_path: str,
    test_path: str,
    lr_params: dict | None = None,
):
    """
    Train a Logistic Regression model on extracted audio features and evaluate on a test set.

    Returns everything needed to save an experiment:
    - trained pipeline
    - evaluation metrics
    - model parameters
    - feature names
    - extra metadata (train/test size)
    """

    if lr_params is None:
        lr_params = {
            "C": 1.0,  # Regularization strength
            "class_weight": "balanced",  # Handle imbalanced classes
            "max_iter": 1000,  # Usually enough to converge
            "random_state": 42,
            "solver": "liblinear",  # Good for small-medium datasets, handles binary classification well
            "penalty": "l2",  # Standard L2 regularization
        }

    # Load datasets
    train_df = pd.read_parquet(train_path)
    train_df.dropna(inplace=True)
    test_df = pd.read_parquet(test_path)
    test_df.dropna(inplace=True)

    # Split features and labels
    def split_xy(df):
        X = df.drop(columns=["label", "filename"], errors="ignore")
        y = df["label"].map({"real": 0, "fake": 1}).values
        return X.values, y, X.columns.tolist()

    X_train, y_train, feature_names = split_xy(train_df)
    X_test, y_test, _ = split_xy(test_df)

    # Build pipeline
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(**lr_params)),
        ]
    )

    # Train
    pipeline.fit(X_train, y_train)

    # Predict
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]  # For ROC AUC

    # Compute metrics
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, average="macro")),
        "recall": float(recall_score(y_test, y_pred, average="macro")),
        "f1": float(f1_score(y_test, y_pred, average="macro")),
        "roc_auc": float(
            roc_auc_score(y_test, y_proba)
        ),  # Use probabilities for ROC AUC
    }

    # Extra metadata for saving
    metadata_extra = {
        "train_samples": X_train.shape[0],
        "test_samples": X_test.shape[0],
    }

    return pipeline, metrics, lr_params, feature_names, metadata_extra


def save_experiment(
    model,
    metrics: dict,
    experiment_dir: str = "experiments",
    experiment_name: str | None = None,
    model_params: dict | None = None,
    feature_names: list | None = None,
    metadata_extra: dict | None = None,
):
    """
    Save a trained model, evaluation metrics, model parameters, and metadata
    to a structured experiment folder.

    Parameters
    ----------
    model : any
        Trained model object (e.g., sklearn pipeline, XGBoost model).

    metrics : dict
        Dictionary containing evaluation metrics.

    experiment_dir : str
        Root directory to store experiments.

    experiment_name : str, optional
        Name of the experiment folder. Auto-generated if None.

    model_params : dict, optional
        Dictionary of model hyperparameters.

    feature_names : list of str, optional
        List of feature names used in training.

    metadata_extra : dict, optional
        Additional metadata to save (dataset info, notes, etc.).

    Returns
    -------
    exp_path : str
        Path to the saved experiment folder.
    """
    # Create experiment folder
    os.makedirs(experiment_dir, exist_ok=True)

    if experiment_name is None:
        experiment_name = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    exp_path = os.path.join(experiment_dir, experiment_name)
    os.makedirs(exp_path, exist_ok=True)

    # Save model
    joblib.dump(model, os.path.join(exp_path, "model.joblib"))

    # Save metrics
    with open(os.path.join(exp_path, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    # Save model parameters
    if model_params is not None:
        with open(os.path.join(exp_path, "model_params.json"), "w") as f:
            json.dump(model_params, f, indent=4)

    # Save metadata
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "num_features": len(feature_names) if feature_names is not None else None,
        "feature_names": feature_names,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }

    if metadata_extra:
        metadata.update(metadata_extra)

    with open(os.path.join(exp_path, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"Experiment saved to: {exp_path}")
    return exp_path


def evaluate_model_on_parquet(
    model,
    test_path: str,
    plots: bool = False,
):
    """
    Evaluate a trained model/pipeline on a parquet test dataset.

    Returns:
    - metrics (dict)
    - extra metadata (dict)
    """

    # Load test dataset
    test_df = pd.read_parquet(test_path)
    test_df.dropna(inplace=True)

    # Split features and labels
    X_test = test_df.drop(columns=["label", "filename"], errors="ignore").values
    y_test = test_df["label"].map({"real": 0, "fake": 1}).values

    # Predict
    y_pred = model.predict(X_test)

    y_proba = None
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]

    # Metrics (explicitly fake=1)
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, pos_label=1)),
        "recall": float(recall_score(y_test, y_pred, pos_label=1)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro")),
    }

    if y_proba is not None:
        metrics["roc_auc"] = float(roc_auc_score(y_test, y_proba))
    else:
        y_scores = model.decision_function(X_test)
        metrics["roc_auc"] = float(roc_auc_score(y_test, y_scores))

    metadata_extra = {
        "test_samples": int(X_test.shape[0]),
        "test_file": test_path,
    }

    # -------------------------
    # Optional visualizations
    # -------------------------
    if plots:
        # Confusion Matrix (most important)
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["real (0)", "fake (1)"],
        )
        disp.plot()
        plt.title("Confusion Matrix")
        plt.show()

        # Precision–Recall curve (recall tuning)
        if y_proba is not None:
            precision, recall, thresholds = precision_recall_curve(y_test, y_proba)

            plt.figure()
            plt.plot(recall, precision)
            plt.xlabel("Recall (fake)")
            plt.ylabel("Precision (fake)")
            plt.title("Precision–Recall Curve")
            plt.grid(True)
            plt.show()

        # Score distribution (debug false negatives)
        if y_proba is not None:
            plt.figure()
            plt.hist(y_proba[y_test == 0], bins=50, alpha=0.6, label="real")
            plt.hist(y_proba[y_test == 1], bins=50, alpha=0.6, label="fake")
            plt.xlabel("P(fake)")
            plt.ylabel("Count")
            plt.title("Prediction Score Distribution")
            plt.legend()
            plt.show()

    return metrics, metadata_extra


def find_best_trained_monel(folder_path, metric):
    best_model = None
    metric_value = 0
    for exp in os.listdir(folder_path):
        experiment_name = exp
        with open(os.path.join(folder_path, exp, "metrics.json"), "r") as f:
            metrics = json.load(f)
        new_metric_value = metrics[metric]
        if new_metric_value > metric_value:
            metric_value = new_metric_value
            best_model = exp
    print(f"Best model according to {metric}: {best_model}, {metric}={metric_value}")
    return best_model


def find_highly_correlated_features(corr_matrix, threshold=0.85):
    """
    Find pairs of features with correlation above the threshold.

    Args:
        corr_matrix: Correlation matrix DataFrame
        threshold: Correlation threshold

    Returns:
        DataFrame with feature pairs and their correlation values
    """
    # Create a copy and set diagonal to NaN to exclude self-correlations
    corr_matrix_copy = corr_matrix.copy()
    mask = np.tril(np.ones(corr_matrix_copy.shape), k=-1).astype(bool)
    corr_matrix_lower = corr_matrix_copy.mask(mask)

    corr_matrix_selected = corr_matrix_lower
    # Find pairs above threshold (both positive and negative)
    high_corr_pairs = []
    for i in range(len(corr_matrix_selected.columns)):
        for j in range(i + 1, len(corr_matrix_selected.columns)):
            corr_value = corr_matrix_selected.iloc[i, j]
            if not np.isnan(corr_value) and abs(corr_value) >= threshold:
                high_corr_pairs.append(
                    {
                        "feature_1": corr_matrix_selected.columns[i],
                        "feature_2": corr_matrix_selected.columns[j],
                        "correlation": corr_value,
                    }
                )

    result_df = pd.DataFrame(high_corr_pairs)
    if len(result_df) > 0:
        result_df = result_df.sort_values("correlation", key=abs, ascending=False)

    return result_df


def grid_search_model(
    model,
    param_grid: dict,
    train_path: str,
    test_path: str,
    *,
    scoring: str = "f1_macro",
    cv: int = 5,
    n_jobs: int = -1,
    verbose: int = 2,
):
    # Load data
    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    # Clean NaNs + infs
    for df in (train_df, test_df):
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)

    # Extract group_id from filename
    for df in (train_df, test_df):
        df["group_id"] = df["filename"].str.extract(r"(file\d+\.(?:wav|mp3))")[0]

    # Split X, y, groups
    def split_xy_groups(df):
        X = df.drop(columns=["label", "filename", "group_id"], errors="ignore")
        y = df["label"].map({"real": 0, "fake": 1}).values
        groups = df["group_id"].values
        return X.values, y, groups, X.columns.tolist()

    X_train, y_train, groups_train, feature_names = split_xy_groups(train_df)
    X_test, y_test, groups_test, _ = split_xy_groups(test_df)

    # GroupKFold
    gkf = GroupKFold(n_splits=cv)

    # Grid search with GroupKFold
    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring=scoring,
        cv=gkf.split(X_train, y_train, groups_train),
        n_jobs=n_jobs,
        verbose=verbose,
    )

    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_

    # Predictions
    y_pred = best_model.predict(X_test)

    # Scores for ROC AUC
    if hasattr(best_model, "decision_function"):
        y_scores = best_model.decision_function(X_test)
    elif hasattr(best_model, "predict_proba"):
        y_scores = best_model.predict_proba(X_test)[:, 1]
    else:
        raise RuntimeError("Model does not support ROC AUC scoring")

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, average="macro")),
        "recall": float(recall_score(y_test, y_pred, average="macro")),
        "f1": float(f1_score(y_test, y_pred, average="macro")),
        "roc_auc": float(roc_auc_score(y_test, y_scores)),
    }

    metadata = {
        "cv_best_score": float(grid.best_score_),
        "train_samples": X_train.shape[0],
        "test_samples": X_test.shape[0],
        "train_groups": len(np.unique(groups_train)),
        "test_groups": len(np.unique(groups_test)),
    }

    return best_model, metrics, grid.best_params_, metadata, feature_names
