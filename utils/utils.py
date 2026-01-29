import os
import json
from re import X
import joblib
import platform
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
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
)

from sklearn.model_selection import GridSearchCV


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
):
    """
    Evaluate a trained model/pipeline on a parquet test dataset.

    Returns:
    - evaluation metrics
    - extra metadata (test size)
    """

    # Load test dataset
    test_df = pd.read_parquet(test_path)
    test_df.dropna(inplace=True)

    # Split features and labels
    X_test = test_df.drop(columns=["label", "filename"], errors="ignore").values
    y_test = test_df["label"].map({"real": 0, "fake": 1}).values

    # Predict
    y_pred = model.predict(X_test)

    # Some models may not support predict_proba
    y_proba = None
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]

    # Compute metrics
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred, average="macro")),
    }

    if y_proba is not None:
        metrics["roc_auc"] = float(roc_auc_score(y_test, y_proba))

    metadata_extra = {"test_samples": X_test.shape[0], "test_file": test_path}

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

    def split_xy(df):
        X = df.drop(columns=["label", "filename"], errors="ignore")
        y = df["label"].map({"real": 0, "fake": 1}).values
        if np.isnan(y).any():
            raise ValueError("Invalid label values detected")
        return X.values, y, X.columns.tolist()

    X_train, y_train, feature_names = split_xy(train_df)
    X_test, y_test, _ = split_xy(test_df)

    # Grid search
    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
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
    }

    return best_model, metrics, grid.best_params_, metadata, feature_names

def train_and_evaluate_decision_tree(
    train_path: str,
    val_path: str | None = None,
    test_path: str | None = None,
    dt_params: dict | None = None,
    criterion: str | None = None,
):
    """
    Train a Decision Tree Classifier on extracted audio features and evaluate.

    At least one of val_path or test_path must be provided.
    """
    if criterion is None:
        criterion = "gini"
    if dt_params is None:
        dt_params = {
            "max_depth": 10,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "max_features": "auto",
            "random_state": 42,
        }

    train_df = pd.read_parquet(train_path)
    train_df.dropna(inplace=True)
    X_train = train_df.drop(columns=["label", "filename"], errors="ignore")
    y_train = train_df["label"].map({"real": 0, "fake": 1}).values
    feature_names = X_train.columns.tolist()

    if val_path is not None:
        val_df = pd.read_parquet(val_path)
        val_df.dropna(inplace=True)
        X_val = val_df.drop(columns=["label", "filename"], errors="ignore")
        y_val = val_df["label"].map({"real": 0, "fake": 1}).values
    else:
        X_val = y_val = None

    if test_path is not None:
        test_df = pd.read_parquet(test_path)
        test_df.dropna(inplace=True)
        X_test = test_df.drop(columns=["label", "filename"], errors="ignore")
        y_test = test_df["label"].map({"real": 0, "fake": 1}).values
    else:
        X_test = y_test = None

    if X_test is None and X_val is None:
        raise ValueError("At least one of val_path or test_path must be provided.")

    if criterion == "gini":
        clf = DecisionTreeClassifier(criterion="gini", **dt_params)
    else:
        clf = DecisionTreeClassifier(criterion="entropy", **dt_params)

    clf.fit(X_train, y_train)
    metrics = {}
    metadata_extra = {"train_samples": X_train.shape[0]}

    if X_test is not None:
        y_pred = clf.predict(X_test)
        y_scores = clf.predict_proba(X_test)[:, 1]
        metadata_extra["test_samples"] = X_test.shape[0]
        metrics["accuracy"] = float(accuracy_score(y_test, y_pred))
        metrics["precision"] = float(precision_score(y_test, y_pred, average="macro"))
        metrics["recall"] = float(recall_score(y_test, y_pred, average="macro"))
        metrics["f1"] = float(f1_score(y_test, y_pred, average="macro"))
        metrics["roc_auc"] = float(roc_auc_score(y_test, y_scores))

    if X_val is not None:
        y_val_pred = clf.predict(X_val)
        y_val_scores = clf.predict_proba(X_val)[:, 1]
        metadata_extra["val_samples"] = X_val.shape[0]
        if X_test is None:
            metrics["accuracy"] = float(accuracy_score(y_val, y_val_pred))
            metrics["precision"] = float(precision_score(y_val, y_val_pred, average="macro"))
            metrics["recall"] = float(recall_score(y_val, y_val_pred, average="macro"))
            metrics["f1"] = float(f1_score(y_val, y_val_pred, average="macro"))
            metrics["roc_auc"] = float(roc_auc_score(y_val, y_val_scores))
        else:
            metrics["val_accuracy"] = float(accuracy_score(y_val, y_val_pred))
            metrics["val_precision"] = float(precision_score(y_val, y_val_pred, average="macro"))
            metrics["val_recall"] = float(recall_score(y_val, y_val_pred, average="macro"))
            metrics["val_f1"] = float(f1_score(y_val, y_val_pred, average="macro"))
            metrics["val_roc_auc"] = float(roc_auc_score(y_val, y_val_scores))

    return clf, metrics, dt_params, feature_names, metadata_extra