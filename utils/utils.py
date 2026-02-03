import os
import json
from re import X
import joblib
import platform
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
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


from sklearn.base import clone
from sklearn.model_selection import ParameterGrid
from joblib import Parallel, delayed



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
    val_results: list | None = None,
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

    val_results : list, optional
        List of validation results.

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

    if val_results is not None:
        val_results_serializable = [
            {k: v for k, v in r.items() if k != "model"}
            for r in val_results
        ]
        with open(os.path.join(exp_path, "val_results.json"), "w") as f:
            json.dump(val_results_serializable, f, indent=4)

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

    # Optional visualizations
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
    # Find pairs above the threshold (both positive and negative)
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




# Helper: load parquet and split X/y
def load_and_prepare_data(train_path, val_path, test_path):
    def clean_and_split(df):
        df = df.copy()
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)
        X = df.drop(columns=["label", "filename"], errors="ignore")
        y = df["label"].map({"real": 0, "fake": 1}).values
        return X.values, y, X.columns.tolist()

    train_df = pd.read_parquet(train_path)
    val_df   = pd.read_parquet(val_path)
    test_df  = pd.read_parquet(test_path)

    X_train, y_train, feature_names = clean_and_split(train_df)
    X_val, y_val, _ = clean_and_split(val_df)
    X_test, y_test, _ = clean_and_split(test_df)

    return X_train, y_train, X_val, y_val, X_test, y_test, feature_names

# Helper: fit and score a single parameter set
def _fit_and_score(params, model, X_train, y_train, X_val, y_val, scoring="f1_macro"):
    candidate = clone(model).set_params(**params)
    candidate.fit(X_train, y_train)
    y_val_pred = candidate.predict(X_val)
    if hasattr(candidate, "decision_function"):
        y_val_scores = candidate.decision_function(X_val)
    elif hasattr(candidate, "predict_proba"):
        y_val_scores = candidate.predict_proba(X_val)[:, 1]
    else:
        raise RuntimeError("Model does not support ROC AUC scoring")

    acc = accuracy_score(y_val, y_val_pred)
    f1 = f1_score(y_val, y_val_pred, average="macro")
    precision = precision_score(y_val, y_val_pred, average="macro"),
    recall = recall_score(y_val, y_val_pred, average="macro"),
    roc_auc = roc_auc_score(y_val, y_val_scores)
    score = f1 if scoring == "f1_macro" else acc

    return {
        "params": params,
        "val_accuracy": acc,
        "val_f1_macro": f1,
        "val_precision": precision,
        "val_recall": recall,
        "val_roc_auc": roc_auc,
        "selection_score": score,
        "model": candidate
    }

# Main parallel grid search
def grid_search_joblib(
    model,
    param_grid: dict,
    train_path: str,
    val_path: str,
    test_path: str,
    *,
    scoring: str = "f1_macro",
    verbose: int | None = 1,
    n_jobs: int = 1,
):
    # Load and prepare data
    X_train, y_train, X_val, y_val, X_test, y_test, feature_names = load_and_prepare_data(
        train_path, val_path, test_path
    )

    # Create grid
    grid = list(ParameterGrid(param_grid))
    if verbose:
        print(f"Number of fits: {len(grid)} with n_jobs={n_jobs} parallel jobs")

    # Run parallel search
    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(_fit_and_score)(params, model, X_train, y_train, X_val, y_val, scoring)
        for params in grid
    )

    # Collect results and find best
    best_score = -np.inf
    best_model = None
    best_params = None
    val_results = []

    for i, res in enumerate(results, 1):
        val_results.append(res)
        if verbose:
            print(f"[{i}] {res['params']} | val_acc={res['val_accuracy']:.4f} | val_f1={res['val_f1_macro']:.4f}")
        if res["selection_score"] > best_score:
            best_score = res["selection_score"]
            best_model = res["model"]
            best_params = res["params"]

    if verbose:
        print("\nBest validation result:")
        print(f"  params: {best_params}")
        print(f"  {scoring}: {best_score:.4f}")

    # Validation metrics for best model
    y_val_pred_best = best_model.predict(X_val)
    if hasattr(best_model, "decision_function"):
        y_val_best_scores = best_model.decision_function(X_val)
    elif hasattr(best_model, "predict_proba"):
        y_val_best_scores = best_model.predict_proba(X_val)[:, 1]
    else:
        raise RuntimeError("Model does not support ROC AUC scoring")
    
    val_metrics = {
        "accuracy": float(accuracy_score(y_val, y_val_pred_best)),
        "precision": float(precision_score(y_val, y_val_pred_best, average="macro")),
        "recall": float(recall_score(y_val, y_val_pred_best, average="macro")),
        "f1": float(f1_score(y_val, y_val_pred_best, average="macro")),
        "roc_auc": float(roc_auc_score(y_val, y_val_best_scores)),

    }

    # Retrain on train + val
    X_final = np.vstack([X_train, X_val])
    y_final = np.concatenate([y_train, y_val])
    final_model = clone(model).set_params(**best_params)
    final_model.fit(X_final, y_final)

    # Final test evaluation
    y_test_pred = final_model.predict(X_test)
    if hasattr(final_model, "decision_function"):
        y_scores = final_model.decision_function(X_test)
    elif hasattr(final_model, "predict_proba"):
        y_scores = final_model.predict_proba(X_test)[:, 1]
    else:
        raise RuntimeError("Model does not support ROC AUC scoring")

    test_metrics = {
        "accuracy": float(accuracy_score(y_test, y_test_pred)),
        "precision": float(precision_score(y_test, y_test_pred, average="macro")),
        "recall": float(recall_score(y_test, y_test_pred, average="macro")),
        "f1": float(f1_score(y_test, y_test_pred, average="macro")),
        "roc_auc": float(roc_auc_score(y_test, y_scores)),
    }

    metadata = {
        "val_best_score": float(best_score),
        "train_samples": X_train.shape[0],
        "val_samples": X_val.shape[0],
        "test_samples": X_test.shape[0],
    }

    return (
        final_model,
        test_metrics,
        val_metrics,
        best_params,
        val_results,
        metadata,
        feature_names,
    )

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
            "max_features": None,
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


def train_and_evaluate_random_forest(
    train_path: str,
    val_path: str | None = None,
    test_path: str | None = None,
    rf_params: dict | None = None,
):

    default_params = {
        "n_estimators": 100,
        "max_features": "sqrt",
        "criterion": "gini",
        #"class_weight": "balanced",
        "random_state": 42,
        "bootstrap": True,
        "oob_score": True,
        "n_jobs": -1
    }
    # Update defaults with user provided params
    if rf_params:
        default_params.update(rf_params)

    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('rf', RandomForestClassifier(**default_params))
    ])

    def load_data(path):
        if path is None:
            return None, None
        df = pd.read_parquet(path)
        X = df.drop(columns=["label", "filename"], errors="ignore")
        y = df["label"].map({"real": 0, "fake": 1}).values # type: ignore
        return X, y

    X_train, y_train = load_data(train_path)
    X_val, y_val = load_data(val_path)
    X_test, y_test = load_data(test_path)

    if X_test is None and X_val is None:
        raise ValueError("At least one of val_path or test_path must be provided.")

    feature_names = X_train.columns.tolist() # type: ignore
    metadata_extra = {"train_samples": X_train.shape[0]} # type: ignore

    print(f"Training on {X_train.shape[0]} samples with {len(feature_names)} features...") # type: ignore
    pipeline.fit(X_train, y_train)


    def get_metrics(X, y, prefix=""):
        if X is None: 
            return {}
        
        y_pred = pipeline.predict(X)
        y_probs = pipeline.predict_proba(X)[:, 1] # Probability of 'fake' (Class 1)
        
        # Determine prefix for dictionary keys (e.g., 'val_accuracy')
        p = f"{prefix}_" if prefix else ""
        
        return {
            f"{p}accuracy": float(accuracy_score(y, y_pred)),
            f"{p}precision": float(precision_score(y, y_pred, average="macro")), # Binary (default)
            f"{p}recall": float(recall_score(y, y_pred, average="macro")),       # Binary (default)
            f"{p}f1": float(f1_score(y, y_pred, average="macro")),               # Binary (default)
            f"{p}roc_auc": float(roc_auc_score(y, y_probs))
        }

    metrics = {}
    
    # Calculate Test Metrics
    if X_test is not None:
        metadata_extra["test_samples"] = X_test.shape[0]
        metrics.update(get_metrics(X_test, y_test, prefix=""))

    # Calculate Val Metrics
    if X_val is not None:
        metadata_extra["val_samples"] = X_val.shape[0]
        # If we have both, prefix val metrics with 'val_'
        prefix = "val" if X_test is not None else "" 
        metrics.update(get_metrics(X_val, y_val, prefix=prefix))

    # Safely get OOB score (only exists if bootstrap=True and oob_score=True)
    oob_score = None
    if default_params.get("bootstrap") and default_params.get("oob_score"):
        oob_score = pipeline.named_steps['rf'].oob_score_

    return pipeline, metrics, default_params, feature_names, metadata_extra, oob_score



def train_and_evaluate_xgboost(
    train_path: str,
    val_path: str | None = None,
    test_path: str | None = None,
    xgb_params: dict | None = None,
):
    X_train, y_train, X_val, y_val, X_test, y_test, feature_names = load_and_prepare_data(train_path, val_path, test_path)
    """
    def load_data(path):
        if path is None: return None, None
        df = pd.read_parquet(path)
        X = df.drop(columns=["label", "filename"], errors="ignore")
        y = df["label"].map({"real": 0, "fake": 1}).values # type: ignore
        return X, y
    X_train, y_train = load_data(train_path)
    X_val, y_val = load_data(val_path)
    X_test, y_test = load_data(test_path)    
    """



    if X_test is None and X_val is None:
        raise ValueError("At least one of val_path or test_path must be provided.")

    #feature_names = X_train.columns.tolist()
    metadata_extra = {"train_samples": X_train.shape[0]}

    if xgb_params is None: xgb_params = {}
    """    
    if "scale_pos_weight" not in xgb_params:
        num_neg = np.sum(y_train == 0) # Real
        num_pos = np.sum(y_train == 1) # Fake
        if num_pos > 0:
            scale_weight = num_neg / num_pos
            xgb_params["scale_pos_weight"] = scale_weight
            print(f"Scale_pos_weight: {scale_weight:.2f}")
    """


    # default_params = {
    #     "max_depth": 6,
    #     "learning_rate": 0.1,
    #     "subsample": 0.8,
    #     "gamma": 0.0,
    #     "colsample_bytree": 0.7,
    #     "n_jobs": -1,
    #     "verbosity": 2,
    #     "eval_metric": "aucpr",
    # }
    
    default_params = {}
    default_params.update(xgb_params)

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("xgb", XGBClassifier(**default_params)),
    ])

    print(f"Training XGBoost on {X_train.shape[0]} samples...")
    pipeline.named_steps["imputer"].fit(X_train)
    X_train_imp = pipeline.named_steps["imputer"].transform(X_train)

    if X_val is not None:
        X_val_imp = pipeline.named_steps["imputer"].transform(X_val)
        #pipeline.named_steps["xgb"].set_params(early_stopping_rounds=5)
        pipeline.named_steps["xgb"].fit(
            X_train_imp,
            y_train,
            eval_set=[(X_val_imp, y_val)],
        )
    else:
        pipeline.named_steps["xgb"].fit(X_train_imp, y_train)

    def get_metrics(X, y, prefix=""):
        if X is None: return {}
        y_pred = pipeline.predict(X)
        y_probs = pipeline.predict_proba(X)[:, 1]
        p = f"{prefix}_" if prefix else ""
        return {
            f"{p}accuracy": float(accuracy_score(y, y_pred)),
            f"{p}precision": float(precision_score(y, y_pred, average="macro")),
            f"{p}recall": float(recall_score(y, y_pred, average="macro")),
            f"{p}f1": float(f1_score(y, y_pred, average="macro")),
            f"{p}roc_auc": float(roc_auc_score(y, y_probs))
        }

    metrics = {}
    if X_test is not None:
        metadata_extra["test_samples"] = X_test.shape[0]
        metrics.update(get_metrics(X_test, y_test, prefix=""))

    if X_val is not None:
        metadata_extra["val_samples"] = X_val.shape[0]
        prefix = "val" if X_test is not None else "" 
        metrics.update(get_metrics(X_val, y_val, prefix=prefix))

    return pipeline, metrics, default_params, feature_names, metadata_extra