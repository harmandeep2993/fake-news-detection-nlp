import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def evaluate_model(
    model,
    X_train_vec,
    X_test_vec,
    y_train,
    y_test,
    config: dict = None
):
    """
    Evaluate trained model and return metrics row.
    """

    # Predictions
    y_train_pred = model.predict(X_train_vec)
    y_test_pred  = model.predict(X_test_vec)

    # Metrics
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy  = accuracy_score(y_test, y_test_pred)

    train_f1 = f1_score(y_train, y_train_pred)
    test_f1  = f1_score(y_test, y_test_pred)

    precision = precision_score(y_test, y_test_pred)
    recall    = recall_score(y_test, y_test_pred)

    row = {
        "model": model.__class__.__name__,
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
        "precision": precision,
        "recall": recall,
        "train_f1": train_f1,
        "test_f1": test_f1,
        "gap_percent": round((train_f1 - test_f1) * 100, 2),
        "config": config
    }

    if config:
        row.update(config)

    return row


def append_result(results_df: pd.DataFrame, row: dict):
    """
    Append evaluation row to results dataframe.
    """
    return pd.concat([results_df, pd.DataFrame([row])], ignore_index=True)