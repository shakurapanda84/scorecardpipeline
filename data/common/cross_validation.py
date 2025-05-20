import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, roc_auc_score

def cross_validate_model(model, X, y, n_splits=5, scoring='accuracy'):
    """
    Perform k-fold cross-validation on a given model and dataset.

    Parameters:
    -----------
    model : sklearn-like estimator
        The model to evaluate.
    X : array-like
        Feature matrix.
    y : array-like
        Target vector.
    n_splits : int, default=5
        Number of folds in k-fold cross-validation.
    scoring : str, default='accuracy'
        Scoring metric to use ('accuracy' or 'roc_auc').

    Returns:
    --------
    dict
        A dictionary containing the mean and standard deviation of the scores.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        if scoring == 'accuracy':
            score = accuracy_score(y_test, y_pred)
        elif scoring == 'roc_auc':
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            score = roc_auc_score(y_test, y_pred_proba)
        else:
            raise ValueError("Unsupported scoring method. Use 'accuracy' or 'roc_auc'.")

        scores.append(score)

    return {
        'mean_score': np.mean(scores),
        'std_score': np.std(scores)
    }

# Example usage
if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from sklearn.linear_model import LogisticRegression

    # Load sample data
    data = load_iris()
    X, y = data.data, data.target

    # Initialize model
    model = LogisticRegression(max_iter=200)

    # Perform cross-validation
    results = cross_validate_model(model, X, y, n_splits=5, scoring='accuracy')
    print("Cross-validation results:", results) 