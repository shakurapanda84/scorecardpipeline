import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def plot_auc_curve(y_true, y_scores, title='ROC Curve'):
    """
    Plot the AUC curve for a given set of true labels and predicted scores.

    Parameters:
    -----------
    y_true : array-like
        True binary labels.
    y_scores : array-like
        Target scores, can either be probability estimates of the positive class, 
        confidence values, or non-thresholded measure of decisions.
    title : str, default='ROC Curve'
        Title of the plot.
    """
    # Compute ROC curve and ROC area
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()

# Example usage
if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    # Load sample data
    data = load_iris()
    X, y = data.data, data.target

    # Convert to binary classification problem
    y = (y == 2).astype(int)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train model
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)

    # Predict probabilities
    y_scores = model.predict_proba(X_test)[:, 1]

    # Plot AUC curve
    plot_auc_curve(y_test, y_scores) 