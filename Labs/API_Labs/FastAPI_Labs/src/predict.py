import joblib

CLASS_NAMES = ['setosa', 'versicolor', 'virginica']


def predict_data(X):
    """
    Predict the class labels for the input data.
    Args:
        X (numpy.ndarray): Input data for which predictions are to be made.
    Returns:
        y_pred (numpy.ndarray): Predicted class labels.
    """
    model = joblib.load("../model/iris_model.pkl")
    y_pred = model.predict(X)
    return y_pred


def predict_with_probability(X):
    """
    Predict class labels along with probabilities for the input data.
    Args:
        X (numpy.ndarray): Input data for which predictions are to be made.
    Returns:
        dict: Contains prediction, class_name, and probabilities.
    """
    model = joblib.load("../model/iris_model.pkl")
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)

    return {
        "prediction": int(y_pred[0]),
        "class_name": CLASS_NAMES[y_pred[0]],
        "probabilities": {
            CLASS_NAMES[i]: round(float(y_proba[0][i]), 4)
            for i in range(len(CLASS_NAMES))
        }
    }