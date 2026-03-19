from sklearn.metrics import accuracy_score

def evaluate(model, X_test, y_test):

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    return accuracy