def preprocess(train, test):

    # Features (WiFi signals)
    X_train = train.iloc[:, :-5]
    y_train = train["FLOOR"]

    X_test = test.iloc[:, :-5]
    y_test = test["FLOOR"]

    # Replace missing signals
    X_train = X_train.replace(100, -110)
    X_test = X_test.replace(100, -110)

    return X_train, X_test, y_train, y_test