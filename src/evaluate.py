# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# def evaluate_model(name, model, X_train, X_test, y_train, y_test):
#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)

#     return {
#         "Model": name,
#         "Accuracy": accuracy_score(y_test, y_pred),
#         "Precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
#         "Recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
#         "F1": f1_score(y_test, y_pred, average="weighted", zero_division=0)
#     }