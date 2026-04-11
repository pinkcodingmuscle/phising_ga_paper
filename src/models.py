# from sklearn.svm import SVC
# from sklearn.linear_model import LogisticRegression, RidgeClassifier
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# from sklearn.naive_bayes import GaussianNB
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.neural_network import MLPClassifier


# def get_models(seed=42):
#     models = {
#         "SVM": SVC(kernel="rbf", random_state=seed),
#         "Logistic Regression": LogisticRegression(max_iter=1000, random_state=seed),
#         "Decision Tree": DecisionTreeClassifier(random_state=seed),
#         "Random Forest": RandomForestClassifier(random_state=seed),
#         "Gradient Boosting": GradientBoostingClassifier(random_state=seed),
#         "Naive Bayes": GaussianNB(),
#         "KNN": KNeighborsClassifier(),
#         "Ridge Classifier": RidgeClassifier(),
#         "DNN (MLP)": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=seed)
#     }
#     return models