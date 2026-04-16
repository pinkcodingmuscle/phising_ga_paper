import pandas as pd
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

def evaluate_final_models(X_train, y_train):
    model_params = {
        'svc': {
            'model': SVC(gamma='auto'),
            'params': {'C': [1, 10, 20], 'kernel': ['rbf', 'linear']}
        },
        'logistic_regression': {
            'model': LogisticRegression(solver='liblinear'), # multi_class removed to fix warning
            'params': {'C': [1, 5, 10]}
        },
        'decision_tree': {
            'model': DecisionTreeClassifier(),
            'params': {'criterion': ['gini', 'entropy'], 'max_depth': [None, 5, 10]}
        }
    }

    scores = []
    for model_name, mp in model_params.items():
        clf = GridSearchCV(mp['model'], mp['params'], cv=5, return_train_score=False)
        clf.fit(X_train, y_train)
        scores.append({
            'model': model_name,
            'best_score': clf.best_score_,
            'best_params': clf.best_params_,
            'best_estimator': clf.best_estimator_
        })
    return pd.DataFrame(scores)