import pandas as pd
from sklearn.feature_selection import mutual_info_classif


def prefilter_features(X_train, X_test, y_train, k=50, seed=42):
    # return all features without filtering if k is larger than the number of 
    # features
    k = min(k, X_train.shape[1])

    # feature importance scores based on mutual information with the 
    # target variable
    mi_scores = mutual_info_classif(X_train, y_train, random_state=seed)

    # build a named Series so we can sort by score while keeping track
    # of which score belongs to which feature name
    mi_series = pd.Series(mi_scores, index=X_train.columns).sort_values(ascending=False)

    # Keep only the top-k features
    selected_features = mi_series.head(k).index.tolist()

    # apply the same column selection to both sets.
    X_train_filtered = X_train[selected_features].copy()
    X_test_filtered = X_test[selected_features].copy()

    # return the filtered datasets, the list of selected features, 
    # and the full mutual information scores for analysis and interpretation
    return X_train_filtered, X_test_filtered, selected_features, mi_series