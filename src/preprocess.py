
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def split_data(X, y, test_size=0.30, seed=42):
    # Stratified split ensures the class distribution is preserved in both sets.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size, # fixed test size for fair comparisons across experiments
        stratify=y, # maintain class balance in train/test splits 
        random_state=seed  # fixed seed for reproducibility
    )
    # return raw split to allow the prefilter to fit only on training data, 
    # preventing test set leakage
    return X_train, X_test, y_train, y_test


def scale_data(X_train, X_test):
    # map every feature to the [0, 1] range
    scaler = MinMaxScaler()

    # train the scaler on the training data to learn the min and max for 
    # each feature
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )

    # apply the same transformation to the test data using the parameters learned
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )

    # return scaled data along with the scaler object in case we need to inverse transform later (e.g., for interpreting feature importance)
    return X_train_scaled, X_test_scaled, scaler