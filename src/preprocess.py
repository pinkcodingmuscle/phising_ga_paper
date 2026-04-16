
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


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
    # compute mean and std on training data, then apply to both train and test 
    # sets to prevent data leakage
    mean = X_train.mean()
    std = X_train.std()

    # avoid division by zero in case of constant features
    std_replaced = std.replace(0, 1)

    # apply z-score normalization: (X - mean) / std
    X_train_scaled = (X_train - mean) / std_replaced
    X_test_scaled = (X_test - mean) / std_replaced

    # train the scaler on the training data to learn the mean and standard deviation for 
    # each feature
    X_train_scaled = pd.DataFrame(
        X_train_scaled,
        columns=X_train.columns,
        index=X_train.index
    )

    # apply the same transformation to the test data using the parameters learned
    X_test_scaled = pd.DataFrame(
        X_test_scaled,
        columns=X_test.columns,
        index=X_test.index
    )

    # return scaled data along with the mean and standard deviation used for scaling
    return X_train_scaled, X_test_scaled, mean, std_replaced