from pathlib import Path
import pandas as pd
from scipy.io import arff


def load_csv(file_path):
    return pd.read_csv(file_path)


def load_arff(file_path):
    data, meta = arff.loadarff(file_path)
    df = pd.DataFrame(data)

    for col in df.select_dtypes([object]).columns:
        df[col] = df[col].apply(
            lambda x: x.decode("utf-8") if isinstance(x, bytes) else x
        )

    return df


def load_dataset(file_path):
    file_path = Path(file_path)
    suffix = file_path.suffix.lower()

    if suffix == ".csv":
        return load_csv(file_path)
    if suffix == ".arff":
        return load_arff(file_path)

    raise ValueError(f"Unsupported file format: {suffix}")