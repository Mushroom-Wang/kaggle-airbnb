from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def del_irrelevant_columns(df: pd.DataFrame, irrelevant_columns: List[str]):
    return df.drop(irrelevant_columns, axis=1)


def process_non_numeric_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for column in df:
        if not pd.api.types.is_numeric_dtype(df[column].dtype):
            types = list(set(df[column].values))
            try:
                types.sort()
            except Exception:
                import ipdb
                ipdb.set_trace()

            _dict = {t: v for v, t in enumerate(types)}
            df[column] = df[column].apply(lambda t: _dict[t])

    return df


def normalize(df: pd.DataFrame):
    return (df-df.mean())/df.std()


def process_non(df: pd.DataFrame, method="average") -> pd.DataFrame:
    if method == "drop":
        return df.dropna()
    elif method == "average":
        df = df.copy()
        for col in df:
            if pd.api.types.is_numeric_dtype(df[col].dtype):
                df[col] = df[col].fillna(df[col].mean())
            else:
                df[col] = df[col].fillna("NaN")

        return df
    elif method == "repair":
        df = df.copy()
        has_nan_col_list = []
        full_col_list = []

        for col in df:
            if df[col].hasnans:
                has_nan_col_list.append(col)
            else:
                full_col_list.append(col)

        for has_nan_col in has_nan_col_list:
            non_index = df[has_nan_col].isnull().to_numpy()
            features = normalize(process_non_numeric_data(df[full_col_list]))
            x = features[np.logical_not(non_index)]
            y_series = df[has_nan_col][np.logical_not(non_index)]
            y_df = y_series.to_frame(name="y")
            y = process_non_numeric_data(y_df).to_numpy().flatten()

            x_fix = features[non_index]
            linear_reg = LinearRegression()
            linear_reg.fit(x, y)
            pred_y = linear_reg.predict(x_fix)

            fixed_series = np.zeros(len(features))
            fixed_series[np.logical_not(non_index)] = y
            fixed_series[non_index] = pred_y
            df[has_nan_col] = fixed_series

            full_col_list.append(has_nan_col)
            # print(f"repaired {sum(non_index)} {has_nan_col}")

    return df


def preprocess_feature(df: pd.DataFrame,
                       irrelevant_features: List[str] = ["id", "Decision"])\
        -> Tuple[np.ndarray, np.ndarray]:
    x_df = del_irrelevant_columns(df, irrelevant_features)
    # feature engineering
    x_df = process_non(x_df)
    x_df = process_non_numeric_data(x_df)
    x_df = normalize(x_df)

    return x_df.to_numpy()
