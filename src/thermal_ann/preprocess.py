import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import joblib
from pathlib import Path


ORDER_MAP = {
    1: 20, 2: 80, 3: 10, 4: 30, 5: 40, 6: 60, 7: 50, 8: 70
}

def fit_transformers(df: pd.DataFrame, num_cols=("Sr", "n"), cat_col="Class"):
    num_scaler = MinMaxScaler().fit(df[list(num_cols)].values)
    x_num = num_scaler.transform(df[list(num_cols)].values)

    lab = LabelEncoder().fit(df[cat_col])
    x_cat = lab.transform(df[cat_col])

    # reproduce your category remapping + /10 scaling
    x_cat = np.vectorize(lambda z: ORDER_MAP.get(z, z))(x_cat)
    x_cat = x_cat / 10.0
    x_proc = np.concatenate((x_num, x_cat.reshape(len(x_cat), 1)), axis=1)

    return x_proc, num_scaler, lab, x_cat

def transform(df: pd.DataFrame, num_scaler: MinMaxScaler, lab: LabelEncoder, num_cols=("Sr", "n"), cat_col="Class"):
    x_num = num_scaler.transform(df[list(num_cols)].values)
    x_cat = lab.transform(df[cat_col])
    x_cat = np.vectorize(lambda z: ORDER_MAP.get(z, z))(x_cat)
    x_cat = x_cat / 10.0
    x_proc = np.concatenate((x_num, x_cat.reshape(len(x_cat), 1)), axis=1)
    return x_proc, x_cat

def save_transformers(num_scaler, lab, outdir="models"):
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(num_scaler, out / "numeric_transformer_ASTN.joblib", compress=True)
    joblib.dump(lab,        out / "categorical_transformer_ASTN.joblib", compress=True)
