import argparse
import pandas as pd
from pathlib import Path
import joblib

from .preprocess import transform

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="CSV with columns Sr,n,Class")
    ap.add_argument("--model-dir", default="models")
    ap.add_argument("--out", default="predictions.csv")
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    model = joblib.load(model_dir / "Thermal_conductivity_model_ANN_ASTN.joblib")
    num_scaler = joblib.load(model_dir / "numeric_transformer_ASTN.joblib")
    lab = joblib.load(model_dir / "categorical_transformer_ASTN.joblib")

    df = pd.read_csv(args.csv)
    Xp, _ = transform(df, num_scaler, lab, num_cols=("Sr", "n"), cat_col="Class")
    df["l_pred"] = model.predict(Xp)
    df.to_csv(args.out, index=False)
    print(f"Saved predictions → {args.out}")

if __name__ == "__main__":
    main()
