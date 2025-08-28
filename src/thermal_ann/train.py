import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import joblib

from .data import load_excel
from .preprocess import fit_transformers, transform, save_transformers
from .model import grid_search_mlpr
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error

def rmse(y_true, y_pred):
    # Works even when sklearn doesn't support squared=False
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def parity_plot(y_true, y_pred, out_path: Path, title="Actual vs ANN thermal conductivities"):
    plt.figure()
    plt.plot(y_true, y_pred, 'rx', markersize=6)
    x = np.linspace(min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max()), 10)
    plt.plot(x, x, color="black", linewidth=1)
    plt.xlabel("Thermal conductivities")
    plt.ylabel("ANN conductivities")
    plt.title(title)
    plt.legend(["ANN vs Actual", "Y = X"])
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()

def per_class_plots(df_out: pd.DataFrame, out_path: Path):
    # 3x3 grid as in your notebook
    x = np.linspace(0, 5, 10)
    fig, axes = plt.subplots(3, 3, figsize=(16, 30))
    classes = sorted(df_out["Class"].unique())
    for ax, c in zip(axes.ravel(), classes):
        sub = df_out[df_out["Class"] == c]
        ax.plot(sub["l_actual"], sub["l_pred"], 'rx', markersize=5)
        ax.plot(x, x, color="black", linewidth=0.5)
        ax.set_xlabel("Thermal conductivities")
        ax.set_ylabel("ANN conductivities")
        ax.set_title(f"Actual vs ANN for class {c}")
        ax.legend(["ANN vs Actual", "Y = X"])
    plt.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--excel", required=True, help="Path to ASTM classification.xlsx")
    ap.add_argument("--target", default="l")
    ap.add_argument("--num-cols", nargs="+", default=["Sr", "n"])
    ap.add_argument("--cat-col", default="Class")
    ap.add_argument("--outdir", default="models")
    ap.add_argument("--results", default="results")
    ap.add_argument("--test-size", type=float, default=0.2)
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    resdir = Path(args.results); resdir.mkdir(parents=True, exist_ok=True)

    df = load_excel(args.excel)
    Xp, num_scaler, lab, x_cat_scaled = fit_transformers(df, num_cols=tuple(args.num_cols), cat_col=args.cat_col)

    # optional heatmap correlation (reproduces your seaborn plot)
    try:
        import seaborn as sns
        df_corr = df.copy()
        df_corr[args.cat_col] = x_cat_scaled
        plt.figure(figsize=(15, 10))
        sns.heatmap(df_corr.corr(numeric_only=True), annot=True)
        plt.tight_layout()
        plt.savefig(resdir / "corr_heatmap.png", dpi=180)
        plt.close()
    except Exception:
        pass

    y = df[args.target].values
    x_train, x_test, y_train, y_test = train_test_split(Xp, y, test_size=args.test_size, random_state=42)

    gs = grid_search_mlpr(x_train, y_train)
    best = gs.best_estimator_

    # metrics
    y_train_pred = best.predict(x_train)
    y_test_pred  = best.predict(x_test)
    metrics = {
        "best_params": gs.best_params_,
        "r2_train": float(r2_score(y_train, y_train_pred)),
        "r2_test":  float(r2_score(y_test,  y_test_pred)),
        "rmse_train": rmse(y_train, y_train_pred),
        "rmse_test":  rmse(y_test,  y_test_pred),
    }
    (resdir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # plots (parity + per-class)
    parity_plot(np.concatenate([y_train, y_test]),
                np.concatenate([y_train_pred, y_test_pred]),
                resdir / "parity.png")

    # build output frame for per-class plots & RMSE printouts
    x_all_cat = np.concatenate([x_train[:, 2], x_test[:, 2]])
    df_out = pd.DataFrame({"Class": x_all_cat,
                           "l_actual": np.concatenate([y_train, y_test]),
                           "l_pred":   np.concatenate([y_train_pred, y_test_pred])})
    per_class_plots(df_out, resdir / "per_class.png")

    # per-class RMSEs (like your printouts)
    class_rmses = {}
    for c in sorted(df_out["Class"].unique()):
        sub = df_out[df_out["Class"] == c]
        class_rmses[str(c)] = float(rmse(sub["l_actual"], sub["l_pred"]))
    (resdir / "per_class_rmse.json").write_text(json.dumps(class_rmses, indent=2))

    # save artifacts (same filenames you used, but in models/)
    joblib.dump(best, outdir / "Thermal_conductivity_model_ANN_ASTN.joblib", compress=True)
    save_transformers(num_scaler, lab, outdir=outdir)

    print(json.dumps(metrics, indent=2))
    print("Per-class RMSE:", json.dumps(class_rmses, indent=2))

if __name__ == "__main__":
    main()
