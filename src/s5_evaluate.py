"""Stage 5: score every model once on 2018, explain gradient boosting, make figures.

Input:  outputs/models/final_models.joblib (stage 4), model table (stage 3)
Output: outputs/tables/table3_test_metrics.csv        all five models on 2018
        outputs/tables/table4_ablation.csv            feature-group ablation
        outputs/tables/table4b_ablation_gains.csv     gain per group, paired 95% interval
        outputs/tables/permutation_importance.csv     top features
        outputs/figures/fig2_confusion.png            confusion matrices
        outputs/figures/fig3_importance.png           permutation importance
        outputs/figures/fig4_ablation.png             ablation bars

Noise check: gradient boosting without early stopping is deterministic, so
re-running with other seeds gives identical scores. Instead, every score gets
a 95% bootstrap interval from resampling the 2018 crashes; a gain smaller than
that interval is reported as no measurable gain.
"""

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

import config
from src import s4_train
from src.utils import ensure_dirs, log

N_BOOT = 200
IMPORTANCE_SAMPLE = 20_000
IMPORTANCE_REPEATS = 5


def bootstrap_macro_f1(y, pred, rng) -> tuple[float, float]:
    """95% interval of macro-F1 from resampling test crashes with replacement."""
    n = len(y)
    scores = [f1_score(y[i], pred[i], average="macro")
              for i in (rng.integers(0, n, n) for _ in range(N_BOOT))]
    return float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))


def metrics_row(name, y, pred, rng) -> dict:
    """Headline and per-class metrics for one model on the test year."""
    prec, rec, _, _ = precision_recall_fscore_support(y, pred, labels=[0, 1, 2], zero_division=0)
    lo, hi = bootstrap_macro_f1(y, pred, rng)
    row = {"model": name, "macro_f1": f1_score(y, pred, average="macro"),
           "macro_f1_ci_low": lo, "macro_f1_ci_high": hi,
           "accuracy": accuracy_score(y, pred),
           "ordinal_mae": float(np.mean(np.abs(y - pred)))}
    for k, cls in enumerate(config.CLASS_NAMES):
        row[f"precision_{cls}"] = prec[k]
        row[f"recall_{cls}"] = rec[k]
    return row


def plot_confusions(cms: dict) -> None:
    """Figure 2: confusion matrices normalised by true class, side by side."""
    fig, axes = plt.subplots(1, len(cms), figsize=(5 * len(cms), 4.2))
    for ax, (name, cm) in zip(np.atleast_1d(axes), cms.items()):
        im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center",
                        color="white" if cm[i, j] > 0.5 else "black")
        ax.set_xticks(range(3), config.CLASS_NAMES)
        ax.set_yticks(range(3), config.CLASS_NAMES)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(name)
    fig.colorbar(im, ax=axes, shrink=0.8, label="Share of true class")
    fig.savefig(config.OUT_FIGURES / "fig2_confusion.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def importance(model, X, y) -> pd.DataFrame:
    """Permutation importance of gradient boosting on a 2018 sample, scored by macro-F1."""
    idx = np.random.default_rng(config.SEED).choice(len(X), size=min(IMPORTANCE_SAMPLE, len(X)), replace=False)
    result = permutation_importance(model, X.iloc[idx], y[idx], scoring="f1_macro",
                                    n_repeats=IMPORTANCE_REPEATS, random_state=config.SEED, n_jobs=-1)
    return (pd.DataFrame({"feature": X.columns, "importance_mean": result.importances_mean,
                          "importance_std": result.importances_std})
            .sort_values("importance_mean", ascending=False).reset_index(drop=True))


def plot_importance(imp: pd.DataFrame) -> None:
    """Figure 3: top 15 features by permutation importance, with spread."""
    top = imp.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.barh(top["feature"], top["importance_mean"], xerr=top["importance_std"], color="#3a6ea5")
    ax.set_xlabel("Drop in macro-F1 when shuffled")
    ax.set_title("Permutation importance, gradient boosting (2018)")
    fig.savefig(config.OUT_FIGURES / "fig3_importance.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def ablation(df, best, rng) -> pd.DataFrame:
    """Refit gradient boosting on cumulative feature groups; score each on 2018."""
    rows, features, preds = [], [], []
    hgb = best["gradient_boosting"]
    _, yte = s4_train.split(df, [config.TEST_YEAR], config.FEATURE_GROUPS[config.ABLATION_ORDER[0]])
    for group in config.ABLATION_ORDER:
        features = features + config.FEATURE_GROUPS[group]
        Xtr, ytr = s4_train.split(df, config.TRAIN_YEARS, features)
        Xte, yte = s4_train.split(df, [config.TEST_YEAR], features)
        model = s4_train.make_hgb(hgb, max_iter=hgb["n_iter"], early_stop=False)
        model.fit(Xtr, ytr, sample_weight=s4_train.weights(ytr, hgb["severe_mult"]))
        pred = model.predict(Xte)
        lo, hi = bootstrap_macro_f1(yte, pred, rng)
        rows.append({"step": " + ".join(config.ABLATION_ORDER[:len(rows) + 1]), "n_features": len(features),
                     "macro_f1": f1_score(yte, pred, average="macro"), "macro_f1_ci_low": lo,
                     "macro_f1_ci_high": hi,
                     "severe_recall": float(np.mean(pred[yte == 2] == 2))})
        preds.append(pred)
        log(f"  ablation {rows[-1]['step']}: macro-F1 {rows[-1]['macro_f1']:.4f} "
            f"[{lo:.4f}, {hi:.4f}], severe recall {rows[-1]['severe_recall']:.3f}")
    gains = paired_gains(yte, preds, rng)
    return pd.DataFrame(rows), gains


def paired_gains(y, preds, rng) -> pd.DataFrame:
    """Gain in macro-F1 from each added feature group, with a paired 95% interval.

    The same resampled crashes score both steps, so the interval is for the
    difference itself. If it includes 0, the group gives no measurable gain.
    """
    n, rows = len(y), []
    samples = [rng.integers(0, n, n) for _ in range(N_BOOT)]
    for k in range(1, len(preds)):
        diffs = [f1_score(y[i], preds[k][i], average="macro") - f1_score(y[i], preds[k - 1][i], average="macro")
                 for i in samples]
        gain = f1_score(y, preds[k], average="macro") - f1_score(y, preds[k - 1], average="macro")
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        rows.append({"added_group": config.ABLATION_ORDER[k], "macro_f1_gain": gain,
                     "gain_ci_low": lo, "gain_ci_high": hi, "measurable": bool(lo > 0 or hi < 0)})
        log(f"  adding {config.ABLATION_ORDER[k]}: gain {gain:+.4f} [{lo:+.4f}, {hi:+.4f}]"
            f"{'' if rows[-1]['measurable'] else ' (no measurable gain)'}")
    return pd.DataFrame(rows)


def plot_ablation(table: pd.DataFrame) -> None:
    """Figure 4: macro-F1 (with 95% interval) and severe recall at each ablation step."""
    labels = ["Roadway", "+ Time", "+ Weather", "+ Traffic"][:len(table)]
    x = np.arange(len(table))
    fig, ax = plt.subplots(figsize=(7, 4.2))
    err = [table["macro_f1"] - table["macro_f1_ci_low"], table["macro_f1_ci_high"] - table["macro_f1"]]
    ax.bar(x - 0.2, table["macro_f1"], 0.4, yerr=err, capsize=3, label="Macro-F1", color="#3a6ea5")
    ax.bar(x + 0.2, table["severe_recall"], 0.4, label="Severe recall", color="#d08c2c")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score on 2018")
    ax.set_title("Feature-group ablation, gradient boosting")
    ax.legend()
    fig.savefig(config.OUT_FIGURES / "fig4_ablation.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def run() -> None:
    """Run stage 5: metrics, confusion matrices, importance, ablation."""
    ensure_dirs(config.OUT_FIGURES, config.OUT_TABLES)
    bundle = joblib.load(s4_train.FINAL_MODELS)
    models, features, best = bundle["models"], bundle["features"], bundle["best"]
    df = s4_train.load_table()
    Xte, yte = s4_train.split(df, [config.TEST_YEAR], features)
    rng = np.random.default_rng(config.SEED)
    log(f"  scoring on {config.TEST_YEAR}: {len(yte):,} crashes (used once, never for tuning)")

    rows, preds = [], {}
    for name, model in models.items():
        preds[name] = s4_train.predict(name, model, Xte)
        rows.append(metrics_row(name, yte, preds[name], rng))
    table3 = pd.DataFrame(rows).round(4)
    table3.to_csv(config.OUT_TABLES / "table3_test_metrics.csv", index=False)
    log("  test results:\n" + table3[["model", "macro_f1", "macro_f1_ci_low", "macro_f1_ci_high",
                                      "accuracy", "recall_Severe", "ordinal_mae"]].to_string(index=False))

    comparison = table3[~table3["model"].isin(["Gradient boosting", "Majority class"])]
    best_other = comparison.loc[comparison["macro_f1"].idxmax(), "model"]
    plot_confusions({name: confusion_matrix(yte, preds[name], labels=[0, 1, 2], normalize="true")
                     for name in ["Gradient boosting", best_other]})

    imp = importance(models["Gradient boosting"], Xte, yte)
    imp.round(5).to_csv(config.OUT_TABLES / "permutation_importance.csv", index=False)
    plot_importance(imp)
    log("  top 5 features: " + ", ".join(imp["feature"].head(5)))

    table4, gains = ablation(df, best, rng)
    table4.round(4).to_csv(config.OUT_TABLES / "table4_ablation.csv", index=False)
    gains.round(4).to_csv(config.OUT_TABLES / "table4b_ablation_gains.csv", index=False)
    plot_ablation(table4)


if __name__ == "__main__":
    run()
