"""Stage 4: train the four comparison models and gradient boosting.

Input:  data/processed/model_table.parquet (stage 3)
Output: outputs/models/best_params.json      chosen settings (committed to git)
        outputs/models/final_models.joblib   models refit on 2014-2017
        outputs/tables/validation_scores.csv macro-F1 per model per fold

Default run: load best_params.json and refit (fast, reproducible).
With --tune: rerun the searches on the three time-ordered folds and rewrite
best_params.json. Scoring is macro-F1 throughout.
"""

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight

import config
from src.s3_features import all_features, assert_no_leakage
from src.utils import ensure_dirs, log

MODEL_NAMES = ["Majority class", "Logistic regression", "Decision tree", "Random forest", "Gradient boosting"]
FINAL_MODELS = config.OUT_MODELS / "final_models.joblib"
SEVERE_MULTIPLIERS = [1, 2, 4, 8]
N_SEARCH = 30


# --- Data helpers --------------------------------------------------------------

def load_table() -> pd.DataFrame:
    """Read the model table written by stage 3."""
    return pd.read_parquet(config.MODEL_TABLE)


def split(df: pd.DataFrame, years, features):
    """Rows for the given years -> (X, y)."""
    part = df[df["year"].isin(years)]
    return part[features], part["label"].to_numpy()


def cat_cols(features):
    """The categorical features within a feature list."""
    return [f for f in features if f in config.CATEGORICAL_FEATURES]


def num_cols(features):
    """The numeric features within a feature list."""
    return [f for f in features if f not in config.CATEGORICAL_FEATURES]


def as_object(X: pd.DataFrame) -> pd.DataFrame:
    """Categoricals -> plain strings with 'missing', for scikit-learn encoders.

    A fixed transformation (no fitting), so it cannot leak test information.
    """
    X = X.copy()
    for c in cat_cols(X.columns):
        X[c] = X[c].astype(object).where(X[c].notna(), "missing")
    return X


def weights(y: np.ndarray, severe_mult: float = 1.0) -> np.ndarray:
    """Balanced sample weights, with an extra multiplier on the severe class."""
    w = compute_sample_weight("balanced", y)
    return np.where(y == 2, w * severe_mult, w)


def macro_f1(y_true, y_pred) -> float:
    """Macro-averaged F1 over the three classes (the project's headline metric)."""
    return f1_score(y_true, y_pred, average="macro")


# --- Model builders ------------------------------------------------------------

def make_logreg(features):
    """Multinomial logistic regression: one-hot categoricals, scaled numerics."""
    prep = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)),
                          ("scale", StandardScaler())]), num_cols(features)),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20), cat_cols(features)),
    ])
    return Pipeline([("prep", prep),
                     ("model", LogisticRegression(class_weight="balanced", max_iter=2000, C=1.0))])


def tree_prep(features):
    """Ordinal codes plus median imputation with missing indicators (tree and forest)."""
    return ColumnTransformer([
        ("num", SimpleImputer(strategy="median", add_indicator=True), num_cols(features)),
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1,
                               encoded_missing_value=-1), cat_cols(features)),
    ])


def make_tree(features, max_depth):
    """Decision tree with balanced class weights."""
    return Pipeline([("prep", tree_prep(features)),
                     ("model", DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=50,
                                                      class_weight="balanced", random_state=config.SEED))])


def make_forest(features, min_samples_leaf, seed=config.SEED):
    """Random forest, 300 trees, balanced class weights per bootstrap sample."""
    return Pipeline([("prep", tree_prep(features)),
                     ("model", RandomForestClassifier(n_estimators=300, min_samples_leaf=min_samples_leaf,
                                                      class_weight="balanced_subsample", n_jobs=-1,
                                                      random_state=seed))])


def make_hgb(params: dict, seed=config.SEED, max_iter=None, early_stop=True):
    """Gradient boosting; reads pandas categoricals and missing values directly."""
    p = {k: v for k, v in params.items() if k not in ("n_iter", "severe_mult")}
    return HistGradientBoostingClassifier(
        **p, max_iter=max_iter or 1000, early_stopping=early_stop, n_iter_no_change=30,
        categorical_features="from_dtype", random_state=seed)


def uses_object_input(name: str) -> bool:
    """scikit-learn encoders need plain strings; gradient boosting takes categoricals."""
    return name in ("Logistic regression", "Decision tree", "Random forest")


# --- Tuning --------------------------------------------------------------------

def sample_hgb_params(rng: np.random.Generator) -> dict:
    """One random gradient boosting configuration from the plan's search ranges."""
    return {
        "learning_rate": float(np.exp(rng.uniform(np.log(0.03), np.log(0.3)))),
        "max_leaf_nodes": int(rng.integers(15, 128)),
        "max_depth": [None, 4, 6, 8, 10][int(rng.integers(0, 5))],
        "min_samples_leaf": int(rng.integers(20, 201)),
        "l2_regularization": float(rng.uniform(0, 10)),
    }


def fit_hgb_fold(df, features, params, train_years, val_year, severe_mult=1.0):
    """Fit gradient boosting on one fold, early-stopping on the validation year."""
    Xtr, ytr = split(df, train_years, features)
    Xva, yva = split(df, [val_year], features)
    model = make_hgb(params)
    model.fit(Xtr, ytr, sample_weight=weights(ytr, severe_mult),
              X_val=Xva, y_val=yva, sample_weight_val=weights(yva, severe_mult))
    pred = model.predict(Xva)
    return macro_f1(yva, pred), recall_score(yva, pred, labels=[2], average="macro"), model.n_iter_


def cv_score(df, features, build, object_input):
    """Mean macro-F1 of a scikit-learn model over the three time-ordered folds."""
    scores = []
    for train_years, val_year in config.CV_FOLDS:
        Xtr, ytr = split(df, train_years, features)
        Xva, yva = split(df, [val_year], features)
        if object_input:
            Xtr, Xva = as_object(Xtr), as_object(Xva)
        model = build()
        model.fit(Xtr, ytr)
        scores.append(macro_f1(yva, model.predict(Xva)))
    return float(np.mean(scores)), scores


def tune(df, features) -> dict:
    """Search every model's settings on the time-ordered folds. Returns best_params."""
    best = {}

    depth_scores = {d: cv_score(df, features, lambda d=d: make_tree(features, d), True)[0]
                    for d in range(4, 13, 2)}
    best["decision_tree"] = {"max_depth": max(depth_scores, key=depth_scores.get)}
    log(f"  decision tree depth scores: { {d: round(s, 4) for d, s in depth_scores.items()} }")

    leaf_scores = {m: cv_score(df, features, lambda m=m: make_forest(features, m), True)[0]
                   for m in (5, 20, 50)}
    best["random_forest"] = {"min_samples_leaf": max(leaf_scores, key=leaf_scores.get)}
    log(f"  random forest leaf scores: { {m: round(s, 4) for m, s in leaf_scores.items()} }")

    rng = np.random.default_rng(config.SEED)
    results = []
    for i in range(N_SEARCH):
        params = sample_hgb_params(rng)
        folds = [fit_hgb_fold(df, features, params, tr, va) for tr, va in config.CV_FOLDS]
        score = float(np.mean([f[0] for f in folds]))
        results.append((score, params, [f[2] for f in folds]))
        log(f"  gradient boosting search {i + 1}/{N_SEARCH}: macro-F1 {score:.4f}")
    score, params, iters = max(results, key=lambda r: r[0])
    log(f"  best gradient boosting macro-F1 {score:.4f}: {params}")

    # Severe-class multiplier, chosen on the 2017 validation fold.
    last_train, last_val = config.CV_FOLDS[-1]
    mult_rows = []
    for m in SEVERE_MULTIPLIERS:
        f1, sev_recall, n_iter = fit_hgb_fold(df, features, params, last_train, last_val, m)
        mult_rows.append({"severe_mult": m, "macro_f1": round(f1, 4), "severe_recall": round(sev_recall, 4),
                          "n_iter": int(n_iter)})
        log(f"  severe weight x{m}: macro-F1 {f1:.4f}, severe recall {sev_recall:.4f}")
    mult_table = pd.DataFrame(mult_rows)
    mult_table.to_csv(config.OUT_TABLES / "severe_weight_choice.csv", index=False)
    chosen = mult_table.loc[mult_table["macro_f1"].idxmax()]

    best["gradient_boosting"] = {**params, "severe_mult": int(chosen["severe_mult"]),
                                 "n_iter": int(chosen["n_iter"])}
    return best


# --- Final models --------------------------------------------------------------

def build_all(features, best, seed=config.SEED):
    """Every model, unfitted, keyed by display name."""
    hgb = best["gradient_boosting"]
    return {
        "Majority class": DummyClassifier(strategy="most_frequent"),
        "Logistic regression": make_logreg(features),
        "Decision tree": make_tree(features, best["decision_tree"]["max_depth"]),
        "Random forest": make_forest(features, best["random_forest"]["min_samples_leaf"], seed),
        "Gradient boosting": make_hgb(hgb, seed=seed, max_iter=hgb["n_iter"], early_stop=False),
    }


def fit_model(name, model, X, y, best):
    """Fit one model on (X, y) with the right input format and weights."""
    if name == "Gradient boosting":
        model.fit(X, y, sample_weight=weights(y, best["gradient_boosting"]["severe_mult"]))
    else:
        model.fit(as_object(X) if uses_object_input(name) else X, y)
    return model


def predict(name, model, X):
    """Predict with the right input format for the model."""
    return model.predict(as_object(X) if uses_object_input(name) else X)


def validation_table(df, features, best) -> pd.DataFrame:
    """Macro-F1 of every model on each validation year, with the chosen settings."""
    rows = []
    for train_years, val_year in config.CV_FOLDS:
        Xtr, ytr = split(df, train_years, features)
        Xva, yva = split(df, [val_year], features)
        for name, model in build_all(features, best).items():
            fit_model(name, model, Xtr, ytr, best)
            rows.append({"model": name, "validation_year": val_year,
                         "macro_f1": round(macro_f1(yva, predict(name, model, Xva)), 4)})
    return pd.DataFrame(rows).pivot(index="model", columns="validation_year", values="macro_f1").loc[MODEL_NAMES]


def run(tune_params: bool = False) -> dict:
    """Run stage 4: tune if asked (or if no saved settings), then refit on 2014-2017."""
    ensure_dirs(config.OUT_MODELS, config.OUT_TABLES)
    features = all_features()
    assert_no_leakage(features)
    df = load_table()

    if tune_params or not config.BEST_PARAMS.exists():
        log("  tuning on the time-ordered folds (this takes a few minutes)")
        best = tune(df, features)
        config.BEST_PARAMS.write_text(json.dumps(best, indent=2))
    else:
        best = json.loads(config.BEST_PARAMS.read_text())
        log(f"  using saved settings from {config.BEST_PARAMS.relative_to(config.ROOT)}")

    val = validation_table(df, features, best)
    val.to_csv(config.OUT_TABLES / "validation_scores.csv")
    log("  validation macro-F1 by year:\n" + val.to_string())

    Xtr, ytr = split(df, config.TRAIN_YEARS, features)
    models = {name: fit_model(name, m, Xtr, ytr, best) for name, m in build_all(features, best).items()}
    joblib.dump({"models": models, "features": features, "best": best}, FINAL_MODELS)
    log(f"  refit all five models on {config.TRAIN_YEARS[0]}-{config.TRAIN_YEARS[-1]} "
        f"({len(ytr):,} crashes)")
    return {"models": models, "features": features, "best": best}


if __name__ == "__main__":
    import sys
    run(tune_params="--tune" in sys.argv)
