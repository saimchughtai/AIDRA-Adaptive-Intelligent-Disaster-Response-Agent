"""
ml_models.py
============
AIDRA – Adaptive Intelligent Disaster Response Agent
AIC-201 CCP | Module 5: Machine Learning Risk & Survival Estimation

Three classifiers are trained to predict binary victim survival
(1 = survives, 0 = does not survive) given five environmental features:

  Feature 1 : severity_code      – 1 (minor) / 2 (moderate) / 3 (critical)
  Feature 2 : distance           – A* path length to medical centre
  Feature 3 : risk_zone_exposure – risk cells along path (0–10)
  Feature 4 : wait_time          – estimated steps before rescue begins
  Feature 5 : kit_available      – 1 if a medical kit is reachable, else 0

Models
  • k-Nearest Neighbours (kNN, k=5)
  • Gaussian Naïve Bayes
  • Multi-Layer Perceptron (MLP, 32→16 hidden, ReLU, Adam)

The best model (highest F1) is selected as the active inference model
and called by the Controller for each victim at rescue time.
"""

import numpy as np
from sklearn.neighbors        import KNeighborsClassifier
from sklearn.naive_bayes      import GaussianNB
from sklearn.neural_network   import MLPClassifier
from sklearn.metrics          import (accuracy_score, precision_score,
                                      recall_score, f1_score, confusion_matrix)
from sklearn.model_selection  import train_test_split
from environment              import SEVERITY, log

# ─────────────────────────────────────────────────────────────────────────────
# DATASET GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def generate_dataset(n_samples: int = 400, seed: int = 42) -> tuple:
    """
    Generate a synthetic survival dataset.

    Scoring function
    ----------------
    s = 0.9 - 0.10·sev - 0.02·dist - 0.08·risk - 0.03·wait + 0.05·kit + ε
    label = 1 (survives) if s > 0.4 else 0

    ε ~ N(0, 0.05) introduces realistic noise.
    Resulting class split ≈ 75% positive (survives).

    Returns
    -------
    X : ndarray (n_samples, 5)
    y : ndarray (n_samples,) — binary labels
    """
    rng = np.random.default_rng(seed)

    sev   = rng.integers(1, 4,  n_samples)          # 1,2,3
    dist  = rng.integers(1, 20, n_samples)           # 1–19 path steps
    risk  = rng.integers(0, 3,  n_samples)           # 0,1,2 risk cells
    wait  = rng.integers(0, 15, n_samples)           # 0–14 steps
    kit   = rng.integers(0, 2,  n_samples)           # 0 or 1
    noise = rng.normal(0, 0.05, n_samples)

    score = (0.90
             - 0.10 * sev
             - 0.02 * dist
             - 0.08 * risk
             - 0.03 * wait
             + 0.05 * kit
             + noise)

    y = (score > 0.40).astype(int)
    X = np.column_stack([sev, dist, risk, wait, kit])

    log(f"Dataset generated: {n_samples} records, "
        f"pos={int(y.sum())} ({100*y.mean():.0f}%), "
        f"neg={int((1-y).sum())} ({100*(1-y).mean():.0f}%)")

    return X, y


# ─────────────────────────────────────────────────────────────────────────────
# MODEL DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────

def build_models() -> dict:
    """Return a fresh dictionary of untrained model instances."""
    return {
        'kNN (k=5)':   KNeighborsClassifier(n_neighbors=5),
        'Naive Bayes': GaussianNB(),
        'MLP (32-16)': MLPClassifier(
            hidden_layer_sizes=(32, 16),
            activation='relu',
            solver='adam',
            max_iter=500,
            random_state=42,
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING AND EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def train_and_evaluate(
    n_samples: int = 400,
    test_size: float = 0.20,
    seed: int = 42,
) -> tuple[dict, dict]:
    """
    Train all three classifiers and evaluate on a stratified hold-out set.

    Parameters
    ----------
    n_samples : number of synthetic training records
    test_size : fraction reserved for testing (default 20 %)
    seed      : random seed for reproducibility

    Returns
    -------
    trained_models : {name: fitted_sklearn_model}
    metrics        : {name: {accuracy, precision, recall, f1, cm, ...}}
    """
    X, y = generate_dataset(n_samples, seed)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    models        = build_models()
    trained       = {}
    metrics       = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc  = round(accuracy_score(y_test, y_pred),                           4)
        prec = round(precision_score(y_test, y_pred, zero_division=0),         4)
        rec  = round(recall_score(y_test, y_pred, zero_division=0),            4)
        f1   = round(f1_score(y_test, y_pred, zero_division=0),                4)
        cm   = confusion_matrix(y_test, y_pred).tolist()

        metrics[name] = {
            'accuracy':  acc,
            'precision': prec,
            'recall':    rec,
            'f1':        f1,
            'cm':        cm,
        }
        trained[name] = model

        log(f"ML [{name}] → Acc={acc}, Prec={prec}, Rec={rec}, F1={f1}")

    return trained, metrics


# ─────────────────────────────────────────────────────────────────────────────
# MODEL SELECTION
# ─────────────────────────────────────────────────────────────────────────────

def select_best_model(trained_models: dict, metrics: dict) -> tuple[object, str]:
    """
    Select the model with the highest F1-score.

    F1 is preferred over accuracy because:
      • The dataset is class-imbalanced (~75 % positive).
      • False negatives (missed high-risk victims) are safety-critical.
      • F1 balances precision and recall symmetrically.

    Returns
    -------
    (best_model_object, best_model_name)
    """
    best_name  = max(metrics, key=lambda n: metrics[n]['f1'])
    best_model = trained_models[best_name]
    log(f"Best ML model selected: {best_name} (F1={metrics[best_name]['f1']})")
    return best_model, best_name


# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────────────────────────────────────────

def predict_survival(
    model,
    severity_str: str,
    distance:     int,
    risk_exposure:int,
    wait_time:    int,
    kit_available:int = 1,
) -> float:
    """
    Predict survival probability for a single victim.

    Parameters
    ----------
    model         : fitted sklearn classifier with predict_proba
    severity_str  : 'critical' | 'moderate' | 'minor'
    distance      : A* path length (steps)
    risk_exposure : risk cells along path
    wait_time     : estimated waiting steps before rescue begins
    kit_available : 1 if a kit is available, else 0

    Returns
    -------
    survival_probability : float in [0.0, 1.0]
    """
    sev  = SEVERITY.get(severity_str, 2)
    feat = np.array([[sev, distance, risk_exposure, wait_time, kit_available]])

    if hasattr(model, 'predict_proba'):
        prob = model.predict_proba(feat)[0][1]
    else:
        # Fallback for models without predict_proba
        prob = float(model.predict(feat)[0])

    return round(float(prob), 4)


def classify_risk(survival_prob: float) -> str:
    """
    Translate survival probability into a human-readable risk category.

    ≥ 0.70 → low risk
    0.40–0.69 → moderate risk
    < 0.40 → high risk (escalate priority)
    """
    if survival_prob >= 0.70:
        return 'low'
    elif survival_prob >= 0.40:
        return 'moderate'
    else:
        return 'high'


# ─────────────────────────────────────────────────────────────────────────────
# FULL PIPELINE (called by Controller)
# ─────────────────────────────────────────────────────────────────────────────

def run_ml_pipeline() -> tuple[object, str, dict]:
    """
    Convenience function: train, evaluate, select best model, return all.

    Returns
    -------
    (best_model, best_model_name, all_metrics)
    """
    log("ML pipeline: training and evaluating models…")
    trained_models, metrics = train_and_evaluate()
    best_model, best_name   = select_best_model(trained_models, metrics)
    return best_model, best_name, metrics, trained_models