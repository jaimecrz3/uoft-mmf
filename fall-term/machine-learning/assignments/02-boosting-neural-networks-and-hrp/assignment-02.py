"""MMF2030H - Machine Learning, Assignment 2

Author: Jaime Corzo Galdó

Part B: XGBoost, neural networks and portfolio allocation.
Input files: mmf2030h_a2_credit_data.csv, mmf2030h_a2_prices.csv
Outputs are saved to the assignment2_outputs folder.
"""

from pathlib import Path
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, leaves_list, linkage
from scipy.optimize import minimize
from scipy.spatial.distance import squareform
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from xgboost import DMatrix, XGBClassifier
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "assignment2_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
SEED = 42


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_credit_data():
    path = BASE_DIR / "mmf2030h_a2_credit_data.csv"
    df = pd.read_csv(path)
    target = "serious_dlqin_2yrs"
    features = [c for c in df.columns if c not in (target, "split")]

    assert len(features) == 10
    assert set(df["split"].unique()) == {"train", "test"}
    assert not df[features + [target, "split"]].isna().any().any()

    train = df[df["split"] == "train"].copy()
    test = df[df["split"] == "test"].copy()
    print(f"Credit data: {len(train)} training rows, {len(test)} test rows")
    return train, test, features, target


# B1: XGBoost and feature importance

def run_b1(train, test, features, target):
    print("\nB1 - XGBoost and SHAP")
    X_train, y_train = train[features], train[target]
    X_test, y_test = test[features], test[target]

    # A minimum Hessian sum per child helps avoid very small leaves.
    model = XGBClassifier(
        n_estimators=250,
        max_depth=3,
        min_child_weight=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        importance_type="gain",
        random_state=SEED,
        n_jobs=1,
    )
    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, probabilities)
    print(f"Test AUC: {auc:.6f}")

    gain = pd.Series(model.feature_importances_, index=features)

    # XGBoost returns one SHAP contribution per feature plus a baseline.
    booster = model.get_booster()
    dtest = DMatrix(X_test, feature_names=features)
    contributions = booster.predict(dtest, pred_contribs=True)
    shap_values = pd.DataFrame(contributions[:, :-1], index=X_test.index, columns=features)
    base_values = contributions[:, -1]

    # SHAP contributions add up to the raw model output (log-odds).
    margins = booster.predict(dtest, output_margin=True)
    assert np.max(np.abs(contributions.sum(axis=1) - margins)) < 1e-4

    individual_shap = shap_values.add_prefix("shap_").copy()
    individual_shap.insert(0, "original_row_index", X_test.index)
    individual_shap["base_log_odds"] = base_values
    individual_shap["predicted_probability"] = probabilities
    individual_shap["actual_default"] = y_test.to_numpy()
    individual_shap.to_csv(OUTPUT_DIR / "B1_SHAP_values.csv", index=False)

    # Use absolute SHAP values so positive and negative effects do not cancel.
    mean_shap = shap_values.abs().mean()
    importance = pd.DataFrame({
        "Gain (relative %)": 100 * gain / gain.sum(),
        "SHAP (relative %)": 100 * mean_shap / mean_shap.sum(),
        "Mean abs SHAP (log-odds)": mean_shap,
    })
    importance.to_csv(OUTPUT_DIR / "B1_feature_importance.csv", index_label="feature")
    print(importance.sort_values("SHAP (relative %)", ascending=False).round(2))
    corr = X_train["revolving_utilization"].corr(X_train["debt_ratio"])
    print(f"Utilization/debt ratio correlation (train): {corr:.3f}")

    order = importance.sort_values("SHAP (relative %)").index
    fig, axes = plt.subplots(1, 2, sharey=True, figsize=(14, 6))
    axes[0].barh(order, importance.loc[order, "Gain (relative %)"])
    axes[1].barh(order, importance.loc[order, "SHAP (relative %)"])
    axes[0].set_title("XGBoost gain (MDI-style)")
    axes[1].set_title("Mean absolute TreeSHAP")
    for ax in axes:
        ax.set_xlabel("Relative importance (%)")
        ax.grid(axis="x", alpha=0.2)
        ax.set_axisbelow(True)
    axes[1].tick_params(axis="y", labelleft=False)
    fig.suptitle("B1 - Feature importance comparison")
    fig.subplots_adjust(left=0.25, right=0.97, top=0.87, bottom=0.13, wspace=0.13)
    fig.savefig(OUTPUT_DIR / "B1_MDI_SHAP_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return auc


# B2: Neural network architectures and L2 regularization

class FeedForward(nn.Module):
    def __init__(self, input_size, hidden_sizes):
        super().__init__()
        sizes = [input_size] + list(hidden_sizes) + [1]
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            if i < len(sizes) - 2:
                layers.append(nn.ReLU())
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x).squeeze(-1)  # Raw logits for BCEWithLogitsLoss


def train_network(X_fit, y_fit, X_val, y_val, hidden_sizes, weight_decay, epochs=100):
    set_seed()
    model = FeedForward(X_fit.shape[1], hidden_sizes)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=weight_decay)
    loss_function = nn.BCEWithLogitsLoss()

    dataset = TensorDataset(
        torch.tensor(X_fit, dtype=torch.float32),
        torch.tensor(y_fit, dtype=torch.float32),
    )
    loader = DataLoader(dataset, batch_size=128, shuffle=True,
                        generator=torch.Generator().manual_seed(SEED))
    x_train = torch.tensor(X_fit, dtype=torch.float32)
    y_train = torch.tensor(y_fit, dtype=torch.float32)
    x_val = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
    history = {"train": [], "val": []}

    for epoch in range(epochs):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = loss_function(model(xb), yb)
            loss.backward()
            optimizer.step()

        # Track the same unpenalized loss for train and validation.
        model.eval()
        with torch.no_grad():
            history["train"].append(float(loss_function(model(x_train), y_train)))
            history["val"].append(float(loss_function(model(x_val), y_val_tensor)))
    return model, history


def run_b2(train, test, features, target, xgboost_auc):
    print("\nB2 - Neural networks")

    # Set aside part of the official training set for validation.
    cutoff = int(0.8 * len(train))
    fit, val = train.iloc[:cutoff], train.iloc[cutoff:]
    scaler = StandardScaler().fit(fit[features])
    X_fit = scaler.transform(fit[features]).astype("float32")
    X_val = scaler.transform(val[features]).astype("float32")
    X_test = scaler.transform(test[features]).astype("float32")
    y_fit = fit[target].to_numpy(dtype="float32")
    y_val = val[target].to_numpy(dtype="float32")
    y_test = test[target].to_numpy()
    print(f"Fit: {len(fit)}, validation: {len(val)}, test: {len(test)}")

    architectures = {
        "Shallow-wide (64)": [64],
        "Deep-narrow (32-16-8)": [32, 16, 8],
    }
    regularization = {"No regularization": 0.0, "With L2": 0.001}
    histories = {}
    records = []

    for name, hidden_sizes in architectures.items():
        for reg_name, decay in regularization.items():
            network, history = train_network(X_fit, y_fit, X_val, y_val, hidden_sizes, decay)
            histories[(name, reg_name)] = history
            network.eval()
            with torch.no_grad():
                logits = network(torch.tensor(X_test, dtype=torch.float32))
                probabilities = torch.sigmoid(logits).numpy()
            auc = roc_auc_score(y_test, probabilities)
            records.append({
                "Architecture": name,
                "Regularization": reg_name,
                "Test AUC": auc,
                "Validation final loss": history["val"][-1],
            })
            print(f"{name}: {reg_name}, test AUC = {auc:.6f}")

    results = pd.DataFrame(records)
    results.to_csv(OUTPUT_DIR / "B2_model_results.csv", index=False)

    # Choose the architecture for loss plots using validation loss, not test AUC.
    unregularized = results[results["Regularization"] == "No regularization"]
    chosen = unregularized.sort_values("Validation final loss").iloc[0]["Architecture"]
    print(f"Architecture shown in loss plots: {chosen}")

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), sharey=True)
    for ax, reg_name in zip(axes, regularization):
        history = histories[(chosen, reg_name)]
        epochs = np.arange(1, len(history["train"]) + 1)
        ax.plot(epochs, history["train"], label="Train loss")
        ax.plot(epochs, history["val"], label="Validation loss")
        ax.set_title(reg_name)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Binary cross-entropy loss")
        ax.grid(alpha=0.25)
        ax.legend()
    fig.suptitle(f"B2 - Loss curves: {chosen} (without vs. with L2)")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "B2_loss_curves.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    labels = [f"{row['Architecture']}\n{row['Regularization']}"
              for _, row in results.iterrows()]
    auc_values = list(results["Test AUC"]) + [xgboost_auc]
    labels.append("XGBoost (B1)")
    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.bar(range(len(labels)), auc_values)
    ax.set_xticks(range(len(labels)), labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Held-out test ROC-AUC")
    ax.set_title("B2 - Architecture and regularization comparison")
    for bar, value in zip(bars, auc_values):
        ax.annotate(f"{value:.4f}", (bar.get_x() + bar.get_width() / 2, value),
                    xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9)
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "B2_test_AUC_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return results


# B3: Hierarchical Risk Parity vs. Markowitz

def cluster_variance(covariance):
    """Variance of a cluster using inverse-variance weights."""
    inverse_variance = 1.0 / np.diag(covariance)
    inverse_variance /= inverse_variance.sum()
    return float(inverse_variance @ covariance @ inverse_variance)


def hrp_weights(covariance, correlation):
    """Single-linkage clustering followed by recursive bisection."""
    distance = np.sqrt((1.0 - correlation.clip(-1.0, 1.0)) / 2.0)

    distance_array = distance.to_numpy(copy=True)
    np.fill_diagonal(distance_array, 0.0)

    tree = linkage(squareform(distance_array, checks=False), method="single")

    # Put nearby assets next to each other before splitting the clusters.
    ordered_assets = list(correlation.index[leaves_list(tree)])
    weights = pd.Series(1.0, index=ordered_assets)
    clusters = [ordered_assets]

    while clusters:
        next_clusters = []
        for cluster in clusters:
            if len(cluster) <= 1:
                continue
            midpoint = len(cluster) // 2
            left, right = cluster[:midpoint], cluster[midpoint:]
            var_left = cluster_variance(covariance.loc[left, left].to_numpy())
            var_right = cluster_variance(covariance.loc[right, right].to_numpy())
            share_left = var_right / (var_left + var_right)
            weights.loc[left] *= share_left
            weights.loc[right] *= 1.0 - share_left
            next_clusters.extend([left, right])
        clusters = next_clusters

    assert np.isclose(weights.sum(), 1.0)
    assert (weights >= 0).all()
    return weights.reindex(covariance.index), tree, ordered_assets


def markowitz_weights(annual_mean, annual_covariance, risk_aversion=4.0):
    """Long-only mean-variance portfolio with weights summing to one.

    Assumptions: historical sample means and risk aversion of 4.
    """
    n_assets = len(annual_mean)
    mu = annual_mean.to_numpy()
    sigma = annual_covariance.to_numpy()

    def objective(w):
        return -float(mu @ w - 0.5 * risk_aversion * (w @ sigma @ w))

    def gradient(w):
        return -(mu - risk_aversion * (sigma @ w))

    result = minimize(
        objective, np.ones(n_assets) / n_assets,
        jac=gradient,
        method="SLSQP",
        bounds=[(0, 1)] * n_assets,
        constraints={"type": "eq", "fun": lambda w: w.sum() - 1,
                     "jac": lambda w: np.ones_like(w)},
        options={"ftol": 1e-12, "maxiter": 5000},
    )
    if not result.success:
        raise RuntimeError(f"Markowitz optimization failed: {result.message}")
    weights = pd.Series(result.x, index=annual_mean.index)
    weights.loc[weights.abs() < 1e-10] = 0.0
    return weights / weights.sum()


def run_b3():
    print("\nB3 - HRP vs. Markowitz")
    path = BASE_DIR / "mmf2030h_a2_prices.csv"
    prices = pd.read_csv(path, parse_dates=["Date"]).set_index("Date")
    assets = [f"TECH_{letter}" for letter in "ABCD"] + [f"STPL_{letter}" for letter in "ABCD"]
    assert len(prices) == 1250 and set(prices.columns) == set(assets)
    prices = prices[assets]

    returns = prices.pct_change().dropna()
    covariance = returns.cov()
    correlation = returns.corr()
    hrp, tree, asset_order = hrp_weights(covariance, correlation)

    # Annualize both the means and covariance using 252 trading days.
    annual_mean = returns.mean() * 252
    annual_covariance = covariance * 252
    markowitz = markowitz_weights(annual_mean, annual_covariance, risk_aversion=4.0)

    weights = pd.DataFrame({"HRP": hrp, "Markowitz mean-variance": markowitz})
    effective_assets = 1.0 / weights.pow(2).sum(axis=0)
    weight_variance = weights.var(axis=0, ddof=0)
    print(f"Prices: {len(prices)}, returns: {len(returns)}")
    print(f"Cluster order: {asset_order}")
    print(f"Covariance condition number: {np.linalg.cond(covariance):.3f}")
    print("Portfolio weights (%):\n", (100 * weights).round(2))
    print("Effective number of assets:\n", effective_assets.round(3))
    print("Weight variance:\n", weight_variance.round(6))

    weights.to_csv(OUTPUT_DIR / "B3_portfolio_weights.csv", index_label="ticker")
    pd.DataFrame({"Effective N": effective_assets, "Weight variance": weight_variance}).to_csv(
        OUTPUT_DIR / "B3_concentration.csv", index_label="portfolio"
    )
    covariance.to_csv(OUTPUT_DIR / "B3_daily_covariance.csv")

    fig, ax = plt.subplots(figsize=(8.5, 5))
    image = ax.imshow(correlation.loc[asset_order, asset_order], vmin=-1, vmax=1,
                      cmap="coolwarm")
    ax.set_xticks(range(8), asset_order, rotation=45, ha="right")
    ax.set_yticks(range(8), asset_order)
    fig.colorbar(image, ax=ax, label="Correlation")
    ax.set_title("B3 - Return correlation (quasi-diagonal order)")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "B3_correlation_heatmap.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    dendrogram(tree, labels=list(correlation.index), ax=ax)
    ax.set_title("B3 - Hierarchical clustering of assets")
    ax.set_ylabel("Correlation distance")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "B3_HRP_dendrogram.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    ax = weights.plot(kind="bar", figsize=(10, 5))
    ax.set_title("B3 - Portfolio weights: HRP vs Markowitz")
    ax.set_ylabel("Portfolio weight")
    ax.set_xlabel("Asset")
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    ax.legend(loc="best")
    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "B3_weights_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return weights, effective_assets


if __name__ == "__main__":
    set_seed()
    train, test, features, target = load_credit_data()
    xgboost_auc = run_b1(train, test, features, target)
    nn_results = run_b2(train, test, features, target, xgboost_auc)
    portfolio_weights, concentration = run_b3()
    print(f"\nDone. Results and plots saved to: {OUTPUT_DIR}")
