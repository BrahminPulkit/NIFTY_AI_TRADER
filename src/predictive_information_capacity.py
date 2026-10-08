"""Step 19C: descriptive information-capacity research, with no model fitting."""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import mutual_info_score, roc_auc_score


def binary_entropy(target: pd.Series) -> float:
    probability = float(pd.Series(target).astype(int).mean())
    if probability <= 0 or probability >= 1:
        return 0.0
    return float(-probability * np.log2(probability) - (1 - probability) * np.log2(1 - probability))


def _quantile_codes(series: pd.Series, bins: int = 10) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce")
    filled = values.fillna(values.median())
    if filled.nunique() < 2:
        return np.zeros(len(filled), dtype=int)
    return pd.qcut(
        filled, q=min(bins, filled.nunique()), duplicates="drop"
    ).cat.codes.to_numpy()


def information_capacity(
    frame: pd.DataFrame, features: list[str], target: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute univariate information measures without fitting a predictor."""
    data = frame[features + [target]].copy()
    y = data[target].astype(int)
    target_entropy = binary_entropy(y)
    numeric = data[features].apply(pd.to_numeric, errors="coerce")
    numeric = numeric.fillna(numeric.median()).fillna(0.0)
    mi_continuous = mutual_info_classif(
        numeric, y, discrete_features=False, random_state=42
    )
    rows = []
    for index, feature in enumerate(features):
        codes = _quantile_codes(data[feature])
        information_gain = float(mutual_info_score(codes, y) / np.log(2))
        rows.append({
            "feature": feature,
            "information_gain_bits": information_gain,
            "conditional_entropy_bits": max(0.0, target_entropy - information_gain),
            "mutual_information_nats": float(mi_continuous[index]),
            "fraction_target_entropy_explained": (
                information_gain / target_entropy if target_entropy else np.nan
            ),
            "non_null_fraction": float(data[feature].notna().mean()),
            "unique_values": int(data[feature].nunique(dropna=True)),
        })
    entropy = pd.DataFrame([
        {"measure": "observations", "value": len(y)},
        {"measure": "positive_observations", "value": int(y.sum())},
        {"measure": "negative_observations", "value": int((1 - y).sum())},
        {"measure": "positive_prevalence", "value": float(y.mean())},
        {"measure": "target_entropy_bits", "value": target_entropy},
        {"measure": "irreducible_entropy_not_identifiable", "value": np.nan},
    ])
    return pd.DataFrame(rows).sort_values(
        ["information_gain_bits", "mutual_information_nats"], ascending=False
    ), entropy


def _population_stability_index(reference: pd.Series, sample: pd.Series) -> float:
    combined = pd.concat([reference, sample]).dropna()
    if combined.nunique() < 2:
        return 0.0
    edges = np.unique(combined.quantile(np.linspace(0, 1, 11)).to_numpy())
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    ref = pd.cut(reference, edges, include_lowest=True).value_counts(normalize=True, sort=False)
    cur = pd.cut(sample, edges, include_lowest=True).value_counts(normalize=True, sort=False)
    ref = ref.clip(lower=1e-6)
    cur = cur.clip(lower=1e-6)
    return float(((cur - ref) * np.log(cur / ref)).sum())


def feature_redundancy(
    frame: pd.DataFrame, features: list[str], information: pd.DataFrame
) -> pd.DataFrame:
    numeric = frame[features].apply(pd.to_numeric, errors="coerce")
    correlation = numeric.corr(method="spearman").abs()
    years = sorted(frame["year"].dropna().unique())
    info_lookup = information.set_index("feature")
    rows = []
    for feature in features:
        peer = correlation[feature].drop(index=feature).sort_values(ascending=False)
        max_corr = float(peer.iloc[0]) if len(peer) else 0.0
        max_peer = str(peer.index[0]) if len(peer) else ""
        psi_values = [
            _population_stability_index(
                numeric.loc[frame.year.eq(years[0]), feature],
                numeric.loc[frame.year.eq(year), feature],
            )
            for year in years[1:]
        ] if len(years) > 1 else [0.0]
        information_gain = float(info_lookup.loc[feature, "information_gain_bits"])
        rows.append({
            "feature": feature,
            "most_correlated_feature": max_peer,
            "max_absolute_spearman": max_corr,
            "highly_redundant_095": max_corr >= 0.95,
            "near_zero_information": information_gain <= 1e-5,
            "information_gain_bits": information_gain,
            "independent_signal_score": information_gain * (1 - max_corr),
            "maximum_yearly_psi": max(psi_values),
            "stability_score": 1 / (1 + max(psi_values)),
        })
    return pd.DataFrame(rows).sort_values(
        ["independent_signal_score", "information_gain_bits"], ascending=False
    )


def _bhattacharyya_normal(a: np.ndarray, b: np.ndarray) -> float:
    mean_a, mean_b = np.mean(a), np.mean(b)
    var_a, var_b = max(np.var(a), 1e-12), max(np.var(b), 1e-12)
    return float(
        0.25 * np.log(0.25 * (var_a / var_b + var_b / var_a + 2))
        + 0.25 * (mean_a - mean_b) ** 2 / (var_a + var_b)
    )


def _overlap_coefficient(a: np.ndarray, b: np.ndarray) -> float:
    combined = np.concatenate([a, b])
    if np.min(combined) == np.max(combined):
        return 1.0
    edges = np.histogram_bin_edges(combined, bins="fd")
    if len(edges) > 101:
        edges = np.linspace(np.min(combined), np.max(combined), 101)
    hist_a, _ = np.histogram(a, bins=edges, density=True)
    hist_b, _ = np.histogram(b, bins=edges, density=True)
    return float(np.sum(np.minimum(hist_a, hist_b) * np.diff(edges)))


def _auc_interval(a: np.ndarray, b: np.ndarray, seed: int = 42) -> tuple[float, float, float]:
    y = np.concatenate([np.ones(len(a)), np.zeros(len(b))])
    scores = np.concatenate([a, b])
    auc = float(roc_auc_score(y, scores))
    auc = max(auc, 1 - auc)
    random = np.random.default_rng(seed)
    boot = []
    for _ in range(300):
        sample_a = random.choice(a, size=len(a), replace=True)
        sample_b = random.choice(b, size=len(b), replace=True)
        sample_y = np.concatenate([np.ones(len(sample_a)), np.zeros(len(sample_b))])
        sample_score = np.concatenate([sample_a, sample_b])
        value = roc_auc_score(sample_y, sample_score)
        boot.append(max(value, 1 - value))
    return auc, float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def class_separability(
    frame: pd.DataFrame, features: list[str], target: str
) -> pd.DataFrame:
    rows = []
    for feature in features:
        values = pd.to_numeric(frame[feature], errors="coerce")
        positive = values.loc[frame[target].eq(1)].dropna().to_numpy(float)
        negative = values.loc[frame[target].eq(0)].dropna().to_numpy(float)
        if len(positive) < 2 or len(negative) < 2:
            continue
        pooled = np.sqrt(
            ((len(positive) - 1) * np.var(positive, ddof=1)
             + (len(negative) - 1) * np.var(negative, ddof=1))
            / (len(positive) + len(negative) - 2)
        )
        cohen = (np.mean(positive) - np.mean(negative)) / pooled if pooled else 0.0
        ks = ks_2samp(positive, negative, alternative="two-sided", method="auto")
        auc, low, high = _auc_interval(positive, negative)
        rows.append({
            "feature": feature,
            "positive_count": len(positive),
            "negative_count": len(negative),
            "ks_statistic": float(ks.statistic),
            "ks_pvalue_unadjusted": float(ks.pvalue),
            "cohens_d": float(cohen),
            "bhattacharyya_distance_normal_approx": _bhattacharyya_normal(
                positive, negative
            ),
            "distribution_overlap_coefficient": _overlap_coefficient(
                positive, negative
            ),
            "orientation_free_univariate_auc": auc,
            "auc_bootstrap_95_low": low,
            "auc_bootstrap_95_high": high,
        })
    result = pd.DataFrame(rows)
    result["ks_pvalue_bonferroni"] = np.minimum(
        1.0, result.ks_pvalue_unadjusted * len(result)
    )
    return result.sort_values(
        ["orientation_free_univariate_auc", "ks_statistic"], ascending=False
    )

