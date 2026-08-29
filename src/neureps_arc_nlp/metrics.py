from __future__ import annotations

from typing import Any

import numpy as np

METRIC_VERSION = "neureps_aa_metrics.20260824.v1"


def validate_coefficients(coefficients: np.ndarray, tolerance: float = 5e-6) -> np.ndarray:
    values = np.asarray(coefficients, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] == 0 or not np.isfinite(values).all():
        raise ValueError("coefficient matrix must be finite and two-dimensional")
    if float(values.min()) < -tolerance:
        raise ValueError("coefficient matrix contains negative values")
    simplex_error = float(np.max(np.abs(values.sum(axis=1) - 1.0)))
    if simplex_error > tolerance:
        raise ValueError(f"coefficient simplex error: {simplex_error}")
    return values


def compute_reconstruction_metrics(X: np.ndarray, A: np.ndarray, Z: np.ndarray) -> dict[str, float]:
    values = np.asarray(X, dtype=np.float64)
    coefficients = validate_coefficients(A)
    archetypes = np.asarray(Z, dtype=np.float64)
    if values.ndim != 2 or archetypes.ndim != 2:
        raise ValueError("X and Z must be two-dimensional")
    if coefficients.shape[0] != values.shape[0] or coefficients.shape[1] != archetypes.shape[0]:
        raise ValueError("X, A, and Z shapes are incompatible")
    reconstruction = coefficients @ archetypes
    residual = values - reconstruction
    sse = float(np.sum(residual * residual))
    mse = float(sse / values.shape[0])
    centered = values - values.mean(axis=0, keepdims=True)
    sst = float(np.sum(centered * centered))
    norm_sq = float(np.sum(values * values))
    return {
        "mse": mse,
        "r2": float(1.0 - sse / sst) if sst > 1e-12 else (1.0 if sse <= 1e-12 else 0.0),
        "paper_variance_explained": (float(1.0 - sse / norm_sq) if norm_sq > 1e-12 else (1.0 if sse <= 1e-12 else 0.0)),
        "relative_mse": float(mse / max(float(np.var(values)), 1e-12)),
    }


def compute_coefficient_metrics(coefficients: np.ndarray) -> dict[str, float | int]:
    values = validate_coefficients(coefficients)
    usage = values.mean(axis=0)
    sorted_usage = np.sort(usage)[::-1]
    cumulative = np.cumsum(sorted_usage)
    assignments = values.argmax(axis=1)
    assignment_share = np.bincount(assignments, minlength=values.shape[1]) / float(values.shape[0])

    def rank_count(threshold: float) -> int:
        return int(np.searchsorted(cumulative, threshold) + 1)

    return {
        "mean_max_weight": float(np.mean(np.max(values, axis=1))),
        "effective_k": float(np.mean(1.0 / np.maximum(np.sum(values * values, axis=1), 1e-12))),
        "active_anchor_count": int(np.sum(usage > 1e-12)),
        "dominant_archetype_share": float(np.max(assignment_share)),
        "rank80_count_mean_contribution": rank_count(0.80),
        "rank90_count_mean_contribution": rank_count(0.90),
        "rank95_count_mean_contribution": rank_count(0.95),
    }


def label_membership(labels: np.ndarray) -> np.ndarray:
    values = np.asarray(labels)
    if values.ndim == 1:
        unique = sorted(set(values.tolist()), key=str)
        if not unique:
            raise ValueError("label array is empty")
        mapping = {value: index for index, value in enumerate(unique)}
        output = np.zeros((len(values), len(unique)), dtype=np.float64)
        output[np.arange(len(values)), [mapping[value] for value in values.tolist()]] = 1.0
        return output
    if values.ndim == 2:
        output = np.asarray(values, dtype=np.float64)
        if output.shape[1] == 0 or not np.isfinite(output).all() or np.any(output < 0):
            raise ValueError("invalid multilabel membership")
        if np.any(output.sum(axis=1) <= 0):
            raise ValueError("multilabel membership contains an empty row")
        return output
    raise ValueError(f"unsupported label shape: {values.shape}")


def compute_label_metrics(coefficients: np.ndarray, labels: np.ndarray) -> dict[str, float | int]:
    values = validate_coefficients(coefficients)
    membership = label_membership(labels)
    if values.shape[0] != membership.shape[0]:
        raise ValueError("coefficient and label row counts differ")

    eps = 1e-12
    usage = values.sum(axis=0)
    label_anchor_mass = membership.T @ values
    conditional = np.divide(
        label_anchor_mass,
        usage[None, :],
        out=np.zeros_like(label_anchor_mass),
        where=usage[None, :] > eps,
    )
    purity = float(np.sum((usage / values.shape[0]) * np.max(conditional, axis=0)))

    joint = label_anchor_mass / float(membership.sum())
    p_label = joint.sum(axis=1, keepdims=True)
    p_anchor = joint.sum(axis=0, keepdims=True)
    expected = p_label @ p_anchor
    mask = (joint > eps) & (expected > eps)
    mutual_information = float(np.sum(joint[mask] * np.log(joint[mask] / expected[mask])))
    positive_labels = p_label[p_label > eps]
    label_entropy = float(-np.sum(positive_labels * np.log(positive_labels)))
    if label_entropy <= eps:
        raise ValueError("label entropy is zero")

    global_top = float(np.max(membership.sum(axis=0) / values.shape[0]))
    if purity + eps < global_top:
        raise ValueError("usage-weighted purity is below the global top-label probability")
    return {
        "usage_weighted_purity": purity,
        "global_top_label_probability": global_top,
        "normalized_label_coefficient_mi": mutual_information / label_entropy,
        "label_entropy": label_entropy,
        "n_labels": int(membership.shape[1]),
    }


def compute_all_metrics(
    *,
    X_train: np.ndarray,
    X_valid: np.ndarray,
    A_train: np.ndarray,
    A_valid: np.ndarray,
    Z: np.ndarray,
    labels_valid: np.ndarray,
) -> dict[str, Any]:
    return {
        "metric_version": METRIC_VERSION,
        "n_train": int(X_train.shape[0]),
        "n_valid": int(X_valid.shape[0]),
        "n_features": int(X_train.shape[1]),
        "reconstruction": {
            "train": compute_reconstruction_metrics(X_train, A_train, Z),
            "valid": compute_reconstruction_metrics(X_valid, A_valid, Z),
        },
        "coefficient_usage": {
            "train": compute_coefficient_metrics(A_train),
            "valid": compute_coefficient_metrics(A_valid),
        },
        "label_alignment_valid": compute_label_metrics(A_valid, labels_valid),
    }
