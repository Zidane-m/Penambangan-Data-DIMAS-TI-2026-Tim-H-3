# utils/metrics.py
"""
Statistik Survei dan Metrik Evaluasi Terbobot (Weighted Metrics & Survey Statistics)
Mendukung pembobotan populasi PISA (W_FSTUWT), replicate weights Fay-BRR, dan aturan Rubin.
"""
import numpy as np
import pandas as pd


def weighted_mean(y, weights):
    """Menghitung rata-rata terbobot sampel survei."""
    mask = ~np.isnan(y) & ~np.isnan(weights)
    if not mask.any():
        return np.nan
    return np.sum(y[mask] * weights[mask]) / np.sum(weights[mask])


def weighted_rmse(y_true, y_pred, weights=None):
    """Root Mean Squared Error terbobot (Weighted RMSE)."""
    if weights is None:
        return np.sqrt(np.mean((y_true - y_pred) ** 2))
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred) & ~np.isnan(weights)
    return np.sqrt(np.average((y_true[mask] - y_pred[mask]) ** 2, weights=weights[mask]))


def weighted_mae(y_true, y_pred, weights=None):
    """Mean Absolute Error terbobot (Weighted MAE)."""
    if weights is None:
        return np.mean(np.abs(y_true - y_pred))
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred) & ~np.isnan(weights)
    return np.average(np.abs(y_true[mask] - y_pred[mask]), weights=weights[mask])


def weighted_r2(y_true, y_pred, weights=None):
    """Koefisien Determinasi R^2 terbobot (Weighted R-Squared)."""
    if weights is None:
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred) & ~np.isnan(weights)
    y_t, y_p, w = y_true[mask], y_pred[mask], weights[mask]
    y_bar = np.average(y_t, weights=w)
    ss_res = np.sum(w * (y_t - y_p) ** 2)
    ss_tot = np.sum(w * (y_t - y_bar) ** 2)
    return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0


def rubin_combine(point_estimates, variances):
    """
    Aturan Penggabungan Rubin (Rubin's Rules) untuk M Plausible Values.
    Menggabungkan variansi sampling (within) dan variansi imputasi (between).
    """
    M = len(point_estimates)
    theta_bar = np.mean(point_estimates)
    U_bar = np.nanmean(variances)
    B = np.var(point_estimates, ddof=1) if M > 1 else 0.0
    T = U_bar + (1.0 + 1.0 / M) * B
    return theta_bar, np.sqrt(T), np.sqrt(U_bar), np.sqrt(B)

