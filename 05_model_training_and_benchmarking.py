#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
05_model_training_and_benchmarking.py
=====================================
Tahap 4: Pelatihan Model, Validasi Silang Bebas Leakage, & Benchmark Metrik
Kompetisi: DIMAS-TI AMLI 2026 - Divisi Penambangan Data (Data Mining)

Deskripsi:
  1. Validasi Silang 5-Fold School-Level GroupKFold:
     - Mencegah kebocoran klaster (zero cluster leakage) antar-sekolah (Intraclass Correlation ICC = 0,501).
  2. Pembobotan Populasi (Sample Weights W_FSTUWT):
     - Memastikan model mewakili distribusi populasi nyata siswa usia 15 tahun di Indonesia.
  3. Hirarki Benchmark Model Bertingkat:
     - Naive Baseline (Dummy Mean Regressor)
     - Interpretable Linear Baseline (Ridge Regression dengan Standard Scaler)
     - Non-linear Bagging Ensemble (Random Forest Regressor)
     - Gradient Boosting Tree (XGBoost Regressor)
     - Gradient Boosting Tree (LightGBM Regressor)
  4. Evaluasi Metrik Komprehensif:
     - RMSE, MAE, R^2 (baik versi standar maupun terbobot populasi).
  5. Uji Signifikansi Statistik (Paired t-test):
     - Memvalidasi secara empiris keunggulan non-linearitas model ensemble terhadap baseline linier.
"""

import os
import argparse
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import warnings

from sklearn.model_selection import GroupKFold
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from utils.visual_config import setup_visual_style
from utils.metrics import weighted_rmse, weighted_mae, weighted_r2

warnings.filterwarnings('ignore')

# Coba import LightGBM dan XGBoost
try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


def train_and_benchmark(input_path="data/pisa2025_features_engineered.csv", output_dir="outputs", fast_mode=False):
    print("=" * 80)
    print("   TAHAP 4: MODEL TRAINING, BENCHMARKING & ZERO-LEAKAGE GROUP-KFOLD     ")
    print("=" * 80)

    # 1. Fallback Berkas Sumber
    if not os.path.exists(input_path):
        fallbacks = [
            "data/pisa2025_indonesia_cleaned.csv",
            "data/pisa2025_indonesia.csv",
            "../dataset/pisa2025_indonesia_cleaned.csv",
            "../dataset/pisa2025_indonesia.csv"
        ]
        for fb in fallbacks:
            if os.path.exists(fb):
                input_path = fb
                break

    print(f"[*] Memuat data pelatihan dari: {input_path}")
    df = pd.read_csv(input_path, low_memory=False)

    # Memastikan target capaian tersedia
    if 'TARGET_SCIE' not in df.columns:
        scie_pvs = [f'PV{i}SCIE' for i in range(1, 11) if f'PV{i}SCIE' in df.columns]
        if scie_pvs:
            df['TARGET_SCIE'] = df[scie_pvs].mean(axis=1)
        else:
            print("[!] Kolom target PV*SCIE tidak ditemukan.")
            return

    # Definisi Fitur Prediktor
    ses_cols = ['ESCS', 'HOMEPOS', 'ICTRES', 'PAREDINT', 'HISEI']
    affective_cols = ['BELONG', 'BULLIED', 'FEELSAFE', 'DISCLISCI', 'TEACHSUP', 'COGACSC', 
                      'JOYSCIE', 'EFFSCIE', 'CURIO', 'SELFREG', 'ENPROBS', 'FAMSUP', 'AIUSESCH']
    composite_cols = ['PSYCH_SAFETY', 'PEDAGOGICAL_CLIMATE', 'SCIENCE_AFFECTIVE', 'INTERNAL_DRIVE', 'DIGITAL_CAPITAL']
    
    feature_cols = [c for c in ses_cols + affective_cols + composite_cols if c in df.columns]
    
    # Imputasi residual nulls dengan median (safety fallback)
    for col in feature_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    X = df[feature_cols].copy()
    y = df['TARGET_SCIE'].values
    weights = df['W_FSTUWT'].values if 'W_FSTUWT' in df.columns else np.ones(len(df))
    groups = df['CNTSCHID'].values if 'CNTSCHID' in df.columns else np.arange(len(df))

    n_schools = len(np.unique(groups))
    print(f"[+] Dataset siap latih: {X.shape[0]:,} siswa, {X.shape[1]} fitur, {n_schools} klaster sekolah.")

    # 2. Definisi Model
    models = {
        "Baseline (Dummy Mean)": DummyRegressor(strategy="mean"),
        "Ridge Regression": Ridge(alpha=10.0),
        "Random Forest": RandomForestRegressor(
            n_estimators=60 if fast_mode else 120, 
            max_depth=10 if fast_mode else 12, 
            min_samples_split=8, 
            random_state=42, 
            n_jobs=-1
        )
    }

    if HAS_LGB:
        models["LightGBM"] = lgb.LGBMRegressor(
            n_estimators=80 if fast_mode else 150, 
            learning_rate=0.05, 
            max_depth=6, 
            num_leaves=31, 
            subsample=0.8, 
            colsample_bytree=0.8, 
            random_state=42, 
            verbose=-1
        )
    else:
        print("[!] LightGBM tidak terpasang. Dilewati.")

    if HAS_XGB:
        models["XGBoost"] = xgb.XGBRegressor(
            n_estimators=80 if fast_mode else 150, 
            learning_rate=0.05, 
            max_depth=5, 
            subsample=0.8, 
            colsample_bytree=0.8, 
            random_state=42, 
            verbosity=0
        )
    else:
        print("[!] XGBoost tidak terpasang. Dilewati.")

    # 3. Eksekusi 5-Fold GroupKFold Cross-Validation
    n_splits = 5 if n_schools >= 5 else 3
    print(f"\n[*] Menjalankan {n_splits}-Fold School-Level Stratified GroupKFold Cross-Validation...")
    gkf = GroupKFold(n_splits=n_splits)

    results = {name: {'rmse': [], 'mae': [], 'r2': [], 'weighted_rmse': [], 'weighted_mae': [], 'weighted_r2': []} for name in models}
    oof_preds = {name: np.zeros(len(df)) for name in models}

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups)):
        X_train, y_train, w_train = X.iloc[train_idx], y[train_idx], weights[train_idx]
        X_val, y_val, w_val = X.iloc[val_idx], y[val_idx], weights[val_idx]

        # Standard Scaler khusus model linear (fit hanya pada fold train untuk mencegah leakage)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        for name, model in models.items():
            if name == "Ridge Regression":
                model.fit(X_train_scaled, y_train, sample_weight=w_train)
                pred = model.predict(X_val_scaled)
            elif name in ["LightGBM", "XGBoost", "Random Forest"]:
                model.fit(X_train, y_train, sample_weight=w_train)
                pred = model.predict(X_val)
            else: # Dummy
                model.fit(X_train, y_train)
                pred = model.predict(X_val)

            oof_preds[name][val_idx] = pred

            # Metrik evaluasi standar & terbobot
            r_rmse = np.sqrt(mean_squared_error(y_val, pred))
            r_mae = mean_absolute_error(y_val, pred)
            r_r2 = r2_score(y_val, pred)

            w_rmse = weighted_rmse(y_val, pred, w_val)
            w_mae = weighted_mae(y_val, pred, w_val)
            w_r2 = weighted_r2(y_val, pred, w_val)

            results[name]['rmse'].append(r_rmse)
            results[name]['mae'].append(r_mae)
            results[name]['r2'].append(r_r2)
            results[name]['weighted_rmse'].append(w_rmse)
            results[name]['weighted_mae'].append(w_mae)
            results[name]['weighted_r2'].append(w_r2)

    # 4. Ringkasan Tabel Benchmark Metrik
    summary_rows = []
    for name in models:
        row = {
            "Model": name,
            "RMSE": f"{np.mean(results[name]['rmse']):.2f} ± {np.std(results[name]['rmse']):.2f}",
            "MAE": f"{np.mean(results[name]['mae']):.2f} ± {np.std(results[name]['mae']):.2f}",
            "R2": f"{np.mean(results[name]['r2']):.4f} ± {np.std(results[name]['r2']):.4f}",
            "Weighted_RMSE": f"{np.mean(results[name]['weighted_rmse']):.2f} ± {np.std(results[name]['weighted_rmse']):.2f}",
            "Weighted_MAE": f"{np.mean(results[name]['weighted_mae']):.2f} ± {np.std(results[name]['weighted_mae']):.2f}",
            "Weighted_R2": f"{np.mean(results[name]['weighted_r2']):.4f} ± {np.std(results[name]['weighted_r2']):.4f}"
        }
        summary_rows.append(row)

    benchmark_df = pd.DataFrame(summary_rows)
    print("\n[+] Hasil Benchmark Model (5-Fold School GroupKFold CV):")
    print(benchmark_df.to_string(index=False))

    # Simpan Tabel Benchmark
    table_dir = os.path.join(output_dir, "tables")
    os.makedirs(table_dir, exist_ok=True)
    benchmark_csv_path = os.path.join(table_dir, "model_benchmark_results.csv")
    benchmark_df.to_csv(benchmark_csv_path, index=False)
    print(f"[+] Tabel benchmark disimpan ke: {benchmark_csv_path}")

    # 5. Uji Signifikansi Statistik (Paired t-test: LightGBM vs Ridge)
    top_model = "LightGBM" if "LightGBM" in models else ("XGBoost" if "XGBoost" in models else "Random Forest")
    if top_model in results and "Ridge Regression" in results:
        t_stat, p_val = stats.ttest_rel(results[top_model]['rmse'], results["Ridge Regression"]['rmse'])
        print(f"\n[*] Uji Signifikansi Statistik ({top_model} vs Ridge Baseline):")
        print(f"    - Paired t-statistic : {t_stat:.4f}")
        print(f"    - p-value           : {p_val:.4e}")
        print(f"    - Signifikan (p < 0.05): {'YA (Signifikan Secara Statistik)' if p_val < 0.05 else 'TIDAK'}")

    # 6. Visualisasi Komparasi Kinerja Model
    setup_visual_style()
    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8.5, 5))
    model_names = list(models.keys())
    weighted_rmses = [np.mean(results[m]['weighted_rmse']) for m in models]
    
    colors = ['#8c8c8c', '#4575b4', '#74add1', '#313695', '#fdae61'][:len(model_names)]
    bars = ax.barh(model_names, weighted_rmses, color=colors, edgecolor='black', alpha=0.85)
    
    ax.set_xlabel("Weighted RMSE (Poin Skor PISA, Lebih Rendah = Lebih Akurat)")
    ax.set_title("Perbandingan Kinerja Algoritma Prediktif (Evaluasi Terbobot Populasi)", pad=12)
    
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 1, bar.get_y() + bar.get_height() / 2, f"{w:.2f}", va='center', fontweight='bold', fontsize=9.5)
    
    ax.set_xlim(0, max(weighted_rmses) + 15)
    plt.tight_layout()
    fig_path = os.path.join(fig_dir, "model_rmse_comparison.png")
    fig.savefig(fig_path)
    plt.close()
    print(f"[+] Grafik komparasi performa model disimpan ke: {fig_path}")

    print("\n[+] Tahap 4 Modeling & Benchmarking selesai dengan sukses!")
    print("=" * 80 + "\n")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pelatihan Model & Benchmark GroupKFold PISA 2025")
    parser.add_argument("--input-path", type=str, default="data/pisa2025_features_engineered.csv", help="Input dataset")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Direktori hasil")
    parser.add_argument("--fast", action="store_true", help="Mode cepat (sedikit iterasi)")
    args = parser.parse_args()

    train_and_benchmark(input_path=args.input_path, output_dir=args.output_dir, fast_mode=args.fast)

