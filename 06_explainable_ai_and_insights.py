#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
06_explainable_ai_and_insights.py
=================================
Tahap 5: Model Explainability (TreeSHAP) & Pembuktian 'The AI Paradox'
Kompetisi: DIMAS-TI AMLI 2026 - Divisi Penambangan Data (Data Mining)

Deskripsi:
  1. Menghitung atribusi nilai Shapley (SHapley Additive exPlanations) menggunakan TreeSHAP.
  2. Global Interpretability:
     - SHAP Beeswarm Plot (peringkat 15 fitur determinan terkuat dan arah dampaknya).
     - Ranking rerata kontribusi marjinal (|SHAP value|).
  3. Non-Linear Interaction & Threshold Analysis:
     - SHAP Dependence Plot pada variabel pemanfaatan AI (AIUSESCH) untuk memvalidasi
       kurva U-terbalik (The AI Paradox: titik jenuh pemanfaatan AI tanpa scaffolding).
     - Interaksi status sosio-ekonomi (ESCS) dan iklim dukungan guru (TEACHSUP).
  4. Local Interpretability (Studi Kasus Resiliensi):
     - SHAP Waterfall Plot mendekomposisi faktor pembeda siswa berprestasi tinggi
       yang berasal dari latar belakang ekonomi paling rentan (Academically Resilient).
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

from utils.visual_config import setup_visual_style

warnings.filterwarnings('ignore')

try:
    import lightgbm as lgb
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False


def run_xai(input_path="data/pisa2025_features_engineered.csv", output_dir="outputs", sample_size=2000):
    print("=" * 80)
    print("    TAHAP 5: EXPLAINABLE AI (TreeSHAP) & PEMBUKTIAN THE AI PARADOX      ")
    print("=" * 80)

    if not HAS_SHAP:
        print("[!] Library 'lightgbm' atau 'shap' tidak ditemukan.")
        print("    Jalankan: pip install lightgbm shap")
        return

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

    print(f"[*] Memuat data untuk analisis XAI dari: {input_path}")
    df = pd.read_csv(input_path, low_memory=False)

    if 'TARGET_SCIE' not in df.columns:
        scie_pvs = [f'PV{i}SCIE' for i in range(1, 11) if f'PV{i}SCIE' in df.columns]
        if scie_pvs:
            df['TARGET_SCIE'] = df[scie_pvs].mean(axis=1)

    ses_cols = ['ESCS', 'HOMEPOS', 'ICTRES', 'PAREDINT', 'HISEI']
    affective_cols = ['BELONG', 'BULLIED', 'FEELSAFE', 'DISCLISCI', 'TEACHSUP', 'COGACSC', 
                      'JOYSCIE', 'EFFSCIE', 'CURIO', 'SELFREG', 'ENPROBS', 'FAMSUP', 'AIUSESCH']
    composite_cols = ['PSYCH_SAFETY', 'PEDAGOGICAL_CLIMATE', 'SCIENCE_AFFECTIVE', 'INTERNAL_DRIVE', 'DIGITAL_CAPITAL']
    
    feature_cols = [c for c in ses_cols + affective_cols + composite_cols if c in df.columns]

    for col in feature_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    X = df[feature_cols].copy()
    y = df['TARGET_SCIE'].values
    weights = df['W_FSTUWT'].values if 'W_FSTUWT' in df.columns else np.ones(len(df))

    # 2. Pelatihan Model LightGBM Terbaik
    print("[*] Melatih model LightGBM untuk kalkulasi TreeSHAP...")
    best_lgb = lgb.LGBMRegressor(
        n_estimators=150, 
        learning_rate=0.05, 
        max_depth=6, 
        num_leaves=31, 
        subsample=0.8, 
        colsample_bytree=0.8, 
        random_state=42, 
        verbose=-1
    )
    best_lgb.fit(X, y, sample_weight=weights)

    # 3. Komputasi TreeSHAP Explainer
    print(f"[*] Menghitung nilai SHAP pada subsampel acak {min(sample_size, len(X)):,} siswa...")
    explainer = shap.TreeExplainer(best_lgb)
    shap_sample = X.sample(min(sample_size, len(X)), random_state=42)
    shap_values = explainer(shap_sample)

    fig_dir = os.path.join(output_dir, "figures")
    table_dir = os.path.join(output_dir, "tables")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(table_dir, exist_ok=True)

    # 4. Global Feature Importance (SHAP Beeswarm Plot)
    print("\n[*] Menghasilkan Gambar 1: SHAP Beeswarm Plot...")
    setup_visual_style()
    fig = plt.figure(figsize=(9.5, 6.5))
    shap.plots.beeswarm(shap_values, max_display=15, show=False)
    plt.title("Analisis Faktor Determinan Prestasi Sains Siswa (SHAP Beeswarm)", fontsize=12, fontweight='bold', pad=14)
    plt.tight_layout()
    fig1_path = os.path.join(fig_dir, "05_shap_summary_beeswarm.png")
    plt.savefig(fig1_path, bbox_inches='tight')
    plt.close()
    print(f"    [+] Disimpan: {fig1_path}")

    # Simpan Tabel Feature Importance
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    fi_df = pd.DataFrame({
        'Feature': feature_cols,
        'Mean_Absolute_SHAP': mean_abs_shap
    }).sort_values('Mean_Absolute_SHAP', ascending=False).reset_index(drop=True)
    fi_csv_path = os.path.join(table_dir, "tabel_feature_importance.csv")
    fi_df.to_csv(fi_csv_path, index=False)
    print(f"    [+] Tabel feature importance disimpan ke: {fi_csv_path}")

    # 5. Non-linear Dependence Plot (The AI Paradox & ESCS Interaction)
    if "AIUSESCH" in feature_cols and "ESCS" in feature_cols:
        print("\n[*] Menghasilkan Gambar 2: SHAP Dependence Plot (The AI Paradox & ESCS)...")
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
        
        # Plot 1: AIUSESCH vs ESCS
        shap.plots.scatter(shap_values[:, "AIUSESCH"], color=shap_values[:, "ESCS"], ax=axes[0], show=False)
        axes[0].set_title("Efek Marginal Penggunaan AI di Sekolah terhadap Skor Sains", fontsize=11, fontweight='bold')
        axes[0].set_xlabel("Tingkat Penggunaan AI di Sekolah (AIUSESCH)")
        axes[0].set_ylabel("Kontribusi SHAP terhadap Skor Sains")

        # Plot 2: ESCS vs TEACHSUP (atau DISCLISCI)
        interact_col = "TEACHSUP" if "TEACHSUP" in feature_cols else feature_cols[0]
        shap.plots.scatter(shap_values[:, "ESCS"], color=shap_values[:, interact_col], ax=axes[1], show=False)
        axes[1].set_title("Efek Marginal Status Sosio-Ekonomi (ESCS) terhadap Skor Sains", fontsize=11, fontweight='bold')
        axes[1].set_xlabel("Indeks Sosio-Ekonomi (ESCS)")
        axes[1].set_ylabel("Kontribusi SHAP terhadap Skor Sains")

        plt.tight_layout()
        fig2_path = os.path.join(fig_dir, "06_shap_dependence_ai_and_escs.png")
        plt.savefig(fig2_path, bbox_inches='tight')
        plt.close()
        print(f"    [+] Disimpan: {fig2_path}")

    # 6. Local Waterfall Plot: Studi Kasus Siswa Resilien
    print("\n[*] Menghasilkan Gambar 3: SHAP Waterfall Plot Studi Kasus Siswa Resilien...")
    if 'ESCS' in df.columns:
        resilient_idx = (df['ESCS'] < df['ESCS'].quantile(0.25)) & (df['TARGET_SCIE'] > df['TARGET_SCIE'].quantile(0.75))
        if resilient_idx.any():
            case_student = X[resilient_idx].iloc[0]
            case_shap = explainer(pd.DataFrame([case_student]))[0]

            fig = plt.figure(figsize=(9.5, 6))
            shap.plots.waterfall(case_shap, max_display=10, show=False)
            plt.title("Studi Kasus Siswa Resilien: Dekomposisi Faktor Pendorong Prestasi Sains", fontsize=11, fontweight='bold', pad=14)
            plt.tight_layout()
            fig3_path = os.path.join(fig_dir, "07_shap_case_study_waterfall.png")
            plt.savefig(fig3_path, bbox_inches='tight')
            plt.close()
            print(f"    [+] Disimpan: {fig3_path}")

    print("\n[+] Tahap 5 Explainable AI selesai dengan sukses!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Explainability & XAI PISA 2025")
    parser.add_argument("--input-path", type=str, default="data/pisa2025_features_engineered.csv", help="Input dataset")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Direktori hasil")
    parser.add_argument("--sample-size", type=int, default=2000, help="Jumlah sampel untuk SHAP")
    args = parser.parse_args()

    run_xai(input_path=args.input_path, output_dir=args.output_dir, sample_size=args.sample_size)

