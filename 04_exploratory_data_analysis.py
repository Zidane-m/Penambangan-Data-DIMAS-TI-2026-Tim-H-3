#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
04_exploratory_data_analysis.py
===============================
Tahap 3: Analisis Eksplorasi Data (EDA) & Statistik Multivariat
Kompetisi: DIMAS-TI AMLI 2026 - Divisi Penambangan Data (Data Mining)

Deskripsi:
  1. Analisis statistik deskriptif terbobot populasi (W_FSTUWT) untuk seluruh domain capaian
     (Sains, Computational Problem Solving LDW, Matematika, dan Membaca).
  2. Uji korelasi multivariat antar-fitur prediktor psikososial dan modal digital terhadap skor capaian.
  3. Visualisasi distribusi skor nasional vs baseline OECD (300 DPI).
  4. Analisis disparitas status sosio-ekonomi (ESCS vs Science Equity Gap).
  5. Boxplot kuartil penggunaan AI di sekolah (AIUSESCH) terhadap skor komputasi & sains.
  6. Matriks korelasi heatmap berresolusi tinggi.
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from utils.visual_config import setup_visual_style, PALETTE_PRIMARY
from utils.metrics import weighted_mean

warnings.filterwarnings('ignore')


def run_eda(input_path="data/pisa2025_features_engineered.csv", output_dir="outputs"):
    print("=" * 80)
    print("     TAHAP 3: EXPLORATORY DATA ANALYSIS (EDA) & STATISTIK SURVEI        ")
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

    print(f"[*] Memuat data analisis dari: {input_path}")
    df = pd.read_csv(input_path, low_memory=False)

    # Setup direktori luaran
    fig_dir = os.path.join(output_dir, "figures")
    table_dir = os.path.join(output_dir, "tables")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(table_dir, exist_ok=True)

    # Menjamin adanya kolom TARGET
    if 'TARGET_SCIE' not in df.columns:
        scie_pvs = [f'PV{i}SCIE' for i in range(1, 11) if f'PV{i}SCIE' in df.columns]
        if scie_pvs:
            df['TARGET_SCIE'] = df[scie_pvs].mean(axis=1)

    weight_col = 'W_FSTUWT' if 'W_FSTUWT' in df.columns else None

    # 2. Ringkasan Statistik Capaian Siswa Indonesia
    print("\n[*] Ringkasan Capaian Skor PISA 2025 Indonesia:")
    targets = [
        ('Sains', 'TARGET_SCIE'),
        ('Komputasi (LDW)', 'TARGET_CMPS'),
        ('Matematika', 'TARGET_MATH'),
        ('Membaca', 'TARGET_READ')
    ]
    
    summary_list = []
    for subject, col in targets:
        if col in df.columns:
            m = df[col].mean()
            s = df[col].std()
            q25 = df[col].quantile(0.25)
            q75 = df[col].quantile(0.75)
            w_m = weighted_mean(df[col].values, df[weight_col].values) if weight_col else m
            print(f"    - {subject:<18}: Mean={m:.2f} (Terbobot={w_m:.2f}), Std={s:.2f}, IQR=[{q25:.2f} - {q75:.2f}]")
            summary_list.append({
                'Domain': subject,
                'Rerata_Tak_Terbobot': round(m, 2),
                'Rerata_Terbobot': round(w_m, 2),
                'Simpangan_Baku': round(s, 2),
                'Q1_25': round(q25, 2),
                'Q3_75': round(q75, 2)
            })

    pd.DataFrame(summary_list).to_csv(os.path.join(table_dir, "skor_capaian_domain_summary.csv"), index=False)

    # 3. Analisis Korelasi Multivariat terhadap Capaian Sains
    ses_cols = ['ESCS', 'HOMEPOS', 'ICTRES', 'PAREDINT', 'HISEI']
    affective_cols = ['BELONG', 'BULLIED', 'FEELSAFE', 'DISCLISCI', 'TEACHSUP', 'COGACSC', 
                      'JOYSCIE', 'EFFSCIE', 'CURIO', 'SELFREG', 'ENPROBS', 'FAMSUP', 'AIUSESCH']
    composite_cols = ['PSYCH_SAFETY', 'PEDAGOGICAL_CLIMATE', 'SCIENCE_AFFECTIVE', 'INTERNAL_DRIVE', 'DIGITAL_CAPITAL']
    
    candidate_features = [c for c in ses_cols + affective_cols + composite_cols if c in df.columns]
    
    if 'TARGET_SCIE' in df.columns and candidate_features:
        print("\n[*] Peringkat Korelasi Linear (r) Fitur terhadap Skor Sains (TARGET_SCIE):")
        corr_series = df[candidate_features].apply(lambda s: s.corr(df['TARGET_SCIE'])).sort_values(ascending=False)
        for feat, r in corr_series.items():
            print(f"    {feat:<22}: r = {r:+.4f}")
        
        corr_df = corr_series.to_frame(name='correlation_with_science').reset_index().rename(columns={'index': 'Feature'})
        corr_df.to_csv(os.path.join(table_dir, "correlation_summary.csv"), index=False)

    # 4. Pembuatan Visualisasi Standar Publikasi (300 DPI)
    print("\n[*] Menghasilkan Visualisasi Standar Publikasi (300 DPI)...")
    setup_visual_style()

    # Grafik 1: Distribusi Skor Sains vs Baseline OECD
    if 'TARGET_SCIE' in df.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(df['TARGET_SCIE'], kde=True, color='#1f77b4', bins=40, ax=ax, stat='density', alpha=0.6)
        nat_mean = df['TARGET_SCIE'].mean()
        ax.axvline(nat_mean, color='#d62728', linestyle='--', linewidth=2, label=f'Rata-rata Indonesia ({nat_mean:.1f})')
        ax.axvline(485, color='#2ca02c', linestyle=':', linewidth=2, label='Rata-rata Baseline OECD (~485)')
        ax.set_title('Distribusi Capaian Literasi Sains PISA 2025 Siswa Indonesia', pad=12)
        ax.set_xlabel('Skor Rata-rata Plausible Values Sains')
        ax.set_ylabel('Densitas Probabilitas')
        ax.legend(frameon=True, facecolor='white', framealpha=0.9)
        plt.tight_layout()
        fig_path = os.path.join(fig_dir, "01_distribusi_skor_sains.png")
        fig.savefig(fig_path)
        plt.close()
        print(f"    [+] Disimpan: {fig_path}")

    # Grafik 2: ESCS vs Capaian Sains (Equity Gap)
    if 'ESCS' in df.columns and 'TARGET_SCIE' in df.columns:
        fig, ax = plt.subplots(figsize=(8, 5.5))
        sample_size = min(3000, len(df))
        sns.regplot(data=df.sample(sample_size, random_state=42), x='ESCS', y='TARGET_SCIE',
                    scatter_kws={'alpha': 0.18, 'color': '#3b528b', 's': 18},
                    line_kws={'color': '#d95f02', 'linewidth': 2.5}, ax=ax)
        ax.set_title('Kesenjangan Status Sosio-Ekonomi (ESCS) terhadap Capaian Sains', pad=12)
        ax.set_xlabel('Indeks Status Ekonomi, Sosial, dan Budaya (ESCS)')
        ax.set_ylabel('Skor Sains Siswa')
        plt.tight_layout()
        fig_path = os.path.join(fig_dir, "02_escs_vs_science_equity.png")
        fig.savefig(fig_path)
        plt.close()
        print(f"    [+] Disimpan: {fig_path}")

    # Grafik 3: Penggunaan AI di Sekolah vs Capaian Sains & Komputasi
    if 'AIUSESCH' in df.columns and 'TARGET_SCIE' in df.columns:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
        valid_ai = df.dropna(subset=['AIUSESCH']).copy()
        valid_ai['AI_BIN'] = pd.qcut(valid_ai['AIUSESCH'], q=4, 
                                     labels=['Kuartil 1\n(Sangat Rendah)', 'Kuartil 2\n(Sedang)', 'Kuartil 3\n(Tinggi)', 'Kuartil 4\n(Sangat Intensif)'])
        
        sns.boxplot(data=valid_ai, x='AI_BIN', y='TARGET_SCIE', palette='Purples', ax=axes[0])
        axes[0].set_title('Penggunaan AI di Sekolah vs Skor Sains', fontweight='bold')
        axes[0].set_xlabel('Tingkat Penggunaan AI di Sekolah')
        axes[0].set_ylabel('Skor Sains Rata-rata')

        target_comp = 'TARGET_CMPS' if 'TARGET_CMPS' in valid_ai.columns else 'TARGET_SCIE'
        sns.boxplot(data=valid_ai, x='AI_BIN', y=target_comp, palette='Blues', ax=axes[1])
        axes[1].set_title(f'Penggunaan AI di Sekolah vs Skor Komputasi (LDW)', fontweight='bold')
        axes[1].set_xlabel('Tingkat Penggunaan AI di Sekolah')
        axes[1].set_ylabel('Skor Komputasi')
        
        plt.tight_layout()
        fig_path = os.path.join(fig_dir, "03_ai_usage_impact.png")
        fig.savefig(fig_path)
        plt.close()
        print(f"    [+] Disimpan: {fig_path}")

    # Grafik 4: Matriks Korelasi Multivariat
    core_corr_feats = [c for c in candidate_features if c in df.columns][:12]
    if 'TARGET_SCIE' in df.columns:
        core_corr_feats.append('TARGET_SCIE')
    
    if len(core_corr_feats) > 3:
        fig, ax = plt.subplots(figsize=(10, 8))
        sub_corr = df[core_corr_feats].corr()
        mask = np.triu(np.ones_like(sub_corr, dtype=bool))
        sns.heatmap(sub_corr, mask=mask, cmap='vlag', vmin=-0.5, vmax=0.5, annot=True, fmt='.2f', 
                    cbar_kws={'label': 'Korelasi Pearson (r)'}, ax=ax, square=True)
        ax.set_title('Matriks Korelasi Multivariat Indeks Psikososial, AI, dan Skor Capaian', pad=12)
        plt.tight_layout()
        fig_path = os.path.join(fig_dir, "04_correlation_heatmap.png")
        fig.savefig(fig_path)
        plt.close()
        print(f"    [+] Disimpan: {fig_path}")

    print("\n[+] Tahap 3 EDA selesai dengan sukses!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exploratory Data Analysis PISA 2025")
    parser.add_argument("--input-path", type=str, default="data/pisa2025_features_engineered.csv", help="Input dataset")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Direktori hasil")
    args = parser.parse_args()

    run_eda(input_path=args.input_path, output_dir=args.output_dir)

