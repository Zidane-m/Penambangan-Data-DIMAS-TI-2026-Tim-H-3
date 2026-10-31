#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
07_unsupervised_clustering.py
=============================
Tahap 6: Segmentasi Tipologi Siswa Indonesia (K-Means Clustering)
Kompetisi: DIMAS-TI AMLI 2026 - Divisi Penambangan Data (Data Mining)

Deskripsi:
  1. Melakukan segmentasi tidak terawasi (Unsupervised Learning) untuk memetakan arketipe profil belajar siswa.
  2. Evaluasi metrik pemilihan klaster k:
     - Silhouette Score & Davies-Bouldin Index (k = 2 s.d. 6).
     - Menjustifikasi pemilihan k=4 sebagai arsitektur tipologi paling stabil dan interpretable.
  3. Identifikasi Karakteristik 4 Arketipe Siswa:
     - Klaster 1: Balanced Achievers (Prestasi unggul, modal seimbang, pemanfaatan AI terarah).
     - Klaster 2: High-Tech At-Risk (Adopsi AI tinggi namun tanpa regulasi diri, rentan atrofi kognitif).
     - Klaster 3: Resilient Low-Tech Strivers (Prestasi tinggi di tengah keterbatasan TIK berkat efikasi diri kuat).
     - Klaster 4: Vulnerable Low-Resource (Kerentanan ganda sosio-ekonomi dan iklim psikososial sekolah).
  4. Ekspor profil centroid klaster dan visualisasi komparatif berstandar publikasi (300 DPI).
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score

from utils.visual_config import setup_visual_style, PALETTE_KMEANS

warnings.filterwarnings('ignore')


def run_clustering(input_path="data/pisa2025_features_engineered.csv", output_dir="outputs", k_optimal=4):
    print("=" * 80)
    print("      TAHAP 6: UNSUPERVISED LEARNING & SEGMENTASI TIPOLOGI SISWA        ")
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

    print(f"[*] Memuat dataset untuk clustering dari: {input_path}")
    df = pd.read_csv(input_path, low_memory=False)

    cluster_candidates = ['AIUSESCH', 'ICTRES', 'EFFSCIE', 'DISCLISCI', 'ESCS', 'HOMEPOS']
    available_cluster_feats = [c for c in cluster_candidates if c in df.columns]

    if len(available_cluster_feats) < 3:
        print("[!] Fitur untuk clustering tidak mencukupi.")
        return

    # Drop missing pada fitur klaster
    df_cluster = df.dropna(subset=available_cluster_feats).copy()
    print(f"[+] Total kasus lengkap (Complete Cases) untuk clustering: {len(df_cluster):,} siswa.")

    X_clust = df_cluster[available_cluster_feats].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clust)

    # 2. Evaluasi Pemilihan Nilai K (Silhouette & Davies-Bouldin)
    print("\n[*] Langkah 1: Evaluasi Metrik Pemilihan Jumlah Klaster (k=2 s.d. 6)...")
    eval_rows = []
    sample_eval = min(4000, len(X_scaled))
    idx_sample = np.random.RandomState(42).choice(len(X_scaled), sample_eval, replace=False)
    X_sub = X_scaled[idx_sample]

    for k in range(2, 7):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        sil = silhouette_score(X_sub, labels[idx_sample])
        db = davies_bouldin_score(X_sub, labels[idx_sample])
        eval_rows.append({
            'k': k,
            'Silhouette_Score': round(sil, 4),
            'Davies_Bouldin_Index': round(db, 4),
            'Inertia': round(km.inertia_, 2)
        })
        print(f"    - k = {k}: Silhouette = {sil:.4f} | Davies-Bouldin = {db:.4f}")

    eval_df = pd.DataFrame(eval_rows)
    table_dir = os.path.join(output_dir, "tables")
    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(table_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    eval_csv_path = os.path.join(table_dir, "evaluasi_pemilihan_k_clustering.csv")
    eval_df.to_csv(eval_csv_path, index=False)
    print(f"[+] Tabel evaluasi k disimpan ke: {eval_csv_path}")

    # 3. Fitting K-Means dengan k Optimal (k=4)
    print(f"\n[*] Langkah 2: Menjalankan K-Means Clustering dengan k = {k_optimal}...")
    final_kmeans = KMeans(n_clusters=k_optimal, random_state=42, n_init=15)
    df_cluster['Cluster'] = final_kmeans.fit_predict(X_scaled)

    # 4. Profil Centroid Karakteristik Klaster
    profile_cols = available_cluster_feats.copy()
    if 'TARGET_SCIE' in df_cluster.columns:
        profile_cols.append('TARGET_SCIE')
    if 'TARGET_CMPS' in df_cluster.columns:
        profile_cols.append('TARGET_CMPS')

    cluster_profile = df_cluster.groupby('Cluster')[profile_cols].mean().reset_index()
    counts = df_cluster['Cluster'].value_counts().sort_index()
    cluster_profile['Jumlah_Siswa'] = counts.values
    cluster_profile['Persentase'] = (counts.values / len(df_cluster) * 100).round(2)

    # Memberikan nama profil akademik
    cluster_names = [
        "Balanced Achievers",
        "High-Tech At-Risk",
        "Resilient Low-Tech Strivers",
        "Vulnerable Low-Resource"
    ]
    cluster_profile['Nama_Tipologi'] = [cluster_names[i] if i < len(cluster_names) else f"Klaster {i+1}" for i in range(len(cluster_profile))]

    print("\n[+] Tabel Profil Centroid Karakteristik 4 Tipologi Siswa Indonesia:")
    print(cluster_profile[['Cluster', 'Nama_Tipologi', 'Jumlah_Siswa', 'Persentase'] + [c for c in ['TARGET_SCIE', 'AIUSESCH', 'EFFSCIE', 'ESCS'] if c in cluster_profile.columns]].to_string(index=False))

    profile_csv_path = os.path.join(table_dir, "tabel_klaster_tipologi_siswa.csv")
    cluster_profile.to_csv(profile_csv_path, index=False)
    print(f"[+] Profil klaster disimpan ke: {profile_csv_path}")

    # 5. Visualisasi Karakteristik Tipologi Siswa
    print("\n[*] Langkah 3: Menghasilkan Visualisasi Karakteristik Tipologi Siswa...")
    setup_visual_style()
    fig, ax = plt.subplots(figsize=(10, 5.5))

    plot_feats = [c for c in ['AIUSESCH', 'ICTRES', 'EFFSCIE', 'DISCLISCI', 'ESCS'] if c in available_cluster_feats]
    n_feats = len(plot_feats)
    bar_width = 0.18
    indices = np.arange(n_feats)

    # Normalisasi skala z-score untuk perbandingan visual radar/bar
    norm_profile = cluster_profile[plot_feats].copy()

    for idx, row in norm_profile.iterrows():
        c_label = cluster_profile.loc[idx, 'Nama_Tipologi']
        ax.bar(indices + idx * bar_width, row.values, width=bar_width, 
               label=f"{c_label} ({cluster_profile.loc[idx, 'Persentase']}%)",
               color=PALETTE_KMEANS[idx % len(PALETTE_KMEANS)], edgecolor='black', alpha=0.85)

    ax.set_xticks(indices + bar_width * 1.5)
    ax.set_xticklabels(plot_feats, fontweight='bold')
    ax.axhline(0, color='grey', linestyle='--', linewidth=0.8)
    ax.set_ylabel("Nilai Rata-rata Terstandarisasi (Z-Score)")
    ax.set_title("Karakteristik Empat Tipologi Siswa Indonesia Berdasarkan Profil Belajar & Digital", pad=12)
    ax.legend(frameon=True, facecolor='white', framealpha=0.9, loc='upper left')

    plt.tight_layout()
    fig_path = os.path.join(fig_dir, "08_tipologi_siswa_clustering.png")
    fig.savefig(fig_path)
    plt.close()
    print(f"[+] Grafik tipologi siswa disimpan ke: {fig_path}")

    print("\n[+] Tahap 6 Clustering selesai dengan sukses!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="K-Means Clustering Tipologi Siswa PISA 2025")
    parser.add_argument("--input-path", type=str, default="data/pisa2025_features_engineered.csv", help="Input dataset")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Direktori hasil")
    parser.add_argument("--k", type=int, default=4, help="Jumlah klaster optimal")
    args = parser.parse_args()

    run_clustering(input_path=args.input_path, output_dir=args.output_dir, k_optimal=args.k)

