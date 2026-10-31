#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
03_feature_engineering.py
=========================
Tahap 2: Rekayasa Fitur Komposit & Indeks Resiliensi Akademik
Kompetisi: DIMAS-TI AMLI 2026 - Divisi Penambangan Data (Data Mining)

Deskripsi:
  1. Agregasi Rata-rata 10 Plausible Values (PV1 - PV10) sesuai metodologi IRT OECD.
  2. Rekayasa 5 Indeks Komposit Domain Psikometris & Digital:
     - PSYCH_SAFETY: Indeks keamanan psikologis sekolah.
     - PEDAGOGICAL_CLIMATE: Kualitas iklim pedagogis & disiplin kelas sains.
     - SCIENCE_AFFECTIVE: Sinergi afektif (minat intrinsik x efikasi diri sains).
     - INTERNAL_DRIVE: Motivasi intrinsik & regulasi diri belajar.
     - DIGITAL_CAPITAL: Akumulasi aset perangkat TIK & intensitas adaptasi AI sekolah.
  3. Formulasi Indeks Resiliensi Akademik (Academic Resilience Index):
     - Menghitung deviasi capaian aktual terhadap ekspektasi latar belakang sosio-ekonomi (ESCS).
     - Flag biner IS_RESILIENT untuk siswa berprestasi di tengah kemiskinan (Low ESCS, High Score).
"""

import os
import argparse
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings('ignore')


def engineer_features(input_path="data/pisa2025_indonesia_cleaned.csv",
                       output_path="data/pisa2025_features_engineered.csv",
                       report_dir="outputs"):
    print("=" * 80)
    print("      TAHAP 2: FEATURE ENGINEERING & INDEKS RESILIENSI AKADEMIK         ")
    print("=" * 80)

    # 1. Fallback Berkas Sumber
    if not os.path.exists(input_path):
        fallbacks = [
            "data/pisa2025_indonesia.csv",
            "../dataset/pisa2025_indonesia_cleaned.csv",
            "../dataset/pisa2025_indonesia.csv"
        ]
        for fb in fallbacks:
            if os.path.exists(fb):
                input_path = fb
                break

    print(f"[*] Memuat data dari: {input_path}")
    df = pd.read_csv(input_path, low_memory=False)

    # 2. Agregasi Plausible Values (Prinsip Metodologi IRT OECD)
    print("\n[*] Langkah 1: Mengagregasi Plausible Values (10 PV per domain)...")
    scie_cols = [f'PV{i}SCIE' for i in range(1, 11) if f'PV{i}SCIE' in df.columns]
    cmps_cols = [f'PV{i}CMPS' for i in range(1, 11) if f'PV{i}CMPS' in df.columns]
    math_cols = [f'PV{i}MATH' for i in range(1, 11) if f'PV{i}MATH' in df.columns]
    read_cols = [f'PV{i}READ' for i in range(1, 11) if f'PV{i}READ' in df.columns]

    if scie_cols:
        df['TARGET_SCIE'] = df[scie_cols].mean(axis=1)
        print(f"    - TARGET_SCIE (Sains) berhasil dibentuk dari {len(scie_cols)} PV.")
    if cmps_cols:
        df['TARGET_CMPS'] = df[cmps_cols].mean(axis=1)
        print(f"    - TARGET_CMPS (Komputasi LDW) berhasil dibentuk dari {len(cmps_cols)} PV.")
    if math_cols:
        df['TARGET_MATH'] = df[math_cols].mean(axis=1)
        print(f"    - TARGET_MATH (Matematika) berhasil dibentuk dari {len(math_cols)} PV.")
    if read_cols:
        df['TARGET_READ'] = df[read_cols].mean(axis=1)
        print(f"    - TARGET_READ (Membaca) berhasil dibentuk dari {len(read_cols)} PV.")

    # 3. Konstruksi Fitur Komposit Domain Psikometris & Digital
    print("\n[*] Langkah 2: Mengonstruksi Indeks Komposit Domain...")
    
    # F1: Psychological Safety Index (Feeling Safe & Belonging - Bullied)
    if all(k in df.columns for k in ['FEELSAFE', 'BELONG', 'BULLIED']):
        df['PSYCH_SAFETY'] = (df['FEELSAFE'] + df['BELONG']) - df['BULLIED']
        print("    [+] PSYCH_SAFETY dibentuk: (FEELSAFE + BELONG) - BULLIED")

    # F2: Pedagogical & Climate Quality (Discipline + Teacher Support + Cognitive Activation)
    if all(k in df.columns for k in ['DISCLISCI', 'TEACHSUP', 'COGACSC']):
        df['PEDAGOGICAL_CLIMATE'] = df['DISCLISCI'] + df['TEACHSUP'] + df['COGACSC']
        print("    [+] PEDAGOGICAL_CLIMATE dibentuk: DISCLISCI + TEACHSUP + COGACSC")

    # F3: Science Affective Synergy (Minat x Efikasi Diri)
    if all(k in df.columns for k in ['JOYSCIE', 'EFFSCIE']):
        df['SCIENCE_AFFECTIVE'] = df['JOYSCIE'] * df['EFFSCIE']
        print("    [+] SCIENCE_AFFECTIVE dibentuk: JOYSCIE * EFFSCIE")

    # F4: Curiosity & Self-Regulation Synergy
    if all(k in df.columns for k in ['CURIO', 'SELFREG', 'ENPROBS']):
        df['INTERNAL_DRIVE'] = df['CURIO'] + df['SELFREG'] + df['ENPROBS']
        print("    [+] INTERNAL_DRIVE dibentuk: CURIO + SELFREG + ENPROBS")

    # F5: Digital Asset & AI Capital
    if all(k in df.columns for k in ['ICTRES', 'AIUSESCH']):
        df['DIGITAL_CAPITAL'] = df['ICTRES'] + df['AIUSESCH']
        print("    [+] DIGITAL_CAPITAL dibentuk: ICTRES + AIUSESCH")

    # 4. Formulasi Resiliensi Akademik (Sosio-Ekonomi vs Capaian Aktual)
    print("\n[*] Langkah 3: Menghitung Indeks Resiliensi Akademik...")
    if 'ESCS' in df.columns and 'TARGET_SCIE' in df.columns:
        valid_mask = ~df['ESCS'].isnull() & ~df['TARGET_SCIE'].isnull()
        p = np.polyfit(df.loc[valid_mask, 'ESCS'], df.loc[valid_mask, 'TARGET_SCIE'], deg=1)
        df['EXPECTED_SCIE_BY_ESCS'] = np.polyval(p, df['ESCS'])
        df['ACADEMIC_RESILIENCE'] = df['TARGET_SCIE'] - df['EXPECTED_SCIE_BY_ESCS']

        low_escs_threshold = df['ESCS'].quantile(0.25)
        national_median_score = df['TARGET_SCIE'].median()
        df['IS_RESILIENT'] = ((df['ESCS'] <= low_escs_threshold) & (df['TARGET_SCIE'] >= national_median_score)).astype(int)

        resilient_count = df['IS_RESILIENT'].sum()
        pct_resilient = (resilient_count / len(df)) * 100
        print(f"    - Garis Dasar Ekspektasi Sosio-Ekonomi: Score = {p[0]:.2f} * ESCS + {p[1]:.2f}")
        print(f"    - Ambang Kuartil Bawah ESCS (Q1): {low_escs_threshold:.3f}")
        print(f"    - Median Nasional Skor Sains: {national_median_score:.2f} poin")
        print(f"    [+] Siswa Berprestasi Resilien (IS_RESILIENT=1): {resilient_count:,} ({pct_resilient:.2f}%)")

    # 5. Simpan Dataset Hasil Rekayasa Fitur
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\n[+] Dataset rekayasa fitur berhasil disimpan ke: {output_path}")
    print(f"[+] Dimensi Akhir: {df.shape[0]:,} baris x {df.shape[1]} kolom")
    print("=" * 80 + "\n")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rekayasa Fitur Komposit & Resiliensi Akademik PISA 2025")
    parser.add_argument("--input-path", type=str, default="data/pisa2025_indonesia_cleaned.csv", help="Dataset bersih input")
    parser.add_argument("--output-path", type=str, default="data/pisa2025_features_engineered.csv", help="Lokasi penyimpanan fitur")
    parser.add_argument("--report-dir", type=str, default="outputs", help="Direktori luaran")
    args = parser.parse_args()

    engineer_features(input_path=args.input_path, output_path=args.output_path, report_dir=args.report_dir)

