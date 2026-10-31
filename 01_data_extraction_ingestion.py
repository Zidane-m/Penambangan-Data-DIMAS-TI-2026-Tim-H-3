#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
01_data_extraction_ingestion.py
===============================
Tahap 0: Ekstraksi Data Mentah & Ingestion (Data Extraction & Verification)
Kompetisi: DIMAS-TI AMLI 2026 - Divisi Penambangan Data (Data Mining)

Deskripsi:
  Membaca berkas mikro-data resmi OECD PISA 2025 (berformat SPSS .sav atau CSV mentah),
  melakukan penapisan sampel khusus siswa Indonesia (CNT == 'IDN' atau '360'),
  dan memvalidasi kelengkapan variabel kunci (ID, stratifikasi, bobot survei, dan skor capaian).
"""

import os
import sys
import argparse
import pandas as pd


def extract_indonesia_data(sav_path=None, csv_source=None, output_path="data/pisa2025_indonesia.csv", quick_mode=False):
    print("=" * 80)
    print("      TAHAP 0: DATA EXTRACTION & INGESTION (PISA 2025 INDONESIA)       ")
    print("=" * 80)

    # 1. Menentukan Sumber Berkas
    if sav_path and os.path.exists(sav_path):
        print(f"[*] Membaca mikro-data SPSS resmi: {sav_path}")
        try:
            import pyreadstat
            print("[*] Mengekstrak baris dengan kode negara Indonesia (IDN / 360)...")
            df, meta = pyreadstat.read_sav(sav_path)
            if "CNT" in df.columns:
                df = df[df["CNT"].astype(str).str.upper().isin(["IDN", "360"])]
            print(f"[+] Berhasil mengekstrak {len(df):,} baris siswa Indonesia dari SPSS.")
        except ImportError:
            print("[!] Library 'pyreadstat' tidak terpasang. Gunakan: pip install pyreadstat")
            sys.exit(1)
    else:
        # Fallback ke berkas CSV yang sudah diekstrak
        candidate_paths = [
            csv_source,
            "data/pisa2025_indonesia.csv",
            "../dataset/pisa2025_indonesia.csv",
            "data/sample_pisa2025_indonesia.csv",
            "../dataset 2/PISA2025_IDN.csv"
        ]
        found_csv = None
        for p in candidate_paths:
            if p and os.path.exists(p):
                found_csv = p
                break

        if not found_csv:
            print("[!] Berkas sumber mentah tidak ditemukan di jalur default.")
            print("    Silakan tentukan path berkas dengan parameter --csv-path atau --sav-path.")
            sys.exit(1)

        print(f"[*] Menggunakan berkas data CSV terdeteksi: {found_csv}")
        nrows = 1000 if quick_mode else None
        df = pd.read_csv(found_csv, low_memory=False, nrows=nrows)

    # 2. Verifikasi Kolom Kunci Sesuai Framework PISA 2025
    essential_cols = {
        "ID & Stratifikasi": ["CNTSCHID", "CNTSTUID", "STRATUM"],
        "Bobot Sampling Populasi": ["W_FSTUWT"],
        "Plausible Values Sains": [f"PV{i}SCIE" for i in range(1, 11)],
        "Indeks Sosio-Ekonomi": ["ESCS", "HOMEPOS", "ICTRES"]
    }

    print("\n[*] Audit Kelengkapan Fitur Kunci:")
    for group_name, cols in essential_cols.items():
        present = [c for c in cols if c in df.columns]
        status = "LENGKAP" if len(present) == len(cols) else f"PARSIAL ({len(present)}/{len(cols)})"
        print(f"    - {group_name:<26}: {status}")

    # 3. Simpan Hasil Ekstraksi
    if quick_mode and output_path == "data/pisa2025_indonesia.csv":
        output_path = "data/sample_pisa2025_indonesia.csv"
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    if not os.path.exists(output_path) or os.path.abspath(output_path) != os.path.abspath(found_csv if 'found_csv' in locals() and found_csv else ""):
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"\n[+] Data siap latih berhasil disimpan ke: {output_path}")
    else:
        print(f"\n[+] Berkas data target sudah terverifikasi di: {output_path}")

    print(f"[+] Total Baris Terdaftar: {len(df):,} siswa | Kolom: {len(df.columns)}")
    print("=" * 80 + "\n")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ekstraksi Data Mentah PISA 2025 Indonesia")
    parser.add_argument("--sav-path", type=str, default=None, help="Path ke file mentah SPSS CY09_MS_STU_PUF.sav")
    parser.add_argument("--csv-path", type=str, default=None, help="Path ke file CSV PISA 2025")
    parser.add_argument("--output-path", type=str, default="data/pisa2025_indonesia.csv", help="Lokasi penyimpanan CSV target")
    parser.add_argument("--quick", action="store_true", help="Mode cepat (1.000 baris sampel)")
    args = parser.parse_args()

    extract_indonesia_data(sav_path=args.sav_path, csv_source=args.csv_path, output_path=args.output_path, quick_mode=args.quick)
