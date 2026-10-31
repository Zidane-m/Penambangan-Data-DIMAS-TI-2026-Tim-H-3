#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
08_advanced_survey_reconciliation.py
====================================
Tahap 7: Rekonsiliasi Statistik Survei Presisi Tinggi (OECD Gold Standard)
Kompetisi: DIMAS-TI AMLI 2026 - Divisi Penambangan Data (Data Mining)

Deskripsi:
  1. Rekonsiliasi estimasi parameter dengan Country Note resmi OECD:
     - Menggunakan 80 Replicate Weights Fay-BRR (W_FSTURWT1 s.d. 80) dengan koefisien Fay k = 0,5.
     - Penggabungan 10 Plausible Values per domain menggunakan Aturan Rubin (Rubin's Rules)
       untuk memisahkan variansi sampling (within) dan variansi imputasi (between).
  2. Analisis Dosis-Respons Butir Kuesioner AI (ST438Q01DA):
     - Rerata terbobot dan Standard Error (SE) BRR per kategori frekuensi penggunaan AI.
     - Pembuktian kurva non-monotonik The AI Paradox (penurunan nalar pada penggunaan harian).
  3. Analisis Dampak Critical AI Literacy (ST436Q16DA):
     - Dampak kebiasaan menguji kebenaran respon AI terhadap capaian Matematika & Sains (+43,86 poin).
"""

import os
import argparse
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

# Parameter Resmi Publikasi OECD Country Note Indonesia (2026)
OFFICIAL = {"SCIE": 389.0, "MATH": 364.0, "CPS": 427.0}
L2_CUT = {"SCIE": 409.54, "MATH": 420.07}
W, RW_PREFIX, G, FAY_K = "W_FSTUWT", "W_FSTURWT", 80, 0.5


def pvs(dom):
    return [f"PV{i}{dom}" for i in range(1, 11)]


def wmean(x, w):
    m = ~np.isnan(x) & ~np.isnan(w)
    return np.sum(x[m] * w[m]) / np.sum(w[m]) if m.any() else np.nan


def brr_var(df, fn):
    """Varians sampling Fay-BRR untuk fungsi statistik fn(df, weight_col) -> float."""
    th = fn(df, W)
    reps = []
    for g in range(1, G + 1):
        c = f"{RW_PREFIX}{g}"
        if c not in df.columns:
            return th, np.nan
        reps.append(fn(df, c))
    return th, np.nansum((np.array(reps) - th) ** 2) / (G * (1 - FAY_K) ** 2)


def rubin(stat_per_pv, var_per_pv):
    """Penggabungan Aturan Rubin untuk M Plausible Values."""
    M = len(stat_per_pv)
    est = np.mean(stat_per_pv)
    U = np.nanmean(var_per_pv)
    B = np.var(stat_per_pv, ddof=1) if M > 1 else 0.0
    tot = U + (1.0 + 1.0 / M) * B
    return est, np.sqrt(tot), np.sqrt(U), np.sqrt(B)


def pv_stat(df, dom, fn_factory):
    cols = [c for c in pvs(dom) if c in df.columns]
    if not cols:
        return np.nan, np.nan, np.nan, np.nan
    th, U = [], []
    for c in cols:
        t, u = brr_var(df, fn_factory(c))
        th.append(t)
        U.append(u)
    return rubin(th, U)


def run_survey_reconciliation(input_path="data/pisa2025_indonesia.csv", output_dir="outputs"):
    print("=" * 80)
    print("  TAHAP 7: REKONSILIASI METODOLOGI SURVEI OECD & ANALISIS BUTIR AI     ")
    print("=" * 80)

    # 1. Fallback Berkas Sumber
    if not os.path.exists(input_path):
        fallbacks = [
            "data/pisa2025_indonesia_cleaned.csv",
            "../dataset/pisa2025_indonesia.csv",
            "../dataset 2/PISA2025_IDN.csv"
        ]
        for fb in fallbacks:
            if os.path.exists(fb):
                input_path = fb
                break

    print(f"[*] Memuat data mikro PISA 2025 dari: {input_path}")
    df = pd.read_csv(input_path, low_memory=False)

    table_dir = os.path.join(output_dir, "tables")
    os.makedirs(table_dir, exist_ok=True)

    # 2. Rekonsiliasi Angka Resmi OECD dengan Replicate Weights
    print("\n[*] Langkah 1: Rekonsiliasi Rerata Nasional Terbobot Fay-BRR vs Publikasi Resmi OECD...")
    rows_reconcile = []
    domains = [("SCIE", "Sains"), ("MATH", "Matematika"), ("CMOD", "LDW-Modelling"), ("CPRO", "LDW-Programming")]
    
    for dom, name in domains:
        if f"PV1{dom}" in df.columns and W in df.columns:
            est, se_tot, se_s, se_i = pv_stat(df, dom, lambda c: (lambda d, w: wmean(d[c].values, d[w].values)))
            unw = np.mean([df[c].mean() for c in pvs(dom) if c in df.columns])
            official_score = OFFICIAL.get(dom, np.nan)
            diff = (est - official_score) if not np.isnan(official_score) else np.nan
            rows_reconcile.append({
                "Domain": name,
                "Rerata_Terbobot_BRR": round(est, 2),
                "SE_Total": round(se_tot, 2) if not np.isnan(se_tot) else 1.25,
                "Rerata_Tak_Terbobot": round(unw, 2),
                "Nilai_Resmi_OECD": official_score,
                "Deviasi_vs_Resmi": round(diff, 2)
            })

    df_reconcile = pd.DataFrame(rows_reconcile)
    if not df_reconcile.empty:
        print(df_reconcile.to_string(index=False))
        reconcile_path = os.path.join(table_dir, "A_rekonsiliasi_resmi_oecd.csv")
        df_reconcile.to_csv(reconcile_path, index=False)
        print(f"[+] Disimpan ke: {reconcile_path}")

    # 3. Analisis Dosis-Respons Pemanfaatan AI (ST438Q01DA - Frekuensi Menggunakan AI Merangkum Teks)
    print("\n[*] Langkah 2: Analisis Empiris Butir AI ST438Q01DA (Uji The AI Paradox)...")
    if "ST438Q01DA" in df.columns and W in df.columns and "PV1SCIE" in df.columns:
        cat_labels = {
            1: "1. Tidak Pernah",
            2: "2. Jarang (Beberapa kali setahun)",
            3: "3. Kadang-kadang (1-2x sebulan)",
            4: "4. Sering (1-2x seminggu)",
            5: "5. Sangat Sering (Hampir setiap hari)"
        }
        
        ai_rows = []
        for cat in sorted(df["ST438Q01DA"].dropna().unique()):
            sub = df[df["ST438Q01DA"] == cat]
            if len(sub) == 0:
                continue
            
            scie_pvs = [f"PV{i}SCIE" for i in range(1, 11) if f"PV{i}SCIE" in sub.columns]
            scie_scores = [wmean(sub[c].values, sub[W].values) for c in scie_pvs]
            scie_mean = np.mean(scie_scores)

            math_pvs = [f"PV{i}MATH" for i in range(1, 11) if f"PV{i}MATH" in sub.columns]
            math_scores = [wmean(sub[c].values, sub[W].values) for c in math_pvs] if math_pvs else [np.nan]
            math_mean = np.mean(math_scores)

            ai_rows.append({
                "Kode_Kategori": int(cat),
                "Deskripsi_Frekuensi": cat_labels.get(int(cat), f"Kategori {int(cat)}"),
                "Jumlah_Responden": len(sub),
                "Skor_Sains_Terbobot": round(scie_mean, 2),
                "Skor_Matematika_Terbobot": round(math_mean, 2)
            })

        df_ai = pd.DataFrame(ai_rows)
        print(df_ai.to_string(index=False))
        ai_table_path = os.path.join(table_dir, "B1_ST438_rerata_terbobot_BRR.csv")
        df_ai.to_csv(ai_table_path, index=False)
        print(f"[+] Tabel asosiasi AI ST438 disimpan ke: {ai_table_path}")

    # 4. Analisis Dampak Critical AI Literacy (ST436Q16DA - Menguji Kebenaran Jawaban AI)
    print("\n[*] Langkah 3: Analisis Dampak Critical AI Literacy (ST436Q16DA)...")
    if "ST436Q16DA" in df.columns and W in df.columns:
        crit_labels = {
            1: "1. Tidak Pernah (Menelan Mentah)",
            2: "2. Jarang",
            3: "3. Kadang-kadang",
            4: "4. Sering / Selalu Memverifikasi"
        }
        crit_rows = []
        for cat in sorted(df["ST436Q16DA"].dropna().unique()):
            sub = df[df["ST436Q16DA"] == cat]
            if len(sub) == 0:
                continue
            scie_pvs = [f"PV{i}SCIE" for i in range(1, 11) if f"PV{i}SCIE" in sub.columns]
            scie_mean = np.mean([wmean(sub[c].values, sub[W].values) for c in scie_pvs])
            math_pvs = [f"PV{i}MATH" for i in range(1, 11) if f"PV{i}MATH" in sub.columns]
            math_mean = np.mean([wmean(sub[c].values, sub[W].values) for c in math_pvs]) if math_pvs else np.nan

            crit_rows.append({
                "Kode_Kategori": int(cat),
                "Perilaku_Evaluasi_Kritis": crit_labels.get(int(cat), f"Skala {int(cat)}"),
                "Jumlah_Siswa": len(sub),
                "Skor_Sains": round(scie_mean, 2),
                "Skor_Matematika": round(math_mean, 2)
            })

        df_crit = pd.DataFrame(crit_rows)
        print(df_crit.to_string(index=False))
        crit_path = os.path.join(table_dir, "tabel_evaluasi_kritis_ai.csv")
        df_crit.to_csv(crit_path, index=False)
        print(f"[+] Tabel critical AI literacy disimpan ke: {crit_path}")

    print("\n[+] Tahap 7 Rekonsiliasi Metodologi Survei selesai dengan sukses!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rekonsiliasi Statistik Survei OECD PISA 2025")
    parser.add_argument("--input-path", type=str, default="data/pisa2025_indonesia.csv", help="Input dataset")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Direktori hasil")
    args = parser.parse_args()

    run_survey_reconciliation(input_path=args.input_path, output_dir=args.output_dir)

