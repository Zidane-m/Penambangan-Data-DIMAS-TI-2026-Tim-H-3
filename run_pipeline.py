#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_pipeline.py
===============
Master Pipeline Runner & End-to-End Orchestrator
Kompetisi: Pagelaran Mahasiswa Pendidikan, Matematika, Sains, dan Teknologi Informasi (DIMAS-TI AMLI 2026)
Divisi: Penambangan Data (Data Mining)

Penggunaan:
  python run_pipeline.py --all            # Eksekusi seluruh alur data mining (Tahap 0 s.d. 7)
  python run_pipeline.py --quick          # Mode uji cepat (menggunakan dataset sampel 1.000 baris)
  python run_pipeline.py --step 4         # Hanya menjalankan Tahap 4 (Modeling & Benchmark)
"""

import os
import sys
import time
import argparse
import subprocess


STEPS = [
    ("01_data_extraction_ingestion.py", "Tahap 0: Ekstraksi Data Mentah & Ingestion"),
    ("02_data_cleaning_and_audit.py", "Tahap 1: Pembersihan Data, Audit Missingness & Imputasi KNN"),
    ("03_feature_engineering.py", "Tahap 2: Rekayasa Fitur Komposit & Indeks Resiliensi"),
    ("04_exploratory_data_analysis.py", "Tahap 3: Exploratory Data Analysis (EDA) & Statistik Survei"),
    ("05_model_training_and_benchmarking.py", "Tahap 4: Pelatihan Model, Validasi GroupKFold & Benchmark"),
    ("06_explainable_ai_and_insights.py", "Tahap 5: Model Explainability (TreeSHAP) & The AI Paradox"),
    ("07_unsupervised_clustering.py", "Tahap 6: Segmentasi Tipologi Siswa (K-Means Clustering)"),
    ("08_advanced_survey_reconciliation.py", "Tahap 7: Rekonsiliasi Statistik Survei OECD & Analisis Butir AI")
]


def run_command(cmd_list, description):
    print("\n" + "=" * 80)
    print(f"[*] MENJALANKAN: {description}")
    print(f"[*] Perintah  : {' '.join(cmd_list)}")
    print("=" * 80)
    t0 = time.time()
    res = subprocess.run(cmd_list)
    elapsed = time.time() - t0
    if res.returncode != 0:
        print(f"[!] GAGAL mengeksekusi {description} (Exit Code: {res.returncode})")
        return False
    print(f"[+] SELESAI dalam {elapsed:.2f} detik.")
    return True


def main():
    parser = argparse.ArgumentParser(description="Master Orchestrator Pipeline Data Mining DIMAS-TI 2026")
    parser.add_argument("--all", action="store_true", help="Jalankan seluruh tahap dari awal hingga akhir")
    parser.add_argument("--quick", action="store_true", help="Mode cepat menggunakan sampel data")
    parser.add_argument("--step", type=int, default=None, choices=range(0, 8), help="Pilih nomor tahap tertentu (0 s.d. 7)")
    args = parser.parse_args()

    python_bin = sys.executable

    print("\n" + "#" * 80)
    print("   PIPELINE PENAMBANGAN DATA DETERMINAN LITERASI MIPA SISWA INDONESIA   ")
    print("      DIMAS-TI AMLI 2026 - DIVISI PENAMBANGAN DATA (DATA MINING)        ")
    print("#" * 80)

    # Menentukan mode eksekusi
    if args.step is not None:
        script, desc = STEPS[args.step]
        cmd = [python_bin, script]
        if args.quick and args.step in [0, 1, 4]:
            cmd.append("--quick" if args.step in [0, 1] else "--fast")
        success = run_command(cmd, desc)
        if not success:
            sys.exit(1)
    else:
        # Menjalankan seluruh tahap secara berurutan
        total_start = time.time()
        for idx, (script, desc) in enumerate(STEPS):
            cmd = [python_bin, script]
            if args.quick and idx in [0, 1, 4]:
                cmd.append("--quick" if idx in [0, 1] else "--fast")
            success = run_command(cmd, desc)
            if not success:
                print(f"\n[!] Pipeline terhenti pada tahap {idx}: {desc}")
                sys.exit(1)

        total_elapsed = time.time() - total_start
        print("\n" + "#" * 80)
        print(f"[+] SELURUH PIPELINE DATA MINING SELESAI DALAM {total_elapsed:.2f} DETIK!")
        print("    Semua artefak grafik telah tersimpan di: outputs/figures/")
        print("    Semua tabel evaluasi telah tersimpan di: outputs/tables/")
        print("    Semua laporan audit telah tersimpan di : outputs/logs/")
        print("#" * 80 + "\n")


if __name__ == "__main__":
    main()

