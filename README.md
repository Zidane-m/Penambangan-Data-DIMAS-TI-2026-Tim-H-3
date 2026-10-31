# Repositori Kode Sumber: Divisi Penambangan Data (Data Mining)
## Pagelaran Mahasiswa Pendidikan, Matematika, Sains, dan Teknologi Informasi (DIMAS-TI AMLI 2026)

---

### Identitas Penelitian
* **Judul Karya Tulis Ilmiah**: Penambangan Data Determinan Literasi MIPA dan Paradoks Kecerdasan Buatan Siswa Indonesia: Integrasi Psikometri IRT, Gradient Boosting, dan Clustering Menuju Transformasi Sistem Cerdas Education 5.0
* **Bidang Lomba**: Divisi Penambangan Data (Data Mining)
* **Dataset Primer**: Data Mikro Resmi OECD PISA 2025 Indonesia (N = 14.168 responden siswa, 419 klaster sekolah, 38 provinsi)
* **Standar Kerangka Kerja**: CRISP-DM (Cross-Industry Standard Process for Data Mining) dan KDD (Knowledge Discovery in Databases)

---

## 1. Ringkasan Eksekutif dan Keunggulan Metodologis

Repositori ini memuat seluruh implementasi kode sumber penambangan data secara utuh, mulai dari ekstraksi data mentah dari basis data internasional OECD, sanitasi dan audit data, rekayasa fitur psikometris, eksplorasi multivariat terbobot, pemodelan prediktif ensemble terawasi, dekomposisi Explainable AI (TreeSHAP), segmentasi profil pembelajar tanpa pengawasan (K-Means), hingga reproduksi visualisasi data interaktif pada Google Colab.

### Karakteristik dan Standar Teknis:
1. **Validasi Bebas Kebocoran Klaster (Zero Cluster Leakage)**: Menerapkan 5-Fold School-Level GroupKFold Cross-Validation. Penyekatan data dilakukan pada tingkat sekolah (CNTSCHID), menjamin tidak ada siswa dari sekolah yang sama berada di set latih dan set uji secara bersamaan untuk mengatasi korelasi intrakelas sekolah (ICC = 0,501).
2. **Representasi Populasi Nyata (Sampling Weights)**: Memasukkan bobot probabilitas siswa OECD (W_FSTUWT) pada pelatihan model dan evaluasi metrik (Weighted RMSE, Weighted MAE, dan Weighted R^2), serta 80 bobot replikasi Fay-BRR (W_FSTURWT1-80) dengan koefisien k = 0,5 untuk estimasi galat baku resmi.
3. **Imputasi Multivariat Presisi Tinggi (KNN Imputer)**: Menangani nilai hilang (Missing at Random) menggunakan K-Nearest Neighbors Imputation (k = 5, distance-weighted) pada prediktor psikososial, mempertahankan struktur kovariansi variabel laten tanpa distorsi distribusi (rata-rata Mean Shift pasca-imputasi mendekati 0,000).
4. **Agregasi Plausible Values (Rubin's Rules)**: Mengintegrasikan 10 Plausible Values per domain kompetensi (Sains, Matematika, dan Computational Problem Solving) sesuai kaidah Item Response Theory (IRT) OECD.
5. **Pembuktian Empiris The AI Paradox**: Mengonfirmasi relasi non-linier terbalik antara frekuensi adopsi AI di sekolah dengan capaian sains, di mana penurunan nalar terjadi pada frekuensi harian akibat cognitive offloading tanpa panduan.
6. **Temuan Critical AI Literacy**: Membuktikan bahwa kebiasaan mengevaluasi kebenaran keluaran chatbot AI (ST436Q16DA) berasosiasi positif hingga +43,86 poin pada capaian matematika siswa.

---

## 2. Struktur Direktori Repositori

```text
DIMAS_TI_2026_Source_Code_Penambangan_Data/
│
├── README.md                                  # Dokumentasi teknis komprehensif repositori
├── requirements.txt                           # Spesifikasi dependensi pustaka Python
├── run_pipeline.py                            # Master runner eksekusi otomatis seluruh pipeline (Tahap 0 s.d. 7)
├── .gitignore                                 # Konfigurasi pengabaian berkas sementara Git
│
├── 01_data_extraction_ingestion.py            # [Tahap 0] Ekstraksi data mikro mentah SPSS (.sav) ke CSV
├── 02_data_cleaning_and_audit.py              # [Tahap 1] Sanitasi integritas ID, filter usia, KNN Imputer k=5
├── 03_feature_engineering.py                  # [Tahap 2] Agregasi 10 PV & formulasi indeks komposit domain
├── 04_exploratory_data_analysis.py            # [Tahap 3] Statistik terbobot populasi, disparitas ESCS, & korelasi
├── 05_model_training_and_benchmarking.py      # [Tahap 4] Benchmark 5 model ML (School GroupKFold) & uji paired t-test
├── 06_explainable_ai_and_insights.py          # [Tahap 5] TreeSHAP (Beeswarm, Dependence Plot, Waterfall Studi Kasus)
├── 07_unsupervised_clustering.py              # [Tahap 6] K-Means Clustering 4 tipologi siswa (Silhouette & DB Index)
├── 08_advanced_survey_reconciliation.py       # [Tahap 7] Rekonsiliasi resmi OECD (Fay-BRR & Rubin's Rules)
│
├── colab_notebooks/                           # Berkas Lampiran Resmi Makalah (Google Colab Ready)
│   ├── Lampiran_7_Reproduksi_Visualisasi_Utama.ipynb  # Notebook Colab Lampiran 7 (Gambar 1 s.d. 6 & Tabel 3, 7)
│   ├── Lampiran_7_Reproduksi_Visualisasi_Utama.py     # Script Python setara Lampiran 7
│   ├── Lampiran_8_Visualisasi_Analisis_Lanjutan.ipynb # Notebook Colab Lampiran 8 (Tabel 4, 5, 8 & Lampiran 5)
│   └── Lampiran_8_Visualisasi_Analisis_Lanjutan.py    # Script Python setara Lampiran 8
│
├── data/                                      # Direktori dataset
│   ├── PISA2025_IDN.csv                       # Data mikro resmi 14.168 siswa Indonesia (553 kolom atribut)
│   ├── PISA2025_labels.csv                    # Kamus label kuesioner dan metadata resmi OECD
│   ├── sample_pisa2025_indonesia.csv          # Sampel 1.000 baris representatif untuk uji coba kilat (Quickstart)
│   ├── pisa2025_indonesia_cleaned.csv         # Dataset hasil pembersihan KNN siap latih
│   └── data_summary/                          # Ringkasan tabel empiris pra-komputasi (CSV dan JSON)
│
├── outputs/                                   # Direktori luaran yang dihasilkan otomatis oleh pipeline
│   ├── figures/                               # Seluruh grafik visualisasi resolusi tinggi (PNG 300 DPI)
│   ├── tables/                                # Tabel benchmark model, korelasi, dan klastering (CSV)
│   └── logs/                                  # Log audit pembersihan data dan ringkasan eksekusi
│
└── utils/                                     # Modul fungsi utilitas
    ├── __init__.py
    ├── metrics.py                             # Formulasi weighted RMSE, MAE, R^2, dan aturan Rubin
    └── visual_config.py                       # Standarisasi gaya visual grafik publikasi ilmiah (300 DPI)
```

---

## 3. Prasyarat Sistem dan Instalasi

### Prasyarat
* Python versi 3.10 atau versi lebih baru
* Sistem Operasi: Windows 10/11, macOS, atau Linux
* Memori RAM: Minimal 8 GB (Disarankan 16 GB untuk komputasi SHAP skala penuh)

### Langkah Pemasangan
1. Buka terminal atau PowerShell, lalu arahkan ke direktori repositori:
   ```bash
   cd DIMAS_TI_2026_Source_Code_Penambangan_Data
   ```

2. Buat dan aktifkan virtual environment Python:
   ```bash
   # Windows (PowerShell)
   python -m venv .venv
   .venv\Scripts\Activate.ps1

   # Linux / macOS
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Pasang seluruh dependensi pustaka:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

---

## 4. Panduan Eksekusi Pipeline

Repositori ini menyediakan fleksibilitas eksekusi, baik melalui otomasi satu perintah maupun secara modular per tahapan.

### Opsi A: Eksekusi Otomatis (Master Pipeline Runner)
Gunakan skrip `run_pipeline.py` untuk mengeksekusi tahapan secara sekuensial:

```bash
# 1. Mode Uji Cepat (Quick Mode ~60 detik)
# Menggunakan sampel representatif 1.000 baris untuk verifikasi sintaks dan pengujian fungsi tanpa kendala memori
python run_pipeline.py --quick

# 2. Mode Penuh (Full Pipeline ~5-8 menit)
# Mengeksekusi seluruh 14.168 baris data mikro PISA 2025 Indonesia dari awal hingga akhir
python run_pipeline.py --all
```

### Opsi B: Eksekusi Modular Per Tahap
Setiap modul dapat dijalankan secara terpisah:

| Tahapan | Perintah Eksekusi | Deskripsi dan Luaran |
|---|---|---|
| **Tahap 0** | `python 01_data_extraction_ingestion.py` | Ekstraksi raw SPSS (.sav) dan validasi struktur data mikro nasional. |
| **Tahap 1** | `python 02_data_cleaning_and_audit.py` | Sanitasi ID, filter usia, KNN Imputer k=5, dan laporan Mean Shift. |
| **Tahap 2** | `python 03_feature_engineering.py` | Agregasi 10 PV target dan pembentukan 30 fitur prediktor Lampiran 1. |
| **Tahap 3** | `python 04_exploratory_data_analysis.py` | Statistik deskriptif terbobot populasi, matriks korelasi, dan disparitas ESCS. |
| **Tahap 4** | `python 05_model_training_and_benchmarking.py` | Benchmark 5 model ML (School GroupKFold) dan uji paired t-test. |
| **Tahap 5** | `python 06_explainable_ai_and_insights.py` | Ekstraksi TreeSHAP: Beeswarm Plot, Dependence Plot, dan Waterfall Siswa Resilien. |
| **Tahap 6** | `python 07_unsupervised_clustering.py` | Evaluasi Silhouette & DB Index, K-Means clustering 4 tipologi siswa (N = 12.599). |
| **Tahap 7** | `python 08_advanced_survey_reconciliation.py` | Rekonsiliasi angka resmi OECD Country Note, estimasi Fay-BRR, dan analisis butir AI. |

### Opsi C: Eksekusi Reproduksi Google Colab (Lampiran 7 dan Lampiran 8 Makalah)
Subfolder `colab_notebooks/` dirancang khusus untuk memfasilitasi verifikasi langsung oleh dewan juri sesuai tautan pada Halaman 39 Naskah Makalah:
1. `Lampiran_7_Reproduksi_Visualisasi_Utama.ipynb`: Mereproduksi Gambar 1 s.d. Gambar 6 serta Tabel 3 dan Tabel 7.
2. `Lampiran_8_Visualisasi_Analisis_Lanjutan.ipynb`: Mereproduksi Tabel 4 (The AI Paradox), Tabel 5 (Critical AI Literacy), Tabel 8 (Komparasi Domain), dan Lampiran 5 (Korelasi Linear).

Kedua berkas notebook telah dilengkapi mekanisme dual-path data loader sehingga dapat langsung dijalankan pada runtime Google Colab dengan mengunggah `PISA2025_IDN.csv`, atau dijalankan secara lokal di VS Code / Jupyter Lab.

---

## 5. Ringkasan Hasil Eksperimen dan Temuan Empiris

### A. Evaluasi Kinerja Model Machine Learning (Tabel 3 Makalah)
Skema validasi: 5-Fold School-Level GroupKFold Cross-Validation terbobot populasi (W_FSTUWT).

| Algoritma Model | Kategori Algoritma | 5-Fold CV R^2 | 5-Fold CV RMSE | Test R^2 | Test RMSE | Test MAE | Pearson r |
|---|---|---|---|---|---|---|---|
| **Ridge Regression** | Linear Baseline | 0,3621 ± 0,0131 | 49,17 ± 1,28 | 0,3550 | 49,10 | 38,87 | 0,5959 |
| **Random Forest** | Bagging Ensemble | 0,4521 ± 0,0126 | 45,57 ± 1,15 | 0,4338 | 46,00 | 36,59 | 0,6599 |
| **XGBoost Regressor** | Gradient Boosting | 0,5261 ± 0,0101 | 42,38 ± 0,98 | 0,5194 | 42,38 | 33,70 | 0,7212 |
| **LightGBM Regressor** | Gradient Boosting | **0,5269 ± 0,0101** | **42,35 ± 0,96** | **0,5244** | **42,16** | **33,42** | **0,7248** |

*Catatan Metrik*: Model non-linier LightGBM mengungguli baseline linier Ridge secara signifikan (Paired t-test: t = -4,45; p = 1,13e-02), membuktikan adanya relasi non-linier kompleks dalam data pendidikan nasional.

### B. Hubungan Frekuensi Chatbot AI terhadap Capaian Sains (Tabel 4 Makalah - The AI Paradox)
Analisis butir kuesioner ST438Q01DA (N valid = 13.684 siswa):
* Kategori 1 (Tidak pernah): 404,14 poin (+3,29 poin terhadap rerata sampel)
* Kategori 2 (Beberapa kali setahun): 382,04 poin (-18,81 poin)
* Kategori 3 (Beberapa kali sebulan): **412,94 poin (+12,09 poin - Titik Puncak)**
* Kategori 4 (Beberapa kali seminggu): 410,71 poin (+9,86 poin)
* Kategori 5 (Setiap hari): **396,97 poin (-3,88 poin - Penurunan drastis sebesar 15,97 poin dari titik puncak)**

### C. Pengaruh Keterampilan Evaluasi Kritis AI (Tabel 5 Makalah - Critical AI Literacy)
Analisis butir kuesioner ST436Q16DA (N valid = 11.562 siswa):
* Kategori "Tidak Pernah": Rerata Matematika = 368,88 | Sains = 392,02 | Proksi Level 2 = 21,1%
* Kategori "Sangat Sering": Rerata Matematika = **412,74** | Sains = **441,54** | Proksi Level 2 = **54,4%**
* Selisih Keunggulan: **+43,86 poin** pada Matematika dan **+49,52 poin** pada Sains.

### D. Empat Tipologi Siswa Indonesia Hasil K-Means Clustering (Tabel 7 Makalah)
Berdasarkan 7 variabel psikometris dan digital pada N = 12.599 kasus lengkap:
1. **Klaster 0: AI-Empowered High Achievers** (25,9% populasi, Rerata MIPA = 448,11 poin): Siswa dengan capaian unggul, fasilitas TIK memadai, dan penggunaan AI terarah.
2. **Klaster 2: Resilient Low-Tech Learners** (23,24% populasi, Rerata MIPA = 382,63 poin): Siswa dari latar belakang TIK terbatas (ICTRES = -1,71) namun berprestasi melampaui kelompok rentan berkat efikasi diri sains tertinggi (+0,41) dan iklim disiplin (+1,13).
3. **Klaster 3: Under-Resourced Strugglers** (39,60% populasi, Rerata MIPA = 372,83 poin): Siswa yang menghadapi keterbatasan ganda pada modal sosio-ekonomi dan fasilitas digital.
4. **Klaster 1: Digitally Distracted Learners** (11,21% populasi, Rerata MIPA = 360,40 poin): Kelompok berisiko dengan akses TIK tinggi namun mencatatkan frekuensi hiburan gawai di sekolah tertinggi (skor 6,60 dari skala 1-9) dan capaian belajar terendah.

---

## 6. Integritas Data dan Kepatuhan Etika

1. **Sumber Data**: Data mikro bersumber dari rilis resmi Organisation for Economic Co-operation and Development (OECD) Programme for International Student Assessment (PISA) 2025.
2. **Anonimisasi**: Seluruh pengidentifikasi individu siswa dan sekolah telah dianonimkan secara ketat menggunakan kode acak standar OECD (CNTSTUID dan CNTSCHID).
3. **Keterbukaan Penelitian (Reproducibility)**: Seluruh script dan notebook dirancang untuk memastikan dewan juri, peneliti, dan akademisi dapat mereplikasi setiap angka, tabel, dan grafik secara independen tanpa data leakage.
