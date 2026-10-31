# 🏆 Repositori Kode Sumber Resmi: Divisi Penambangan Data (Data Mining)
## Pagelaran Mahasiswa Pendidikan, Matematika, Sains, dan Teknologi Informasi (DIMAS-TI AMLI 2026)

---

### 📄 Judul Penelitian:
> **"PENAMBANGAN DATA DETERMINAN LITERASI MIPA DAN PARADOKS KECERDASAN BUATAN SISWA INDONESIA: INTEGRASI PSIKOMETRI IRT, GRADIENT BOOSTING, DAN CLUSTERING MENUJU TRANSFORMASI SISTEM CERDAS EDUCATION 5.0"**

* **Bidang Lomba**: Divisi Penambangan Data (Data Mining)
* **Dataset Utama**: Data Mikro Resmi OECD PISA 2025 Indonesia ($N = 14.168$ siswa, 419 sekolah)
* **Framework Standar**: CRISP-DM (*Cross-Industry Standard Process for Data Mining*) & KDD (*Knowledge Discovery in Databases*)

---

## 📌 Ringkasan Eksekutif & Keunggulan Metodologis

Repositori ini menyatukan seluruh rangkaian kode sumber (*source code*) penambangan data secara profesional, mulai dari penapisan data mentah, audit kualitas dan sanitasi, rekayasa fitur domain, eksplorasi data survei multivariat, pemodelan prediktif terawasi, penjelas model (Explainable AI), segmentasi profil tidak terawasi (Clustering), hingga rekonsiliasi statistik survei presisi tinggi.

### 🌟 Nilai Keunggulan Metodologis (Technical Highlights):
1. **Pencegahan Kebocoran Klaster (Zero Cluster Leakage)**: Menerapkan skema validasi silang **5-Fold School-Level GroupKFold Cross-Validation** berbobot populasi. Tidak ada data siswa dari sekolah yang sama yang bocor antara set latih dan validasi (*Intraclass Correlation Coefficient* $\text{ICC} = 0{,}501$).
2. **Representasi Populasi Nyata**: Memasukkan bobot survei resmi OECD (`W_FSTUWT`) pada seluruh metrik evaluasi (*Weighted RMSE, MAE, R²*) dan 80 *Replicate Weights* Fay-BRR (`W_FSTURWT1-80`) dengan koefisien Fay $k = 0{,}5$.
3. **Agregasi Plausible Values (Rubin's Rules)**: Mengintegrasikan 10 *Plausible Values* (PV1 s.d. PV10) per domain literasi menggunakan aturan penggabungan Rubin untuk memisahkan variansi sampling (*within*) dan variansi imputasi psikometri (*between*).
4. **Imputasi Multivariat Presisi Tinggi (KNN Imputer)**: Mempertahankan struktur kovariansi variabel laten psikososial tanpa mereduksi variansi alami data survei (*Mean Shift* pasca-imputasi mendekati $0{,}000$).
5. **Pembuktian Empiris The AI Paradox**: Mengonfirmasi kurva non-linier dosis-respons penggunaan AI generatif di sekolah (penurunan capaian pada penggunaan harian tanpa bimbingan akibat *cognitive offloading*).
6. **Kekuatan Critical AI Literacy**: Menemukan bahwa keterampilan mengevaluasi kebenaran respon AI (`ST436Q16DA`) berasosiasi positif hingga $+43{,}86$ poin pada capaian matematika siswa.

---

## 📂 Struktur Direktori Repositori

```text
DIMAS_TI_2026_Source_Code_Penambangan_Data/
│
├── README.md                                  # Dokumentasi komprehensif repositori & panduan juri
├── requirements.txt                           # Daftar dependensi library Python
├── run_pipeline.py                            # Master Runner: Eksekusi otomatis seluruh pipeline (Tahap 0-7)
│
├── 01_data_extraction_ingestion.py            # [Tahap 0] Ekstraksi data mikro resmi SPSS (.sav) ke CSV
├── 02_data_cleaning_and_audit.py              # [Tahap 1] Sanitasi ID, filter usia, audit missingness, imputasi KNN
├── 03_feature_engineering.py                  # [Tahap 2] Agregasi 10 PV & konstruksi 5 indeks domain psikometris
├── 04_exploratory_data_analysis.py            # [Tahap 3] Statistik deskriptif terbobot, disparitas ESCS, & korelasi
├── 05_model_training_and_benchmarking.py      # [Tahap 4] Pelatihan 5 model ML, 5-Fold GroupKFold, & uji signifikansi
├── 06_explainable_ai_and_insights.py          # [Tahap 5] TreeSHAP (Beeswarm, Dependence Plot, Waterfall Studi Kasus)
├── 07_unsupervised_clustering.py              # [Tahap 6] K-Means Clustering (Silhouette & Davies-Bouldin) 4 tipologi
├── 08_advanced_survey_reconciliation.py       # [Tahap 7] Rekonsiliasi OECD Country Note, Fay-BRR, & butir kuesioner AI
│
├── reproduksi_visualisasi_pisa2025.ipynb      # Google Colab Notebook terpadu reproduksi visual (Grafik 300 DPI)
│
├── colab_notebooks/                           # Berkas Lampiran Resmi Makalah (Google Colab Ready)
│   ├── Lampiran_7_Reproduksi_Visualisasi_Utama.ipynb  # [Lampiran 7] Colab reproduksi Gambar 1-6 & Tabel 3, 7
│   ├── Lampiran_7_Reproduksi_Visualisasi_Utama.py     # Script Python setara Lampiran 7
│   ├── Lampiran_8_Visualisasi_Analisis_Lanjutan.ipynb # [Lampiran 8] Colab reproduksi Tabel 4, 5, 8 & Lampiran 5
│   └── Lampiran_8_Visualisasi_Analisis_Lanjutan.py    # Script Python setara Lampiran 8
│
├── data/                                      # Direktori penyimpanan data
│   ├── PISA2025_IDN.csv                       # Data mikro resmi Indonesia (14.168 siswa, 553 kolom)
│   ├── PISA2025_labels.csv                    # Kamus label variabel resmi OECD PISA 2025
│   ├── sample_pisa2025_indonesia.csv          # Sampel representatif 1.000 baris untuk uji coba kilat (Quickstart)
│   ├── pisa2025_indonesia_cleaned.csv         # Dataset hasil pembersihan siap latih (14.168 siswa)
│   └── data_summary/                          # Ringkasan tabel empiris (CSV & JSON)
│
├── outputs/                                   # Direktori luaran hasil eksekusi pipeline
│   ├── figures/                               # Grafik visualisasi resolusi tinggi (PNG 300 DPI)
│   ├── tables/                                # Tabel benchmark model, korelasi, dan klastering (CSV)
│   └── logs/                                  # Catatan audit data cleaning & log eksekusi
│
└── utils/                                     # Modul utilitas
    ├── __init__.py
    ├── metrics.py                             # Formulasi weighted RMSE, MAE, R², dan aturan Rubin
    └── visual_config.py                       # Standarisasi gaya visual grafik publikasi ilmiah (300 DPI)
```

---

## ⚡ Panduan Cepat Menjalankan Kode (Quickstart Guide)

### 1. Prasyarat Lingkungan (Environment Setup)
Pastikan Python 3.10 atau versi lebih baru telah terpasang. Buat dan aktifkan *virtual environment*:

```bash
# Membuat virtual environment
python -m venv .venv

# Aktivasi di Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Aktivasi di Linux / macOS
source .venv/bin/activate

# Memasang seluruh dependensi
pip install -r requirements.txt
```

### 2. Eksekusi Otomatis Sekali Perintah (Master Pipeline Runner)

Anda dapat menjalankan seluruh pipeline dari Tahap 0 sampai Tahap 7 secara berurutan dengan perintah:

```bash
# Menjalankan seluruh alur data mining end-to-end
python run_pipeline.py --all

# Mode Uji Cepat (Menggunakan dataset sampel 1.000 baris untuk verifikasi sintaks dalam ~15 detik)
python run_pipeline.py --quick
```

### 3. Eksekusi Modular Per Tahapan

Setiap tahapan dirancang modular dan dapat dijalankan secara terpisah:

| Tahap | Perintah Eksekusi | Deskripsi & Luaran Utama |
|---|---|---|
| **Tahap 0** | `python 01_data_extraction_ingestion.py` | Ekstraksi raw data SPSS `.sav` / verifikasi dataset CSV Indonesia. |
| **Tahap 1** | `python 02_data_cleaning_and_audit.py` | Sanitasi integritas ID, filter usia (14.5-17.5), imputasi KNN $k=5$, & laporan Mean Shift. |
| **Tahap 2** | `python 03_feature_engineering.py` | Agregasi 10 PV target & pembuatan 5 indeks komposit domain serta indeks resiliensi. |
| **Tahap 3** | `python 04_exploratory_data_analysis.py` | Analisis statistik terbobot populasi, matriks korelasi, dan visualisasi distribusi skor. |
| **Tahap 4** | `python 05_model_training_and_benchmarking.py` | Benchmark Ridge, RF, XGBoost, LightGBM (5-Fold School GroupKFold) & Paired t-test. |
| **Tahap 5** | `python 06_explainable_ai_and_insights.py` | Ekstraksi TreeSHAP: Beeswarm Plot, Dependence Plot (AI Paradox), & Waterfall Siswa Resilien. |
| **Tahap 6** | `python 07_unsupervised_clustering.py` | Evaluasi Silhouette & Davies-Bouldin, segmentasi 4 tipologi profil siswa Indonesia. |
| **Tahap 7** | `python 08_advanced_survey_reconciliation.py` | Rekonsiliasi angka resmi OECD, estimasi Fay-BRR (80 replicate weights), & analisis butir AI. |

---

## 📊 Hasil Benchmark Model Machine Learning (Tahap 4)

Berdasarkan validasi silang **5-Fold School-Level GroupKFold** terbobot populasi (`W_FSTUWT`):

| Model Algoritma | Kategori Model | Test RMSE (Poin) | Test MAE (Poin) | 5-Fold CV $R^2$ | Uji Signifikansi vs Linear ($p$-value) |
|---|---|---|---|---|---|
| **Baseline (Dummy Mean)** | Naive Baseline | $60{,}92 \pm 0{,}84$ | $49{,}82 \pm 0{,}65$ | $-0{,}0012 \pm 0{,}0015$ | - |
| **Ridge Regression** | Interpretable Linear | $48{,}85 \pm 0{,}92$ | $39{,}14 \pm 0{,}71$ | $0{,}3550 \pm 0{,}0184$ | Baseline |
| **Random Forest** | Bagging Tree Ensemble | $42{,}88 \pm 0{,}76$ | $33{,}95 \pm 0{,}62$ | $0{,}5082 \pm 0{,}0121$ | $p < 0{,}0001$ (Signifikan) |
| **XGBoost Regressor** | Gradient Boosting | $42{,}21 \pm 0{,}68$ | $33{,}41 \pm 0{,}55$ | $0{,}5238 \pm 0{,}0115$ | $p < 0{,}0001$ (Signifikan) |
| **LightGBM Regressor** | Gradient Boosting (Terpilih) | **$42{,}16 \pm 0{,}65$** | **$33{,}38 \pm 0{,}52$** | **$0{,}5269 \pm 0{,}0101$** | **$p = 3{,}14 \times 10^{-5}$** (Signifikan) |

> **Temuan Kunci**: Model non-linier *LightGBM* mengungguli baseline linier *Ridge* dengan penurunan galat $\Delta\text{RMSE} = -6{,}69$ poin dan peningkatan daya jelas variansi $\Delta R^2 = +17{,}19\%$, membuktikan adanya relasi non-linier kuat dalam ekosistem pendidikan Indonesia.

---

## 🔍 Temuan Explainable AI & Segmentasi Siswa

1. **The AI Paradox**: Siswa yang menggunakan AI dengan frekuensi sedang/bulanan memiliki capaian sains tertinggi ($412{,}94$ poin), sedangkan siswa yang menggunakan AI secara intensif harian tanpa pengawasan guru mengalami penurunan drastis hingga $396{,}97$ poin (penurunan $-15{,}97$ poin) akibat ketergantungan kognitif (*cognitive offloading*).
2. **Empat Tipologi Profil Siswa Indonesia (K-Means Clustering, $k=4$)**:
   - **Klaster 1: Balanced Achievers** ($23{,}8\%$): Prestasi tinggi ($438{,}2$ poin), efikasi diri positif, pemanfaatan AI terarah.
   - **Klaster 2: High-Tech At-Risk** ($21{,}4\%$): Sarana digital lengkap & penggunaan AI intensif, namun disiplin rendah dan prestasi menurun ($374{,}5$ poin).
   - **Klaster 3: Resilient Low-Tech Strivers** ($27{,}6\%$): Sarana digital terbatas dan SES rendah, namun capaian sains tinggi ($411{,}8$ poin) berkat efikasi diri dan regulasi diri yang tangguh.
   - **Klaster 4: Vulnerable Low-Resource** ($27{,}2\%$): Menghadapi kerentanan ganda (ketiadaan akses teknologi dan iklim sekolah yang kurang mendukung, rata-rata skor $358{,}1$ poin).

---

## 📓 Reproduksi Visualisasi Data di Google Colab

Seluruh grafik publikasi dapat direproduksi secara interaktif melalui notebook:
[`reproduksi_visualisasi_pisa2025.ipynb`](reproduksi_visualisasi_pisa2025.ipynb).

Fitur notebook:
- **Dual-Mode Data Loader**: Dapat otomatis mendeteksi file dataset CSV lokal, atau beralih ke mode *Standalone Empirical Matrix* jika dijalankan di lingkungan Google Colab tanpa perlu mengunggah berkas dataset besar.
- Menghasilkan 6 gambar utama beresolusi 300 DPI dengan palet warna ramah buta warna (*colorblind-friendly*).

---

## ⚖️ Lisensi & Kepatuhan Etika Data

Dataset PISA 2025 adalah data mikro publik berlisensi resmi dari *Organisation for Economic Co-operation and Development* (OECD). Seluruh data siswa telah dianonimkan (*fully anonymized*). Penggunaan kode ini ditujukan untuk kepentingan kompetisi akademik DIMAS-TI AMLI 2026 dan pengembangan ilmu pengetahuan (*Educational Data Mining*).

