# ============================================================================
# LAMPIRAN 7 (VERSI PERBAIKAN): Reproduksi Grafik Utama Makalah
# Semua angka DIHITUNG dari PISA2025_IDN.csv. Tidak ada angka yang diketik manual.
#
# Perbaikan dibanding versi lama:
#  1. Gambar 1  : metrik dihitung dari model asli (Ridge, RF, XGBoost, LightGBM) dengan
#                 5-Fold School-Level GroupKFold. CatBoost dihapus (tidak ada di Tabel 3).
#  2. Gambar 2  : memakai prediksi model asli pada data uji (versi lama = data simulasi+noise).
#  3. Gambar 4  : feature importance dihitung dari model (versi lama = angka diketik). Jenis "split"
#                 dipakai agar sebanding dengan makalah; "gain" disimpan sebagai pembanding.
#  4. Gambar 5  : SHAP memakai 30 fitur penuh (versi lama hanya 11 fitur, tanpa imputasi/GroupKFold).
#  5. Gambar 6  : K-Means BENAR-BENAR dijalankan (versi lama tidak ada KMeans dan angkanya manual).
#                 Gawai hiburan ditulis "skor skala 1-9", BUKAN "jam/hari".
#  6. Tambahan  : tabel pemilihan k (elbow, Silhouette, Davies-Bouldin untuk k = 2..8).
#
# Cara pakai di Google Colab: jalankan sel berurutan; unggah PISA2025_IDN.csv saat diminta.
# Pasang paket bila perlu:  !pip install lightgbm xgboost shap
# Penanda "# %%" memisahkan sel (Colab/VS Code membacanya sebagai sel terpisah).
# ============================================================================

# %% [SEL 1] Setup, pemuatan data, dan target psikometrik
import os, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({"font.family": "DejaVu Sans", "figure.dpi": 150})

SEED = 42
OUT = "figures"
os.makedirs(OUT, exist_ok=True)


def load_data(path="PISA2025_IDN.csv"):
    candidates = [path, os.path.join("data", path), os.path.join("..", "data", path), os.path.join("..", path)]
    for p in candidates:
        if os.path.exists(p):
            return pd.read_csv(p, low_memory=False)
    from google.colab import files  # hanya jalan di Colab
    print("Pilih berkas PISA2025_IDN.csv")
    name = list(files.upload().keys())[0]
    return pd.read_csv(name, low_memory=False)


df = load_data()
for subj in ["SCIE", "MATH", "CMOD", "CPRO", "CMPS"]:
    df[f"TARGET_{subj}"] = df[[f"PV{i}{subj}" for i in range(1, 11)]].mean(axis=1)
df["TARGET_MIPA"] = df[[f"PV{i}{s}" for s in ["SCIE", "MATH"] for i in range(1, 11)]].mean(axis=1)

W = df["W_FSTUWT"].values
print(f"Data dimuat: {len(df):,} siswa, {df['CNTSCHID'].nunique()} sekolah, {df.shape[1]} kolom")
print(f"Rerata MIPA (tidak terbobot): {df['TARGET_MIPA'].mean():.2f} | "
      f"SD: {df['TARGET_MIPA'].std():.2f} | "
      f"terbobot: {np.average(df['TARGET_MIPA'], weights=W):.2f}")

# %% [SEL 2] 30 fitur prediktor (sesuai Lampiran 1) dan fungsi metrik
FEATURES = {
    # kode: (nama tampil, kategori)
    "HISEI": ("Pekerjaan Tertinggi Orang Tua", "Sosio-ekonomi"),
    "COGABIL": ("Adaptabilitas Kognitif", "Karakter & Kognisi"),
    "GOALSET": ("Penetapan Tujuan", "Karakter & Kognisi"),
    "SELFREG": ("Regulasi Diri", "Karakter & Kognisi"),
    "ENPROBS": ("Minat Pemecahan Masalah", "Karakter & Kognisi"),
    "PAREDINT": ("Pendidikan Orang Tua (Tahun)", "Sosio-ekonomi"),
    "CURIO": ("Rasa Ingin Tahu", "Karakter & Kognisi"),
    "IC173Q02JA": ("Frekuensi TIK di Kelas Matematika", "Edu-Tech & AI"),
    "ST326Q05JA": ("Gawai Hiburan di Luar Sekolah", "Edu-Tech & AI"),
    "IC173Q03JA": ("Frekuensi TIK di Kelas Sains", "Edu-Tech & AI"),
    "AIUSESCH": ("Penggunaan AI di Sekolah", "Edu-Tech & AI"),
    "ST326Q04JA": ("Gawai Hiburan di Sekolah", "Edu-Tech & AI"),
    "ST326Q02JA": ("Gawai Belajar di Luar Sekolah", "Edu-Tech & AI"),
    "ST250Q03JA": ("Software Edukasi di Rumah", "Sosio-ekonomi"),
    "ST326Q01JA": ("Gawai Belajar di Sekolah", "Edu-Tech & AI"),
    "ST097Q06DA": ("Distraksi Gawai saat Jam Sains", "Edu-Tech & AI"),
    "ESCS": ("Indeks Sosial-Ekonomi & Budaya", "Sosio-ekonomi"),
    "COGACSC": ("Aktivasi Kognitif Guru", "Iklim Belajar & Pedagogi"),
    "STBELSCIE": ("Keyakinan Hakikat Sains", "Karakter & Kognisi"),
    "EFFSCIE": ("Efikasi Diri Sains", "Karakter & Kognisi"),
    "SCIECOLLR": ("Penalaran Kolaboratif Sains", "Iklim Belajar & Pedagogi"),
    "ST250Q02JA": ("Komputer Rumah untuk Tugas", "Sosio-ekonomi"),
    "TEACHSUP": ("Dukungan Guru Sains", "Iklim Belajar & Pedagogi"),
    "JOYSCIE": ("Kesenangan Belajar Sains", "Karakter & Kognisi"),
    "DISCLISCI": ("Iklim Disiplin Kelas Sains", "Iklim Belajar & Pedagogi"),
    "BELONG": ("Rasa Memiliki terhadap Sekolah", "Iklim Belajar & Pedagogi"),
    "FEELSAFE": ("Rasa Aman di Sekolah", "Iklim Belajar & Pedagogi"),
    "ICTRES": ("Ketersediaan Sumber Daya TIK", "Edu-Tech & AI"),
    "HOMEPOS": ("Kepemilikan Fasilitas Rumah", "Sosio-ekonomi"),
    "MALE": ("Gender (Laki-laki)", "Sosio-ekonomi"),
}
COLS = list(FEATURES)
CAT_COLOR = {"Edu-Tech & AI": "#1f77b4", "Karakter & Kognisi": "#ff7f0e",
             "Sosio-ekonomi": "#2ca02c", "Iklim Belajar & Pedagogi": "#9467bd"}
assert len(COLS) == 30 and all(c in df.columns for c in COLS)

X = df[COLS].values
y = df["TARGET_MIPA"].values
groups = df["CNTSCHID"].values


def metrics(y_true, y_pred, w=None):
    """R2, RMSE, MAE; w=None -> tidak terbobot, w=array -> terbobot survei."""
    w = np.ones(len(y_true)) if w is None else w
    ybar = np.average(y_true, weights=w)
    r2 = 1 - np.sum(w * (y_true - y_pred) ** 2) / np.sum(w * (y_true - ybar) ** 2)
    rmse = np.sqrt(np.average((y_true - y_pred) ** 2, weights=w))
    mae = np.average(np.abs(y_true - y_pred), weights=w)
    return r2, rmse, mae


# %% [SEL 3] Gambar 1: perbandingan 4 algoritma (GroupKFold level sekolah + uji sekolah terpisah)
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def make_model(name):
    imp = SimpleImputer(strategy="median")  # imputasi median DI DALAM pipeline -> hanya belajar dari data latih
    if name == "Ridge":
        return make_pipeline(imp, StandardScaler(), Ridge(alpha=1.0))
    if name == "Random Forest":
        return make_pipeline(imp, RandomForestRegressor(
            n_estimators=200, min_samples_leaf=10, max_features=0.5, n_jobs=-1, random_state=SEED))
    if name == "XGBoost":
        return make_pipeline(imp, xgb.XGBRegressor(
            n_estimators=300, learning_rate=0.03, max_depth=4, subsample=0.8,
            colsample_bytree=0.8, random_state=SEED, n_jobs=-1))
    if name == "LightGBM":
        return make_pipeline(imp, lgb.LGBMRegressor(
            n_estimators=300, learning_rate=0.03, num_leaves=15, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.8, random_state=SEED, verbose=-1, n_jobs=-1))


MODELS = ["Ridge", "Random Forest", "XGBoost", "LightGBM"]

# Uji akhir: 20% SEKOLAH ditahan (tidak ada sekolah yang muncul di data latih dan uji sekaligus)
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
tr_idx, te_idx = next(gss.split(X, y, groups))
assert not set(groups[tr_idx]) & set(groups[te_idx]), "Kebocoran sekolah!"

rows, preds = [], {}
for name in MODELS:
    cv_r2u, cv_rmu, cv_r2w, cv_rmw = [], [], [], []
    for a, b in GroupKFold(n_splits=5).split(X[tr_idx], y[tr_idx], groups[tr_idx]):
        ia, ib = tr_idx[a], tr_idx[b]
        m = make_model(name).fit(X[ia], y[ia])
        p = m.predict(X[ib])
        r2u, rmu, _ = metrics(y[ib], p)
        r2w, rmw, _ = metrics(y[ib], p, W[ib])
        cv_r2u.append(r2u); cv_rmu.append(rmu); cv_r2w.append(r2w); cv_rmw.append(rmw)
    final = make_model(name).fit(X[tr_idx], y[tr_idx])
    p = final.predict(X[te_idx])
    preds[name] = (final, p)
    r2u, rmu, mau = metrics(y[te_idx], p)
    r2w, rmw, maw = metrics(y[te_idx], p, W[te_idx])
    rows.append({
        "Model": name,
        "CV R2 (tak terbobot)": f"{np.mean(cv_r2u):.4f} ± {np.std(cv_r2u):.4f}",
        "CV RMSE (tak terbobot)": f"{np.mean(cv_rmu):.2f}",
        "CV R2 (terbobot)": f"{np.mean(cv_r2w):.4f} ± {np.std(cv_r2w):.4f}",
        "CV RMSE (terbobot)": f"{np.mean(cv_rmw):.2f}",
        "Test R2": round(r2u, 4), "Test RMSE": round(rmu, 2), "Test MAE": round(mau, 2),
        "Test R2 (terbobot)": round(r2w, 4), "Test RMSE (terbobot)": round(rmw, 2),
        "Pearson r": round(float(np.corrcoef(y[te_idx], p)[0, 1]), 4),
    })
    print(f"selesai: {name}")

res = pd.DataFrame(rows)
res.to_csv(f"{OUT}/tabel3_hasil_model.csv", index=False)
print("\nTABEL 3 (hasil dari kode ini; samakan angka makalah dengan tabel ini):")
print(res.to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
colors = ["#4c78a8", "#72b7b2", "#f58518", "#54a24b"]
axes[0].bar(res["Model"], res["Test R2"], color=colors)
axes[0].set_ylabel("Test R²"); axes[0].set_title("Test R² (makin tinggi makin baik)")
axes[1].bar(res["Model"], res["Test RMSE"], color=colors)
axes[1].set_ylabel("Test RMSE (poin PISA)"); axes[1].set_title("Test RMSE (makin rendah makin baik)")
for ax, col in zip(axes, ["Test R2", "Test RMSE"]):
    for i, v in enumerate(res[col]):
        ax.annotate(f"{v:.4f}" if col == "Test R2" else f"{v:.2f}", (i, v), ha="center", va="bottom",
                    fontsize=9, fontweight="bold")
    ax.tick_params(axis="x", rotation=15)
plt.suptitle("Gambar 1. Perbandingan Kinerja Algoritma (uji pada sekolah yang tidak dipakai melatih)", y=1.02)
plt.tight_layout(); plt.savefig(f"{OUT}/gambar1_perbandingan_model.png", bbox_inches="tight"); plt.show()

# %% [SEL 4] Gambar 2: aktual vs prediksi (prediksi ASLI model LightGBM pada data uji)
lgbm, p_lgb = preds["LightGBM"]
y_te = y[te_idx]
r = np.corrcoef(y_te, p_lgb)[0, 1]
fig, ax = plt.subplots(figsize=(6.5, 6))
ax.scatter(y_te, p_lgb, alpha=0.25, s=18, color="#1f77b4", edgecolors="none",
           label=f"Siswa uji (N = {len(y_te):,})")
lo, hi = min(y_te.min(), p_lgb.min()) - 10, max(y_te.max(), p_lgb.max()) + 10
ax.plot([lo, hi], [lo, hi], "--", color="#d62728", lw=2, label="Garis ideal (y = x)")
m_, b_ = np.polyfit(y_te, p_lgb, 1)
xs = np.array([lo, hi]); ax.plot(xs, m_ * xs + b_, color="#2ca02c", lw=2, label=f"Tren model (r = {r:.3f})")
ax.set_xlabel("Skor MIPA aktual (Plausible Values)", fontweight="bold")
ax.set_ylabel("Skor MIPA prediksi LightGBM", fontweight="bold")
ax.set_title("Gambar 2. Aktual vs Prediksi LightGBM (data uji)"); ax.legend(loc="upper left")
plt.tight_layout(); plt.savefig(f"{OUT}/gambar2_aktual_vs_prediksi.png", bbox_inches="tight"); plt.show()

# %% [SEL 5] Gambar 3: kurva respons-dosis The AI Paradox (ST438Q01DA) + selang kepercayaan
lab = ["Kat 1\nTidak pernah", "Kat 2\nBeberapa kali/tahun", "Kat 3\nBeberapa kali/bulan",
       "Kat 4\nBeberapa kali/minggu", "Kat 5\nSetiap hari"]
fig, axes = plt.subplots(1, 2, figsize=(13.5, 5))
for ax, col, title, color in [(axes[0], "TARGET_SCIE", "Sains", "#2b5c8f"),
                              (axes[1], "TARGET_MATH", "Matematika", "#800020")]:
    g = df.dropna(subset=["ST438Q01DA"]).groupby("ST438Q01DA")[col]
    mean, se = g.mean().values, (g.std() / np.sqrt(g.count())).values
    ax.errorbar(range(5), mean, yerr=1.96 * se, marker="o", color=color, lw=2.5, capsize=4,
                label="Rerata ± 95% CI (tak terbobot)")
    ax.axhline(df[col].mean(), ls="--", color="#e6550d", label=f"Rerata sampel ({df[col].mean():.2f})")
    ax.scatter(2, mean[2], s=150, color="#2ca02c", zorder=5, label=f"Puncak Kat 3 ({mean[2]:.2f})")
    ax.scatter(4, mean[4], s=150, color="#d62728", zorder=5, label=f"Harian Kat 5 ({mean[4]:.2f})")
    for i, v in enumerate(mean):
        ax.annotate(f"{v:.2f}", (i, v + 2.2), ha="center", fontsize=8.5, fontweight="bold")
    ax.set_xticks(range(5)); ax.set_xticklabels(lab, fontsize=8)
    ax.set_title(f"The AI Paradox: {title} vs Frekuensi Chatbot AI"); ax.set_ylabel(f"Skor {title}")
    ax.legend(loc="lower right", fontsize=8)
plt.suptitle("Gambar 3. Pola non-monotonik (asosiasi, bukan sebab-akibat)", y=1.02)
plt.tight_layout(); plt.savefig(f"{OUT}/gambar3_ai_paradox.png", bbox_inches="tight"); plt.show()

# %% [SEL 6] Gambar 4: feature importance dari model LightGBM (split, dinormalkan; gain sebagai pembanding)
booster = lgbm.steps[-1][1]
# Makalah memakai importance bawaan LightGBM ("split" = seberapa sering fitur dipakai memecah node).
# Importance "gain" ikut dihitung sebagai pembanding: urutannya bisa BERBEDA (lihat kolom gain).
imp_split = booster.booster_.feature_importance(importance_type="split").astype(float)
imp_gain = booster.booster_.feature_importance(importance_type="gain")
imp = imp_split / imp_split.sum()
fi = pd.DataFrame({"kode": COLS, "importance": imp, "gain_norm": imp_gain / imp_gain.sum()})
fi["nama"] = fi["kode"].map(lambda c: FEATURES[c][0])
fi["kategori"] = fi["kode"].map(lambda c: FEATURES[c][1])
fi = fi.sort_values("importance", ascending=False).reset_index(drop=True)
fi.to_csv(f"{OUT}/lampiran4_feature_importance.csv", index=False)
top = fi.head(12).iloc[::-1]
fig, ax = plt.subplots(figsize=(9.5, 6))
ax.barh(top["nama"], top["importance"], color=[CAT_COLOR[k] for k in top["kategori"]], edgecolor="black")
for yv, v in enumerate(top["importance"]):
    ax.annotate(f"{v:.4f}", (v + 0.001, yv), va="center", fontsize=8.5)
handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in CAT_COLOR.values()]
ax.legend(handles, CAT_COLOR.keys(), loc="lower right")
ax.set_xlabel("Kepentingan relatif fitur (split count, dinormalkan)")
ax.set_title("Gambar 4. 12 Fitur Paling Berpengaruh (LightGBM)")
plt.tight_layout(); plt.savefig(f"{OUT}/gambar4_feature_importance.png", bbox_inches="tight"); plt.show()
print(fi.head(12).to_string(index=False))

# %% [SEL 7] Gambar 5: SHAP beeswarm dengan 30 fitur penuh
import shap
imputer = lgbm.steps[0][1]
X_te_imp = pd.DataFrame(imputer.transform(X[te_idx]), columns=[FEATURES[c][0] for c in COLS])
sample = X_te_imp.sample(min(1500, len(X_te_imp)), random_state=SEED)
explainer = shap.TreeExplainer(booster)
sv = explainer.shap_values(sample)
plt.figure(figsize=(9, 7))
shap.summary_plot(sv, sample, max_display=14, show=False)
plt.title("Gambar 5. SHAP Beeswarm (30 fitur, imputasi median dari data latih)")
plt.tight_layout(); plt.savefig(f"{OUT}/gambar5_shap.png", bbox_inches="tight"); plt.show()

# Arah SHAP aktivasi kognitif guru (untuk dicek vs klaim "marginal negatif")
cg = FEATURES["COGACSC"][0]
rho = np.corrcoef(sample[cg].values, sv[:, list(sample.columns).index(cg)])[0, 1]
print(f"Korelasi nilai {cg} dengan SHAP-nya: {rho:+.3f}  (negatif = nilai tinggi menurunkan prediksi)")

# %% [SEL 8] K-Means: pemilihan k (elbow, Silhouette, Davies-Bouldin) untuk k = 2..8
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score

CL_FEATS = ["AIUSESCH", "ICTRES", "EFFSCIE", "ST326Q04JA", "DISCLISCI", "COGABIL", "ESCS"]
dc = df.dropna(subset=CL_FEATS).copy()          # kasus lengkap pada 7 variabel
print(f"N klaster = {len(dc):,}  (makalah: 12.599)")
Z = StandardScaler().fit_transform(dc[CL_FEATS])

scan = []
for k in range(2, 9):
    km = KMeans(n_clusters=k, n_init=20, random_state=SEED).fit(Z)
    scan.append({"k": k, "inertia": km.inertia_,
                 "silhouette": silhouette_score(Z, km.labels_, sample_size=8000, random_state=SEED),
                 "davies_bouldin": davies_bouldin_score(Z, km.labels_),
                 "klaster_terkecil_%": 100 * np.bincount(km.labels_).min() / len(Z)})
scan = pd.DataFrame(scan).round(4)
scan.to_csv(f"{OUT}/pemilihan_k.csv", index=False)
print(scan.to_string(index=False))

fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
axes[0].plot(scan["k"], scan["inertia"], "o-"); axes[0].set_title("Elbow (inertia)")
axes[1].plot(scan["k"], scan["silhouette"], "o-", color="#2ca02c"); axes[1].set_title("Silhouette (tinggi = baik)")
axes[2].plot(scan["k"], scan["davies_bouldin"], "o-", color="#d62728"); axes[2].set_title("Davies-Bouldin (rendah = baik)")
for ax in axes:
    ax.axvline(4, ls=":", color="gray"); ax.set_xlabel("k")
plt.tight_layout(); plt.savefig(f"{OUT}/gambar_pemilihan_k.png", bbox_inches="tight"); plt.show()

# %% [SEL 9] Gambar 6: tipologi siswa (K-Means k = 4), penamaan otomatis berdasarkan aturan eksplisit
K = 4
km = KMeans(n_clusters=K, n_init=20, random_state=SEED).fit(Z)
dc["klaster"] = km.labels_
sil = silhouette_score(Z, km.labels_, sample_size=8000, random_state=SEED)
dbi = davies_bouldin_score(Z, km.labels_)
print(f"k = 4 | Silhouette = {sil:.4f} | Davies-Bouldin = {dbi:.3f}")

prof = dc.groupby("klaster").agg(N=("TARGET_MIPA", "size"), MIPA=("TARGET_MIPA", "mean"),
                                 **{c: (c, "mean") for c in CL_FEATS}).round(2)
prof["persen_sampel"] = (100 * prof["N"] / prof["N"].sum()).round(2)

# Aturan penamaan (eksplisit & dapat diaudit):
#  - skor MIPA tertinggi            -> AI-Empowered High Achievers
#  - hiburan gawai di sekolah tertinggi -> Digitally Distracted
#  - dua sisanya: indeks psikologis (EFFSCIE + DISCLISCI + COGABIL) lebih tinggi -> Resilient Low-Tech,
#                 yang lain -> Under-Resourced Strugglers
names = {}
names[prof["MIPA"].idxmax()] = "AI-Empowered High Achievers"
rest = [c for c in prof.index if c not in names]
names[prof.loc[rest, "ST326Q04JA"].idxmax()] = "Digitally Distracted"
rest = [c for c in prof.index if c not in names]
psy = prof.loc[rest, ["EFFSCIE", "DISCLISCI", "COGABIL"]].sum(axis=1)
names[psy.idxmax()] = "Resilient Low-Tech"
names[psy.idxmin()] = "Under-Resourced Strugglers"
prof["tipologi"] = prof.index.map(names)
order = ["AI-Empowered High Achievers", "Resilient Low-Tech", "Under-Resourced Strugglers", "Digitally Distracted"]
prof = prof.sort_values("tipologi", key=lambda s: s.map(order.index))
prof.to_csv(f"{OUT}/tabel7_profil_klaster.csv")
print("\nTABEL 7 (hasil dari kode ini):")
print(prof.to_string())
print(f"\nRerata MIPA sampel keseluruhan: {df['TARGET_MIPA'].mean():.2f}")

pal = ["#2ca02c", "#1f77b4", "#ff7f0e", "#d62728"]
short = {"AI-Empowered High Achievers": "AI-Empowered\nHigh Achievers", "Resilient Low-Tech": "Resilient\nLow-Tech",
         "Under-Resourced Strugglers": "Under-Resourced\nStrugglers", "Digitally Distracted": "Digitally\nDistracted"}
xl = [f"{short[t]}\nN={int(n):,}\n({p:.1f}% sampel)" for t, n, p in zip(prof["tipologi"], prof["N"], prof["persen_sampel"])]
fig, axes = plt.subplots(1, 3, figsize=(17, 5.2), gridspec_kw={"width_ratios": [1, 1.5, 0.8]})
axes[0].bar(range(4), prof["MIPA"], color=pal)
axes[0].axhline(df["TARGET_MIPA"].mean(), ls="--", color="gray", label="Rerata sampel")
for i, v in enumerate(prof["MIPA"]):
    axes[0].annotate(f"{v:.1f}", (i, v + 2), ha="center", fontweight="bold")
axes[0].set_ylim(300, 480); axes[0].set_ylabel("Rerata skor MIPA"); axes[0].set_title("(a) Capaian MIPA")
axes[0].set_xticks(range(4)); axes[0].set_xticklabels(xl, fontsize=7.5, rotation=0); axes[0].legend(fontsize=8)

idx_cols = ["AIUSESCH", "ICTRES", "EFFSCIE", "DISCLISCI", "COGABIL", "ESCS"]
idx_lab = ["AI sekolah", "Fasilitas TIK", "Efikasi diri", "Disiplin kelas", "Adaptabilitas", "Sosio-ekonomi"]
wd = 0.2
for i, (_, row) in enumerate(prof.iterrows()):
    axes[1].bar(np.arange(6) + (i - 1.5) * wd, row[idx_cols].values.astype(float), wd,
                color=pal[i], label=row["tipologi"])
axes[1].axhline(0, color="black", lw=0.8)
axes[1].set_xticks(range(6)); axes[1].set_xticklabels(idx_lab, fontsize=8, rotation=15)
axes[1].set_ylabel("Indeks (0 = rerata internasional OECD)"); axes[1].set_title("(b) Profil indeks psikometrik")
axes[1].legend(fontsize=7)

axes[2].bar(range(4), prof["ST326Q04JA"], color=pal)
for i, v in enumerate(prof["ST326Q04JA"]):
    axes[2].annotate(f"{v:.2f}", (i, v + 0.1), ha="center", fontweight="bold")
axes[2].set_ylim(0, 9); axes[2].set_ylabel("Skor kategori (skala 1-9, BUKAN jam)")
axes[2].set_title("(c) Gawai hiburan di sekolah"); axes[2].set_xticks(range(4))
axes[2].set_xticklabels(["High", "Resil.", "Under", "Distr."], fontsize=8)
plt.suptitle(f"Gambar 6. Empat Tipologi Siswa (K-Means, k = 4; Silhouette = {sil:.2f}; Davies-Bouldin = {dbi:.2f})", y=1.02)
plt.tight_layout(); plt.savefig(f"{OUT}/gambar6_tipologi_klaster.png", bbox_inches="tight"); plt.show()
print(f"\nSemua gambar & tabel tersimpan di folder '{OUT}/'")
