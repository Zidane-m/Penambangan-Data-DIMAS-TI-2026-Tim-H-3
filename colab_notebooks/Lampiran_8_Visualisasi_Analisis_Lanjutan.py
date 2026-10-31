# ============================================================================
# LAMPIRAN 8 (VERSI PERBAIKAN): Analisis Lanjutan (Tabel 4, Tabel 5, Tabel 8)
# Semua angka DIHITUNG dari PISA2025_IDN.csv.
#
# Perbaikan dibanding versi lama:
#  1. Sel Gambar 6 (K-Means dengan angka manual) DIHAPUS dari lampiran ini; K-Means yang
#     benar-benar dihitung ada di Lampiran_7_FIXED.py.
#  2. Tabel 5 kini juga menghitung proporsi proksi Level 2 (>= 420) dan UJI SENSITIVITAS
#     ambang (400/420/450), serta menyajikan selisih +43,86 (matematika) dan +49,52 (sains).
#  3. Tabel 8: dibuat BAR CHART empat domain (CMOD, CPRO, Sains, Matematika) sesuai Slide 10,
#     lengkap dengan rerata terbobot (W_FSTUWT) dan tak terbobot. Versi lama hanya KDE dengan CMPS.
#  4. Tabel 4: ditambah selang kepercayaan 95% dan rerata terbobot, agar klaim
#     "non-monotonik" bisa diperiksa terhadap ketidakpastian sampling.
#  5. Label diperbaiki: "Rerata Nasional (400,85)" -> "Rerata sampel" (400,85 adalah rerata
#     TAK terbobot; rerata nasional terbobot sains = 388,67).
#
# Pasang paket bila perlu di Colab: seaborn dan matplotlib sudah tersedia.
# ============================================================================

# %% [SEL 1] Setup, pemuatan data, target psikometrik
import os, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({"font.family": "DejaVu Sans", "figure.dpi": 150})
OUT = "figures"
os.makedirs(OUT, exist_ok=True)


def load_data(path="PISA2025_IDN.csv"):
    candidates = [path, os.path.join("data", path), os.path.join("..", "data", path), os.path.join("..", path)]
    for p in candidates:
        if os.path.exists(p):
            return pd.read_csv(p, low_memory=False)
    from google.colab import files
    print("Pilih berkas PISA2025_IDN.csv")
    return pd.read_csv(list(files.upload().keys())[0], low_memory=False)


df = load_data()
for subj in ["SCIE", "MATH", "CMOD", "CPRO", "CMPS"]:
    df[f"TARGET_{subj}"] = df[[f"PV{i}{subj}" for i in range(1, 11)]].mean(axis=1)
df["TARGET_MIPA"] = df[[f"PV{i}{s}" for s in ["SCIE", "MATH"] for i in range(1, 11)]].mean(axis=1)
W = df["W_FSTUWT"]


def wmean(s, w=W):
    ok = s.notna()
    return np.average(s[ok], weights=w[ok])


print(f"Data dimuat: {len(df):,} siswa | rerata sains {df['TARGET_SCIE'].mean():.2f} (terbobot {wmean(df['TARGET_SCIE']):.2f}) | "
      f"matematika {df['TARGET_MATH'].mean():.2f} (terbobot {wmean(df['TARGET_MATH']):.2f})")

# %% [SEL 2] Tabel 4 + Gambar 3: The AI Paradox (ST438Q01DA)
kat = {1: "Tidak pernah", 2: "Beberapa kali/tahun", 3: "Beberapa kali/bulan", 4: "Beberapa kali/minggu", 5: "Setiap hari"}
d = df.dropna(subset=["ST438Q01DA"]).copy()
rows = []
for k, g in d.groupby("ST438Q01DA"):
    n = len(g)
    se = g["TARGET_SCIE"].std() / np.sqrt(n)
    rows.append({"Kategori": int(k), "Frekuensi": kat[int(k)], "N": n, "Persen": round(100 * n / len(d), 1),
                 "Sains": round(g["TARGET_SCIE"].mean(), 2), "CI95_sains": f"±{1.96 * se:.2f}",
                 "Sains (terbobot)": round(wmean(g["TARGET_SCIE"], g["W_FSTUWT"]), 2),
                 "Matematika": round(g["TARGET_MATH"].mean(), 2)})
t4 = pd.DataFrame(rows)
t4["Selisih thd rerata sampel sains"] = (t4["Sains"] - df["TARGET_SCIE"].mean()).round(2)
t4.to_csv(f"{OUT}/tabel4_ai_paradox.csv", index=False)
print("TABEL 4 (N valid = %d):" % len(d)); print(t4.to_string(index=False))
print("\nCatatan: N valid = %d. Lampiran 2 makalah menulis 13.684, periksa konsistensinya." % len(d))
print(f"Harian (Kat 5) vs puncak (Kat 3): {t4.loc[4,'Sains'] - t4.loc[2,'Sains']:+.2f} poin sains")
print(f"Tidak pernah (Kat 1) {t4.loc[0,'Sains']:.2f} vs tahunan (Kat 2) {t4.loc[1,'Sains']:.2f}  -> bukan kurva U terbalik murni")

lab = [f"Kat {i}\n{kat[i]}" for i in range(1, 6)]
fig, axes = plt.subplots(1, 2, figsize=(13.5, 5))
for ax, col, title, color in [(axes[0], "TARGET_SCIE", "Sains", "#2b5c8f"), (axes[1], "TARGET_MATH", "Matematika", "#800020")]:
    g = d.groupby("ST438Q01DA")[col]
    mean, se = g.mean().values, (g.std() / np.sqrt(g.count())).values
    ax.errorbar(range(5), mean, yerr=1.96 * se, marker="o", color=color, lw=2.5, capsize=4, label="Rerata ± 95% CI")
    ax.axhline(df[col].mean(), ls="--", color="#e6550d", label=f"Rerata sampel ({df[col].mean():.2f})")
    ax.scatter(2, mean[2], s=150, color="#2ca02c", zorder=5, label=f"Puncak ({mean[2]:.2f})")
    ax.scatter(4, mean[4], s=150, color="#d62728", zorder=5, label=f"Harian ({mean[4]:.2f})")
    for i, v in enumerate(mean):
        ax.annotate(f"{v:.2f}", (i, v + 2.2), ha="center", fontsize=8.5, fontweight="bold")
    ax.set_xticks(range(5)); ax.set_xticklabels(lab, fontsize=8)
    ax.set_title(f"The AI Paradox: {title}"); ax.set_ylabel(f"Skor {title}"); ax.legend(fontsize=8, loc="lower right")
plt.suptitle("Gambar 3. Pola non-monotonik pada pengguna aktif (asosiasi, bukan sebab-akibat)", y=1.02)
plt.tight_layout(); plt.savefig(f"{OUT}/gambar3_ai_paradox.png", bbox_inches="tight"); plt.show()

# %% [SEL 3] Tabel 5 + grafik Slide 8: Critical AI Literacy (ST436Q16DA) + uji sensitivitas ambang
ev = df[df["ST436Q16DA"].isin([1, 2, 3, 4])].copy()
nm = {1: "Tidak pernah", 2: "Kadang-kadang", 3: "Sering", 4: "Sangat sering"}
rows = []
for k, g in ev.groupby("ST436Q16DA"):
    rows.append({"Kategori": nm[int(k)], "N": len(g), "Persen": round(100 * len(g) / len(ev), 1),
                 "Matematika": round(g["TARGET_MATH"].mean(), 2), "Sains": round(g["TARGET_SCIE"].mean(), 2),
                 "L2 proksi >=400 (%)": round(100 * (g["TARGET_MIPA"] >= 400).mean(), 1),
                 "L2 proksi >=420 (%)": round(100 * (g["TARGET_MIPA"] >= 420).mean(), 1),
                 "L2 proksi >=450 (%)": round(100 * (g["TARGET_MIPA"] >= 450).mean(), 1)})
t5 = pd.DataFrame(rows)
t5.to_csv(f"{OUT}/tabel5_critical_ai_literacy.csv", index=False)
print(f"TABEL 5 (N valid = {len(ev):,}; hilang struktural = {100 * (1 - len(ev) / len(df)):.2f}%):")
print(t5.to_string(index=False))
dm = t5.loc[3, "Matematika"] - t5.loc[0, "Matematika"]
ds = t5.loc[3, "Sains"] - t5.loc[0, "Sains"]
print(f"\nSelisih 'Sangat sering' vs 'Tidak pernah': matematika {dm:+.2f}, sains {ds:+.2f}")
print("Catatan: komposit MIPA >= 420 adalah PROKSI buatan, bukan ambang resmi Level 2 PISA per domain.")

fig, ax = plt.subplots(figsize=(8, 4.8))
vals = t5["L2 proksi >=420 (%)"].values
bars = ax.bar(t5["Kategori"], vals, color=["#c6dbef", "#9ecae1", "#4292c6", "#08519c"])
for b, v in zip(bars, vals):
    ax.annotate(f"{v:.1f}%", (b.get_x() + b.get_width() / 2, v + 1), ha="center", fontweight="bold")
ax.set_ylim(0, 70); ax.set_ylabel("Siswa dengan komposit MIPA ≥ 420 (%)")
ax.set_xlabel("Seberapa sering siswa memverifikasi jawaban chatbot AI")
ax.set_title(f"Critical AI Literacy: matematika {dm:+.1f} poin, sains {ds:+.1f} poin (asosiasi)")
plt.tight_layout(); plt.savefig(f"{OUT}/slide8_critical_ai_literacy.png", bbox_inches="tight"); plt.show()

# %% [SEL 4] Tabel 8 + grafik Slide 10: kesenjangan komputasi vs matematika formal
dom = {"Pemodelan Komputasi (CMOD)": "TARGET_CMOD", "Pemrograman Logis (CPRO)": "TARGET_CPRO",
       "Sains": "TARGET_SCIE", "Matematika": "TARGET_MATH"}
t8 = pd.DataFrame([{"Domain": k, "Tak terbobot": round(df[v].mean(), 2),
                    "Terbobot (W_FSTUWT)": round(wmean(df[v]), 2), "SD": round(df[v].std(), 2)}
                   for k, v in dom.items()])
t8.to_csv(f"{OUT}/tabel8_domain.csv", index=False)
print("TABEL 8:"); print(t8.to_string(index=False))
gap_u = t8.loc[0, "Tak terbobot"] - t8.loc[3, "Tak terbobot"]
gap_w = t8.loc[0, "Terbobot (W_FSTUWT)"] - t8.loc[3, "Terbobot (W_FSTUWT)"]
print(f"\nSelisih CMOD - Matematika: tak terbobot {gap_u:+.2f} | terbobot {gap_w:+.2f}")
print("Catatan: skala antar-domain PISA tidak sepenuhnya sebanding; baca sebagai indikasi, bukan perbandingan setara.")

fig, ax = plt.subplots(figsize=(8.5, 5))
cols = ["#2ca02c", "#98df8a", "#1f77b4", "#d62728"]
bars = ax.bar(t8["Domain"], t8["Tak terbobot"], color=cols)
for b, v in zip(bars, t8["Tak terbobot"]):
    ax.annotate(f"{v:.2f}", (b.get_x() + b.get_width() / 2, v + 3), ha="center", fontweight="bold")
ax.set_ylim(300, 480); ax.set_ylabel("Rerata skor sampel (tak terbobot)")
ax.set_title(f"Komputasi vs Matematika Formal: selisih CMOD - Matematika = {gap_u:+.2f} poin")
ax.tick_params(axis="x", labelsize=9)
plt.tight_layout(); plt.savefig(f"{OUT}/slide10_domain_bar.png", bbox_inches="tight"); plt.show()

# Distribusi (KDE) dengan ambang proksi 420; garis OECD ~485 DIHAPUS karena tidak dihitung dari data
fig, ax = plt.subplots(figsize=(9.5, 4.8))
for col, c, lb in [("TARGET_CMOD", "#2ca02c", "CMOD"), ("TARGET_SCIE", "#1f77b4", "Sains"), ("TARGET_MATH", "#d62728", "Matematika")]:
    sns.kdeplot(df[col], fill=True, alpha=0.25, color=c, label=f"{lb} (rerata {df[col].mean():.1f})", ax=ax)
ax.axvline(420, ls=":", color="#333", lw=2, label="Ambang proksi 420")
ax.set_xlabel("Skor"); ax.set_title("Distribusi capaian: komputasi vs sains vs matematika"); ax.legend()
plt.tight_layout(); plt.savefig(f"{OUT}/gambar_kde_domain.png", bbox_inches="tight"); plt.show()

# %% [SEL 5] Lampiran 5: korelasi linier dengan AIUSESCH (alasan perlunya model non-linier)
rows = []
for name, col in [("Sains", "TARGET_SCIE"), ("Matematika", "TARGET_MATH"), ("CMOD", "TARGET_CMOD"), ("CPRO", "TARGET_CPRO")]:
    ok = df[["AIUSESCH", col]].dropna()
    rows.append({"Domain": name, "r Pearson thd AIUSESCH": round(ok["AIUSESCH"].corr(ok[col]), 4), "N": len(ok)})
t_l5 = pd.DataFrame(rows)
t_l5.to_csv(f"{OUT}/lampiran5_korelasi.csv", index=False)
print("LAMPIRAN 5:"); print(t_l5.to_string(index=False))
print(f"\nSemua tabel & gambar tersimpan di folder '{OUT}/'")
