# utils/visual_config.py
"""
Konfigurasi Visualisasi Standar Publikasi Ilmiah (300 DPI)
Mengadopsi palet warna ramah buta warna (colorblind-friendly), tipografi formal,
dan penataan layout otomatis.
"""
import matplotlib.pyplot as plt

PALETTE_PRIMARY = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
PALETTE_CONTRAST = ['#3b528b', '#21918c', '#5ec962', '#fde725', '#d95f02', '#7570b3']
PALETTE_KMEANS = ['#2b5c8f', '#d95f02', '#2a9d8f', '#7570b3']


def setup_visual_style():
    """Mengatur konfigurasi global matplotlib untuk publikasi karya tulis."""
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    plt.rcParams['axes.edgecolor'] = '#cccccc'
    plt.rcParams['axes.linewidth'] = 0.8
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['axes.titleweight'] = 'bold'
    plt.rcParams['axes.labelsize'] = 10
    plt.rcParams['xtick.labelsize'] = 9
    plt.rcParams['ytick.labelsize'] = 9
    plt.rcParams['legend.fontsize'] = 9
    plt.rcParams['figure.titlesize'] = 13
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['savefig.bbox'] = 'tight'

