# Wellum: Dashboard AI Assistant Pemantauan Tingkat Kesejahteraan Akademik Digital Mahasiswa sebagai Basis Intervensi Institusional Berbasis Data

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0.0-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Ensemble-EB5424?style=for-the-badge&logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io/)
[![CatBoost](https://img.shields.io/badge/CatBoost-GBDT-FFCC00?style=for-the-badge&logo=catboost&logoColor=black)](https://catboost.ai/)
[![Design](https://img.shields.io/badge/UI%2FUX-Apple%20HIG-0071E3?style=for-the-badge&logo=apple&logoColor=white)](https://developer.apple.com/design/human-interface-guidelines/)
[![Validation](https://img.shields.io/badge/Validation%20F1-0.692-10B981?style=for-the-badge)](https://techphoria.unsri.ac.id)

**Karya Inovasi Kompetisi Data Mining — TECHPHORIA 2026**  
*Fakultas Ilmu Komputer, Universitas Sriwijaya*  
**Tema:** *"Ignite Innovation, Build the Future"*

[Fitur Utama](#-fitur-utama-wellum) •
[Arsitektur Machine Learning](#-arsitektur-machine-learning-sota) •
[Instalasi & Panduan Cepat](#-panduan-instalasi--menjalankan) •
[Matriks Intervensi](#-matriks-intervensi-preskriptif-institusional) •
[Struktur Proyek](#-struktur-direktori-proyek)

</div>

---

## 📌 Latar Belakang & Urgensi

Di era pascapandemi dan akselerasi perkuliahan *hybrid*, mahasiswa perguruan tinggi terpapar disrupsi digital yang intensif. Pola perilaku seperti *bedtime procrastination* berbasis gawai, beban kognitif berlebih (*notification overload*), serta tingginya tingkat kecemasan akademik secara empiris memicu penurunan drastis pada kesehatan fisik, stabilitas mental, dan Indeks Prestasi Kumulatif (IPK).

Sebagian besar perguruan tinggi masih menerapkan pendekatan **reaktif-kuratif**—intervensi baru dilakukan setelah mahasiswa mengalami putus studi (*drop out*) atau kegagalan studi massal. 

**Wellum (Wellbeing Intelligence Engine)** hadir sebagai terobosan sistem berbasis kecerdasan buatan (*AI Assistant*) untuk mentransformasi tata kelola kemahasiswaan dari reaktif menjadi **prediktif-preskriptif**. Dengan menganalisis indikator perilaku harian, ritme sirkadian, serta stres akademik, Wellum memetakan tingkat kesejahteraan akademik digital mahasiswa ke dalam tiga tingkat (`Low`, `Medium`, `High`) dan menyajikan **rekomendasi intervensi institusional yang terpersonalisasi**.

```
[ Data Perilaku Digital Mahasiswa ]
               │
               ▼
[ 150+ Rekayasa Fitur Psiko-Kronobiologi ]
               │
               ▼
[ 4-Seed Bagged Grandmaster Ensemble (20 Folds) ]
               │
               ▼
[ Dual-Stream Soft-Voting & Kalibrasi Prior PRDC ]
               │
               ▼
[ Dashboard Wellum & Rekomendasi Preskriptif Bertingkat ]
```

---

## ✨ Fitur Utama Wellum

Sistem Wellum dirancang mengikuti standar **Apple Human Interface Guidelines (HIG)** dengan antarmuka yang elegan, responsif, minimalis, dan berfokus pada kejelasan data (*clarity, deference, and depth*).

### 1. Web Dashboard Berstandar Apple Design (`Interface/`)
- **Estetika Premium**: Mengadopsi palet warna monokromatik Apple dengan aksen *Action Blue* (`#0066cc`), tipografi *SF Pro / Inter*, efek *frosted glass* (*glassmorphism*), dan animasi mikro yang halus.
- **Drag-and-Drop Batch Ingestion**: Kemampuan mengunggah dataset survei berskala ribuan responden (format `.csv` identik dengan `test.csv`) dengan pemrosesan instan (*sub-second inference*).
- **Executive Analytics Overview**:
  - Kartu KPI makro: Total Mahasiswa, Rata-rata Skor Kesejahteraan Kampus, Distribusi Kelas Kesejahteraan, dan Mahasiswa Butuh Intervensi Segera.
  - Distribusi proporsi kelas interaktif (`Low`: Zona Kritis Merah, `Medium`: Zona Perhatian Kuning, `High`: Zona Prima Hijau).
  - Radar Risiko Multi-Dimensi: Memetakan 6 pilar risiko kampus (Krisis Tidur, Distraksi Digital, Stres Ujian, Prokrastinasi, Kelelahan Fisik, Performa Akademik).
  - Matriks Korelasi & Analisis Komparatif Antarfakultas / Program Studi.
- **Interactive Student Roster & Live Editing**:
  - Tabel daftar mahasiswa lengkap dengan pencarian cepat (*instant search*), penyaringan status risiko, dan paginasi.
  - **Fitur Edit Mandiri (JSON-Backed)**: Administrator/konselor dapat menyunting data responden secara langsung melalui antarmuka web, tersimpan secara persisten ke berkas JSON sesi dengan *audit trail*.
- **Student Profiling & Detail Modal**:
  - Modal inspeksi personal menampilkan metrik individual, radar risiko mahasiswa, serta kartu diagnosa perilaku.
  - **Matriks Rekomendasi Preskriptif 3 Tingkat**: Aksi nyata yang harus dilakukan mahasiswa, dosen pembimbing akademik (PA), dan unit bimbingan konseling.
- **Ekspor Data & Submission**:
  - Unduh berkas prediksi resmi berformat CSV siap unggah/lapor.
  - Ekspor data hasil inferensi lengkap beserta skor probabilitas dalam format JSON.

### 2. Companion Streamlit Explorer (`app.py`)
- Simulator inferensi individu (*what-if analysis*) dengan *slider* interaktif parameter tidur, stres, durasi gawai, dan asupan kafein.
- Galeri transparansi model dan visualisasi beresolusi tinggi (*Explainable AI*).

---

## 🔬 Arsitektur Machine Learning (SOTA)

Sistem Wellum dibangun di atas metodologi pemodelan yang mengoptimalkan metrik **Macro F1-Score** pada data yang memiliki ketimpangan kelas alami (*Low* 48.8%, *Medium* 35.8%, *High* 15.4%):

| Komponen | Spesifikasi Teknis | Dampak Ilmiah |
| :--- | :--- | :--- |
| **Feature Engineering** | **150+ Fitur Domain Terarah** | Menangkap interaksi psiko-kronobiologi seperti *Academic Vitality to Strain Ratio* (AVSR, $r=+0.33$), *Circadian Disruption Index* (CDI), *Cognitive Friction Index*, dan rasio kafein terhadap tidur. |
| **Model Ensemble** | **4-Seed Zoo (20 Folds)** | Kombinasi heterogen: XGBoost, CatBoost, RandomForest, ExtraTrees, dan Huber Continuous Regressor dengan 4 variasi inisialisasi benih acak (`SEEDS = [42, 100, 2024, 777]`). |
| **Dual-Stream Fusion** | **Probability + Continuous Score** | Mengintegrasikan ekspektasi kelas kontinu dari *soft-voting classifier* ($72\%$) dengan prediksi *Huber continuous regressor* ($28\%$) untuk menstabilkan probabilitas di batas kelas (*decision boundary*). |
| **Post-Processing** | **Prior-Regularized Distribution Calibration (PRDC)** | Optimasi batas ambang non-linear ($[t_1, t_2]$) via Nelder-Mead Simplex dengan penalti deviasi prior populasi, menjamin tidak terjadi *under-prediction* pada kelas minoritas (*High*). |
| **Performa Validasi** | **Out-of-Fold Macro F1: ~0.692** | Peningkatan signifikan sebesar **+3.6%** dibandingkan baseline tunggal (0.658), dengan variansi prediksi terpangkas sebesar $50\%$. |

---

## 🛠️ Struktur Direktori Proyek

```text
TECHPO/
├── Interface/                          # Production Web Application (Flask + Apple HIG)
│   ├── flask_app.py                    # Server Flask, routing, batch engine, session store
│   ├── data/                           # Direktori penyimpanan JSON sesi responden
│   │   └── .gitkeep
│   ├── static/
│   │   └── css/
│   │       └── design.css              # Apple Human Interface System (SF Pro, glassmorphism)
│   └── templates/
│       ├── base.html                   # Master layout Apple-style dengan navigasi & footer
│       ├── index.html                  # Landing page & drag-and-drop CSV upload zone
│       ├── dashboard.html              # Executive campus analytics, chart visual, KPI cards
│       └── students.html               # Student roster table, live JSON editing, detail modal
│
├── figures/                            # 12 Visualisasi Publikasi Resmi (High-Res 300 DPI)
│   ├── vis_01_target_donut.png         # VIS-01: Distribusi Prior Target
│   ├── vis_02_correlation_heatmap.png  # VIS-02: Spearman Correlation Matrix
│   ├── vis_03_violin_distribution.png  # VIS-03: The Sleep-Stress Paradox
│   ├── vis_04_data_pipeline_sankey.png # VIS-04: Sanitasi & Rekayasa Fitur
│   ├── vis_05_system_architecture.png  # VIS-05: Vector Blueprint Arsitektur Sistem
│   ├── vis_06_shap_feature_importance.png # VIS-06: SHAP Top Predictors
│   ├── vis_07_shap_dependence.png      # VIS-07: Interaksi Non-Linear Fitur
│   ├── vis_08_latent_space_manifold.png # VIS-08: Pemisahan Manifold t-SNE / PCA
│   ├── vis_09_confusion_matrix.png     # VIS-09: Confusion Matrix Terkalibrasi PRDC
│   ├── vis_10_threshold_landscape.png  # VIS-10: Optimasi Simplex PRDC
│   ├── vis_11_radar_benchmark.png      # VIS-11: Benchmark Algoritma Heterogen
│   └── vis_12_dashboard_mockup.png     # VIS-12: Blueprint Antarmuka Preskriptif
│
├── models/                             # Penyimpanan Serialized Model Artifacts
│   ├── wellum_model_bundle.joblib      # Model bundle produksi (bisa dibuat ulang via script)
│   └── .gitkeep
│
├── wellum_engine.py                    # Inference Core, feature preprocessor & prescriptive logic
├── eduwell_engine.py                   # Legacy alias module untuk kompatibilitas penuh
├── app.py                              # Streamlit Companion Exploration Dashboard
├── train_and_save_model.py             # Script otomatis retraining & bundle packager produksi
├── generate_visualizations.py          # Script pembangkit otomatis 12 gambar resolusi tinggi
├── experiment_v6_sota_multiseed_champion.ipynb # Notebook riset utama (SOTA Champion)
├── notebook_visualisasi_lengkap.ipynb  # Notebook visualisasi dan analisis eksplorasi
├── TECHNICAL_REPORT_TECHPHORIA_2026.md # Technical report lengkap standar akademik
├── BLUEPRINT_DAN_RANCANGAN_LENGKAP.md  # Master blueprint & rincian metodologi sistem
├── train.csv                           # Dataset pelatihan kompetisi (bersih & berlabel)
├── test.csv                            # Dataset pengujian kompetisi (tanpa label)
├── submission_v6_sota_multiseed.csv    # Berkas hasil inferensi terbaik
├── requirements.txt                    # Daftar pustaka & dependensi Python
├── .gitignore                          # Konfigurasi filter berkas Git & GitHub
└── README.md                           # Dokumentasi utama proyek
```

---

## 🚀 Panduan Instalasi & Menjalankan

### 1. Prasyarat Sistem
- Python versi **3.10**, **3.11**, atau **3.12**
- Git (opsional untuk *version control*)
- Peramban web modern (*Google Chrome*, *Safari*, *Mozilla Firefox*, atau *Microsoft Edge*)

### 2. Kloning Repositori & Persiapan Lingkungan
```bash
# Kloning repositori
git clone https://github.com/username/Wellum.git
cd Wellum

# Buat lingkungan virtual (sangat disarankan)
python -m venv venv

# Aktivasi lingkungan virtual
# Pada Windows (PowerShell / Command Prompt):
venv\Scripts\activate
# Pada macOS / Linux:
source venv/bin/activate

# Pasang seluruh dependensi yang dibutuhkan
pip install -r requirements.txt
```

### 3. Membangkitkan / Retrain Model Bundle (Jika Diperlukan)
*Catatan: Model bundle biner berukuran >100 MB diabaikan dari repositori demi mematuhi batas ukuran berkas GitHub. Anda dapat membangkitkan model bundle produksi secara langsung hanya dalam hitungan menit menggunakan perintah:*

```bash
python train_and_save_model.py
```
Script ini akan:
1. Melakukan sanitasi data dan mengekstraksi 150+ fitur dari `train.csv`.
2. Melatih ansambel multi-model (XGBoost, CatBoost, RandomForest, ExtraTrees, Huber Regressor) melintasi 5 folds bertingkat.
3. Melakukan kalibrasi ambang batas PRDC menggunakan algoritma Nelder-Mead.
4. Menyimpan *ready-to-serve bundle* ke `models/wellum_model_bundle.joblib`.
5. Menguji inferensi pada sampel data `test.csv`.

---

### 4. Menjalankan Dashboard Utama (Flask Web App)

Aplikasi utama berbasis Flask dengan antarmuka Apple Design System:

```bash
python Interface/flask_app.py
```
Akses dashboard pada peramban melalui URL:
👉 **[http://localhost:5000](http://localhost:5000)**

#### Alur Penggunaan Dashboard:
1. **Unggah Berkas**: Pada halaman utama, seret dan lepas (*drag-and-drop*) berkas `test.csv` (atau format survei sejenis).
2. **Lihat Executive Overview**: Sistem akan otomatis mengarahkan ke dashboard analitik kampus dengan grafik distribusi, radar risiko, dan ringkasan populasi.
3. **Navigasi ke Student Roster**: Klik menu **"Students"** di navigasi atas untuk melihat daftar seluruh responden.
4. **Inspeksi Personal**: Klik tombol **"View"** pada baris mahasiswa untuk membuka modal detail yang berisi metrik individual, radar risiko, dan rekomendasi intervensi preskriptif.
5. **Sunting Data Mahasiswa**: Klik tombol **"Edit"** untuk menyunting data profil responden (misal memperbarui jam tidur atau durasi gawai). Sistem akan langsung menghitung ulang prediksi secara *real-time* dan menyimpannya ke berkas JSON sesi.
6. **Ekspor Hasil**: Klik tombol **"Export Predictions (CSV)"** untuk mengunduh berkas submission resmi.

---

### 5. Menjalankan Companion Streamlit Explorer

Untuk eksplorasi interaktif berbasis widget Streamlit:

```bash
streamlit run app.py
```
Akses pada peramban melalui URL:
👉 **[http://localhost:8501](http://localhost:8501)**

---

## 📊 Katalog Visualisasi Resmi (300 DPI)

Seluruh visualisasi teknis berstandar publikasi akademik telah dibangkitkan dan tersimpan di folder `figures/`. Anda dapat membangkitkan ulang seluruh figur menggunakan perintah:

```bash
python generate_visualizations.py
```

| ID Gambar | Nama Berkas | Deskripsi Visual |
| :---: | :--- | :--- |
| **VIS-01** | `vis_01_target_donut.png` | Donut chart rasio ketimpangan kelas target (`Low`, `Medium`, `High`) beserta nilai absolut responden. |
| **VIS-02** | `vis_02_correlation_heatmap.png` | Heatmap korelasi rank Spearman antara fitur-fitur perilaku utama terhadap tingkat kesejahteraan. |
| **VIS-03** | `vis_03_violin_distribution.png` | Violin plot gabungan fenomena *Sleep-Stress Paradox* pada mahasiswa. |
| **VIS-04** | `vis_04_data_pipeline_sankey.png` | Diagram alir Sankey proses transformasi data kotor menjadi 150+ fitur rekayasa. |
| **VIS-05** | `vis_05_system_architecture.png` | Blueprint vektor komputasi: 150+ fitur, 4-Seed Multi-Algorithm Zoo, fusi probabilitas, dan kalibrasi PRDC. |
| **VIS-06** | `vis_06_shap_feature_importance.png` | Diagram batang 25 fitur paling berpengaruh berdasarkan SHAP Value / Importance Gain. |
| **VIS-07** | `vis_07_shap_dependence.png` | Scatter plot interdependensi non-linear antara durasi tidur, stres akademik, dan skor output. |
| **VIS-08** | `vis_08_latent_space_manifold.png` | Proyeksi manifold t-SNE / PCA yang memperlihatkan pemisahan kelas hasil representasi fitur. |
| **VIS-09** | `vis_09_confusion_matrix.png` | Heatmap matriks konfusi Out-of-Fold setelah kalibrasi batas keputusan PRDC. |
| **VIS-10** | `vis_10_threshold_landscape.png` | Kontur optimasi landscape Nelder-Mead Simplex mencari titik ambang batas optimal $[t_1, t_2]$. |
| **VIS-11** | `vis_11_radar_benchmark.png` | Radar perbandingan multi-metrik (F1, Precision, Recall, AUC, Stabilitas) antarmodel individual vs ensemble. |
| **VIS-12** | `vis_12_dashboard_mockup.png` | Cetak biru arsitektur antarmuka sistem intervensi preskriptif perguruan tinggi. |

---

## 🎯 Matriks Intervensi Preskriptif Institusional

Sebagai perwujudan tema *"Build the Future"*, Wellum tidak hanya berhenti pada diagnosis label kelas, melainkan menyediakan mesin preskriptif tiga tingkat aksi:

```
┌────────────────────────────────────────────────────────────────────────┐
│                   WELLUM PRESCRIPTIVE TAXONOMY                         │
├───────────────────┬────────────────────────────────────────────────────┤
│ LEVEL 1: PRIMER   │ 📢 Edukasi Higienitas Digital & Ritme Sirkadian    │
│ (Promotif)        │    Aplikasi pengingat istirahat gawai, lokakarya   │
│                   │    manajemen waktu, penyediaan ruang tenang.       │
├───────────────────┼────────────────────────────────────────────────────┤
│ LEVEL 2: SEKUNDER │ 👥 Pendampingan Akademik Terarah                   │
│ (Preventif)       │    Bimbingan intensif Dosen PA, penyesuaian beban  │
│                   │    tugas, pembentukan peer-assisted study group.   │
├───────────────────┼────────────────────────────────────────────────────┤
│ LEVEL 3: TERSIER  │ 🩺 Konseling Psikologis & Intervensi Klinis        │
│ (Kuratif)         │    Rujukan unit Bimbingan & Konseling Kampus,       │
│                   │    konseling kognitif perilaku, pemantauan intensif│
└───────────────────┴────────────────────────────────────────────────────┘
```

---

## 💻 Spesifikasi Lingkungan Pengujian

Sistem telah diuji dan divalidasi pada lingkungan komputasi berikut:
- **Sistem Operasi**: Windows 11 Pro 64-bit / Linux Ubuntu 22.04 LTS
- **Bahasa**: Python 3.12.3
- **Framework Web**: Flask 3.0.0, Jinja2, Werkzeug
- **Pustaka ML**: Scikit-Learn 1.4+, XGBoost 2.0+, CatBoost 1.2+, SciPy 1.12+
- **Frontend**: Vanilla CSS (Apple HIG Design Tokens), Vanilla JavaScript (ES6+), SVG Icons

---

## 📜 Lisensi & Penghargaan

Proyek ini dikembangkan khusus untuk ajang **TECHPHORIA 2026 Data Mining Competition** di Fakultas Ilmu Komputer, Universitas Sriwijaya, dengan mengusung semangat *"Ignite Innovation, Build the Future"*.

Seluruh kode sumber dilisensikan di bawah lisensi **MIT License**. Silakan gunakan, pelajari, dan kembangkan untuk kemajuan dunia pendidikan dan kesehatan mental mahasiswa.

---

<div align="center">
  <sub>Dibangun dengan dedikasi untuk memajukan ekosistem pendidikan tinggi yang adaptif, empatik, dan berdaya saing tinggi.</sub><br>
  <b>Wellum AI Engine © 2026</b>
</div>
