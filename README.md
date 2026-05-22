# 🧩 Sudoku AI — Machine Learning Intelligence System

> Aplikasi permainan Sudoku berbasis desktop yang mengintegrasikan **5 model Machine Learning** untuk menganalisis perilaku pemain secara real-time, mengklasifikasikan tipe pemain, memprediksi skor, mendeteksi anomali sesi, dan memberikan rekomendasi adaptif.

---

## 📋 Daftar Isi

1. [Deskripsi Proyek](#1-deskripsi-proyek)
2. [Fitur Utama](#2-fitur-utama)
3. [Arsitektur Machine Learning](#3-arsitektur-machine-learning)
4. [Struktur Folder](#4-struktur-folder)
5. [Cara Instalasi](#5-cara-instalasi)
6. [Cara Menjalankan](#6-cara-menjalankan)
7. [Alur Permainan](#7-alur-permainan)
8. [Sistem Achievement](#8-sistem-achievement)
9. [Format Data Pemain](#9-format-data-pemain)
10. [Kontribusi & Lisensi](#10-kontribusi--lisensi)

---

## 1. Deskripsi Proyek

Sudoku AI adalah permainan Sudoku desktop yang dibangun dengan Python menggunakan **Tkinter** sebagai antarmuka grafis. Proyek ini menggabungkan logika permainan Sudoku klasik dengan kecerdasan buatan berbasis *supervised* dan *unsupervised learning* untuk menciptakan pengalaman bermain yang adaptif dan personal.

Setiap sesi permainan yang diselesaikan pemain direkam dan digunakan untuk melatih ulang model secara otomatis di latar belakang (*continuous improvement*), sehingga rekomendasi yang diberikan semakin akurat seiring waktu.

Proyek ini dikembangkan untuk keperluan **Pameran**, **HAKI**, dan **Penilaian Tugas Akhir Semester (UAS)**.

---

## 2. Fitur Utama

### Permainan
- **Dua ukuran grid**: 4×4 (Pemula) dan 9×9 (Klasik)
- **Tiga tingkat kesulitan**: Easy, Normal, Hard — masing-masing dengan palet warna, tingkat kekosongan sel, dan tema visual yang berbeda
- **Tema UI**: Dark Mode dan Light Mode yang dapat diubah kapan saja
- **Highlight sel adaptif**: Baris, kolom, dan kotak yang terhubung dengan sel terpilih disorot otomatis
- **Sistem petunjuk (Hint)**: Petunjuk diberikan secara cerdas berdasarkan waktu idle dan tingkat kesalahan pemain
- **Mode Auto-Fill**: Mengisi kandidat angka yang valid secara otomatis
- **Undo/Redo**: Riwayat gerakan dapat dibatalkan dan diulang
- **Sistem skor**: Skor dihitung secara adil berdasarkan waktu, kesalahan, penggunaan hint, dan tingkat kesulitan

### Antarmuka
- **Animasi latar belakang**: Partikel bergerak yang mengikuti tema aktif
- **Musik latar dan efek suara**: Dihasilkan secara prosedural dan diputar menggunakan Pygame
- **Attractor Screen**: Layar demo otomatis yang menampilkan AI memecahkan Sudoku menggunakan algoritma *MRV Backtracking* (aktif setelah 45 detik idle di halaman login)
- **Easter Egg**: Overlay animasi tersembunyi dengan pemutaran video dan musik eksklusif

### Analitik & ML
- **Dashboard Performa**: Ringkasan sesi dengan visualisasi skor, tipe pemain, anomali, dan prediksi sesi berikutnya
- **Kartu Skor PNG**: Dapat diekspor sebagai gambar menggunakan Pillow
- **Continuous Retraining**: Semua model dilatih ulang di *background thread* setelah setiap sesi disimpan

### Multi-Pemain
- Mendukung banyak akun pemain dalam satu perangkat
- Data setiap pemain tersimpan terpisah dalam `sudoku_data.json`

---

## 3. Arsitektur Machine Learning

Proyek ini menggunakan **5 model Machine Learning** yang masing-masing memiliki tanggung jawab spesifik. Setiap model dilatih di notebook Jupyter tersendiri, diekspor sebagai file `.pkl`, dan dimuat oleh `Sudoku.py` saat runtime.

### Ringkasan Model

| File PKL | Notebook | Algoritma | Tugas |
|---|---|---|---|
| `KNN.pkl` | `Model_KNN.ipynb` | K-Nearest Neighbors + StandardScaler | Klasifikasi tipe pemain |
| `RFR.pkl` | `Model_Random_Forest_Regressor.ipynb` | Random Forest Regressor | Prediksi skor sesi berikutnya |
| `ISO.pkl` | `Model_Isolation_Forest.ipynb` | Isolation Forest + StandardScaler | Deteksi sesi anomali |
| `GBM.pkl` | `Model_GBM.ipynb` | HistGradientBoosting / RFC + StandardScaler | Rekomendasi tingkat kesulitan |
| `Multioutput_regressor_GBM.pkl` | `Model_Multioutput_Regressor.ipynb` | MultiOutputRegressor (GBM/RF) + StandardScaler | Prediksi profil performa 11 target |

Selain kelima model di atas, terdapat satu model tambahan:

| File PKL | Keterangan |
|---|---|
| `HintTimer_RFR.pkl` | Mini Random Forest Regressor untuk memprediksi waktu idle optimal sebelum memberikan hint otomatis |

---

### 3.1 KNN — Klasifikasi Tipe Pemain (`KNN.pkl`)

**Tujuan**: Mengklasifikasikan pemain ke dalam 5 kategori berdasarkan perilaku bermain.

**Kelas (Label)**:
- `Speedrunner` — Cepat dan akurat
- `Careful` — Lambat tapi minim kesalahan
- `Learner` — Sedang berkembang
- `Struggling` — Banyak kesalahan dan mengandalkan hint
- `Inconsistent` — Performa tidak menentu

**Fitur Input (6 fitur)**:

| Fitur | Deskripsi |
|---|---|
| `avg_time_per_cell` | Rata-rata waktu per sel (detik) |
| `error_rate` | Proporsi gerakan yang salah |
| `hint_rate` | Proporsi gerakan yang menggunakan hint |
| `completion_rate` | Rasio sesi yang berhasil diselesaikan |
| `near_miss_rate` | Rasio error yang hampir benar |
| `guessing_rate` | Rasio error yang terdeteksi sebagai tebakan |

**Training**: 2.500 data sintetis per kelas (GridSearchCV dengan StratifiedKFold-5, metrik F1-macro). Model hanya diperbarui jika CV F1-macro dari data aktual ≥ F1 notebook + 0,5%.

---

### 3.2 Random Forest Regressor — Prediksi Skor (`RFR.pkl`)

**Tujuan**: Memprediksi skor sesi berikutnya berdasarkan tren 3 sesi terakhir.

**Fitur Input (4 fitur)**:

| Fitur | Deskripsi |
|---|---|
| `session_idx` | Indeks urutan sesi |
| `time_per_cell` | Waktu per sel kosong (detik) |
| `error_rate` | Proporsi gerakan yang salah |
| `hint_rate` | Proporsi gerakan yang menggunakan hint |

**Catatan**: Model ini tidak memerlukan StandardScaler karena Random Forest invariant terhadap scaling. Validasi menggunakan OOB Score (Out-of-Bag), bukan training R². Model baru hanya disimpan jika OOB Score lebih tinggi dari model sebelumnya.

---

### 3.3 Isolation Forest — Deteksi Anomali (`ISO.pkl`)

**Tujuan**: Mendeteksi apakah sesi terakhir merupakan anomali (performa tidak biasa, potensi kecurangan, atau kondisi tidak normal).

**Fitur Input**: Sama dengan 6 fitur KNN (`avg_time_per_cell`, `error_rate`, `hint_rate`, `completion_rate`, `near_miss_rate`, `guessing_rate`).

**Metode**: Scaler dilatih **hanya** pada data normal di training set. Threshold optimal ditentukan menggunakan sweep F1-score pada validation set. Skor anomali dihitung sebagai negasi dari `decision_function` agar nilai lebih besar = lebih anomali.

**Output**: `"normal"` | `"anomaly"` | `"unknown"` beserta alasan teks.

---

### 3.4 GBM — Rekomendasi Kesulitan (`GBM.pkl`)

**Tujuan**: Merekomendasikan tingkat kesulitan (Easy / Normal / Hard) yang sesuai dengan kemampuan pemain saat ini.

**Fitur Input (8 fitur)**: `tpc`, `er`, `hr`, `cr`, `nmr`, `gur`, `diff` (encoded), dan `skill_score`.

**Model**: `HistGradientBoostingClassifier` (GBM) dan `RandomForestClassifier` diperbandingkan menggunakan Balanced Accuracy dan F1-macro. Model terbaik dipilih dan disimpan beserta StandardScaler-nya.

**Training Data**: 5.000 data sintetis + oversampling data aktual (faktor ×3 jika tersedia).

---

### 3.5 MultiOutput Regressor — Profil Performa (`Multioutput_regressor_GBM.pkl`)

**Tujuan**: Memprediksi 11 target performa sekaligus berdasarkan observasi sesi terkini.

**Fitur Input (8 fitur)**: `tpc`, `er`, `hr`, `cr`, `nmr`, `gur`, `moves`, `score`.

**Target Output (11 target)**:

| Target | Deskripsi |
|---|---|
| `exp_tpc` – `exp_score` | Nilai yang diharapkan (versi smooth dari input) |
| `speed_idx` | Indeks kecepatan (0–100) |
| `accuracy_idx` | Indeks akurasi (0–100) |
| `consistency_idx` | Indeks konsistensi (0–100) |
| `independence_idx` | Indeks kemandirian/tanpa hint (0–100) |

**Model**: `MultiOutputRegressor` dengan `HistGradientBoostingRegressor` sebagai estimator dasar, dilatih pada 5.000 sampel (5 arketipe × 1.000 sampel) menggunakan RandomizedSearchCV.

---

### Mekanisme Continuous Improvement

Setelah setiap sesi berhasil disimpan, `Sudoku.py` menjalankan fungsi `_ml_schedule_retrain()` di *background daemon thread*. Fungsi ini:
1. Melatih ulang KNN, RFR, dan Isolation Forest
2. Melatih ulang GBM (rekomendasi kesulitan) dan MultiOutput Regressor
3. Melatih ulang HintTimer RFR

Semua retrain dilindungi dengan `threading.Lock` untuk mencegah kondisi race. Model baru hanya menimpa file `.pkl` lama jika metrik validasinya lebih baik.

---

## 4. Struktur Folder

```
PROJECT_UAS/
├── Sudoku.py                          # Entry point — game & ML runtime
├── sudoku_data.json                   # Data sesi & profil semua pemain (auto-generated)
│
├── Assets/
│   ├── logo.png                       # Logo aplikasi (header & icon)
│   ├── music.mp3                      # Musik latar permainan
│   ├── easter_egg.mp3                 # Audio Easter Egg
│   └── easter_egg.mp4                 # Video Easter Egg
│
├── Models/
│   └── Files/
│       ├── KNN.pkl                    # KNeighborsClassifier + StandardScaler
│       ├── RFR.pkl                    # RandomForestRegressor
│       ├── ISO.pkl                    # IsolationForest + StandardScaler
│       ├── GBM.pkl                    # HistGradientBoosting/RFC + StandardScaler
│       ├── Multioutput_regressor_GBM.pkl  # MultiOutputRegressor + StandardScaler
│       └── HintTimer_RFR.pkl         # Mini-RFR untuk hint timing
│
├── Model_KNN.ipynb                    # Notebook training KNN
├── Model_Random_Forest_Regressor.ipynb  # Notebook training RFR
├── Model_Isolation_Forest.ipynb       # Notebook training Isolation Forest
├── Model_GBM.ipynb                    # Notebook training GBM
├── Model_Multioutput_Regressor.ipynb  # Notebook training MultiOutput
│
└── requirements.txt                   # Daftar dependensi Python
```

---

## 5. Cara Instalasi

### Prasyarat
- Python **3.10 – 3.12** (disarankan)
- pip

### Langkah Instalasi

```bash
# 1. Clone atau ekstrak folder proyek
cd PROJECT_UAS

# 2. (Opsional) Buat virtual environment
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows

# 3. Instal semua dependensi
pip install -r requirements.txt
```

> **Catatan**: `tkinter` sudah termasuk dalam instalasi Python standar dan tidak perlu diinstal secara terpisah. Pada beberapa distribusi Linux, mungkin perlu menjalankan `sudo apt install python3-tk`.

---

## 6. Cara Menjalankan

### Menjalankan Permainan

```bash
python Sudoku.py
```

### Melatih Ulang Model (Opsional)

Buka dan jalankan masing-masing notebook secara berurutan. Setiap notebook akan menghasilkan file `.pkl` di folder `Models/Files/`:

```bash
jupyter notebook
# Buka: Model_KNN.ipynb
# Buka: Model_Random_Forest_Regressor.ipynb
# Buka: Model_Isolation_Forest.ipynb
# Buka: Model_GBM.ipynb
# Buka: Model_Multioutput_Regressor.ipynb
```

> **Catatan**: File `.pkl` yang sudah ada di folder `Models/Files/` sudah cukup untuk menjalankan permainan. Notebook hanya diperlukan jika ingin melatih ulang model atau mengeksplorasi data.

---

## 7. Alur Permainan

```
Login Screen
    ↓ (masukkan nama pemain)
Player Select Screen
    ↓ (pilih atau buat profil)
Grid Select Screen
    ↓ (pilih 4×4 atau 9×9)
Difficulty Screen
    ↓ (pilih Easy / Normal / Hard)
Game Screen
    ↓ (mainkan Sudoku)
Achievement Popup  ←── (jika ada badge baru)
    ↓
Performance Dashboard
    ↓ (lihat analitik sesi)
Grid Select Screen  ←── (main lagi)
```

Jika tidak ada interaksi selama **45 detik** di halaman Login, **Attractor Screen** akan muncul secara otomatis — menampilkan AI yang memecahkan Sudoku 9×9 secara animasi. Layar ini dirancang untuk mode pameran (*booth display*).

---

## 8. Sistem Achievement

Terdapat **20 achievement** yang dibagi dalam 6 tier:

| Tier | Nama | Contoh Achievement |
|---|---|---|
| 1 | First Steps | First Win, Marathon (10 sesi), Veteran (25 sesi) |
| 2 | Streak & Konsistensi | Consistent (3x berturut), Unbeatable (5x), Comeback |
| 3 | Kecepatan | Lightning (4×4 < 60 dtk), Speed Demon (9×9 < 5 menit) |
| 4 | Akurasi & Kemandirian | No Hints, Flawless (0 error), Perfect (0 hint + 0 error) |
| 5 | Kesulitan & Grid | Hard Expert, Iron Will, Master 9×9, Explorer |
| 6 | Skor | Genius (skor > 800), Expert (Hard + skor > 500) |

Achievement dievaluasi otomatis setelah setiap sesi selesai dan ditampilkan melalui popup animasi sebelum masuk ke Performance Dashboard.

---

## 9. Format Data Pemain

Semua data pemain disimpan di `sudoku_data.json` dengan format berikut:

```json
{
  "players": {
    "namapengguna": {
      "achievements": ["first_win", "marathon"],
      "sessions": [
        {
          "timestamp": 1700000000,
          "difficulty": "Normal",
          "grid_size": 3,
          "total_time": 312,
          "moves": 48,
          "errors": 3,
          "hints_used": 1,
          "auto_used": 0,
          "score": 425,
          "completed": true,
          "empty_cells": 45,
          "time_per_cell": 6.93,
          "near_miss": 1,
          "guessing": 2
        }
      ]
    }
  }
}
```

Data ditulis secara *atomic* (tulis ke file `.tmp` terlebih dahulu, lalu `os.replace()` ke file asli) untuk mencegah korupsi data jika aplikasi ditutup paksa.

---

## 10. Kontribusi & Lisensi

Proyek ini dikembangkan sebagai bagian dari **Tugas Akhir Semester (UAS)** dan diajukan untuk keperluan **HAKI (Hak Kekayaan Intelektual)**.

Seluruh kode dalam repositori ini merupakan karya orisinal. Penggunaan ulang kode untuk keperluan komersial tanpa izin tertulis dari pemilik tidak diperbolehkan.

---

*Sudoku AI — Machine Learning Intelligence System*