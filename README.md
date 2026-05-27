# Sudoku AI

> **Game Sudoku berbasis Python dengan sistem Kecerdasan Buatan adaptif**  
> *A Python-based Sudoku game powered by an adaptive Artificial Intelligence system*

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![pygame](https://img.shields.io/badge/pygame-2.5%2B-00B140?style=flat)
![Pillow](https://img.shields.io/badge/Pillow-10%2B-8A2BE2?style=flat)
![Version](https://img.shields.io/badge/version-1.0.0-58A6FF?style=flat)
![License](https://img.shields.io/badge/license-All%20Rights%20Reserved-FF7B7B?style=flat)

---

## Daftar Isi / Table of Contents

- [Tentang Proyek](#tentang-proyek)
- [Fitur Unggulan](#fitur-unggulan)
- [Model Machine Learning](#model-machine-learning)
- [Struktur Proyek](#struktur-proyek)
- [Persyaratan Sistem](#persyaratan-sistem)
- [Instalasi](#instalasi)
- [Cara Menjalankan](#cara-menjalankan)
- [Panduan Penggunaan](#panduan-penggunaan)
- [Data Pemain](#data-pemain)
- [Mode Demo](#mode-demo)
- [Kredit Akademik](#kredit-akademik)
- [Hak Cipta](#hak-cipta)

---

## Tentang Proyek

**Sudoku AI** adalah aplikasi permainan Sudoku berbasis Python yang mengintegrasikan enam model *Machine Learning* untuk menciptakan pengalaman bermain yang adaptif dan personal. Setiap sesi pemain dianalisis secara real-time oleh AI engine untuk menentukan tipe pemain, merekomendasikan tingkat kesulitan, memprediksi skor, mendeteksi perilaku anomali, dan memutuskan kapan waktu yang tepat untuk memberikan petunjuk.

Proyek ini dibangun di atas arsitektur OOP dengan lebih dari 8.000 baris kode Python murni menggunakan `tkinter` sebagai antarmuka grafis utama, `pygame` untuk sistem audio, dan `Pillow` untuk pembuatan *score card* PNG.

---

## Fitur Unggulan

### Gameplay
- **Dua ukuran grid**, yaitu 4x4 (pemula) dan 9x9 (standar)
- **Tiga tingkat kesulitan**, yaitu Easy, Normal, Hard dengan puzzle generator berbasis backtracking + MRV
- **Multi-player login**, baik sistem akun lokal maupun riwayat sesi tersimpan per pemain
- **Auto-solve**, yaitu fitur bantuan penyelesaian otomatis dengan animasi
- **Hint adaptif**, artinya petunjuk diberikan secara cerdas berdasarkan kondisi pemain saat ini
- **Timer sesi** dan kalkulasi skor berlapis (kecepatan, akurasi, penalti hint)

### User Interface
- **Dua tema visual**, yaitu Dark (default) dan Light, dapat diganti kapan saja
- **Blur overlay**, berupa efek *frosted glass* di setiap popup menggunakan `PIL.ImageFilter`
- **Animated background**, yaitu partikel dinamis pada layar utama
- **Tooltip** informatif di seluruh kontrol
- **Tutorial overlay** interaktif untuk pemain baru

### Sistem Pencapaian (20 Badge)
Pencapaian dikelompokkan dalam 6 tier:

| Tier | Kategori | Contoh Badge |
|------|----------|--------------|
| 1 | First Steps | First Win, Marathon (10 sesi), Veteran (25 sesi) |
| 2 | Streak | Consistent (3x), Unbeatable (5x), Serial Winner (7x) |
| 3 | Speed | Lightning (4x4 <60s), Speed Demon (9x9 <5 menit) |
| 4 | Accuracy | No Hints, Flawless, Perfect (tanpa error dan hint) |
| 5 | Difficulty | Hard Expert, Iron Will, Master 9x9, Explorer |
| 6 | Score | Genius (>800), Expert (Hard >500) |

### Score Card & Performa
- **Score Card PNG**, yaitu kartu skor bergaya visual tinggi yang dapat disimpan ke folder `Card/`
- **Performance Dashboard**, yaitu grafik dan statistik sesi dengan analisis mendalam dari ML engine
- **Achievement Popup** berupa animasi penghargaan setelah sesi selesai

### Audio
- Musik latar MP3 dengan kontrol on/off
- **SFX Engine programatik** dengan enam efek suara unik (correct, error, win, achievement, select, hover, click) dibuat secara generatif menggunakan `pygame` tanpa file audio eksternal

### Easter Egg
Klik judul aplikasi sebanyak 7 kali untuk mengaktifkan overlay video tersembunyi (membutuhkan `opencv-python`).

---

## Model Machine Learning

Seluruh model dilatih menggunakan data sintetis berlabel dan disimpan dalam format `.pkl`. Model di-*load* saat startup dan di-*retrain* secara inkremental di latar belakang setiap kali sesi baru selesai (*continuous learning*).

### Ringkasan Model

| File `.pkl` | Notebook Pelatihan | Algoritma | Task | Input | Output |
|---|---|---|---|---|---|
| `Player_Classifier.pkl` | `Model_Classify_Player.ipynb` | KNN + StandardScaler | Multiclass Classification | 6 fitur sesi | 5 tipe pemain |
| `Score_Prediction.pkl` | `Model_Score_Prediction.ipynb` | HistGradientBoostingRegressor | Regression | 4 fitur sesi | Prediksi skor |
| `Detect_Anomaly.pkl` | `Model_Anomaly_Detection.ipynb` | IsolationForest | Unsupervised Anomaly Detection | 6 fitur sesi | Normal / Anomali |
| `Difficulty_Recommender.pkl` | `Model_Difficulty_Recommender.ipynb` | RandomForestClassifier | Multiclass Classification | 8 fitur agregat | Easy / Normal / Hard |
| `Performance_Prediction.pkl` | `Model_Performance_Prediction.ipynb` | MultiOutputRegressor (HistGBR) | Multi-output Regression | 8 fitur sesi | 11 target metrik |
| `Hint_Timer.pkl` | `Model_Hint_Timer.ipynb` | RandomForestRegressor | Regression | Fitur sesi + konteks | Threshold waktu hint |

### Fitur Input (Feature Vector)

Semua model menggunakan subset dari 8 fitur utama berikut:

| Fitur | Simbol | Deskripsi |
|-------|--------|-----------|
| `avg_time_per_cell` | `tpc` | Rata-rata detik per sel yang dikerjakan |
| `error_rate` | `er` | Proporsi langkah yang salah terhadap total langkah |
| `hint_rate` | `hr` | Proporsi hint yang digunakan terhadap total langkah |
| `completion_rate` | `cr` | 1.0 = puzzle selesai, 0.0 = tidak selesai |
| `near_miss_rate` | `nmr` | Rasio *hampir benar* terhadap total error |
| `guessing_rate` | `gur` | Rasio tebak-acak terhadap total error |
| `avg_moves` | - | Rata-rata total langkah per sesi |
| `avg_score` | - | Rata-rata skor per sesi |

### Tipe Pemain

Model klasifikasi membagi pemain ke dalam 5 kategori:

| Label | Tipe | Karakteristik |
|-------|------|---------------|
| 0 | **Speedrunner** | Cepat, akurat tinggi, hampir tidak pernah pakai hint |
| 1 | **Careful** | Lambat dan sangat teliti, jarang salah |
| 2 | **Learner** | Performa moderat, sedang berkembang |
| 3 | **Struggling** | Banyak error dan hint, jarang menyelesaikan puzzle |
| 4 | **Inconsistent** | Variansi tinggi antar sesi, pola tidak menentu |

### Anomali yang Dideteksi

Model `IsolationForest` (unsupervised) mendeteksi pola mencurigakan:

| Pola Anomali | Ciri Khas |
|---|---|
| *Cheat Speed* | Selesai sangat cepat dengan nol error |
| *Hint Abuse* | Penggunaan hint di atas 70% dari seluruh langkah |
| *Impossible Combo* | Error sangat tinggi tetapi puzzle tetap selesai |
| *Idle Session* | AFK panjang tanpa progres berarti |
| *Corrupted Data* | Nilai metrik yang saling kontradiktif |

---

## Struktur Proyek

```
SUDOKU/
├── Assets/
│   ├── logo.png              # Logo aplikasi
│   ├── music.mp3             # Musik latar permainan
│   ├── easter_egg.mp3        # Audio easter egg
│   └── easter_egg.mp4        # Video easter egg (butuh opencv-python)
│
├── Models/
│   └── Files/
│       ├── Detect_Anomaly.pkl
│       ├── Difficulty_Recommender.pkl
│       ├── Hint_Timer.pkl
│       ├── Performance_Prediction.pkl
│       ├── Player_Classifier.pkl
│       └── Score_Prediction.pkl
│
├── Model_Anomaly_Detection.ipynb       # Training: IsolationForest
├── Model_Classify_Player.ipynb         # Training: KNN Classifier
├── Model_Difficulty_Recommender.ipynb  # Training: RFC / HistGBR
├── Model_Hint_Timer.ipynb              # Training: RandomForestRegressor
├── Model_Performance_Prediction.ipynb  # Training: MultiOutputRegressor
├── Model_Score_Prediction.ipynb        # Training: HistGBR / XGB / LGBM
│
├── Sudoku.py                 # Entry point utama aplikasi
├── README.md
├── requirements.txt
│
├── player_data.json          # (dibuat otomatis) Data sesi semua pemain
└── Card/                     # (dibuat otomatis) Score card PNG per pemain
    └── <username>_<timestamp>.png
```

---

## Persyaratan Sistem

- **Python** 3.10 atau lebih baru
- **OS** : Windows 10/11, macOS 12+, atau Linux (Ubuntu 20.04+)
- **RAM** : minimal 512 MB
- `tkinter` sudah termasuk dalam instalasi Python standar

---

## Instalasi

### 1. Clone repositori

```bash
git clone https://github.com/<username>/sudoku-ai.git
cd sudoku-ai
```

### 2. Buat virtual environment (disarankan)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependensi runtime

```bash
pip install numpy scikit-learn pygame Pillow matplotlib
```

Atau gunakan `requirements.txt` (sudah mencakup dependensi notebook):

```bash
pip install -r requirements.txt
```

### 4. (Opsional) Aktifkan easter egg video

```bash
pip install opencv-python
```

### 5. Pastikan file model tersedia

Pastikan enam file `.pkl` ada di folder `Models/Files/`. Jika belum ada, jalankan masing-masing notebook pelatihan terlebih dahulu (lihat bagian [Training Ulang Model](#training-ulang-model)).

---

## Cara Menjalankan

```bash
python Sudoku.py
```

Saat startup, konsol akan menampilkan status dependensi dan kondisi setiap model:

```
====================================================
  Sudoku AI  v1.0.0
  Mode    : NORMAL
  sklearn : tersedia
  pygame  : tersedia
  PIL     : tersedia
  cv2     : tersedia
----------------------------------------------------
  [OK  ] Player_Classifier
  [OK  ] Score_Prediction
  [OK  ] Detect_Anomaly
  [OK  ] Difficulty_Recommender
  [OK  ] Performance_Prediction
  [OK  ] Hint_Timer
====================================================
```

Jika ada model yang `[MISS]`, aplikasi tetap berjalan dengan prediksi *fallback* bawaan.

---

## Panduan Penggunaan

1. **Login / Daftar** : Masukkan nama pengguna pada layar awal. Akun baru dibuat otomatis.
2. **Pilih Ukuran Grid** : 4x4 untuk sesi cepat, 9x9 untuk tantangan penuh.
3. **Pilih Kesulitan** : AI akan merekomendasikan tingkat yang sesuai berdasarkan riwayat Anda.
4. **Bermain** : Klik sel, ketik angka. Tekan `Hint` jika butuh bantuan.
5. **Selesai** : Lihat Achievement Popup (jika ada badge baru), kemudian Performance Dashboard.
6. **Score Card** : Simpan kartu skor sebagai PNG dari layar hasil akhir.

### Kontrol Keyboard

| Tombol | Fungsi |
|--------|--------|
| `1-9` / `1-4` | Input angka ke sel yang dipilih |
| `Delete` / `Backspace` | Hapus angka di sel aktif |
| `Arrow Keys` | Navigasi antar sel |
| `H` | Minta hint |

---

## Data Pemain

Data seluruh pemain disimpan lokal dalam `player_data.json` di direktori yang sama dengan `Sudoku.py`. Struktur utamanya:

```json
{
  "players": {
    "<username>": {
      "sessions": [
        {
          "difficulty": "Normal",
          "grid_size": 3,
          "total_time": 312.5,
          "moves": 45,
          "errors": 2,
          "hints_used": 1,
          "completed": true,
          "score": 680,
          "near_miss": 1,
          "guessing": 0
        }
      ],
      "achievements": ["pemula_berhasil", "tanpa_cela"]
    }
  }
}
```

File ini juga digunakan sebagai data nyata untuk proses *continuous learning*. Artinya model akan di-retrain secara inkremental di latar belakang setiap kali sesi baru selesai dan disimpan.

---

## Training Ulang Model

Jika ingin melatih ulang model dari awal, buka dan jalankan notebook sesuai urutan berikut:

1. `Model_Classify_Player.ipynb` → `Player_Classifier.pkl`
2. `Model_Anomaly_Detection.ipynb` → `Detect_Anomaly.pkl`
3. `Model_Score_Prediction.ipynb` → `Score_Prediction.pkl`
4. `Model_Difficulty_Recommender.ipynb` → `Difficulty_Recommender.pkl`
5. `Model_Performance_Prediction.ipynb` → `Performance_Prediction.pkl`
6. `Model_Hint_Timer.ipynb` → `Hint_Timer.pkl`

Semua notebook menggunakan data sintetis yang di-*generate* secara internal. Jika `player_data.json` sudah berisi cukup data nyata, notebook `Model_Difficulty_Recommender` dan `Model_Score_Prediction` juga dapat dilatih menggunakan data tersebut.

Install dependensi tambahan untuk notebook:

```bash
pip install pandas seaborn scipy xgboost lightgbm notebook
```

---

## Mode Demo

Jika lingkungan tidak mendukung pemuatan model `.pkl` (misalnya saat pameran tanpa koneksi atau dependensi tidak lengkap), aktifkan **Demo Mode** di baris berikut pada `Sudoku.py`:

```python
DEMO_MODE = True  # baris ~295 di Sudoku.py
```

Saat `DEMO_MODE = True`, seluruh prediksi ML mengembalikan nilai *fallback* yang aman tanpa memanggil model `sklearn`, sehingga presentasi tetap berjalan lancar.

---

## Kredit Akademik

Proyek ini dikembangkan untuk memenuhi kebutuhan:

- **Ujian Akhir Semester (UAS)** Mata Kuliah *Machine Learning for Intelligence System*
- **Pameran Karya Mahasiswa** Universitas Bunda Mulia
- **Pendaftaran HAKI** sebagai Ciptaan Program Komputer

| Detail | Informasi |
|--------|-----------|
| Penulis | Samuel Lie |
| Institusi | Universitas Bunda Mulia (UBM) |
| Program Studi | Data Science |
| Mata Kuliah | Machine Learning for Intelligence System |
| Dosen Pengampu | Puguh Hiskiawan, S.Si., M.Si., Ph.D. |
| Tahun | 2026 |

---

## Hak Cipta

**Copyright &copy; 2026 Samuel Lie. Seluruh hak dilindungi.**

Karya ini diajukan untuk pendaftaran **Hak Kekayaan Intelektual (HAKI)** sebagai Ciptaan Program Komputer. Dilarang keras menyalin, mendistribusikan, atau memodifikasi kode, aset, atau model dalam repositori ini tanpa izin tertulis dari penulis.

> *This work is submitted for Intellectual Property Rights (HAKI) registration as a Computer Program Creation. Unauthorized copying, distribution, or modification of any code, asset, or model in this repository is strictly prohibited without written permission from the author.*