# ⬛ Sudoku AI — Machine Learning Intelligence System

Sudoku berbasis GUI Python dengan sistem **Machine Learning** yang terintegrasi penuh. Program ini mampu mengklasifikasi tipe pemain, merekomendasikan tingkat kesulitan secara adaptif, mendeteksi sesi anomali, dan memprediksi performa pemain ke depannya — semua secara real-time.

---

## 📁 Struktur Proyek

```
sudoku-ai/
├── Sudoku.py                  # Aplikasi utama (GUI + ML engine)
├── Sudoku_ML_Models.ipynb     # Notebook training & evaluasi model
├── requirements.txt           # Dependensi Python
├── sudoku_data.json           # Data sesi pemain (dibuat otomatis)
├── sudoku_music.mp3           # File musik (diunduh otomatis dari Google Drive)
├── KNN.pkl                    # Model tipe pemain (dibuat otomatis)
├── LR.pkl                     # Model prediksi skor (dibuat otomatis)
├── ISO.pkl                    # Model deteksi anomali (dibuat otomatis)
├── RFC.pkl                    # Model rekomendasi kesulitan (dibuat otomatis)
└── Multi.pkl                  # Model profil statistik (dibuat otomatis)
```

> File `.pkl` dan `.json` dibuat **otomatis** saat pertama kali program dijalankan. Tidak perlu dibuat manual.

---

## 🚀 Cara Menjalankan

### 1. Clone repositori

```bash
git clone https://github.com/username/sudoku-ai.git
cd sudoku-ai
```

### 2. (Opsional) Buat virtual environment

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 3. Install dependensi

```bash
pip install -r requirements.txt
```

> **Catatan untuk tkinter:** Jika muncul error `ModuleNotFoundError: No module named 'tkinter'`, jalankan:
> - **Ubuntu/Debian:** `sudo apt-get install python3-tk`
> - **macOS (Homebrew):** `brew install python-tk`
> - **Windows:** tkinter sudah termasuk dalam installer Python resmi.

### 4. Jalankan game

```bash
python Sudoku.py
```

### 5. (Opsional) Jalankan notebook training model

```bash
jupyter notebook Sudoku_ML_Models.ipynb
```

---

## 🎮 Fitur Utama

| Fitur | Deskripsi |
|-------|-----------|
| **Grid 4×4 & 9×9** | Dua ukuran papan dengan tingkat kesulitan Easy / Normal / Hard |
| **AI Solver** | Backtracking MRV (Minimum Remaining Values) dengan animasi langkah |
| **Draft / Pencil Mode** | Mode coret-coretan kandidat angka (Hard mode) dengan auto-fill constraint propagation |
| **Tema Dark & Light** | Toggle tema instan tanpa restart |
| **Musik Latar** | Diunduh otomatis dari Google Drive, toggle dengan tombol `M` |
| **Leaderboard** | Papan skor semua pemain, dapat difilter per grid dan kesulitan |
| **Performance Dashboard** | Grafik interaktif tren performa tiap sesi |

---

## 🧠 Sistem Machine Learning

Lima model scikit-learn yang bekerja secara otomatis di balik layar:

| File PKL | Model | Fungsi |
|----------|-------|--------|
| `KNN.pkl` | `KNeighborsClassifier` + `StandardScaler` | Klasifikasi tipe pemain (5 kelas) |
| `LR.pkl` | `LinearRegression` | Prediksi skor sesi berikutnya |
| `ISO.pkl` | `IsolationForest` + `StandardScaler` | Deteksi sesi anomali |
| `RFC.pkl` | `RandomForestClassifier` + `StandardScaler` | Rekomendasi tingkat kesulitan |
| `Multi.pkl` | `MultiOutputRegressor(RFR)` + `StandardScaler` | Prediksi 11 metrik profil pemain |

### Tipe Pemain yang Dikenali

| Tipe | Ciri |
|------|------|
| ⚡ Speedrunner | Cepat & akurat |
| 🧩 Careful | Hati-hati & teliti |
| 📚 Learner | Sedang berkembang |
| 💪 Struggling | Butuh bantuan |
| 🎲 Inconsistent | Tidak konsisten |

### Fitur Input Model

Setiap sesi pemain direpresentasikan oleh vektor fitur berikut:

| Fitur | Keterangan |
|-------|-----------|
| `tpc` | Waktu rata-rata per gerakan (detik) |
| `er` | Error rate (error / total gerakan) |
| `hr` | Hint rate (hint / total gerakan) |
| `cr` | Completion rate (0 atau 1 per sesi) |
| `nmr` | Near-miss rate |
| `gur` | Guessing rate |
| `moves` | Rata-rata jumlah gerakan (RFC & Multi) |
| `score` | Skor sesi (RFC & Multi) |

---

## 📊 Notebook Training (`Sudoku_ML_Models.ipynb`)

Notebook ini mencakup:

1. **Generasi Data Sintetis** — 1.200 sampel per kelas untuk KNN, 2.000 sampel untuk RFC & Multi dengan noise Gaussian dan overlap antar kelas
2. **EDA** — Distribusi fitur per tipe pemain dan heatmap korelasi
3. **Training & Tuning** — GridSearchCV, StratifiedKFold-5, learning curve
4. **Evaluasi** — Classification report, confusion matrix, MSE, MAE, R²
5. **Export** — Semua model disimpan ke `.pkl` untuk dipakai langsung oleh `Sudoku.py`

---

## ⌨️ Shortcut Keyboard

| Tombol | Aksi |
|--------|------|
| `1–9` | Input angka ke sel yang dipilih |
| `Backspace / Delete` | Hapus angka di sel |
| `Arrow keys` | Navigasi sel |
| `D` | Toggle draft/pencil mode (Hard mode) |
| `Enter / Space` | Konfirmasi naked single di draft mode |
| `M` | Toggle musik latar |
| `Esc` | Toggle fullscreen |

---

## 🔧 Kompatibilitas

| Komponen | Versi |
|----------|-------|
| Python | ≥ 3.9 |
| scikit-learn | ≥ 1.3, < 1.6 |
| numpy | ≥ 1.24, < 2.0 |
| pandas | ≥ 2.0 (notebook) |
| matplotlib | ≥ 3.7 (dashboard & notebook) |
| seaborn | ≥ 0.13 (notebook) |
| pygame | ≥ 2.5 (opsional, musik) |
| tkinter | stdlib (bawaan Python) |

---

## 📝 Catatan

- File `.pkl` divalidasi sebelum disimpan ulang — model lama tidak akan ditimpa kecuali model baru memiliki akurasi yang lebih baik.
- Retrain model berjalan di **background thread** agar tidak memblokir UI.
- Data sesi disimpan di `sudoku_data.json` dalam direktori yang sama dengan `Sudoku.py`.
- Musik diunduh satu kali dan disimpan lokal; tidak ada unduhan ulang jika file sudah ada.
