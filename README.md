<div align="center">

<!-- HEADER BANNER -->
<img src="logo.png" alt="Sudoku AI Logo" width="96" height="96" />

# 🧩 Sudoku AI - ML Intelligence System

> **Game Sudoku berbasis Python** dengan kecerdasan buatan terintegrasi.  
> Dari klasifikasi tipe pemain, prediksi skor, hingga deteksi anomali - semua berjalan secara _real-time_ langsung di dalam game.

<br/>

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![tkinter](https://img.shields.io/badge/GUI-Tkinter-orange?style=for-the-badge)](https://docs.python.org/3/library/tkinter.html)
[![scikit-learn](https://img.shields.io/badge/ML-scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org)
[![Lines](https://img.shields.io/badge/Lines%20of%20Code-9%2C666-purple?style=for-the-badge)]()
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)]()

<br/>

</div>

---

## 📖 Daftar Isi

- [Tentang Proyek](#-tentang-proyek)
- [Fitur Unggulan](#-fitur-unggulan)
- [Arsitektur ML](#-arsitektur-machine-learning)
- [Tampilan & Layar](#-tampilan--layar)
- [Struktur Proyek](#-struktur-proyek)
- [Persyaratan Sistem](#-persyaratan-sistem)
- [Instalasi & Menjalankan](#-instalasi--menjalankan)
- [File Model ML](#-file-model-ml)
- [Cara Bermain](#-cara-bermain)
- [Achievement System](#-achievement-system)
- [Kontribusi](#-kontribusi)

---

## 🎯 Tentang Proyek

**Sudoku AI** bukan sekadar game Sudoku biasa. Di balik tampilan yang bersih dan responsif, terdapat sebuah **sistem Machine Learning** yang terus belajar dari pola bermainmu - mengenali tipe pemainmu, memprediksi performa berikutnya, merekomendasikan tingkat kesulitan yang tepat, bahkan mendeteksi kalau sesimu terasa "tidak biasa".

Proyek ini dibangun sepenuhnya dengan Python, menggunakan `tkinter` sebagai antarmuka grafis dan `scikit-learn` sebagai fondasi model ML-nya. Seluruh logika mulai dari generasi puzzle, solver AI, hingga pipeline ML berjalan dalam **satu file Python tunggal** sepanjang hampir 10.000 baris.

---

## ✨ Fitur Unggulan

### 🎮 Game Engine

- **Dua ukuran grid:** 4×4 (kotak 2×2) dan 9×9 (kotak 3×3)
- **Tiga tingkat kesulitan:**
  - 🍃 **Easy** - 35% sel dihapus, palet hijau tosca
  - ⚡ **Normal** - 50% sel dihapus, palet biru safir
  - 🔥 **Hard** - 65% sel dihapus, palet merah coral
- **Highlight pintar:** baris, kolom, dan kotak yang berkaitan dengan sel terpilih langsung diwarnai secara adaptif sesuai tema
- **Auto-fill kandidat:** opsi isi kandidat otomatis (draft mode) khusus Hard
- **Sistem skor yang adil:** skor dihitung berdasarkan kecepatan _per sel yang dikerjakan sendiri_, bukan total waktu - hint dan auto-fill langsung memotong skor

### 🤖 AI Demo (Attractor Screen)

- Saat layar login idle selama ±45 detik, aplikasi secara otomatis menampilkan **live demo** AI memainkan Sudoku menggunakan **MRV Backtracking Solver**
- Visualisasi langkah demi langkah: warna sel berubah setiap kali AI mengisi angka atau melakukan _backtrack_
- Panel kiri menampilkan statistik global semua pemain, panel kanan menampilkan insight ML real-time

### 🏆 Sistem Prestasi (Achievement)

- **22 badge** yang terbagi dalam 6 tier: mulai dari kemenangan pertama hingga "Perfect" (tanpa error & tanpa hint)
- Badge ditampilkan via popup animasi yang muncul tepat setelah puzzle selesai
- Lihat detail semua badge di bagian [Achievement System](#-achievement-system)

### 📊 Performance Dashboard

- Dashboard lengkap yang muncul setiap kali selesai bermain, menampilkan:
  - **Grafik tren performa** interaktif (Matplotlib) - pilih metrik: skor, waktu, error, hint, atau auto
  - **Tipe pemain** hasil klasifikasi KNN beserta confidence %
  - **Rekomendasi AI:** grid dan tingkat kesulitan berikutnya
  - **Insight AI** berupa analisis tekstual singkat dari pola bermainmu
  - **Score Card eksportable** - PNG bergaya kartu dengan gradient dan statistik lengkap

### 🎵 Audio & Efek Suara

- Background music (MP3 looping) via `pygame.mixer` - toggle dengan tombol `M` atau ikon di pojok kanan bawah
- **SFX generatif** dihasilkan secara programatis menggunakan NumPy (tanpa file audio eksternal):
  - ✅ Sine wave 880 Hz untuk input benar
  - ❌ Sawtooth + noise 200 Hz untuk input salah
  - 🎉 Arpeggio C-E-G fanfare saat puzzle selesai

### 🌗 Tema Dark / Light

- Toggle tema kapan saja via ikon matahari/bulan di pojok kanan atas
- Seluruh palet warna (50+ token warna) beralih secara real-time termasuk palet per-kesulitan
- State tema dipertahankan saat berpindah layar dengan mekanisme `_rebuild_fn`

### 👥 Multi-Pemain & Leaderboard

- Profil pemain tersimpan per-username di `sudoku_data.json`
- Layar **Ganti Pemain** dua panel: daftar pemain di kiri, detail profil + riwayat sesi di kanan
- **Leaderboard global** dengan filter grid dan kesulitan, ditampilkan sebagai tabel scrollable
- **Kiosk Mode:** tekan `F5` atau `Ctrl+Shift+R` untuk reset cepat tanpa dialog - cocok untuk instalasi pameran

---

## 🧠 Arsitektur Machine Learning

Seluruh logika ML terpusat di kelas `PlayerMLEngine`. Terdapat **6 model** yang bekerja bersama:

| Model | File PKL | Fungsi | Scaler |
|---|---|---|---|
| **KNeighborsClassifier** | `KNN.pkl` | Klasifikasi tipe pemain (5 kelas) | StandardScaler |
| **RandomForestRegressor** | `RFR.pkl` | Prediksi skor sesi berikutnya | Tidak (tree-based) |
| **IsolationForest** | `ISO.pkl` | Deteksi sesi anomali | StandardScaler |
| **RandomForestClassifier** (GBM) | `GBM.pkl` | Rekomendasi tingkat kesulitan | StandardScaler |
| **MultiOutputRegressor** | `Multioutput_regressor_GBM.pkl` | Profil skill 4 dimensi | StandardScaler |
| **RandomForestRegressor** (hint) | `HintTimer_RFR.pkl` | Ambang batas idle sebelum hint muncul | Tidak |

### Alur Kerja ML

```
┌─────────────────────────────────────────────────────┐
│                  Data Sesi Pemain                   │
│  [waktu, error, hint, selesai, near_miss, guessing] │
└───────────────────┬─────────────────────────────────┘
                    │  Feature Extraction
                    ▼
┌─────────────────────────────────────────────────────┐
│                  PlayerMLEngine                     │
│                                                     │
│  ┌──────────────┐    ┌──────────────────────────┐  │
│  │ KNN Classify │───▶│  Tipe Pemain + Conf %     │  │
│  └──────────────┘    └──────────────────────────┘  │
│                                                     │
│  ┌──────────────┐    ┌──────────────────────────┐  │
│  │   RFR Score  │───▶│  Prediksi Skor Berikutnya │  │
│  └──────────────┘    └──────────────────────────┘  │
│                                                     │
│  ┌──────────────┐    ┌──────────────────────────┐  │
│  │    ISO Det.  │───▶│  Normal / Anomali         │  │
│  └──────────────┘    └──────────────────────────┘  │
│                                                     │
│  ┌──────────────┐    ┌──────────────────────────┐  │
│  │   GBM Diff   │───▶│  Rekomendasi Kesulitan    │  │
│  └──────────────┘    └──────────────────────────┘  │
│                                                     │
│  ┌──────────────┐    ┌──────────────────────────┐  │
│  │ Multi Skill  │───▶│  Speed / Akurasi / dst.   │  │
│  └──────────────┘    └──────────────────────────┘  │
└─────────────────────────────────────────────────────┘
```

### Continuous Improvement

Setelah setiap sesi disimpan, `_ml_schedule_retrain()` menjalankan retrain **semua model di background thread** menggunakan `threading.Lock` agar tidak ada dua retrain berjalan bersamaan. Model lama hanya ditimpa jika model baru benar-benar lebih baik:

- **KNN:** retrain hanya disimpan jika CV F1-macro baru ≥ F1 notebook + 0.5%
- **RFR:** retrain hanya disimpan jika OOB score baru lebih tinggi dari R² sebelumnya
- **Proteksi data minim:** jika sesi aktual < 20 (KNN) atau < 15 (RFR), model notebook yang jauh lebih representatif tetap digunakan

### Tipe Pemain

KNN mengklasifikasikan pemain ke dalam 5 tipe berdasarkan 6 fitur rata-rata:

| Tipe | Ciri Khas |
|---|---|
| ⚡ **Speedrunner** | Waktu per sel ≤ 4 detik, error rate < 5% |
| 🧩 **Careful** | Waktu per sel ≥ 14 detik, teliti dan jarang salah |
| 📚 **Learner** | Performa sedang, masih dalam proses belajar |
| 💪 **Struggling** | Error rate tinggi > 30%, sering pakai hint |
| 🎲 **Inconsistent** | Variance tinggi, performa tidak stabil |

---

## 🖥️ Tampilan & Layar

```
SudokuApp (root controller)
│
├── LoginScreen            - Input username, idle → Attractor
├── AttractorScreen        - Demo AI solver + statistik global
├── PlayerSelectScreen     - Daftar pemain + detail profil (2 panel)
├── GridSizeScreen         - Pilih 4×4 atau 9×9
├── DifficultyScreen       - Pilih Easy / Normal / Hard
├── GameScreen             - Layar utama bermain
│   ├── Board (Canvas)     - Grid interaktif + highlight
│   ├── Numpad             - Tombol angka 1–9 (atau 1–4)
│   ├── Sidebar            - Timer, skor, hint, error counter
│   └── ML Panel (toggle) - Insight real-time dari semua model
├── PerformanceDashboard   - Statistik sesi + chart + rekomendasi
├── LeaderboardWindow      - Top pemain per grid & kesulitan
├── AchievementPopup       - Animasi badge baru yang diraih
└── TutorialOverlay        - Panduan singkat cara bermain
```

Setiap layar adalah subkelas `tk.Frame` yang ditempatkan via `.place()` di atas `root`. Pergantian layar dilakukan dengan `_clear()` + inisialisasi layar baru - state tema dipertahankan lewat `_rebuild_fn`.

---

## 📁 Struktur Proyek

```
sudoku-ai/
│
├── Sudoku.py                    # File utama - seluruh logika ada di sini
├── sudoku_data.json             # Data pemain (dibuat otomatis saat pertama kali dijalankan)
├── sudoku_music.mp3             # Background music (opsional)
├── logo.png                     # Ikon aplikasi (opsional)
│
├── Models/
│   └── Files/
│       ├── KNN.pkl                          # KNeighborsClassifier + StandardScaler
│       ├── RFR.pkl                          # RandomForestRegressor (skor)
│       ├── ISO.pkl                          # IsolationForest + threshold
│       ├── GBM.pkl                          # Classifier rekomendasi kesulitan
│       ├── Multioutput_regressor_GBM.pkl    # MultiOutputRegressor (skill profile)
│       └── HintTimer_RFR.pkl               # RFR untuk ambang batas hint
│
└── Card/
    └── SudokuAI_<username>_<timestamp>.png  # Score card yang diekspor
```

> Folder `Models/Files/` dan `Card/` dibuat **otomatis** saat aplikasi pertama kali dijalankan. File PKL diisi secara bertahap saat model pertama kali dilatih.

---

## 💻 Persyaratan Sistem

| Kebutuhan | Versi Minimum |
|---|---|
| Python | 3.8+ |
| tkinter | Bawaan Python (termasuk di instalasi standar) |
| NumPy | 1.20+ |
| scikit-learn | 0.24+ _(opsional, tapi disarankan)_ |
| Pillow (PIL) | 8.0+ _(opsional, untuk ekspor Score Card)_ |
| pygame | 2.0+ _(opsional, untuk musik dan SFX)_ |

Aplikasi tetap berjalan penuh meskipun `scikit-learn`, `Pillow`, atau `pygame` tidak terinstal - masing-masing modul memiliki fallback graceful.

---

## 🚀 Instalasi & Menjalankan

### 1. Clone Repositori

```bash
git clone https://github.com/username/sudoku-ai.git
cd sudoku-ai
```

### 2. Buat Virtual Environment (disarankan)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Instal Dependensi

**Instalasi lengkap (semua fitur aktif):**

```bash
pip install numpy scikit-learn Pillow pygame
```

**Instalasi minimal (hanya game tanpa ML penuh dan audio):**

```bash
pip install numpy
```

### 4. Jalankan Aplikasi

```bash
python Sudoku.py
```

Aplikasi akan terbuka dalam mode **fullscreen**. Tekan `Esc` atau tutup jendela untuk keluar.

### 5. Tambahkan File Opsional

Letakkan file berikut di direktori yang sama dengan `Sudoku.py` untuk mengaktifkan fitur tambahan:

```
sudoku-ai/
├── sudoku_music.mp3   # Musik latar (format MP3)
├── logo.png           # Ikon aplikasi dan header
└── Models/Files/      # Letakkan file .pkl dari notebook Anda di sini
```

---

## 📦 File Model ML

Model `*.pkl` dapat dihasilkan dari **notebook Jupyter** yang disertakan, atau dibiarkan kosong - aplikasi akan melatih model dari data sintetis bawaan dan memperbaruinya seiring data pemain bertambah.

| File | Notebook Sumber | Keterangan |
|---|---|---|
| `KNN.pkl` | `Model_KNN.ipynb` | Klasifikasi tipe pemain |
| `RFR.pkl` | `Model_Random_Forest_Regressor.ipynb` | Prediksi skor |
| `ISO.pkl` | `Model_Isolation_Forest.ipynb` | Deteksi anomali + threshold optimal |
| `GBM.pkl` | `Model_GBM.ipynb` | Rekomendasi kesulitan |
| `Multioutput_regressor_GBM.pkl` | `Model_Multioutput_Regressor.ipynb` | Profil skill |
| `HintTimer_RFR.pkl` | - | Dilatih otomatis oleh aplikasi |

Jika file PKL belum ada, aplikasi tetap berfungsi dengan model yang dilatih dari data sintetis 20 sampel bawaan, dan secara otomatis meningkat kualitasnya seiring sesi pemain bertambah.

---

## 🕹️ Cara Bermain

1. **Login** - masukkan nama pemain (baru atau yang sudah ada)
2. **Pilih Grid** - 4×4 untuk pemula, 9×9 untuk tantangan
3. **Pilih Kesulitan** - Easy, Normal, atau Hard
4. **Bermain** - klik sel, lalu klik angka di numpad atau tekan angka di keyboard
5. **Shortcut Berguna:**

| Tombol | Fungsi |
|---|---|
| `1`–`9` | Isi angka di sel terpilih |
| `Delete` / `Backspace` | Hapus isi sel |
| `H` | Tampilkan hint (1 sel) |
| `M` | Toggle musik |
| `F5` / `Ctrl+Shift+R` | Reset cepat (Kiosk Mode) |
| `Esc` | Kembali ke layar sebelumnya |

6. **Setelah selesai** - dashboard performa tampil otomatis, badge baru (jika ada) muncul sebelumnya
7. **Ekspor Score Card** - tekan tombol di dashboard untuk menyimpan kartu PNG ke folder `Card/`

---

## 🏅 Achievement System

Terdapat **22 badge** yang terbagi dalam 6 tier. Setiap badge hanya diberikan sekali per pemain.

### Tier 1 - First Steps
| Badge | Kondisi |
|---|---|
| 🟢 **First Win** | Selesaikan puzzle pertama |
| 🟠 **Marathon** | Selesaikan 10 sesi total |
| 🩷 **Veteran** | Selesaikan 25 sesi total |

### Tier 2 - Streak & Konsistensi
| Badge | Kondisi |
|---|---|
| ❤️ **Consistent** | 3 sesi berturut-turut selesai |
| 🟣 **Unbeatable** | 5 sesi berturut-turut selesai |
| 🩷 **Serial Winner** | 7 sesi berturut-turut selesai |
| 🟠 **Comeback** | Selesai setelah sesi sebelumnya gagal |

### Tier 3 - Kecepatan
| Badge | Kondisi |
|---|---|
| 🥇 **Lightning** | 4×4 selesai di bawah 60 detik |
| 🥇 **Speed Flash** | 4×4 selesai di bawah 30 detik |
| 🔵 **Speed Demon** | 9×9 selesai di bawah 5 menit |

### Tier 4 - Akurasi & Kemandirian
| Badge | Kondisi |
|---|---|
| 🔵 **No Hints** | Selesai tanpa satu pun hint |
| 🟣 **Flawless** | Selesai tanpa satu pun error |
| 🥇 **Perfect** | Selesai tanpa error DAN tanpa hint |
| 🟢 **Efficient** | Selesai dengan kurang dari 3 error |

### Tier 5 - Kesulitan & Grid
| Badge | Kondisi |
|---|---|
| ❤️ **Hard Expert** | Selesaikan puzzle Hard |
| ❤️ **Iron Will** | Selesaikan Hard tanpa hint/auto |
| 🟣 **Master 9×9** | Selesaikan puzzle 9×9 |
| 🔵 **Explorer** | Mainkan semua 3 tingkat kesulitan |

### Tier 6 - Skor
| Badge | Kondisi |
|---|---|
| 🥇 **Genius** | Raih skor > 800 dalam satu sesi |
| ❤️ **Expert** | Raih skor > 500 di mode Hard |

---

## 🏗️ Detail Teknis Menarik

**Atomic Write** - data JSON dan file PKL tidak pernah terkorupsi saat crash. Write dilakukan ke file `.tmp` terlebih dahulu, baru `os.replace()` secara atomik ke file asli.

**PKL Cache RAM** - semua model di-load dari disk hanya sekali per sesi. Akses berikutnya langsung dari `_PKL_CACHE` dictionary di memori (~0ms).

**Thread Safety** - retrain model berjalan di background thread dengan `threading.Lock`. Tidak ada dua retrain yang berjalan bersamaan, dan hasil dikembalikan ke main thread via `root.after(0, callback)`.

**Deduplication Sesi** - setiap sesi memiliki _fingerprint_ unik dari 11 field. Sesi duplikat (misalnya akibat crash atau double-save) dibuang secara otomatis.

**MRV Solver** - solver AI menggunakan **Minimum Remaining Values** heuristic: pilih sel dengan kandidat paling sedikit terlebih dahulu. Jauh lebih efisien dari backtracking biasa karena deteksi kegagalan lebih awal dan search tree lebih kecil.

---

## 🤝 Kontribusi

Pull request dan issue sangat disambut! Beberapa area yang bisa dikembangkan:

- Tambah ukuran grid 6×6 atau 16×16
- Integrasi model ML yang lebih canggih (XGBoost, neural network)
- Export riwayat sesi ke CSV
- Mode multiplayer online
- Lokalisasi bahasa lain

Pastikan kode mengikuti gaya yang sudah ada (docstring Bahasa Indonesia, token warna dari design tokens) sebelum mengajukan PR.

---

## 📄 Lisensi

Didistribusikan di bawah lisensi **MIT**. Lihat file `LICENSE` untuk detail lengkap.

---

<div align="center">

Dibuat dengan ❤️ dan banyak ☕  
_"Setiap angka di papan adalah keputusan. Setiap keputusan adalah data."_

</div>