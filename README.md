# 🧩 Sudoku AI — ML Intelligence System

> Sudoku klasik yang ditingkatkan dengan kecerdasan buatan: profil pemain adaptif, rekomendasi kesulitan real-time, solver otomatis, dan sistem skor kompetitif — semuanya dalam satu aplikasi desktop Python.

---

## 📸 Tampilan

| Login & Tema | Papan Permainan | Profil Pemain |
|:---:|:---:|:---:|
| Dark / Light mode | Draft Mode (Hard) | Statistik & ML |

---

## ✨ Fitur Utama

### 🎮 Gameplay
- **Dua ukuran grid** — 4×4 (santai) dan 9×9 (standar)
- **Tiga tingkat kesulitan** — Easy, Normal, Hard, masing-masing dengan persentase sel kosong yang berbeda
- **Mode Draft (Hard only)** — tandai angka kandidat di sudut sel layaknya Sudoku profesional, dengan tampilan mini 3×3 per sel
- **Sistem Hati** — jumlah hint terbatas sesuai ukuran grid (4 hati untuk 4×4, 9 hati untuk 9×9)
- **Deteksi near-miss & asal tebak** — game mencatat apakah kesalahan berasal dari hampir benar (1 error per sel) atau asal tebak (2+ error per sel)
- **Timer aktif** — dihitung sejak angka pertama dimasukkan

### 🤖 AI & Machine Learning
Sudoku AI menggunakan **5 model ML** yang dilatih ulang di background setiap kali sesi selesai:

| Model | File | Fungsi |
|---|---|---|
| `KNeighborsClassifier` | `KNN.pkl` | Klasifikasi tipe pemain |
| `LinearRegression` | `LR.pkl` | Prediksi skor sesi berikutnya |
| `IsolationForest` | `ISO.pkl` | Deteksi anomali performa |
| `RandomForestClassifier` | `RFC.pkl` | Rekomendasi difficulty |
| `MultiOutputRegressor` | `Multi.pkl` | Proyeksi multi-metrik |

**Tipe pemain yang dikenali:**
- ⚡ **Speedrunner** — cepat, presisi tinggi, jarang pakai hint
- 🛡 **Careful** — lambat tapi akurat, konsisten
- 📚 **Learner** — error rate sedang, sedang berkembang
- 💪 **Struggling** — banyak error atau hint, butuh dukungan
- 🎲 **Inconsistent** — performa tidak menentu, sering menebak

### 🔢 Solver Otomatis
Algoritma **Backtracking MRV (Minimum Remaining Values)** + Forward Checking:
- Memilih sel dengan kemungkinan kandidat paling sedikit terlebih dahulu
- Eliminasi kandidat otomatis (forward checking)
- Tombol `⚡ Auto` mengisi kandidat otomatis di mode Draft

### 📊 Sistem Skor
Skor dihitung berdasarkan formula multi-faktor:

```
Skor = Base × Difficulty Multiplier × Speed Bonus − Error Penalty − Hint Penalty
```

- **Base score** dari kecepatan per sel
- **Multiplier** — Easy ×1.0, Normal ×1.5, Hard ×2.2
- **Penalti error** — setiap kesalahan memotong skor
- **Bonus near-miss** — mendekati benar tapi salah tetap diapresiasi
- **Penalti tebak** — asal tebak (2+ error per sel) dihukum lebih berat

### 🏆 Leaderboard
- Menampilkan **satu skor terbaik per pemain** (bukan semua sesi)
- Filter berdasarkan ukuran grid (4×4 / 9×9) dan kesulitan (All / Easy / Normal / Hard)
- Tampil dengan ranking, waktu, moves, error, dan skor

### 👥 Manajemen Pemain
- Daftar semua pemain terdaftar dengan profil lengkap
- Panel dua kolom: list pemain di kiri, detail statistik di kanan
- Statistik mencakup: win rate, error rate, hint rate, avg time/sel, total playtime, best score
- Badge tipe pemain dengan warna dan ikon unik

---

## 🖥 Tampilan Antarmuka

### Tema
Aplikasi mendukung dua tema yang dapat diubah kapan saja tanpa kehilangan progress:

| Tema | Latar | Aksen |
|---|---|---|
| 🌑 **Dark** (default) | `#0D1117` | Biru `#58A6FF` |
| ☀️ **Light** | `#F0F2F5` | Biru `#1A6FBF` |

Tombol toggle tema berada di pojok kanan atas layar.

### Layar & Navigasi
```
Login Screen
  ├── Start Playing (username baru)
  ├── Daftar Pemain (pilih akun lama)
  └── Leaderboard

Daftar Pemain
  └── Pilih Pemain → Grid Size Screen
                        └── Difficulty Screen
                              └── Game Screen
                                    └── Performance Dashboard
```

### Panel Sidebar (saat bermain)
- Ganti difficulty tanpa keluar game
- Kontrol: New Game, Hint, Leaderboard, Ganti Pemain, Logout
- Mode Draft (hanya Hard): toggle, Auto Fill, Hapus
- AI Solver: Backtracking MRV
- Statistik sesi: Moves, Errors, Hints, Auto, Hampir Benar, Asal Tebak
- Indikator Hati (hint tersisa)
- Tombol HINT besar

---

## 🎵 Musik Latar
Aplikasi mengunduh file musik `.mp3` otomatis dari Google Drive saat pertama dijalankan (jika koneksi tersedia). Musik dapat di-toggle dengan shortcut **`[M]`**. Memerlukan library `pygame`.

---

## 🛠 Instalasi & Cara Menjalankan

### Persyaratan Sistem
- Python **3.8+**
- OS: Windows / macOS / Linux (dengan Tkinter)

### Dependensi

```bash
pip install numpy scikit-learn pygame
```

> Tkinter sudah termasuk dalam instalasi Python standar. Jika tidak ada, install via:
> `sudo apt install python3-tk` (Linux/Debian)

### Menjalankan

```bash
python Sudoku.py
```

### File yang Dihasilkan
Aplikasi membuat beberapa file otomatis di direktori yang sama:

| File | Keterangan |
|---|---|
| `sudoku_data.json` | Data pemain, sesi, dan statistik |
| `KNN.pkl` | Model klasifikasi pemain |
| `LR.pkl` | Model prediksi skor |
| `ISO.pkl` | Model deteksi anomali |
| `RFC.pkl` | Model rekomendasi difficulty |
| `Multi.pkl` | Model proyeksi multi-metrik |
| `sudoku_music.mp3` | File musik (diunduh otomatis) |

---

## ⌨️ Shortcut Keyboard

| Tombol | Fungsi |
|---|---|
| `1` – `9` | Input angka ke sel aktif |
| `Backspace` / `Delete` | Hapus angka di sel aktif |
| `Arrow keys` | Pindah sel |
| `D` | Toggle Mode Draft |
| `M` | Toggle musik |
| `Esc` | Toggle Fullscreen |
| `Enter` / `Space` | Konfirmasi kandidat tunggal (draft mode) |

---

## 🏗 Struktur Kode

```
Sudoku.py
├── Konfigurasi & Tema          (baris ~60–400)
│   ├── Dark / Light theme
│   └── Difficulty themes (per mode)
├── Logika Puzzle               (~540–830)
│   ├── generate_full_board()
│   ├── generate_puzzle()
│   └── solve_backtracking_mrv()
├── Sistem Skor                 (~580–640)
│   └── calculate_score()
├── ML Engine                   (~832–1240)
│   └── class PlayerMLEngine
│       ├── KNN, LR, IsolationForest
│       ├── RFC, MultiOutputRegressor
│       ├── classify_player()
│       ├── recommend_difficulty()
│       └── predict_next_score()
├── Layar UI                    (~1238–5082)
│   ├── AnimatedBG              — background bintang animasi
│   ├── LoginScreen             — halaman awal
│   ├── GridSizeScreen          — pilih 4×4 / 9×9
│   ├── DifficultyScreen        — pilih kesulitan + rekomendasi AI
│   ├── GameScreen              — papan utama + sidebar
│   ├── PerformanceDashboard    — hasil & analisis pasca game
│   ├── LeaderboardWindow       — papan skor global
│   └── PlayerSelectScreen      — manajemen & profil pemain
└── App Controller              (~5083–5480)
    └── class SudokuApp
        ├── Navigasi antar layar
        ├── Sistem rebuild tema
        └── Preservasi state game saat ganti tema
```

---

## 🔧 Konfigurasi Kustom

Edit bagian `KONFIGURASI` di awal file untuk menyesuaikan:

```python
# Ganti link musik Google Drive
GDRIVE_LINK_MUSIC = "https://drive.google.com/file/d/..."

# Nama file model ML
PKL_KNN   = "KNN.pkl"
PKL_LR    = "LR.pkl"
PKL_ISO   = "ISO.pkl"
PKL_RFC   = "RFC.pkl"
PKL_MULTI = "Multi.pkl"
```

---

## 🐛 Catatan Teknis

- Model ML otomatis di-retrain di **background thread** setelah setiap sesi selesai menggunakan `threading.Lock` untuk mencegah retrain ganda
- Ganti tema **tidak mereset papan atau timer** — state game disimpan penuh dan dipulihkan ke instance baru
- Saat ganti tema, timer lama dihentikan terlebih dahulu sebelum widget dihancurkan untuk mencegah duplikasi loop `_tick`
- Pilihan pemain di `PlayerSelectScreen` dipertahankan saat tema diganti via callback `on_highlight`
- Leaderboard hanya menampilkan **satu entri terbaik per pemain** agar ranking tidak didominasi oleh satu orang

---

## 📄 Lisensi

Proyek ini dibuat untuk keperluan edukasi dan pengembangan pribadi.

---

<div align="center">

Dibuat dengan ❤️ menggunakan Python · Tkinter · scikit-learn · pygame

</div>
