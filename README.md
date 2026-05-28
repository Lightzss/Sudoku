<div align="center">

<img src="Assets/logo.png" alt="Sudoku AI Logo" width="120" />

# SUDOKU AI

**Game Sudoku berbasis Python yang ditenagai 6 model Machine Learning terintegrasi untuk menganalisis, memprediksi, dan beradaptasi dengan gaya bermain pemain secara *real-time*.**

<br/>

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![pygame](https://img.shields.io/badge/pygame-2.5%2B-1A1A2E?style=for-the-badge&logo=pygame&logoColor=white)](https://pygame.org)
[![Pillow](https://img.shields.io/badge/Pillow-10.0%2B-11557C?style=for-the-badge&logo=python&logoColor=white)](https://pillow.readthedocs.io)
[![License](https://img.shields.io/badge/License-All%20Rights%20Reserved-red?style=for-the-badge)](./README.md)
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey?style=for-the-badge)](./README.md)
[![Version](https://img.shields.io/badge/Version-1.0.0-brightgreen?style=for-the-badge)](./README.md)

</div>

---

## 🧩 ***About Project***

**Sudoku AI** adalah game Sudoku desktop lengkap yang dibangun sepenuhnya dengan Python, mengintegrasikan *pipeline* 6 model *Machine Learning* langsung ke dalam alur permainan. Program ini membuat *puzzle* melalui algoritma *backtracking* acak, mencatat setiap aksi pemain selama sesi berlangsung, lalu mengumpankan data tersebut ke rangkaian model *sklearn* yang mengklasifikasikan tipe pemain, memprediksi skor berikutnya, mendeteksi sesi tidak wajar, merekomendasikan tingkat kesulitan, membangun profil kemampuan 11 dimensi, serta menghitung timer hint adaptif. Seluruh proses itu berjalan di *background thread* tanpa mengganggu jalannya game.

Antarmuka dibangun dengan `tkinter` dan dilengkapi dua tema (Dark theme / Light theme), animasi latar partikel bergerak, efek suara yang dibuat menggunakan `NumPy` dan `pygame`, tampilan *Leaderboard* dengan animasi blur, *Performance Dashboard* yang dapat di-*scroll* dengan grafik `matplotlib` interaktif, *Draft Mode* sistem *draft mode* untuk tingkat kesulitan Hard, serta *AI Solver* yang memutar ulang langkah *MRV Backtracking* secara animasi di layar.

Semua data pemain tersimpan persisten di `player_data.json`. Kartu skor diekspor sebagai gambar PNG ke folder `Card/` menggunakan `Pillow`.

---

## ✨ ***Key Features***

### 🎮 ***Game Features***

- ***Two grid sizes*** : 4x4 (box 2x2, angka 1 hingga 4) dan 9x9 (box 3x3, angka 1 hingga 9)
- ***Three difficulty levels*** : Easy (35% sel dikosongkan), Normal (50% dikosongkan), Hard (65% dikosongkan)
- ***Heart system*** : Setiap pemain memulai dengan N percobaan. Easy/Normal maksimal 5 error, Hard maksimal 3 *error*, bersifat adaptif berdasarkan tipe pemain
- ***Draft Mode*** (Hard only) : Sistem kandidat pensil dengan notasi sudut 9 posisi, *auto-fill* kandidat (⚡ Auto), konfirmasi *naked-single*, dan eliminasi kandidat otomatis setelah konfirmasi
- ***MRV Backtracking AI Solver*** : Animasi penyelesaian *step-by-step* menggunakan algortima *backtracking* dengan fungsi heuristik *Minimum Remaining Values* (MRV) dan *Forward Checking*
- ***Dark / Light theme*** : Sistem pemilihan tampilan antara tema gelap dan terang
- ***Animated background*** : 35 partikel mengambang yang dirender dalam *loop Canvas* setiap 50ms
- ***Procedural sound effects*** : 7 SFX dibuat menggunakan `NumPy` (`correct`, `error`, `win`, `achievement`, `select`, `hover`, `click`)
- ***Background music*** : Diputar dari `Assets/music.mp3` menggunakan `pygame`
- ***Attractor / Demo screen*** : Aktif otomatis setelah 30 detik *idle* pada tampilan login, menampilkan *AI solver* langsung, panel statistik global, dan kartu fitur ML
- ***Tutorial overlay*** : Muncul otomatis di game pertama pemain baru dan tidak akan muncul lagi
- ***20 achievement badges*** : Tersusun dalam 6 bagian yaitu `First Steps`, `Streak`, `Speed`, `Accuracy`, `Difficulty`, dan `Score`
- ***Leaderboard*** : *Overlay slide-up* dengan blur latar, bisa difilter berdasarkan ukuran grid (4x4 / 9x9) dan difficulty (Semua / Easy / Normal / Hard), menampilkan top-25 dengan `highlight` medali
- ***Score card export*** : Kartu PNG selebar 860px tersimpan di `Card/SudokuAI_{username}_{timestamp}.png`
- ***Multi-player profiles*** : Setiap *username* memiliki riwayat sesi, pencapaian, dan *state ML* yang independen
- ***Score formula*** : Multi-variable: `(time_score - error_penalty - behavior_penalty - hint_penalty - auto_penalty) x difficulty_multiplier` dengan pengali Easy=1.0, Normal=1.8, Hard=3.0
- ***Near-miss and guessing detection*** : *Error* pertama pada satu sel dicatat sebagai *near-miss*, error berulang pada sel yang sama ditandai sebagai perilaku *guessing*


### 🤖 ***AI / ML Features***

- ***KNN Player Classifier*** : Mengklasifikasikan riwayat sesi ke dalam 5 tipe pemain beserta persentase keyakinannya 
- ***HistGradientBoosting Score Predictor*** : Memprediksi skor yang akan dicapai pemain di sesi berikutnya berdasarkan tren terbaru
- ***IsolationForest Anomaly Detector*** : Mendeteksi sesi yang menyimpang secara signifikan dari pola normal pemain
- ***RandomForest Difficulty Recommender*** : Merekomendasikan Easy / Normal / Hard beserta alasannya, ditampilkan sebagai badge AI pada tampilan *Difficulty Selection*
- ***MultiOutputRegressor Performance Profiler*** : Memprediksi 11 dimensi kemampuan sekaligus (`speed index`, `accuracy index`, `consistency index`, `independence index`, dan 7 target metrik mentah)
- ***GradientBoosting Hint Timer*** : Menghitung ambang batas *idle* adaptif (8 hingga 120 detik) sebelum *hint* ditawarkan, menggunakan 15 fitur terrekayasa termasuk `log_tpc`, `err_x_hint`, `hint_pressure`, dan `patience_proxy`
- ***Continuous learning*** : Model dilatih ulang di *background daemon thread* setelah setiap sesi selesai menggunakan `player_data.json` terbaru, dan hanya disimpan jika CV score baru mengalahkan model sebelumnya
- ***PKL cache warming*** : Setiap file `.pkl` di-*preload* ke RAM saat *startup* untuk inferensi instan
- ***Rule-based fallback*** : Jika `sklearn` tidak tersedia atau data tidak cukup, setiap model diganti dengan aturan deterministik
- ***Live analysis panel*** : Tekan tombol `I` selama permainan untuk melihat *output ML* secara *real-time*

---

## 🧠 ***AI Model Architecture***

| File Model | Algoritma | Fitur Input | Fungsi dalam Game |
|---|---|---|---|
| `Player_Classifier.pkl` | KNeighborsClassifier + StandardScaler | 6 fitur: `avg_time_per_cell`, `error_rate`, `hint_rate`, `completion_rate`, `near_miss_rate`, `guessing_rate` | Mengklasifikasikan pemain ke dalam `Speedrunner` / `Careful` / `Learner` / `Struggling` / `Inconsistent` beserta persentase keyakinannya. Ditampilkan pada *Dashboard* dan *Difficulty Selection*. Dilatih dari 2.500 sampel sintetis (500 per kelas). |
| `Score_Prediction.pkl` | HistGradientBoostingRegressor | 4 fitur: `session_idx`, `time_per_cell`, `error_rate`, `hint_rate` | Memprediksi skor yang akan diraih pemain di sesi berikutnya. Ditampilkan di *Performance Dashboard*. Membutuhkan minimal 3 sesi. Dibandingkan dengan RFR, XGBoost, dan LightGBM saat pelatihan. HistGBR terpilih sebagai pemenang. |
| `Detect_Anomaly.pkl` | IsolationForest + StandardScaler | 6 fitur yang sama dengan *classifier* | Mendeteksi sesi dengan pola tidak wajar (misalnya penyelesaian terlalu cepat atau penggunaan *hint* ekstrem). `contamination=0.05`. *Threshold* diatur dari evaluasi *notebook*. Hasil ditampilkan di *Dashboard* sebagai `Normal` / `Anomali` / `Tidak Diketahui`. |
| `Difficulty_Recommender.pkl` | RandomForestClassifier + StandardScaler | 8 fitur agregat: `tpc`, `er`, `hr`, `cr`, `nmr`, `gur`, `avg_moves`, `avg_score` | Merekomendasikan tingkat kesulitan berikutnya (Easy=0 / Normal=1 / Hard=2). Ditampilkan sebagai *badge* AI di layar *Difficulty Selection*. Dibandingkan *head-to-head* dengan HistGradientBoostingClassifier saat *training*. Data sintetis menggunakan distribusi Beta/Gamma dengan label *noise* 15%. |
| `Performance_Prediction.pkl` | MultiOutputRegressor(HistGradientBoostingRegressor) + StandardScaler | 8 fitur sesi | Memprediksi 11 target sekaligus: `exp_tpc`, `exp_er`, `exp_hr`, `exp_cr`, `exp_nmr`, `exp_gur`, `exp_score`, `speed_idx`, `accuracy_idx`, `consistency_idx`, `independence_idx`. Empat indeks kemampuan (0 hingga 100) menggerakkan progress bar animasi di *Dashboard*. |
| `Hint_Timer.pkl` | GradientBoostingRegressor + StandardScaler | 15 fitur: 9 dasar + `log_tpc`, `err_x_hint`, `diff_x_grid`, `hint_pressure`, `move_density`, `patience_proxy` | Menghitung *threshold idle* adaptif dalam detik sebelum *hint* ditawarkan. batasi hanya dalam rentang 8-120 detik. Dilatih dengan `n_estimators=107`, `learning_rate=0.069`, `max_depth=4`. |

<details>
<summary><strong>Player Type Definitions</strong></summary>

| Tipe | Warna | Karakteristik |
|---|---|---|
| ⚡ `Speedrunner` | Emas | Waktu per sel rendah (maks 4 detik), *error rate* di bawah 5%, jarang memakai *hint* |
| 🧩 `Careful` | Hijau | Waktu per sel tinggi (minimal 14 detik), sangat akurat, jarang memakai *hint* |
| 📚 `Learner` | Biru | Kecepatan dan *error* moderat, penggunaan *hint* moderat |
| 💪 `Struggling` | Merah | *Error* rate tinggi (di atas 30%) atau *hint rate* tinggi (di atas 35%), *completion rate* rendah |
| 🎲 `Inconsistent` | Oranye | Variansi timing tinggi, metrik campuran tidak konsisten |

</details>

---

## 📁 ***Folder Structure***

```
SUDOKU/
│
├── Assets/                              # Aset statis yang dimuat saat runtime
│   ├── easter_egg.mp3                   # Audio tersembunyi (lihat bagian Easter Egg)
│   ├── easter_egg.mp4                   # Video tersembunyi (butuh opencv-python + Pillow)
│   ├── logo.png                         # Logo aplikasi di Login, Sidebar, Attractor screen
│   └── music.mp3                        # Musik latar yang diputar via pygame
│
├── Card/                                # Dibuat otomatis saat pertama kali dijalankan
│   └── SudokuAI_{user}_{ts}.png         # File PNG kartu skor yang diekspor (lebar 860px)
│
├── Models/
│   └── Files/                           # File biner model terlatih
│       ├── Detect_Anomaly.pkl           # IsolationForest + StandardScaler
│       ├── Difficulty_Recommender.pkl   # RandomForestClassifier + StandardScaler
│       ├── Hint_Timer.pkl               # GradientBoostingRegressor + StandardScaler
│       ├── Performance_Prediction.pkl   # MultiOutputRegressor(HistGBR) + StandardScaler
│       ├── Player_Classifier.pkl        # KNeighborsClassifier + StandardScaler
│       └── Score_Prediction.pkl         # HistGradientBoostingRegressor (tanpa scaler)
│
├── Model_Anomaly_Detection.ipynb        # Notebook pelatihan: IsolationForest
├── Model_Classify_Player.ipynb          # Notebook pelatihan: KNN dengan GridSearchCV
├── Model_Difficulty_Recommender.ipynb   # Notebook pelatihan: RFC vs HistGBR head-to-head
├── Model_Hint_Timer.ipynb               # Notebook pelatihan: GBR dengan feature engineering
├── Model_Performance_Prediction.ipynb   # Notebook pelatihan: MultiOutputRegressor
├── Model_Score_Prediction.ipynb         # Notebook pelatihan: RFR vs XGB vs LGBM vs HistGBR
│
├── player_data.json                     # Dibuat otomatis: basis data pemain persisten
├── requirements.txt                     # Dependensi Python
├── README.md                            # File ini
└── Sudoku.py                            # Titik masuk aplikasi utama (satu file tunggal)
```

---

## ⚙️ ***Installation and Running***

### *Prerequisites*

- Python 3.10+
- pip

### Langkah 1 - *Clone* atau unduh proyek

```bash
git clone https://github.com/your-username/sudoku-ai.git
cd sudoku-ai
```

### Langkah 2 - Instal dependensi

```bash
pip install -r requirements.txt
```

Library yang diinstal dari `requirements.txt`:

```
numpy>=1.24.0
scikit-learn>=1.3.0
pygame>=2.5.0
Pillow>=10.0.0
matplotlib>=3.7.0
opencv-python>=4.8.0
pandas>=2.0.0
seaborn>=0.12.0
scipy>=1.11.0
xgboost>=1.7.0
lightgbm>=4.0.0
notebook>=7.0.0
ipykernel>=6.0.0
```

> `opencv-python` dan `tkinter` bersifat opsional saat runtime. Game tetap berjalan dengan baik meski salah satunya tidak tersedia. `tkinter` sudah dibundel di sebagian besar distribusi Python. Di Linux mungkin perlu menjalankan `sudo apt install python3-tk`.

### Langkah 3 - Jalankan game

```bash
python Sudoku.py
```

Sebuah startup check akan tercetak di terminal, menampilkan library dan file model yang terdeteksi:

```
====================================================
  Sudoku AI  v1.0.0
  sklearn : Available
  pygame  : Available
  PIL     : Available
  cv2     : Available
----------------------------------------------------
[READY]   Player Classifier Model
[READY]   Score Prediction Model
[READY]   Anomaly Detection Model
[READY]   Difficulty Recommender Model
[READY]   Performance Prediction Model
[READY]   Hint Timer Model
====================================================
```

### Langkah 4 - (Opsional) *Retraining* model

Buka notebook mana saja di Jupyter lalu jalankan semua sel. Setiap notebook menyimpan outputnya ke `Models/Files/` secara otomatis.

```bash
jupyter notebook
```

---

## 🕹️ ***How to Play***

### Alur Permainan

```
*Login Screen*  →  *Grid Selection* (4x4 / 9x9)  →  *Difficulty Selection*  →  *Game*  →  *Performance Dashboard*
```

### Kontrol Keyboard

| Tombol | Fungsi |
|---|---|
| `1` hingga `9` | Memasukkan angka ke sel yang dipilih |
| `Backspace` / `Delete` | Menghapus isi sel yang dipilih |
| `Tombol Panah` | Memindahkan seleksi antar sel |
| `D` | Mulai *Draft Mode* (khusus difficulty Hard) |
| `Enter` / `Space` | Konfirmasi kandidat draft (*Draft Mode*, khusus Hard) |
| `I` | Tampilkan panel *Live ML Analysis* |
| `Esc` | Membuka dialog konfirmasi kembali ke pemilihan *grid* |
| `F5` | *Logout* dan kembali ke layar Login |

### Kontrol Mouse

- **Klik kiri sel** - Memilih sel tersebut
- **Tombol Numpad** - Memasukkan angka (tombol otomatis nonaktif jika angka sudah penuh ditempatkan)
- **Tombol ⌫ Delete** - Menghapus isi sel yang dipilih
- **Tombol 💡 HINT** - Menggunakan satu nyawa untuk mengungkap nilai benar sel yang dipilih
- **Tombol 🤖 Solve** - Menyaksikan AI menyelesaikan game secara *stey-by-step*
- **Tombol ⚡ Auto (khusus Hard)** - Mengisi semua tanda kandidat pensil secara otomatis (pengurangan 50 poin per penggunaan)

### Aturan Permainan

- Setiap baris, kolom, dan kotak kecil harus berisi masing-masing angka tepat satu kali.
- Penempatan yang salah menghabiskan satu nyawa (♥). Kehilangan semua nyawa mengakhiri sesi.
- Easy dan Normal memperbolehkan maksimal 5 error, Hard maksimal 3 error. Batas ini bersifat adaptif berdasarkan tipe pemain.
- Tombol `HINT` di mode Hard juga menambah hitungan *auto-used* yang memengaruhi skor.
- Skor akhir dihitung dari efisiensi waktu, tingkat *error*, penggunaan hint, dan pengali *difficulty*.

### Formula Skor

```
`skor = (time_score - error_penalty - behavior_penalty - hint_penalty - auto_penalty) x difficulty_multiplier`

keterangan:
  `time_score`       = max(0, 1000 - (total_waktu / player_cells) x 10)
  `error_penalty`    = min(350, (errors / player_cells) x 500)
  `behavior_penalty` = guessing x 25 + near_miss x 8
  `hint_penalty`     = hints_used x 200
  `auto_penalty`     = auto_used x 50
  `difficulty_mult`  = Easy: 1.0 / Normal: 1.8 / Hard: 3.0
```

---

## 📦 ***Output Files***

### `player_data.json`

Dibuat otomatis di *root* proyek setelah sesi pertama. Menyimpan seluruh data pemain menggunakan penulisan atomik (tulis ke `.tmp` lalu `os.replace`).

```json
{
  "players": {
    "samuel": {
      "created_at": 1748000000.0,
      "tutorial_done": true,
      "achievements": ["pemula_berhasil", "kilat", "tanpa_cela"],
      "sessions": [
        {
          "username": "samuel",
          "timestamp": 1748000100.0,
          "difficulty": "Normal",
          "grid_size": 3,
          "total_time": 187.4,
          "moves": 42,
          "errors": 1,
          "hints_used": 0,
          "auto_used": 0,
          "near_miss": 1,
          "guessing": 0,
          "completed": true,
          "score": 820,
          "empty_cells": 41,
          "time_per_cell": 4.46,
          "hearts_left": 4,
          "max_hearts": 9,
          "max_errors": 5,
          "lose_reason": null
        }
      ]
    }
  }
}
```

Field sesi utama: `timestamp`, `difficulty`, `grid_size`, `total_time`, `moves`, `errors`, `hints_used`, `auto_used`, `near_miss`, `guessing`, `completed`, `score`, `empty_cells`, `time_per_cell`, `hearts_left`, `max_hearts`, `max_errors`, `lose_reason`.

### Folder `Card/`

Gambar PNG kartu skor diekspor ke sini ketika pemain menekan **🖼 SIMPAN SCORECARD** di *Performance Dashboard*.

- Format nama file: `SudokuAI_{username}_{unix_timestamp}.png`
- Lebar: 860px (tinggi menyesuaikan konten)
- Dibuat seluruhnya dengan `Pillow` tanpa font eksternal atau template tambahan
- Konten kartu: nama pemain, statistik sesi, badge tipe pemain, progress bar kemampuan, jumlah pencapaian, dan pesan rekomendasi AI

---

## 🏅 ***Achievements***

Game melacak 20 *badge achievements* yang tersebar dalam 6 tier. Setiap *badge* disimpan berdasarkan ID-nya di `player_data.json` dan ditampilkan di *Performance Dashboard*.

<details>
<summary><strong>View all 20 achievements</strong></summary>

| Badge | ID | Kondisi Unlock |
|---|---|---|
| 🟢 `First Win` | `pemula_berhasil` | Selesaikan *puzzle* pertamamu |
| 🟠 `Marathon` | `maraton` | Selesaikan total 10 sesi |
| 🟣 `Veteran` | `veteran` | Selesaikan total 25 sesi |
| 🔴 `Consistent` | `konsisten` | Selesaikan 3 sesi berturut-turut |
| 🟣 `Unbeatable` | `tak_terkalahkan` | Selesaikan 5 sesi berturut-turut |
| 🟣 `Serial Winner` | `serial_winner` | Selesaikan 7 sesi berturut-turut |
| 🟠 `Comeback` | `comeback` | Selesaikan sesi setelah sesi sebelumnya gagal |
| 🟡 `Lightning` | `kilat` | Selesaikan 4x4 dalam waktu di bawah 60 detik |
| 🟡 `Speed Flash` | `cepat_kilat` | Selesaikan 4x4 dalam waktu di bawah 30 detik |
| 🔵 `Speed Demon` | `speed_demon` | Selesaikan 9x9 dalam waktu di bawah 5 menit |
| 🔵 `No Hints` | `tanpa_petunjuk` | Selesaikan tanpa menggunakan satu pun *hint* atau *auto* |
| 🟣 `Flawless` | `tanpa_cela` | Selesaikan tanpa satu pun *error* |
| 🟡 `Perfect` | `sempurna` | Selesaikan tanpa *error* dan tanpa *hint* |
| 🟢 `Efficient` | `efisien` | Selesaikan dengan kurang dari 3 *error* |
| 🔴 `Hard Expert` | `ahli_hard` | Selesaikan *puzzle* Hard |
| 🔴 `Iron Will` | `tanpa_menyerah_hard` | Selesaikan Hard tanpa satu pun *hint* atau *auto* |
| 🟣 `Master 9x9` | `master_9x9` | Selesaikan *puzzle* 9x9 |
| 🔵 `Explorer` | `explorer` | Mainkan ketiga tingkat kesulitan |
| 🟡 `Genius` | `jenius` | Raih skor di atas 800 dalam satu sesi |
| 🔴 `Expert` | `pakar` | Raih skor di atas 500 di mode Hard |

</details>

---

## 🛠️ ***Tech Stack***

<div align="center">

[![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![tkinter](https://img.shields.io/badge/tkinter-GUI%20Framework-4A90D9?style=flat-square)](https://docs.python.org/3/library/tkinter.html)
[![pygame](https://img.shields.io/badge/pygame-Audio%20%26%20SFX-1A1A2E?style=flat-square)](https://pygame.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML%20Models-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![NumPy](https://img.shields.io/badge/NumPy-Array%20%26%20SFX%20Engine-013243?style=flat-square&logo=numpy&logoColor=white)](https://numpy.org)
[![Pillow](https://img.shields.io/badge/Pillow-Image%20%26%20Score%20Card-11557C?style=flat-square)](https://pillow.readthedocs.io)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-Performance%20Charts-11557C?style=flat-square)](https://matplotlib.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-Easter%20Egg%20Video-5C3EE8?style=flat-square&logo=opencv&logoColor=white)](https://opencv.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-Score%20Prediction%20Benchmark-FF6600?style=flat-square)](https://xgboost.readthedocs.io)
[![LightGBM](https://img.shields.io/badge/LightGBM-Score%20Prediction%20Benchmark-02A388?style=flat-square)](https://lightgbm.readthedocs.io)
[![pandas](https://img.shields.io/badge/pandas-Data%20Analysis-150458?style=flat-square&logo=pandas&logoColor=white)](https://pandas.pydata.org)
[![seaborn](https://img.shields.io/badge/seaborn-Notebook%20Viz-4C72B0?style=flat-square)](https://seaborn.pydata.org)
[![scipy](https://img.shields.io/badge/scipy-Statistical%20Distributions-8CAAE6?style=flat-square)](https://scipy.org)

</div>

---

## 🥚 ***Easter Egg***

<details>
<summary><strong>Click to reveal the hidden feature</strong></summary>

Pada tampilan Login, **klik logo sebanyak 7 kali dalam waktu 4 detik**.

Jika berhasil dipicu:

1. `Assets/easter_egg.mp3` diputar menggunakan `pygame` sebagai audio.
2. Jika `opencv-python` dan `Pillow` keduanya terinstal, `Assets/easter_egg.mp4` dirender *frame-per-frame* langsung di dalam tampilan `tkinter` sebagai *overlay* video layar penuh.
3. Jika file video tidak ditemukan, animasi *fallback* ditampilkan sebagai gantinya.
4. Sembarang penekanan tombol atau klik *mouse* akan menutup *overlay*.

*Easter egg counter* direset jika jeda antar klik lebih dari 4 detik.

</details>

---

## 🎓 ***Academic Information***

<div align="center">

| Keterangan | Detail |
|---|---|
| **Nama Proyek** | SUDOKU AI |
| **Versi** | 1.0.0 |
| **Program Studi** | Sains Data |
| **Fakultas** | Fakultas Teknologi & Desain |
| **Universitas** | Universitas Bunda Mulia Kampus Serpong |
| **Mata Kuliah** | *Machine Learning for Intelligence System* |
| **Dosen Pengampu** | Puguh Hiskiawan., S.Si., M.Si., Ph.D. |
| **Tahun** | 2026 |

</div>

> Proyek ini diajukan sebagai karya Ujian Akhir Semester (UAS) untuk mata kuliah Machine Learning for Intelligence System. Proses pendaftaran Hak Kekayaan Intelektual (HAKI) sebagai Ciptaan Program Komputer sedang dalam proses.

---

<div align="center">

*Made with 🧠 and ♥ by **Samuel Lie***
Universitas Bunda Mulia · 2026

</div>
