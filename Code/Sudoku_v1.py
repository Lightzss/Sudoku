import tkinter as tk
from tkinter import messagebox
import random
import time
import copy
import json
import os

# ── MACHINE LEARNING IMPORTS ─────────────────────────────────────────────────
# Gunakan scikit-learn jika tersedia; fallback ke rule-based jika tidak.
try:
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    import numpy as np
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# =====================================================
# DATA STORAGE
# =====================================================
DATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sudoku_data.json")

def load_data():
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, "r") as f:
                return json.load(f)
        except:
            pass
    return {"players": {}}

def save_data(data):
    with open(DATA_FILE, "w") as f:
        json.dump(data, f, indent=2)

# =====================================================
# DESIGN TOKENS
# =====================================================
C_BG        = "#0D1117"
C_SURFACE   = "#161B22"
C_SURFACE2  = "#1F2937"
C_BORDER    = "#30363D"
C_ACCENT    = "#58A6FF"
C_ACCENT2   = "#7EE787"
C_WARN      = "#F0883E"
C_ERROR     = "#FF7B7B"
C_PURPLE    = "#BC8CFF"
C_PINK      = "#FF7EDB"
C_GOLD      = "#FFD700"
C_TEXT      = "#E6EDF3"
C_TEXT_DIM  = "#8B949E"
C_WHITE     = "#FFFFFF"

# Highlight cross-line colors (row/col/box)
C_CROSS_ROW  = "#1A2535"   # subtle blue tint for row
C_CROSS_COL  = "#1A2535"   # subtle blue tint for col
C_CROSS_BOX  = "#1B2B1B"   # subtle green tint for box
C_SELECTED   = "#2A4A7A"   # selected cell
C_SAME_NUM   = "#2A3A5A"   # same number highlight

DIFF_THEMES = {
    # ─── EASY: palet hijau toska — nuansa tenang & ramah ───────────────────
    "Easy": {
        "accent":         "#7EE787",   # hijau cerah utama
        "accent2":        "#2EA043",
        "cell_bg":        "#0D2818",   # latar sel kosong
        "cell_fixed_bg":  "#143321",   # latar angka sistem
        "cell_fixed_fg":  "#7EE787",   # teks angka sistem
        "cell_user_fg":   "#A8F0B0",   # teks input user
        "highlight":      "#1F6B3A",   # sel terpilih (hijau solid)
        "hover":          "#153825",
        "error_bg":       "#3B1212",   # latar input salah
        "error_fg":       "#FF7B7B",
        "grid_line":      "#2EA043",
        "remove_pct":     0.35,
        "emoji":          "🌱",
        # ── Highlight warna adaptif ──────────────────────────────────────
        # Palet: toska-hijau — kontras jelas di atas cell_bg gelap
        "hl_box":         "#0E3020",   # kotak yg sama (ringan)
        "hl_rowcol":      "#144A2A",   # baris/kolom (lebih terang)
        "hl_same_bg":     "#1B6B38",   # angka yg sama (paling mencolok)
        "hl_same_fg_fix": "#B8FFD0",   # fg fixed same-num
        "hl_same_fg_usr": "#D0FFE0",   # fg user same-num
        "hl_sel_border":  "#7EE787",   # border efek sel terpilih
    },
    # ─── NORMAL: palet biru safir — tajam & fokus ──────────────────────────
    "Normal": {
        "accent":         "#58A6FF",
        "accent2":        "#1F6FEB",
        "cell_bg":        "#0D1A2E",
        "cell_fixed_bg":  "#112244",
        "cell_fixed_fg":  "#79C0FF",
        "cell_user_fg":   "#B0D4FF",
        "highlight":      "#1C4880",   # biru solid
        "hover":          "#142D47",
        "error_bg":       "#3B1212",
        "error_fg":       "#FF7B7B",
        "grid_line":      "#1F6FEB",
        "remove_pct":     0.50,
        "emoji":          "⚡",
        # ── Highlight warna adaptif ──────────────────────────────────────
        # Palet: biru safir — kontras tinggi di latar biru gelap
        "hl_box":         "#0E2040",   # kotak yg sama
        "hl_rowcol":      "#152E5A",   # baris/kolom
        "hl_same_bg":     "#1A4A8C",   # angka yg sama
        "hl_same_fg_fix": "#A8D4FF",
        "hl_same_fg_usr": "#C8E4FF",
        "hl_sel_border":  "#58A6FF",
    },
    # ─── HARD: palet merah coral — intens & menegangkan ────────────────────
    "Hard": {
        "accent":         "#FF7B7B",
        "accent2":        "#DA3633",
        "cell_bg":        "#2D0D0D",
        "cell_fixed_bg":  "#3D1515",
        "cell_fixed_fg":  "#FF7B7B",
        "cell_user_fg":   "#FFAAAA",
        "highlight":      "#6B2020",   # merah solid terang
        "hover":          "#421515",
        "error_bg":       "#5A1010",
        "error_fg":       "#FF4444",
        "grid_line":      "#DA3633",
        "remove_pct":     0.65,
        "emoji":          "🔥",
        # ── Highlight warna adaptif ──────────────────────────────────────
        # Palet: merah-ungu — hangat, mencolok di atas latar merah gelap
        "hl_box":         "#3A1010",   # kotak yg sama (merah sangat gelap)
        "hl_rowcol":      "#501A1A",   # baris/kolom (merah tua)
        "hl_same_bg":     "#7A2020",   # angka yg sama (merah cerah)
        "hl_same_fg_fix": "#FFB8B8",
        "hl_same_fg_usr": "#FFD0D0",
        "hl_sel_border":  "#FF7B7B",
    },
}

FONT_BTN    = ("Segoe UI", 10, "bold")
FONT_BTN_SM = ("Segoe UI", 9, "bold")
FONT_SMALL  = ("Segoe UI", 9)
FONT_BODY   = ("Segoe UI", 11)
FONT_TIMER  = ("Consolas", 22, "bold")

# =====================================================
# SHARED UI HELPERS
# =====================================================
_GRADIENT_COLORS = ["#BC8CFF", "#58A6FF", "#7EE787", "#F0883E"]

def draw_gradient_bar(canvas, colors=None, height=8):
    """Draw a smooth multi-stop horizontal gradient onto a Canvas.
    Safe to call from .after() — queries real widget width at render time.
    """
    if colors is None:
        colors = _GRADIENT_COLORS
    canvas.update_idletasks()
    w = canvas.winfo_width()
    if w < 2:
        w = canvas.winfo_screenwidth() if hasattr(canvas, "winfo_screenwidth") else 1280
    n    = len(colors)
    seg  = max(1, w // max(n - 1, 1))
    steps = 20
    for i in range(n - 1):
        c1, c2 = colors[i], colors[i + 1]
        x1, x2 = i * seg, (i + 1) * seg
        for k in range(steps):
            t   = k / steps
            r_  = int(int(c1[1:3], 16) * (1 - t) + int(c2[1:3], 16) * t)
            g_  = int(int(c1[3:5], 16) * (1 - t) + int(c2[3:5], 16) * t)
            b_  = int(int(c1[5:7], 16) * (1 - t) + int(c2[5:7], 16) * t)
            xi  = int(x1 + k * (x2 - x1) / steps)
            xj  = int(x1 + (k + 1) * (x2 - x1) / steps)
            canvas.create_rectangle(xi, 0, xj, height,
                                    fill=f"#{r_:02x}{g_:02x}{b_:02x}",
                                    outline="")
    canvas.create_rectangle((n - 1) * seg, 0, w, height,
                             fill=colors[-1], outline="")

# =====================================================
# SUDOKU LOGIC
# =====================================================
def calculate_score(difficulty, total_time, empty_cells, errors,
                    hints_used, completed, near_miss=0, guessing=0):
    """
    Skor yang adil: hint TIDAK boleh meningkatkan skor.
    ─────────────────────────────────────────────────────────────
    Prinsip:
    • Skor didasarkan pada waktu per sel yang BENAR-BENAR diselesaikan
      oleh pemain sendiri (bukan yang diisi hint).
    • Setiap hint langsung memotong skor besar karena:
        - hint mengisi sel (mengurangi waktu sisa)
        - sekaligus dikenai penalti hints_penalty eksplisit
    • Error: dikenai penalti proporsional.
    • "Asal tebak" (guessing): penalti ekstra dibanding "hampir benar".
    ─────────────────────────────────────────────────────────────
    """
    if not completed:
        return 0

    diff_mult = {"Easy": 1.0, "Normal": 1.8, "Hard": 3.0}.get(difficulty, 1.0)
    N = max(empty_cells, 1)

    # Sel yang benar-benar dikerjakan pemain
    player_cells = max(1, N - hints_used)

    # Waktu per sel yang dikerjakan sendiri
    time_per_player_cell = total_time / player_cells

    # Base time score: makin cepat per sel → makin tinggi
    time_score = max(0.0, 1000.0 - time_per_player_cell * 10.0)

    # Penalti error (proporsional terhadap jumlah sel)
    error_rate    = errors / player_cells
    error_penalty = min(350, int(error_rate * 500))

    # Penalti jenis error: asal tebak lebih berat dari hampir benar
    behavior_penalty = guessing * 25 + near_miss * 8

    # Penalti hint — SANGAT besar agar hint tidak meng-inflate skor
    # Setiap hint = melewati 1 sel, dikenai 200 poin + proporsional
    hint_penalty = hints_used * 200

    raw   = time_score - error_penalty - behavior_penalty - hint_penalty
    final = max(0, int(raw * diff_mult))
    return final
    for r in range(N):
        for c in range(N):
            if board[r][c] == 0:
                return (r, c)
    return None

def find_empty(board, N):
    for i in range(N):
        for j in range(N):
            if board[i][j] == 0:
                return (i, j)
    return None

def is_valid(board, num, pos, N, BOX):
    r, c = pos
    for i in range(N):
        if board[r][i] == num and c != i: return False
    for i in range(N):
        if board[i][c] == num and r != i: return False
    bx, by = c // BOX, r // BOX
    for i in range(by*BOX, by*BOX+BOX):
        for j in range(bx*BOX, bx*BOX+BOX):
            if board[i][j] == num and (i, j) != pos: return False
    return True

def get_candidates(board, r, c, N, BOX):
    return [n for n in range(1, N+1) if is_valid(board, n, (r, c), N, BOX)]

def generate_full_board(N, BOX):
    board = [[0]*N for _ in range(N)]
    def fill(b):
        empty = find_empty(b, N)
        if not empty: return True
        r, c = empty
        nums = list(range(1, N+1))
        random.shuffle(nums)
        for n in nums:
            if is_valid(b, n, (r, c), N, BOX):
                b[r][c] = n
                if fill(b): return True
                b[r][c] = 0
        return False
    fill(board)
    return board

def generate_puzzle(N, BOX, remove_pct):
    board = generate_full_board(N, BOX)
    solution = copy.deepcopy(board)
    puzzle = copy.deepcopy(board)
    total = N * N
    remove_count = min(int(total * remove_pct), total - (N + 1))
    indices = [(r, c) for r in range(N) for c in range(N)]
    random.shuffle(indices)
    for i in range(remove_count):
        r, c = indices[i]
        puzzle[r][c] = 0
    return puzzle, solution

# =====================================================
# OPTIMAL BACKTRACKING: MRV + Forward Checking
# =====================================================
# Ini adalah versi paling optimal dari backtracking:
# - MRV (Minimum Remaining Values): pilih sel dengan
#   kandidat paling sedikit terlebih dahulu
# - Forward Checking: cek apakah sel lain masih punya
#   kandidat valid setelah penempatan angka
# - Early termination jika ada sel dengan 0 kandidat
# =====================================================
def solve_backtracking_mrv(start_board, N, BOX):
    """
    Backtracking dengan MRV (Minimum Remaining Values) heuristic.
    Memilih sel kosong dengan jumlah kandidat paling sedikit (paling constrained).
    Jauh lebih efisien daripada backtracking biasa karena:
    1. Sel dengan 1 kandidat langsung diisi (forced assignment)
    2. Deteksi kegagalan lebih awal (forward checking)
    3. Search tree jauh lebih kecil
    """
    expanded = [0]
    start = time.time()
    hist = []
    working = [row[:] for row in start_board]

    def find_mrv_cell(b):
        """Cari sel kosong dengan kandidat paling sedikit (MRV)"""
        best_pos = None
        best_count = N + 1
        for r in range(N):
            for c in range(N):
                if b[r][c] == 0:
                    cands = get_candidates(b, r, c, N, BOX)
                    if len(cands) == 0:
                        return None, []   # Dead end, backtrack
                    if len(cands) < best_count:
                        best_count = len(cands)
                        best_pos = (r, c)
                        best_cands = cands
                        if best_count == 1:
                            return best_pos, best_cands  # Forced
        return best_pos, best_cands if best_pos else []

    def bt(b):
        expanded[0] += 1
        pos, cands = find_mrv_cell(b)
        if pos is None:
            # Check if actually solved or dead end
            if find_empty(b, N) is None:
                return True   # Solved!
            return False      # Dead end
        r, c = pos
        for num in cands:
            b[r][c] = num
            hist.append((r, c, num))
            if bt(b):
                return True
            b[r][c] = 0
            hist.append((r, c, 0))
        return False

    if bt(working):
        return hist, expanded[0], time.time() - start
    return None, expanded[0], time.time() - start

# =====================================================
# MACHINE LEARNING MODULE  (v2 — sklearn powered)
# =====================================================
# Dataset sintetis untuk melatih KNN Classifier.
# Setiap baris: [avg_time_per_cell, error_rate, hint_rate,
#                completion_rate, near_miss_rate, guessing_rate]
# Label: indeks PLAYER_TYPES_ORDER
# ─────────────────────────────────────────────────────────────
_PLAYER_TYPES_ORDER = [
    "Speedrunner", "Careful", "Learner", "Struggling", "Inconsistent"
]
_KNN_SYNTHETIC_X = [
    # Speedrunner: cepat (tpc rendah), akurat, jarang hint, sering selesai
    [2.0, 0.02, 0.03, 1.0, 0.4, 0.1],
    [3.0, 0.03, 0.02, 0.95, 0.3, 0.1],
    [1.5, 0.01, 0.01, 1.0, 0.2, 0.05],
    [4.0, 0.04, 0.04, 0.90, 0.5, 0.2],
    # Careful: lambat, akurat, jarang hint, sering selesai
    [18.0, 0.04, 0.05, 1.0, 0.6, 0.1],
    [22.0, 0.03, 0.04, 0.95, 0.5, 0.05],
    [15.0, 0.05, 0.06, 0.90, 0.7, 0.15],
    [25.0, 0.02, 0.03, 1.0, 0.4, 0.1],
    # Learner: sedang, kadang error, sedang hint
    [8.0, 0.15, 0.20, 0.70, 0.5, 0.3],
    [10.0, 0.12, 0.18, 0.65, 0.6, 0.25],
    [7.0, 0.18, 0.22, 0.75, 0.55, 0.35],
    [9.0, 0.10, 0.15, 0.80, 0.45, 0.2],
    # Struggling: lambat, banyak error, banyak hint, jarang selesai
    [20.0, 0.40, 0.50, 0.30, 0.4, 0.8],
    [25.0, 0.45, 0.55, 0.25, 0.35, 0.85],
    [18.0, 0.35, 0.45, 0.35, 0.5, 0.7],
    [30.0, 0.50, 0.60, 0.20, 0.3, 0.9],
    # Inconsistent: variance tinggi (waktu moderat tapi tidak stabil)
    [8.0, 0.25, 0.20, 0.60, 0.6, 0.5],
    [6.0, 0.30, 0.15, 0.55, 0.7, 0.55],
    [12.0, 0.20, 0.25, 0.65, 0.5, 0.4],
    [5.0, 0.35, 0.30, 0.50, 0.65, 0.6],
]
_KNN_SYNTHETIC_Y = (
    [0]*4 +   # Speedrunner
    [1]*4 +   # Careful
    [2]*4 +   # Learner
    [3]*4 +   # Struggling
    [4]*4     # Inconsistent
)


class PlayerMLEngine:
    """
    ML Engine v2 — scikit-learn + Rule-Based Fallback
    ──────────────────────────────────────────────────
    Fitur ML nyata (aktif jika sklearn tersedia):
    • KNeighborsClassifier  — klasifikasi tipe pemain + confidence %
    • LinearRegression       — prediksi skor sesi berikutnya
    • IsolationForest        — deteksi sesi anomali (tidak biasa)
    • StandardScaler         — normalisasi fitur sebelum training

    Jika sklearn tidak tersedia, fallback ke rule-based scoring.
    """
    PLAYER_TYPES = {
        "Speedrunner":  {"color": "#FFD700", "emoji": "⚡", "desc": "Cepat & akurat"},
        "Careful":      {"color": "#7EE787", "emoji": "🧩", "desc": "Hati-hati & teliti"},
        "Learner":      {"color": "#58A6FF", "emoji": "📚", "desc": "Sedang belajar"},
        "Struggling":   {"color": "#FF7B7B", "emoji": "💪", "desc": "Butuh bantuan"},
        "Inconsistent": {"color": "#F0883E", "emoji": "🎲", "desc": "Tidak konsisten"},
    }

    def __init__(self):
        self.sessions       = []
        self.adaptive_idle  = 15.0
        # sklearn model instances (None sampai di-train)
        self._knn           = None
        self._knn_scaler    = None
        self._lr            = None
        self._iso           = None
        self._iso_scaler    = None
        self._models_dirty  = True   # flag: perlu re-train saat ada sesi baru
        # Pre-train KNN pada data sintetis agar siap pakai sejak sesi pertama
        if SKLEARN_AVAILABLE:
            self._pretrain_knn()

    # ── Pre-training KNN dengan data sintetis ────────────────────────
    def _pretrain_knn(self):
        """Train KNN di data sintetis saja (dijalankan saat init)."""
        try:
            X = np.array(_KNN_SYNTHETIC_X, dtype=float)
            y = np.array(_KNN_SYNTHETIC_Y, dtype=int)
            scaler = StandardScaler()
            X_sc   = scaler.fit_transform(X)
            knn    = KNeighborsClassifier(n_neighbors=3, metric="euclidean")
            knn.fit(X_sc, y)
            self._knn        = knn
            self._knn_scaler = scaler
        except Exception:
            pass

    # ── Update semua model saat ada data sesi aktual ─────────────────
    def _train_models(self):
        """
        Re-train semua model ML menggunakan gabungan data sintetis
        + sesi aktual pemain. Dipanggil lazy saat dibutuhkan.
        """
        if not SKLEARN_AVAILABLE or not self._models_dirty:
            return
        self._models_dirty = False
        n = len(self.sessions)

        # ── 1. KNN: gabung sintetis + aktual ─────────────────────────
        try:
            X_syn = np.array(_KNN_SYNTHETIC_X, dtype=float)
            y_syn = np.array(_KNN_SYNTHETIC_Y, dtype=int)

            if n >= 1:
                feats_actual = [self._session_to_vector(s) for s in self.sessions]
                X_act = np.array(feats_actual, dtype=float)
                # Label aktual dari rule-based classifier untuk bootstrapping
                y_act = np.array(
                    [_PLAYER_TYPES_ORDER.index(self._rule_based_type(s))
                     for s in self.sessions], dtype=int)
                X_all = np.vstack([X_syn, X_act])
                y_all = np.concatenate([y_syn, y_act])
            else:
                X_all, y_all = X_syn, y_syn

            scaler = StandardScaler()
            X_sc   = scaler.fit_transform(X_all)
            k      = min(5, len(X_all))
            knn    = KNeighborsClassifier(n_neighbors=k, metric="euclidean")
            knn.fit(X_sc, y_all)
            self._knn        = knn
            self._knn_scaler = scaler
        except Exception:
            pass

        # ── 2. Linear Regression: prediksi skor sesi berikutnya ──────
        if n >= 3:
            try:
                # Feature: [session_index, tpc, error_rate, hint_rate]
                X_lr, y_lr = [], []
                for i, s in enumerate(self.sessions):
                    mv  = max(s.get("moves", 1), 1)
                    tpc = s.get("total_time", 0) / mv
                    er  = s.get("errors", 0) / mv
                    hr  = s.get("hints_used", 0) / mv
                    sc  = s.get("score", 0) or 0
                    X_lr.append([i, tpc, er, hr])
                    y_lr.append(sc)
                X_lr = np.array(X_lr, dtype=float)
                y_lr = np.array(y_lr, dtype=float)
                lr   = LinearRegression()
                lr.fit(X_lr, y_lr)
                self._lr = lr
            except Exception:
                self._lr = None

        # ── 3. Isolation Forest: deteksi sesi anomali ────────────────
        if n >= 5:
            try:
                X_iso = np.array(
                    [self._session_to_vector(s) for s in self.sessions],
                    dtype=float)
                scaler_iso = StandardScaler()
                X_iso_sc   = scaler_iso.fit_transform(X_iso)
                iso        = IsolationForest(
                    contamination=0.15, random_state=42, n_estimators=50)
                iso.fit(X_iso_sc)
                self._iso        = iso
                self._iso_scaler = scaler_iso
            except Exception:
                self._iso        = None
                self._iso_scaler = None

    def _session_to_vector(self, s):
        """Konversi 1 sesi dict → feature vector [6 nilai]."""
        mv   = max(s.get("moves", 1), 1)
        tpc  = s.get("total_time", 0) / mv
        er   = s.get("errors", 0)    / mv
        hr   = s.get("hints_used", 0) / mv
        cr   = 1.0 if s.get("completed", False) else 0.0
        total_err = max(s.get("errors", 0), 1)
        nmr  = s.get("near_miss", 0)  / total_err
        gur  = s.get("guessing", 0)   / total_err
        return [tpc, er, hr, cr, nmr, gur]

    def _rule_based_type(self, s):
        """Klasifikasi tipe untuk 1 sesi — dipakai sebagai label bootstrap."""
        mv  = max(s.get("moves", 1), 1)
        tpc = s.get("total_time", 0) / mv
        er  = s.get("errors", 0)     / mv
        hr  = s.get("hints_used", 0) / mv
        cr  = 1.0 if s.get("completed", False) else 0.0
        if tpc <= 4 and er < 0.05: return "Speedrunner"
        if tpc >= 14 and er < 0.10: return "Careful"
        if er > 0.30 or hr > 0.35: return "Struggling"
        if er > 0.15 or hr > 0.18: return "Learner"
        return "Inconsistent"

    def add_session(self, s):
        self.sessions.append(s)
        self._models_dirty = True
        self._update_thresholds()

    def _update_thresholds(self):
        if len(self.sessions) >= 2:
            avg = sum(s.get("total_time", 60) for s in self.sessions) / len(self.sessions)
            self.adaptive_idle = max(10.0, min(30.0, avg * 0.20))

    def extract_features(self):
        if not self.sessions:
            return {"avg_time_per_cell": 0, "error_rate": 0,
                    "hint_rate": 0, "completion_rate": 0,
                    "avg_moves": 0, "sessions_count": 0,
                    "near_miss_rate": 0, "guessing_rate": 0,
                    "avg_time_per_empty_cell": 0}
        n = len(self.sessions)
        tpc  = sum(s.get("total_time", 0) / max(s.get("moves", 1), 1) for s in self.sessions) / n
        er   = sum(s.get("errors", 0) / max(s.get("moves", 1), 1) for s in self.sessions) / n
        hr   = sum(s.get("hints_used", 0) / max(s.get("moves", 1), 1) for s in self.sessions) / n
        cr   = sum(1 for s in self.sessions if s.get("completed", False)) / n
        am   = sum(s.get("moves", 0) for s in self.sessions) / n
        # New: near-miss & guessing rates (per error move)
        total_err = sum(s.get("errors", 0) for s in self.sessions) or 1
        nmr  = sum(s.get("near_miss", 0) for s in self.sessions) / total_err
        gur  = sum(s.get("guessing", 0) for s in self.sessions) / total_err
        # Time per empty cell (direct performance metric)
        tpec = sum(s.get("time_per_cell", s.get("total_time",0)/max(s.get("moves",1),1))
                   for s in self.sessions) / n
        return {"avg_time_per_cell": tpc, "error_rate": er,
                "hint_rate": hr, "completion_rate": cr,
                "avg_moves": am, "sessions_count": n,
                "near_miss_rate": nmr, "guessing_rate": gur,
                "avg_time_per_empty_cell": tpec}

    def classify_player(self):
        """
        Klasifikasi tipe pemain.
        - Jika sklearn tersedia → KNN Classifier (trained on synthetic + actual data)
        - Fallback → Rule-based weighted scoring
        Returns: (player_type_str, feat_dict)
        """
        feat = self.extract_features()
        if feat["sessions_count"] == 0:
            return "Learner", feat

        if SKLEARN_AVAILABLE and self._knn is not None:
            try:
                self._train_models()
                fv  = np.array([[
                    feat["avg_time_per_cell"], feat["error_rate"],
                    feat["hint_rate"],          feat["completion_rate"],
                    feat["near_miss_rate"],      feat["guessing_rate"],
                ]], dtype=float)
                fv_sc = self._knn_scaler.transform(fv)
                idx   = int(self._knn.predict(fv_sc)[0])
                pt    = _PLAYER_TYPES_ORDER[idx]
                return pt, feat
            except Exception:
                pass   # fallback di bawah

        # ── Rule-based fallback ───────────────────────────────────────
        tpc, er, hr, cr = (feat["avg_time_per_cell"], feat["error_rate"],
                           feat["hint_rate"], feat["completion_rate"])
        scores = {
            "Speedrunner":  max(0,100-tpc*10)*0.4 + max(0,100-er*200)*0.4 + cr*100*0.2,
            "Careful":      min(100,tpc*5)*0.3    + max(0,100-er*150)*0.4 + cr*100*0.3,
            "Learner":      max(0,60-abs(tpc-8)*5)*0.3 + max(0,70-er*100)*0.4 + (1-hr)*100*0.3,
            "Struggling":   min(100,er*300)*0.3   + min(100,hr*400)*0.4   + max(0,100-cr*100)*0.3,
            "Inconsistent": 20,
        }
        if len(self.sessions) >= 3:
            times = [s.get("total_time", 0) for s in self.sessions]
            mt = sum(times) / len(times)
            cv = (sum((t-mt)**2 for t in times) / len(times))**0.5 / (mt+1)
            scores["Inconsistent"] = min(100, cv*150)
        return max(scores, key=scores.get), feat

    def classify_player_confidence(self):
        """
        Kembalikan (player_type, confidence_pct, feat).
        Confidence = probabilitas kelas teratas dari KNN (0–100).
        Jika sklearn tidak tersedia → confidence = 0 (tidak diketahui).
        """
        pt, feat = self.classify_player()
        conf = 0.0
        if SKLEARN_AVAILABLE and self._knn is not None:
            try:
                self._train_models()
                fv = np.array([[
                    feat["avg_time_per_cell"], feat["error_rate"],
                    feat["hint_rate"],          feat["completion_rate"],
                    feat["near_miss_rate"],      feat["guessing_rate"],
                ]], dtype=float)
                fv_sc = self._knn_scaler.transform(fv)
                proba = self._knn.predict_proba(fv_sc)[0]
                conf  = float(max(proba)) * 100
            except Exception:
                pass
        return pt, conf, feat

    def predict_next_score(self):
        """
        Prediksi skor sesi berikutnya menggunakan Linear Regression.
        Return: (predicted_score: int, available: bool)
        Butuh minimal 3 sesi untuk aktif.
        """
        if not SKLEARN_AVAILABLE or len(self.sessions) < 3:
            return None, False
        try:
            self._train_models()
            if self._lr is None:
                return None, False
            n   = len(self.sessions)
            # Feature sesi berikutnya: estimasi dari rata-rata 3 sesi terakhir
            recent = self.sessions[-3:]
            mv_avg  = sum(max(s.get("moves",1),1) for s in recent) / 3
            tpc_avg = sum(s.get("total_time",0)/max(s.get("moves",1),1) for s in recent) / 3
            er_avg  = sum(s.get("errors",0)/max(s.get("moves",1),1) for s in recent) / 3
            hr_avg  = sum(s.get("hints_used",0)/max(s.get("moves",1),1) for s in recent) / 3
            # Asumsi perbaikan 5% pada tpc dan er (learning curve)
            X_next = np.array([[n, tpc_avg*0.95, er_avg*0.95, hr_avg*0.95]])
            pred   = max(0, int(self._lr.predict(X_next)[0]))
            return pred, True
        except Exception:
            return None, False

    def detect_anomaly(self):
        """
        Deteksi apakah sesi TERAKHIR adalah anomali (tidak biasa).
        Menggunakan Isolation Forest — aktif setelah 5 sesi.
        Return: ("normal"|"anomaly"|"unknown", reason_str)
        """
        if not SKLEARN_AVAILABLE or len(self.sessions) < 5:
            return "unknown", "Data belum cukup (min 5 sesi)"
        try:
            self._train_models()
            if self._iso is None or self._iso_scaler is None:
                return "unknown", "Model belum siap"
            last_vec = np.array([self._session_to_vector(self.sessions[-1])],
                                dtype=float)
            last_sc  = self._iso_scaler.transform(last_vec)
            result   = self._iso.predict(last_sc)[0]
            score    = self._iso.decision_function(last_sc)[0]
            if result == -1:
                if score < -0.2:
                    reason = "Sesi ini sangat berbeda dari biasanya"
                else:
                    reason = "Performa sedikit di luar pola normal"
                return "anomaly", reason
            else:
                return "normal", "Sesi ini konsisten dengan pola kamu"
        except Exception:
            return "unknown", "Tidak dapat dianalisis"

    def recommend_difficulty(self):
        feat = self.extract_features()
        if feat["sessions_count"] == 0:
            return "Easy"
        skill = (max(0,100-feat["avg_time_per_cell"]*8)*0.35 +
                 max(0,100-feat["error_rate"]*250)*0.35 +
                 feat["completion_rate"]*100*0.30)
        last_done = self.sessions[-1].get("completed", False) if self.sessions else False
        if skill >= 70 and last_done: return "Hard"
        if skill >= 40: return "Normal"
        return "Easy"

    def should_give_hint(self, idle, errors, moves):
        if idle > self.adaptive_idle: return True, "idle"
        if moves > 5 and errors / max(moves, 1) > 0.4: return True, "errors"
        return False, None

    def get_summary(self):
        pt, conf, feat = self.classify_player_confidence()
        pred_score, pred_avail = self.predict_next_score()
        anom_status, anom_reason = self.detect_anomaly()
        return {
            "player_type":             pt,
            "features":                feat,
            "recommended_difficulty":  self.recommend_difficulty(),
            "type_info":               self.PLAYER_TYPES.get(pt, {}),
            # ── ML tambahan ──────────────────────────────────────────
            "ml_confidence":           conf,           # 0–100
            "predicted_next_score":    pred_score,     # int | None
            "predicted_score_avail":   pred_avail,     # bool
            "anomaly_status":          anom_status,    # "normal"|"anomaly"|"unknown"
            "anomaly_reason":          anom_reason,    # str
            "sklearn_active":          SKLEARN_AVAILABLE,
        }

# =====================================================
# ANIMATED BACKGROUND CANVAS
# =====================================================
class AnimatedBG(tk.Canvas):
    def __init__(self, master, **kw):
        super().__init__(master, **kw)
        self._pts = []
        self.after(50, self._init)

    def _init(self):
        sw = self.winfo_screenwidth()
        sh = self.winfo_screenheight()
        cols = ["#1A2840","#1A3028","#2A1A30","#1C2010","#201C10"]
        for _ in range(35):
            self._pts.append({
                "x": random.uniform(0, sw), "y": random.uniform(0, sh),
                "r": random.uniform(1.5, 4.5),
                "vx": random.uniform(-0.3, 0.3), "vy": random.uniform(-0.3, 0.3),
                "c": random.choice(cols)
            })
        self._run()

    def _run(self):
        sw = self.winfo_screenwidth()
        sh = self.winfo_screenheight()
        self.delete("pt")
        for p in self._pts:
            p["x"] = (p["x"] + p["vx"]) % sw
            p["y"] = (p["y"] + p["vy"]) % sh
            self.create_oval(p["x"]-p["r"], p["y"]-p["r"],
                             p["x"]+p["r"], p["y"]+p["r"],
                             fill=p["c"], outline="", tags="pt")
        self.after(50, self._run)

# =====================================================
# TOOLTIP
# =====================================================
class Tooltip:
    def __init__(self, w, text):
        self.w, self.text, self.tip = w, text, None
        w.bind("<Enter>", self.show)
        w.bind("<Leave>", self.hide)

    def show(self, _=None):
        if self.tip: return
        x = self.w.winfo_rootx() + 20
        y = self.w.winfo_rooty() + self.w.winfo_height() + 4
        self.tip = tk.Toplevel(self.w)
        self.tip.wm_overrideredirect(True)
        self.tip.wm_geometry(f"+{x}+{y}")
        tk.Label(self.tip, text=self.text, bg="#21262D", fg=C_TEXT,
                 font=FONT_SMALL, padx=8, pady=4,
                 highlightbackground=C_BORDER, highlightthickness=1).pack()

    def hide(self, _=None):
        if self.tip:
            self.tip.destroy()
            self.tip = None

# =====================================================
# SCREEN: LOGIN  (fixed — no clipped text, recent users)
# =====================================================
class LoginScreen(tk.Frame):
    def __init__(self, master, on_login):
        super().__init__(master, bg=C_BG)
        self.on_login = on_login
        self.data = load_data()
        self._build()

    def _build(self):
        self.place(relx=0, rely=0, relwidth=1, relheight=1)

        bg = AnimatedBG(self, bg=C_BG, highlightthickness=0)
        bg.place(relx=0, rely=0, relwidth=1, relheight=1)

        # Center card — fixed height, enough for all content
        card = tk.Frame(self, bg=C_SURFACE,
                        highlightbackground=C_BORDER, highlightthickness=1)
        card.place(relx=0.5, rely=0.5, anchor="center", width=460, height=570)

        # Top gradient bar
        gbar = tk.Canvas(card, height=8, bg=C_SURFACE, highlightthickness=0)
        gbar.pack(fill="x")
        gbar.after(120, lambda: self._gradient(gbar))

        # Branding
        tk.Label(card, text="⬛", font=("Segoe UI", 44),
                 bg=C_SURFACE, fg=C_ACCENT).pack(pady=(22, 0))
        tk.Label(card, text="SUDOKU AI",
                 font=("Segoe UI", 26, "bold"), bg=C_SURFACE, fg=C_TEXT).pack()
        tk.Label(card, text="Machine Learning Intelligence System",
                 font=("Segoe UI", 10), bg=C_SURFACE, fg=C_TEXT_DIM).pack(pady=(2, 0))

        tk.Frame(card, height=1, bg=C_BORDER).pack(fill="x", padx=30, pady=16)

        # Username input
        inp = tk.Frame(card, bg=C_SURFACE)
        inp.pack(padx=40, fill="x")
        tk.Label(inp, text="USERNAME", font=("Segoe UI", 9, "bold"),
                 bg=C_SURFACE, fg=C_TEXT_DIM).pack(anchor="w")

        self.username_var = tk.StringVar()
        ebox = tk.Frame(inp, bg=C_BORDER, pady=1, padx=1)
        ebox.pack(fill="x", pady=(4, 0))
        self.entry = tk.Entry(ebox, textvariable=self.username_var,
                              font=("Segoe UI", 13), bg=C_SURFACE2, fg=C_TEXT,
                              insertbackground=C_ACCENT, relief="flat", bd=8)
        self.entry.pack(fill="x")
        self.entry.bind("<Return>", lambda _: self._login())
        self.entry.bind("<FocusIn>",  lambda _: ebox.config(bg=C_ACCENT))
        self.entry.bind("<FocusOut>", lambda _: ebox.config(bg=C_BORDER))

        self.err_lbl = tk.Label(inp, text="", font=FONT_SMALL,
                                bg=C_SURFACE, fg=C_ERROR)
        self.err_lbl.pack(anchor="w", pady=(4, 0))

        # START PLAYING button
        tk.Button(card, text="START PLAYING  →",
                  font=("Segoe UI", 12, "bold"),
                  bg=C_ACCENT, fg=C_BG,
                  activebackground="#79BFFF", activeforeground=C_BG,
                  relief="flat", cursor="hand2", pady=10,
                  command=self._login).pack(padx=40, fill="x", pady=(8, 0))

        # Divider
        div = tk.Frame(card, bg=C_SURFACE)
        div.pack(padx=40, fill="x", pady=(12, 6))
        tk.Frame(div, height=1, bg=C_BORDER).pack(side="left", fill="x", expand=True)
        tk.Label(div, text=" atau ", font=FONT_SMALL, bg=C_SURFACE, fg=C_TEXT_DIM).pack(side="left")
        tk.Frame(div, height=1, bg=C_BORDER).pack(side="right", fill="x", expand=True)

        # Leaderboard + player count row
        bot = tk.Frame(card, bg=C_SURFACE)
        bot.pack(padx=40, fill="x")
        tk.Button(bot, text="🏆  Lihat Leaderboard",
                  font=FONT_BTN, bg=C_SURFACE, fg=C_GOLD,
                  activebackground=C_SURFACE2, activeforeground=C_GOLD,
                  relief="flat", cursor="hand2", pady=6,
                  command=lambda: LeaderboardWindow(self.master, load_data())
                  ).pack(fill="x")

        n_p = len(self.data.get("players", {}))
        tk.Label(card,
                 text=f"👤 {n_p} pemain terdaftar",
                 font=FONT_SMALL, bg=C_SURFACE, fg=C_TEXT_DIM).pack(pady=(6, 0))

        # Recent players (quick re-login)
        players = list(self.data.get("players", {}).keys())
        if players:
            tk.Frame(card, height=1, bg=C_BORDER).pack(fill="x", padx=30, pady=(10, 6))
            tk.Label(card, text="LOGIN CEPAT",
                     font=("Segoe UI", 8, "bold"), bg=C_SURFACE, fg=C_TEXT_DIM).pack()
            recent_row = tk.Frame(card, bg=C_SURFACE)
            recent_row.pack(pady=(4, 8))
            for p in players[-5:]:   # max 5 terbaru
                btn = tk.Button(recent_row, text=p,
                                font=("Segoe UI", 9),
                                bg=C_SURFACE2, fg=C_ACCENT,
                                activebackground=C_BORDER, activeforeground=C_ACCENT,
                                relief="flat", cursor="hand2", padx=8, pady=4,
                                command=lambda u=p: self._quick_login(u))
                btn.pack(side="left", padx=3)

        self.entry.focus_set()

    def _gradient(self, c):
        c.update_idletasks()
        w = c.winfo_width()
        segs = [(0, C_ACCENT), (w//3, C_PURPLE), (2*w//3, C_PINK)]
        for i, (x, col) in enumerate(segs):
            x2 = segs[i+1][0] if i+1 < len(segs) else w
            c.create_rectangle(x, 0, x2+2, 8, fill=col, outline="")

    def _login(self):
        u = self.username_var.get().strip()
        if not u:
            return self._err("Username tidak boleh kosong!")
        if len(u) < 3:
            return self._err("Username minimal 3 karakter!")
        if len(u) > 20:
            return self._err("Username maksimal 20 karakter!")
        data = load_data()
        is_new = u not in data["players"]
        if is_new:
            data["players"][u] = {"sessions": [], "created_at": time.time()}
            save_data(data)
        greeting = "Selamat datang" if is_new else "Halo kembali"
        self.on_login(u, is_new, greeting)

    def _quick_login(self, u):
        greeting = "Halo kembali"
        self.on_login(u, False, greeting)

    def _err(self, msg):
        self.err_lbl.config(text=f"⚠ {msg}")
        self.after(2500, lambda: self.err_lbl.config(text=""))

# =====================================================
# SCREEN: GRID SIZE SELECT  (Redesigned v3 — bulletproof layout)
# =====================================================
class GridSizeScreen(tk.Frame):
    # Sample puzzle cells untuk preview mini-grid
    # (row, col, value, is_fixed)
    _SAMPLE_4 = [
        (0,0,3,True),(0,1,1,True),(0,3,2,True),
        (1,2,4,True),(1,3,3,True),
        (2,0,4,True),(2,1,2,True),
        (3,0,1,True),(3,2,3,True),(3,3,4,True),
    ]
    _SAMPLE_9 = [
        (0,0,5,True),(0,1,3,True),(0,4,7,True),
        (1,0,6,True),(1,3,1,True),(1,4,9,True),(1,5,5,True),
        (2,1,9,True),(2,2,8,True),(2,7,6,True),
        (3,0,8,True),(3,4,6,True),(3,8,3,True),
        (4,0,4,True),(4,3,8,True),(4,5,3,True),(4,8,1,True),
        (5,0,7,True),(5,4,2,True),(5,8,6,True),
        (6,1,6,True),(6,6,2,True),(6,7,8,True),
        (7,3,4,True),(7,4,1,True),(7,5,9,True),(7,8,5,True),
        (8,4,8,True),(8,7,7,True),(8,8,9,True),
    ]

    def __init__(self, master, username, greeting, on_select):
        super().__init__(master, bg=C_BG)
        self.username  = username
        self.greeting  = greeting
        self.on_select = on_select
        self._build()

    # ── Gradient bar ─────────────────────────────────────────────
    def _draw_gradient(self, canvas, colors):
        draw_gradient_bar(canvas, colors)  # delegate to shared helper

    # ── Mini sudoku grid preview ──────────────────────────────────
    def _draw_mini_grid(self, canvas, box, cells, color, size):
        N       = box * box
        pad     = 6
        avail   = size - pad * 2
        cell_sz = avail / N

        # Alternating block backgrounds
        for br in range(box):
            for bc in range(box):
                x1 = pad + bc * box * cell_sz
                y1 = pad + br * box * cell_sz
                x2 = x1 + box * cell_sz
                y2 = y1 + box * cell_sz
                blk_bg = "#0F1F35" if (br + bc) % 2 == 0 else "#0A1525"
                canvas.create_rectangle(x1, y1, x2, y2, fill=blk_bg, outline="")

        # Thin cell grid lines
        for i in range(1, N):
            xi = pad + i * cell_sz
            yi = pad + i * cell_sz
            is_block = (i % box == 0)
            col = color if is_block else "#1E2D40"
            w_  = 2   if is_block else 1
            canvas.create_line(pad, yi, pad + avail, yi, fill=col, width=w_)
            canvas.create_line(xi, pad, xi, pad + avail, fill=col, width=w_)

        # Outer border glow (2-pass: wide dim + narrow bright)
        # Tkinter hanya support hex 6-digit (#RRGGBB), bukan 8-digit RGBA.
        # Buat versi "redup" dari accent color dengan blend ke background gelap.
        try:
            r_c = int(color[1:3], 16)
            g_c = int(color[3:5], 16)
            b_c = int(color[5:7], 16)
            # Blend 25% accent + 75% near-black untuk efek glow redup
            dim = "#{:02x}{:02x}{:02x}".format(
                int(r_c * 0.25 + 8 * 0.75),
                int(g_c * 0.25 + 8 * 0.75),
                int(b_c * 0.25 + 8 * 0.75),
            )
        except Exception:
            dim = "#1A2030"
        canvas.create_rectangle(pad - 2, pad - 2,
                                 pad + avail + 2, pad + avail + 2,
                                 outline=dim, width=3)
        canvas.create_rectangle(pad, pad, pad + avail, pad + avail,
                                 outline=color, width=2)

        # Numbers
        font_sz = max(7, int(cell_sz * 0.55))
        for (r, c, val, _) in cells:
            if r >= N or c >= N: continue
            cx = pad + c * cell_sz + cell_sz / 2
            cy = pad + r * cell_sz + cell_sz / 2
            # Filled-cell highlight
            hs = cell_sz * 0.82
            canvas.create_rectangle(
                cx - hs/2, cy - hs/2, cx + hs/2, cy + hs/2,
                fill="#132030", outline="")
            canvas.create_text(cx, cy, text=str(val),
                                fill=color,
                                font=("Segoe UI", font_sz, "bold"),
                                anchor="center")

    # ── Main layout ───────────────────────────────────────────────
    def _build(self):
        self.place(relx=0, rely=0, relwidth=1, relheight=1)

        # Animated particle background
        abg = AnimatedBG(self, bg=C_BG, highlightthickness=0)
        abg.place(relx=0, rely=0, relwidth=1, relheight=1)

        # ── Top header ───────────────────────────────────────────
        hdr = tk.Frame(self, bg=C_SURFACE)
        hdr.pack(fill="x")

        # Rainbow gradient accent stripe
        gbar = tk.Canvas(hdr, height=8, bg=C_SURFACE, highlightthickness=0)
        gbar.pack(fill="x")
        gbar.after(100, lambda: self._draw_gradient(
            gbar, ["#BC8CFF", "#58A6FF", "#7EE787", "#F0883E"]))

        hdr_inner = tk.Frame(hdr, bg=C_SURFACE)
        hdr_inner.pack(pady=20)

        greet_row = tk.Frame(hdr_inner, bg=C_SURFACE)
        greet_row.pack()
        tk.Label(greet_row, text="👤 ",
                 font=("Segoe UI", 20), bg=C_SURFACE, fg=C_TEXT_DIM).pack(side="left")
        tk.Label(greet_row, text=f"{self.greeting}, ",
                 font=("Segoe UI", 24), bg=C_SURFACE, fg=C_TEXT_DIM).pack(side="left")
        tk.Label(greet_row, text=f"{self.username}!",
                 font=("Segoe UI", 24, "bold"), bg=C_SURFACE, fg=C_TEXT).pack(side="left")

        tk.Label(hdr_inner,
                 text="Pilih ukuran grid untuk memulai petualangan Sudoku",
                 font=("Segoe UI", 11), bg=C_SURFACE, fg=C_TEXT_DIM).pack(pady=(5, 0))

        # ── Center — cards sit here ───────────────────────────────
        # Use place so cards are vertically centered relative to remaining space
        # We rely on a pack+expand wrapper instead of absolute coords
        center_wrap = tk.Frame(self, bg=C_BG)
        center_wrap.pack(side="top", fill="both", expand=True)

        # Inner frame — grid manager for the two cards
        inner = tk.Frame(center_wrap, bg=C_BG)
        inner.place(relx=0.5, rely=0.5, anchor="center")

        # Section label above cards
        sec_row = tk.Frame(inner, bg=C_BG)
        sec_row.grid(row=0, column=0, columnspan=2, pady=(0, 24))
        tk.Frame(sec_row, width=55, height=1, bg=C_BORDER).pack(side="left", padx=(0,12))
        tk.Label(sec_row, text="PILIH UKURAN GRID",
                 font=("Segoe UI", 10, "bold"), bg=C_BG, fg=C_TEXT_DIM).pack(side="left")
        tk.Frame(sec_row, width=55, height=1, bg=C_BORDER).pack(side="left", padx=(12,0))

        # Two cards side-by-side
        grids = [
            {
                "box": 2, "full": "4 × 4",
                "color": "#BC8CFF", "tag_bg": "#1E0E36",
                "tag": "PEMULA", "cta": "MULAI 4 × 4",
                "cells": self._SAMPLE_4,
                "features": [
                    ("🎯", "Angka 1 – 4"),
                    ("⚡", "Selesai dalam menit"),
                    ("🎓", "Sempurna untuk pemula"),
                    ("😊", "Aturan mudah dipahami"),
                ],
            },
            {
                "box": 3, "full": "9 × 9",
                "color": "#58A6FF", "tag_bg": "#071830",
                "tag": "KLASIK", "cta": "MULAI 9 × 9",
                "cells": self._SAMPLE_9,
                "features": [
                    ("🎯", "Angka 1 – 9"),
                    ("🧠", "Butuh logika & strategi"),
                    ("🏆", "Format Sudoku resmi"),
                    ("🔥", "Tantangan sesungguhnya"),
                ],
            },
        ]
        for col_idx, g in enumerate(grids):
            self._card(inner, g, col_idx)

        # ── Footer ───────────────────────────────────────────────
        foot = tk.Frame(self, bg=C_SURFACE, pady=11)
        foot.pack(fill="x", side="bottom")
        tk.Label(foot,
                 text="💡  Tidak yakin? Mulai dari 4×4 untuk memahami aturan dasar Sudoku",
                 font=("Segoe UI", 9), bg=C_SURFACE, fg=C_TEXT_DIM).pack()

    # ── Individual card ───────────────────────────────────────────
    def _card(self, parent, g, col_idx):
        color   = g["color"]
        box     = g["box"]
        GRID_SZ = 164 if box == 3 else 136

        # ── Outer padding wrapper ─────────────────────────────────
        outer = tk.Frame(parent, bg=C_BG)
        outer.grid(row=1, column=col_idx, padx=20, sticky="n")

        # ── Card shell — NO pack_propagate(False) ─────────────────
        # Width is enforced by a hidden spacer Frame inside.
        card = tk.Frame(outer, bg=C_SURFACE,
                        highlightbackground=C_BORDER,
                        highlightthickness=1,
                        cursor="hand2")
        card.pack()

        CARD_W = 290
        # Invisible width-enforcer (must be the first child)
        spacer = tk.Frame(card, bg=C_SURFACE, height=1, width=CARD_W)
        spacer.pack(side="top")       # pack first so it sets horizontal extent
        spacer.lower()                # visually behind everything

        # ── Tag badge ────────────────────────────────────────────
        tag_bar = tk.Frame(card, bg=C_SURFACE)
        tag_bar.pack(fill="x", padx=18, pady=(16, 0))
        tk.Label(tag_bar, text=f"  {g['tag']}  ",
                 font=("Segoe UI", 8, "bold"),
                 bg=g["tag_bg"], fg=color,
                 padx=6, pady=3).pack(side="left")

        # ── Mini grid preview ─────────────────────────────────────
        preview_wrap = tk.Frame(card, bg=C_SURFACE)
        preview_wrap.pack(pady=(14, 0))

        # Coloured border ring (2px padding each side)
        ring = tk.Frame(preview_wrap, bg=color, padx=2, pady=2)
        ring.pack()
        dark_bg = tk.Frame(ring, bg="#080F1A")
        dark_bg.pack()

        cv = tk.Canvas(dark_bg, width=GRID_SZ, height=GRID_SZ,
                       bg="#080F1A", highlightthickness=0)
        cv.pack()
        # Draw after layout is committed so canvas has real dimensions
        cv.after(150, lambda c=cv, b=box, cl=g["cells"], col=color, sz=GRID_SZ:
                 self._draw_mini_grid(c, b, cl, col, sz))

        # ── Grid size label ───────────────────────────────────────
        size_row = tk.Frame(card, bg=C_SURFACE)
        size_row.pack(pady=(16, 2))
        tk.Label(size_row, text=g["full"],
                 font=("Segoe UI", 32, "bold"),
                 bg=C_SURFACE, fg=color).pack()

        tk.Label(card,
                 text=f"Blok {box}×{box}  ·  Grid {box*box}×{box*box} sel",
                 font=("Segoe UI", 9),
                 bg=C_SURFACE, fg=C_TEXT_DIM).pack()

        # ── Divider ───────────────────────────────────────────────
        divider = tk.Frame(card, bg=C_BORDER, height=1)
        divider.pack(fill="x", padx=18, pady=(16, 12))

        # ── Feature list ─────────────────────────────────────────
        feat_wrap = tk.Frame(card, bg=C_SURFACE)
        feat_wrap.pack(fill="x", padx=18)
        for emoji, label in g["features"]:
            row_f = tk.Frame(feat_wrap, bg=C_SURFACE)
            row_f.pack(fill="x", pady=3)
            tk.Label(row_f, text=emoji,
                     font=("Segoe UI", 12), bg=C_SURFACE,
                     fg=C_TEXT, width=3,
                     anchor="center").pack(side="left")
            tk.Label(row_f, text=label,
                     font=("Segoe UI", 10), bg=C_SURFACE,
                     fg=C_TEXT, anchor="w").pack(side="left", padx=(4, 0))

        # ── CTA button ────────────────────────────────────────────
        btn_wrap = tk.Frame(card, bg=C_SURFACE)
        btn_wrap.pack(fill="x", padx=18, pady=(18, 20))
        btn = tk.Button(btn_wrap,
                        text=f"▶   {g['cta']}",
                        font=("Segoe UI", 11, "bold"),
                        bg=color, fg="#0D1117",
                        activebackground="#FFFFFF",
                        activeforeground="#0D1117",
                        relief="flat", cursor="hand2", pady=12,
                        command=lambda b=box: self.on_select(b))
        btn.pack(fill="x")

        # ── Hover glow ────────────────────────────────────────────
        def _enter(_):
            card.config(highlightbackground=color, highlightthickness=2)

        def _leave(_):
            card.config(highlightbackground=C_BORDER, highlightthickness=1)

        hover_targets = [card, tag_bar, preview_wrap, ring, dark_bg,
                         cv, size_row, feat_wrap, btn_wrap, spacer]
        for w in hover_targets:
            w.bind("<Enter>", _enter)
            w.bind("<Leave>", _leave)
        # Click anywhere on card selects it
        for w in [card, spacer, preview_wrap, size_row, feat_wrap]:
            w.bind("<Button-1>", lambda _, b=box: self.on_select(b))

# =====================================================
# SCREEN: DIFFICULTY SELECT  (Redesigned v2)
# =====================================================
class DifficultyScreen(tk.Frame):

    # Canvas-drawn icon painters — each receives (canvas, size, color)
    @staticmethod
    def _icon_easy(cv, sz, col):
        """Minimalist seedling / leaf icon."""
        cx, cy = sz // 2, sz // 2
        # Stem
        cv.create_line(cx, cy + sz//3, cx, cy - sz//8,
                       fill=col, width=3, capstyle="round")
        # Left leaf
        cv.create_arc(cx - sz//3, cy - sz//3, cx + sz//8, cy + sz//12,
                      start=20, extent=140, fill=col, outline="")
        # Right leaf
        cv.create_arc(cx - sz//8, cy - sz//3, cx + sz//3, cy + sz//12,
                      start=20, extent=140, fill=col, outline="")
        # Ground dot
        cv.create_oval(cx - 3, cy + sz//3 - 3, cx + 3, cy + sz//3 + 3,
                       fill=col, outline="")

    @staticmethod
    def _icon_normal(cv, sz, col):
        """Lightning bolt icon."""
        pts = [
            sz*0.55, sz*0.08,
            sz*0.25, sz*0.50,
            sz*0.48, sz*0.50,
            sz*0.35, sz*0.92,
            sz*0.72, sz*0.42,
            sz*0.50, sz*0.42,
        ]
        cv.create_polygon(pts, fill=col, outline="")

    @staticmethod
    def _icon_hard(cv, sz, col):
        """Flame icon."""
        # Outer flame
        pts_out = [
            sz*0.50, sz*0.05,
            sz*0.72, sz*0.30,
            sz*0.80, sz*0.55,
            sz*0.72, sz*0.78,
            sz*0.50, sz*0.92,
            sz*0.28, sz*0.78,
            sz*0.20, sz*0.55,
            sz*0.28, sz*0.30,
        ]
        cv.create_polygon(pts_out, fill=col, smooth=True, outline="")
        # Inner highlight (lighter core)
        try:
            r_ = min(255, int(int(col[1:3], 16) * 0.5 + 255 * 0.5))
            g_ = min(255, int(int(col[3:5], 16) * 0.5 + 255 * 0.5))
            b_ = min(255, int(int(col[5:7], 16) * 0.5 + 255 * 0.5))
            inner = f"#{r_:02x}{g_:02x}{b_:02x}"
        except Exception:
            inner = "#FFFFFF"
        pts_in = [
            sz*0.50, sz*0.30,
            sz*0.62, sz*0.48,
            sz*0.66, sz*0.62,
            sz*0.50, sz*0.75,
            sz*0.34, sz*0.62,
            sz*0.38, sz*0.48,
        ]
        cv.create_polygon(pts_in, fill=inner, smooth=True, outline="")

    _ICON_PAINTERS = {
        "Easy":   _icon_easy.__func__,
        "Normal": _icon_normal.__func__,
        "Hard":   _icon_hard.__func__,
    }

    def __init__(self, master, username, grid_size, on_select):
        super().__init__(master, bg=C_BG)
        self.username  = username
        self.grid_size = grid_size
        self.on_select = on_select
        self.data      = load_data()
        self._build()

    # ── Main layout ───────────────────────────────────────────────
    def _build(self):
        self.place(relx=0, rely=0, relwidth=1, relheight=1)

        abg = AnimatedBG(self, bg=C_BG, highlightthickness=0)
        abg.place(relx=0, rely=0, relwidth=1, relheight=1)

        N          = self.grid_size * self.grid_size
        grid_label = f"{N}×{N}"

        # ── ML data ──────────────────────────────────────────────
        sessions = self.data["players"].get(self.username, {}).get("sessions", [])
        ml       = PlayerMLEngine()
        ml.sessions = [s for s in sessions
                       if s.get("grid_size") == self.grid_size]
        rec           = ml.recommend_difficulty()
        p_type, feat  = ml.classify_player()
        has_history   = feat["sessions_count"] > 0

        # ── Header ───────────────────────────────────────────────
        hdr = tk.Frame(self, bg=C_SURFACE)
        hdr.pack(fill="x")

        # Gradient bar (same as grid selection screen)
        gbar = tk.Canvas(hdr, height=8, bg=C_SURFACE, highlightthickness=0)
        gbar.pack(fill="x")
        gbar.after(100, lambda: draw_gradient_bar(gbar))

        hdr_inner = tk.Frame(hdr, bg=C_SURFACE)
        hdr_inner.pack(pady=18)

        # Title row
        title_row = tk.Frame(hdr_inner, bg=C_SURFACE)
        title_row.pack()
        tk.Label(title_row,
                 text=f"Grid {grid_label}",
                 font=("Segoe UI", 24, "bold"),
                 bg=C_SURFACE, fg=C_TEXT).pack(side="left")
        tk.Label(title_row,
                 text="  —  Pilih Kesulitan",
                 font=("Segoe UI", 24),
                 bg=C_SURFACE, fg=C_TEXT_DIM).pack(side="left")

        # AI recommendation badge
        if has_history:
            rec_colors = {"Easy": "#7EE787", "Normal": "#58A6FF", "Hard": "#FF7B7B"}
            rc = rec_colors.get(rec, C_ACCENT)
            badge_row = tk.Frame(hdr_inner, bg=C_SURFACE)
            badge_row.pack(pady=(8, 0))
            tk.Label(badge_row,
                     text=f"  🤖  AI merekomendasikan: ",
                     font=("Segoe UI", 10), bg=C_SURFACE, fg=C_TEXT_DIM).pack(side="left")
            tk.Label(badge_row,
                     text=f" {rec} ",
                     font=("Segoe UI", 10, "bold"),
                     bg=rc, fg="#0D1117", padx=6, pady=1).pack(side="left")
            tk.Label(badge_row,
                     text=f"   ·   Tipe: {p_type}",
                     font=("Segoe UI", 10), bg=C_SURFACE, fg=C_TEXT_DIM).pack(side="left")
        else:
            tk.Label(hdr_inner,
                     text="🤖  Pilih tingkat kesulitan untuk memulai!",
                     font=("Segoe UI", 10), bg=C_SURFACE, fg=C_TEXT_DIM).pack(pady=(6, 0))

        # ── Center — cards ────────────────────────────────────────
        center_wrap = tk.Frame(self, bg=C_BG)
        center_wrap.pack(side="top", fill="both", expand=True)

        inner = tk.Frame(center_wrap, bg=C_BG)
        inner.place(relx=0.5, rely=0.5, anchor="center")

        # Section label
        sec = tk.Frame(inner, bg=C_BG)
        sec.grid(row=0, column=0, columnspan=3, pady=(0, 22))
        tk.Frame(sec, width=50, height=1, bg=C_BORDER).pack(side="left", padx=(0, 12))
        tk.Label(sec, text="TINGKAT KESULITAN",
                 font=("Segoe UI", 10, "bold"),
                 bg=C_BG, fg=C_TEXT_DIM).pack(side="left")
        tk.Frame(sec, width=50, height=1, bg=C_BORDER).pack(side="left", padx=(12, 0))

        diff_configs = {
            "Easy": {
                "color": "#7EE787", "tag_bg": "#0A2010",
                "pct":   int(DIFF_THEMES["Easy"]["remove_pct"] * 100),
                "lines": ["Cocok untuk pemula", "Waktu lebih santai", "Aturan dasar Sudoku"],
            },
            "Normal": {
                "color": "#58A6FF", "tag_bg": "#071830",
                "pct":   int(DIFF_THEMES["Normal"]["remove_pct"] * 100),
                "lines": ["Tantangan menengah", "Butuh strategi", "Format kompetitif"],
            },
            "Hard": {
                "color": "#FF7B7B", "tag_bg": "#280808",
                "pct":   int(DIFF_THEMES["Hard"]["remove_pct"] * 100),
                "lines": ["Untuk ahli Sudoku", "Konsentrasi penuh", "Draft mode tersedia"],
            },
        }
        for col_idx, (diff, cfg) in enumerate(diff_configs.items()):
            is_rec = (diff == rec and has_history)
            self._card(inner, col_idx, diff, cfg, is_rec)

        # ── Footer stats bar ──────────────────────────────────────
        if has_history:
            foot = tk.Frame(self, bg=C_SURFACE)
            foot.pack(fill="x", side="bottom")

            # Thin accent line at top of footer
            foot_bar = tk.Canvas(foot, height=3, bg=C_SURFACE, highlightthickness=0)
            foot_bar.pack(fill="x")
            foot_bar.after(120, lambda: draw_gradient_bar(foot_bar, height=3))

            foot_inner = tk.Frame(foot, bg=C_SURFACE)
            foot_inner.pack(pady=10)
            tk.Label(foot_inner, text="STATISTIK GRID INI",
                     font=("Segoe UI", 8, "bold"),
                     bg=C_SURFACE, fg=C_TEXT_DIM).pack(pady=(0, 6))

            stat_row = tk.Frame(foot_inner, bg=C_SURFACE)
            stat_row.pack()

            stat_items = [
                ("🎮", "Sesi",       str(feat["sessions_count"])),
                ("❌", "Error Rate", f"{feat['error_rate']*100:.0f}%"),
                ("✅", "Completion", f"{feat['completion_rate']*100:.0f}%"),
                ("🧠", "Tipe",       p_type),
            ]
            for icon, lbl, val in stat_items:
                cell = tk.Frame(stat_row, bg=C_SURFACE2,
                                highlightbackground=C_BORDER,
                                highlightthickness=1)
                cell.pack(side="left", padx=5)
                inner_c = tk.Frame(cell, bg=C_SURFACE2)
                inner_c.pack(padx=14, pady=6)
                top_r = tk.Frame(inner_c, bg=C_SURFACE2)
                top_r.pack()
                tk.Label(top_r, text=icon,
                         font=("Segoe UI", 10), bg=C_SURFACE2,
                         fg=C_TEXT).pack(side="left")
                tk.Label(top_r, text=f"  {lbl}",
                         font=("Segoe UI", 8), bg=C_SURFACE2,
                         fg=C_TEXT_DIM).pack(side="left")
                tk.Label(inner_c, text=val,
                         font=("Segoe UI", 13, "bold"),
                         bg=C_SURFACE2, fg=C_TEXT).pack()
        else:
            # Minimal footer when no history
            foot = tk.Frame(self, bg=C_SURFACE, pady=10)
            foot.pack(fill="x", side="bottom")
            tk.Label(foot,
                     text="💡  Selesaikan puzzle pertamamu untuk melihat statistik dan rekomendasi AI",
                     font=("Segoe UI", 9), bg=C_SURFACE, fg=C_TEXT_DIM).pack()

    # ── Individual difficulty card ─────────────────────────────────
    def _card(self, parent, col_idx, diff, cfg, is_rec):
        color    = cfg["color"]
        CARD_W   = 220
        ICON_SZ  = 72
        painter  = self._ICON_PAINTERS.get(diff)

        outer = tk.Frame(parent, bg=C_BG)
        outer.grid(row=1, column=col_idx, padx=14, sticky="n")

        # Card outer frame — highlighted if recommended
        card = tk.Frame(outer, bg=C_SURFACE,
                        highlightbackground=color if is_rec else C_BORDER,
                        highlightthickness=2 if is_rec else 1,
                        cursor="hand2")
        card.pack()

        # Width enforcer
        spacer = tk.Frame(card, bg=C_SURFACE, height=1, width=CARD_W)
        spacer.pack(side="top")
        spacer.lower()

        # ── Top accent stripe (full-width colored bar) ────────────
        stripe = tk.Frame(card, bg=color, height=5)
        stripe.pack(fill="x")

        # ── "REKOMENDASI AI" banner when applicable ───────────────
        if is_rec:
            banner = tk.Frame(card, bg=color)
            banner.pack(fill="x")
            tk.Label(banner,
                     text="⭐  REKOMENDASI AI",
                     font=("Segoe UI", 8, "bold"),
                     bg=color, fg="#0D1117",
                     pady=4).pack()

        # ── Canvas icon ───────────────────────────────────────────
        icon_wrap = tk.Frame(card, bg=C_SURFACE)
        icon_wrap.pack(pady=(18, 4))
        cv = tk.Canvas(icon_wrap, width=ICON_SZ, height=ICON_SZ,
                       bg=C_SURFACE, highlightthickness=0)
        cv.pack()
        if painter:
            cv.after(80, lambda c=cv, p=painter, sz=ICON_SZ, col=color:
                     p(c, sz, col))

        # ── Difficulty name ───────────────────────────────────────
        tk.Label(card, text=diff.upper(),
                 font=("Segoe UI", 20, "bold"),
                 bg=C_SURFACE, fg=color).pack()

        # ── % removed pill ────────────────────────────────────────
        pct_row = tk.Frame(card, bg=C_SURFACE)
        pct_row.pack(pady=(4, 0))
        pct_pill = tk.Frame(pct_row, bg=cfg["tag_bg"])
        pct_pill.pack()
        tk.Label(pct_pill,
                 text=f"  {cfg['pct']}% sel dikosongkan  ",
                 font=("Segoe UI", 8, "bold"),
                 bg=cfg["tag_bg"], fg=color,
                 pady=3).pack()

        # ── Divider ───────────────────────────────────────────────
        tk.Frame(card, height=1, bg=C_BORDER).pack(fill="x", padx=14, pady=(14, 10))

        # ── Feature lines ─────────────────────────────────────────
        feat_wrap = tk.Frame(card, bg=C_SURFACE)
        feat_wrap.pack(padx=14, pady=(0, 4))
        for ln in cfg["lines"]:
            row_f = tk.Frame(feat_wrap, bg=C_SURFACE)
            row_f.pack(fill="x", pady=2)
            tk.Label(row_f, text="▸",
                     font=("Segoe UI", 9), bg=C_SURFACE,
                     fg=color, width=2).pack(side="left")
            tk.Label(row_f, text=ln,
                     font=("Segoe UI", 9), bg=C_SURFACE,
                     fg=C_TEXT, anchor="w").pack(side="left", padx=(2, 0))

        # ── CTA Button ────────────────────────────────────────────
        btn_wrap = tk.Frame(card, bg=C_SURFACE)
        btn_wrap.pack(fill="x", padx=14, pady=(14, 18))
        tk.Button(btn_wrap,
                  text=f"▶   MULAI {diff.upper()}",
                  font=("Segoe UI", 10, "bold"),
                  bg=color, fg="#0D1117",
                  activebackground="#FFFFFF",
                  activeforeground="#0D1117",
                  relief="flat", cursor="hand2", pady=10,
                  command=lambda d=diff: self.on_select(d)).pack(fill="x")

        # ── Hover glow ────────────────────────────────────────────
        def _enter(_):
            card.config(highlightbackground=color, highlightthickness=2)

        def _leave(_):
            card.config(highlightbackground=color if is_rec else C_BORDER,
                        highlightthickness=2 if is_rec else 1)

        hover_targets = [card, icon_wrap, cv, feat_wrap, btn_wrap, spacer, stripe]
        for w in hover_targets:
            w.bind("<Enter>", _enter)
            w.bind("<Leave>", _leave)
        for w in [card, icon_wrap, feat_wrap, spacer]:
            w.bind("<Button-1>", lambda _, d=diff: self.on_select(d))

# =====================================================
# LEADERBOARD WINDOW  (tabbed by grid+difficulty)
# =====================================================
class LeaderboardWindow(tk.Toplevel):
    # Fixed pixel widths for each column (must match header + row)
    COL_WIDTHS = [55, 140, 100, 90, 75, 75, 75]
    COL_HEADS  = ["RANK", "PLAYER", "DIFFICULTY", "TIME", "MOVES", "ERRORS", "SCORE"]

    def __init__(self, master, data):
        super().__init__(master)
        self.title("🏆 Leaderboard")
        self.configure(bg=C_BG)
        self.geometry("740x580")
        self.resizable(False, False)
        self.data = data
        self._active_grid = "9x9"
        self._active_diff = "All"
        self._build()
        self.grab_set()

    def _build(self):
        # Header
        hdr = tk.Frame(self, bg=C_SURFACE, pady=16)
        hdr.pack(fill="x")
        tk.Label(hdr, text="🏆  HALL OF FAME",
                 font=("Segoe UI", 20, "bold"), bg=C_SURFACE, fg=C_GOLD).pack()
        tk.Label(hdr, text="Best performances across all players",
                 font=FONT_SMALL, bg=C_SURFACE, fg=C_TEXT_DIM).pack()

        # Filter tabs
        tab_outer = tk.Frame(self, bg=C_SURFACE2)
        tab_outer.pack(fill="x")
        self._tab_frame = tab_outer

        # Grid filter
        gf = tk.Frame(tab_outer, bg=C_SURFACE2)
        gf.pack(side="left", padx=12, pady=8)
        tk.Label(gf, text="GRID:", font=("Segoe UI", 8, "bold"),
                 bg=C_SURFACE2, fg=C_TEXT_DIM).pack(side="left", padx=(0, 6))
        self._grid_btns = {}
        for lbl in ["4x4", "9x9"]:
            b = tk.Button(gf, text=lbl, font=FONT_SMALL,
                          relief="flat", cursor="hand2", padx=10, pady=3,
                          command=lambda g=lbl: self._set_grid(g))
            b.pack(side="left", padx=2)
            self._grid_btns[lbl] = b

        sep = tk.Frame(tab_outer, width=1, bg=C_BORDER)
        sep.pack(side="left", fill="y", padx=6)

        # Difficulty filter
        df = tk.Frame(tab_outer, bg=C_SURFACE2)
        df.pack(side="left", padx=4, pady=8)
        tk.Label(df, text="MODE:", font=("Segoe UI", 8, "bold"),
                 bg=C_SURFACE2, fg=C_TEXT_DIM).pack(side="left", padx=(0, 6))
        self._diff_btns = {}
        for lbl in ["All", "Easy", "Normal", "Hard"]:
            col = {"Easy": "#7EE787", "Normal": "#58A6FF", "Hard": "#FF7B7B"}.get(lbl, C_TEXT)
            b = tk.Button(df, text=lbl, font=FONT_SMALL,
                          relief="flat", cursor="hand2", padx=10, pady=3, fg=col,
                          command=lambda d=lbl: self._set_diff(d))
            b.pack(side="left", padx=2)
            self._diff_btns[lbl] = b

        # Table area (fixed canvas for alignment)
        self._table_frame = tk.Frame(self, bg=C_BG)
        self._table_frame.pack(fill="both", expand=True, padx=16, pady=10)

        # Close
        tk.Button(self, text="TUTUP", font=FONT_BTN,
                  bg=C_ACCENT, fg=C_BG, relief="flat", cursor="hand2",
                  command=self.destroy, pady=8).pack(padx=20, pady=10, fill="x")

        self._refresh()

    def _set_grid(self, g):
        self._active_grid = g
        self._refresh()

    def _set_diff(self, d):
        self._active_diff = d
        self._refresh()

    def _refresh(self):
        # Update tab button styles
        diff_colors = {"Easy": "#7EE787", "Normal": "#58A6FF",
                       "Hard": "#FF7B7B", "All": C_TEXT}
        for lbl, btn in self._grid_btns.items():
            active = (lbl == self._active_grid)
            btn.config(bg=C_ACCENT if active else C_SURFACE,
                       fg=C_BG if active else C_TEXT_DIM)
        for lbl, btn in self._diff_btns.items():
            active = (lbl == self._active_diff)
            col = diff_colors.get(lbl, C_TEXT)
            btn.config(bg=col if active else C_SURFACE,
                       fg=C_BG if active else col)

        # Clear table
        for w in self._table_frame.winfo_children():
            w.destroy()

        # Build entries
        box = 2 if self._active_grid == "4x4" else 3
        entries = []
        for uname, pdata in self.data.get("players", {}).items():
            for s in pdata.get("sessions", []):
                if not s.get("completed"): continue
                if s.get("grid_size", 3) != box: continue
                diff = s.get("difficulty", "Normal")
                if self._active_diff != "All" and diff != self._active_diff: continue
                t = s.get("total_time", 0)
                score = s.get("score") or calculate_score(
                    diff, t,
                    s.get("empty_cells", max(s.get("moves",1),1)),
                    s.get("errors",0), s.get("hints_used",0), True,
                    s.get("near_miss",0), s.get("guessing",0))
                entries.append({
                    "username": uname, "difficulty": diff,
                    "time": t, "moves": s.get("moves", 0),
                    "errors": s.get("errors", 0), "score": score
                })
        entries.sort(key=lambda x: -x["score"])

        # Header row
        hrow = tk.Frame(self._table_frame, bg=C_SURFACE2)
        hrow.pack(fill="x")
        for head, w in zip(self.COL_HEADS, self.COL_WIDTHS):
            tk.Label(hrow, text=head, width=0,
                     font=("Segoe UI", 9, "bold"),
                     bg=C_SURFACE2, fg=C_TEXT_DIM,
                     padx=0, pady=8, anchor="center"
                     ).pack(side="left", ipadx=0)
            # Use a fixed Frame to enforce column width
            # We actually place a frame of exact pixel width:
        # ---- rebuild using canvas for pixel-perfect alignment ----
        hrow.destroy()
        self._draw_header()

        # Scrollable rows
        sf = tk.Frame(self._table_frame, bg=C_BG)
        sf.pack(fill="both", expand=True)
        cv = tk.Canvas(sf, bg=C_BG, highlightthickness=0)
        sb = tk.Scrollbar(sf, orient="vertical", command=cv.yview)
        inner = tk.Frame(cv, bg=C_BG)
        inner.bind("<Configure>", lambda e: cv.configure(scrollregion=cv.bbox("all")))
        cv.create_window((0, 0), window=inner, anchor="nw")
        cv.configure(yscrollcommand=sb.set)
        cv.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

        ri = {1: "🥇", 2: "🥈", 3: "🥉"}
        dc = {"Easy": "#7EE787", "Normal": "#58A6FF", "Hard": "#FF7B7B"}

        for i, e in enumerate(entries[:25], 1):
            rbg = C_SURFACE if i % 2 == 0 else C_SURFACE2
            t = e["time"]
            ts = f"{int(t//60):02}:{int(t%60):02}"
            vals = [
                (ri.get(i, str(i)),       C_GOLD),
                (e["username"],           C_TEXT),
                (e["difficulty"],         dc.get(e["difficulty"], C_TEXT)),
                (ts,                      C_TEXT),
                (str(e["moves"]),         C_TEXT),
                (str(e["errors"]),        C_ERROR if e["errors"] > 5 else C_TEXT),
                (str(e["score"]),         C_ACCENT),
            ]
            self._draw_row(inner, vals, rbg)

        if not entries:
            tk.Label(inner,
                     text="Belum ada data!\nJadi yang pertama menyelesaikan puzzle 🚀",
                     font=FONT_BODY, bg=C_BG, fg=C_TEXT_DIM,
                     justify="center").pack(pady=30)

    def _draw_header(self):
        total_w = sum(self.COL_WIDTHS)
        cv = tk.Canvas(self._table_frame, height=34,
                       width=total_w, bg=C_SURFACE2, highlightthickness=0)
        cv.pack(fill="x")
        x = 0
        for head, w in zip(self.COL_HEADS, self.COL_WIDTHS):
            cv.create_rectangle(x, 0, x+w, 34, fill=C_SURFACE2, outline="")
            cv.create_text(x + w//2, 17, text=head,
                           fill=C_TEXT_DIM, font=("Segoe UI", 9, "bold"), anchor="center")
            x += w

    def _draw_row(self, parent, vals, bg):
        total_w = sum(self.COL_WIDTHS)
        cv = tk.Canvas(parent, height=34,
                       width=total_w, bg=bg, highlightthickness=0)
        cv.pack(fill="x")
        x = 0
        for (txt, fg), w in zip(vals, self.COL_WIDTHS):
            cv.create_rectangle(x, 0, x+w, 34, fill=bg, outline="")
            cv.create_text(x + w//2, 17, text=txt,
                           fill=fg, font=("Segoe UI", 10), anchor="center")
            x += w

# =====================================================
# PERFORMANCE DASHBOARD
# =====================================================
class PerformanceDashboard(tk.Frame):
    def __init__(self, master, controller, username, session, ml):
        super().__init__(master, bg=C_BG)
        self.controller = controller
        self.session    = session
        self.ml         = ml
        self.username   = username
        self._scroll_canvas = None
        self._mousewheel_bound = False
        self.place(relx=0, rely=0, relwidth=1, relheight=1)
        self._build()

    def _build(self):
        s     = self.session
        stats = self.ml.get_summary()
        pt    = stats["player_type"]
        feat  = stats["features"]
        rec   = stats["recommended_difficulty"]
        ti    = stats["type_info"]
        rc    = {"Easy": "#7EE787", "Normal": "#58A6FF", "Hard": "#FF7B7B"}
        tc    = ti.get("color", C_ACCENT)

        # ── Header ───────────────────────────────────────────────────
        hdr = tk.Frame(self, bg=C_SURFACE)
        hdr.pack(fill="x")

        gbar = tk.Canvas(hdr, height=6, bg=C_SURFACE, highlightthickness=0)
        gbar.pack(fill="x")
        gbar.after(80, lambda: draw_gradient_bar(gbar, height=6))

        hdr_inner = tk.Frame(hdr, bg=C_SURFACE)
        hdr_inner.pack(pady=14)
        tk.Label(hdr_inner, text="📊  PERFORMANCE ANALYSIS",
                 font=("Segoe UI", 18, "bold"), bg=C_SURFACE, fg=C_TEXT).pack()
        tk.Label(hdr_inner, text=f"Laporan sesi untuk  @{self.username}",
                 font=FONT_SMALL, bg=C_SURFACE, fg=C_TEXT_DIM).pack(pady=(2,0))

        # ── Scrollable body ───────────────────────────────────────────
        scroll_row = tk.Frame(self, bg=C_BG)
        scroll_row.pack(fill="both", expand=True)
        sbr = tk.Scrollbar(scroll_row, orient="vertical")
        sbr.pack(side="right", fill="y")
        sc = tk.Canvas(scroll_row, bg=C_BG, highlightthickness=0,
                       yscrollcommand=sbr.set)
        sc.pack(side="left", fill="both", expand=True)
        sbr.config(command=sc.yview)
        body = tk.Frame(sc, bg=C_BG)
        wid  = sc.create_window((0,0), window=body, anchor="nw")
        body.bind("<Configure>", lambda _: sc.configure(scrollregion=sc.bbox("all")))
        sc.bind("<Configure>",   lambda e: sc.itemconfig(wid, width=e.width))
        self._scroll_canvas = sc
        self._bind_mousewheel()

        # Padding wrapper
        pw = tk.Frame(body, bg=C_BG)
        pw.pack(fill="x", padx=24, pady=18)

        # ── Row 1: Player type + quick stats ──────────────────────────
        row1 = tk.Frame(pw, bg=C_BG)
        row1.pack(fill="x", pady=(0,16))

        # Player type card
        pcard = tk.Frame(row1, bg=C_SURFACE,
                         highlightbackground=tc, highlightthickness=2)
        pcard.pack(side="left", padx=(0,12), ipadx=18, ipady=14)
        tk.Label(pcard, text="TIPE PEMAINMU",
                 font=("Segoe UI", 9, "bold"), bg=C_SURFACE, fg=C_TEXT_DIM).pack()
        tk.Label(pcard, text=ti.get("emoji","🎮"),
                 font=("Segoe UI", 36), bg=C_SURFACE, fg=C_TEXT).pack(pady=2)
        tk.Label(pcard, text=pt,
                 font=("Segoe UI", 16, "bold"), bg=C_SURFACE, fg=tc).pack()
        tk.Label(pcard, text=ti.get("desc",""),
                 font=FONT_SMALL, bg=C_SURFACE, fg=C_TEXT_DIM).pack()

        # Recommendation card
        rcard = tk.Frame(row1, bg=C_SURFACE,
                         highlightbackground=C_BORDER, highlightthickness=1)
        rcard.pack(side="left", ipadx=14, ipady=14)
        tk.Label(rcard, text="REKOMENDASI AI",
                 font=("Segoe UI", 9, "bold"), bg=C_SURFACE, fg=C_TEXT_DIM).pack()
        tk.Label(rcard, text=rec,
                 font=("Segoe UI", 16, "bold"),
                 bg=C_SURFACE, fg=rc.get(rec, C_ACCENT)).pack(pady=4)
        tk.Label(rcard, text="Level berikutnya yang disarankan",
                 font=FONT_SMALL, bg=C_SURFACE, fg=C_TEXT_DIM).pack()

        # ── Quick actions + recommendation routing ───────────────────
        self.recommended_grid = 2 if rec == "Easy" else 3
        rec_grid_label = f"{self.recommended_grid}x{self.recommended_grid}"
        act = tk.Frame(pw, bg=C_SURFACE,
                       highlightbackground=C_BORDER, highlightthickness=1)
        act.pack(fill="x", pady=(0, 16))
        tk.Frame(act, bg=C_ACCENT2, height=4).pack(fill="x")
        act_inner = tk.Frame(act, bg=C_SURFACE)
        act_inner.pack(fill="x", padx=16, pady=12)

        tk.Label(act_inner, text="🎮  AKSI CEPAT",
                 font=("Segoe UI", 10, "bold"),
                 bg=C_SURFACE, fg=C_TEXT).pack(anchor="w")
        tk.Label(act_inner,
                 text=f"Rekomendasi AI saat ini mengarahkan ke grid {rec_grid_label}.",
                 font=FONT_SMALL, bg=C_SURFACE, fg=C_TEXT_DIM).pack(anchor="w", pady=(0, 10))

        act_btns = tk.Frame(act_inner, bg=C_SURFACE)
        act_btns.pack(fill="x")
        tk.Button(act_btns, text="🔄  PLAY AGAIN",
                  font=FONT_BTN, bg=C_ACCENT, fg=C_BG,
                  relief="flat", cursor="hand2", pady=10,
                  command=self.controller._play_again).pack(side="left", fill="x", expand=True, padx=(0, 4))
        tk.Button(act_btns, text=f"🤖  COBA REKOMENDASI AI → {rec_grid_label}",
                  font=FONT_BTN, bg=C_GOLD, fg=C_BG,
                  relief="flat", cursor="hand2", pady=10,
                  command=lambda: self.controller._start_recommended_grid(
                      self.recommended_grid, self.ml.recommend_difficulty())
                  ).pack(side="left", fill="x", expand=True, padx=4)
        tk.Button(act_btns, text="🚪  EXIT",
                  font=FONT_BTN, bg=C_ERROR, fg=C_WHITE,
                  relief="flat", cursor="hand2", pady=10,
                  command=self.controller._exit).pack(side="left", fill="x", expand=True, padx=(4, 0))

        # ── Row 2: Session stats grid ──────────────────────────────────
        tk.Label(pw, text="STATISTIK SESI INI",
                 font=("Segoe UI", 11, "bold"), bg=C_BG, fg=C_TEXT).pack(anchor="w")

        g = tk.Frame(pw, bg=C_BG)
        g.pack(fill="x", pady=(6,16))

        t        = s.get("total_time", 0)
        ts_str   = f"{int(t//60):02}:{int(t%60):02}"
        ec       = s.get("empty_cells", max(s.get("moves",1),1))
        tpc      = s.get("time_per_cell", t / max(ec, 1))
        score    = s.get("score") or calculate_score(
            s.get("difficulty","Normal"), t, ec,
            s.get("errors",0), s.get("hints_used",0), s.get("completed",False),
            s.get("near_miss",0), s.get("guessing",0))
        grid_sz  = s.get("grid_size", 3)
        nm       = s.get("near_miss", 0)
        gu       = s.get("guessing", 0)
        hl       = s.get("hearts_left", "?")
        mh       = s.get("max_hearts", grid_sz * grid_sz)

        items = [
            ("⏱️", "Total Waktu",         ts_str,                        C_ACCENT),
            ("📐", "Waktu / Sel Kosong",  f"{tpc:.1f}s",                 "#58A6FF"),
            ("✅", "Moves",               str(s.get("moves",0)),          C_ACCENT2),
            ("❌", "Errors",              str(s.get("errors",0)),         C_ERROR),
            ("💡", "Hints Terpakai",      str(s.get("hints_used",0)),    C_WARN),
            ("♥",  "Hati Tersisa",        f"{hl}/{mh}",                  C_ERROR),
            ("🎯", "Hampir Benar",        str(nm),                       "#F0883E"),
            ("🎲", "Asal Tebak",          str(gu),                       C_ERROR),
            ("🏆", "Skor",                str(score),                    C_GOLD),
            ("🎮", "Grid",                f"{grid_sz*grid_sz}×{grid_sz*grid_sz}", C_PURPLE),
        ]
        cols = 4
        for i, (ico, lbl, val, col) in enumerate(items):
            rr, cc = divmod(i, cols)
            cf = tk.Frame(g, bg=C_SURFACE,
                          highlightbackground=C_BORDER, highlightthickness=1)
            cf.grid(row=rr, column=cc, padx=4, pady=4,
                    sticky="nsew", ipadx=8, ipady=8)
            top_r = tk.Frame(cf, bg=C_SURFACE)
            top_r.pack()
            tk.Label(top_r, text=ico,
                     font=("Segoe UI", 12), bg=C_SURFACE, fg=col).pack(side="left")
            tk.Label(top_r, text=f"  {lbl}",
                     font=("Segoe UI", 8), bg=C_SURFACE, fg=C_TEXT_DIM).pack(side="left")
            tk.Label(cf, text=val,
                     font=("Segoe UI", 15, "bold"), bg=C_SURFACE, fg=col).pack()
        for cc in range(cols):
            g.columnconfigure(cc, weight=1)

        # ── Row 3: Behaviour analysis ──────────────────────────────────
        bcard = tk.Frame(pw, bg=C_SURFACE,
                         highlightbackground=C_BORDER, highlightthickness=1)
        bcard.pack(fill="x", pady=(0,16))
        tk.Frame(bcard, bg=C_WARN, height=4).pack(fill="x")
        binner = tk.Frame(bcard, bg=C_SURFACE)
        binner.pack(fill="x", padx=16, pady=12)

        tk.Label(binner, text="🧠  ANALISIS PERILAKU",
                 font=("Segoe UI", 10, "bold"),
                 bg=C_SURFACE, fg=C_TEXT).pack(anchor="w", pady=(0,8))

        b_items = [
            ("🎯 Hampir Benar (Near Miss)",
             f"{nm} sel",
             "Pemain mengetahui area jawaban tapi belum tepat — 1 kesalahan per sel.",
             "#F0883E"),
            ("🎲 Asal Tebak (Guessing)",
             f"{gu} sel",
             "Pemain mencoba angka berulang di sel yang sama — tidak tahu jawabannya.",
             C_ERROR),
            ("📐 Waktu/Sel (Latency)",
             f"{tpc:.1f}s per sel",
             "Kecepatan penyelesaian per sel kosong. Makin rendah = makin efisien.",
             "#58A6FF"),
            ("💔 Surrender Rate",
             f"{s.get('hints_used',0)}/{mh} hint",
             "Seberapa sering menyerah dan meminta bantuan. Makin rendah = lebih berani.",
             C_PURPLE),
        ]
        for ico_lbl, val_str, desc, col in b_items:
            brow = tk.Frame(binner, bg=C_SURFACE2,
                            highlightbackground=C_BORDER, highlightthickness=1)
            brow.pack(fill="x", pady=3)
            bl = tk.Frame(brow, bg=col, width=4)
            bl.pack(side="left", fill="y")
            bt = tk.Frame(brow, bg=C_SURFACE2)
            bt.pack(side="left", fill="x", expand=True, padx=(10,10), pady=8)
            top = tk.Frame(bt, bg=C_SURFACE2)
            top.pack(fill="x")
            tk.Label(top, text=ico_lbl,
                     font=("Segoe UI", 10, "bold"),
                     bg=C_SURFACE2, fg=C_TEXT).pack(side="left")
            tk.Label(top, text=val_str,
                     font=("Segoe UI", 10, "bold"),
                     bg=C_SURFACE2, fg=col).pack(side="right")
            tk.Label(bt, text=desc,
                     font=("Segoe UI", 8),
                     bg=C_SURFACE2, fg=C_TEXT_DIM, anchor="w").pack(anchor="w")

        # ── Row 4: Skill bars ──────────────────────────────────────────
        scard = tk.Frame(pw, bg=C_SURFACE,
                         highlightbackground=C_BORDER, highlightthickness=1)
        scard.pack(fill="x", pady=(0,16))
        tk.Frame(scard, bg=C_ACCENT, height=4).pack(fill="x")
        sinner = tk.Frame(scard, bg=C_SURFACE)
        sinner.pack(fill="x", padx=16, pady=12)
        tk.Label(sinner, text="📊  ANALISIS KEMAMPUAN",
                 font=("Segoe UI", 10, "bold"),
                 bg=C_SURFACE, fg=C_TEXT).pack(anchor="w", pady=(0,8))
        for sn, sv, sc_col in [
            ("Kecepatan",    max(0, min(100, 100 - feat["avg_time_per_cell"]*8)),  C_GOLD),
            ("Akurasi",      max(0, min(100, 100 - feat["error_rate"]*250)),        C_ACCENT2),
            ("Konsistensi",  feat["completion_rate"]*100,                           C_ACCENT),
            ("Kemandirian",  max(0, min(100, (1-feat["hint_rate"])*100)),           C_PURPLE),
        ]:
            self._bar(sinner, sn, sv, sc_col)

        # ── Row 4b: ML INSIGHTS (sklearn) ─────────────────────────────
        ml_conf       = stats.get("ml_confidence", 0)
        pred_score    = stats.get("predicted_next_score")
        pred_avail    = stats.get("predicted_score_avail", False)
        anom_status   = stats.get("anomaly_status", "unknown")
        anom_reason   = stats.get("anomaly_reason", "")
        sklearn_on    = stats.get("sklearn_active", False)

        mlcard = tk.Frame(pw, bg=C_SURFACE,
                          highlightbackground=C_BORDER, highlightthickness=1)
        mlcard.pack(fill="x", pady=(0, 16))
        tk.Frame(mlcard, bg=C_PURPLE, height=4).pack(fill="x")
        ml_inner = tk.Frame(mlcard, bg=C_SURFACE)
        ml_inner.pack(fill="x", padx=16, pady=12)

        # Header row: judul + status sklearn
        ml_head = tk.Frame(ml_inner, bg=C_SURFACE)
        ml_head.pack(fill="x", pady=(0, 8))
        tk.Label(ml_head, text="🤖  ML INSIGHTS",
                 font=("Segoe UI", 10, "bold"),
                 bg=C_SURFACE, fg=C_TEXT).pack(side="left")
        sk_badge_col = C_ACCENT2 if sklearn_on else C_TEXT_DIM
        sk_badge_txt = "✔ scikit-learn aktif" if sklearn_on else "⚠ mode rule-based"
        tk.Label(ml_head, text=f"  [{sk_badge_txt}]",
                 font=("Segoe UI", 8), bg=C_SURFACE,
                 fg=sk_badge_col).pack(side="left")

        ml_cols = tk.Frame(ml_inner, bg=C_SURFACE)
        ml_cols.pack(fill="x")

        # ── Card: KNN Confidence ─────────────────────────────────────
        knn_card = tk.Frame(ml_cols, bg=C_SURFACE2,
                            highlightbackground=C_BORDER, highlightthickness=1)
        knn_card.pack(side="left", fill="both", expand=True,
                      padx=(0, 6), ipadx=10, ipady=10)
        tk.Label(knn_card, text="KNN Classifier",
                 font=("Segoe UI", 8, "bold"),
                 bg=C_SURFACE2, fg=C_TEXT_DIM).pack(anchor="w")
        tk.Label(knn_card, text="Tipe Pemain",
                 font=("Segoe UI", 9), bg=C_SURFACE2, fg=C_TEXT_DIM).pack(anchor="w")
        tk.Label(knn_card,
                 text=f"{ti.get('emoji','🎮')}  {pt}",
                 font=("Segoe UI", 13, "bold"),
                 bg=C_SURFACE2, fg=tc).pack(anchor="w", pady=(2, 0))
        if sklearn_on and ml_conf > 0:
            conf_str = f"Confidence: {ml_conf:.1f}%"
            conf_col = C_ACCENT2 if ml_conf >= 70 else (C_WARN if ml_conf >= 45 else C_ERROR)
        else:
            conf_str = "Confidence: rule-based"
            conf_col = C_TEXT_DIM
        tk.Label(knn_card, text=conf_str,
                 font=("Segoe UI", 8), bg=C_SURFACE2, fg=conf_col).pack(anchor="w")
        # Mini confidence bar
        if sklearn_on and ml_conf > 0:
            bar_bg = tk.Frame(knn_card, bg=C_BORDER, height=6)
            bar_bg.pack(fill="x", pady=(4, 0))
            bar_fill = tk.Frame(bar_bg, bg=conf_col, height=6)
            bar_fill.place(x=0, y=0)
            def _grow_conf_bar(bg=bar_bg, fill=bar_fill, pct=ml_conf):
                bg.update_idletasks()
                w = int(bg.winfo_width() * pct / 100)
                fill.config(width=max(4, w))
            bar_bg.after(300, _grow_conf_bar)

        # ── Card: Linear Regression Score Prediction ─────────────────
        lr_card = tk.Frame(ml_cols, bg=C_SURFACE2,
                           highlightbackground=C_BORDER, highlightthickness=1)
        lr_card.pack(side="left", fill="both", expand=True,
                     padx=(0, 6), ipadx=10, ipady=10)
        tk.Label(lr_card, text="Linear Regression",
                 font=("Segoe UI", 8, "bold"),
                 bg=C_SURFACE2, fg=C_TEXT_DIM).pack(anchor="w")
        tk.Label(lr_card, text="Prediksi Skor Berikutnya",
                 font=("Segoe UI", 9), bg=C_SURFACE2, fg=C_TEXT_DIM).pack(anchor="w")
        if pred_avail and pred_score is not None:
            cur_score = s.get("score") or 0
            delta     = pred_score - cur_score
            delta_col = C_ACCENT2 if delta >= 0 else C_ERROR
            delta_str = f"(+{delta})" if delta >= 0 else f"({delta})"
            tk.Label(lr_card, text=f"🏆 {pred_score}",
                     font=("Segoe UI", 13, "bold"),
                     bg=C_SURFACE2, fg=C_GOLD).pack(anchor="w", pady=(2, 0))
            tk.Label(lr_card, text=f"vs saat ini {cur_score} {delta_str}",
                     font=("Segoe UI", 8), bg=C_SURFACE2, fg=delta_col).pack(anchor="w")
        else:
            min_needed = 3 - min(len(self.ml.sessions), 3)
            tk.Label(lr_card, text="—",
                     font=("Segoe UI", 13, "bold"),
                     bg=C_SURFACE2, fg=C_TEXT_DIM).pack(anchor="w", pady=(2, 0))
            tk.Label(lr_card,
                     text=f"Butuh {min_needed} sesi lagi",
                     font=("Segoe UI", 8),
                     bg=C_SURFACE2, fg=C_TEXT_DIM).pack(anchor="w")

        # ── Card: Isolation Forest Anomaly Detection ──────────────────
        iso_card = tk.Frame(ml_cols, bg=C_SURFACE2,
                            highlightbackground=C_BORDER, highlightthickness=1)
        iso_card.pack(side="left", fill="both", expand=True,
                      ipadx=10, ipady=10)
        tk.Label(iso_card, text="Isolation Forest",
                 font=("Segoe UI", 8, "bold"),
                 bg=C_SURFACE2, fg=C_TEXT_DIM).pack(anchor="w")
        tk.Label(iso_card, text="Anomaly Detection",
                 font=("Segoe UI", 9), bg=C_SURFACE2, fg=C_TEXT_DIM).pack(anchor="w")
        anom_emoji  = {"normal": "✅", "anomaly": "⚠️", "unknown": "❓"}
        anom_colors = {"normal": C_ACCENT2, "anomaly": C_WARN, "unknown": C_TEXT_DIM}
        anom_labels = {"normal": "Sesi Normal", "anomaly": "Sesi Anomali", "unknown": "Belum Cukup Data"}
        tk.Label(iso_card,
                 text=f"{anom_emoji.get(anom_status,'❓')} {anom_labels.get(anom_status,'—')}",
                 font=("Segoe UI", 13, "bold"),
                 bg=C_SURFACE2, fg=anom_colors.get(anom_status, C_TEXT_DIM)
                 ).pack(anchor="w", pady=(2, 0))
        tk.Label(iso_card, text=anom_reason,
                 font=("Segoe UI", 8), bg=C_SURFACE2,
                 fg=C_TEXT_DIM, wraplength=160, justify="left").pack(anchor="w")

        # ── Buttons ────────────────────────────────────────────────────
        btns = tk.Frame(pw, bg=C_BG)
        btns.pack(fill="x", pady=(4, 0))
        tk.Button(btns, text="🔄  PLAY AGAIN",
                  font=FONT_BTN, bg=C_ACCENT, fg=C_BG,
                  relief="flat", cursor="hand2", pady=10,
                  command=lambda: [self.destroy(),
                                   self.master.event_generate("<<PlayAgain>>")]
                  ).pack(side="left", fill="x", expand=True, padx=(0,4))
        tk.Button(btns, text="🏆  LEADERBOARD",
                  font=FONT_BTN, bg=C_GOLD, fg=C_BG,
                  relief="flat", cursor="hand2", pady=10,
                  command=lambda: LeaderboardWindow(self.winfo_toplevel(), load_data())
                  ).pack(side="left", fill="x", expand=True, padx=4)
        tk.Button(btns, text="🚪  EXIT",
                  font=FONT_BTN, bg=C_ERROR, fg=C_WHITE,
                  relief="flat", cursor="hand2", pady=10,
                  command=lambda: [self.destroy(),
                                   self.master.event_generate("<<ExitGame>>")]
                  ).pack(side="left", fill="x", expand=True, padx=(4,0))

    def _play_again(self):
        self.controller._play_again()

    def _start_recommendation(self):
        self.controller._start_recommended_grid(self.recommended_grid, self.ml.recommend_difficulty())

    def _exit(self):
        self.controller._exit()

    def _bind_mousewheel(self):
        """Enable mouse-wheel scrolling for the whole dashboard area."""
        if self._mousewheel_bound:
            return
        top = self.winfo_toplevel()
        top.bind_all("<MouseWheel>", self._on_mousewheel, add="+")
        top.bind_all("<Button-4>", self._on_mousewheel, add="+")
        top.bind_all("<Button-5>", self._on_mousewheel, add="+")
        self._mousewheel_bound = True

    def _on_mousewheel(self, event):
        canvas = getattr(self, "_scroll_canvas", None)
        if canvas is None or not canvas.winfo_exists():
            return
        if getattr(event, "num", None) == 4:
            canvas.yview_scroll(-1, "units")
        elif getattr(event, "num", None) == 5:
            canvas.yview_scroll(1, "units")
        else:
            delta = getattr(event, "delta", 0)
            if delta:
                canvas.yview_scroll(int(-1 * (delta / 120)), "units")

    def destroy(self):
        try:
            if self._mousewheel_bound:
                top = self.winfo_toplevel()
                top.unbind_all("<MouseWheel>")
                top.unbind_all("<Button-4>")
                top.unbind_all("<Button-5>")
                self._mousewheel_bound = False
        except Exception:
            pass
        super().destroy()

    def _bar(self, parent, name, val, color):
        row = tk.Frame(parent, bg=C_BG)
        row.pack(fill="x", pady=2)
        tk.Label(row, text=name, font=("Segoe UI", 9),
                 bg=C_BG, fg=C_TEXT_DIM, width=14, anchor="w").pack(side="left")
        bg = tk.Frame(row, bg=C_SURFACE2, height=12)
        bg.pack(side="left", fill="x", expand=True)
        bar = tk.Frame(bg, bg=color, height=12, width=4)
        bar.place(x=0, y=0)
        def grow():
            bg.update_idletasks()
            target = int(bg.winfo_width() * val / 100)
            def step(cur):
                if cur < target:
                    nxt = min(cur + max(1, target//20), target)
                    bar.config(width=nxt)
                    bg.after(15, lambda: step(nxt))
            step(4)
        bg.after(250, grow)
        tk.Label(row, text=f"{val:.0f}%", font=("Segoe UI", 9, "bold"),
                 bg=C_BG, fg=color, width=5).pack(side="right")

# =====================================================
# MAIN GAME SCREEN
# =====================================================
# Draft Mode (Hard only):
#   - Setiap sel kosong punya Canvas yang bisa menampilkan
#     angka draft kecil di 9 posisi (seperti Sudoku pro)
#   - Toggle mode draft dengan tombol ✏ atau shortcut "D"
#   - Saat draft mode aktif, angka yang diketik masuk ke
#     draft corner kecil, bukan ke sel utama
#   - Tombol "✔ Konfirmasi" commit semua draft ke board
#   - Tombol "✗ Hapus Draft" bersihkan semua draft
# =====================================================
class GameScreen(tk.Frame):
    # Ukuran canvas per sel (pixel) — disesuaikan per grid
    CELL_PX_9 = 58   # untuk 9x9
    CELL_PX_4 = 90   # untuk 4x4

    def __init__(self, master, username, grid_size, difficulty, on_finish):
        super().__init__(master, bg=C_BG)
        self.username   = username
        self.grid_size  = grid_size
        self.difficulty = difficulty
        self.on_finish  = on_finish
        self.N          = grid_size * grid_size
        self.BOX        = grid_size
        self.theme      = DIFF_THEMES[difficulty]

        self.CELL_PX = self.CELL_PX_4 if self.N == 4 else self.CELL_PX_9

        # Board state
        self.puzzle        = []
        self.solution      = []
        self.current_board = []
        self.selected      = None
        self.canvases      = {}   # (r,c) -> Canvas widget

        # Draft state — only in Hard mode
        # draft_board[r][c] = set of draft digits (1..N)
        self.draft_board   = {}
        self.draft_mode    = False   # True = pensil aktif
        self.numpad_btns   = {}

        # Stats
        self.timer_running    = False
        self.start_time       = 0
        self.elapsed          = 0
        self.game_over        = False
        self.error_count      = 0
        self.move_count       = 0
        self.hints_used       = 0
        self.last_action      = time.time()
        self.idle_after       = None
        self.hint_shown       = False

        # ── New analytics ──────────────────────────────────────────
        # Hati: total = N (grid size), setiap hint butuh 1 hati
        self.max_hearts       = self.N          # 4 untuk 4×4, 9 untuk 9×9
        self.hearts           = self.max_hearts
        # Per-cell error tracking untuk deteksi perilaku tebak vs hampir benar
        self.cell_errors      = {}   # {(r,c): count of wrong attempts}
        self.cell_last_time   = {}   # {(r,c): time of last input attempt}
        self.near_miss_count  = 0    # 1 error per sel → hampir tahu
        self.guessing_count   = 0    # 2+ errors per sel → asal tebak
        self.empty_cells      = 0    # diisi saat _start_new_game()

        # ML
        data     = load_data()
        sessions = [s for s in data["players"].get(username, {}).get("sessions", [])
                    if s.get("grid_size") == grid_size]
        self.ml  = PlayerMLEngine()
        self.ml.sessions = sessions

        self._build()
        self._start_new_game()

    # ──────────────────────────────────────────────────
    def _build(self):
        self.place(relx=0, rely=0, relwidth=1, relheight=1)

        # ═══ SIDEBAR ═══
        sb = tk.Frame(self, bg="#0A0E14", width=236)
        sb.pack(side="left", fill="y")
        sb.pack_propagate(False)

        tk.Frame(sb, height=1, bg=C_BORDER).pack(fill="x")
        brand = tk.Frame(sb, bg="#0A0E14", pady=14)
        brand.pack(fill="x")
        tk.Label(brand, text="⬛ SUDOKU AI",
                 font=("Segoe UI", 13, "bold"), bg="#0A0E14", fg=C_TEXT).pack()
        tk.Label(brand, text=f"@{self.username}  |  {self.N}×{self.N}",
                 font=FONT_SMALL, bg="#0A0E14", fg=C_TEXT_DIM).pack()
        tk.Frame(sb, height=1, bg=C_BORDER).pack(fill="x")

        # Difficulty buttons
        df = tk.Frame(sb, bg="#0A0E14", pady=10)
        df.pack(fill="x", padx=10)
        tk.Label(df, text="DIFFICULTY", font=("Segoe UI", 8, "bold"),
                 bg="#0A0E14", fg=C_TEXT_DIM).pack(anchor="w")
        for d in ["Easy", "Normal", "Hard"]:
            b = tk.Button(df, text=DIFF_THEMES[d]["emoji"] + "  " + d,
                          font=FONT_BTN_SM, relief="flat", cursor="hand2", pady=4,
                          command=lambda dd=d: self._change_difficulty(dd))
            b.pack(fill="x", pady=1)
            b.config(bg=DIFF_THEMES[d]["accent"] if d == self.difficulty else C_SURFACE2,
                     fg=C_BG if d == self.difficulty else C_TEXT_DIM)
            setattr(self, f"_dbtn_{d}", b)

        tk.Frame(sb, height=1, bg=C_BORDER).pack(fill="x", padx=10, pady=4)

        # Controls
        cf = tk.Frame(sb, bg="#0A0E14")
        cf.pack(fill="x", padx=10)
        tk.Label(cf, text="KONTROL", font=("Segoe UI", 8, "bold"),
                 bg="#0A0E14", fg=C_TEXT_DIM).pack(anchor="w")
        self._sb_btn(cf, "🔄  New Game",    "#DA3633", self._new_game)
        self._sb_btn(cf, "💡  Hint",         "#E67E22", self._give_hint)
        self._sb_btn(cf, "🏆  Leaderboard",  C_GOLD,
                     lambda: LeaderboardWindow(self.master, load_data()))

        tk.Frame(cf, height=1, bg=C_BORDER).pack(fill="x", pady=(6, 3))

        # Ganti Pemain — lebih menonjol dengan bg berwarna redup
        btn_gp = tk.Button(cf,
            text="🔙  Ganti Pemain",
            font=FONT_BTN_SM,
            bg="#2D1A4A", fg=C_PURPLE,
            activebackground=C_PURPLE, activeforeground="#0D1117",
            relief="flat", cursor="hand2", pady=6, anchor="w",
            command=self._change_player)
        btn_gp.pack(fill="x", pady=1)

        # Logout — background merah redup agar beda dari Ganti Pemain
        btn_lo = tk.Button(cf,
            text="🚪  Logout",
            font=FONT_BTN_SM,
            bg="#2A0808", fg=C_ERROR,
            activebackground=C_ERROR, activeforeground="#0D1117",
            relief="flat", cursor="hand2", pady=6, anchor="w",
            command=self._logout)
        btn_lo.pack(fill="x", pady=1)

        tk.Frame(cf, height=1, bg=C_BORDER).pack(fill="x", pady=(3, 0))

        tk.Frame(sb, height=1, bg=C_BORDER).pack(fill="x", padx=10, pady=4)

        # Draft Mode Panel (only shown in Hard)
        self.draft_panel = tk.Frame(sb, bg="#0A0E14")
        self.draft_panel.pack(fill="x", padx=10)
        if self.difficulty == "Hard":
            self._build_draft_panel()

        tk.Frame(sb, height=1, bg=C_BORDER).pack(fill="x", padx=10, pady=4)

        # AI Solver
        af = tk.Frame(sb, bg="#0A0E14")
        af.pack(fill="x", padx=10)
        tk.Label(af, text="AI SOLVER", font=("Segoe UI", 8, "bold"),
                 bg="#0A0E14", fg=C_TEXT_DIM).pack(anchor="w")
        btn_bt = self._sb_btn(af, "🤖  Backtracking MRV", "#1ABC9C", self._run_backtrack)
        Tooltip(btn_bt, "Backtracking + MRV heuristic: solver optimal")

        tk.Frame(sb, height=1, bg=C_BORDER).pack(fill="x", padx=10, pady=4)

        # Session stats
        sf = tk.Frame(sb, bg="#0A0E14")
        sf.pack(fill="x", padx=10)
        tk.Label(sf, text="STATISTIK SESI", font=("Segoe UI", 8, "bold"),
                 bg="#0A0E14", fg=C_TEXT_DIM).pack(anchor="w")
        self.lbl_moves  = self._stat_row(sf, "Moves",  "0")
        self.lbl_errors = self._stat_row(sf, "Errors", "0")
        self.lbl_hints  = self._stat_row(sf, "Hints",  "0")

        # ── Behaviour analysis rows ────────────────────────────────
        tk.Frame(sf, height=1, bg=C_BORDER).pack(fill="x", pady=(4,2))
        self.lbl_nearmiss = self._stat_row(sf, "Hampir Benar", "0",
                                           col="#F0883E")
        self.lbl_guessing = self._stat_row(sf, "Asal Tebak",   "0",
                                           col=C_ERROR)

        # ── Hearts bar ─────────────────────────────────────────────
        tk.Frame(sf, height=1, bg=C_BORDER).pack(fill="x", pady=(6,4))
        heart_lbl_row = tk.Frame(sf, bg="#0A0E14")
        heart_lbl_row.pack(fill="x")
        tk.Label(heart_lbl_row, text="HATI (HINT)",
                 font=("Segoe UI", 8, "bold"),
                 bg="#0A0E14", fg=C_TEXT_DIM).pack(side="left")
        self.lbl_hearts_count = tk.Label(heart_lbl_row,
                 text=f"{self.hearts}/{self.max_hearts}",
                 font=("Segoe UI", 8, "bold"),
                 bg="#0A0E14", fg=C_ERROR)
        self.lbl_hearts_count.pack(side="right")

        self.heart_row = tk.Frame(sf, bg="#0A0E14")
        self.heart_row.pack(fill="x", pady=(3,0))
        self.heart_labels = []
        for _ in range(self.max_hearts):
            lbl = tk.Label(self.heart_row, text="♥",
                           font=("Segoe UI", 11),
                           bg="#0A0E14", fg=C_ERROR)
            lbl.pack(side="left")
            self.heart_labels.append(lbl)
        self._update_hearts_ui()

        tk.Label(sb, text="Esc = Toggle Fullscreen",
                 font=("Segoe UI", 8, "italic"),
                 bg="#0A0E14", fg=C_TEXT_DIM).pack(side="bottom", pady=8)

        # ═══ MAIN AREA ═══
        main = tk.Frame(self, bg=C_BG)
        main.pack(side="right", fill="both", expand=True)

        # Top bar
        topbar = tk.Frame(main, bg=C_SURFACE, pady=10)
        topbar.pack(fill="x")
        title_row = tk.Frame(topbar, bg=C_SURFACE)
        title_row.pack()
        tk.Label(title_row, text="SUDOKU",
                 font=("Segoe UI", 24, "bold"), bg=C_SURFACE, fg=C_TEXT).pack(side="left")
        self.diff_badge = tk.Label(
            title_row,
            text=f" {self.theme['emoji']} {self.difficulty}  {self.N}×{self.N} ",
            font=("Segoe UI", 11, "bold"),
            bg=self.theme["accent"], fg=C_BG, padx=6, pady=2)
        self.diff_badge.pack(side="left", padx=8)

        # Draft mode indicator badge (shows in topbar)
        self.draft_badge = tk.Label(
            title_row, text="",
            font=("Segoe UI", 10, "bold"),
            bg="#4A2060", fg="#E8AAFF", padx=8, pady=2)
        # Only shown when draft mode active

        self.timer_var = tk.StringVar(value="00:00")
        self.timer_lbl = tk.Label(topbar, textvariable=self.timer_var,
                                  font=FONT_TIMER, bg=C_SURFACE, fg=C_ERROR)
        self.timer_lbl.pack()

        self.status_var = tk.StringVar(value="Pilih sel dan ketik angka untuk mulai...")
        self.status_lbl = tk.Label(topbar, textvariable=self.status_var,
                                   font=FONT_SMALL, bg=C_SURFACE, fg=C_TEXT_DIM)
        self.status_lbl.pack()

        # Board (canvas-based)
        board_wrap = tk.Frame(main, bg=C_BG)
        board_wrap.pack(expand=True)
        self.grid_container = tk.Frame(board_wrap,
                                       bg=self.theme["grid_line"], padx=3, pady=3)
        self.grid_container.pack()
        self._build_grid()

        # Numpad area
        numpad_outer = tk.Frame(main, bg=C_BG, pady=8)
        numpad_outer.pack()
        self.numpad_row = tk.Frame(numpad_outer, bg=C_BG)
        self.numpad_row.pack()
        self._build_numpad()

        self.master.bind("<Key>",    self._on_key)
        self.master.bind("<Escape>", self._on_esc)
        self._idle_check()

    # ── Sidebar helpers ──────────────────────────────
    def _sb_btn(self, parent, text, color, cmd):
        b = tk.Button(parent, text=text, font=FONT_BTN_SM,
                      bg=C_SURFACE2, fg=color,
                      activebackground=color, activeforeground=C_BG,
                      relief="flat", cursor="hand2", pady=5, anchor="w",
                      command=cmd)
        b.pack(fill="x", pady=1)
        return b

    def _stat_row(self, parent, label, val, col=None):
        r = tk.Frame(parent, bg="#0A0E14")
        r.pack(fill="x", pady=1)
        tk.Label(r, text=label, font=("Segoe UI", 9),
                 bg="#0A0E14", fg=C_TEXT_DIM).pack(side="left")
        lbl = tk.Label(r, text=val, font=("Segoe UI", 9, "bold"),
                       bg="#0A0E14", fg=col if col else C_TEXT)
        lbl.pack(side="right")
        return lbl

    # ── Draft Panel (Hard only) ───────────────────────
    def _build_draft_panel(self):
        for w in self.draft_panel.winfo_children():
            w.destroy()
        if self.difficulty != "Hard":
            return

        # Section title
        title_row = tk.Frame(self.draft_panel, bg="#0A0E14")
        title_row.pack(fill="x")
        tk.Label(title_row, text="✏ MODE DRAFT",
                 font=("Segoe UI", 8, "bold"),
                 bg="#0A0E14", fg="#E8AAFF").pack(side="left")
        tk.Label(title_row, text="  [D]",
                 font=("Segoe UI", 8),
                 bg="#0A0E14", fg="#8B6FAE").pack(side="left")

        # Toggle button
        self.draft_toggle_btn = tk.Button(
            self.draft_panel,
            text="✏  Aktifkan Draft Mode",
            font=FONT_BTN_SM,
            bg=C_SURFACE2, fg="#BC8CFF",
            activebackground="#4A2060", activeforeground="#E8AAFF",
            relief="flat", cursor="hand2", pady=6,
            command=self._toggle_draft_mode)
        self.draft_toggle_btn.pack(fill="x", pady=(4, 2))

        # Confirm & clear row
        action_row = tk.Frame(self.draft_panel, bg="#0A0E14")
        action_row.pack(fill="x", pady=2)

        self.confirm_btn = tk.Button(
            action_row,
            text="✔ Konfirmasi",
            font=("Segoe UI", 8, "bold"),
            bg="#1B4D2E", fg="#7EE787",
            activebackground="#2EA043", activeforeground=C_BG,
            relief="flat", cursor="hand2", pady=5,
            command=self._confirm_all_drafts)
        self.confirm_btn.pack(side="left", fill="x", expand=True, padx=(0, 2))

        self.clear_draft_btn = tk.Button(
            action_row,
            text="✗ Hapus Draft",
            font=("Segoe UI", 8, "bold"),
            bg="#3D1515", fg="#FF7B7B",
            activebackground="#5A1E1E", activeforeground=C_BG,
            relief="flat", cursor="hand2", pady=5,
            command=self._clear_all_drafts)
        self.clear_draft_btn.pack(side="right", fill="x", expand=True, padx=(2, 0))

        # Tips
        tip = tk.Label(self.draft_panel,
                       text="1 angka per kotak, lalu\nkonfirmasi ke board",
                       font=("Segoe UI", 8),
                       bg="#0A0E14", fg="#6B4E8E",
                       justify="left")
        tip.pack(anchor="w", pady=(4, 0))

    # ── Grid (Canvas cells) ───────────────────────────
    def _build_grid(self):
        for w in self.grid_container.winfo_children():
            w.destroy()
        self.blocks.clear() if hasattr(self, "blocks") else None
        self.blocks   = {}
        self.canvases = {}

        px = self.CELL_PX

        for br in range(self.BOX):
            for bc in range(self.BOX):
                blk = tk.Frame(self.grid_container,
                               bg=self.theme["grid_line"], padx=1, pady=1)
                blk.grid(row=br, column=bc, padx=2, pady=2)
                self.blocks[(br, bc)] = blk

        for r in range(self.N):
            for c in range(self.N):
                parent = self.blocks[(r // self.BOX, c // self.BOX)]
                # Always initialize with cell_bg of the NEW theme.
                # _draw_board will immediately repaint with precise colors,
                # but this prevents any flash of the old theme color.
                cv = tk.Canvas(parent, width=px, height=px,
                               bg=self.theme["cell_bg"],
                               highlightthickness=0, relief="flat",
                               cursor="hand2")
                cv.grid(row=r % self.BOX, column=c % self.BOX,
                        padx=1, pady=1)
                cv.bind("<Button-1>", lambda e, rr=r, cc=c: self._on_click(rr, cc))
                self.canvases[(r, c)] = cv

    # ── Numpad ────────────────────────────────────────
    def _build_numpad(self):
        for w in self.numpad_row.winfo_children():
            w.destroy()
        self.numpad_btns.clear()

        for n in range(1, self.N + 1):
            btn = tk.Button(self.numpad_row, text=str(n),
                            font=("Segoe UI", 13, "bold"),
                            bg=C_SURFACE2, fg=C_TEXT,
                            activebackground=self.theme["highlight"],
                            activeforeground=C_TEXT,
                            relief="flat", cursor="hand2",
                            width=3, height=1, pady=4,
                            command=lambda v=n: self._input_number(v))
            btn.pack(side="left", padx=2)
            self.numpad_btns[n] = btn

        del_btn = tk.Button(self.numpad_row, text="⌫",
                            font=("Segoe UI", 13),
                            bg="#3D1515", fg=C_ERROR,
                            activebackground="#5A1E1E", activeforeground=C_ERROR,
                            relief="flat", cursor="hand2",
                            width=3, height=1, pady=4,
                            command=self._delete_cell)
        del_btn.pack(side="left", padx=2)

    def _update_numpad(self):
        for n in range(1, self.N + 1):
            valid_count = 0
            for r in range(self.N):
                for c in range(self.N):
                    if self.current_board[r][c] == n:
                        self.current_board[r][c] = 0
                        ok = is_valid(self.current_board, n, (r, c), self.N, self.BOX)
                        self.current_board[r][c] = n
                        if ok:
                            valid_count += 1
            btn = self.numpad_btns.get(n)
            if not btn: continue
            if valid_count >= self.N:
                btn.config(state="disabled", bg="#1A1A1A", fg="#444444",
                           text=f"✓{n}", font=("Segoe UI", 10, "bold"),
                           cursor="arrow")
            else:
                btn.config(state="normal", bg=C_SURFACE2, fg=C_TEXT,
                           text=str(n), font=("Segoe UI", 13, "bold"),
                           cursor="hand2")

    # ── Draft Mode Logic ──────────────────────────────
    def _toggle_draft_mode(self):
        """Toggle draft mode on/off (Hard only)"""
        if self.difficulty != "Hard": return
        self.draft_mode = not self.draft_mode
        self._update_draft_ui()
        if self.draft_mode:
            self.status_var.set("✏ Draft Mode aktif — ketik 1 angka per kotak, Enter untuk konfirmasi")
        else:
            self.status_var.set("📝 Draft Mode nonaktif — kembali ke mode normal")
        self._draw_board()

    def _update_draft_ui(self):
        """Update visual state of draft toggle button and top badge.
        Guard terhadap kondisi None (sudah di-reset saat ganti difficulty)
        ataupun widget yang sudah destroyed agar tidak melempar TclError.
        """
        btn = getattr(self, "draft_toggle_btn", None)
        if btn is None:
            # Widget belum dibuat atau sudah di-reset — pastikan badge disembunyikan
            try:
                self.draft_badge.pack_forget()
            except Exception:
                pass
            return
        try:
            if not btn.winfo_exists():
                self.draft_toggle_btn = None
                try:
                    self.draft_badge.pack_forget()
                except Exception:
                    pass
                return
        except Exception:
            return
        if self.draft_mode:
            self.draft_toggle_btn.config(
                text="✏  Draft Mode: AKTIF",
                bg="#4A2060", fg="#E8AAFF",
                activebackground="#6A30A0")
            self.draft_badge.config(text=" ✏ MODE DRAFT  [1 angka/kotak] ")
            self.draft_badge.pack(side="left", padx=4)
        else:
            self.draft_toggle_btn.config(
                text="✏  Aktifkan Draft Mode",
                bg=C_SURFACE2, fg="#BC8CFF",
                activebackground="#4A2060")
            self.draft_badge.pack_forget()

    def _confirm_all_drafts(self):
        """Commit semua draft (1 angka per sel) ke board utama sekaligus."""
        committed = 0
        for r in range(self.N):
            for c in range(self.N):
                if self.puzzle[r][c] != 0: continue
                if self.current_board[r][c] != 0: continue
                num = self.draft_board.get((r, c), 0)
                if num == 0: continue
                ok = is_valid(self.current_board, num, (r, c), self.N, self.BOX)
                self.current_board[r][c] = num
                self.draft_board.pop((r, c), None)
                self.move_count += 1
                if not ok:
                    self.error_count += 1
                committed += 1
        if committed:
            self._update_stat_labels()
            self._update_numpad()
            self._draw_board()
            self.status_var.set(f"✔ {committed} draft dikonfirmasi ke board")
            self._check_win()
        else:
            self.status_var.set("ℹ Belum ada draft untuk dikonfirmasi")

    def _confirm_single_draft(self, r, c):
        """Commit draft dari sel yang sedang dipilih ke board utama."""
        if self.puzzle[r][c] != 0: return
        if self.current_board[r][c] != 0: return
        num = self.draft_board.get((r, c), 0)
        if num == 0:
            self.status_var.set("ℹ Sel ini belum punya draft")
            return
        ok = is_valid(self.current_board, num, (r, c), self.N, self.BOX)
        self.current_board[r][c] = num
        self.draft_board.pop((r, c), None)
        self.move_count += 1
        if not ok:
            self.error_count += 1
        if not self.timer_running: self._start_timer()
        self._update_stat_labels()
        self._update_numpad()
        self._draw_board()
        status_icon = "✔" if ok else "⚠"
        self.status_var.set(f"{status_icon} Draft [{num}] dikonfirmasi di ({r+1},{c+1})")
        self._check_win()

    def _clear_all_drafts(self):
        """Hapus semua draft di seluruh board."""
        count = len(self.draft_board)
        self.draft_board.clear()
        self._draw_board()
        if count:
            self.status_var.set(f"✗ {count} draft dihapus dari board")
        else:
            self.status_var.set("ℹ Tidak ada draft untuk dihapus")

    def _add_draft(self, r, c, num):
        """
        Set draft angka di sel tertentu.
        Hanya 1 angka per sel yang diperbolehkan.
        Jika angka yang sama diklik lagi → hapus draft (toggle).
        """
        current_draft = self.draft_board.get((r, c), 0)
        if current_draft == num:
            # Toggle off: angka sama diklik lagi
            self.draft_board.pop((r, c), None)
            self.status_var.set(f"✏ Draft di ({r+1},{c+1}) dihapus")
        else:
            # Set angka baru (mengganti draft sebelumnya jika ada)
            self.draft_board[(r, c)] = num
            self.status_var.set(
                f"✏ Draft [{num}] di ({r+1},{c+1}) — tekan Enter/Space untuk konfirmasi"
            )
        self._draw_board()

    # ── Game State ────────────────────────────────────
    def _reset_all_canvas_colors(self):
        """
        Reset ALL canvas backgrounds and content to the current theme.
        Called at the start of every new game to guarantee zero color bleed
        from a previous difficulty. We clear content here (cv.delete) so
        stale numbers from the old puzzle/theme never linger on screen.
        """
        t = self.theme
        for cv in self.canvases.values():
            cv.config(bg=t["cell_bg"])
            cv.delete("all")
        for blk in self.blocks.values():
            blk.config(bg=t["grid_line"])
        self.grid_container.config(bg=t["grid_line"])

    def _start_new_game(self):
        self.puzzle, self.solution = generate_puzzle(
            self.N, self.BOX, self.theme["remove_pct"])
        self.current_board = [row[:] for row in self.puzzle]
        self.draft_board   = {}
        self.draft_mode    = False
        self.selected      = None
        self.timer_running = False
        self.start_time    = 0
        self.elapsed       = 0
        self.game_over     = False
        self.error_count   = 0
        self.move_count    = 0
        self.hints_used    = 0
        self.last_action   = time.time()
        self.hint_shown    = False

        # Reset analytics per-game
        self.hearts          = self.max_hearts
        self.cell_errors     = {}
        self.cell_last_time  = {}
        self.near_miss_count = 0
        self.guessing_count  = 0
        self.empty_cells     = sum(
            1 for r in range(self.N)
            for c in range(self.N)
            if self.puzzle[r][c] == 0
        )

        self.timer_var.set("00:00")
        self.timer_lbl.config(fg=C_ERROR)
        self.status_var.set("Pilih sel dan ketik angka untuk mulai...")
        # Clear and recolor all canvases for the current theme
        self._reset_all_canvas_colors()
        self._update_draft_ui()
        self._update_stat_labels()
        self._update_hearts_ui()
        self._update_numpad()
        # Flush pending geometry events (misalnya akibat rebuild numpad atau
        # perubahan draft panel) sebelum _draw_board menulis ke canvas.
        # Ini mencegah situation di mana create_text() dipanggil sebelum
        # geometry manager selesai memproses perubahan layout.
        self.update_idletasks()
        self._draw_board()

    def _new_game(self):
        self._stop_timer()
        self._start_new_game()

    def _change_difficulty(self, diff):
        self._stop_timer()
        self.difficulty = diff
        self.theme      = DIFF_THEMES[diff]
        t = self.theme

        # ── 1. Sidebar difficulty buttons ─────────────────────────────
        for d in ["Easy", "Normal", "Hard"]:
            b = getattr(self, f"_dbtn_{d}")
            b.config(
                bg=DIFF_THEMES[d]["accent"] if d == diff else C_SURFACE2,
                fg=C_BG if d == diff else C_TEXT_DIM
            )

        # ── 2. Topbar badge ───────────────────────────────────────────
        self.diff_badge.config(
            text=f" {t['emoji']} {diff}  {self.N}×{self.N} ",
            bg=t["accent"]
        )

        # ── 3. Grid frame & block separators ─────────────────────────
        self.grid_container.config(bg=t["grid_line"])
        for blk in self.blocks.values():
            blk.config(bg=t["grid_line"])

        # ── 4. REPAINT EXISTING canvases — DO NOT rebuild ─────────────
        # Rebuilding canvas widgets (destroy + create) causes a Tkinter
        # race: newly created widgets haven't been committed to the screen
        # by the geometry manager yet, so create_text() silently discards
        # text drawn on them → numbers invisible until the next click.
        # Fix: keep the SAME canvas objects, just change their bg color
        # and clear content. _draw_board will repaint everything correctly.
        for cv in self.canvases.values():
            cv.config(bg=t["cell_bg"])
            cv.delete("all")

        # ── 5. Draft panel ────────────────────────────────────────────
        # PENTING: Reset referensi atribut ke None SEBELUM menghancurkan
        # widget children. Jika tidak, self.draft_toggle_btn masih menunjuk
        # ke widget yang sudah destroyed, sehingga pemanggilan .config()
        # selanjutnya di _update_draft_ui() akan melempar TclError dan
        # membatalkan seluruh callback — termasuk _draw_board() yang berakibat
        # angka puzzle tidak muncul sampai user mengklik board.
        self.draft_toggle_btn  = None
        self.confirm_btn       = None
        self.clear_draft_btn   = None
        for w in self.draft_panel.winfo_children():
            w.destroy()
        if diff == "Hard":
            self._build_draft_panel()

        # ── 6. Numpad ─────────────────────────────────────────────────
        self._build_numpad()

        # ── 7. Generate new puzzle & repaint ──────────────────────────
        self._start_new_game()

    def _change_player(self):
        self._stop_timer()
        self.master.event_generate("<<ChangePlayer>>")

    def _logout(self):
        self._stop_timer()
        self.master.event_generate("<<Logout>>")

    # ── Event Handlers ────────────────────────────────
    def _on_click(self, r, c):
        self.selected = (r, c)
        if not self.game_over:
            self.last_action = time.time()
        self._draw_board()

    def _on_key(self, event):
        if self.game_over: return

        # Draft mode toggle
        if event.char.lower() == "d" and self.difficulty == "Hard":
            self._toggle_draft_mode()
            return

        # Confirm single draft: Enter or Space (when draft mode & single sel)
        if event.keysym in ("Return", "space") and self.draft_mode and self.selected:
            self._confirm_single_draft(*self.selected)
            return

        if event.keysym in ("BackSpace", "Delete"):
            self._delete_cell()
        elif event.char in [str(i) for i in range(1, self.N + 1)]:
            self._input_number(int(event.char))
        elif event.keysym in ("Up", "Down", "Left", "Right"):
            self._move_sel(event.keysym)

    def _on_esc(self, _):
        fs = self.master.attributes("-fullscreen")
        self.master.attributes("-fullscreen", not fs)

    def _move_sel(self, key):
        if not self.selected:
            self.selected = (0, 0)
            self._draw_board()
            return
        r, c = self.selected
        d = {"Up": (-1,0), "Down": (1,0), "Left": (0,-1), "Right": (0,1)}[key]
        self.selected = (max(0, min(self.N-1, r+d[0])),
                         max(0, min(self.N-1, c+d[1])))
        self._draw_board()

    def _input_number(self, num):
        if self.game_over or not self.selected: return
        r, c = self.selected
        if self.puzzle[r][c] != 0: return  # fixed cell

        # ── Draft mode: add/toggle draft angka ───────
        if self.draft_mode and self.difficulty == "Hard":
            if self.current_board[r][c] != 0: return  # already confirmed
            if not self.timer_running: self._start_timer()
            self.last_action = time.time()
            self.hint_shown  = False
            self._add_draft(r, c, num)
            return

        # ── Normal mode: isi langsung ─────────────────
        count = sum(1 for rr in range(self.N) for cc in range(self.N)
                    if self.current_board[rr][cc] == num)
        if count >= self.N: return

        if not self.timer_running: self._start_timer()
        self.last_action = time.time()
        self.hint_shown  = False
        self.move_count += 1

        self.current_board[r][c] = 0
        num_valid = is_valid(self.current_board, num, (r, c), self.N, self.BOX)
        self.current_board[r][c] = num

        if not num_valid:
            self.error_count += 1
            now  = time.time()
            prev = self.cell_errors.get((r, c), 0)
            last = self.cell_last_time.get((r, c), 0)

            if prev == 0:
                # First error on this cell → "hampir benar" (near miss)
                self.near_miss_count += 1
            else:
                # Subsequent error on same cell → "asal tebak" (guessing)
                # Especially if repeated quickly (< 4 seconds gap)
                self.guessing_count += 1
                if prev == 1:
                    # Reclassify: the earlier "near miss" was actually guessing
                    self.near_miss_count = max(0, self.near_miss_count - 1)

            self.cell_errors[(r, c)] = prev + 1
            self.cell_last_time[(r, c)] = now
        else:
            # Correct input: clear per-cell error state for this cell
            self.cell_errors.pop((r, c), None)
            self.cell_last_time.pop((r, c), None)

        # Clear any draft on this cell when committing normally
        self.draft_board.pop((r, c), None)

        self._update_stat_labels()
        self._update_numpad()
        self._draw_board()
        self._check_win()

    def _delete_cell(self):
        if not self.selected or self.game_over: return
        r, c = self.selected
        if self.puzzle[r][c] != 0: return

        if self.draft_mode and self.difficulty == "Hard":
            had = self.draft_board.pop((r, c), 0)
            self._draw_board()
            if had:
                self.status_var.set(f"✗ Draft [{had}] di ({r+1},{c+1}) dihapus")
            else:
                self.status_var.set(f"ℹ Tidak ada draft di ({r+1},{c+1})")
            return

        self.current_board[r][c] = 0
        self.draft_board.pop((r, c), None)
        self.last_action = time.time()
        self._update_numpad()
        self._draw_board()

    # ── Cross-line helpers ────────────────────────────
    def _get_cross_cells(self, sel_r, sel_c):
        row_col  = set()
        box_only = set()
        for i in range(self.N):
            if i != sel_c: row_col.add((sel_r, i))
            if i != sel_r: row_col.add((i, sel_c))
        br, bc = sel_r // self.BOX, sel_c // self.BOX
        for i in range(br*self.BOX, br*self.BOX+self.BOX):
            for j in range(bc*self.BOX, bc*self.BOX+self.BOX):
                if (i, j) != (sel_r, sel_c) and (i, j) not in row_col:
                    box_only.add((i, j))
        return row_col, box_only

    def _is_cell_error(self, r, c):
        val = self.current_board[r][c]
        if val == 0: return False
        self.current_board[r][c] = 0
        valid = is_valid(self.current_board, val, (r, c), self.N, self.BOX)
        self.current_board[r][c] = val
        return not valid

    # ── Canvas-based board drawing ────────────────────
    def _draw_board(self):
        if not self.current_board: return
        t   = self.theme
        sel = self.selected
        px  = self.CELL_PX

        if sel:
            row_col_cross, box_cross = self._get_cross_cells(*sel)
            sel_val = self.current_board[sel[0]][sel[1]]
        else:
            row_col_cross, box_cross = set(), set()
            sel_val = 0

        # Font sizes (canvas text)
        main_font_sz  = 20 if self.N == 4 else 16
        draft_font_sz = 9  if self.N == 9 else 12

        for r in range(self.N):
            for c in range(self.N):
                cv       = self.canvases[(r, c)]
                val      = self.current_board[r][c]
                is_fixed = (self.puzzle[r][c] != 0)
                is_sel   = (sel == (r, c))
                in_rowcol = (r, c) in row_col_cross
                in_box    = (r, c) in box_cross
                same_num  = (val != 0 and val == sel_val and not is_sel)
                # draft_val and has_draft computed in render block below

                # ── Determine background ──────────────────────────
                if is_fixed:
                    bg = t["cell_fixed_bg"]
                    fg = t["cell_fixed_fg"]
                else:
                    if val != 0:
                        is_err = self._is_cell_error(r, c)
                        bg = t["error_bg"] if is_err else t["cell_bg"]
                        fg = t["error_fg"] if is_err else t["cell_user_fg"]
                    else:
                        bg = t["cell_bg"]
                        fg = t["cell_user_fg"]

                # ── Highlight layers (low → high priority) ────────
                if in_box and not is_sel:
                    bg = t["hl_box"]
                if in_rowcol and not is_sel:
                    bg = t["hl_rowcol"]
                if same_num:
                    bg = t["hl_same_bg"]
                    fg = t["hl_same_fg_fix"] if is_fixed else t["hl_same_fg_usr"]
                if is_sel:
                    bg = t["highlight"]
                    if is_fixed:
                        fg = t["cell_fixed_fg"]
                    elif val != 0 and self._is_cell_error(r, c):
                        fg = t["error_fg"]
                    else:
                        fg = t["cell_user_fg"]

                # ── Draft: single int per cell ────────────────────
                # draft is int (1-9/1-4), 0 = no draft
                draft_val = self.draft_board.get((r, c), 0) if val == 0 else 0
                has_draft = (draft_val != 0 and self.difficulty == "Hard")

                # Draft cells get a subtle violet background tint
                cell_bg = bg
                if has_draft and not is_sel:
                    cell_bg = self._blend_draft(bg)

                # ── Render canvas ──────────────────────────────────
                # CRITICAL: config bg FIRST, then delete+redraw.
                # If delete comes first, Tkinter briefly exposes the old
                # widget background before the new bg is applied.
                cv.config(bg=cell_bg)
                cv.delete("all")

                if val != 0:
                    # ── Confirmed number: large, centered ──────────
                    cv.create_text(
                        px // 2, px // 2,
                        text=str(val),
                        fill=fg,
                        font=("Segoe UI", main_font_sz, "bold"),
                        anchor="center"
                    )

                elif has_draft:
                    # ── Draft number: centered, italic-style, distinct color ──
                    # Slightly smaller than confirmed, with dashed border to signal
                    # it is NOT yet committed.
                    draft_fg_col = "#C89EFF" if is_sel else "#9B6DDB"
                    draft_sz     = main_font_sz - 2

                    # Dashed border to visually signal "temporary"
                    dash_col = "#6B3AAA" if not is_sel else "#BC8CFF"
                    # Draw dashed rectangle manually with small segments
                    seg = 6
                    for x in range(2, px-2, seg*2):
                        cv.create_line(x, 2, min(x+seg, px-2), 2,
                                       fill=dash_col, width=1)
                        cv.create_line(x, px-2, min(x+seg, px-2), px-2,
                                       fill=dash_col, width=1)
                    for y in range(2, px-2, seg*2):
                        cv.create_line(2, y, 2, min(y+seg, px-2),
                                       fill=dash_col, width=1)
                        cv.create_line(px-2, y, px-2, min(y+seg, px-2),
                                       fill=dash_col, width=1)

                    # Draft digit — centered, italic look via oblique
                    cv.create_text(
                        px // 2, px // 2,
                        text=str(draft_val),
                        fill=draft_fg_col,
                        font=("Segoe UI", draft_sz, "bold"),
                        anchor="center"
                    )

                    # Small "draft" label top-left corner
                    cv.create_text(
                        4, 4,
                        text="✏",
                        fill="#6B3AAA" if not is_sel else "#BC8CFF",
                        font=("Segoe UI", 7),
                        anchor="nw"
                    )

                    # Enter indicator bottom-right when selected
                    if is_sel:
                        cv.create_text(
                            px - 4, px - 4,
                            text="↵",
                            fill="#7EE787",
                            font=("Segoe UI", 9, "bold"),
                            anchor="se"
                        )

                else:
                    # ── Empty cell in draft mode: show pencil hint ─
                    if not is_fixed and self.draft_mode:
                        cv.create_text(
                            px // 2, px // 2,
                            text="✏",
                            fill="#3A2050",
                            font=("Segoe UI", 11),
                            anchor="center"
                        )

                # ── Selected cell border glow ──────────────────────
                if is_sel:
                    cv.create_rectangle(
                        1, 1, px-2, px-2,
                        outline=t["hl_sel_border"],
                        width=2
                    )

    # _draw_draft_numbers removed — draft is now always single value,
    # rendered directly in _draw_board as a large centered digit.

    def _blend_draft(self, hex_bg):
        """Blend a cell background with a purple draft tint"""
        try:
            r1,g1,b1 = int(hex_bg[1:3],16),int(hex_bg[3:5],16),int(hex_bg[5:7],16)
            # Purple tint: #3A1A5A
            r2,g2,b2 = 0x3A, 0x1A, 0x5A
            # 40% tint
            rr = int(r1*0.6 + r2*0.4)
            gg = int(g1*0.6 + g2*0.4)
            bb = int(b1*0.6 + b2*0.4)
            return f"#{rr:02x}{gg:02x}{bb:02x}"
        except:
            return hex_bg

    def _blend(self, hex1, hex2):
        try:
            r1,g1,b1 = int(hex1[1:3],16),int(hex1[3:5],16),int(hex1[5:7],16)
            r2,g2,b2 = int(hex2[1:3],16),int(hex2[3:5],16),int(hex2[5:7],16)
            return f"#{(r1+r2)//2:02x}{(g1+g2)//2:02x}{(b1+b2)//2:02x}"
        except:
            return hex1

    # ── Win check ─────────────────────────────────────
    def _is_board_valid_and_complete(self):
        expected = set(range(1, self.N + 1))
        for r in range(self.N):
            for c in range(self.N):
                if self.current_board[r][c] == 0: return False
        for r in range(self.N):
            if set(self.current_board[r]) != expected: return False
        for c in range(self.N):
            if {self.current_board[r][c] for r in range(self.N)} != expected: return False
        for br in range(self.BOX):
            for bc in range(self.BOX):
                box = set()
                for i in range(self.BOX):
                    for j in range(self.BOX):
                        box.add(self.current_board[br*self.BOX+i][bc*self.BOX+j])
                if box != expected: return False
        return True

    def _check_win(self):
        if not self._is_board_valid_and_complete(): return
        self.game_over = True
        self._stop_timer()
        self.status_var.set("🎉 SELAMAT! Puzzle selesai!")
        self.timer_lbl.config(fg=C_ACCENT2)
        self._flash_win()
        session = self._build_session(completed=True)
        self._save_session(session)
        self.master.after(1600, lambda: self.on_finish(session, self.ml))

    def _flash_win(self):
        t  = self.theme
        cs = list(self.canvases.values())
        pulse_cols = [t["highlight"], t["accent"], t["cell_bg"]]
        def step(i):
            col = pulse_cols[i % len(pulse_cols)]
            for cv in cs:
                cv.config(bg=col)
            if i < 6:
                self.master.after(140, lambda: step(i+1))
            else:
                self._draw_board()
        step(0)

    def _give_hint(self, auto=False):
        if self.game_over: return

        # Cek ketersediaan hati
        if self.hearts <= 0:
            self.status_var.set("💔 Tidak ada hati tersisa! Kamu sudah kehabisan hint.")
            self._banner("💔 Hati habis! Selesaikan sendiri atau mulai game baru.",
                         auto_hint=False)
            return

        for r in range(self.N):
            for c in range(self.N):
                if self.puzzle[r][c] == 0:
                    val = self.current_board[r][c]
                    if val == 0 or self._is_cell_error(r, c):
                        self.current_board[r][c] = self.solution[r][c]
                        self.draft_board.pop((r, c), None)
                        # Clear error tracking for this cell (hint fills it correctly)
                        self.cell_errors.pop((r, c), None)
                        self.cell_last_time.pop((r, c), None)
                        self.hints_used += 1
                        # Consume 1 heart
                        self.hearts = max(0, self.hearts - 1)
                        if not self.timer_running: self._start_timer()
                        src = "idle" if auto else "manual"
                        remaining = self.hearts
                        heart_str = "♥" * remaining + "♡" * (self.max_hearts - remaining)
                        self.status_var.set(
                            f"💡 Hint ({src}) — sisa hati: {heart_str}")
                        self._update_stat_labels()
                        self._update_hearts_ui()
                        self._update_numpad()
                        self._draw_board()
                        self._check_win()
                        return
        self.status_var.set("✅ Semua sel sudah terisi dengan benar!")

    def _idle_check(self):
        if self.game_over: return
        if self.timer_running:
            idle = time.time() - self.last_action
            give, reason = self.ml.should_give_hint(
                idle, self.error_count, self.move_count)
            if give and not self.hint_shown:
                self.hint_shown = True
                if reason == "idle":
                    self._banner(f"Sudah diam {idle:.0f} detik. Perlu hint? 💡",
                                 auto_hint=True)
                elif reason == "errors":
                    self._banner("Banyak kesalahan! Coba turunkan level? 💡",
                                 suggest_lower=True)
        self.idle_after = self.master.after(3000, self._idle_check)

    def _banner(self, msg, auto_hint=False, suggest_lower=False):
        ban = tk.Frame(self, bg="#1C2530",
                       highlightbackground=C_WARN, highlightthickness=1)
        ban.place(relx=0.5, y=8, anchor="n")
        tk.Label(ban, text=msg, font=FONT_BODY,
                 bg="#1C2530", fg=C_WARN, padx=12, pady=7).pack(side="left")
        if auto_hint:
            tk.Button(ban, text="Beri Hint", font=FONT_BTN_SM,
                      bg=C_WARN, fg=C_BG, relief="flat", cursor="hand2", padx=8,
                      command=lambda: [self._give_hint(auto=True), ban.destroy()]
                      ).pack(side="left", padx=4)
        if suggest_lower:
            diffs = ["Easy", "Normal", "Hard"]
            idx   = diffs.index(self.difficulty)
            if idx > 0:
                low = diffs[idx-1]
                tk.Button(ban, text=f"Turun ke {low}", font=FONT_BTN_SM,
                          bg=C_WARN, fg=C_BG, relief="flat", cursor="hand2", padx=8,
                          command=lambda l=low: [self._change_difficulty(l), ban.destroy()]
                          ).pack(side="left", padx=4)
        tk.Button(ban, text="✕", font=FONT_BTN_SM,
                  bg="#1C2530", fg=C_TEXT_DIM, relief="flat", cursor="hand2",
                  command=ban.destroy).pack(side="left", padx=4)
        self.master.after(8000, lambda: ban.destroy() if ban.winfo_exists() else None)

    # ── Stats / Timer ─────────────────────────────────
    def _update_stat_labels(self):
        self.lbl_moves.config(text=str(self.move_count))
        self.lbl_errors.config(text=str(self.error_count),
                               fg=C_ERROR if self.error_count > 0 else C_TEXT)
        self.lbl_hints.config(text=str(self.hints_used))
        self.lbl_nearmiss.config(text=str(self.near_miss_count),
                                  fg="#F0883E" if self.near_miss_count else C_TEXT)
        self.lbl_guessing.config(text=str(self.guessing_count),
                                  fg=C_ERROR if self.guessing_count else C_TEXT)

    def _update_hearts_ui(self):
        """Refresh heart icons: filled red = remaining, dim = spent."""
        for i, lbl in enumerate(self.heart_labels):
            if i < self.hearts:
                lbl.config(fg=C_ERROR, text="♥")
            else:
                lbl.config(fg="#3A1010", text="♡")
        remaining = self.hearts
        total     = self.max_hearts
        self.lbl_hearts_count.config(
            text=f"{remaining}/{total}",
            fg=C_ERROR if remaining > 0 else "#3A1010")

    def _start_timer(self):
        if not self.timer_running and not self.game_over:
            self.timer_running = True
            self.start_time    = time.time()
            self.timer_lbl.config(fg=C_ACCENT2)
            self._tick()

    def _stop_timer(self):
        self.timer_running = False
        if self.idle_after:
            try: self.master.after_cancel(self.idle_after)
            except: pass

    def _tick(self):
        if self.timer_running:
            self.elapsed = time.time() - self.start_time
            m, s = int(self.elapsed // 60), int(self.elapsed % 60)
            self.timer_var.set(f"{m:02}:{s:02}")
            self.master.after(500, self._tick)

    def _build_session(self, completed):
        tpc = self.elapsed / max(self.empty_cells, 1)
        return {
            "username":       self.username,
            "difficulty":     self.difficulty,
            "grid_size":      self.grid_size,
            "total_time":     self.elapsed,
            "moves":          self.move_count,
            "errors":         self.error_count,
            "hints_used":     self.hints_used,
            "completed":      completed,
            "timestamp":      time.time(),
            # ── New fields ─────────────────────────────────────────
            "empty_cells":    self.empty_cells,
            "time_per_cell":  round(tpc, 3),     # Total Time / Empty Cell
            "near_miss":      self.near_miss_count,
            "guessing":       self.guessing_count,
            "hearts_left":    self.hearts,
            "max_hearts":     self.max_hearts,
            "score":          calculate_score(
                self.difficulty, self.elapsed, self.empty_cells,
                self.error_count, self.hints_used, completed,
                self.near_miss_count, self.guessing_count),
        }

    def _save_session(self, s):
        data = load_data()
        if self.username not in data["players"]:
            data["players"][self.username] = {"sessions": [], "created_at": time.time()}
        data["players"][self.username]["sessions"].append(s)
        self.ml.add_session(s)
        save_data(data)

    # ── AI Solver ─────────────────────────────────────
    def _animate(self, hist, label):
        if not hist:
            messagebox.showerror("Error", "Solver tidak menemukan solusi.",
                                 parent=self.master)
            return
        self._stop_timer()
        self.game_over = True
        self.selected  = None
        self.status_var.set(f"🤖 {label}...")
        delay = 5 if self.N == 9 else 50

        def step(i):
            if i < len(hist):
                r, c, val = hist[i]
                self.current_board[r][c] = val
                cv = self.canvases[(r, c)]
                cv.delete("all")
                px = self.CELL_PX
                cv.config(bg=self.theme["highlight"] if val else self.theme["cell_bg"])
                if val:
                    cv.create_text(
                        px//2, px//2, text=str(val),
                        fill=self.theme["accent"],
                        font=("Segoe UI", 16 if self.N==9 else 20, "bold"),
                        anchor="center")
                self.master.after(delay, lambda: step(i+1))
            else:
                self._draw_board()
                self._update_numpad()
                self.status_var.set(f"✅ {label}")
        step(0)

    def _run_backtrack(self):
        self.current_board = [row[:] for row in self.puzzle]
        self.draft_board   = {}
        hist, exp, t = solve_backtracking_mrv(self.puzzle, self.N, self.BOX)
        self._animate(hist,
                      f"Backtracking MRV — {exp} nodes | {t*1000:.1f}ms")

# =====================================================
# SCREEN: PLAYER SELECT  (Ganti Pemain)
# =====================================================
# =====================================================
# SCREEN: GANTI PEMAIN  (2-panel: list + detail)
# =====================================================
class PlayerSelectScreen(tk.Frame):
    """
    Layar Ganti Pemain — dua panel:
    LEFT  : scrollable daftar pemain (compact rows)
    RIGHT : profil lengkap pemain yang dipilih (detail stats + ML)
    """

    _TYPE_COLORS = {
        "Speedrunner":  "#58A6FF",
        "Careful":      "#7EE787",
        "Learner":      "#BC8CFF",
        "Struggling":   "#FF7B7B",
        "Inconsistent": "#F0883E",
    }
    _TYPE_EMOJIS = {
        "Speedrunner":  "⚡",
        "Careful":      "🛡",
        "Learner":      "📚",
        "Struggling":   "💪",
        "Inconsistent": "🎲",
    }

    def __init__(self, master, current_user, on_select, on_new_player):
        super().__init__(master, bg=C_BG)
        self.current_user  = current_user
        self.on_select     = on_select
        self.on_new_player = on_new_player
        self.data          = load_data()
        self._selected     = current_user   # which player detail panel shows
        self._row_widgets  = {}             # name → card frame (for highlight)
        self._detail_frame = None
        self._build()

    # ── Stat computation ─────────────────────────────────────────
    def _get_stats(self, username):
        sessions = self.data["players"].get(username, {}).get("sessions", [])
        ml = PlayerMLEngine()
        ml.sessions = sessions
        p_type, feat = ml.classify_player()
        rec          = ml.recommend_difficulty()
        completed    = [s for s in sessions if s.get("completed", False)]
        scores = [
            s.get("score") or calculate_score(
                s.get("difficulty","Normal"),
                s.get("total_time", 1),
                s.get("empty_cells", max(s.get("moves",1),1)),
                s.get("errors",0),
                s.get("hints_used",0),
                s.get("completed",False),
                s.get("near_miss",0),
                s.get("guessing",0)
            )
            for s in sessions
        ]
        return {
            "sessions":        sessions,
            "n_sess":          len(sessions),
            "n_done":          len(completed),
            "best_score":      max(scores, default=0),
            "completion_rate": feat["completion_rate"],
            "error_rate":      feat["error_rate"],
            "hint_rate":       feat["hint_rate"],
            "avg_time":        feat["avg_time_per_cell"],
            "total_playtime":  sum(s.get("total_time",0) for s in sessions),
            "player_type":     p_type,
            "type_color":      self._TYPE_COLORS.get(p_type, C_ACCENT),
            "type_emoji":      self._TYPE_EMOJIS.get(p_type, "🎮"),
            "recommended":     rec,
            "feat":            feat,
        }

    def _fmt_time(self, s):
        s = int(s)
        if s >= 3600: return f"{s//3600}j {(s%3600)//60}m"
        if s >= 60:   return f"{s//60}m {s%60}s"
        return f"{s}s"

    def _initials(self, name):
        p = name.strip().split()
        if len(p) >= 2: return (p[0][0]+p[1][0]).upper()
        return name[:2].upper() if len(name)>=2 else name[0].upper()

    # ── Draw avatar ───────────────────────────────────────────────
    @staticmethod
    def _draw_avatar(canvas, size, color, initials):
        pad = 3
        # Outer ring
        canvas.create_oval(pad, pad, size-pad, size-pad,
                           outline=color, width=2, fill="")
        # Dark filled inner circle
        try:
            r_ = int(int(color[1:3],16)*0.22)
            g_ = int(int(color[3:5],16)*0.22)
            b_ = int(int(color[5:7],16)*0.22)
            inner = f"#{r_:02x}{g_:02x}{b_:02x}"
        except Exception:
            inner = "#0D1117"
        canvas.create_oval(pad+3, pad+3, size-pad-3, size-pad-3,
                           fill=inner, outline="")
        canvas.create_text(size//2, size//2, text=initials,
                           fill=color,
                           font=("Segoe UI", int(size*0.38), "bold"),
                           anchor="center")

    # ── Draw animated skill bar ───────────────────────────────────
    def _skill_bar(self, parent, label, pct, color, bg_col):
        row = tk.Frame(parent, bg=bg_col)
        row.pack(fill="x", pady=3)
        tk.Label(row, text=label,
                 font=("Segoe UI", 9), bg=bg_col, fg=C_TEXT_DIM,
                 width=16, anchor="w").pack(side="left")
        track = tk.Frame(row, bg=C_SURFACE2, height=10)
        track.pack(side="left", fill="x", expand=True)
        fill = tk.Frame(track, bg=color, height=10, width=4)
        fill.place(x=0, y=0)
        def _grow():
            track.update_idletasks()
            target = int(track.winfo_width() * pct / 100)
            def _step(cur):
                if cur < target:
                    nxt = min(cur + max(1, target//18), target)
                    fill.config(width=nxt)
                    track.after(12, lambda: _step(nxt))
            _step(4)
        track.after(300, _grow)
        tk.Label(row, text=f"{pct:.0f}%",
                 font=("Segoe UI", 9, "bold"),
                 bg=bg_col, fg=color, width=5).pack(side="right")

    # ── Main build ────────────────────────────────────────────────
    def _build(self):
        self.place(relx=0, rely=0, relwidth=1, relheight=1)

        abg = AnimatedBG(self, bg=C_BG, highlightthickness=0)
        abg.place(relx=0, rely=0, relwidth=1, relheight=1)

        # ── Header ───────────────────────────────────────────────
        hdr = tk.Frame(self, bg=C_SURFACE)
        hdr.pack(fill="x")

        gbar = tk.Canvas(hdr, height=8, bg=C_SURFACE, highlightthickness=0)
        gbar.pack(fill="x")
        gbar.after(100, lambda: draw_gradient_bar(gbar))

        hdr_inner = tk.Frame(hdr, bg=C_SURFACE)
        hdr_inner.pack(pady=16)

        title_row = tk.Frame(hdr_inner, bg=C_SURFACE)
        title_row.pack()
        tk.Label(title_row, text="👥  ",
                 font=("Segoe UI", 20), bg=C_SURFACE, fg=C_PURPLE).pack(side="left")
        tk.Label(title_row, text="Ganti Pemain",
                 font=("Segoe UI", 22, "bold"), bg=C_SURFACE, fg=C_TEXT).pack(side="left")

        tk.Label(hdr_inner,
                 text=f"Saat ini login sebagai  @{self.current_user}  ·  klik pemain untuk melihat profil",
                 font=("Segoe UI", 9), bg=C_SURFACE, fg=C_TEXT_DIM).pack(pady=(4,0))

        # ── Body: 2-column layout ─────────────────────────────────
        body = tk.Frame(self, bg=C_BG)
        body.pack(fill="both", expand=True)

        # LEFT PANEL — player list ────────────────────────────────
        left = tk.Frame(body, bg="#0A0E14", width=360)
        left.pack(side="left", fill="y")
        left.pack_propagate(False)

        # Left header
        lhdr = tk.Frame(left, bg="#0A0E14", pady=12)
        lhdr.pack(fill="x", padx=16)
        tk.Label(lhdr, text="PEMAIN TERDAFTAR",
                 font=("Segoe UI", 9, "bold"),
                 bg="#0A0E14", fg=C_TEXT_DIM).pack(side="left")
        all_players = self.data.get("players", {})
        tk.Label(lhdr, text=str(len(all_players)),
                 font=("Segoe UI", 9, "bold"),
                 bg="#0A0E14", fg=C_PURPLE).pack(side="right")

        tk.Frame(left, height=1, bg=C_BORDER).pack(fill="x")

        # Scrollable list
        sc = tk.Canvas(left, bg="#0A0E14", highlightthickness=0)
        sc.pack(side="left", fill="both", expand=True)

        sb = tk.Scrollbar(left, orient="vertical", command=sc.yview)
        sc.configure(yscrollcommand=sb.set)

        list_frame = tk.Frame(sc, bg="#0A0E14")
        win_id = sc.create_window((0,0), window=list_frame, anchor="nw")

        list_frame.bind("<Configure>",
                        lambda _: sc.configure(scrollregion=sc.bbox("all")))
        sc.bind("<Configure>",
                lambda e: sc.itemconfig(win_id, width=e.width))
        sc.bind("<MouseWheel>",
                lambda e: sc.yview_scroll(int(-1*(e.delta/120)), "units"))
        list_frame.bind("<MouseWheel>",
                        lambda e: sc.yview_scroll(int(-1*(e.delta/120)), "units"))

        if not all_players:
            tk.Label(list_frame, text="Belum ada pemain.",
                     font=("Segoe UI",11), bg="#0A0E14", fg=C_TEXT_DIM).pack(pady=30)
        else:
            def sort_key(n):
                nsess = len(all_players[n].get("sessions",[]))
                return (0 if n==self.current_user else 1, -nsess)
            for name in sorted(all_players, key=sort_key):
                self._player_row(list_frame, name, sc)
            if len(all_players) > 6:
                sb.pack(side="right", fill="y")

        # Divider between panels
        tk.Frame(body, width=1, bg=C_BORDER).pack(side="left", fill="y")

        # RIGHT PANEL — detail view ──────────────────────────────
        self._right_panel = tk.Frame(body, bg=C_BG)
        self._right_panel.pack(side="left", fill="both", expand=True)
        self._refresh_detail(self._selected)

        # ── Footer ───────────────────────────────────────────────
        foot = tk.Frame(self, bg=C_SURFACE)
        foot.pack(fill="x", side="bottom")

        foot_bar = tk.Canvas(foot, height=3, bg=C_SURFACE, highlightthickness=0)
        foot_bar.pack(fill="x")
        foot_bar.after(120, lambda: draw_gradient_bar(foot_bar, height=3))

        foot_inner = tk.Frame(foot, bg=C_SURFACE)
        foot_inner.pack(pady=12)

        tk.Button(foot_inner,
                  text="➕  PEMAIN BARU",
                  font=("Segoe UI", 10, "bold"),
                  bg=C_PURPLE, fg=C_TEXT,
                  activebackground="#9B5FEF", activeforeground=C_TEXT,
                  relief="flat", cursor="hand2", pady=10, padx=24,
                  command=self.on_new_player).pack(side="left", padx=(0,10))

        tk.Button(foot_inner,
                  text="↩  KEMBALI",
                  font=("Segoe UI", 10),
                  bg=C_SURFACE2, fg=C_TEXT_DIM,
                  activebackground=C_BORDER, activeforeground=C_TEXT,
                  relief="flat", cursor="hand2", pady=10, padx=18,
                  command=lambda: self.on_select(self.current_user)).pack(side="left")

    # ── Left panel: compact player row ───────────────────────────
    def _player_row(self, parent, name, scroll_canvas):
        sessions   = self.data["players"][name].get("sessions", [])
        n_sess     = len(sessions)
        ml         = PlayerMLEngine()
        ml.sessions = sessions
        p_type, _  = ml.classify_player()
        type_color = self._TYPE_COLORS.get(p_type, C_ACCENT)
        type_emoji = self._TYPE_EMOJIS.get(p_type, "🎮")
        is_me      = (name == self.current_user)
        is_sel     = (name == self._selected)

        BG_NORMAL   = "#0A0E14"
        BG_ACTIVE   = "#12182A"
        BG_SELECTED = "#131B2E"

        row_bg = BG_SELECTED if is_sel else (BG_ACTIVE if is_me else BG_NORMAL)

        row = tk.Frame(parent, bg=row_bg, cursor="hand2")
        row.pack(fill="x")

        # Left accent strip
        strip = tk.Frame(row, width=3, bg=C_PURPLE if is_me else (type_color if is_sel else "#0A0E14"))
        strip.pack(side="left", fill="y")

        inner = tk.Frame(row, bg=row_bg, pady=10, padx=12)
        inner.pack(side="left", fill="x", expand=True)

        # Avatar
        AV = 38
        av = tk.Canvas(inner, width=AV, height=AV,
                       bg=row_bg, highlightthickness=0)
        av.pack(side="left")
        av.after(80, lambda c=av, col=type_color, s=AV, n=name:
                 self._draw_avatar(c, s, col, self._initials(n)))

        # Name + type
        text_col = tk.Frame(inner, bg=row_bg)
        text_col.pack(side="left", padx=(10,0), fill="x", expand=True)

        name_row = tk.Frame(text_col, bg=row_bg)
        name_row.pack(anchor="w")
        tk.Label(name_row,
                 text=f"@{name}",
                 font=("Segoe UI", 11, "bold"),
                 bg=row_bg, fg=C_TEXT).pack(side="left")
        if is_me:
            tk.Label(name_row, text="  ●",
                     font=("Segoe UI", 9, "bold"),
                     bg=row_bg, fg="#7EE787").pack(side="left")

        sub = tk.Frame(text_col, bg=row_bg)
        sub.pack(anchor="w")
        tk.Label(sub, text=f"{type_emoji} {p_type}",
                 font=("Segoe UI", 8, "bold"),
                 bg=row_bg, fg=type_color).pack(side="left")
        tk.Label(sub, text=f"  ·  {n_sess} sesi",
                 font=("Segoe UI", 8),
                 bg=row_bg, fg=C_TEXT_DIM).pack(side="left")

        # Right arrow
        tk.Label(inner, text="›",
                 font=("Segoe UI", 16),
                 bg=row_bg, fg=type_color if is_sel else C_BORDER).pack(side="right")

        # Separator
        tk.Frame(parent, height=1, bg="#0F1520").pack(fill="x")

        # Store reference for highlight update
        self._row_widgets[name] = (row, strip, inner, text_col, name_row, sub, av)

        # Scroll wheel propagation
        for w in [row, inner, text_col, name_row, sub, av]:
            w.bind("<MouseWheel>",
                   lambda e: scroll_canvas.yview_scroll(int(-1*(e.delta/120)),"units"))

        # Click → update selection + detail
        def _click(_, n=name):
            self._selected = n
            self._refresh_row_highlights()
            self._refresh_detail(n)

        for w in [row, inner, text_col, av, name_row, sub]:
            w.bind("<Button-1>", _click)

        # Hover
        def _enter(_):
            if name != self._selected:
                row.config(bg="#0F1827")
                inner.config(bg="#0F1827")
        def _leave(_):
            bg_ = BG_SELECTED if self._selected==name else (BG_ACTIVE if is_me else BG_NORMAL)
            row.config(bg=bg_)
            inner.config(bg=bg_)

        for w in [row, inner]:
            w.bind("<Enter>", _enter)
            w.bind("<Leave>", _leave)

    # ── Highlight selected row in list ────────────────────────────
    def _refresh_row_highlights(self):
        for name, widgets in self._row_widgets.items():
            row, strip, inner, text_col, name_row, sub, av = widgets
            is_sel = (name == self._selected)
            is_me  = (name == self.current_user)
            sessions = self.data["players"][name].get("sessions",[])
            ml = PlayerMLEngine(); ml.sessions = sessions
            p_type, _ = ml.classify_player()
            tc = self._TYPE_COLORS.get(p_type, C_ACCENT)
            bg_ = "#131B2E" if is_sel else ("#12182A" if is_me else "#0A0E14")
            row.config(bg=bg_)
            inner.config(bg=bg_)
            strip.config(bg=C_PURPLE if is_me else (tc if is_sel else "#0A0E14"))

    # ── Scroll helper — bind wheel to every descendant ───────────
    def _bind_scroll_all(self, widget, scroll_fn):
        """
        Recursively bind <MouseWheel> (Windows/macOS) and <Button-4/5>
        (Linux) to every widget in the subtree rooted at `widget`.
        This ensures the detail panel always scrolls regardless of which
        child the cursor is hovering over.
        """
        widget.bind("<MouseWheel>",
                    lambda e: scroll_fn(int(-1 * (e.delta / 120)), "units"),
                    add="+")
        # Linux scroll events
        widget.bind("<Button-4>",  lambda e: scroll_fn(-1, "units"), add="+")
        widget.bind("<Button-5>",  lambda e: scroll_fn(+1, "units"), add="+")
        for child in widget.winfo_children():
            self._bind_scroll_all(child, scroll_fn)

    # ── Right panel: full player detail ──────────────────────────
    def _refresh_detail(self, username):
        # Clear old content
        for w in self._right_panel.winfo_children():
            w.destroy()

        if username not in self.data.get("players", {}):
            tk.Label(self._right_panel,
                     text="Pilih pemain dari daftar untuk melihat profil.",
                     font=("Segoe UI", 12), bg=C_BG, fg=C_TEXT_DIM).pack(expand=True)
            return

        st       = self._get_stats(username)
        tc       = st["type_color"]
        is_me    = (username == self.current_user)
        PANEL_BG = C_BG

        # ── Scrollable area: correct structure ────────────────────
        # container holds Canvas + Scrollbar side-by-side (both are
        # children of container, NOT of each other).
        container = tk.Frame(self._right_panel, bg=PANEL_BG)
        container.pack(fill="both", expand=True)

        sbr = tk.Scrollbar(container, orient="vertical")
        sbr.pack(side="right", fill="y")

        sc = tk.Canvas(container, bg=PANEL_BG, highlightthickness=0,
                       yscrollcommand=sbr.set)
        sc.pack(side="left", fill="both", expand=True)
        sbr.config(command=sc.yview)

        detail = tk.Frame(sc, bg=PANEL_BG)
        wid    = sc.create_window((0, 0), window=detail, anchor="nw")

        # Keep the inner frame as wide as the canvas viewport
        sc.bind("<Configure>",
                lambda e: sc.itemconfig(wid, width=e.width))
        # Update scrollregion whenever content height changes
        detail.bind("<Configure>",
                    lambda _: sc.configure(scrollregion=sc.bbox("all")))

        def _scroll(amount, unit):
            sc.yview_scroll(amount, unit)

        # Bind scroll on canvas and scrollbar itself
        sc.bind("<MouseWheel>",
                lambda e: _scroll(int(-1 * (e.delta / 120)), "units"))
        sc.bind("<Button-4>",  lambda e: _scroll(-1, "units"))
        sc.bind("<Button-5>",  lambda e: _scroll(+1, "units"))

        # After all content is built we recursively bind every child widget
        # so hovering ANY element still scrolls the panel.
        # We defer this until after pack/grid have run (after=0 = next idle).
        def _bind_all_later():
            self._bind_scroll_all(detail, _scroll)
        detail.after(0, _bind_all_later)

        pad = tk.Frame(detail, bg=PANEL_BG)
        pad.pack(fill="x", padx=32, pady=20)

        # ── Hero section ──────────────────────────────────────────
        hero = tk.Frame(pad, bg=C_SURFACE,
                        highlightbackground=tc, highlightthickness=2)
        hero.pack(fill="x", pady=(0,18))

        # Colored top stripe
        tk.Frame(hero, bg=tc, height=5).pack(fill="x")

        hero_inner = tk.Frame(hero, bg=C_SURFACE)
        hero_inner.pack(fill="x", padx=20, pady=16)

        # Avatar (large)
        AV = 72
        av_cv = tk.Canvas(hero_inner, width=AV, height=AV,
                          bg=C_SURFACE, highlightthickness=0)
        av_cv.pack(side="left")
        av_cv.after(80, lambda c=av_cv, col=tc, s=AV:
                    self._draw_avatar(c, s, col, self._initials(username)))

        hero_text = tk.Frame(hero_inner, bg=C_SURFACE)
        hero_text.pack(side="left", padx=(16,0), fill="x", expand=True)

        name_r = tk.Frame(hero_text, bg=C_SURFACE)
        name_r.pack(anchor="w")
        tk.Label(name_r, text=f"@{username}",
                 font=("Segoe UI", 20, "bold"),
                 bg=C_SURFACE, fg=C_TEXT).pack(side="left")
        if is_me:
            tk.Label(name_r, text="  ● AKTIF",
                     font=("Segoe UI", 10, "bold"),
                     bg=C_SURFACE, fg="#7EE787").pack(side="left")

        type_r = tk.Frame(hero_text, bg=C_SURFACE)
        type_r.pack(anchor="w", pady=(4,0))
        # Compute a dark tinted background from tc (12% brightness) — no 8-digit hex
        try:
            rb = int(int(tc[1:3],16)*0.15)
            gb = int(int(tc[3:5],16)*0.15)
            bb = int(int(tc[5:7],16)*0.15)
            badge_tint = f"#{rb:02x}{gb:02x}{bb:02x}"
        except Exception:
            badge_tint = "#1A1A2E"
        badge_bg = tk.Frame(type_r, bg=badge_tint)
        badge_bg.pack(side="left")
        tk.Label(badge_bg,
                 text=f"  {st['type_emoji']}  {st['player_type']}  ",
                 font=("Segoe UI", 10, "bold"),
                 bg=badge_tint, fg=tc, pady=3, padx=2).pack()

        # AI recommendation pill
        rec_col = {"Easy":"#7EE787","Normal":"#58A6FF","Hard":"#FF7B7B"}.get(st["recommended"],C_ACCENT)
        rec_r = tk.Frame(hero_text, bg=C_SURFACE)
        rec_r.pack(anchor="w", pady=(6,0))
        tk.Label(rec_r, text="AI Rekomendasikan: ",
                 font=("Segoe UI", 9), bg=C_SURFACE, fg=C_TEXT_DIM).pack(side="left")
        tk.Label(rec_r, text=f"  {st['recommended']}  ",
                 font=("Segoe UI", 9, "bold"),
                 bg=rec_col, fg="#0D1117", padx=4, pady=2).pack(side="left")

        # Total playtime
        tk.Label(rec_r, text=f"  ·  Total waktu: {self._fmt_time(st['total_playtime'])}",
                 font=("Segoe UI", 9),
                 bg=C_SURFACE, fg=C_TEXT_DIM).pack(side="left")

        # ── Quick stats row ───────────────────────────────────────
        qs = tk.Frame(pad, bg=PANEL_BG)
        qs.pack(fill="x", pady=(0,18))

        qitems = [
            ("🎮", "Total Sesi",  str(st["n_sess"]),                       "#58A6FF"),
            ("✅", "Selesai",     f"{st['n_done']} ({int(st['completion_rate']*100)}%)", "#7EE787"),
            ("🏆", "Best Score",  str(st["best_score"]),                    C_GOLD),
            ("⏱", "Rata Waktu/sel",f"{st['avg_time']:.1f}s",               "#BC8CFF"),
        ]
        for i, (ico, lbl, val, col) in enumerate(qitems):
            cell = tk.Frame(qs, bg=C_SURFACE,
                            highlightbackground=C_BORDER, highlightthickness=1)
            cell.grid(row=0, column=i, padx=5, sticky="nsew", ipadx=8, ipady=10)
            tk.Label(cell, text=ico,
                     font=("Segoe UI", 16), bg=C_SURFACE, fg=col).pack()
            tk.Label(cell, text=val,
                     font=("Segoe UI", 14, "bold"),
                     bg=C_SURFACE, fg=col).pack()
            tk.Label(cell, text=lbl,
                     font=("Segoe UI", 8),
                     bg=C_SURFACE, fg=C_TEXT_DIM).pack()
        for i in range(4):
            qs.columnconfigure(i, weight=1)

        # ── Skill analysis bars ───────────────────────────────────
        skill_card = tk.Frame(pad, bg=C_SURFACE,
                              highlightbackground=C_BORDER, highlightthickness=1)
        skill_card.pack(fill="x", pady=(0,18))

        sk_hdr = tk.Frame(skill_card, bg=C_SURFACE)
        sk_hdr.pack(fill="x", padx=16, pady=(12,8))
        tk.Label(sk_hdr, text="📊  ANALISIS KEMAMPUAN",
                 font=("Segoe UI", 10, "bold"),
                 bg=C_SURFACE, fg=C_TEXT).pack(side="left")

        sk_body = tk.Frame(skill_card, bg=C_SURFACE)
        sk_body.pack(fill="x", padx=16, pady=(0,14))

        feats = st["feat"]
        bars = [
            ("Kecepatan",     max(0, min(100, 100 - feats["avg_time_per_cell"]*8)), C_GOLD),
            ("Akurasi",       max(0, min(100, 100 - feats["error_rate"]*250)),      "#7EE787"),
            ("Konsistensi",   feats["completion_rate"]*100,                         "#58A6FF"),
            ("Kemandirian",   max(0, min(100, (1-feats["hint_rate"])*100)),         "#BC8CFF"),
        ]
        for lbl, pct, col in bars:
            self._skill_bar(sk_body, lbl, pct, col, C_SURFACE)

        # ── Recent session history ────────────────────────────────
        recent = st["sessions"][-8:][::-1]   # up to 8, newest first
        if recent:
            hist_card = tk.Frame(pad, bg=C_SURFACE,
                                 highlightbackground=C_BORDER, highlightthickness=1)
            hist_card.pack(fill="x", pady=(0,18))

            tk.Label(hist_card, text="🕒  RIWAYAT SESI TERBARU",
                     font=("Segoe UI", 10, "bold"),
                     bg=C_SURFACE, fg=C_TEXT,
                     padx=16, pady=10, anchor="w").pack(fill="x")
            tk.Frame(hist_card, height=1, bg=C_BORDER).pack(fill="x")

            diff_col = {"Easy":"#7EE787","Normal":"#58A6FF","Hard":"#FF7B7B"}
            for s in recent:
                srow = tk.Frame(hist_card, bg=C_SURFACE)
                srow.pack(fill="x", padx=16, pady=5)

                dc = diff_col.get(s.get("difficulty","Normal"), C_ACCENT)
                # Difficulty badge
                tk.Label(srow, text=f" {s.get('difficulty','?')} ",
                         font=("Segoe UI", 8, "bold"),
                         bg=dc, fg="#0D1117", padx=3).pack(side="left")
                # Grid
                gs = s.get("grid_size",3)
                tk.Label(srow, text=f"  {gs*gs}×{gs*gs}",
                         font=("Segoe UI", 9), bg=C_SURFACE, fg=C_TEXT_DIM).pack(side="left")
                # Completed / DNF
                done = s.get("completed", False)
                tk.Label(srow,
                         text="  ✅ Selesai" if done else "  ⏸ Berhenti",
                         font=("Segoe UI", 9),
                         bg=C_SURFACE, fg="#7EE787" if done else C_TEXT_DIM).pack(side="left")
                # Time
                t = int(s.get("total_time", 0))
                tk.Label(srow, text=f"  ⏱ {t//60:02}:{t%60:02}",
                         font=("Segoe UI", 9),
                         bg=C_SURFACE, fg=C_TEXT_DIM).pack(side="left")
                # Errors
                err = s.get("errors", 0)
                tk.Label(srow, text=f"  ❌ {err}",
                         font=("Segoe UI", 9),
                         bg=C_SURFACE, fg=C_ERROR if err else C_TEXT_DIM).pack(side="left")
                # Score
                sc_val = s.get("score") or calculate_score(
                    s.get("difficulty","Normal"),
                    int(s.get("total_time",1)),
                    s.get("empty_cells", max(s.get("moves",1),1)),
                    s.get("errors",0),
                    s.get("hints_used",0),
                    s.get("completed",False),
                    s.get("near_miss",0),
                    s.get("guessing",0)
                )
                tk.Label(srow, text=f"  🏆 {sc_val}",
                         font=("Segoe UI", 9, "bold"),
                         bg=C_SURFACE, fg=C_GOLD).pack(side="left")

                tk.Frame(hist_card, height=1, bg="#0F1520").pack(fill="x", padx=16)

        # ── CTA / Konfirmasi Section ──────────────────────────────
        cta_outer = tk.Frame(pad, bg=C_SURFACE,
                             highlightbackground=C_BORDER, highlightthickness=1)
        cta_outer.pack(fill="x", pady=(0, 8))

        # Thin accent stripe at top of CTA box
        tk.Frame(cta_outer, bg=tc, height=3).pack(fill="x")

        cta_inner = tk.Frame(cta_outer, bg=C_SURFACE)
        cta_inner.pack(fill="x", padx=18, pady=14)

        if is_me:
            # Already active — show "continue" in muted style
            tk.Label(cta_inner,
                     text="✓  Ini adalah akunmu saat ini",
                     font=("Segoe UI", 10),
                     bg=C_SURFACE, fg=C_TEXT_DIM).pack(anchor="w", pady=(0,8))
            tk.Button(cta_inner,
                      text="↩   LANJUTKAN",
                      font=("Segoe UI", 11, "bold"),
                      bg=C_SURFACE2, fg=C_TEXT,
                      activebackground=C_BORDER, activeforeground=C_TEXT,
                      relief="flat", cursor="hand2", pady=11,
                      command=lambda: self.on_select(username)).pack(fill="x")
        else:
            # Confirm switch — two-step: preview → confirm button
            info_row = tk.Frame(cta_inner, bg=C_SURFACE)
            info_row.pack(fill="x", pady=(0,10))

            tk.Label(info_row, text="⚠",
                     font=("Segoe UI", 14), bg=C_SURFACE, fg=C_WARN).pack(side="left")
            msg_col = tk.Frame(info_row, bg=C_SURFACE)
            msg_col.pack(side="left", padx=(8,0))
            tk.Label(msg_col,
                     text=f"Kamu akan berganti ke  @{username}",
                     font=("Segoe UI", 10, "bold"),
                     bg=C_SURFACE, fg=C_TEXT, anchor="w").pack(anchor="w")
            tk.Label(msg_col,
                     text=f"Sesi @{self.current_user} yang sedang berjalan akan dihentikan.",
                     font=("Segoe UI", 9),
                     bg=C_SURFACE, fg=C_TEXT_DIM, anchor="w").pack(anchor="w")

            # Two buttons side by side: Cancel + Confirm
            btn_row = tk.Frame(cta_inner, bg=C_SURFACE)
            btn_row.pack(fill="x")

            tk.Button(btn_row,
                      text="✕   BATAL",
                      font=("Segoe UI", 10),
                      bg=C_SURFACE2, fg=C_TEXT_DIM,
                      activebackground=C_BORDER, activeforeground=C_TEXT,
                      relief="flat", cursor="hand2", pady=10,
                      command=lambda: self.on_select(self.current_user)
                      ).pack(side="left", fill="x", expand=True, padx=(0,6))

            tk.Button(btn_row,
                      text=f"✓   KONFIRMASI — MASUK SEBAGAI @{username.upper()}",
                      font=("Segoe UI", 10, "bold"),
                      bg=tc, fg="#0D1117",
                      activebackground="#FFFFFF", activeforeground="#0D1117",
                      relief="flat", cursor="hand2", pady=10,
                      command=lambda u=username: self.on_select(u)
                      ).pack(side="left", fill="x", expand=True)


class SudokuApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Sudoku AI — ML Intelligence System")
        self.root.attributes("-fullscreen", True)
        self.root.configure(bg=C_BG)

        self.username   = None
        self.grid_size  = 3      # BOX size
        self.difficulty = "Normal"
        self.screen     = None

        self.root.bind("<<PlayAgain>>",        self._play_again)
        self.root.bind("<<ExitGame>>",         self._exit)
        self.root.bind("<<ChangePlayer>>",     self._show_player_select)  # → PlayerSelectScreen
        self.root.bind("<<Logout>>",           self._logout)

        self._show_login()

    def _clear(self):
        if self.screen:
            try:
                self.screen.place_forget()
                self.screen.destroy()
            except: pass
            self.screen = None

    def _show_login(self):
        self._clear()
        self.screen = LoginScreen(self.root, self._on_login)

    def _on_login(self, username, is_new, greeting):
        self.username = username
        self._clear()
        self.screen = GridSizeScreen(
            self.root, username, greeting,
            on_select=self._on_grid_selected)

    def _on_grid_selected(self, box):
        self.grid_size = box
        self._clear()
        self.screen = DifficultyScreen(
            self.root, self.username, self.grid_size,
            on_select=self._on_diff_selected)

    def _on_diff_selected(self, diff):
        self.difficulty = diff
        self._show_game()

    def _show_game(self):
        self._clear()
        self.screen = GameScreen(
            self.root,
            username=self.username,
            grid_size=self.grid_size,
            difficulty=self.difficulty,
            on_finish=self._on_finish)

    def _on_finish(self, session, ml):
        self._clear()
        self.screen = PerformanceDashboard(
            self.root, self, self.username, session, ml)

    def _start_recommended_grid(self, box, difficulty=None):
        self.grid_size = box
        if difficulty in {"Easy", "Normal", "Hard"}:
            self.difficulty = difficulty
        else:
            self.difficulty = "Easy" if box == 2 else "Normal"
        self._show_game()

    def _play_again(self, _=None):
        self._clear()
        self.screen = DifficultyScreen(
            self.root, self.username, self.grid_size,
            on_select=self._on_diff_selected)

    # ── Ganti Pemain → tampilkan PlayerSelectScreen ───────────────
    def _show_player_select(self, _=None):
        self._clear()
        self.screen = PlayerSelectScreen(
            self.root,
            current_user=self.username,
            on_select=self._on_player_selected,
            on_new_player=self._show_login,
        )

    def _on_player_selected(self, username):
        """Pemain dipilih dari PlayerSelectScreen — langsung ke grid select."""
        self.username = username
        self._clear()
        # Greeting berbeda untuk pemain yang sudah ada vs baru
        data    = load_data()
        is_new  = username not in data.get("players", {})
        greeting = "Halo" if is_new else "Halo kembali"
        self.screen = GridSizeScreen(
            self.root, username, greeting,
            on_select=self._on_grid_selected)

    # ── Logout → reset semua state lalu ke login ─────────────────
    def _logout(self, _=None):
        self.username   = None
        self.grid_size  = 3
        self.difficulty = "Normal"
        self._show_login()

    def _exit(self, _=None):
        if messagebox.askyesno("Keluar", "Yakin ingin keluar?", parent=self.root):
            self.root.destroy()

    def run(self):
        self.root.mainloop()

# =====================================================
# ENTRY POINT
# =====================================================
if __name__ == "__main__":
    app = SudokuApp()
    app.run()