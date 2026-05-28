# =============================================================================
# BAGIAN 0: METADATA PROYEK
# =============================================================================
# Nama Aplikasi  : Sudoku AI - Game Sudoku dengan Sistem Kecerdasan Buatan
# Versi          : 1.0.0
# Deskripsi      : Game Sudoku berbasis Python dengan integrasi model ML untuk
#                  rekomendasi kesulitan, deteksi anomali, prediksi skor, dan
#                  klasifikasi tipe pemain secara adaptif.
# Penulis        : Samuel Lie
# Institusi      : Universitas Bunda Mulia
# Mata Kuliah    : Machine Learning for Intelligence System
# Dosen Pengampu : Puguh Hiskiawan., S.Si., M.Si., Ph.D.
# Tahun          : 2026
# Hak Cipta      : (c) 2026 Samuel Lie. Seluruh hak dilindungi.
#                  Karya ini diajukan untuk pendaftaran HAKI sebagai
#                  Ciptaan Program Komputer.
# Teknologi      : Python 3, tkinter, pygame, scikit-learn, Pillow, NumPy
# Dependensi     : Lihat requirements.txt
# =============================================================================

# =============================================================================
# BAGIAN 1: IMPORT LIBRARY
# =============================================================================

# (1) Standard library
import os
import sys
import math
import json
import time
import copy
import pickle
import random
import threading
import subprocess
import datetime
import traceback

# (2) Third-party
import tkinter as tk
from tkinter import messagebox
import numpy as np

# (3) Optional
try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

try:
    from PIL import (Image as _PilImage, ImageDraw as _PilDraw,
                     ImageFont as _PilFont, ImageFilter as _PilFilter,
                     ImageEnhance as _PilEnhance, ImageGrab as _PilGrab,
                     ImageTk as _PilImageTk)
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.ensemble import (IsolationForest, RandomForestClassifier,
                                  RandomForestRegressor,
                                  HistGradientBoostingRegressor)
    from sklearn.multioutput import MultiOutputRegressor
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    cv2 = None
    CV2_AVAILABLE = False

# ML continuous-improvement helpers
_ml_retrain_lock = threading.Lock() # prevent concurrent retrains

# [DEMO-POINT] Continuous learning - retrain di background thread
# Menjadwalkan retrain model ML di background thread setelah sesi terbaru tersimpan, sambil mencegah dua proses retrain berjalan bersamaan.
def _ml_schedule_retrain(ml_instance):
    # Fungsi bantu ini memecah logika ml schedule retrain agar langkah kecilnya lebih rapi dan mudah dipakai ulang di bagian ini.
    def _do():
        if not _ml_retrain_lock.acquire(blocking=False):
            return
        try:
            ml_instance._models_dirty = True
            ml_instance._train_models()
            ml_instance._ml_dirty = True
            ml_instance._train_ml_models(force=True)
            ml_instance._train_hint_model()
        except Exception:
            pass
        finally:
            _ml_retrain_lock.release()
    threading.Thread(target=_do, daemon=True).start()

# Blur-overlay helper
# Mengambil screenshot area jendela, memberi blur dan efek gelap, lalu mengembalikannya sebagai PhotoImage untuk latar popup.
def _grab_blur_bg(root, radius: int = 14, darken: float = 0.38):
    if not PIL_AVAILABLE:
        return None
    try:
        root.update_idletasks()
        x = root.winfo_rootx()
        y = root.winfo_rooty()
        w = root.winfo_width()
        h = root.winfo_height()
        if w < 10 or h < 10:
            return None

        full = _PilGrab.grab()
        phys_w, phys_h = full.size

        logi_w = root.winfo_screenwidth()
        logi_h = root.winfo_screenheight()

        sx = phys_w / logi_w if logi_w > 0 else 1.0
        sy = phys_h / logi_h if logi_h > 0 else 1.0

        cx  = int(round(x * sx))
        cy  = int(round(y * sy))
        cx2 = int(round((x + w) * sx))
        cy2 = int(round((y + h) * sy))

        cx  = max(0, min(cx,  phys_w))
        cy  = max(0, min(cy,  phys_h))
        cx2 = max(0, min(cx2, phys_w))
        cy2 = max(0, min(cy2, phys_h))

        img = full.crop((cx, cy, cx2, cy2))

        if img.size != (w, h):
            img = img.resize((w, h), _PilImage.LANCZOS)

        img = img.filter(_PilFilter.GaussianBlur(radius=radius))
        img = _PilEnhance.Brightness(img).enhance(darken)
        return _PilImageTk.PhotoImage(img)
    except Exception:
        return None

# Mengambil area jendela yang sama seperti _grab_blur_bg, tetapi mengembalikan objek PIL Image agar bisa diproses lagi.
def _grab_blur_pil(root, radius: int = 14, darken: float = 0.40):
    if not PIL_AVAILABLE:
        return None
    try:
        root.update_idletasks()
        x = root.winfo_rootx()
        y = root.winfo_rooty()
        w = root.winfo_width()
        h = root.winfo_height()
        if w < 10 or h < 10:
            return None
        full = _PilGrab.grab()
        phys_w, phys_h = full.size
        logi_w = root.winfo_screenwidth()
        logi_h = root.winfo_screenheight()
        sx = phys_w / logi_w if logi_w > 0 else 1.0
        sy = phys_h / logi_h if logi_h > 0 else 1.0
        cx  = int(round(x * sx))
        cy  = int(round(y * sy))
        cx2 = int(round((x + w) * sx))
        cy2 = int(round((y + h) * sy))
        cx  = max(0, min(cx,  phys_w))
        cy  = max(0, min(cy,  phys_h))
        cx2 = max(0, min(cx2, phys_w))
        cy2 = max(0, min(cy2, phys_h))
        img = full.crop((cx, cy, cx2, cy2))
        if img.size != (w, h):
            img = img.resize((w, h), _PilImage.LANCZOS)
        img = img.filter(_PilFilter.GaussianBlur(radius=radius))
        img = _PilEnhance.Brightness(img).enhance(darken)
        return img
    except Exception:
        return None

# Menempatkan background blur atau fallback polos pada widget target agar overlay tampil rapi di atas layar aktif.
def _place_blur_canvas(parent, root, radius: int = 14, darken: float = 0.38,
                       pre_captured=None):
    photo = pre_captured if pre_captured is not None else \
            _grab_blur_bg(root, radius=radius, darken=darken)

    if photo:
        lbl = tk.Label(parent, image=photo, bd=0, highlightthickness=0)
        lbl._blur_photo_ref = photo
        lbl.place(relx=0, rely=0, relwidth=1, relheight=1)
        return lbl
    else:
        is_dark = (_CURRENT_THEME_NAME == "dark")
        fb = tk.Frame(parent, bg="#050810" if is_dark else "#8090A8")
        fb.place(relx=0, rely=0, relwidth=1, relheight=1)
        return fb

# Menurunkan ikon pojok ke bawah overlay yang sedang tampil supaya elemen penting di popup tidak saling menutupi.
def _corner_icons_lower():
    try:
        app = _APP_INSTANCE
        if app is None:
            return
        try: app._theme_btn.lower()
        except Exception: pass
        try: app._music_btn.lower()
        except Exception: pass
        app._corner_overlay_paused = True
    except Exception:
        pass

# Mengembalikan ikon pojok ke posisi teratas setelah popup ditutup agar UI utama tampil normal lagi.
def _corner_icons_restore():
    try:
        app = _APP_INSTANCE
        if app is None:
            return
        app._corner_overlay_paused = False
        app._raise_overlay()
    except Exception:
        pass

# SOUND EFFECTS ENGINE
_SFX_CORRECT     = None
_SFX_ERROR       = None
_SFX_WIN         = None
_SFX_ACHIEVEMENT = None
_SFX_SELECT      = None
_SFX_HOVER       = None
_SFX_CLICK       = None

# Membangun seluruh efek suara aplikasi secara prosedural memakai NumPy dan pygame lalu menyimpannya ke variabel global.
def _build_sfx():
    global _SFX_CORRECT, _SFX_ERROR, _SFX_WIN, _SFX_ACHIEVEMENT, \
           _SFX_SELECT, _SFX_HOVER, _SFX_CLICK
    if not PYGAME_AVAILABLE:
        return
    try:
        _np = np
        SR = 44100
        # Fungsi bantu ini memecah logika build sfx agar langkah kecilnya lebih rapi dan mudah dipakai ulang di bagian ini.
        def _make_sound(arr):
            arr = _np.clip(arr, -1.0, 1.0)
            s16 = (arr * 32767).astype(_np.int16)
            stereo = _np.column_stack([s16, s16])
            return pygame.sndarray.make_sound(stereo)
        # Membuat efek suara correct
        t = _np.linspace(0, 0.08, int(SR * 0.08), endpoint=False)
        _SFX_CORRECT = _make_sound(_np.sin(2 * _np.pi * 880 * t) * _np.exp(-t * 40) * 0.45)
        
        # Membuat efek suara error
        t = _np.linspace(0, 0.12, int(SR * 0.12), endpoint=False)
        saw = 2 * (t * 200 % 1) - 1
        noise = _np.random.uniform(-0.3, 0.3, len(t))
        _SFX_ERROR = _make_sound((saw * 0.5 + noise * 0.15) * _np.exp(-t * 25) * 0.55)
        
        # Membuat efek suara win
        chunks = []
        for freq in [523.25, 659.25, 783.99]:
            t = _np.linspace(0, 0.15, int(SR * 0.15), endpoint=False)
            env = _np.exp(-t * 6)
            chunks.append((_np.sin(2*_np.pi*freq*t)*0.5 + _np.sin(2*_np.pi*freq*2*t)*0.2)*env*0.6)
        _SFX_WIN = _make_sound(_np.concatenate(chunks))
        
        # Membuat efek suara achievement
        ach_chunks = []
        for freq in [392.00, 493.88, 587.33, 783.99]:
            t_a = _np.linspace(0, 0.06, int(SR * 0.06), endpoint=False)
            env_a = _np.exp(-t_a * 10)
            wave  = (_np.sin(2*_np.pi*freq*t_a) * 0.55
                   + _np.sin(2*_np.pi*freq*2*t_a) * 0.20
                   + _np.sin(2*_np.pi*freq*3*t_a) * 0.08) * env_a
            ach_chunks.append(wave)
        t_s = _np.linspace(0, 0.12, int(SR * 0.12), endpoint=False)
        env_s = _np.exp(-t_s * 5)
        shimmer = (_np.sin(2*_np.pi*1046.5*t_s)    * 0.45
                 + _np.sin(2*_np.pi*1318.5*t_s)    * 0.25
                 + _np.sin(2*_np.pi*1567.98*t_s)   * 0.15) * env_s
        ach_chunks.append(shimmer)
        _SFX_ACHIEVEMENT = _make_sound(_np.concatenate(ach_chunks) * 0.7)
        
        # Membuat efek suara select
        sel_chunks = []
        for freq in [523.25, 659.25]:
            t_sel = _np.linspace(0, 0.06, int(SR * 0.06), endpoint=False)
            env_sel = _np.exp(-t_sel * 18)
            wave_sel = (_np.sin(2 * _np.pi * freq * t_sel) * 0.55
                      + _np.sin(2 * _np.pi * freq * 2 * t_sel) * 0.18) * env_sel
            sel_chunks.append(wave_sel)
        _SFX_SELECT = _make_sound(_np.concatenate(sel_chunks) * 0.75)
        
        # Membuat efek suara hover
        t_hov = _np.linspace(0, 0.028, int(SR * 0.028), endpoint=False)
        env_hov = _np.exp(-t_hov * 120)
        _SFX_HOVER = _make_sound(_np.sin(2 * _np.pi * 1200 * t_hov) * env_hov * 0.18)

        # Membuat efek suara click
        t_clk = _np.linspace(0, 0.055, int(SR * 0.055), endpoint=False)
        env_clk = _np.exp(-t_clk * 55)
        _SFX_CLICK = _make_sound(
            (_np.sin(2 * _np.pi * 440 * t_clk) * 0.5
           + _np.sin(2 * _np.pi * 880 * t_clk) * 0.12) * env_clk * 0.65)
    except Exception:
        pass

# Memutar objek pygame.Sound secara non-blocking tanpa menghentikan alur utama aplikasi.
def _play_sfx(sound_obj):
    if not PYGAME_AVAILABLE or sound_obj is None:
        return
    try:
        sound_obj.play()
    except Exception:
        pass

# =============================================================================
# BAGIAN 4: DESIGN TOKENS (WARNA, FONT, TEMA)
# =============================================================================

# =============================================================================
# BAGIAN 4A: IDENTITAS APLIKASI
# =============================================================================
APP_NAME    = "Sudoku AI"
APP_VERSION = "1.0.0"

# =============================================================================
# BAGIAN 4B: DEMO MODE
# =============================================================================

DEMO_MODE = False # Saat DEMO_MODE = True, semua fungsi ML mengembalikan nilai fallback yang aman tanpa memanggil model sklearn

DEMO_DIFFICULTY   = "Normal"        # fallback difficulty recommendation
DEMO_PLAYER_TYPE  = "Learner"       # fallback player classification
DEMO_SCORE_PRED   = 750             # fallback predicted next score
DEMO_ANOMALY_FLAG = False           # fallback anomaly status (False = normal)

# ACHIEVEMENT ENGINE
# [DEMO-POINT] Dictionary 15+ achievement dengan kriteria unik
ACHIEVEMENTS = {
    # TIER 1: First Steps
    "pemula_berhasil":     {"id":"pemula_berhasil",     "nama":"First Win",          "emoji":"WIN",
                            "desc":"Selesaikan puzzle pertamamu!",                  "warna":"#7EE787"},
    "maraton":             {"id":"maraton",             "nama":"Marathon",            "emoji":"MAR",
                            "desc":"Selesaikan 10 sesi total.",                     "warna":"#F0883E"},
    "veteran":             {"id":"veteran",             "nama":"Veteran",            "emoji":"VET",
                            "desc":"Selesaikan 25 sesi total.",                     "warna":"#FF7EDB"},
    # TIER 2: Streak & Consistency
    "konsisten":           {"id":"konsisten",           "nama":"Consistent",         "emoji":"3X",
                            "desc":"Selesaikan 3 sesi berturut-turut.",             "warna":"#FF7B7B"},
    "tak_terkalahkan":     {"id":"tak_terkalahkan",     "nama":"Unbeatable",         "emoji":"5X",
                            "desc":"Selesaikan 5 sesi berturut-turut.",             "warna":"#BC8CFF"},
    "serial_winner":       {"id":"serial_winner",       "nama":"Serial Winner",      "emoji":"7X",
                            "desc":"Selesaikan 7 sesi berturut-turut.",             "warna":"#FF7EDB"},
    "comeback":            {"id":"comeback",            "nama":"Comeback",           "emoji":"BCK",
                            "desc":"Selesaikan setelah sesi sebelumnya gagal.",     "warna":"#F0883E"},
    # TIER 3: Speed
    "kilat":               {"id":"kilat",               "nama":"Lightning",          "emoji":"60s",
                            "desc":"Selesaikan 4x4 di bawah 60 detik.",            "warna":"#FFD700"},
    "cepat_kilat":         {"id":"cepat_kilat",         "nama":"Speed Flash",        "emoji":"30s",
                            "desc":"Selesaikan 4x4 di bawah 30 detik.",            "warna":"#FFD700"},
    "speed_demon":         {"id":"speed_demon",         "nama":"Speed Demon",        "emoji":"5M",
                            "desc":"Selesaikan 9x9 di bawah 5 menit.",             "warna":"#58A6FF"},
    # TIER 4: Accuracy & Independence
    "tanpa_petunjuk":      {"id":"tanpa_petunjuk",      "nama":"No Hints",           "emoji":"NH",
                            "desc":"Selesaikan puzzle tanpa satu pun hint.",        "warna":"#58A6FF"},
    "tanpa_cela":          {"id":"tanpa_cela",          "nama":"Flawless",           "emoji":"FL",
                            "desc":"Selesaikan puzzle tanpa satu pun error.",       "warna":"#BC8CFF"},
    "sempurna":            {"id":"sempurna",            "nama":"Perfect",            "emoji":"PF",
                            "desc":"Selesaikan tanpa error dan tanpa hint.",        "warna":"#FFD700"},
    "efisien":             {"id":"efisien",             "nama":"Efficient",          "emoji":"EF",
                            "desc":"Selesaikan dengan kurang dari 3 error.",        "warna":"#7EE787"},
    # TIER 5: Difficulty & Grid
    "ahli_hard":           {"id":"ahli_hard",           "nama":"Hard Expert",        "emoji":"HRD",
                            "desc":"Selesaikan puzzle Hard.",                       "warna":"#FF7B7B"},
    "tanpa_menyerah_hard": {"id":"tanpa_menyerah_hard", "nama":"Iron Will",          "emoji":"IW",
                            "desc":"Selesaikan Hard tanpa memakai hint/auto.",      "warna":"#FF7B7B"},
    "master_9x9":          {"id":"master_9x9",          "nama":"Master 9x9",         "emoji":"9x9",
                            "desc":"Selesaikan puzzle 9x9.",                        "warna":"#BC8CFF"},
    "explorer":            {"id":"explorer",            "nama":"Explorer",           "emoji":"EXP",
                            "desc":"Mainkan semua 3 tingkat kesulitan.",            "warna":"#58A6FF"},
    # TIER 6: Score
    "jenius":              {"id":"jenius",              "nama":"Genius",             "emoji":"800",
                            "desc":"Raih skor di atas 800 dalam satu sesi.",        "warna":"#FFD700"},
    "pakar":               {"id":"pakar",               "nama":"Expert",             "emoji":"500",
                            "desc":"Raih skor di atas 500 di mode Hard.",           "warna":"#FF7B7B"},
}

# [DEMO-POINT] Engine evaluasi pencapaian berbasis statistik sesi
# Memeriksa sesi permainan yang baru selesai, lalu membuka badge baru jika kriteria pencapaiannya terpenuhi.
def _evaluate_achievements(username, session, all_sessions):
    if not session.get("completed", False):
        return []
    data     = load_data()
    p        = data["players"].setdefault(username, {})
    unlocked = set(p.get("achievements", []))
    earned   = []

    done   = [s for s in all_sessions if s.get("completed", False)]
    diff   = session.get("difficulty", "Normal")
    gs     = session.get("grid_size", 3)
    t_sec  = session.get("total_time", 9999)
    hints  = session.get("hints_used", 0)
    auto   = session.get("auto_used", 0)
    errors = session.get("errors", 0)
    score  = int(session.get("score", 0) or 0)

    # Fungsi bantu ini memecah logika evaluate achievements agar langkah kecilnya lebih rapi dan mudah dipakai ulang di bagian ini.
    def _earn(aid):
        if aid not in unlocked:
            earned.append(aid)

    if len(done) >= 1:  _earn("pemula_berhasil")
    if len(done) >= 10: _earn("maraton")
    if len(done) >= 25: _earn("veteran")

    if len(done) >= 3 and all(s.get("completed") for s in done[-3:]):
        _earn("konsisten")
    if len(done) >= 5 and all(s.get("completed") for s in done[-5:]):
        _earn("tak_terkalahkan")
    if len(done) >= 7 and all(s.get("completed") for s in done[-7:]):
        _earn("serial_winner")
    cur_fp = _session_fingerprint(session)
    prev_sessions = [s for s in all_sessions if _session_fingerprint(s) != cur_fp]
    if prev_sessions and not prev_sessions[-1].get("completed", False):
        _earn("comeback")

    if gs == 2 and t_sec < 60:  _earn("kilat")
    if gs == 2 and t_sec < 30:  _earn("cepat_kilat")
    if gs == 3 and t_sec < 300: _earn("speed_demon")

    if hints == 0 and auto == 0: _earn("tanpa_petunjuk")
    if errors == 0:              _earn("tanpa_cela")
    if hints == 0 and auto == 0 and errors == 0: _earn("sempurna")
    if errors < 3:               _earn("efisien")

    if diff == "Hard":                         _earn("ahli_hard")
    if diff == "Hard" and hints == 0 and auto == 0: _earn("tanpa_menyerah_hard")
    if gs == 3:                                _earn("master_9x9")
    all_diffs = {s.get("difficulty", "Normal") for s in all_sessions}
    if {"Easy", "Normal", "Hard"}.issubset(all_diffs): _earn("explorer")

    if score > 800:                _earn("jenius")
    if diff == "Hard" and score > 500: _earn("pakar")

    if earned:
        unlocked.update(earned)
        p["achievements"] = list(unlocked)
        save_data(data)
    return earned

# =============================================================================
# BAGIAN 2: KONSTANTA PATH DAN ASET
# =============================================================================

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
ASSET_DIR   = os.path.join(BASE_DIR, "Assets")
MODEL_DIR   = os.path.join(BASE_DIR, "Models", "Files")
CARD_DIR    = os.path.join(BASE_DIR, "Card")

os.makedirs(ASSET_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(CARD_DIR,  exist_ok=True)

IMAGE_LOGO       = os.path.join(ASSET_DIR, "logo.png")
AUDIO_MUSIC      = os.path.join(ASSET_DIR, "music.mp3")
AUDIO_EASTER_EGG = os.path.join(ASSET_DIR, "easter_egg.mp3")
VIDEO_EASTER_EGG = os.path.join(ASSET_DIR, "easter_egg.mp4")
FILE_PLAYER_DATA = os.path.join(BASE_DIR,  "player_data.json")

_PKL_DIR    = MODEL_DIR
MUSIC_FILE  = AUDIO_MUSIC
DATA_FILE   = FILE_PLAYER_DATA

# =============================================================================
# BAGIAN 3: KONSTANTA MODEL ML
# =============================================================================

MODEL_CLASSIFIER = os.path.join(MODEL_DIR, "Player_Classifier.pkl")
MODEL_SCORE      = os.path.join(MODEL_DIR, "Score_Prediction.pkl")
MODEL_ANOMALY    = os.path.join(MODEL_DIR, "Detect_Anomaly.pkl")
MODEL_DIFFICULTY = os.path.join(MODEL_DIR, "Difficulty_Recommender.pkl")
MODEL_PERFORMANCE = os.path.join(MODEL_DIR, "Performance_Prediction.pkl")
MODEL_HINT_TIMER = os.path.join(MODEL_DIR, "Hint_Timer.pkl")

PKL_CLF     = "Player_Classifier.pkl"        # KNeighborsClassifier + StandardScaler
PKL_SCORE   = "Score_Prediction.pkl"         # HistGradientBoostingRegressor (prediksi skor)
PKL_ANO     = "Detect_Anomaly.pkl"           # IsolationForest + StandardScaler
PKL_DIFF    = "Difficulty_Recommender.pkl"   # RandomForestClassifier (rekomendasi difficulty)
PKL_PERFORM = "Performance_Prediction.pkl"   # MultiOutputRegressor + HistGBR (profil skill)
PKL_HINT    = "Hint_Timer.pkl"               # GradientBoostingRegressor (timer hint adaptif)

# -- Data storage -------------------------------------------------------------

# Membaca data pemain dari file JSON dan mengembalikan struktur kosong jika file belum tersedia atau gagal dibaca.
def load_data():
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"players": {}}

# Menyimpan data pemain ke file JSON secara atomic supaya file lama tetap aman jika proses berhenti di tengah jalan.
def save_data(data):
    tmp = DATA_FILE + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, DATA_FILE)
    except Exception:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise

# Menormalkan username dengan menghapus spasi, mengubah huruf menjadi kecil, lalu menggabungkannya untuk perbandingan yang konsisten.
def _normalize_username(username):
    return "".join(str(username).strip().lower().split())

# =============================================================================
# PKL MODEL HELPERS
# =============================================================================

# Menyusun path lengkap file model .pkl dari folder Models berdasarkan nama yang diminta.
def _pkl_path(name: str) -> str:
    return os.path.join(_PKL_DIR, name if name.endswith(".pkl") else f"{name}.pkl")

# Cache PKL di RAM - load sekali dari disk, akses berikutnya instan dari memori.
# Ini menghilangkan disk I/O berulang saat user ganti-ganti pemain di PlayerSelectScreen.
_PKL_CACHE: dict = {}

# Memuat objek model dari file pkl dan menyimpannya ke cache RAM agar akses berikutnya lebih cepat.
def _load_pkl(name: str):
    if name in _PKL_CACHE:
        return _PKL_CACHE[name]
    path = _pkl_path(name)
    try:
        if os.path.exists(path):
            with open(path, "rb") as f:
                obj = pickle.load(f)
            _PKL_CACHE[name] = obj
            return obj
    except Exception:
        pass
    return None

# [DEMO-POINT] PKL cache warming - preload semua model ke RAM
# Memuat lebih dulu semua model .pkl ke cache RAM saat startup supaya prediksi berikutnya tidak perlu I/O disk.
def _warmup_pkl_cache():
    for name in (PKL_CLF, PKL_SCORE, PKL_ANO, PKL_DIFF, PKL_PERFORM, PKL_HINT):
        try:
            _load_pkl(name)
        except Exception:
            pass

# Memuat semua model ML ke dictionary pada saat startup dan memberi nilai None jika ada file yang gagal dibuka.
def load_all_models() -> dict:
    model_paths = {
        "anomaly":     MODEL_ANOMALY,
        "difficulty":  MODEL_DIFFICULTY,
        "hint_timer":  MODEL_HINT_TIMER,
        "performance": MODEL_PERFORMANCE,
        "classifier":  MODEL_CLASSIFIER,
        "score":       MODEL_SCORE,
    }
    models = {}
    for name, path in model_paths.items():
        try:
            import pickle as _pkl
            with open(path, "rb") as _f:
                models[name] = _pkl.load(_f)
        except Exception as e:
            print(f"  [WARN] Gagal load model '{name}': {e}")
            models[name] = None
    return models

# Menampilkan status komponen aplikasi saat startup, termasuk mode, library opsional, dan keberadaan file model.
def startup_check() -> None:
    w = 52
    print("=" * w)
    print(f"  {APP_NAME}  v{APP_VERSION}")
    if DEMO_MODE:
        print("⚠ DEMO MODE ACTIVE")
    print(f"  sklearn : {'Available' if SKLEARN_AVAILABLE else 'Not Available'}")
    print(f"  pygame  : {'Available' if PYGAME_AVAILABLE else 'Not Available'}")
    print(f"  PIL     : {'Available' if PIL_AVAILABLE else 'Not Available'}")
    print(f"  cv2     : {'Available' if CV2_AVAILABLE else 'Not Available'}")
    print("-" * w)
    model_map = {
        "Player Classifier Model":      PKL_CLF,
        "Score Prediction Model":       PKL_SCORE,
        "Anomaly Detection Model":      PKL_ANO,
        "Difficulty Recommender Model": PKL_DIFF,
        "Performance Prediction Model": PKL_PERFORM,
        "Hint Timer Model":             PKL_HINT,
    }

    for label, pkl_name in model_map.items():
        path = _pkl_path(pkl_name)
        status = "READY" if os.path.exists(path) else "MISSING"
        status_teks = f"[{status}]"
        print(f"{status_teks.ljust(9)} {label}")
    print("=" * w)

# Menyimpan objek ke file pkl secara atomic lalu memperbarui cache RAM agar data lama tidak stale.
def _save_pkl(name: str, obj) -> bool:
    path = _pkl_path(name)
    tmp  = path + ".tmp"
    try:
        with open(tmp, "wb") as f:
            pickle.dump(obj, f)
        os.replace(tmp, path)
        _PKL_CACHE[name] = obj
        return True
    except Exception:
        try:
            os.remove(tmp)
        except OSError:
            pass
        return False

# Mencari username yang sudah terdaftar dengan membandingkan versi yang sudah dinormalisasi.
def _find_existing_username(data, username):
    target = _normalize_username(username)
    for existing in (data or {}).get("players", {}):
        if _normalize_username(existing) == target:
            return existing
    return None

# DESIGN TOKENS
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


# THEME SYSTEM  (Dark ↔ Light)
_DARK_THEME = {
    "C_BG":        "#0D1117",
    "C_SURFACE":   "#161B22",
    "C_SURFACE2":  "#1F2937",
    "C_BORDER":    "#30363D",
    "C_ACCENT":    "#58A6FF",
    "C_ACCENT2":   "#7EE787",
    "C_WARN":      "#F0883E",
    "C_ERROR":     "#FF7B7B",
    "C_PURPLE":    "#BC8CFF",
    "C_PINK":      "#FF7EDB",
    "C_GOLD":      "#FFD700",
    "C_TEXT":      "#E6EDF3",
    "C_TEXT_DIM":  "#8B949E",
    "C_WHITE":     "#FFFFFF",
    "C_CROSS_ROW": "#1A2535",
    "C_CROSS_COL": "#1A2535",
    "C_CROSS_BOX": "#1B2B1B",
    "C_SELECTED":  "#2A4A7A",
    "C_SAME_NUM":  "#2A3A5A",
    "ANIM_COLS":   ["#1A2840","#1A3028","#2A1A30","#1C2010","#201C10"],
    "GRADIENT":    ["#BC8CFF", "#58A6FF", "#7EE787", "#F0883E"],
    # Sidebar & special buttons
    "C_SIDEBAR":           "#0A0E14",
    "C_SIDEBAR_PURPLE_BG": "#2D1A4A",
    "C_SIDEBAR_RED_BG":    "#2A0808",
    "C_NUMPAD_DEL_BG":     "#3D1515",
    "C_NUMPAD_DEL_HOV":    "#5A1E1E",
    "C_NUMPAD_DIS_BG":     "#1A1A1A",
    "C_NUMPAD_DIS_FG":     "#444444",
    # Mini-grid preview
    "C_MINI_BLK1":  "#0F1F35",
    "C_MINI_BLK2":  "#0A1525",
    "C_MINI_LINE":  "#1E2D40",
    "C_MINI_CELL":  "#132030",
    "C_SIDEBAR_ACTIVE":   "#12182A",
    "C_SIDEBAR_SELECTED": "#131B2E",
    "C_SIDEBAR_HOVER":    "#0F1827",
    "C_SIDEBAR_SEP":      "#0F1520",
}

_LIGHT_THEME = {
    "C_BG":        "#F0F4FA",
    "C_SURFACE":   "#FFFFFF",
    "C_SURFACE2":  "#E8EDF5",
    "C_BORDER":    "#C5CFE0",
    "C_ACCENT":    "#0969DA",
    "C_ACCENT2":   "#1A7F37",
    "C_WARN":      "#BC4C00",
    "C_ERROR":     "#CF222E",
    "C_PURPLE":    "#8250DF",
    "C_PINK":      "#BF3989",
    "C_GOLD":      "#9A6700",
    "C_TEXT":      "#1C2330",
    "C_TEXT_DIM":  "#57606A",
    "C_WHITE":     "#FFFFFF",
    "C_CROSS_ROW": "#DBEAFE",
    "C_CROSS_COL": "#DBEAFE",
    "C_CROSS_BOX": "#DCFCE7",
    "C_SELECTED":  "#BFDBFE",
    "C_SAME_NUM":  "#BBF7D0",
    "ANIM_COLS":   ["#C8D9F0","#C8F0DC","#DCC8F0","#F0E0C8","#F0C8DC"],
    "GRADIENT":    ["#8250DF", "#0969DA", "#1A7F37", "#BC4C00"],
    # Sidebar & special buttons
    "C_SIDEBAR":           "#E2E8F4",
    "C_SIDEBAR_PURPLE_BG": "#EDE9FE",
    "C_SIDEBAR_RED_BG":    "#FFE4E6",
    "C_NUMPAD_DEL_BG":     "#FFE4E6",
    "C_NUMPAD_DEL_HOV":    "#FECDD3",
    "C_NUMPAD_DIS_BG":     "#E5E7EB",
    "C_NUMPAD_DIS_FG":     "#9CA3AF",
    # Mini-grid preview
    "C_MINI_BLK1":  "#DBEAFE",
    "C_MINI_BLK2":  "#EFF6FF",
    "C_MINI_LINE":  "#BFDBFE",
    "C_MINI_CELL":  "#E0EDFF",
    "C_SIDEBAR_ACTIVE":   "#D4DBEC",
    "C_SIDEBAR_SELECTED": "#C8D3E8",
    "C_SIDEBAR_HOVER":    "#D0D8EE",
    "C_SIDEBAR_SEP":      "#C5CFE0",
}

_CURRENT_THEME_NAME = "dark"

# Mengembalikan nama tema aktif yang sedang dipakai aplikasi.
def get_theme_name():
    return _CURRENT_THEME_NAME

# New theme-aware globals (set by apply_theme)
C_SIDEBAR           = "#0A0E14"
C_SIDEBAR_PURPLE_BG = "#2D1A4A"
C_SIDEBAR_RED_BG    = "#2A0808"
C_NUMPAD_DEL_BG     = "#3D1515"
C_NUMPAD_DEL_HOV    = "#5A1E1E"
C_NUMPAD_DIS_BG     = "#1A1A1A"
C_NUMPAD_DIS_FG     = "#444444"
C_MINI_BLK1         = "#0F1F35"
C_MINI_BLK2         = "#0A1525"
C_MINI_LINE         = "#1E2D40"
C_MINI_CELL         = "#132030"
C_SIDEBAR_ACTIVE    = "#12182A"
C_SIDEBAR_SELECTED  = "#131B2E"
C_SIDEBAR_HOVER     = "#0F1827"
C_SIDEBAR_SEP       = "#0F1520"

# Menerapkan tema yang dipilih ke semua konstanta warna global agar tampilan seluruh layar ikut berubah.
def apply_theme(name: str):
    global _CURRENT_THEME_NAME
    global C_BG, C_SURFACE, C_SURFACE2, C_BORDER, C_ACCENT, C_ACCENT2
    global C_WARN, C_ERROR, C_PURPLE, C_PINK, C_GOLD, C_TEXT, C_TEXT_DIM, C_WHITE
    global C_SIDEBAR, C_SIDEBAR_PURPLE_BG, C_SIDEBAR_RED_BG
    global C_NUMPAD_DEL_BG, C_NUMPAD_DEL_HOV, C_NUMPAD_DIS_BG, C_NUMPAD_DIS_FG
    global C_MINI_BLK1, C_MINI_BLK2, C_MINI_LINE, C_MINI_CELL
    global C_SIDEBAR_ACTIVE, C_SIDEBAR_SELECTED, C_SIDEBAR_HOVER, C_SIDEBAR_SEP
    global _GRADIENT_COLORS
    _CURRENT_THEME_NAME = name
    t = _DARK_THEME if name == "dark" else _LIGHT_THEME
    C_BG        = t["C_BG"]
    C_SURFACE   = t["C_SURFACE"]
    C_SURFACE2  = t["C_SURFACE2"]
    C_BORDER    = t["C_BORDER"]
    C_ACCENT    = t["C_ACCENT"]
    C_ACCENT2   = t["C_ACCENT2"]
    C_WARN      = t["C_WARN"]
    C_ERROR     = t["C_ERROR"]
    C_PURPLE    = t["C_PURPLE"]
    C_PINK      = t["C_PINK"]
    C_GOLD      = t["C_GOLD"]
    C_TEXT      = t["C_TEXT"]
    C_TEXT_DIM  = t["C_TEXT_DIM"]
    C_WHITE     = t["C_WHITE"]
    C_SIDEBAR           = t["C_SIDEBAR"]
    C_SIDEBAR_PURPLE_BG = t["C_SIDEBAR_PURPLE_BG"]
    C_SIDEBAR_RED_BG    = t["C_SIDEBAR_RED_BG"]
    C_NUMPAD_DEL_BG     = t["C_NUMPAD_DEL_BG"]
    C_NUMPAD_DEL_HOV    = t["C_NUMPAD_DEL_HOV"]
    C_NUMPAD_DIS_BG     = t["C_NUMPAD_DIS_BG"]
    C_NUMPAD_DIS_FG     = t["C_NUMPAD_DIS_FG"]
    C_MINI_BLK1         = t["C_MINI_BLK1"]
    C_MINI_BLK2         = t["C_MINI_BLK2"]
    C_MINI_LINE         = t["C_MINI_LINE"]
    C_MINI_CELL         = t["C_MINI_CELL"]
    C_SIDEBAR_ACTIVE    = t["C_SIDEBAR_ACTIVE"]
    C_SIDEBAR_SELECTED  = t["C_SIDEBAR_SELECTED"]
    C_SIDEBAR_HOVER     = t["C_SIDEBAR_HOVER"]
    C_SIDEBAR_SEP       = t["C_SIDEBAR_SEP"]
    _GRADIENT_COLORS    = t["GRADIENT"]

DIFF_THEMES = {
    # Mode Easy
    "Easy": {
        "accent":         "#7EE787",
        "accent2":        "#2EA043",
        "cell_bg":        "#0D2818",
        "cell_fixed_bg":  "#143321",
        "cell_user_bg":   "#1E5235",
        "cell_fixed_fg":  "#7EE787",
        "cell_user_fg":   "#A8F0B0",
        "highlight":      "#1F6B3A",
        "hover":          "#153825",
        "error_bg":       "#3B1212",
        "error_fg":       "#FF7B7B",
        "grid_line":      "#2EA043",
        "remove_pct":     0.35,
        "emoji":          "🍃",
        # Highlight warna adaptif
        "hl_box":         "#112C1E",
        "hl_rowcol":      "#174F2E",
        "hl_same_bg":     "#1B6B38",
        "hl_same_fg_fix": "#B8FFD0",
        "hl_same_fg_usr": "#D0FFE0",
        "hl_sel_border":  "#7EE787",
        "error_indicator": "#FF6B6B",
    },
    # Mode Normal
    "Normal": {
        "accent":         "#58A6FF",
        "accent2":        "#1F6FEB",
        "cell_bg":        "#0D1A2E",
        "cell_fixed_bg":  "#112244",
        "cell_user_bg":   "#1C3C6E",
        "cell_fixed_fg":  "#79C0FF",
        "cell_user_fg":   "#B0D4FF",
        "highlight":      "#1C4880",
        "hover":          "#142D47",
        "error_bg":       "#3B1212",
        "error_fg":       "#FF7B7B",
        "grid_line":      "#1F6FEB",
        "remove_pct":     0.50,
        "emoji":          "⚡",
        # Highlight warna adaptif
        "hl_box":         "#102244",
        "hl_rowcol":      "#163460",
        "hl_same_bg":     "#1A4A8C",
        "hl_same_fg_fix": "#A8D4FF",
        "hl_same_fg_usr": "#C8E4FF",
        "hl_sel_border":  "#58A6FF",
        "error_indicator": "#FF6B6B",
    },
    # Mode Hard
    "Hard": {
        "accent":         "#FF7B7B",
        "accent2":        "#DA3633",
        "cell_bg":        "#2D0D0D",
        "cell_fixed_bg":  "#3D1515",
        "cell_user_bg":   "#7E3030",
        "cell_fixed_fg":  "#FF7B7B",
        "cell_user_fg":   "#FFAAAA",
        "highlight":      "#6B2020",
        "hover":          "#421515",
        "error_bg":       "#3A1E00",
        "error_fg":       "#FF8C00",
        "grid_line":      "#DA3633",
        "remove_pct":     0.65,
        "emoji":          "🔥",
        # Highlight warna adaptif
        "hl_box":         "#3E1414",
        "hl_rowcol":      "#562020",
        "hl_same_bg":     "#7A2020",
        "hl_same_fg_fix": "#FFB8B8",
        "hl_same_fg_usr": "#FFD0D0",
        "hl_sel_border":  "#FF7B7B",
        "error_indicator": "#FF8C00",
    },
}

DIFF_THEMES_LIGHT = {
    # Mode Easy
    "Easy": {
        "accent":         "#1A7F37",
        "accent2":        "#0D5622",
        "cell_bg":        "#F0FFF4",
        "cell_fixed_bg":  "#DCFCE7",
        "cell_user_bg":   "#96E0B4",
        "cell_fixed_fg":  "#15803D",
        "cell_user_fg":   "#166534",
        "highlight":      "#BBF7D0",
        "hover":          "#D1FAE5",
        "error_bg":       "#FFE4E6",
        "error_fg":       "#CF222E",
        "grid_line":      "#1A7F37",
        "remove_pct":     0.35,
        "emoji":          "🍃",
        "hl_box":         "#C8F5DC",
        "hl_rowcol":      "#9DE8BA",
        "hl_same_bg":     "#6EE7B7",
        "hl_same_fg_fix": "#064E3B",
        "hl_same_fg_usr": "#065F46",
        "hl_sel_border":  "#1A7F37",
        "error_indicator": "#CF222E",
    },
    # Mode Normal
    "Normal": {
        "accent":         "#0969DA",
        "accent2":        "#0550AE",
        "cell_bg":        "#EFF6FF",
        "cell_fixed_bg":  "#DBEAFE",
        "cell_user_bg":   "#A8C4F5",
        "cell_fixed_fg":  "#1D4ED8",
        "cell_user_fg":   "#1E40AF",
        "highlight":      "#BFDBFE",
        "hover":          "#E0ECFF",
        "error_bg":       "#FFE4E6",
        "error_fg":       "#CF222E",
        "grid_line":      "#0969DA",
        "remove_pct":     0.50,
        "emoji":          "⚡",
        "hl_box":         "#CDE5FF",
        "hl_rowcol":      "#A8C8FA",
        "hl_same_bg":     "#93C5FD",
        "hl_same_fg_fix": "#1E3A8A",
        "hl_same_fg_usr": "#1E40AF",
        "hl_sel_border":  "#0969DA",
        "error_indicator": "#CF222E",
    },
    # Mode Hard
    "Hard": {
        "accent":         "#CF222E",
        "accent2":        "#A40E26",
        "cell_bg":        "#FFF5F5",
        "cell_fixed_bg":  "#FFE4E6",
        "cell_user_bg":   "#EDAABB",
        "cell_fixed_fg":  "#BE123C",
        "cell_user_fg":   "#9F1239",
        "highlight":      "#FECDD3",
        "hover":          "#FFE4E6",
        "error_bg":       "#FFE8CC",
        "error_fg":       "#CC5500",
        "grid_line":      "#CF222E",
        "remove_pct":     0.65,
        "emoji":          "🔥",
        "hl_box":         "#FFD6DA",
        "hl_rowcol":      "#F8A8B4",
        "hl_same_bg":     "#FDA4AF",
        "hl_same_fg_fix": "#881337",
        "hl_same_fg_usr": "#9F1239",
        "hl_sel_border":  "#CF222E",
        "error_indicator": "#CC5500",
    },
}

# Mengambil konfigurasi warna yang sesuai untuk difficulty tertentu berdasarkan tema aktif.
def get_diff_theme(difficulty: str) -> dict:
    themes = DIFF_THEMES if _CURRENT_THEME_NAME == "dark" else DIFF_THEMES_LIGHT
    return themes.get(difficulty, themes["Normal"])

FONT_BTN    = ("Segoe UI", 10, "bold")
FONT_BTN_SM = ("Segoe UI", 9, "bold")
FONT_SMALL  = ("Segoe UI", 9)
FONT_BODY   = ("Segoe UI", 11)
FONT_TIMER  = ("Consolas", 22, "bold")

# SHARED UI HELPERS
_GRADIENT_COLORS = ["#BC8CFF", "#58A6FF", "#7EE787", "#F0883E"]

# Menggambar bar gradasi horizontal multi-stop pada Canvas dengan lebar yang disesuaikan widget aktif.
def draw_gradient_bar(canvas, colors=None, height=8):
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

# SUDOKU LOGIC
# [DEMO-POINT] Formula skor multi-variabel: waktu, error, hint, difficulty
# Menghitung skor akhir berdasarkan difficulty, waktu, error, hint, dan aksi tambahan lain yang memengaruhi performa.
def calculate_score(difficulty, total_time, empty_cells, errors,
                    hints_used, completed, near_miss=0, guessing=0, auto_used=0):
    if not completed:
        return 0
    
    diff_mult            = {"Easy": 1.0, "Normal": 1.8, "Hard": 3.0}.get(difficulty, 1.0)
    N                    = max(empty_cells, 1)
    player_cells         = max(1, N - hints_used)
    time_per_player_cell = total_time / player_cells
    time_score           = max(0.0, 1000.0 - time_per_player_cell * 10.0)
    error_rate           = errors / player_cells
    error_penalty        = min(350, int(error_rate * 500))
    behavior_penalty     = guessing * 25 + near_miss * 8
    hint_penalty         = hints_used * 200
    auto_penalty         = auto_used * 50
    raw                  = time_score - error_penalty - behavior_penalty - hint_penalty - auto_penalty
    return max(0, int(raw * diff_mult))

# Membuat identitas stabil untuk satu sesi permainan supaya duplikasi data bisa dicegah.
def _session_fingerprint(session):
    if not isinstance(session, dict):
        return None
    return (
        session.get("username"),
        session.get("timestamp"),
        session.get("difficulty"),
        session.get("grid_size"),
        round(float(session.get("total_time", 0) or 0), 3),
        int(session.get("moves", 0) or 0),
        int(session.get("errors", 0) or 0),
        int(session.get("hints_used", 0) or 0),
        int(session.get("auto_used", 0) or 0),
        bool(session.get("completed", False)),
        int(session.get("score", 0) or 0),
    )

# Menghapus sesi duplikat berdasarkan fingerprint sambil mempertahankan urutan data yang asli.
def _dedupe_sessions(sessions):
    seen = set()
    unique = []
    for session in sessions or []:
        fp = _session_fingerprint(session)
        if fp is None or fp not in seen:
            if fp is not None:
                seen.add(fp)
            unique.append(session)
    return unique

# =============================================================================
# BAGIAN 5: ALGORITMA SUDOKU INTI
# =============================================================================

# [DEMO-POINT] Pencarian sel kosong - titik awal backtracking MRV
# Mencari sel kosong pertama pada papan Sudoku dengan urutan row-major.
def find_empty(board, N):
    for i in range(N):
        for j in range(N):
            if board[i][j] == 0:
                return (i, j)
    return None

# [DEMO-POINT] Validator aturan Sudoku: baris / kolom / kotak
# Memeriksa apakah angka tertentu boleh ditempatkan pada posisi yang dituju sesuai aturan Sudoku.
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

# [DEMO-POINT] MRV heuristic - kandidat nilai per sel
# Mengambil semua angka valid untuk satu sel berdasarkan kondisi baris, kolom, dan kotak kecil.
def get_candidates(board, r, c, N, BOX):
    return [n for n in range(1, N+1) if is_valid(board, n, (r, c), N, BOX)]

# [DEMO-POINT] Generator papan valid penuh (backtracking acak)
# Membangkitkan papan Sudoku penuh yang valid menggunakan backtracking acak.
def generate_full_board(N, BOX):
    board = [[0]*N for _ in range(N)]
    # Fungsi bantu ini memecah logika generate full board agar langkah kecilnya lebih rapi dan mudah dipakai ulang di bagian ini.
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

# [DEMO-POINT] Puzzle generation dengan kontrol tingkat kesulitan
# Membuat puzzle Sudoku dari papan penuh dengan menghapus sebagian sel sesuai tingkat kesulitan.
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

# OPTIMAL BACKTRACKING: MRV + Forward Checking
# Ini adalah versi paling optimal dari backtracking:
# - MRV (Minimum Remaining Values): pilih sel dengan
#   kandidat paling sedikit terlebih dahulu
# - Forward Checking: cek apakah sel lain masih punya
#   kandidat valid setelah penempatan angka
# - Early termination jika ada sel dengan 0 kandidat
# [DEMO-POINT] Solver Sudoku - Backtracking + MRV + Forward Checking
# Menyelesaikan Sudoku dengan backtracking, MRV, dan forward checking supaya pencarian lebih efisien.
def solve_backtracking_mrv(start_board, N, BOX):
    expanded = [0]
    start = time.time()
    hist = []
    working = [row[:] for row in start_board]

    # Fungsi bantu ini memecah logika solve backtracking mrv agar langkah kecilnya lebih rapi dan mudah dipakai ulang di bagian ini.
    def find_mrv_cell(b):
        best_pos = None
        best_count = N + 1
        for r in range(N):
            for c in range(N):
                if b[r][c] == 0:
                    cands = get_candidates(b, r, c, N, BOX)
                    if len(cands) == 0:
                        return None, []
                    if len(cands) < best_count:
                        best_count = len(cands)
                        best_pos = (r, c)
                        best_cands = cands
                        if best_count == 1:
                            return best_pos, best_cands
        return best_pos, best_cands if best_pos else []

    # Fungsi bantu ini memecah logika solve backtracking mrv agar langkah kecilnya lebih rapi dan mudah dipakai ulang di bagian ini.
    def bt(b):
        expanded[0] += 1
        pos, cands = find_mrv_cell(b)
        if pos is None:
            if find_empty(b, N) is None:
                return True
            return False
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

# MACHINE LEARNING MODULE
# Dataset sintetis untuk melatih KNN Classifier.
# Setiap baris: [avg_time_per_cell, error_rate, hint_rate,
#                completion_rate, near_miss_rate, guessing_rate]
# Label: indeks PLAYER_TYPES_ORDER
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

# =============================================================================
# BAGIAN 6: SISTEM AI / ML
# =============================================================================

# [DEMO-POINT] Engine ML utama - 6 model terintegrasi
class PlayerMLEngine:
    """
    ML Engine v2 - scikit-learn + Rule-Based Fallback
    ──────────────────────────────────────────────────
    Fitur ML nyata (aktif jika sklearn tersedia):
    • KNeighborsClassifier          - klasifikasi tipe pemain + confidence %  (Player_Classifier.pkl)
    • HistGradientBoostingRegressor - prediksi skor sesi berikutnya  (Score_Prediction.pkl)
    • IsolationForest               - deteksi sesi anomali (threshold dari notebook)  (Detect_Anomaly.pkl)
    • RandomForestClassifier        - rekomendasi difficulty  (Difficulty_Recommender.pkl)
    • MultiOutputRegressor+HistGBR  - profil skill 11 dimensi  (Performance_Prediction.pkl)
    • GradientBoostingRegressor     - threshold hint timer adaptif  (Hint_Timer.pkl)
    • StandardScaler                - normalisasi fitur (KNN, ISO, RFC, Multi - bukan HistGBR)

    Jika sklearn tidak tersedia, fallback ke rule-based scoring.
    """
    PLAYER_TYPES = {
        "Speedrunner":  {"color": "#FFD700", "emoji": "⚡", "desc": "Cepat & akurat"},
        "Careful":      {"color": "#7EE787", "emoji": "🧩", "desc": "Hati-hati & teliti"},
        "Learner":      {"color": "#58A6FF", "emoji": "📚", "desc": "Sedang belajar"},
        "Struggling":   {"color": "#FF7B7B", "emoji": "💪", "desc": "Butuh bantuan"},
        "Inconsistent": {"color": "#F0883E", "emoji": "🎲", "desc": "Tidak konsisten"},
    }

    # Menginisialisasi objek PlayerMLEngine dan menyiapkan state awal, referensi penting, serta elemen yang dibutuhkan sebelum layar dipakai.
    def __init__(self):
        self.sessions       = []
        self.adaptive_idle  = 15.0
        self._train_lock    = threading.Lock()
        self._knn           = None
        self._knn_scaler    = None
        self._rfr           = None
        self._iso           = None
        self._iso_scaler    = None
        self._iso_threshold = 0.0
        self._models_dirty  = True
        if SKLEARN_AVAILABLE:
            self._pretrain_knn()

    # Pre-training KNN dengan data sintetis
    # [DEMO-POINT] Pre-training KNN dari data sintetis 200 sampel
    # Menangani proses pretrain knn pada PlayerMLEngine sambil menjaga state internal tetap konsisten.
    def _pretrain_knn(self):
        try:
            cached = _load_pkl(PKL_CLF)
            if cached is not None:
                self._knn        = cached["model"]
                self._knn_scaler = cached["scaler"]
                meta = cached.get("meta", {})
                self._knn_best_k       = int(meta.get("n_neighbors", 3))
                self._knn_best_metric  = str(meta.get("metric", "euclidean"))
                self._knn_best_weights = str(meta.get("weights", "uniform"))
                _cv_acc = float(
                    meta.get("cv_f1_macro",
                             meta.get("cv_accuracy", 0.0))
                )
                if _cv_acc <= 0.0:
                    _cv_acc = float(cached.get("accuracy", 0.0))
                if "cv_f1_macro" not in meta and _cv_acc > 0.0:
                    meta["cv_f1_macro"] = _cv_acc
                return
            X = np.array(_KNN_SYNTHETIC_X, dtype=float)
            y = np.array(_KNN_SYNTHETIC_Y, dtype=int)
            scaler = StandardScaler()
            X_sc   = scaler.fit_transform(X)
            knn    = KNeighborsClassifier(n_neighbors=3, metric="euclidean")
            knn.fit(X_sc, y)
            self._knn        = knn
            self._knn_scaler = scaler
            self._knn_best_k       = 3
            self._knn_best_metric  = "euclidean"
            self._knn_best_weights = "uniform"
            _save_pkl(PKL_CLF, {
                "model":    knn,
                "scaler":   scaler,
                "accuracy": 0.0,
                "X_syn":    X.tolist(),
                "y_syn":    y.tolist(),
                "meta": {
                    "n_neighbors": 3,
                    "metric":      "euclidean",
                    "weights":     "uniform",
                    "cv_f1_macro": 0.0,
                    "n_train":     len(X),
                    "source":      "pretrain_fallback",
                },
            })
        except Exception:
            pass

    # perbarui semua model ketika ada data sesi nyata
    # [DEMO-POINT] Training KNN + HistGBR + IsolationForest dari sesi nyata
    # Menangani proses train models pada PlayerMLEngine sambil menjaga state internal tetap konsisten.
    def _train_models(self):
        if not SKLEARN_AVAILABLE or not self._models_dirty:
            return
        if not self._train_lock.acquire(blocking=False):
            return
        self._models_dirty = False
        n = len(self.sessions)
        try:
            if n == 0:
                cached = _load_pkl(PKL_CLF)
                if cached is not None:
                    self._knn        = cached["model"]
                    self._knn_scaler = cached["scaler"]
            else:
                try:
                    from sklearn.model_selection import cross_val_score, StratifiedKFold
                    _syn_base = _load_pkl(PKL_CLF)
                    if (
                        _syn_base is not None
                        and "X_syn" in _syn_base
                        and "y_syn" in _syn_base
                    ):
                        X_syn = np.array(_syn_base["X_syn"], dtype=float)
                        y_syn = np.array(_syn_base["y_syn"], dtype=int)
                    else:
                        X_syn = np.array(_KNN_SYNTHETIC_X, dtype=float)
                        y_syn = np.array(_KNN_SYNTHETIC_Y, dtype=int)

                    _meta = (_syn_base or {}).get("meta", {})
                    _best_k       = int(_meta.get("n_neighbors",
                                       getattr(self, "_knn_best_k", 3)))
                    _best_metric  = str(_meta.get("metric",
                                       getattr(self, "_knn_best_metric", "euclidean")))
                    _best_weights = str(_meta.get("weights",
                                       getattr(self, "_knn_best_weights", "uniform")))
                    _notebook_cv_f1 = float(
                        _meta.get("cv_f1_macro",
                                  _meta.get("cv_accuracy", 0.0))
                    )

                    all_sessions = list(self.sessions)
                    _n_actual = len(all_sessions)
                    try:
                        _all_data = load_data()
                        for _payload in _all_data.get("players", {}).values():
                            for _s in _payload.get("sessions", []):
                                _fp = _session_fingerprint(_s)
                                if not any(_session_fingerprint(x) == _fp for x in self.sessions):
                                    all_sessions.append(_s)
                    except Exception:
                        pass

                    if _n_actual < 20 and _syn_base is not None:
                        self._knn        = _syn_base["model"]
                        self._knn_scaler = _syn_base["scaler"]
                    else:
                        feats_actual = [self._session_to_vector(s) for s in all_sessions]
                        X_act = np.array(feats_actual, dtype=float)
                        y_act = np.array(
                            [_PLAYER_TYPES_ORDER.index(self._rule_based_type(s))
                             for s in all_sessions], dtype=int)
                        X_all = np.vstack([X_syn, X_act])
                        y_all = np.concatenate([y_syn, y_act])

                        scaler = StandardScaler()
                        X_sc   = scaler.fit_transform(X_all)
                        k = min(_best_k, len(X_all))
                        knn = KNeighborsClassifier(
                            n_neighbors=k,
                            metric=_best_metric,
                            weights=_best_weights,
                        )
                        knn.fit(X_sc, y_all)

                        _n_splits = min(5, len(np.unique(y_all)),
                                        min(np.bincount(y_all)))
                        _n_splits = max(2, _n_splits)
                        _cv_scores = cross_val_score(
                            knn, X_sc, y_all,
                            cv=StratifiedKFold(n_splits=_n_splits, shuffle=True,
                                               random_state=42),
                            scoring="f1_macro",
                        )
                        new_cv_f1 = float(_cv_scores.mean())

                        _meta_ref_f1 = float(
                            _meta.get("cv_f1_macro",
                                      _meta.get("cv_accuracy", 0.0))
                        )
                        _ref_f1 = _meta_ref_f1 if _meta_ref_f1 > 0 else (
                            float(_syn_base["accuracy"]) if _syn_base and "accuracy" in _syn_base else 0.0
                        )
                        _save_new_knn = new_cv_f1 >= _ref_f1 + 0.005

                        self._knn        = knn
                        self._knn_scaler = scaler
                        if _save_new_knn:
                            _save_pkl(PKL_CLF, {
                                "model":    knn,
                                "scaler":   scaler,
                                "accuracy": new_cv_f1,
                                "X_syn":    X_syn.tolist(),
                                "y_syn":    y_syn.tolist(),
                                "meta": {
                                    "n_neighbors":  k,
                                    "metric":       _best_metric,
                                    "weights":      _best_weights,
                                    "cv_f1_macro":  new_cv_f1,
                                    "n_train":      len(X_all),
                                    "n_actual":     _n_actual,
                                },
                            })
                except Exception:
                    pass

            if n >= 3:
                cached_rfr = _load_pkl(PKL_SCORE)
                if cached_rfr is not None and n < 15:
                    self._rfr = cached_rfr["model"]
                else:
                    try:
                        X_rfr, y_rfr = [], []
                        for i, s in enumerate(self.sessions):
                            mv  = max(s.get("moves", 1), 1)
                            tpc = s.get("time_per_cell",
                                        s.get("total_time", 0) / max(s.get("empty_cells", mv), 1))
                            er  = s.get("errors", 0)     / mv
                            hr  = s.get("hints_used", 0) / mv
                            sc  = s.get("score", 0) or 0
                            X_rfr.append([i, tpc, er, hr])
                            y_rfr.append(sc)
                        X_rfr = np.array(X_rfr, dtype=float)
                        y_rfr = np.array(y_rfr, dtype=float)
                        rfr = HistGradientBoostingRegressor(
                            max_iter=200,
                            learning_rate=0.10,
                            min_samples_leaf=20,
                            random_state=42,
                        )
                        rfr.fit(X_rfr, y_rfr)

                        from sklearn.model_selection import cross_val_score, KFold
                        _kf_rfr = KFold(n_splits=min(5, len(X_rfr)), shuffle=True, random_state=42)
                        _cv_rfr = cross_val_score(rfr, X_rfr, y_rfr, cv=_kf_rfr, scoring="r2")
                        new_oob = float(_cv_rfr.mean())
                        _old_ref_r2 = cached_rfr.get("r2", 0.0) if cached_rfr else 0.0
                        _save_new_rfr = new_oob > _old_ref_r2
                        self._rfr = rfr
                        if _save_new_rfr:
                            _save_pkl(PKL_SCORE, {
                                "model": rfr,
                                "r2":    new_oob,
                                "meta": {
                                    "model_type":    "HistGradientBoostingRegressor",
                                    "needs_scaler":  False,
                                    "feature_names": ["session_idx", "time_per_cell",
                                                      "error_rate", "hint_rate"],
                                    "n_train":       len(X_rfr),
                                    "score_type":    "cv_r2_mean",
                                },
                            })
                    except Exception:
                        self._rfr = None

            if n >= 5:
                cached_iso = _load_pkl(PKL_ANO)
                if cached_iso is not None:
                    self._iso           = cached_iso["model"]
                    self._iso_scaler    = cached_iso["scaler"]
                    self._iso_threshold = float(cached_iso.get("optimal_threshold", 0.0))
                else:
                    try:
                        X_iso = np.array(
                            [self._session_to_vector(s) for s in self.sessions],
                            dtype=float)
                        scaler_iso = StandardScaler()
                        X_iso_sc   = scaler_iso.fit_transform(X_iso)
                        iso        = IsolationForest(
                            contamination=0.05, random_state=42, n_estimators=100)
                        iso.fit(X_iso_sc)
                        self._iso           = iso
                        self._iso_scaler    = scaler_iso
                        self._iso_threshold = 0.0
                    except Exception:
                        self._iso           = None
                        self._iso_scaler    = None
                        self._iso_threshold = 0.0
        finally:
            self._train_lock.release()

    # Menangani proses session to vector pada PlayerMLEngine sambil menjaga state internal tetap konsisten.
    def _session_to_vector(self, s):
        mv   = max(s.get("moves", 1), 1)
        tpc  = s.get("total_time", 0) / mv
        er   = s.get("errors", 0)    / mv
        hr   = s.get("hints_used", 0) / mv
        cr   = 1.0 if s.get("completed", False) else 0.0
        total_err = max(s.get("errors", 0), 1)
        nmr  = s.get("near_miss", 0)  / total_err
        gur  = s.get("guessing", 0)   / total_err
        return [tpc, er, hr, cr, nmr, gur]

    # Menangani proses rule based type pada PlayerMLEngine sambil menjaga state internal tetap konsisten.
    def _rule_based_type(self, s):
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

    # Menangani proses add session pada PlayerMLEngine sambil menjaga state internal tetap konsisten.
    def add_session(self, s):
        self.sessions.append(s)
        self._models_dirty = True
        self._update_thresholds()

    # Memperbarui thresholds pada PlayerMLEngine agar data, status, dan tampilan tetap selaras.
    def _update_thresholds(self):
        if len(self.sessions) >= 2:
            avg = sum(s.get("total_time", 60) for s in self.sessions) / len(self.sessions)
            self.adaptive_idle = max(10.0, min(30.0, avg * 0.20))

    # Menangani proses extract features pada PlayerMLEngine sambil menjaga state internal tetap konsisten.
    def extract_features(self):
        if not self.sessions:
            return {"avg_time_per_cell": 0, "error_rate": 0,
                    "hint_rate": 0, "completion_rate": 0,
                    "avg_moves": 0, "sessions_count": 0,
                    "near_miss_rate": 0, "guessing_rate": 0,
                    "avg_time_per_empty_cell": 0, "auto_rate": 0}
        n = len(self.sessions)
        tpc  = sum(s.get("total_time", 0) / max(s.get("moves", 1), 1) for s in self.sessions) / n
        er   = sum(s.get("errors", 0) / max(s.get("moves", 1), 1) for s in self.sessions) / n
        hr   = sum(s.get("hints_used", 0) / max(s.get("moves", 1), 1) for s in self.sessions) / n
        cr   = sum(1 for s in self.sessions if s.get("completed", False)) / n
        am   = sum(s.get("moves", 0) for s in self.sessions) / n
        total_err = sum(s.get("errors", 0) for s in self.sessions) or 1
        nmr  = sum(s.get("near_miss", 0) for s in self.sessions) / total_err
        gur  = sum(s.get("guessing", 0) for s in self.sessions) / total_err
        ar   = sum(s.get("auto_used", 0) for s in self.sessions) / max(am * n, 1)
        tpec = sum(s.get("time_per_cell", s.get("total_time",0)/max(s.get("moves",1),1))
                   for s in self.sessions) / n
        return {"avg_time_per_cell": tpc, "error_rate": er,
                "hint_rate": hr, "completion_rate": cr,
                "avg_moves": am, "sessions_count": n,
                "near_miss_rate": nmr, "guessing_rate": gur,
                "avg_time_per_empty_cell": tpec, "auto_rate": ar}

    # [DEMO-POINT] Klasifikasi tipe pemain via KNN (rule-based fallback)
    # Menangani proses classify player pada PlayerMLEngine sambil menjaga state internal tetap konsisten.
    def classify_player(self):
        feat = self.extract_features()
        if DEMO_MODE:
            return DEMO_PLAYER_TYPE, feat
        if feat["sessions_count"] == 0:
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
                pass

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

    # [DEMO-POINT] Klasifikasi tipe pemain + confidence % (0-100)
    # Menangani proses classify player confidence pada PlayerMLEngine sambil menjaga state internal tetap konsisten.
    def classify_player_confidence(self):
        if DEMO_MODE:
            return DEMO_PLAYER_TYPE, 85.0, {}
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

    # [DEMO-POINT] Prediksi skor sesi berikutnya (HistGradientBoostingRegressor)
    # Menangani proses predict next score pada PlayerMLEngine sambil menjaga state internal tetap konsisten.
    def predict_next_score(self):
        if DEMO_MODE:
            return DEMO_SCORE_PRED, True
        if not SKLEARN_AVAILABLE or len(self.sessions) < 3:
            return None, False
        try:
            self._train_models()
            if self._rfr is None:
                cached = _load_pkl(PKL_SCORE)
                if cached is not None:
                    self._rfr = cached["model"]
                else:
                    return None, False
            n      = len(self.sessions)
            recent = self.sessions[-3:]
            tpc_avg = sum(
                s.get("time_per_cell",
                      s.get("total_time", 0) / max(s.get("empty_cells",
                                                          max(s.get("moves", 1), 1)), 1))
                for s in recent) / 3
            er_avg  = sum(s.get("errors", 0) / max(s.get("moves", 1), 1)
                          for s in recent) / 3
            hr_avg  = sum(s.get("hints_used", 0) / max(s.get("moves", 1), 1)
                          for s in recent) / 3
            X_next = np.array([[n, tpc_avg * 0.95, er_avg * 0.95, hr_avg * 0.95]])
            pred   = max(0, int(self._rfr.predict(X_next)[0]))
            return pred, True
        except Exception:
            return None, False

    # [DEMO-POINT] Deteksi sesi anomali via IsolationForest
    # Menangani proses detect anomaly pada PlayerMLEngine sambil menjaga state internal tetap konsisten.
    def detect_anomaly(self):
        if DEMO_MODE:
            status = "normal" if not DEMO_ANOMALY_FLAG else "anomaly"
            return status, ""
        if not SKLEARN_AVAILABLE or len(self.sessions) < 5:
            return "unknown", "Data belum cukup (min 5 sesi)"
        try:
            self._train_models()
            if self._iso is None or self._iso_scaler is None:
                return "unknown", "Model belum siap"
            last_vec  = np.array([self._session_to_vector(self.sessions[-1])],
                                dtype=float)
            last_sc   = self._iso_scaler.transform(last_vec)

            score     = float(-self._iso.decision_function(last_sc)[0])
            threshold = getattr(self, "_iso_threshold", 0.0)

            if score >= threshold:
                margin = score - threshold
                if margin >= 0.05:
                    reason = "Sesi ini sangat berbeda dari biasanya"
                else:
                    reason = "Performa sedikit di luar pola normal"
                return "anomaly", reason
            else:
                return "normal", "Sesi ini konsisten dengan pola kamu"
        except Exception:
            return "unknown", "Tidak dapat dianalisis"

    # [DEMO-POINT] Rekomendasi difficulty (RandomForestClassifier)
    # Menangani proses recommend difficulty pada PlayerMLEngine sambil menjaga state internal tetap konsisten.
    def recommend_difficulty(self):
        feat = self.extract_features()
        if DEMO_MODE:
            return DEMO_DIFFICULTY
        if feat["sessions_count"] == 0:
            return "Easy"
        skill = (max(0,100-feat["avg_time_per_cell"]*8)*0.35 +
                 max(0,100-feat["error_rate"]*250)*0.35 +
                 feat["completion_rate"]*100*0.30)
        last_done = self.sessions[-1].get("completed", False) if self.sessions else False
        if skill >= 70 and last_done: return "Hard"
        if skill >= 40: return "Normal"
        return "Easy"

    # [DEMO-POINT] Hint timer adaptif (GradientBoostingRegressor)
    # Menangani proses should give hint pada PlayerMLEngine sambil menjaga state internal tetap konsisten.
    def should_give_hint(self, idle, errors, moves):
        if idle > self.adaptive_idle: return True, "idle"
        if moves > 5 and errors / max(moves, 1) > 0.4: return True, "errors"
        return False, None

    # Menangani proses get summary pada PlayerMLEngine sambil menjaga state internal tetap konsisten.
    def get_summary(self):
        pt, conf, feat = self.classify_player_confidence()
        pred_score, pred_avail = self.predict_next_score()
        anom_status, anom_reason = self.detect_anomaly()
        return {
            "player_type":             pt,
            "features":                feat,
            "recommended_difficulty":  self.recommend_difficulty(),
            "type_info":               self.PLAYER_TYPES.get(pt, {}),
            "ml_confidence":           conf,
            "predicted_next_score":    pred_score,
            "predicted_score_avail":   pred_avail,
            "anomaly_status":          anom_status,
            "anomaly_reason":          anom_reason,
            "sklearn_active":          SKLEARN_AVAILABLE,
        }

# =============================================================================
# ML PATCH
# =============================================================================

# Mengonversi label difficulty menjadi angka agar bisa dipakai oleh model ML.
def _ml_diff_to_int(diff):
    return {"Easy": 0, "Normal": 1, "Hard": 2}.get(diff, 1)

# Mengonversi angka hasil model kembali ke label difficulty yang mudah dibaca.
def _ml_int_to_diff(idx):
    return ["Easy", "Normal", "Hard"][max(0, min(2, int(idx)))]

# [DEMO-POINT] Rule-based fallback klasifikasi tipe pemain
# Menerjemahkan metrik performa menjadi tipe pemain berbasis aturan sebelum prediksi model dipakai.
def _ml_player_type_from_metrics(tpc, er, hr, cr, nmr, gur):
    if tpc <= 4 and er < 0.05 and hr < 0.05:
        return "Speedrunner"
    if tpc >= 14 and er < 0.10:
        return "Careful"
    if er > 0.30 or hr > 0.35:
        return "Struggling"
    if er > 0.15 or hr > 0.18:
        return "Learner"
    if cr < 0.55 and (nmr > 0.50 or gur > 0.50):
        return "Inconsistent"
    return "Inconsistent"

# Menyusun vektor fitur dari satu sesi permainan agar mudah diproses model ML.
def _ml_session_vector(session):
    mv = max(session.get("moves", 1), 1)
    tpc = session.get("time_per_cell",
                      session.get("total_time", 0.0) / max(session.get("empty_cells", mv), 1))
    er  = session.get("errors", 0.0)      / mv
    hr  = session.get("hints_used", 0.0)  / mv
    cr  = 1.0 if session.get("completed", False) else 0.0
    total_err = max(session.get("errors", 0), 1)
    nmr = session.get("near_miss", 0.0)  / total_err
    gur = session.get("guessing", 0.0)   / total_err
    sc  = float(session.get("score", 0) or 0)
    return [float(tpc), float(er), float(hr), float(cr),
            float(nmr), float(gur), float(mv), sc]

# Menggabungkan fitur beberapa sesi menjadi ringkasan statistik untuk analisis tingkat lanjut.
def _ml_aggregate_vector(sessions):
    if not sessions:
        return [0.0] * 8
    n = len(sessions)
    tpc = sum(
        s.get("time_per_cell",
              s.get("total_time", 0.0) / max(s.get("empty_cells",
                                                     max(s.get("moves", 1), 1)), 1))
        for s in sessions) / n
    er       = sum(s.get("errors", 0.0)     / max(s.get("moves", 1), 1) for s in sessions) / n
    hr       = sum(s.get("hints_used", 0.0) / max(s.get("moves", 1), 1) for s in sessions) / n
    cr       = sum(1.0 for s in sessions if s.get("completed", False)) / n
    total_err = sum(s.get("errors", 0) for s in sessions) or 1
    nmr      = sum(s.get("near_miss", 0) for s in sessions) / total_err
    gur      = sum(s.get("guessing", 0)  for s in sessions) / total_err
    avg_moves = sum(s.get("moves", 0)    for s in sessions) / n
    avg_score = sum(float(s.get("score", 0) or 0) for s in sessions) / n
    return [float(tpc), float(er), float(hr), float(cr),
            float(nmr), float(gur), float(avg_moves), float(avg_score)]

# Menyusun target label pelatihan dari data sesi agar model supervised bisa dilatih.
def _ml_targets(session):
    mv = max(session.get("moves", 1), 1)
    tpc = session.get("time_per_cell",
                      session.get("total_time", 0.0) / max(session.get("empty_cells", mv), 1))
    er = session.get("errors", 0.0) / mv
    hr = session.get("hints_used", 0.0) / mv
    cr = 1.0 if session.get("completed", False) else 0.0
    total_err = max(session.get("errors", 0), 1)
    nmr = session.get("near_miss", 0.0) / total_err
    gur = session.get("guessing", 0.0) / total_err
    sc = float(session.get("score", 0) or 0)
    return [
        max(0.2, float(tpc)),
        max(0.0, float(er)),
        max(0.0, float(hr)),
        max(0.0, min(1.0, float(cr))),
        max(0.0, float(nmr)),
        max(0.0, float(gur)),
        max(0.0, float(sc)),
        max(0.0, min(100.0, 100.0 - float(tpc) * 8.0)),
        max(0.0, min(100.0, 100.0 - float(er) * 250.0)),
        max(0.0, min(100.0, float(cr) * 100.0)),
        max(0.0, min(100.0, 100.0 - float(hr) * 120.0)),
    ]

# Mengumpulkan seluruh sesi permainan dari data pemain untuk dipakai sebagai dataset ML.
def _ml_all_sessions():
    data = load_data()
    out = []
    for payload in data.get("players", {}).values():
        sessions = sorted(payload.get("sessions", []), key=lambda s: s.get("timestamp", 0))
        if sessions:
            out.append(sessions)
    return out

# pertahankan perilaku asli PlayerMLEngine dan tambahkan model ML extended
_orig_pmle_init = PlayerMLEngine.__init__
_orig_pmle_add_session = PlayerMLEngine.add_session
_orig_pmle_extract = PlayerMLEngine.extract_features
_orig_pmle_classify_conf = PlayerMLEngine.classify_player_confidence
_orig_pmle_predict_next_score = PlayerMLEngine.predict_next_score
_orig_pmle_detect_anomaly = PlayerMLEngine.detect_anomaly

# Menyiapkan engine ML, memuat model yang tersedia, dan membangun fallback saat file model tidak lengkap.
def _ml_init(self):
    _orig_pmle_init(self)
    self._rec_model     = None
    self._rec_scaler    = None
    self._stats_model   = None
    self._stats_scaler  = None
    self._iso_threshold = 0.0
    self._ml_dirty      = True
    self._hint_rfr      = None
    self._hint_scaler   = None
    self._knn_best_k       = 3
    self._knn_best_metric  = "euclidean"
    self._knn_best_weights = "uniform"
    if SKLEARN_AVAILABLE:
        _rfr_pkg = _load_pkl(PKL_SCORE)
        if _rfr_pkg is not None:
            self._rfr = _rfr_pkg["model"]
        _hint_pkg = _load_pkl(PKL_HINT)
        if _hint_pkg is not None:
            self._hint_rfr    = _hint_pkg["model"]
            self._hint_scaler = _hint_pkg.get("scaler", None)
    self._train_ml_models(force=False)

# Menambahkan sesi baru ke dataset ML lalu memperbarui statistik turunan yang dibutuhkan model.
def _ml_add_session(self, s):
    _orig_pmle_add_session(self, s)
    self._ml_dirty = True

# [DEMO-POINT] Training RFC + MultiOutputRegressor (11 dimensi skill)
# Melatih model ML utama dari data sesi yang tersedia lalu memperbarui cache model internal.
def _train_ml_models(self, force=False):
    if not SKLEARN_AVAILABLE:
        return
    if not force and not getattr(self, "_ml_dirty", True):
        return
    self._ml_dirty = False

    if not force:
        cached_gbm = _load_pkl(PKL_DIFF)
        if cached_gbm is not None:
            self._rec_model  = cached_gbm["model"]
            self._rec_scaler = cached_gbm["scaler"]
        cached_multi = _load_pkl(PKL_PERFORM)
        if cached_multi is not None:
            self._stats_model  = cached_multi["model"]
            self._stats_scaler = cached_multi["scaler"]
        if self._rec_model is not None and self._stats_model is not None:
            return

    _real_sess_count = sum(len(s) for s in _ml_all_sessions())

    _cached_gbm_pre = _load_pkl(PKL_DIFF)
    _notebook_gbm_acc = (_cached_gbm_pre or {}).get(
        "f1_score",
        (_cached_gbm_pre or {}).get("accuracy", 0.0)
    )
    _protect_gbm = (
        _cached_gbm_pre is not None
        and _real_sess_count < 30
    )
    if _protect_gbm:
        self._rec_model  = _cached_gbm_pre["model"]
        self._rec_scaler = _cached_gbm_pre["scaler"]
    else:
        X_rec, y_rec = [], []
        for sessions in _ml_all_sessions():
            if len(sessions) < 2:
                continue
            for i in range(1, len(sessions)):
                prefix = sessions[:i]
                X_rec.append(_ml_aggregate_vector(prefix))
                y_rec.append(_ml_diff_to_int(sessions[i].get("difficulty", "Normal")))
        if len(X_rec) < 8:
            rng = np.random.default_rng(19)
            for _ in range(800):
                tpc = float(rng.uniform(1.0, 30.0))
                er = float(rng.uniform(0.0, 0.60))
                hr = float(rng.uniform(0.0, 0.60))
                cr = float(rng.uniform(0.0, 1.0))
                nmr = float(rng.uniform(0.0, 1.0))
                gur = float(rng.uniform(0.0, 1.0))
                avg_moves = float(rng.uniform(4.0, 60.0))
                avg_score = float(rng.uniform(0.0, 1000.0))
                skill = (max(0.0, 100.0 - tpc * 7.0) * 0.32 +
                         max(0.0, 100.0 - er * 220.0) * 0.28 +
                         cr * 100.0 * 0.22 +
                         max(0.0, 100.0 - hr * 140.0) * 0.10 +
                         max(0.0, 100.0 - gur * 110.0) * 0.08)
                if skill >= 72:
                    label = 2
                elif skill >= 42:
                    label = 1
                else:
                    label = 0
                X_rec.append([tpc, er, hr, cr, nmr, gur, avg_moves, avg_score])
                y_rec.append(label)
        X_rec = np.array(X_rec, dtype=float)
        y_rec = np.array(y_rec, dtype=int)
        rec_scaler = StandardScaler()
        X_rec_sc = rec_scaler.fit_transform(X_rec)
        from sklearn.model_selection import cross_val_score, StratifiedKFold
        rec_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            random_state=42,
            class_weight="balanced_subsample",
        )
        rec_model.fit(X_rec_sc, y_rec)

        _n_cv = min(5, min(np.bincount(y_rec)), len(np.unique(y_rec)))
        _n_cv = max(2, _n_cv)
        _cv_f1 = cross_val_score(
            rec_model, X_rec_sc, y_rec,
            cv=StratifiedKFold(n_splits=_n_cv, shuffle=True, random_state=42),
            scoring="f1_macro",
        )
        new_rfc_f1 = float(_cv_f1.mean())
        _save_new_rfc = new_rfc_f1 > _notebook_gbm_acc
        if _save_new_rfc:
            self._rec_model  = rec_model
            self._rec_scaler = rec_scaler
            _save_pkl(PKL_DIFF, {
                "model":    rec_model,
                "scaler":   rec_scaler,
                "f1_score": new_rfc_f1,
                "meta": {
                    "n_train":    len(X_rec),
                    "n_actual":   _real_sess_count,
                    "score_type": "cv_f1_macro",
                },
            })
        else:
            if _cached_gbm_pre is not None:
                self._rec_model  = _cached_gbm_pre["model"]
                self._rec_scaler = _cached_gbm_pre["scaler"]
            else:
                self._rec_model  = rec_model
                self._rec_scaler = rec_scaler

    _cached_multi_pre = _load_pkl(PKL_PERFORM)
    _cached_r2_pre    = _cached_multi_pre.get("r2", 0) if _cached_multi_pre else 0
    _protect_multi = (
        _cached_multi_pre is not None
        and _cached_r2_pre >= 0.75
        and _real_sess_count < 100
    )

    if _protect_multi:
        self._stats_model  = _cached_multi_pre["model"]
        self._stats_scaler = _cached_multi_pre["scaler"]
    else:
        X_stats, Y_stats = [], []
        for sessions in _ml_all_sessions():
            for s in sessions:
                X_stats.append(_ml_session_vector(s))
                Y_stats.append(_ml_targets(s))
        if len(X_stats) < 8:
            X_stats, Y_stats = [], []
            rng = np.random.default_rng(7)
            for _ in range(800):
                tpc = float(rng.uniform(1.0, 30.0))
                er = float(rng.uniform(0.0, 0.60))
                hr = float(rng.uniform(0.0, 0.60))
                cr = float(rng.uniform(0.0, 1.0))
                nmr = float(rng.uniform(0.0, 1.0))
                gur = float(rng.uniform(0.0, 1.0))
                mv = float(rng.uniform(4.0, 60.0))
                sc = max(0.0, 1000.0 - tpc * 18.0 - er * 600.0 - hr * 500.0 + cr * 300.0)
                X_stats.append([tpc, er, hr, cr, nmr, gur, mv, sc])
                Y_stats.append([
                    max(0.2, tpc * rng.uniform(0.92, 1.08)),
                    max(0.0, er * rng.uniform(0.90, 1.10)),
                    max(0.0, hr * rng.uniform(0.90, 1.10)),
                    max(0.0, min(1.0, cr * rng.uniform(0.92, 1.05))),
                    max(0.0, nmr * rng.uniform(0.90, 1.12)),
                    max(0.0, gur * rng.uniform(0.90, 1.12)),
                    max(0.0, sc * rng.uniform(0.90, 1.08)),
                    max(0.0, min(100.0, 100.0 - tpc * 8.0 + rng.normal(0, 5))),
                    max(0.0, min(100.0, 100.0 - er * 250.0 + rng.normal(0, 5))),
                    max(0.0, min(100.0, cr * 100.0 + rng.normal(0, 4))),
                    max(0.0, min(100.0, 100.0 - hr * 120.0 + rng.normal(0, 5))),
                ])
        X_stats = np.array(X_stats, dtype=float)
        Y_stats = np.array(Y_stats, dtype=float)
        stats_scaler = StandardScaler()
        X_stats_sc = stats_scaler.fit_transform(X_stats)
        stats_model = MultiOutputRegressor(
            HistGradientBoostingRegressor(max_iter=200, random_state=42))
        stats_model.fit(X_stats_sc, Y_stats)

        try:
            from sklearn.model_selection import KFold
            _kf = KFold(n_splits=min(5, len(X_stats_sc)), shuffle=True, random_state=42)
            _cv_r2_scores = []
            for _tr, _val in _kf.split(X_stats_sc):
                _m = MultiOutputRegressor(
                    HistGradientBoostingRegressor(max_iter=100, random_state=42))
                _m.fit(X_stats_sc[_tr], Y_stats[_tr])
                _cv_r2_scores.append(_m.score(X_stats_sc[_val], Y_stats[_val]))
            new_multi_r2 = float(np.mean(_cv_r2_scores))
        except Exception:
            new_multi_r2 = float(stats_model.score(X_stats_sc, Y_stats)) * 0.80

        _save_new_multi = new_multi_r2 > _cached_r2_pre
        if _save_new_multi:
            self._stats_model  = stats_model
            self._stats_scaler = stats_scaler
            _save_pkl(PKL_PERFORM, {
                "model":  stats_model,
                "scaler": stats_scaler,
                "r2":     new_multi_r2,
                "meta": {
                    "n_train":    len(X_stats),
                    "n_actual":   _real_sess_count,
                    "score_type": "cv_r2_mean",
                },
            })
        else:
            if _cached_multi_pre is not None:
                self._stats_model  = _cached_multi_pre["model"]
                self._stats_scaler = _cached_multi_pre["scaler"]
            else:
                self._stats_model  = stats_model
                self._stats_scaler = stats_scaler

# Memprediksi difficulty yang paling cocok berdasarkan fitur sesi dan profil pemain.
def _ml_predict_difficulty(self):
    if DEMO_MODE:
        return DEMO_DIFFICULTY, 85.0
    try:
        if getattr(self, "_ml_dirty", True) or self._rec_model is None or self._rec_scaler is None:
            self._train_ml_models(force=False)
        feat = _ml_aggregate_vector(self.sessions)
        X = np.array([feat], dtype=float)
        X_sc = self._rec_scaler.transform(X)
        idx = int(self._rec_model.predict(X_sc)[0])
        proba = self._rec_model.predict_proba(X_sc)[0]
        return _ml_int_to_diff(idx), float(max(proba)) * 100.0
    except Exception:
        return None, 0.0

# Memprediksi profil performa pemain dari data sesi yang sudah dikumpulkan.
def _ml_predict_profile(self, session=None):
    if session is None:
        session = self.sessions[-1] if self.sessions else None
    if session is None:
        return {
            "expected_time_per_cell": 0.0,
            "expected_error_rate": 0.0,
            "expected_hint_rate": 0.0,
            "expected_completion_rate": 0.0,
            "expected_near_miss_rate": 0.0,
            "expected_guessing_rate": 0.0,
            "expected_score": 0.0,
            "speed_index": 0.0,
            "accuracy_index": 0.0,
            "consistency_index": 0.0,
            "independence_index": 0.0,
        }
    if getattr(self, "_ml_dirty", True) or self._stats_model is None or self._stats_scaler is None:
        self._train_ml_models(force=False)
    vec = np.array([_ml_session_vector(session)], dtype=float)
    try:
        vec_sc = self._stats_scaler.transform(vec)
        pred = self._stats_model.predict(vec_sc)[0]
        return {
            "expected_time_per_cell": float(max(0.2, pred[0])),
            "expected_error_rate": float(max(0.0, pred[1])),
            "expected_hint_rate": float(max(0.0, pred[2])),
            "expected_completion_rate": float(max(0.0, min(1.0, pred[3]))),
            "expected_near_miss_rate": float(max(0.0, pred[4])),
            "expected_guessing_rate": float(max(0.0, pred[5])),
            "expected_score": float(max(0.0, pred[6])),
            "speed_index": float(max(0.0, min(100.0, pred[7]))),
            "accuracy_index": float(max(0.0, min(100.0, pred[8]))),
            "consistency_index": float(max(0.0, min(100.0, pred[9]))),
            "independence_index": float(max(0.0, min(100.0, pred[10]))),
        }
    except Exception:
        mv  = max(session.get("moves", 1), 1)
        tpc = session.get("time_per_cell",
                          session.get("total_time", 0.0) / max(session.get("empty_cells", mv), 1))
        er  = session.get("errors", 0.0)     / mv
        hr  = session.get("hints_used", 0.0) / mv
        cr = 1.0 if session.get("completed", False) else 0.0
        total_err = max(session.get("errors", 0), 1)
        nmr = session.get("near_miss", 0.0) / total_err
        gur = session.get("guessing", 0.0) / total_err
        score = float(session.get("score", 0) or 0)
        return {
            "expected_time_per_cell": float(tpc),
            "expected_error_rate": float(er),
            "expected_hint_rate": float(hr),
            "expected_completion_rate": float(cr),
            "expected_near_miss_rate": float(nmr),
            "expected_guessing_rate": float(gur),
            "expected_score": float(score),
            "speed_index": float(max(0.0, min(100.0, 100.0 - tpc * 8.0))),
            "accuracy_index": float(max(0.0, min(100.0, 100.0 - er * 250.0))),
            "consistency_index": float(max(0.0, min(100.0, cr * 100.0))),
            "independence_index": float(max(0.0, min(100.0, 100.0 - hr * 120.0))),
        }

# Memberi rekomendasi tantangan berikutnya berdasarkan performa terakhir dan status pemain.
def _ml_recommend_next_challenge(self):
    latest = self.sessions[-1] if self.sessions else None
    if latest is None:
        return {
            "difficulty": "Easy",
            "grid_size": 2,
            "confidence": 0.0,
            "reason": "belum ada riwayat",
        }

    profile = _ml_predict_profile(self, latest)
    pred_diff, pred_conf = _ml_predict_difficulty(self)

    grid = int(latest.get("grid_size", 3) or 3)
    completed = bool(latest.get("completed", False))
    score = float(latest.get("score", 0) or 0)
    tpc = float(latest.get("time_per_cell", latest.get("total_time", 0.0) / max(latest.get("moves", 1), 1)))
    errors = int(latest.get("errors", 0))
    hints = int(latest.get("hints_used", 0))
    max_hearts = int(latest.get("max_hearts", grid * grid * grid))
    moves = max(int(latest.get("moves", 1)), 1)
    error_rate = errors / moves

    try:
        pt_name, _, _ = _orig_pmle_classify_conf(self)
    except Exception:
        pt_name = "Inconsistent"

    speed_index = float(profile.get("speed_index", 0.0))
    accuracy_index = float(profile.get("accuracy_index", 0.0))
    independence_index = float(profile.get("independence_index", 0.0))

    if grid <= 2:
        if completed and (
            score >= 180
            or speed_index >= 45.0
            or accuracy_index >= 50.0
            or (tpc <= 12.0 and error_rate <= 0.35 and hints <= max(1, moves // 6))
            or independence_index >= 45.0
        ):
            return {
                "difficulty": "Normal",
                "grid_size": 3,
                "confidence": max(pred_conf, 65.0),
                "reason": "lulus 2x2, naik ke 3x3",
            }

        if completed and score >= 90:
            return {
                "difficulty": "Normal",
                "grid_size": 3,
                "confidence": max(pred_conf, 55.0),
                "reason": "selesai baik, coba 3x3",
            }

        return {
            "difficulty": "Easy",
            "grid_size": 2,
            "confidence": max(pred_conf, 50.0),
            "reason": "perlu stabilisasi di 2x2",
        }

    hint_heavy = hints >= max(3, max_hearts // 2)
    if error_rate > 0.30 or hint_heavy or pt_name == "Struggling":
        if completed and score >= 100:
            return {
                "difficulty": "Normal",
                "grid_size": 3,
                "confidence": max(pred_conf, 60.0),
                "reason": "selesai, tapi error rate masih tinggi - kuasai Normal dulu",
            }
        elif completed:
            return {
                "difficulty": "Easy",
                "grid_size": 3,
                "confidence": max(pred_conf, 55.0),
                "reason": "banyak kesalahan, coba Easy 3×3 untuk membangun ritme",
            }
        else:
            return {
                "difficulty": "Easy",
                "grid_size": 2,
                "confidence": max(pred_conf, 55.0),
                "reason": "belum selesai dan banyak error - turun dulu ke 2×2",
            }

    if completed and (
        score >= 650
        and error_rate <= 0.15
        and hints <= max(1, moves // 8)
    ):
        return {
            "difficulty": "Hard",
            "grid_size": 3,
            "confidence": max(pred_conf, 70.0),
            "reason": "performa sangat baik, siap tantangan lebih berat",
        }

    if completed and pt_name in ("Speedrunner", "Careful") and (
        speed_index >= 60.0 or accuracy_index >= 60.0 or independence_index >= 60.0
    ):
        return {
            "difficulty": "Hard",
            "grid_size": 3,
            "confidence": max(pred_conf, 65.0),
            "reason": "siap tantangan yang lebih berat",
        }

    if (not completed) and (error_rate > 0.35 or hints >= max(2, moves // 4)):
        return {
            "difficulty": "Easy",
            "grid_size": 2,
            "confidence": max(pred_conf, 55.0),
            "reason": "turun sebentar untuk pemulihan",
        }

    return {
        "difficulty": "Normal",
        "grid_size": 3,
        "confidence": max(pred_conf, 50.0),
        "reason": "pertahankan ritme saat ini di level Normal",
    }

# Memberi rekomendasi tingkat kesulitan yang paling pas untuk pemain saat ini.
def _ml_recommend_difficulty(self):
    if DEMO_MODE:
        return DEMO_DIFFICULTY
    plan = _ml_recommend_next_challenge(self)
    return plan["difficulty"]

# Menyusun ringkasan hasil analisis ML dalam format yang mudah dibaca oleh layar dashboard.
def _ml_get_summary(self, session=None):
    pt, conf, _ = _orig_pmle_classify_conf(self)
    pred_score, pred_avail = _orig_pmle_predict_next_score(self)
    anom_status, anom_reason = _orig_pmle_detect_anomaly(self)
    profile = _ml_predict_profile(self, session)
    plan = _ml_recommend_next_challenge(self)
    raw_feat = _orig_pmle_extract(self)
    _rfr_meta = {}
    try:
        _rfr_pkg = _load_pkl(PKL_SCORE)
        if _rfr_pkg is not None:
            _rfr_meta = {
                "r2":           _rfr_pkg.get("r2", 0.0),
                "needs_scaler": _rfr_pkg.get("meta", {}).get("needs_scaler", False),
            }
    except Exception:
        pass
    features = {
        "avg_time_per_cell": profile["expected_time_per_cell"],
        "error_rate":        profile["expected_error_rate"],
        "hint_rate":         profile["expected_hint_rate"],
        "completion_rate":   profile["expected_completion_rate"],
        "avg_moves":         raw_feat.get("avg_moves", 0),
        "sessions_count":    raw_feat.get("sessions_count", 0),
        "near_miss_rate":    profile["expected_near_miss_rate"],
        "guessing_rate":     profile["expected_guessing_rate"],
        "avg_time_per_empty_cell": profile["expected_time_per_cell"],
    }
    return {
        "player_type":             pt,
        "features":                features,
        "raw_features":            raw_feat,
        "recommended_difficulty":  plan["difficulty"],
        "recommended_grid_size":   plan["grid_size"],
        "recommended_reason":      plan["reason"],
        "recommended_confidence":  plan["confidence"],
        "type_info":               self.PLAYER_TYPES.get(pt, {}),
        "ml_confidence":           conf,
        "predicted_next_score":    pred_score,
        "predicted_score_avail":   pred_avail,
        "anomaly_status":          anom_status,
        "anomaly_reason":          anom_reason,
        "sklearn_active":          SKLEARN_AVAILABLE,
        "ml_profile":              profile,
        "rfr_meta":                _rfr_meta,
    }

# Hint Timer: training
# [DEMO-POINT] Training GBR untuk prediksi timer hint adaptif
# Melatih model penentu waktu hint berdasarkan ritme permainan dan pola kesulitan sesi.
def _ml_train_hint_model(self):
    if not SKLEARN_AVAILABLE:
        return

    cached_pkg = _load_pkl(PKL_HINT)
    if cached_pkg is None:
        return

    all_player_sessions = _ml_all_sessions()
    flat = [s for player_sess in all_player_sessions for s in player_sess]

    if len(flat) < 10:
        if self._hint_rfr is None:
            self._hint_rfr    = cached_pkg["model"]
            self._hint_scaler = cached_pkg.get("scaler", None)
        return

    # Fungsi bantu ini memecah logika ml train hint model agar langkah kecilnya lebih rapi dan mudah dipakai ulang di bagian ini.
    def _hint_session_to_fv(s):
        moves     = max(s.get("moves", 1), 1)
        total_t   = float(s.get("total_time", 0) or 0)
        tpc       = float(s.get("time_per_cell", total_t / moves))
        er        = s.get("errors", 0)     / moves
        hr        = s.get("hints_used", 0) / moves
        cr        = 1.0 if s.get("completed", False) else 0.0
        total_err = max(s.get("errors", 0), 1)
        nmr       = s.get("near_miss", 0)  / total_err
        gur       = s.get("guessing", 0)   / total_err
        grid_n    = float(int(s.get("grid_size", 3)) ** 2)
        diff_int  = float({"Easy": 0, "Normal": 1, "Hard": 2}.get(
            s.get("difficulty", "Normal"), 1))
        orig = [tpc, er, hr, cr, nmr, gur, grid_n, diff_int, 0.5]
        log_tpc        = np.log1p(tpc)
        err_x_hint     = er * hr
        diff_x_grid    = diff_int * (grid_n / 81.0)
        hint_pressure  = hr / (cr + 0.01)
        move_density   = moves / grid_n
        patience_proxy = (1.0 - er) * cr
        return orig + [log_tpc, err_x_hint, diff_x_grid,
                       hint_pressure, move_density, patience_proxy]

    # Fungsi bantu ini memecah logika ml train hint model agar langkah kecilnya lebih rapi dan mudah dipakai ulang di bagian ini.
    def _hint_label(s):
        moves     = max(s.get("moves", 1), 1)
        total_t   = float(s.get("total_time", 0) or 0)
        hr        = s.get("hints_used", 0) / moves
        grid_n    = int(s.get("grid_size", 3)) ** 2
        diff_int  = {"Easy": 0, "Normal": 1, "Hard": 2}.get(
            s.get("difficulty", "Normal"), 1)
        inter_move  = total_t / moves
        patience    = 1.5 + diff_int * 0.4 + (grid_n / 81) * 0.6
        hint_factor = max(0.5, 1.0 - hr * 0.6)
        raw = inter_move * patience * hint_factor
        if raw > 60:
            raw = 60 + np.log1p(raw - 60) * 10
        return max(8.0, min(120.0, raw))

    try:
        X_np = np.array([_hint_session_to_fv(s) for s in flat], dtype=float)
        y_np = np.array([_hint_label(s) for s in flat], dtype=float)

        scaler = StandardScaler()
        X_sc   = scaler.fit_transform(X_np)

        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.model_selection import KFold, cross_val_score
        model = GradientBoostingRegressor(
            n_estimators=107,
            learning_rate=0.069,
            max_depth=4,
            min_samples_leaf=6,
            subsample=0.97,
            max_features=0.6,
            random_state=42,
        )
        model.fit(X_sc, y_np)

        _old_mae = float(cached_pkg.get("cv_mae_mean", 9999.0))
        _kf      = KFold(n_splits=min(5, len(X_np)), shuffle=True, random_state=42)
        _cv      = cross_val_score(model, X_sc, y_np, cv=_kf,
                                   scoring="neg_mean_absolute_error")
        new_mae  = float(-_cv.mean())

        if new_mae < _old_mae:
            self._hint_rfr    = model
            self._hint_scaler = scaler
            _save_pkl(PKL_HINT, {
                "model":        model,
                "scaler":       scaler,
                "feature_names": [
                    "avg_tpc", "error_rate", "hint_rate", "completion_rate",
                    "near_miss_rate", "guessing_rate", "grid_n", "diff_int",
                    "remaining_pct", "log_tpc", "err_x_hint", "diff_x_grid",
                    "hint_pressure", "move_density", "patience_proxy",
                ],
                "n_features":   15,
                "cv_mae_mean":  new_mae,
                "cv_mae_std":   float(_cv.std()),
                "n_train":      len(X_np),
            })
        else:
            self._hint_rfr    = cached_pkg["model"]
            self._hint_scaler = cached_pkg.get("scaler", None)
    except Exception:
        if self._hint_rfr is None:
            self._hint_rfr    = cached_pkg["model"]
            self._hint_scaler = cached_pkg.get("scaler", None)

# Hint Timer: prediksi threshold
# [DEMO-POINT] Komputasi threshold hint adaptif dari persentil data
# Menghitung ambang waktu hint dari statistik sesi dan parameter latihan yang tersedia.
def _ml_compute_hint_threshold(self, grid_size=3, difficulty="Normal",
                                remaining_pct=1.0):
    feat            = _orig_pmle_extract(self)
    avg_tpc         = float(feat.get("avg_time_per_cell", 10.0))
    error_rate      = float(feat.get("error_rate", 0.0))
    hint_rate       = float(feat.get("hint_rate", 0.0))
    completion_rate = float(feat.get("completion_rate", 0.5))
    near_miss_rate  = float(feat.get("near_miss_rate", 0.0))
    guessing_rate   = float(feat.get("guessing_rate", 0.0))

    grid_n   = float(int(grid_size) ** 2)
    diff_int = float({"Easy": 0, "Normal": 1, "Hard": 2}.get(difficulty, 1))
    rem_pct  = float(max(0.0, min(1.0, remaining_pct)))

    orig = [avg_tpc, error_rate, hint_rate, completion_rate,
            near_miss_rate, guessing_rate, grid_n, diff_int, rem_pct]

    log_tpc        = float(np.log1p(avg_tpc))
    err_x_hint     = error_rate * hint_rate
    diff_x_grid    = diff_int * (grid_n / 81.0)
    hint_pressure  = hint_rate / (completion_rate + 0.01)
    avg_moves_est  = float(feat.get("avg_moves", grid_n * 1.5))
    move_density   = avg_moves_est / grid_n
    patience_proxy = (1.0 - error_rate) * completion_rate

    fv_15 = np.array([orig + [log_tpc, err_x_hint, diff_x_grid,
                               hint_pressure, move_density, patience_proxy]],
                      dtype=float)
    fv_9  = np.array([orig], dtype=float)

    # Fungsi bantu ini memecah logika ml compute hint threshold agar langkah kecilnya lebih rapi dan mudah dipakai ulang di bagian ini.
    def _try_predict(model, scaler, fv_full, fv_short):
        try:
            n_feat = (model.n_features_in_
                      if hasattr(model, "n_features_in_") else 15)
            fv = fv_full if n_feat >= 15 else fv_short
            fv_sc = scaler.transform(fv) if scaler is not None else fv
            return float(model.predict(fv_sc)[0])
        except Exception:
            return None

    hint_rfr    = getattr(self, "_hint_rfr",    None)
    hint_scaler = getattr(self, "_hint_scaler", None)
    if hint_rfr is not None:
        pred = _try_predict(hint_rfr, hint_scaler, fv_15, fv_9)
        if pred is not None:
            return max(8.0, min(120.0, pred))

    try:
        cached = _load_pkl(PKL_HINT)
        if cached is not None:
            self._hint_rfr    = cached["model"]
            self._hint_scaler = cached.get("scaler", None)
            pred = _try_predict(self._hint_rfr, self._hint_scaler, fv_15, fv_9)
            if pred is not None:
                return max(8.0, min(120.0, pred))
    except Exception:
        pass

    base     = max(8.0, avg_tpc * 2.0)
    grid_mul = 1.0 + (grid_n - 4.0) / 77.0 * 0.5
    diff_mul = 1.0 + diff_int * 0.3
    hint_adj = max(0.5, 1.0 - hint_rate * 0.5)
    rem_adj  = 0.8 + rem_pct * 0.4
    return max(8.0, min(120.0, base * grid_mul * diff_mul * hint_adj * rem_adj))

# Hint Timer: pengganti should_give_hint
# Menentukan apakah hint sebaiknya diberikan berdasarkan kondisi saat ini dan ambang adaptif.
def _ml_should_give_hint(self, idle, errors, moves,
                          grid_size=3, difficulty="Normal", remaining_pct=1.0):
    threshold = _ml_compute_hint_threshold(self, grid_size, difficulty, remaining_pct)
    if idle > threshold:
        return True, "idle"
    if moves > 5 and errors / max(moves, 1) > 0.4:
        return True, "errors"
    return False, None

PlayerMLEngine.__init__ = _ml_init
PlayerMLEngine.add_session = _ml_add_session
PlayerMLEngine._train_ml_models = _train_ml_models
PlayerMLEngine.predict_stat_profile = _ml_predict_profile
PlayerMLEngine.recommend_next_challenge = _ml_recommend_next_challenge
PlayerMLEngine.recommend_difficulty = _ml_recommend_difficulty
PlayerMLEngine.get_summary = _ml_get_summary
PlayerMLEngine._train_hint_model      = _ml_train_hint_model
PlayerMLEngine.compute_hint_threshold = _ml_compute_hint_threshold
PlayerMLEngine.should_give_hint       = _ml_should_give_hint

# [DEMO-POINT] Agregasi data ML + profil skill untuk dashboard
# Menyusun data sesi yang dipakai untuk panel dashboard performa dan rekomendasi.
def _ml_dashboard_session(session, ml):
    disp = copy.deepcopy(session)
    try:
        prof = ml.predict_stat_profile(session)
        disp["ml_profile"] = prof
    except Exception:
        disp["ml_profile"] = {}
    return disp


# =============================================================================
# BAGIAN 7: KOMPONEN UI DAN LAYAR (SCREEN CLASSES)
# =============================================================================

class AnimatedBG(tk.Canvas):
    """
    AnimatedBG - Canvas tkinter dengan animasi 35 titik mengambang sebagai latar.

    Deskripsi:
        Menggambar titik-titik berwarna yang bergerak perlahan (loop 50ms).
        Dipakai di layar Login dan pemilihan grid/difficulty.

    Atribut:
        _pts (list[dict]): Titik animasi dengan posisi, kecepatan, dan warna.
    """
    # Menginisialisasi objek AnimatedBG dan menyiapkan state awal, referensi penting, serta elemen yang dibutuhkan sebelum layar dipakai.
    def __init__(self, master, **kw):
        super().__init__(master, **kw)
        self._pts = []
        self.after(50, self._init)

    # Menangani proses init pada AnimatedBG sambil menjaga state internal tetap konsisten.
    def _init(self):
        sw = self.winfo_screenwidth()
        sh = self.winfo_screenheight()
        t = _DARK_THEME if _CURRENT_THEME_NAME == "dark" else _LIGHT_THEME
        cols = t["ANIM_COLS"]
        for _ in range(35):
            self._pts.append({
                "x": random.uniform(0, sw), "y": random.uniform(0, sh),
                "r": random.uniform(1.5, 4.5),
                "vx": random.uniform(-0.3, 0.3), "vy": random.uniform(-0.3, 0.3),
                "c": random.choice(cols)
            })
        self._run()

    # Menangani proses run pada AnimatedBG sambil menjaga state internal tetap konsisten.
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

# SCREEN: ATTRACTOR / DEMO  (idle screen for booth)
# Aktif otomatis setelah IDLE_TIMEOUT detik tanpa interaksi
# di LoginScreen.  Menampilkan:
#   Kiri  - statistik global pemain hari ini
#   Tengah - board Sudoku 9x9 yang dipecahkan AI secara
#             animasi lambat (replay hist dari solve_backtracking_mrv)
#   Kanan  - kartu fitur ML
# Sembarang key / klik / gerakan mouse → kembali ke login.
class AttractorScreen(tk.Frame):
    """
    AttractorScreen - Layar demo idle yang tampil setelah IDLE_TIMEOUT detik.

    Deskripsi:
        Tiga panel: statistik global (kiri), AI solver animasi (tengah),
        kartu fitur ML (kanan). Dirancang untuk demo booth/pameran.
        Sembarang interaksi → kembali ke LoginScreen.

    Atribut:
        IDLE_TIMEOUT (int): Detik idle sebelum tampil (default 45).
        STEP_DELAY (int): ms per langkah animasi solver (default 110).
    """
    IDLE_TIMEOUT  = 30     # detik idle sebelum attractor muncul
    STEP_DELAY    = 110    # ms per langkah solver (terlihat lambat & "berpikir")
    DONE_PAUSE    = 2800   # ms jeda setelah puzzle selesai sebelum puzzle baru
    CELL_PX       = 50     # ukuran piksel tiap sel grid

    # Palet warna sesuai tema
    @staticmethod
    # Menangani proses tc pada AttractorScreen sambil menjaga state internal tetap konsisten.
    def _tc():
        if _CURRENT_THEME_NAME == "dark":
            return {
                "bg":       "#060A10",
                "board_bg": "#0D1A2E",
                "box_line": "#3D85C8",
                "fixed":    ("#112244", "#79C0FF"),
                "ai":       ("#0C2E18", "#7EE787"),
                "backtrack":("#3B1212", "#FF7B7B"),
                "empty":    ("#0D1A2E", "#2A3A4A"),
            }
        else:
            return {
                "bg":       C_BG,
                "board_bg": "#EFF6FF",
                "box_line": "#0969DA",
                "fixed":    ("#DBEAFE", "#1D4ED8"),
                "ai":       ("#DCFCE7", "#166534"),
                "backtrack":("#FFE4E6", "#BE123C"),
                "empty":    ("#EFF6FF", "#94A3B8"),
            }

    # Menginisialisasi objek AttractorScreen dan menyiapkan state awal, referensi penting, serta elemen yang dibutuhkan sebelum layar dipakai.
    def __init__(self, master, on_dismiss):
        tc = self._tc()
        super().__init__(master, bg=tc["bg"])
        self.on_dismiss  = on_dismiss
        self._step_job   = None
        self._blink_job  = None
        self._hist       = []
        self._hist_idx   = 0
        self._board      = []
        self._puzzle     = []
        self._canvases   = {}
        self._running    = True
        self._last_motion = time.time()
        self._build()
        self.after(80, self._new_puzzle)

    # Layout utama
    # Membangun bagian antarmuka pada AttractorScreen dan menyiapkan widget supaya state tampilan tetap konsisten.
    def _build(self):
        tc  = self._tc()
        bg  = tc["bg"]
        self.place(relx=0, rely=0, relwidth=1, relheight=1)

        abg = AnimatedBG(self, bg=bg, highlightthickness=0)
        abg.place(relx=0, rely=0, relwidth=1, relheight=1)

        self.bind_all("<Key>",       self._on_keypress,  add="+")
        self.bind_all("<Button-1>",  self._dismiss,      add="+")
        self.bind_all("<Motion>",    self._on_motion,    add="+")

        hdr = tk.Frame(self, bg=bg)
        hdr.pack(fill="x", side="top")

        hdr_inner = tk.Frame(hdr, bg=bg)
        hdr_inner.pack(pady=(18, 6))

        title_row = tk.Frame(hdr_inner, bg=bg)
        title_row.pack()

        _att_logo = None
        _att_logo_path = IMAGE_LOGO
        if os.path.exists(_att_logo_path):
            try:
                from PIL import Image as _AttImg, ImageTk as _AttITk
                _att_pil  = _AttImg.open(_att_logo_path).convert("RGBA")
                _att_h    = 42
                _att_w    = int(_att_pil.width * _att_h / _att_pil.height)
                _att_pil  = _att_pil.resize((_att_w, _att_h), _AttImg.LANCZOS)
                _att_logo = _AttITk.PhotoImage(_att_pil)
            except Exception:
                _att_logo = None

        _border_col = C_ACCENT
        _inner_col  = bg

        if _att_logo:
            _att_wrap  = tk.Frame(title_row, bg=_border_col, padx=2, pady=2)
            _att_wrap.pack(side="left", padx=(0, 10))
            _att_inner = tk.Frame(_att_wrap, bg=_inner_col, padx=4, pady=4)
            _att_inner.pack()
            _att_lbl   = tk.Label(_att_inner, image=_att_logo, bg=_inner_col)
            _att_lbl.image = _att_logo
            _att_lbl.pack()
        else:
            tk.Label(title_row, text="⬛",
                     font=("Segoe UI", 30), bg=bg, fg=C_ACCENT).pack(side="left", padx=(0, 8))

        tk.Label(title_row, text="SUDOKU AI",
                 font=("Segoe UI", 30, "bold"),
                 bg=bg, fg=C_TEXT).pack(side="left")
        tk.Label(title_row, text="   LIVE DEMO",
                 font=("Segoe UI", 18, "bold"),
                 bg=bg, fg=C_ACCENT).pack(side="left")

        tk.Label(hdr_inner,
                 text="AI memainkan Sudoku menggunakan MRV Backtracking Solver",
                 font=("Segoe UI", 11),
                 bg=bg, fg=C_TEXT_DIM).pack(pady=(4, 0))

        gbar = tk.Canvas(hdr, height=4, bg=bg, highlightthickness=0)
        gbar.pack(fill="x", pady=(10, 0))
        gbar.after(120, lambda: draw_gradient_bar(gbar))

        content_wrap = tk.Frame(self, bg=bg)
        content_wrap.pack(side="top", fill="both", expand=True)

        content = tk.Frame(content_wrap, bg=bg)
        content.place(relx=0.5, rely=0.5, anchor="center")

        left = tk.Frame(content, bg=C_SURFACE, width=248,
                         highlightbackground=C_BORDER, highlightthickness=1)
        left.pack(side="left", padx=(0, 18), fill="y")
        left.pack_propagate(False)
        self._build_stats_panel(left)

        center = tk.Frame(content, bg=bg)
        center.pack(side="left")

        self.status_var = tk.StringVar(value="⚙  Menghasilkan puzzle...")
        tk.Label(center, textvariable=self.status_var,
                 font=("Segoe UI", 10, "bold"),
                 bg=bg, fg=C_ACCENT2).pack(pady=(0, 8))

        self.grid_frame = tk.Frame(center, bg=bg)
        self.grid_frame.pack()

        self.moves_var = tk.StringVar(value="Langkah: 0  |  Backtrack: 0")
        tk.Label(center, textvariable=self.moves_var,
                 font=("Consolas", 10),
                 bg=bg, fg=C_TEXT_DIM).pack(pady=(8, 0))

        right = tk.Frame(content, bg=C_SURFACE, width=248,
                          highlightbackground=C_BORDER, highlightthickness=1)
        right.pack(side="left", padx=(18, 0), fill="y")
        right.pack_propagate(False)
        self._build_ml_panel(right)

        foot = tk.Frame(self, bg=bg)
        foot.pack(fill="x", side="bottom")

        gbar2 = tk.Canvas(foot, height=4, bg=bg, highlightthickness=0)
        gbar2.pack(fill="x")
        gbar2.after(160, lambda: draw_gradient_bar(gbar2))

        tap_row = tk.Frame(foot, bg=bg)
        tap_row.pack(pady=14)

        self._tap_lbl = tk.Label(
            tap_row,
            text="▶   PRESS ANY BUTTON TO START   ◀",
            font=("Segoe UI", 13, "bold"),
            bg=bg, fg=C_ACCENT,
        )
        self._tap_lbl.pack()
        self._blink_start()

    # Panel kiri: statistik global
    # Membangun stats panel pada AttractorScreen dan menyiapkan widget supaya state tampilan tetap konsisten.
    def _build_stats_panel(self, parent):
        data    = load_data()
        players = data.get("players", {})
        today   = time.strftime("%Y-%m-%d")

        total_players   = len(players)
        sessions_today  = 0
        sessions_all    = 0
        completions     = 0
        best_per_player = {}

        for pname, pdata in players.items():
            for s in pdata.get("sessions", []):
                sessions_all += 1
                ts = s.get("timestamp", 0)
                try:
                    if time.strftime("%Y-%m-%d", time.localtime(ts)) == today:
                        sessions_today += 1
                except Exception:
                    pass
                if s.get("completed"):
                    completions += 1
                sc = int(s.get("score", 0) or 0)
                if sc > best_per_player.get(pname, 0):
                    best_per_player[pname] = sc

        completion_pct = completions / max(sessions_all, 1) * 100

        tk.Label(parent, text="STATISTIK GLOBAL",
                 font=("Segoe UI", 10, "bold"),
                 bg=C_SURFACE, fg=C_TEXT).pack(anchor="w", padx=14, pady=(14, 6))
        tk.Frame(parent, height=1, bg=C_BORDER).pack(fill="x")

        stat_items = [
            ("Pemain",          str(total_players),          C_ACCENT),
            ("Game Hari Ini",   str(sessions_today),         C_ACCENT2),
            ("Total Game",      str(sessions_all),           C_TEXT),
            ("Completion Rate", f"{completion_pct:.0f}%",   C_ACCENT2),
        ]
        sf = tk.Frame(parent, bg=C_SURFACE)
        sf.pack(fill="x", padx=12, pady=8)
        for lbl, val, color in stat_items:
            row = tk.Frame(sf, bg=C_SURFACE2,
                           highlightbackground=C_BORDER, highlightthickness=1)
            row.pack(fill="x", pady=3)
            inn = tk.Frame(row, bg=C_SURFACE2)
            inn.pack(fill="x", padx=10, pady=6)
            tk.Label(inn, text=lbl,
                     font=("Segoe UI", 8), bg=C_SURFACE2, fg=C_TEXT_DIM).pack(anchor="w")
            tk.Label(inn, text=val,
                     font=("Segoe UI", 15, "bold"),
                     bg=C_SURFACE2, fg=color).pack(anchor="w")

        tk.Frame(parent, height=1, bg=C_BORDER).pack(fill="x", pady=(2, 0))
        tk.Label(parent, text="🏆  TOP PEMAIN",
                 font=("Segoe UI", 10, "bold"),
                 bg=C_SURFACE, fg=C_GOLD).pack(anchor="w", padx=14, pady=(10, 6))

        top3 = sorted(best_per_player.items(), key=lambda x: x[1], reverse=True)[:3]

        rank_medals = ["🥇", "🥈", "🥉"]
        rank_colors = [C_GOLD, "#B0B8C8", "#CD9B6A"]

        lf = tk.Frame(parent, bg=C_SURFACE)
        lf.pack(fill="x", padx=14)
        if top3:
            for i, (name, score) in enumerate(top3):
                medal  = rank_medals[i]
                mcolor = rank_colors[i]
                r = tk.Frame(lf, bg=C_SURFACE)
                r.pack(fill="x", pady=4)
                tk.Label(r, text=medal,
                         font=("Segoe UI", 16),
                         bg=C_SURFACE, fg=mcolor).pack(side="left")
                tk.Label(r, text=f" {name[:13]}",
                         font=("Segoe UI", 10, "bold"),
                         bg=C_SURFACE, fg=C_TEXT).pack(side="left")
                tk.Label(r, text=str(score),
                         font=("Segoe UI", 9),
                         bg=C_SURFACE, fg=C_ACCENT2).pack(side="right")
        else:
            tk.Label(lf, text="Belum ada data skor",
                     font=FONT_SMALL, bg=C_SURFACE, fg=C_TEXT_DIM).pack()

    # Panel kanan: fitur ML
    # Membangun ml panel pada AttractorScreen dan menyiapkan widget supaya state tampilan tetap konsisten.
    def _build_ml_panel(self, parent):
        tk.Label(parent, text="🤖  FITUR AI & ML",
                 font=("Segoe UI", 10, "bold"),
                 bg=C_SURFACE, fg=C_PURPLE).pack(anchor="w", padx=14, pady=(14, 6))
        tk.Frame(parent, height=1, bg=C_BORDER).pack(fill="x")

        features = [
            ("🎯", "K-Nearest Neighbour",
             "Klasifikasi tipe pemain berdasarkan\n6 fitur sesi bermain", C_ACCENT),
            ("📈", "HistGradientBoosting Regressor",
             "Prediksi skor sesi berikutnya\nberdasarkan pola historis pemain", C_ACCENT2),
            ("🔍", "Isolation Forest",
             "Deteksi sesi anomali dari\npola bermain yang tidak wajar", C_WARN),
            ("⚡", "MRV Backtracking Solver",
             "Algoritma AI optimal dengan\nheuristic Minimum Remaining Values", C_PURPLE),
            ("🧩", "RFC Difficulty Recommender",
             "Rekomendasi tingkat difficulty\nberikutnya melalui RFC", C_PINK),
            ("⏱️",  "GBR Hint Timer",
             "Threshold hint adaptif dilatih\ndengan Gradient Boosting Regressor", C_GOLD),
        ]

        ff = tk.Frame(parent, bg=C_SURFACE)
        ff.pack(fill="x", padx=12, pady=8)
        for icon, title, desc, color in features:
            card = tk.Frame(ff, bg=C_SURFACE2,
                            highlightbackground=color, highlightthickness=1)
            card.pack(fill="x", pady=3)
            inn = tk.Frame(card, bg=C_SURFACE2)
            inn.pack(fill="x", padx=10, pady=6)
            tk.Label(inn, text=f"{icon}  {title}",
                     font=("Segoe UI", 8, "bold"),
                     bg=C_SURFACE2, fg=color).pack(anchor="w")
            tk.Label(inn, text=desc,
                     font=("Segoe UI", 7),
                     bg=C_SURFACE2, fg=C_TEXT_DIM, justify="left").pack(anchor="w")

        tk.Frame(parent, height=1, bg=C_BORDER).pack(fill="x", pady=(4, 0))
        badge_row = tk.Frame(parent, bg=C_SURFACE)
        badge_row.pack(pady=10)
        active = SKLEARN_AVAILABLE
        tk.Label(badge_row,
                 text=f"  {'sklearn: AKTIF' if active else 'sklearn: tidak tersedia'}  ",
                 font=("Segoe UI", 8, "bold"),
                 bg=C_SURFACE2,
                 fg=C_ACCENT2 if active else C_TEXT_DIM,
                 padx=8, pady=4).pack()

    # Board rendering
    # Pola dua level identik dengan GameScreen._build_grid:
    #   Lv-1  grid 3×3 - Frame(bg=box_line_color, padx=2, pady=2)
    #          ditempatkan dengan padx=4, pady=4 → garis TEBAL antar kotak
    #   Lv-2  canvas sel - grid(padx=1, pady=1) di dalam grid
    #          → garis TIPIS antar sel; bg grid terlihat sebagai garis tersebut

    # Membangun board pada AttractorScreen dan menyiapkan widget supaya state tampilan tetap konsisten.
    def _build_board(self, N, BOX):
        tc = self._tc()
        for w in self.grid_frame.winfo_children():
            try:
                w.destroy()
            except Exception:
                pass
        self._canvases = {}
        self._blocks   = {}
        CELL = self.CELL_PX

        for br in range(BOX):
            for bc in range(BOX):
                blk = tk.Frame(
                    self.grid_frame,
                    bg=tc["box_line"],
                    padx=2, pady=2,
                )
                blk.grid(row=br, column=bc, padx=4, pady=4)
                self._blocks[(br, bc)] = blk

        for r in range(N):
            for c in range(N):
                parent = self._blocks[(r // BOX, c // BOX)]
                cv = tk.Canvas(
                    parent,
                    width=CELL, height=CELL,
                    bg=tc["empty"][0],
                    highlightthickness=0,
                )
                cv.grid(row=r % BOX, column=c % BOX, padx=1, pady=1)
                self._canvases[(r, c)] = cv

    # Menggambar cell pada AttractorScreen sesuai state yang sedang aktif.
    def _draw_cell(self, r, c, value, state="empty"):
        cv = self._canvases.get((r, c))
        if not cv:
            return
        CELL = self.CELL_PX
        cv.delete("all")
        tc = self._tc()
        state_colors = {
            "fixed":     tc["fixed"],
            "ai":        tc["ai"],
            "backtrack": tc["backtrack"],
            "empty":     tc["empty"],
        }
        bg, fg = state_colors.get(state, state_colors["empty"])
        cv.config(bg=bg)
        if value:
            cv.create_text(CELL // 2, CELL // 2,
                           text=str(value),
                           font=("Segoe UI", int(CELL * 0.40), "bold"),
                           fill=fg, anchor="center")

    # Solver animation
    # Menangani proses new puzzle pada AttractorScreen sambil menjaga state internal tetap konsisten.
    def _new_puzzle(self):
        if not self._running:
            return
        self.status_var.set("⚙  Menghasilkan puzzle baru...")
        self.moves_var.set("Langkah: 0  |  Backtrack: 0")

        # Menangani proses gen pada AttractorScreen sambil menjaga state internal tetap konsisten.
        def _gen():
            try:
                N, BOX = 9, 3
                puzzle, _ = generate_puzzle(N, BOX, remove_pct=0.50)
                hist, _, _ = solve_backtracking_mrv(
                    [row[:] for row in puzzle], N, BOX)
                if self._running:
                    self.after(0, lambda: self._start_animation(
                        puzzle, hist or [], N, BOX))
            except Exception:
                pass

        threading.Thread(target=_gen, daemon=True).start()

    # Memulai animation pada AttractorScreen dan menyalakan mekanisme pendukung yang dibutuhkan.
    def _start_animation(self, puzzle, hist, N, BOX):
        if not self._running:
            return
        self._build_board(N, BOX)
        self._board    = [row[:] for row in puzzle]
        self._puzzle   = [row[:] for row in puzzle]
        self._hist     = hist
        self._hist_idx = 0
        self._backtrack_count = 0

        for r in range(N):
            for c in range(N):
                val = puzzle[r][c]
                self._draw_cell(r, c, val or None, "fixed" if val else "empty")

        self.status_var.set("🤖  AI sedang berpikir  (MRV Backtracking)...")
        self._cancel_step()
        self._schedule_step()

    # Menangani proses schedule step pada AttractorScreen sambil menjaga state internal tetap konsisten.
    def _schedule_step(self):
        if not self._running:
            return
        self._step_job = self.after(self.STEP_DELAY, self._do_step)

    # Menangani proses cancel step pada AttractorScreen sambil menjaga state internal tetap konsisten.
    def _cancel_step(self):
        if self._step_job:
            try:
                self.after_cancel(self._step_job)
            except Exception:
                pass
            self._step_job = None

    # Menangani proses do step pada AttractorScreen sambil menjaga state internal tetap konsisten.
    def _do_step(self):
        if not self._running:
            return
        if self._hist_idx >= len(self._hist):
            total = len(self._hist)
            self.moves_var.set(
                f"Total langkah: {total}  |  Backtrack: {self._backtrack_count}")

            # Menangani proses countdown pada AttractorScreen sambil menjaga state internal tetap konsisten.
            def _countdown(sisa):
                if not self._running:
                    return
                if sisa <= 0:
                    self._step_job = self.after(0, self._new_puzzle)
                    return
                try:
                    self.status_var.set(
                        f"✅  Puzzle terpecahkan!  Puzzle baru dalam {sisa} detik...")
                except Exception:
                    return
                self._step_job = self.after(1000, lambda: _countdown(sisa - 1))

            _countdown(3)
            return

        r, c, val = self._hist[self._hist_idx]
        self._hist_idx += 1

        if self._puzzle[r][c] != 0:
            state = "fixed"
        elif val == 0:
            state = "backtrack"
            self._backtrack_count += 1
        else:
            state = "ai"

        self._board[r][c] = val
        self._draw_cell(r, c, val if val else None, state)
        self.moves_var.set(
            f"Langkah: {self._hist_idx}  |  Backtrack: {self._backtrack_count}")
        self._schedule_step()

    # Blink label "tap to start"
    # Menangani proses blink start pada AttractorScreen sambil menjaga state internal tetap konsisten.
    def _blink_start(self):
        self._blink_tap()

    # Menangani proses blink tap pada AttractorScreen sambil menjaga state internal tetap konsisten.
    def _blink_tap(self):
        if not self._running:
            return
        try:
            cur = self._tap_lbl.cget("fg")
            nxt = C_ACCENT2 if cur == C_ACCENT else C_ACCENT
            self._tap_lbl.config(fg=nxt)
            self._blink_job = self.after(650, self._blink_tap)
        except Exception:
            pass

    # Dismiss
    # Menangani event keypress pada AttractorScreen dan memperbarui state yang terkait.
    def _on_keypress(self, event=None):
        if event and event.keysym in ("Shift_L", "Shift_R",
                                      "Control_L", "Control_R",
                                      "Alt_L", "Alt_R",
                                      "Super_L", "Super_R",
                                      "Caps_Lock", "Num_Lock"):
            return
        self._dismiss(event)

    # Menangani event motion pada AttractorScreen dan memperbarui state yang terkait.
    def _on_motion(self, event=None):
        now = time.time()
        if now - self._last_motion > 2.0:
            self._last_motion = now
            self._dismiss(event)

    # Menangani proses dismiss pada AttractorScreen sambil menjaga state internal tetap konsisten.
    def _dismiss(self, event=None):
        if not self._running:
            return
        self._running = False
        self._cancel_step()
        if self._blink_job:
            try:
                self.after_cancel(self._blink_job)
            except Exception:
                pass
        try:
            self.unbind_all("<Key>")
            self.unbind_all("<Button-1>")
            self.unbind_all("<Motion>")
        except Exception:
            pass
        if callable(self.on_dismiss):
            self.on_dismiss()

# TOOLTIP
class Tooltip:
    """
    Tooltip - Popup teks kecil saat hover di atas widget.

    Atribut:
        w (tk.Widget): Widget target.
        text (str): Teks yang ditampilkan.
        tip (tk.Toplevel | None): Jendela tooltip aktif, None jika tersembunyi.

    Contoh Penggunaan:
        Tooltip(btn, "Klik untuk detail skor")
    """
    # Menginisialisasi objek Tooltip dan menyiapkan state awal, referensi penting, serta elemen yang dibutuhkan sebelum layar dipakai.
    def __init__(self, w, text):
        self.w, self.text, self.tip = w, text, None
        w.bind("<Enter>", self.show)
        w.bind("<Leave>", self.hide)

    # Menampilkan Tooltip dan menyiapkan tampilan agar bisa langsung dipakai.
    def show(self, _=None):
        if self.tip: return
        x = self.w.winfo_rootx() + 20
        y = self.w.winfo_rooty() + self.w.winfo_height() + 4
        self.tip = tk.Toplevel(self.w)
        self.tip.wm_overrideredirect(True)
        self.tip.wm_geometry(f"+{x}+{y}")
        tk.Label(self.tip, text=self.text, bg=C_SURFACE2, fg=C_TEXT,
                 font=FONT_SMALL, padx=8, pady=4,
                 highlightbackground=C_BORDER, highlightthickness=1).pack()

    # Menyembunyikan Tooltip dan merapikan state tampilan yang tidak lagi dipakai.
    def hide(self, _=None):
        if self.tip:
            self.tip.destroy()
            self.tip = None

# SCREEN: LOGIN  (fixed - no clipped text, recent users)
class LoginScreen(tk.Frame):
    """
    LoginScreen - Layar login tempat pemain memasukkan username.

    Deskripsi:
        Form input username + daftar pemain lama + idle timer ke AttractorScreen.

    Atribut:
        on_login (callable): Callback dengan username saat login.
        _last_activity (float): Timestamp aktivitas terakhir.
        _attractor_job (str | None): ID after() untuk idle check.
    """
    # Detik idle sebelum AttractorScreen diaktifkan
    _ATTRACTOR_IDLE = AttractorScreen.IDLE_TIMEOUT

    # Menginisialisasi objek LoginScreen dan menyiapkan state awal, referensi penting, serta elemen yang dibutuhkan sebelum layar dipakai.
    def __init__(self, master, on_login, on_browse_players=None, on_attractor=None):
        super().__init__(master, bg=C_BG)
        self.on_login          = on_login
        self.on_browse_players = on_browse_players
        self.on_attractor      = on_attractor
        self.data              = load_data()
        self._last_activity    = time.time()
        self._attractor_job    = None
        self._build()
        self._schedule_attractor_check()

    # Membangun bagian antarmuka pada LoginScreen dan menyiapkan widget supaya state tampilan tetap konsisten.
    def _build(self):
        self.place(relx=0, rely=0, relwidth=1, relheight=1)

        bg = AnimatedBG(self, bg=C_BG, highlightthickness=0)
        bg.place(relx=0, rely=0, relwidth=1, relheight=1)

        card = tk.Frame(self, bg=C_SURFACE,
                        highlightbackground=C_BORDER, highlightthickness=1)
        card.place(relx=0.5, rely=0.5, anchor="center", width=500, height=555)

        gbar = tk.Canvas(card, height=8, bg=C_SURFACE, highlightthickness=0)
        gbar.pack(fill="x")
        gbar.after(120, lambda: self._gradient(gbar))

        _app = self.master
        while _app and not isinstance(_app, tk.Tk):
            _app = getattr(_app, 'master', None)
        _logo = getattr(_app, '_logo_tk', None) if _app else None
        self._egg_clicks  = 0
        self._egg_last_ts = 0.0
        if _logo:
            _logo_wrap = tk.Frame(card, bg=C_ACCENT, padx=3, pady=3,
                                  cursor="hand2")
            _logo_wrap.pack(pady=(20, 0))
            _logo_inner = tk.Frame(_logo_wrap, bg=C_SURFACE, padx=6, pady=6)
            _logo_inner.pack()
            _logo_lbl = tk.Label(_logo_inner, image=_logo, bg=C_SURFACE,
                                 cursor="hand2")
            _logo_lbl.pack()
            for _w in (_logo_wrap, _logo_inner, _logo_lbl):
                _w.bind("<Button-1>", self._logo_click)
        else:
            _fallback_lbl = tk.Label(card, text="⬛", font=("Segoe UI", 44),
                                     bg=C_SURFACE, fg=C_ACCENT,
                                     cursor="hand2")
            _fallback_lbl.pack(pady=(20, 0))
            _fallback_lbl.bind("<Button-1>", self._logo_click)

        tk.Label(card, text="SUDOKU AI",
                 font=("Segoe UI", 26, "bold"), bg=C_SURFACE, fg=C_TEXT).pack()
        tk.Label(card, text="Machine Learning Intelligence System",
                 font=("Segoe UI", 10), bg=C_SURFACE, fg=C_TEXT_DIM).pack(pady=(2, 0))

        tk.Frame(card, height=1, bg=C_BORDER).pack(fill="x", padx=28, pady=16)

        inp = tk.Frame(card, bg=C_SURFACE)
        inp.pack(padx=40, fill="x")
        tk.Label(inp, text="Input Username",
                 font=("Segoe UI", 9, "bold"),
                 bg=C_SURFACE, fg=C_TEXT_DIM).pack(anchor="w")

        self.username_var = tk.StringVar()
        ebox = tk.Frame(inp, bg=C_BORDER, pady=1, padx=1)
        ebox.pack(fill="x", pady=(4, 0))
        self.entry = tk.Entry(ebox, textvariable=self.username_var,
                              font=("Segoe UI", 13), bg=C_SURFACE2, fg=C_TEXT,
                              insertbackground=C_ACCENT, relief="flat", bd=8)
        self.entry.pack(fill="x")
        self.entry.bind("<Return>",   lambda _: self._login())
        self.entry.bind("<FocusIn>",  lambda _: ebox.config(bg=C_ACCENT))
        self.entry.bind("<FocusOut>", lambda _: ebox.config(bg=C_BORDER))
        self.entry.bind("<Key>", self._reset_activity, add="+")

        self.err_lbl = tk.Label(inp, text="", font=FONT_SMALL,
                                bg=C_SURFACE, fg=C_ERROR)
        self.err_lbl.pack(anchor="w", pady=(4, 0))

        tk.Button(card, text="START",
                  font=("Segoe UI", 12, "bold"),
                  bg=C_ACCENT, fg=C_BG,
                  activebackground="#79BFFF", activeforeground=C_BG,
                  relief="flat", cursor="hand2", pady=10,
                  command=lambda: (_play_sfx(_SFX_SELECT), self._login())
                  ).pack(padx=40, fill="x", pady=(8, 0))

        tk.Button(card, text="PLAYER LIST",
                  font=("Segoe UI", 11, "bold"),
                  bg=C_SURFACE2, fg=C_TEXT,
                  activebackground=C_BORDER, activeforeground=C_TEXT,
                  relief="flat", cursor="hand2", pady=10,
                  highlightbackground=C_ACCENT, highlightthickness=1,
                  command=lambda: (_play_sfx(_SFX_CLICK), self._open_player_login())
                  ).pack(padx=40, fill="x", pady=(10, 0))

        tk.Button(card, text="🏆  LEADERBOARD",
                  font=("Segoe UI", 11, "bold"),
                  bg=C_SURFACE2, fg=C_GOLD,
                  activebackground=C_BORDER, activeforeground=C_GOLD,
                  relief="flat", cursor="hand2", pady=10,
                  highlightbackground=C_GOLD, highlightthickness=1,
                  command=lambda: (_play_sfx(_SFX_CLICK), LeaderboardWindow(self.master, load_data()))
                  ).pack(padx=40, fill="x", pady=(10, 0))

        n_p = len(self.data.get("players", {}))
        tk.Label(card,
                 text=f"👤 {n_p} player registered",
                 font=FONT_SMALL, bg=C_SURFACE, fg=C_TEXT_DIM).pack(pady=(12, 20))

        self.entry.focus_set()

        self.bind_all("<Button-1>", self._reset_activity, add="+")

    # Idle-tracker helpers
    # Menangani proses reset activity pada LoginScreen sambil menjaga state internal tetap konsisten.
    def _reset_activity(self, event=None):
        self._last_activity = time.time()

    # Menangani proses schedule attractor check pada LoginScreen sambil menjaga state internal tetap konsisten.
    def _schedule_attractor_check(self):
        if not self.winfo_exists():
            return
        elapsed = time.time() - self._last_activity
        if elapsed >= self._ATTRACTOR_IDLE and callable(self.on_attractor):
            self._cancel_attractor_check()
            self.on_attractor()
            return
        self._attractor_job = self.after(5000, self._schedule_attractor_check)

    # Menangani proses cancel attractor check pada LoginScreen sambil menjaga state internal tetap konsisten.
    def _cancel_attractor_check(self):
        if self._attractor_job:
            try:
                self.after_cancel(self._attractor_job)
            except Exception:
                pass
            self._attractor_job = None

    # Menghancurkan widget pada LoginScreen dan melepaskan resource yang masih aktif.
    def destroy(self):
        self._cancel_attractor_check()
        super().destroy()

    # Menangani proses gradient pada LoginScreen sambil menjaga state internal tetap konsisten.
    def _gradient(self, c):
        c.update_idletasks()
        w = c.winfo_width()
        segs = [(0, C_ACCENT), (w//3, C_PURPLE), (2*w//3, C_PINK)]
        for i, (x, col) in enumerate(segs):
            x2 = segs[i+1][0] if i+1 < len(segs) else w
            c.create_rectangle(x, 0, x2+2, 8, fill=col, outline="")

    # Menangani proses open player login pada LoginScreen sambil menjaga state internal tetap konsisten.
    def _open_player_login(self):
        if callable(self.on_browse_players):
            self.on_browse_players()
        else:
            PlayerSelectScreen(self.master, current_user=None,
                               on_select=self.on_login, on_new_player=lambda: None)

    # Menangani proses login pada LoginScreen sambil menjaga state internal tetap konsisten.
    def _login(self):
        u = self.username_var.get().strip()
        if not u:
            return self._err("Username tidak boleh kosong!")
        if len(u) < 3:
            return self._err("Username minimal 3 karakter!")
        if len(u) > 20:
            return self._err("Username maksimal 20 karakter!")

        data = load_data()
        existing = _find_existing_username(data, u)
        if existing:
            return self._err(f'Username "{existing}" sudah dipakai. Gunakan tombol DAFTAR PEMAIN.')
        data["players"][u] = {"sessions": [], "created_at": time.time()}
        save_data(data)
        self.on_login(u, True, "Selamat datang")

    # Menangani proses err pada LoginScreen sambil menjaga state internal tetap konsisten.
    def _err(self, msg):
        self.err_lbl.config(text=f"⚠ {msg}")
        self.after(2500, lambda: self.err_lbl.config(text=""))

    # Menangani proses logo click pada LoginScreen sambil menjaga state internal tetap konsisten.
    def _logo_click(self, event=None):
        now = time.time()
        if now - self._egg_last_ts > 4.0:
            self._egg_clicks = 0
        self._egg_last_ts = now
        self._egg_clicks += 1
        if self._egg_clicks >= 7:
            _root = self.master
            while _root and not isinstance(_root, tk.Tk):
                _root = getattr(_root, "master", None)
            if _root:
                _trigger_easter_egg(_root)
# SCREEN: GRID SIZE SELECT  (Redesigned v3 - bulletproof layout)
class GridSizeScreen(tk.Frame):
    """
    GridSizeScreen - Layar pemilihan ukuran grid (4×4 atau 9×9).

    Deskripsi:
        Dua kartu dengan preview mini-grid. Pilihan menentukan N dan BOX
        untuk seluruh sesi berikutnya.

    Atribut:
        username (str): Username pemain aktif.
        on_select (callable): Callback dengan grid_size (2 atau 3).
    """
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

    # Menginisialisasi objek GridSizeScreen dan menyiapkan state awal, referensi penting, serta elemen yang dibutuhkan sebelum layar dipakai.
    def __init__(self, master, username, greeting, on_select):
        super().__init__(master, bg=C_BG)
        self.username  = username
        self.greeting  = greeting
        self.on_select = on_select
        self._build()

    # Gradient bar
    # Menggambar gradient pada GridSizeScreen sesuai state yang sedang aktif.
    def _draw_gradient(self, canvas, colors):
        draw_gradient_bar(canvas, colors)

    # Mini sudoku grid preview
    # Menggambar mini grid pada GridSizeScreen sesuai state yang sedang aktif.
    def _draw_mini_grid(self, canvas, box, cells, color, size):
        N       = box * box
        pad     = 6
        avail   = size - pad * 2
        cell_sz = avail / N

        for br in range(box):
            for bc in range(box):
                x1 = pad + bc * box * cell_sz
                y1 = pad + br * box * cell_sz
                x2 = x1 + box * cell_sz
                y2 = y1 + box * cell_sz
                blk_bg = C_MINI_BLK1 if (br + bc) % 2 == 0 else C_MINI_BLK2
                canvas.create_rectangle(x1, y1, x2, y2, fill=blk_bg, outline="")

        for i in range(1, N):
            xi = pad + i * cell_sz
            yi = pad + i * cell_sz
            is_block = (i % box == 0)
            col = color if is_block else C_MINI_LINE
            w_  = 2   if is_block else 1
            canvas.create_line(pad, yi, pad + avail, yi, fill=col, width=w_)
            canvas.create_line(xi, pad, xi, pad + avail, fill=col, width=w_)

        try:
            r_c = int(color[1:3], 16)
            g_c = int(color[3:5], 16)
            b_c = int(color[5:7], 16)
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

        font_sz = max(7, int(cell_sz * 0.55))
        for (r, c, val, _) in cells:
            if r >= N or c >= N: continue
            cx = pad + c * cell_sz + cell_sz / 2
            cy = pad + r * cell_sz + cell_sz / 2
            hs = cell_sz * 0.82
            canvas.create_rectangle(
                cx - hs/2, cy - hs/2, cx + hs/2, cy + hs/2,
                fill=C_MINI_CELL, outline="")
            canvas.create_text(cx, cy, text=str(val),
                                fill=color,
                                font=("Segoe UI", font_sz, "bold"),
                                anchor="center")

    # Main layout
    # Membangun bagian antarmuka pada GridSizeScreen dan menyiapkan widget supaya state tampilan tetap konsisten.
    def _build(self):
        self.place(relx=0, rely=0, relwidth=1, relheight=1)

        abg = AnimatedBG(self, bg=C_BG, highlightthickness=0)
        abg.place(relx=0, rely=0, relwidth=1, relheight=1)

        hdr = tk.Frame(self, bg=C_SURFACE)
        hdr.pack(fill="x")

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

        center_wrap = tk.Frame(self, bg=C_BG)
        center_wrap.pack(side="top", fill="both", expand=True)

        inner = tk.Frame(center_wrap, bg=C_BG)
        inner.place(relx=0.5, rely=0.5, anchor="center")

        sec_row = tk.Frame(inner, bg=C_BG)
        sec_row.grid(row=0, column=0, columnspan=2, pady=(0, 24))
        tk.Frame(sec_row, width=55, height=1, bg=C_BORDER).pack(side="left", padx=(0,12))
        tk.Label(sec_row, text="PILIH UKURAN GRID",
                 font=("Segoe UI", 10, "bold"), bg=C_BG, fg=C_TEXT_DIM).pack(side="left")
        tk.Frame(sec_row, width=55, height=1, bg=C_BORDER).pack(side="left", padx=(12,0))

        grids = [
            {
                "box": 2, "full": "4 × 4",
                "color": "#BC8CFF", "tag_bg": C_SURFACE2,
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
                "color": "#58A6FF", "tag_bg": C_SURFACE2,
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

        foot = tk.Frame(self, bg=C_SURFACE, pady=11)
        foot.pack(fill="x", side="bottom")
        tk.Label(foot,
                 text="💡  Tidak yakin? Mulai dari 4×4 untuk memahami aturan dasar Sudoku",
                 font=("Segoe UI", 9), bg=C_SURFACE, fg=C_TEXT_DIM).pack()

    # Individual card
    # Menangani proses card pada GridSizeScreen sambil menjaga state internal tetap konsisten.
    def _card(self, parent, g, col_idx):
        color   = g["color"]
        box     = g["box"]
        GRID_SZ = 164 if box == 3 else 136

        outer = tk.Frame(parent, bg=C_BG)
        outer.grid(row=1, column=col_idx, padx=20, sticky="n")

        card = tk.Frame(outer, bg=C_SURFACE,
                        highlightbackground=C_BORDER,
                        highlightthickness=1,
                        cursor="hand2")
        card.pack()

        CARD_W = 290
        spacer = tk.Frame(card, bg=C_SURFACE, height=1, width=CARD_W)
        spacer.pack(side="top")
        spacer.lower()

        tag_bar = tk.Frame(card, bg=C_SURFACE)
        tag_bar.pack(fill="x", padx=18, pady=(16, 0))
        tk.Label(tag_bar, text=f"  {g['tag']}  ",
                 font=("Segoe UI", 8, "bold"),
                 bg=g["tag_bg"], fg=color,
                 padx=6, pady=3).pack(side="left")

        preview_wrap = tk.Frame(card, bg=C_SURFACE)
        preview_wrap.pack(pady=(14, 0))

        ring = tk.Frame(preview_wrap, bg=color, padx=2, pady=2)
        ring.pack()
        dark_bg = tk.Frame(ring, bg=C_BG)
        dark_bg.pack()

        cv = tk.Canvas(dark_bg, width=GRID_SZ, height=GRID_SZ,
                       bg=C_BG, highlightthickness=0)
        cv.pack()
        cv.after(150, lambda c=cv, b=box, cl=g["cells"], col=color, sz=GRID_SZ:
                 self._draw_mini_grid(c, b, cl, col, sz))

        size_row = tk.Frame(card, bg=C_SURFACE)
        size_row.pack(pady=(16, 2))
        tk.Label(size_row, text=g["full"],
                 font=("Segoe UI", 32, "bold"),
                 bg=C_SURFACE, fg=color).pack()

        tk.Label(card,
                 text=f"Block {box}×{box}  ·  Grid {box*box}×{box*box} sel",
                 font=("Segoe UI", 9),
                 bg=C_SURFACE, fg=C_TEXT_DIM).pack()

        divider = tk.Frame(card, bg=C_BORDER, height=1)
        divider.pack(fill="x", padx=18, pady=(16, 12))

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

        btn_wrap = tk.Frame(card, bg=C_SURFACE)
        btn_wrap.pack(fill="x", padx=18, pady=(18, 20))
        btn = tk.Button(btn_wrap,
                        text=f"▶   {g['cta']}",
                        font=("Segoe UI", 11, "bold"),
                        bg=color, fg=C_BG,
                        activebackground="#FFFFFF",
                        activeforeground=C_BG,
                        relief="flat", cursor="hand2", pady=12,
                        command=lambda b=box: (_play_sfx(_SFX_SELECT), self.on_select(b)))
        btn.pack(fill="x")

        # Menangani interaksi enter pada GridSizeScreen agar respons UI tetap konsisten.
        def _enter(_):
            card.config(highlightbackground=color, highlightthickness=2)
            _play_sfx(_SFX_HOVER)

        # Menangani interaksi leave pada GridSizeScreen agar respons UI tetap konsisten.
        def _leave(_):
            card.config(highlightbackground=C_BORDER, highlightthickness=1)

        hover_targets = [card, tag_bar, preview_wrap, ring, dark_bg,
                         cv, size_row, feat_wrap, btn_wrap, spacer]
        for w in hover_targets:
            w.bind("<Enter>", _enter)
            w.bind("<Leave>", _leave)
        for w in [card, spacer, preview_wrap, size_row, feat_wrap]:
            w.bind("<Button-1>", lambda _, b=box: (_play_sfx(_SFX_SELECT), self.on_select(b)))

# SCREEN: DIFFICULTY SELECT  (Redesigned v2)
class DifficultyScreen(tk.Frame):
    """
    DifficultyScreen - Layar pemilihan tingkat kesulitan (Easy/Normal/Hard).

    Deskripsi:
        Tiga kartu dengan ikon vektor prosedural. Badge rekomendasi AI
        ditampilkan jika model RFC tersedia (confidence ≥ 55%).

    Atribut:
        username (str): Username pemain aktif.
        grid_size (int): Ukuran kotak dari GridSizeScreen.
        on_select (callable): Callback dengan difficulty (str).
    """

    # Canvas-drawn icon painters - each receives (canvas, size, color)
    @staticmethod
    # Menangani proses icon easy pada DifficultyScreen sambil menjaga state internal tetap konsisten.
    def _icon_easy(cv, sz, col):
        cx, cy = sz // 2, sz // 2
        cv.create_line(cx, cy + sz//3, cx, cy - sz//8,
                       fill=col, width=3, capstyle="round")
        cv.create_arc(cx - sz//3, cy - sz//3, cx + sz//8, cy + sz//12,
                      start=20, extent=140, fill=col, outline="")
        cv.create_arc(cx - sz//8, cy - sz//3, cx + sz//3, cy + sz//12,
                      start=20, extent=140, fill=col, outline="")
        cv.create_oval(cx - 3, cy + sz//3 - 3, cx + 3, cy + sz//3 + 3,
                       fill=col, outline="")

    @staticmethod
    # Menangani proses icon normal pada DifficultyScreen sambil menjaga state internal tetap konsisten.
    def _icon_normal(cv, sz, col):
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
    # Menangani proses icon hard pada DifficultyScreen sambil menjaga state internal tetap konsisten.
    def _icon_hard(cv, sz, col):
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

    # Menginisialisasi objek DifficultyScreen dan menyiapkan state awal, referensi penting, serta elemen yang dibutuhkan sebelum layar dipakai.
    def __init__(self, master, username, grid_size, on_select, on_back=None):
        super().__init__(master, bg=C_BG)
        self.username     = username
        self.current_user = username
        self.grid_size    = grid_size
        self.on_select    = on_select
        self.on_back      = on_back
        self.data         = load_data()
        self._build()

    # Main layout
    # Membangun bagian antarmuka pada DifficultyScreen dan menyiapkan widget supaya state tampilan tetap konsisten.
    def _build(self):
        self.place(relx=0, rely=0, relwidth=1, relheight=1)

        abg = AnimatedBG(self, bg=C_BG, highlightthickness=0)
        abg.place(relx=0, rely=0, relwidth=1, relheight=1)

        N          = self.grid_size * self.grid_size
        grid_label = f"{N}×{N}"

        sessions = self.data["players"].get(self.username, {}).get("sessions", [])
        ml       = PlayerMLEngine()
        ml.sessions = [s for s in sessions
                       if s.get("grid_size") == self.grid_size]
        rec           = ml.recommend_difficulty()
        p_type, feat  = ml.classify_player()
        has_history   = feat["sessions_count"] > 0

        hdr = tk.Frame(self, bg=C_SURFACE)
        hdr.pack(fill="x")

        gbar = tk.Canvas(hdr, height=8, bg=C_SURFACE, highlightthickness=0)
        gbar.pack(fill="x")
        gbar.after(100, lambda: draw_gradient_bar(gbar))

        hdr_inner = tk.Frame(hdr, bg=C_SURFACE)
        hdr_inner.pack(pady=18, fill="x")

        if callable(self.on_back):
            back_btn = tk.Button(
                hdr,
                text="←  Back to Grid Selection",
                font=("Segoe UI", 9),
                bg=C_SURFACE,
                fg=C_TEXT_DIM,
                activebackground=C_SURFACE2,
                activeforeground=C_TEXT,
                relief="flat",
                cursor="hand2",
                padx=8, pady=4,
                highlightbackground=C_BORDER,
                highlightthickness=1,
                command=lambda: (_play_sfx(_SFX_CLICK), self.on_back()),
            )
            back_btn.place(x=28, y=14)
            back_btn.bind("<Enter>", lambda _: back_btn.config(
                bg=C_SURFACE2, fg=C_TEXT))
            back_btn.bind("<Leave>", lambda _: back_btn.config(
                bg=C_SURFACE, fg=C_TEXT_DIM))

        N_label = self.grid_size * self.grid_size
        title_row = tk.Frame(hdr_inner, bg=C_SURFACE)
        title_row.pack()
        tk.Label(title_row, text="⚔️  ",
                 font=("Segoe UI", 20), bg=C_SURFACE, fg=C_PURPLE).pack(side="left")
        tk.Label(title_row,
                 text=f"Select Difficulties · {N_label}×{N_label}",
                 font=("Segoe UI", 22, "bold"),
                 bg=C_SURFACE, fg=C_TEXT).pack(side="left")

        subtitle = f"Bermain sebagai  @{self.current_user}  ·  pilih tingkat kesulitan untuk mulai"
        tk.Label(hdr_inner, text=subtitle,
                 font=("Segoe UI", 9), bg=C_SURFACE, fg=C_TEXT_DIM).pack(pady=(4, 0))

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
                     bg=rc, fg=C_BG, padx=6, pady=1).pack(side="left")
            tk.Label(badge_row,
                     text=f"   ·   Tipe: {p_type}",
                     font=("Segoe UI", 10), bg=C_SURFACE, fg=C_TEXT_DIM).pack(side="left")
        else:
            tk.Label(hdr_inner,
                     text="🤖  Pilih tingkat kesulitan untuk memulai!",
                     font=("Segoe UI", 10), bg=C_SURFACE, fg=C_TEXT_DIM).pack(pady=(6, 0))

        center_wrap = tk.Frame(self, bg=C_BG)
        center_wrap.pack(side="top", fill="both", expand=True)

        inner = tk.Frame(center_wrap, bg=C_BG)
        inner.place(relx=0.5, rely=0.5, anchor="center")

        sec = tk.Frame(inner, bg=C_BG)
        sec.grid(row=0, column=0, columnspan=3, pady=(0, 22))
        tk.Frame(sec, width=50, height=1, bg=C_BORDER).pack(side="left", padx=(0, 12))
        tk.Label(sec, text="TINGKAT KESULITAN",
                 font=("Segoe UI", 10, "bold"),
                 bg=C_BG, fg=C_TEXT_DIM).pack(side="left")
        tk.Frame(sec, width=50, height=1, bg=C_BORDER).pack(side="left", padx=(12, 0))

        diff_configs = {
            "Easy": {
                "color": "#7EE787", "tag_bg": C_SURFACE2,
                "pct":   int(DIFF_THEMES["Easy"]["remove_pct"] * 100),
                "lines": ["Cocok untuk pemula", "Waktu lebih santai", "Aturan dasar Sudoku"],
            },
            "Normal": {
                "color": "#58A6FF", "tag_bg": C_SURFACE2,
                "pct":   int(DIFF_THEMES["Normal"]["remove_pct"] * 100),
                "lines": ["Tantangan menengah", "Butuh strategi", "Format kompetitif"],
            },
            "Hard": {
                "color": "#FF7B7B", "tag_bg": C_SURFACE2,
                "pct":   int(DIFF_THEMES["Hard"]["remove_pct"] * 100),
                "lines": ["Untuk ahli Sudoku", "Konsentrasi penuh", "Draft mode tersedia"],
            },
        }
        for col_idx, (diff, cfg) in enumerate(diff_configs.items()):
            is_rec = (diff == rec and has_history)
            self._card(inner, col_idx, diff, cfg, is_rec)

        if has_history:
            foot = tk.Frame(self, bg=C_SURFACE)
            foot.pack(fill="x", side="bottom")

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
            foot = tk.Frame(self, bg=C_SURFACE, pady=10)
            foot.pack(fill="x", side="bottom")
            tk.Label(foot,
                     text="💡  Selesaikan puzzle pertamamu untuk melihat statistik dan rekomendasi AI",
                     font=("Segoe UI", 9), bg=C_SURFACE, fg=C_TEXT_DIM).pack()

    # Individual difficulty card
    # Menangani proses card pada DifficultyScreen sambil menjaga state internal tetap konsisten.
    def _card(self, parent, col_idx, diff, cfg, is_rec):
        color    = cfg["color"]
        CARD_W   = 220
        ICON_SZ  = 72
        painter  = self._ICON_PAINTERS.get(diff)

        outer = tk.Frame(parent, bg=C_BG)
        outer.grid(row=1, column=col_idx, padx=14, sticky="n")

        card = tk.Frame(outer, bg=C_SURFACE,
                        highlightbackground=color if is_rec else C_BORDER,
                        highlightthickness=2 if is_rec else 1,
                        cursor="hand2")
        card.pack()

        spacer = tk.Frame(card, bg=C_SURFACE, height=1, width=CARD_W)
        spacer.pack(side="top")
        spacer.lower()

        stripe = tk.Frame(card, bg=color, height=5)
        stripe.pack(fill="x")

        if is_rec:
            banner = tk.Frame(card, bg=color)
            banner.pack(fill="x")
            tk.Label(banner,
                     text="⭐  REKOMENDASI AI",
                     font=("Segoe UI", 8, "bold"),
                     bg=color, fg=C_BG,
                     pady=4).pack()

        icon_wrap = tk.Frame(card, bg=C_SURFACE)
        icon_wrap.pack(pady=(18, 4))
        cv = tk.Canvas(icon_wrap, width=ICON_SZ, height=ICON_SZ,
                       bg=C_SURFACE, highlightthickness=0)
        cv.pack()
        if painter:
            cv.after(80, lambda c=cv, p=painter, sz=ICON_SZ, col=color:
                     p(c, sz, col))

        tk.Label(card, text=diff.upper(),
                 font=("Segoe UI", 20, "bold"),
                 bg=C_SURFACE, fg=color).pack()

        pct_row = tk.Frame(card, bg=C_SURFACE)
        pct_row.pack(pady=(4, 0))
        pct_pill = tk.Frame(pct_row, bg=cfg["tag_bg"])
        pct_pill.pack()
        tk.Label(pct_pill,
                 text=f"  {cfg['pct']}% sel dikosongkan  ",
                 font=("Segoe UI", 8, "bold"),
                 bg=cfg["tag_bg"], fg=color,
                 pady=3).pack()

        tk.Frame(card, height=1, bg=C_BORDER).pack(fill="x", padx=14, pady=(14, 10))

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

        btn_wrap = tk.Frame(card, bg=C_SURFACE)
        btn_wrap.pack(fill="x", padx=14, pady=(14, 18))
        tk.Button(btn_wrap,
                  text=f"▶   MULAI {diff.upper()}",
                  font=("Segoe UI", 10, "bold"),
                  bg=color, fg=C_BG,
                  activebackground="#FFFFFF",
                  activeforeground=C_BG,
                  relief="flat", cursor="hand2", pady=10,
                  command=lambda d=diff: (_play_sfx(_SFX_SELECT), self.on_select(d))).pack(fill="x")

        # Menangani interaksi enter pada DifficultyScreen agar respons UI tetap konsisten.
        def _enter(_):
            card.config(highlightbackground=color, highlightthickness=2)
            _play_sfx(_SFX_HOVER)

        # Menangani interaksi leave pada DifficultyScreen agar respons UI tetap konsisten.
        def _leave(_):
            card.config(highlightbackground=color if is_rec else C_BORDER,
                        highlightthickness=2 if is_rec else 1)

        hover_targets = [card, icon_wrap, cv, feat_wrap, btn_wrap, spacer, stripe]
        for w in hover_targets:
            w.bind("<Enter>", _enter)
            w.bind("<Leave>", _leave)
        for w in [card, icon_wrap, feat_wrap, spacer]:
            w.bind("<Button-1>", lambda _, d=diff: (_play_sfx(_SFX_SELECT), self.on_select(d)))

# LEADERBOARD WINDOW  (tabbed by grid+difficulty)
class LeaderboardWindow(tk.Frame):
    """
    LeaderboardWindow - Overlay popup papan peringkat dengan animasi slide-up dan blur.

    Deskripsi:
        Menampilkan sebagai overlay penuh di atas layar aktif (bukan jendela baru).
        Warna menyesuaikan tema aktif (dark/light) secara otomatis saat dibuka.
        Filter berdasarkan grid (4×4/9×9) dan difficulty (All/Easy/Normal/Hard).
        Mendukung scroll mouse wheel; fallback top 25 jika scroll bermasalah.
        Animasi: slide-up dari bawah layar dalam 220ms.

    Atribut:
        data (dict): Data seluruh pemain dari load_data().
        _active_grid (str): Filter grid aktif (default ``"9x9"``).
        _active_diff (str): Filter difficulty aktif (default ``"All"``).
        _pal (dict): Palet warna dinamis sesuai tema saat overlay dibuka.
    """

    COL_RATIOS = [0.08, 0.22, 0.16, 0.14, 0.12, 0.12, 0.16]
    COL_HEADS  = ["RANK", "PLAYER", "DIFFICULTY", "TIME", "MOVES", "ERRORS", "SCORE"]

    # Medal colors - tidak bergantung tema
    _C_BRONZE = "#CD7F32"

    @staticmethod
    # Membangun palette pada LeaderboardWindow dan menyiapkan widget supaya state tampilan tetap konsisten.
    def _build_palette() -> dict:
        t    = _DARK_THEME if _CURRENT_THEME_NAME == "dark" else _LIGHT_THEME
        dark = _CURRENT_THEME_NAME == "dark"

        return {
            "card":        t["C_BG"],
            "header":      t["C_SIDEBAR"],
            "row_a":       t["C_SURFACE"],
            "row_b":       t["C_BG"],
            "filter_bar":  t["C_SIDEBAR"],
            "border":      t["C_BORDER"],
            "footer":      t["C_SIDEBAR"],

            "btn_inactive_bg": t["C_SURFACE2"],
            "btn_inactive_fg": t["C_TEXT_DIM"],
            "btn_close_bg":    t["C_SURFACE2"],
            "btn_close_fg":    t["C_TEXT_DIM"],

            "text":         t["C_TEXT"],
            "text_dim":     t["C_TEXT_DIM"],
            "text_header":  t["C_TEXT_DIM"],

            "gold":         t["C_GOLD"],
            "silver":       t["C_TEXT_DIM"],
            "accent_gold":  t["C_GOLD"],

            "diff_easy":    t["C_ACCENT2"],
            "diff_normal":  t["C_ACCENT"],
            "diff_hard":    t["C_ERROR"],

            "row_hover":    t["C_SURFACE2"],

            "dim": "#060d18" if dark else "#2a3550",
        }

    # Menginisialisasi objek LeaderboardWindow dan menyiapkan state awal, referensi penting, serta elemen yang dibutuhkan sebelum layar dipakai.
    def __init__(self, master, data):
        root = master if isinstance(master, tk.Tk) else master.winfo_toplevel()
        super().__init__(root, bg="#000000")
        self.root   = root
        self.data   = data
        self._active_grid = "9x9"
        self._active_diff = "All"
        self._row_canvases = []
        self._blur_pil = None
        self._pal = self._build_palette()
        _bg_photo = None
        if PIL_AVAILABLE:
            try:
                root.update_idletasks()
                pil_img = _grab_blur_pil(root, radius=14, darken=0.40)
                self._blur_pil = pil_img
                if pil_img:
                    _bg_photo = _PilImageTk.PhotoImage(pil_img)
            except Exception:
                pass

        self.place(relx=0, rely=0, relwidth=1, relheight=1)
        self.lift()

        if _bg_photo:
            bg_lbl = tk.Label(self, image=_bg_photo, bd=0, highlightthickness=0)
            bg_lbl._blur_photo_ref = _bg_photo
            bg_lbl.place(relx=0, rely=0, relwidth=1, relheight=1)
            bg_lbl.bind("<Button-1>", lambda e: self._close())
        else:
            fb = tk.Frame(self, bg="#1a2030")
            fb.place(relx=0, rely=0, relwidth=1, relheight=1)
            fb.bind("<Button-1>", lambda e: self._close())

        SW = root.winfo_screenwidth()
        SH = root.winfo_screenheight()
        self._card_w = min(820, int(SW * 0.82))
        self._card_h = min(620, int(SH * 0.85))
        self._card = tk.Frame(self, bg=self._pal["card"],
                              highlightbackground=self._pal["border"],
                              highlightthickness=1)
        self._anim_step = 0
        self._anim_total = 12
        self._card.place(relx=0.5, rely=0.65,
                         anchor="center",
                         width=self._card_w, height=self._card_h)
        self._build(self._card)
        self.bind("<Button-1>", lambda e: self._close()
                  if e.widget is self else None)
        if _APP_INSTANCE is not None:
            _APP_INSTANCE._active_overlay = self
        _corner_icons_lower()
        self.focus_set()
        self.bind("<Escape>", lambda e: (self._close(), "break"))
        self.after(16, self._animate)

    # Menangani proses animate pada LeaderboardWindow sambil menjaga state internal tetap konsisten.
    def _animate(self):
        self._anim_step += 1
        t = self._anim_step / self._anim_total
        ease = 1 - (1 - t) ** 3
        target_rely = 0.5
        start_rely  = 0.65
        current = start_rely + (target_rely - start_rely) * ease
        self._card.place(relx=0.5, rely=current,
                         anchor="center",
                         width=self._card_w, height=self._card_h)
        if self._anim_step < self._anim_total:
            self.after(16, self._animate)

    # Menangani proses close pada LeaderboardWindow sambil menjaga state internal tetap konsisten.
    def _close(self):
        if _APP_INSTANCE is not None:
            if getattr(_APP_INSTANCE, "_active_overlay", None) is self:
                _APP_INSTANCE._active_overlay = None
            _APP_INSTANCE._last_screen_change = time.time()
        _corner_icons_restore()
        try:
            self.destroy()
        except Exception:
            pass

    # Membangun bagian antarmuka pada LeaderboardWindow dan menyiapkan widget supaya state tampilan tetap konsisten.
    def _build(self, card):
        hdr = tk.Frame(card, bg=self._pal["header"], pady=0)
        hdr.pack(fill="x")

        tk.Frame(hdr, height=3, bg=self._pal["gold"]).pack(fill="x")

        title_row = tk.Frame(hdr, bg=self._pal["header"])
        title_row.pack(pady=(14, 10))

        tk.Label(title_row, text="🏆",
                 font=("Segoe UI", 26), bg=self._pal["header"],
                 fg=self._pal["gold"]).pack(side="left", padx=(0, 10))

        title_col = tk.Frame(title_row, bg=self._pal["header"])
        title_col.pack(side="left")
        tk.Label(title_col, text="HALL OF FAME",
                 font=("Segoe UI", 20, "bold"),
                 bg=self._pal["header"], fg=self._pal["gold"]).pack(anchor="w")
        tk.Label(title_col, text="Performa terbaik semua pemain",
                 font=("Segoe UI", 9),
                 bg=self._pal["header"], fg=self._pal["text_dim"]).pack(anchor="w")

        filter_bar = tk.Frame(card, bg=self._pal["filter_bar"], pady=0)
        filter_bar.pack(fill="x")
        tk.Frame(filter_bar, height=1, bg=self._pal["border"]).pack(fill="x")

        inner_filter = tk.Frame(filter_bar, bg=self._pal["filter_bar"])
        inner_filter.pack(pady=8, padx=16, anchor="w")

        tk.Label(inner_filter, text="GRID",
                 font=("Segoe UI", 8, "bold"),
                 bg=self._pal["filter_bar"], fg=self._pal["text_dim"]).pack(side="left", padx=(0, 6))
        self._grid_btns = {}
        for lbl in ["4x4", "9x9"]:
            b = tk.Button(inner_filter, text=lbl,
                          font=("Segoe UI", 8, "bold"),
                          relief="flat", cursor="hand2",
                          padx=12, pady=4, bd=0,
                          command=lambda g=lbl: self._set_grid(g))
            b.pack(side="left", padx=2)
            self._grid_btns[lbl] = b

        tk.Frame(inner_filter, width=1, bg=self._pal["border"]).pack(
            side="left", fill="y", padx=10)

        tk.Label(inner_filter, text="MODE",
                 font=("Segoe UI", 8, "bold"),
                 bg=self._pal["filter_bar"], fg=self._pal["text_dim"]).pack(side="left", padx=(0, 6))
        self._diff_btns = {}
        for lbl in ["All", "Easy", "Normal", "Hard"]:
            b = tk.Button(inner_filter, text=lbl,
                          font=("Segoe UI", 8, "bold"),
                          relief="flat", cursor="hand2",
                          padx=12, pady=4, bd=0,
                          command=lambda d=lbl: self._set_diff(d))
            b.pack(side="left", padx=2)
            self._diff_btns[lbl] = b

        tk.Frame(filter_bar, height=1, bg=self._pal["border"]).pack(fill="x")

        self._table_host = tk.Frame(card, bg=self._pal["card"])
        self._table_host.pack(fill="both", expand=True, padx=0, pady=0)

        foot = tk.Frame(card, bg=self._pal["footer"])
        foot.pack(fill="x")
        tk.Frame(foot, height=1, bg=self._pal["border"]).pack(fill="x")
        close_btn = tk.Button(foot, text="✕  TUTUP",
                              font=("Segoe UI", 10, "bold"),
                              bg=self._pal["btn_close_bg"], fg=self._pal["btn_close_fg"],
                              relief="flat", cursor="hand2",
                              pady=10, bd=0,
                              activebackground="#c0392b",
                              activeforeground="#ffffff",
                              command=self._close)
        close_btn.pack(fill="x", padx=0)

        close_btn.bind("<Enter>", lambda e: close_btn.config(
            bg="#E53935", fg="#ffffff"))
        close_btn.bind("<Leave>", lambda e: close_btn.config(
            bg=self._pal["btn_close_bg"], fg=self._pal["btn_close_fg"]))

        self._refresh()

    # Mengatur grid pada LeaderboardWindow dan menerapkan nilai yang dipilih ke state internal.
    def _set_grid(self, g):
        self._active_grid = g
        self._refresh()

    # Mengatur diff pada LeaderboardWindow dan menerapkan nilai yang dipilih ke state internal.
    def _set_diff(self, d):
        self._active_diff = d
        self._refresh()

    # Menyegarkan tampilan pada LeaderboardWindow setelah data atau pilihan pengguna berubah.
    def _refresh(self):
        DIFF_COLORS = {
            "Easy":   self._pal["diff_easy"],
            "Normal": self._pal["diff_normal"],
            "Hard":   self._pal["diff_hard"],
            "All":    self._pal["text"],
        }
        txt_on_color = "#ffffff" if _CURRENT_THEME_NAME == "dark" else "#ffffff"
        for lbl, btn in self._grid_btns.items():
            active = lbl == self._active_grid
            btn.config(
                bg=self._pal["gold"]            if active else self._pal["btn_inactive_bg"],
                fg=txt_on_color                 if active else self._pal["btn_inactive_fg"])
        for lbl, btn in self._diff_btns.items():
            active = lbl == self._active_diff
            col = DIFF_COLORS.get(lbl, self._pal["text"])
            if active:
                fg_active = self._pal["card"] if lbl == "All" else txt_on_color
                btn.config(bg=col, fg=fg_active)
            else:
                btn.config(bg=self._pal["btn_inactive_bg"], fg=col)

        for w in self._table_host.winfo_children():
            w.destroy()
        self._row_canvases.clear()

        box = 2 if self._active_grid == "4x4" else 3
        best = {}
        for uname, pdata in self.data.get("players", {}).items():
            for s in pdata.get("sessions", []):
                if not s.get("completed"):
                    continue
                if s.get("grid_size", 3) != box:
                    continue
                diff = s.get("difficulty", "Normal")
                if self._active_diff != "All" and diff != self._active_diff:
                    continue
                t = s.get("total_time", 0)
                sc = s.get("score") or calculate_score(
                    diff, t,
                    s.get("empty_cells", max(s.get("moves", 1), 1)),
                    s.get("errors", 0), s.get("hints_used", 0), True,
                    s.get("near_miss", 0), s.get("guessing", 0))
                entry = {"username": uname, "difficulty": diff,
                         "time": t, "moves": s.get("moves", 0),
                         "errors": s.get("errors", 0), "score": sc}
                if uname not in best or sc > best[uname]["score"]:
                    best[uname] = entry

        entries = sorted(best.values(), key=lambda x: -x["score"])

        self.update_idletasks()
        avail_w = self._table_host.winfo_width() or (self._card_w - 2)
        if avail_w < 200:
            avail_w = self._card_w - 2
        col_ws = [max(30, int(avail_w * r)) for r in self.COL_RATIOS]
        col_ws[-1] += avail_w - sum(col_ws)

        self._draw_header(col_ws, avail_w)

        if not entries:
            tk.Label(self._table_host,
                     text="Belum ada data, jadilah yang pertama! 🚀",
                     font=("Segoe UI", 11),
                     bg=self._pal["card"], fg=self._pal["text_dim"]).pack(pady=40)
            return

        scroll_frame = tk.Frame(self._table_host, bg=self._pal["card"])
        scroll_frame.pack(fill="both", expand=True)

        cv = tk.Canvas(scroll_frame, bg=self._pal["card"],
                       highlightthickness=0)
        sb = tk.Scrollbar(scroll_frame, orient="vertical",
                          command=cv.yview, width=8)
        inner = tk.Frame(cv, bg=self._pal["card"])
        inner.bind("<Configure>",
                   lambda e: cv.configure(scrollregion=cv.bbox("all")))
        cv.create_window((0, 0), window=inner, anchor="nw")
        cv.configure(yscrollcommand=sb.set)

        cv.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

        # Menangani event wheel pada LeaderboardWindow dan memperbarui state yang terkait.
        def _on_wheel(e):
            try:
                cv.yview_scroll(int(-1 * (e.delta / 120)), "units")
            except Exception:
                pass
        cv.bind("<MouseWheel>", _on_wheel)
        cv.bind("<Button-4>",   lambda e: cv.yview_scroll(-1, "units"))
        cv.bind("<Button-5>",   lambda e: cv.yview_scroll(+1, "units"))
        inner.bind("<MouseWheel>", _on_wheel)

        MEDAL  = {1: ("🥇", self._pal["gold"]),
                  2: ("🥈", self._pal["silver"]),
                  3: ("🥉", self._C_BRONZE)}
        DC     = {"Easy":   self._pal["diff_easy"],
                  "Normal": self._pal["diff_normal"],
                  "Hard":   self._pal["diff_hard"]}

        for idx, e in enumerate(entries[:25], 1):
            medal, rank_col = MEDAL.get(idx, (str(idx), self._pal["text_dim"]))
            t = e["time"]
            ts = f"{int(t//60):02}:{int(t%60):02}"
            err_col = self._pal["diff_hard"] if e["errors"] > 5 else self._pal["text_dim"]

            vals = [
                (medal,                                rank_col),
                (e["username"],                        self._pal["text"]),
                (e["difficulty"],                      DC.get(e["difficulty"], self._pal["text_dim"])),
                (ts,                                   self._pal["text"]),
                (str(e["moves"]),                      self._pal["text"]),
                (str(e["errors"]),                     err_col),
                (str(e["score"]),                      self._pal["gold"]),
            ]

            row_bg = self._pal["row_a"] if idx % 2 == 0 else self._pal["row_b"]

            if idx <= 3:
                _, rank_c = MEDAL[idx]
                self._draw_row(inner, vals, row_bg, col_ws,
                               avail_w, accent=rank_c, top3=True)
            else:
                self._draw_row(inner, vals, row_bg, col_ws, avail_w)

    # Menggambar header pada LeaderboardWindow sesuai state yang sedang aktif.
    def _draw_header(self, col_ws, total_w):
        H = 36
        cv = tk.Canvas(self._table_host, height=H, width=total_w,
                       bg=self._pal["header"], highlightthickness=0)
        cv.pack(fill="x")

        cv.create_line(0, H-1, total_w, H-1, fill=self._pal["border"], width=1)

        x = 0
        for head, w in zip(self.COL_HEADS, col_ws):
            cv.create_text(x + w // 2, H // 2,
                           text=head, fill=self._pal["text_header"],
                           font=("Segoe UI", 8, "bold"),
                           anchor="center")
            x += w

    # Menggambar row pada LeaderboardWindow sesuai state yang sedang aktif.
    def _draw_row(self, parent, vals, bg, col_ws, total_w,
                  accent=None, top3=False):
        H = 38
        cv = tk.Canvas(parent, height=H, width=total_w,
                       bg=bg, highlightthickness=0)
        cv.pack(fill="x")

        cv.create_line(0, H-1, total_w, H-1, fill=self._pal["border"], width=1)

        if top3 and accent:
            cv.create_rectangle(0, 2, 3, H-2, fill=accent, outline="")

        x = 0
        for (txt, fg), w in zip(vals, col_ws):
            font = ("Segoe UI", 10, "bold") if top3 else ("Segoe UI", 10)
            cv.create_text(x + w // 2, H // 2,
                           text=txt, fill=fg,
                           font=font, anchor="center")
            x += w

        # Menangani event enter pada LeaderboardWindow dan memperbarui state yang terkait.
        def _on_enter(e, c=cv, b=bg):
            c.configure(bg=self._pal["row_hover"])
            c.itemconfigure("all", fill="")
        # Menangani event leave pada LeaderboardWindow dan memperbarui state yang terkait.
        def _on_leave(e, c=cv, b=bg):
            c.configure(bg=b)

        cv.bind("<Enter>", _on_enter)
        cv.bind("<Leave>", _on_leave)

        # Menangani proses find canvas scroll pada LeaderboardWindow sambil menjaga state internal tetap konsisten.
        def _find_canvas_scroll(widget):
            try:
                p = widget.master
                while p:
                    if isinstance(p, tk.Canvas) and p.cget("yscrollcommand"):
                        return p
                    p = getattr(p, "master", None)
            except Exception:
                pass
            return None

        # Menangani proses wheel pada LeaderboardWindow sambil menjaga state internal tetap konsisten.
        def _wheel(e, c=cv):
            sc = _find_canvas_scroll(c)
            if sc:
                try:
                    sc.yview_scroll(int(-1 * (e.delta / 120)), "units")
                except Exception:
                    pass

        cv.bind("<MouseWheel>", _wheel)
        cv.bind("<Button-4>",   lambda e: _wheel(type("E", (), {"delta": 120})()))
        cv.bind("<Button-5>",   lambda e: _wheel(type("E", (), {"delta": -120})()))


# PERFORMANCE DASHBOARD
class PerformanceDashboard(tk.Frame):
    """
    PerformanceDashboard - Dashboard analitik pasca-game dengan integrasi ML.

    Deskripsi:
        Menampilkan tipe pemain (KNN), prediksi skor, deteksi anomali,
        rekomendasi difficulty, profil skill, chart riwayat, dan pencapaian.

    Atribut:
        controller (SudokuApp): Referensi ke app untuk navigasi.
        session (dict): Sesi terakhir yang selesai.
        ml (PlayerMLEngine): Instance ML engine.
        recommended_difficulty (str): Rekomendasi dari model RFC.
    """
    # Menginisialisasi objek PerformanceDashboard dan menyiapkan state awal, referensi penting, serta elemen yang dibutuhkan sebelum layar dipakai.
    def __init__(self, master, controller, username, session, ml):
        super().__init__(master, bg=C_BG)
        self.controller = controller
        self.session    = session
        self.ml         = ml
        self.username   = username

        self.recommended_grid = 3
        self.recommended_difficulty = "Normal"
        self.recommended_reason = ""

        self._scroll_canvas = None
        self._mousewheel_bound = False

        self._chart_metric_var = tk.StringVar(master=self, value="score")
        self._chart_selected_index = None
        self._chart_canvas = None
        self._chart_fig = None
        self._chart_ax = None
        self._chart_info_var = tk.StringVar(master=self, value="")
        self._chart_point_meta = []
        self._chart_host = None
        self._chart_canvas_widget = None
        self._chart_selected_marker = None
        self._chart_metric_buttons = {}

        self.place(relx=0, rely=0, relwidth=1, relheight=1)
        self._build()

    # BUILD
    # Membangun bagian antarmuka pada PerformanceDashboard dan menyiapkan widget supaya state tampilan tetap konsisten.
    def _build(self):
        s     = self.session
        stats = self.ml.get_summary()
        pt    = stats["player_type"]
        feat  = stats["features"]
        rec   = stats["recommended_difficulty"]
        ti    = stats["type_info"]
        rc    = {"Easy": "#7EE787", "Normal": "#58A6FF", "Hard": "#FF7B7B"}
        tc    = ti.get("color", C_ACCENT)

        recommended_grid = int(stats.get("recommended_grid_size", 2 if rec == "Easy" else 3))
        if recommended_grid not in (2, 3):
            recommended_grid = 2 if rec == "Easy" else 3
        self.recommended_grid = recommended_grid
        self.recommended_difficulty = stats.get("recommended_difficulty", rec) or rec
        self.recommended_reason = stats.get("recommended_reason", "") or self._build_recommendation_reason(pt, feat, self.recommended_difficulty)

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

        scroll_row = tk.Frame(self, bg=C_BG)
        scroll_row.pack(fill="both", expand=True)
        sbr = tk.Scrollbar(scroll_row, orient="vertical")
        sbr.pack(side="right", fill="y")
        sc = tk.Canvas(scroll_row, bg=C_BG, highlightthickness=0, yscrollcommand=sbr.set)
        sc.pack(side="left", fill="both", expand=True)
        sbr.config(command=sc.yview)
        body = tk.Frame(sc, bg=C_BG)
        wid  = sc.create_window((0,0), window=body, anchor="nw")
        body.bind("<Configure>", lambda _: sc.configure(scrollregion=sc.bbox("all")))
        sc.bind("<Configure>",   lambda e: sc.itemconfig(wid, width=e.width))
        self._scroll_canvas = sc
        self._bind_mousewheel()

        pw = tk.Frame(body, bg=C_BG)
        pw.pack(fill="x", padx=24, pady=18)

        hero = tk.Frame(pw, bg=C_BG)
        hero.pack(fill="x", pady=(0, 18))

        left_col = tk.Frame(hero, bg=C_BG, width=360)
        left_col.pack(side="left", fill="y", padx=(0, 14))
        left_col.pack_propagate(False)

        right_col = tk.Frame(hero, bg=C_BG)
        right_col.pack(side="left", fill="both", expand=True)

        self._build_type_card(left_col, pt, ti, tc, feat, stats)
        self._build_recommendation_card(left_col, rec, tc)
        self._build_chart_card(right_col, stats)

        snap_subtitle = ("Kamu menyelesaikan puzzle ini!" if s.get("completed")
                         else "Sesi ini berakhir dengan kekalahan.")
        self._section_title(pw, "STATISTIK SESI INI", snap_subtitle)
        self._build_stat_tiles(pw, s)

        insight_row = tk.Frame(pw, bg=C_BG)
        insight_row.pack(fill="x", pady=(0, 16))

        left_insight = tk.Frame(insight_row, bg=C_BG)
        left_insight.pack(side="left", fill="both", expand=True, padx=(0, 12))
        right_insight = tk.Frame(insight_row, bg=C_BG)
        right_insight.pack(side="left", fill="both", expand=True)

        self._build_behavior_card(left_insight, s, feat)
        self._build_skill_ml_card(right_insight, feat, stats, s)

        self._build_achievements_card(pw)

        self._section_title(pw, "AKSI CEPAT", "Lanjut main, eksplor rekomendasi, atau keluar.")
        btns = tk.Frame(pw, bg=C_BG)
        btns.pack(fill="x", pady=(0, 6))

        _gold_fg = C_BG if _CURRENT_THEME_NAME == "dark" else C_WHITE

        self._action_button(btns, "🔄  PLAY AGAIN", C_ACCENT, C_BG,
                            lambda: self.master.event_generate("<<PlayAgain>>"),
                            side="left", padx=(0, 6))
        self._action_button(btns, "🤖  COBA REKOMENDASI AI", C_PURPLE, C_BG,
                            self._start_recommendation,
                            side="left", padx=6)
        self._action_button(btns, "🏆  LEADERBOARD", C_GOLD, _gold_fg,
                            lambda: LeaderboardWindow(self.winfo_toplevel(), load_data()),
                            side="left", padx=6)
        self._action_button(btns, "🚪  LOGOUT", C_ERROR, C_WHITE,
                            lambda: self.master.event_generate("<<Logout>>"),
                            side="left", padx=(6,0))

        btns2 = tk.Frame(pw, bg=C_BG)
        btns2.pack(fill="x", pady=(4, 0))
        self._action_button(btns2, "🖼  SIMPAN SCORECARD", C_GOLD, _gold_fg,
                            self._save_score_card_action,
                            side="left", padx=0)

    # Membangun achievements card pada PerformanceDashboard dan menyiapkan widget supaya state tampilan tetap konsisten.
    def _build_achievements_card(self, parent):
        _data          = load_data()
        _pdata         = _data["players"].get(self.username, {})
        unlocked       = set(_pdata.get("achievements", []))
        total_ach      = len(ACHIEVEMENTS)
        total_unlocked = len(unlocked)

        self._section_title(
            parent,
            "ACHIEVEMENTS",
            f"{total_unlocked}/{total_ach} badge diraih"
        )

        outer = tk.Frame(parent, bg=C_SURFACE,
                         highlightbackground=C_BORDER, highlightthickness=1)
        outer.pack(fill="x", pady=(0, 16))

        prog_host = tk.Frame(outer, bg=C_SURFACE2, height=4)
        prog_host.pack(fill="x", padx=14, pady=(10, 6))
        prog_host.pack_propagate(False)
        fill_pct  = total_unlocked / max(total_ach, 1)
        prog_fill = tk.Frame(prog_host,
                             bg=C_GOLD if fill_pct >= 1.0 else C_ACCENT, height=4)
        prog_fill.place(relx=0, rely=0, relwidth=fill_pct, relheight=1.0)

        ordered = (
            [aid for aid in ACHIEVEMENTS if aid     in unlocked] +
            [aid for aid in ACHIEVEMENTS if aid not in unlocked]
        )

        COLS        = 5
        badge_frame = tk.Frame(outer, bg=C_SURFACE)
        badge_frame.pack(fill="x", padx=10, pady=(4, 10))

        for idx, aid in enumerate(ordered):
            info   = ACHIEVEMENTS.get(aid, {})
            nama   = info.get("nama", aid)
            label  = info.get("emoji", "?")
            warna  = info.get("warna", "#58A6FF")
            desc   = info.get("desc", "")
            is_on  = aid in unlocked

            r, c = divmod(idx, COLS)

            bg_col   = C_SURFACE2
            border   = warna  if is_on else C_BORDER
            nama_col = C_TEXT if is_on else C_TEXT_DIM
            dim_col  = C_TEXT_DIM

            cell = tk.Frame(badge_frame, bg=bg_col,
                            highlightbackground=border, highlightthickness=1)
            cell.grid(row=r, column=c, padx=4, pady=3, sticky="nsew")

            hdr = tk.Frame(cell, bg=bg_col)
            hdr.pack(fill="x", padx=6, pady=(7, 0))

            dot_cv = tk.Canvas(hdr, width=12, height=12, bg=bg_col,
                               highlightthickness=0)
            dot_cv.pack(side="left")
            dot_color = warna if is_on else C_BORDER
            dot_cv.create_oval(1, 1, 11, 11, fill=dot_color, outline="")

            tk.Label(hdr,
                     text=("UNLOCKED" if is_on else "LOCKED"),
                     font=("Segoe UI", 6),
                     bg=bg_col,
                     fg=warna if is_on else C_BORDER).pack(side="right")

            tk.Label(cell, text=nama,
                     font=("Segoe UI", 8, "bold" if is_on else "normal"),
                     bg=bg_col, fg=nama_col,
                     wraplength=145, justify="left",
                     anchor="w").pack(fill="x", padx=7, pady=(3, 0))

            if desc:
                tk.Label(cell, text=desc,
                         font=("Segoe UI", 7),
                         bg=bg_col, fg=dim_col,
                         wraplength=145, justify="left",
                         anchor="w").pack(fill="x", padx=7, pady=(2, 6))
            else:
                tk.Frame(cell, bg=bg_col, height=6).pack()

        for ci in range(COLS):
            badge_frame.columnconfigure(ci, weight=1)

        if total_unlocked < total_ach:
            remaining = total_ach - total_unlocked
            tk.Label(outer,
                     text=f"  {remaining} badge lagi menunggumu. Terus bermain!",
                     font=("Segoe UI", 8), bg=C_SURFACE, fg=C_TEXT_DIM,
                     anchor="w").pack(fill="x", padx=14, pady=(0, 8))

    # HERO CARDS
    # Membangun type card pada PerformanceDashboard dan menyiapkan widget supaya state tampilan tetap konsisten.
    def _build_type_card(self, parent, pt, ti, tc, feat, stats):
        card = tk.Frame(parent, bg=C_SURFACE, highlightbackground=tc, highlightthickness=2)
        card.pack(fill="x", pady=(0, 14))

        top = tk.Frame(card, bg=C_SURFACE)
        top.pack(fill="x", padx=16, pady=(14, 6))
        badge = tk.Label(top, text="TIPE PEMAINMU",
                         font=("Segoe UI", 9, "bold"),
                         bg=C_SURFACE, fg=C_TEXT_DIM)
        badge.pack(anchor="w")

        mid = tk.Frame(card, bg=C_SURFACE)
        mid.pack(fill="x", padx=16)

        icon_box = tk.Frame(mid, bg=C_SURFACE)
        icon_box.pack(side="left", padx=(0, 14))
        icon = tk.Label(icon_box, text=ti.get("emoji", "🎮"),
                        font=("Segoe UI", 40), bg=C_SURFACE, fg=tc)
        icon.pack()

        text_box = tk.Frame(mid, bg=C_SURFACE)
        text_box.pack(side="left", fill="both", expand=True)
        tk.Label(text_box, text=pt, font=("Segoe UI", 20, "bold"),
                 bg=C_SURFACE, fg=tc).pack(anchor="w")
        tk.Label(text_box, text=ti.get("desc", ""),
                 font=("Segoe UI", 10), bg=C_SURFACE, fg=C_TEXT_DIM).pack(anchor="w", pady=(2,0))
        tk.Label(text_box,
                 text=f"{feat['avg_time_per_cell']:.1f}s/sel  ·  {feat['error_rate']*100:.0f}% error  ·  {feat['completion_rate']*100:.0f}% selesai",
                 font=("Segoe UI", 8),
                 bg=C_SURFACE, fg=C_TEXT_DIM, wraplength=220, justify="left").pack(anchor="w", pady=(8,0))

        chips = tk.Frame(card, bg=C_SURFACE)
        chips.pack(fill="x", padx=16, pady=(12, 14))
        grid_val = "4×4" if self.recommended_grid == 2 else "9×9"
        self._chip(chips, "Grid", grid_val, C_ACCENT)
        self._chip(chips, "Confidence", f"{stats.get('ml_confidence', 0):.0f}%", C_WARN if stats.get("ml_confidence", 0) < 70 else C_ACCENT2)
        self._chip(chips, "Sessions", str(feat.get("sessions_count", 0)), C_PURPLE)

    # Membangun recommendation card pada PerformanceDashboard dan menyiapkan widget supaya state tampilan tetap konsisten.
    def _build_recommendation_card(self, parent, rec, tc):
        card = tk.Frame(parent, bg=C_SURFACE, highlightbackground=tc, highlightthickness=1)
        card.pack(fill="x")
        top = tk.Frame(card, bg=C_SURFACE)
        top.pack(fill="x", padx=16, pady=(14, 4))
        tk.Label(top, text="REKOMENDASI AI",
                 font=("Segoe UI", 9, "bold"), bg=C_SURFACE, fg=C_TEXT_DIM).pack(anchor="w")

        title_row = tk.Frame(card, bg=C_SURFACE)
        title_row.pack(fill="x", padx=16, pady=(0, 2))
        tk.Label(title_row, text=self.recommended_difficulty,
                 font=("Segoe UI", 22, "bold"), bg=C_SURFACE,
                 fg={"Easy": "#7EE787", "Normal": "#58A6FF", "Hard": "#FF7B7B"}.get(self.recommended_difficulty, C_ACCENT)).pack(side="left")
        grid_txt = "4×4 (2×2 box)" if self.recommended_grid == 2 else "9×9 (3×3 box)"
        tk.Label(title_row, text=f"  •  {grid_txt}",
                 font=("Segoe UI", 10), bg=C_SURFACE, fg=C_TEXT_DIM).pack(side="left", pady=(10,0))

        reason = self.recommended_reason or "Rekomendasi ini disusun dari pola kecepatan, akurasi, dan konsistensi sesi terakhir."
        tk.Label(card, text=reason, font=("Segoe UI", 9),
                 bg=C_SURFACE, fg=C_TEXT, wraplength=280, justify="left").pack(anchor="w", padx=16, pady=(8, 10))

        row = tk.Frame(card, bg=C_SURFACE)
        row.pack(fill="x", padx=16, pady=(0, 14))
        self._action_button(row, "▶  Coba Sekarang", tc, C_BG,
                            self._start_recommendation, side="left", padx=0)

    # CHART
    # Membangun chart card pada PerformanceDashboard dan menyiapkan widget supaya state tampilan tetap konsisten.
    def _build_chart_card(self, parent, stats):
        card = tk.Frame(parent, bg=C_SURFACE, highlightbackground=C_BORDER, highlightthickness=1)
        card.pack(fill="both", expand=True)

        head = tk.Frame(card, bg=C_SURFACE)
        head.pack(fill="x", padx=16, pady=(14, 10))
        left = tk.Frame(head, bg=C_SURFACE)
        left.pack(side="left")
        tk.Label(left, text="PERFORMANCE TREND",
                 font=("Segoe UI", 10, "bold"), bg=C_SURFACE, fg=C_TEXT).pack(anchor="w")
        tk.Label(left, text="Grafik interaktif riwayat sesi",
                 font=("Segoe UI", 8), bg=C_SURFACE, fg=C_TEXT_DIM).pack(anchor="w")

        chips = tk.Frame(head, bg=C_SURFACE)
        chips.pack(side="right")
        _is_hard_session = (self.session or {}).get("difficulty", "") == "Hard"
        _chart_metrics = [("Skor", "score"), ("Waktu", "time"), ("Errors", "errors"), ("Hints", "hints")]
        if _is_hard_session:
            _chart_metrics.append(("Auto", "auto"))
        for label, metric in _chart_metrics:
            self._metric_chip(chips, label, metric)
        self._update_metric_chip_states()

        chart_outer = tk.Frame(card, bg=C_SURFACE2, highlightbackground=C_BORDER, highlightthickness=1)
        chart_outer.pack(fill="both", expand=True, padx=16, pady=(0, 10))

        self._chart_host = chart_outer
        self._render_chart(self._chart_metric_var.get())

        info_bar = tk.Frame(card, bg=C_SURFACE)
        info_bar.pack(fill="x", padx=16, pady=(0, 14))
        tk.Label(info_bar, textvariable=self._chart_info_var,
                 font=("Segoe UI", 8), bg=C_SURFACE, fg=C_TEXT_DIM,
                 anchor="w", justify="left").pack(fill="x")

    # Menangani proses metric chip pada PerformanceDashboard sambil menjaga state internal tetap konsisten.
    def _metric_chip(self, parent, label, metric):
        btn = tk.Button(
            parent,
            text=label,
            font=("Segoe UI", 8, "bold"),
            bg=C_SURFACE2,
            fg=C_TEXT_DIM,
            activebackground=C_SURFACE2,
            activeforeground=C_TEXT,
            relief="flat",
            cursor="hand2",
            padx=10,
            pady=4,
            highlightthickness=1,
            highlightbackground=C_BORDER,
            bd=0,
            command=lambda m=metric: self._set_chart_metric(m),
        )
        btn.pack(side="left", padx=3)
        self._chart_metric_buttons[metric] = btn

    # Memperbarui metric chip states pada PerformanceDashboard agar data, status, dan tampilan tetap selaras.
    def _update_metric_chip_states(self):
        active = self._chart_metric_var.get()
        for metric, btn in self._chart_metric_buttons.items():
            if not btn.winfo_exists():
                continue
            if metric == active:
                btn.config(bg=C_ACCENT, fg=C_BG, activebackground=C_ACCENT, activeforeground=C_BG,
                           highlightbackground=C_ACCENT, highlightthickness=2)
            else:
                btn.config(bg=C_SURFACE2, fg=C_TEXT_DIM, activebackground=C_SURFACE2, activeforeground=C_TEXT,
                           highlightbackground=C_BORDER, highlightthickness=1)

    # Mengatur chart metric pada PerformanceDashboard dan menerapkan nilai yang dipilih ke state internal.
    def _set_chart_metric(self, metric):
        self._chart_metric_var.set(metric)
        self._update_metric_chip_states()
        self._render_chart(metric)

    # Menangani proses render chart pada PerformanceDashboard sambil menjaga state internal tetap konsisten.
    def _render_chart(self, metric):
        try:
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            import matplotlib.pyplot as plt
        except Exception as exc:
            self._chart_info_var.set(f"Matplotlib tidak tersedia: {exc}")
            return

        if self._chart_host is None or not self._chart_host.winfo_exists():
            return

        for child in self._chart_host.winfo_children():
            child.destroy()
        self._chart_point_meta = []
        self._chart_selected_index = None
        self._chart_selected_marker = None

        sessions = _dedupe_sessions(list(self.ml.sessions or []))

        if self.session is not None:
            current_ts = self.session.get("timestamp")
            already_in = (
                current_ts is not None
                and any(s.get("timestamp") == current_ts for s in sessions)
            )
            if not already_in:
                current_fp = _session_fingerprint(self.session)
                if current_fp not in {_session_fingerprint(s) for s in sessions}:
                    sessions.append(self.session)

        sessions = _dedupe_sessions(sessions)

        labels = [f"S{i+1}" for i in range(len(sessions))]
        x = list(range(1, len(sessions) + 1))

        # Menangani proses score of pada PerformanceDashboard sambil menjaga state internal tetap konsisten.
        def _score_of(s):
            t = s.get("total_time", 0)
            ec = s.get("empty_cells", max(s.get("moves", 1), 1))
            return s.get("score") or calculate_score(
                s.get("difficulty", "Normal"), t, ec,
                s.get("errors", 0), s.get("hints_used", 0),
                s.get("completed", False), s.get("near_miss", 0),
                s.get("guessing", 0), s.get("auto_used", 0)
            )

        metric_map = {
            "score":  ("Skor",        [int(_score_of(s)) for s in sessions],                    C_GOLD),
            "time":   ("Total Waktu (detik)", [float(s.get("total_time", 0)) for s in sessions], C_ACCENT),
            "errors": ("Errors",      [int(s.get("errors", 0)) for s in sessions],               C_ERROR),
            "hints":  ("Hints",       [int(s.get("hints_used", 0)) for s in sessions],          C_WARN),
            "auto":   ("Auto Fill",   [int(s.get("auto_used", 0)) for s in sessions],           C_PURPLE),
        }
        title, y, color = metric_map.get(metric, metric_map["score"])

        fig = plt.Figure(figsize=(5.4, 3.4), dpi=100)
        fig.patch.set_facecolor(C_SURFACE2)
        ax = fig.add_subplot(111)
        ax.set_facecolor(C_SURFACE2)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(C_BORDER)
        ax.spines["bottom"].set_color(C_BORDER)
        ax.tick_params(colors=C_TEXT_DIM, labelsize=8)
        ax.grid(True, axis="y", alpha=0.16)

        if len(x) == 1:
            ax.plot(x, y, color=color, linewidth=2, alpha=0.8, zorder=2)
        else:
            ax.plot(x, y, color=color, linewidth=2.4, marker="o", markersize=5, zorder=2)
            ax.fill_between(x, y, [0]*len(y), color=color, alpha=0.12, zorder=1)

        ax.scatter(x, y, s=36, color=color, alpha=0.95, zorder=3)

        for xi, yi, lbl in zip(x, y, labels):
            self._chart_point_meta.append({"x": xi, "y": yi, "label": lbl, "value": yi})

        ax.set_title(title, color=C_TEXT, fontsize=11, fontweight="bold", pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, color=C_TEXT_DIM)
        if metric == "time":
            ax.set_ylabel("detik", color=C_TEXT_DIM, fontsize=8)
        elif metric == "score":
            ax.set_ylabel("skor", color=C_TEXT_DIM, fontsize=8)
        elif metric == "auto":
            ax.set_ylabel("kali", color=C_TEXT_DIM, fontsize=8)
        else:
            ax.set_ylabel(metric.capitalize(), color=C_TEXT_DIM, fontsize=8)

        ymax = max(y) if y else 1
        ax.set_ylim(0, max(5, ymax * 1.25))

        if y:
            avg = sum(y) / len(y)
            ax.axhline(avg, color=C_TEXT_DIM, linewidth=1, linestyle="--", alpha=0.45)
            ax.text(0.98, 0.94, f"avg {avg:.1f}", transform=ax.transAxes,
                    ha="right", va="top", color=C_TEXT_DIM, fontsize=8)

        self._chart_fig = fig
        self._chart_ax = ax

        canvas = FigureCanvasTkAgg(fig, master=self._chart_host)
        canvas.draw()
        widget = canvas.get_tk_widget()
        widget.pack(fill="both", expand=True)
        self._chart_canvas = canvas
        self._chart_canvas_widget = widget

        self._chart_info_var.set("Arahkan mouse ke titik untuk melihat detail sesi. Klik untuk menandai titik aktif.")
        canvas.mpl_connect("motion_notify_event", self._on_chart_hover)
        canvas.mpl_connect("button_press_event", self._on_chart_click)
    # Menangani event chart hover pada PerformanceDashboard dan memperbarui state yang terkait.
    def _on_chart_hover(self, event):
        if self._chart_ax is None or event.inaxes != self._chart_ax or not self._chart_point_meta:
            return
        if event.xdata is None:
            return
        idx = min(range(len(self._chart_point_meta)), key=lambda i: abs(self._chart_point_meta[i]["x"] - event.xdata))
        pt = self._chart_point_meta[idx]
        self._chart_info_var.set(f"{pt['label']}  ·  nilai {pt['value']}")
        # Hover hanya mengubah teks info; chart tidak di-redraw agar tidak flicker/putih.

    # Menangani event chart click pada PerformanceDashboard dan memperbarui state yang terkait.
    def _on_chart_click(self, event):
        if self._chart_ax is None or event.inaxes != self._chart_ax or not self._chart_point_meta:
            return
        if event.xdata is None:
            return
        idx = min(range(len(self._chart_point_meta)), key=lambda i: abs(self._chart_point_meta[i]["x"] - event.xdata))
        self._highlight_chart_point(idx)

    # Menangani proses highlight chart point pada PerformanceDashboard sambil menjaga state internal tetap konsisten.
    def _highlight_chart_point(self, idx):
        if self._chart_ax is None or self._chart_fig is None or not self._chart_point_meta:
            return
        self._chart_selected_index = idx
        pt = self._chart_point_meta[idx]

        try:
            if self._chart_selected_marker is not None:
                self._chart_selected_marker.remove()
        except Exception:
            pass

        ax = self._chart_ax
        self._chart_selected_marker = ax.scatter(
            [pt["x"]], [pt["y"]],
            s=180, facecolors=C_BG, edgecolors=C_WHITE, linewidths=1.8, zorder=6
        )
        ax.scatter([pt["x"]], [pt["y"]], s=70, color=C_WHITE, zorder=7)
        self._chart_info_var.set(f"{pt['label']} dipilih  ·  nilai {pt['value']}")
        if self._chart_canvas is not None:
            self._chart_canvas.draw_idle()
    # SECTION BUILDERS
    # Menangani proses section title pada PerformanceDashboard sambil menjaga state internal tetap konsisten.
    def _section_title(self, parent, title, subtitle=""):
        wrap = tk.Frame(parent, bg=C_BG)
        wrap.pack(fill="x", pady=(0, 8))
        row = tk.Frame(wrap, bg=C_BG)
        row.pack(fill="x")
        tk.Label(row, text=title, font=("Segoe UI", 11, "bold"),
                 bg=C_BG, fg=C_TEXT).pack(side="left")
        if subtitle:
            tk.Label(row, text=f"  {subtitle}", font=("Segoe UI", 8),
                     bg=C_BG, fg=C_TEXT_DIM).pack(side="left")

    # Membangun stat tiles pada PerformanceDashboard dan menyiapkan widget supaya state tampilan tetap konsisten.
    def _build_stat_tiles(self, parent, s):
        t      = s.get("total_time", 0)
        ts_str = f"{int(t//60):02}:{int(t%60):02}"
        ec     = s.get("empty_cells", max(s.get("moves", 1), 1))
        tpc    = s.get("time_per_cell", t / max(ec, 1))
        score  = s.get("score") or calculate_score(
            s.get("difficulty", "Normal"), t, ec, s.get("errors", 0),
            s.get("hints_used", 0), s.get("completed", False),
            s.get("near_miss", 0), s.get("guessing", 0))
        grid_sz = s.get("grid_size", 3)
        nm      = s.get("near_miss", 0)
        gu      = s.get("guessing", 0)
        hl      = s.get("hearts_left", "?")
        mh      = s.get("max_hearts", grid_sz * grid_sz)
        me      = s.get("max_errors", "?")
        completed = s.get("completed", False)

        if completed:
            banner_col  = "#1A3A1A" if _CURRENT_THEME_NAME == "dark" else "#DCFCE7"
            banner_fg   = C_ACCENT2
            banner_text = "✅  COMPLETED"
        else:
            lose_reason = s.get("lose_reason", "")
            if lose_reason == "max_errors":
                banner_text = f"💀  GAME OVER"
            else:
                banner_text = "❌  PUZZLE TIDAK SELESAI"
            banner_col  = "#3B1212" if _CURRENT_THEME_NAME == "dark" else "#FFE4E6"
            banner_fg   = C_ERROR

        banner = tk.Frame(parent, bg=banner_col,
                          highlightbackground=banner_fg, highlightthickness=1)
        banner.pack(fill="x", pady=(0, 10))
        tk.Label(banner, text=banner_text,
                 font=("Segoe UI", 11, "bold"),
                 bg=banner_col, fg=banner_fg, pady=8).pack()

        _show_auto = (s.get("difficulty", "") == "Hard")

        items = [
            ("⏱️", "Total Waktu",        ts_str,                                 C_ACCENT),
            ("📐", "Waktu / Sel",        f"{tpc:.1f}s",                         "#58A6FF"),
            ("✅", "Moves",              str(s.get("moves", 0)),                C_ACCENT2),
            ("❌", "Errors / Batas",     f"{s.get('errors', 0)}/{me}",          C_ERROR),
            ("💡", "Hints",              str(s.get("hints_used", 0)),           C_WARN),
            ("♥",  "Hati Tersisa",       f"{hl}/{mh}",                          C_ERROR),
            ("🎯", "Near Miss",          str(nm),                               "#F0883E"),
            ("🎲", "Guessing",           str(gu),                               C_ERROR),
            ("🏆", "Skor",               str(score),                            C_GOLD),
            ("🎮", "Grid",               f"{grid_sz*grid_sz}×{grid_sz*grid_sz}", C_PURPLE),
        ]
        if _show_auto:
            items.append(("⚡", "Auto Fill", str(s.get("auto_used", 0)), C_PURPLE))

        grid = tk.Frame(parent, bg=C_BG)
        grid.pack(fill="x", pady=(6, 16))
        cols = 5
        total_items = len(items)
        last_row_count = total_items % cols
        for i, (ico, lbl, val, col) in enumerate(items):
            rr, cc = divmod(i, cols)
            is_last_sole = (last_row_count == 1 and i == total_items - 1)
            cf = tk.Frame(grid, bg=C_SURFACE, highlightbackground=C_BORDER, highlightthickness=1)
            if is_last_sole:
                cf.grid(row=rr, column=2, columnspan=1, padx=4, pady=4, sticky="nsew")
            else:
                cf.grid(row=rr, column=cc, padx=4, pady=4, sticky="nsew")
            top_r = tk.Frame(cf, bg=C_SURFACE)
            top_r.pack(fill="x", padx=10, pady=(8, 0))
            tk.Label(top_r, text=ico, font=("Segoe UI", 12), bg=C_SURFACE, fg=col).pack(side="left")
            tk.Label(top_r, text=f"  {lbl}", font=("Segoe UI", 8), bg=C_SURFACE, fg=C_TEXT_DIM).pack(side="left")
            tk.Label(cf, text=val, font=("Segoe UI", 15, "bold"), bg=C_SURFACE, fg=col).pack(pady=(6, 10))
        for cc in range(cols):
            grid.columnconfigure(cc, weight=1)

    # Membangun behavior card pada PerformanceDashboard dan menyiapkan widget supaya state tampilan tetap konsisten.
    def _build_behavior_card(self, parent, s, feat):
        card = tk.Frame(parent, bg=C_SURFACE, highlightbackground=C_BORDER, highlightthickness=1)
        card.pack(fill="both", expand=True)

        top = tk.Frame(card, bg=C_SURFACE)
        top.pack(fill="x", padx=16, pady=(14, 8))
        tk.Label(top, text="🧠  ANALISIS PERILAKU",
                 font=("Segoe UI", 10, "bold"), bg=C_SURFACE, fg=C_TEXT).pack(anchor="w")
        tk.Label(top, text="Bukan sekadar angka - ini memetakan gaya bermain yang terlihat.",
                 font=("Segoe UI", 8), bg=C_SURFACE, fg=C_TEXT_DIM).pack(anchor="w", pady=(2,0))

        inner = tk.Frame(card, bg=C_SURFACE)
        inner.pack(fill="both", expand=True, padx=16, pady=(0, 14))

        _show_auto_beh = (s.get("difficulty", "") == "Hard")

        b_items = [
            ("🎯", "Near Miss", s.get("near_miss", 0), "#F0883E", "Paham area jawaban, tetapi masih meleset tipis."),
            ("🎲", "Guessing", s.get("guessing", 0), C_ERROR, "Menebak angka berulang di sel yang sama."),
            ("📐", "Latency", f"{s.get('time_per_cell', s.get('total_time', 0)/max(s.get('moves',1),1)):.1f}s", C_ACCENT, "Semakin kecil, semakin efisien."),
            ("💔", "Surrender", f"{s.get('hints_used', 0)}/{s.get('max_hearts', feat.get('sessions_count', 1) or 1)}", C_PURPLE, "Frekuensi memakai hint dibanding kapasitas hidup."),
        ]
        if _show_auto_beh:
            b_items.append(("⚡", "Auto Fill", str(s.get("auto_used", 0)), "#BC8CFF", "Penggunaan Auto isi kandidat (-50 poin/kali, Hard mode)."))

        for idx, (ico, ttl, val, col, desc) in enumerate(b_items):
            row = tk.Frame(inner, bg=C_SURFACE2, highlightbackground=C_BORDER, highlightthickness=1)
            row.grid(row=idx//2, column=idx%2, padx=5, pady=5, sticky="nsew")
            bl = tk.Frame(row, bg=col, width=5)
            bl.pack(side="left", fill="y")
            box = tk.Frame(row, bg=C_SURFACE2)
            box.pack(side="left", fill="both", expand=True, padx=12, pady=10)
            head = tk.Frame(box, bg=C_SURFACE2)
            head.pack(fill="x")
            tk.Label(head, text=f"{ico}  {ttl}", font=("Segoe UI", 10, "bold"),
                     bg=C_SURFACE2, fg=C_TEXT).pack(side="left")
            tk.Label(head, text=str(val), font=("Segoe UI", 12, "bold"),
                     bg=C_SURFACE2, fg=col).pack(side="right")
            tk.Label(box, text=desc, font=("Segoe UI", 8),
                     bg=C_SURFACE2, fg=C_TEXT_DIM, wraplength=250, justify="left").pack(anchor="w", pady=(6,0))

        total_rows = (len(b_items) + 1) // 2
        for r in range(total_rows):
            inner.rowconfigure(r, weight=1)
        for c in range(2):
            inner.columnconfigure(c, weight=1)

    # Membangun skill ml card pada PerformanceDashboard dan menyiapkan widget supaya state tampilan tetap konsisten.
    def _build_skill_ml_card(self, parent, feat, stats, s):
        card = tk.Frame(parent, bg=C_SURFACE, highlightbackground=C_BORDER, highlightthickness=1)
        card.pack(fill="both", expand=True)

        top = tk.Frame(card, bg=C_SURFACE)
        top.pack(fill="x", padx=16, pady=(14, 8))
        tk.Label(top, text="📊  ANALISIS KEMAMPUAN + ML",
                 font=("Segoe UI", 10, "bold"), bg=C_SURFACE, fg=C_TEXT).pack(anchor="w")
        tk.Label(top, text="Ringkasan kompetensi dan status model dalam satu panel.",
                 font=("Segoe UI", 8), bg=C_SURFACE, fg=C_TEXT_DIM).pack(anchor="w", pady=(2,0))

        ml_prof = s.get("ml_profile", {}) if s else {}
        bar_box = tk.Frame(card, bg=C_SURFACE)
        bar_box.pack(fill="x", padx=16, pady=(0, 12))
        skill_bars = [
            ("Kecepatan",
             float(ml_prof.get("speed_index",
                   max(0, min(100, 100 - feat["avg_time_per_cell"] * 8)))),
             C_GOLD),
            ("Akurasi",
             float(ml_prof.get("accuracy_index",
                   max(0, min(100, 100 - feat["error_rate"] * 250)))),
             C_ACCENT2),
            ("Konsistensi",
             float(ml_prof.get("consistency_index",
                   feat["completion_rate"] * 100)),
             C_ACCENT),
            ("Kemandirian",
             float(ml_prof.get("independence_index",
                   max(0, min(100, (1 - feat["hint_rate"] - feat.get("auto_rate", 0) * 0.4) * 100)))),
             C_PURPLE),
        ]
        for sn, sv, sc_col in skill_bars:
            self._bar(bar_box, sn, sv, sc_col)

        ml_conf    = stats.get("ml_confidence", 0)
        pred_score  = stats.get("predicted_next_score")
        pred_avail  = stats.get("predicted_score_avail", False)
        anom_status = stats.get("anomaly_status", "unknown")
        anom_reason = stats.get("anomaly_reason", "")
        sklearn_on  = stats.get("sklearn_active", False)

        sub = tk.Frame(card, bg=C_SURFACE)
        sub.pack(fill="both", expand=True, padx=16, pady=(0, 14))

        cols = tk.Frame(sub, bg=C_SURFACE)
        cols.pack(fill="x")
        self._mini_ml_card(cols, "KNN CLASSIFIER", f"{stats['type_info'].get('emoji','🎮')}  {stats['player_type']}",
                           f"Confidence {ml_conf:.1f}%" if sklearn_on else "Rule-based fallback",
                           C_ACCENT2 if ml_conf >= 70 else (C_WARN if ml_conf >= 45 else C_ERROR),
                           0, 1)
        pred_txt = f"🏆 {pred_score}" if pred_avail and pred_score is not None else "-"
        pred_sub  = f"vs saat ini {s.get('score', 0) or 0}" if pred_avail and pred_score is not None else "Butuh minimal 3 sesi"
        self._mini_ml_card(cols, "HistGBR", "Prediksi Skor Berikutnya",
                           f"{pred_txt}  ·  {pred_sub}",
                           C_GOLD if pred_avail and pred_score is not None else C_TEXT_DIM,
                           1, 1)
        anom_emoji  = {"normal": "✅", "anomaly": "⚠️", "unknown": "❓"}
        anom_colors = {"normal": C_ACCENT2, "anomaly": C_WARN, "unknown": C_TEXT_DIM}
        anom_labels = {"normal": "Sesi Normal", "anomaly": "Sesi Anomali", "unknown": "Belum Cukup Data"}
        self._mini_ml_card(cols, "ISOLATION FOREST", "Anomaly Detection",
                           f"{anom_emoji.get(anom_status,'❓')} {anom_labels.get(anom_status,'-')}\n{anom_reason}",
                           anom_colors.get(anom_status, C_TEXT_DIM), 2, 1)

    # Menangani proses mini ml card pada PerformanceDashboard sambil menjaga state internal tetap konsisten.
    def _mini_ml_card(self, parent, title, subtitle, body, color, col, row):
        card = tk.Frame(parent, bg=C_SURFACE2, highlightbackground=C_BORDER, highlightthickness=1)
        padx = (0, 6) if col == 0 else ((3, 3) if col == 1 else (6, 0))
        card.pack(side="left", fill="both", expand=True, padx=padx, ipadx=10, ipady=10)
        tk.Label(card, text=title, font=("Segoe UI", 8, "bold"),
                 bg=C_SURFACE2, fg=C_TEXT_DIM).pack(anchor="w")
        tk.Label(card, text=subtitle, font=("Segoe UI", 9),
                 bg=C_SURFACE2, fg=C_TEXT_DIM).pack(anchor="w")
        tk.Label(card, text=body, font=("Segoe UI", 11, "bold"),
                 bg=C_SURFACE2, fg=color, wraplength=180, justify="left").pack(anchor="w", pady=(4,0))

    # Menangani proses chip pada PerformanceDashboard sambil menjaga state internal tetap konsisten.
    def _chip(self, parent, label, value, color):
        c = tk.Frame(parent, bg=C_SURFACE2, highlightbackground=color, highlightthickness=1)
        c.pack(side="left", padx=(0, 8))
        tk.Label(c, text=f" {label} ", font=("Segoe UI", 8, "bold"), bg=C_SURFACE2, fg=C_TEXT_DIM).pack(side="left", padx=(6, 2))
        tk.Label(c, text=value, font=("Segoe UI", 8, "bold"), bg=C_SURFACE2, fg=color).pack(side="left", padx=(0, 6))

    # Menangani proses action button pada PerformanceDashboard sambil menjaga state internal tetap konsisten.
    def _action_button(self, parent, text, bg, fg, cmd, side="left", padx=4, fill=True):
        btn = tk.Button(parent, text=text, font=FONT_BTN, bg=bg, fg=fg,
                        relief="flat", cursor="hand2", pady=10, command=cmd)
        if fill:
            btn.pack(side=side, fill="x", expand=True, padx=padx)
        else:
            btn.pack(side=side, padx=padx)
        return btn

    # Membangun recommendation reason pada PerformanceDashboard dan menyiapkan widget supaya state tampilan tetap konsisten.
    def _build_recommendation_reason(self, pt, feat, rec):
        if rec == "Hard":
            return "Performa kamu stabil dan cepat. Dashboard mengarah ke tantangan yang lebih padat agar progres tetap naik."
        if rec == "Normal":
            return "Keseimbangan kecepatan dan akurasi kamu sudah bagus; level menengah akan menjaga ritme tanpa terasa terlalu mudah."
        return "Polanya masih cocok untuk eksplorasi ringan. Bangun konsistensi dulu sebelum naik ke tantangan lebih tinggi."

    # ACTIONS / UTILS
    # Menyegarkan dashboard pada PerformanceDashboard setelah data atau pilihan pengguna berubah.
    def _refresh_dashboard(self):
        try:
            self.controller._refresh_dashboard()
        except Exception:
            pass

    # Menangani proses play again pada PerformanceDashboard sambil menjaga state internal tetap konsisten.
    def _play_again(self):
        self.controller._play_again()

    # Memulai recommendation pada PerformanceDashboard dan menyalakan mekanisme pendukung yang dibutuhkan.
    def _start_recommendation(self):
        self.controller._start_recommended_grid(self.recommended_grid, self.ml.recommend_difficulty())

    # Menangani proses logout from dashboard pada PerformanceDashboard sambil menjaga state internal tetap konsisten.
    def _logout_from_dashboard(self):
        self.master.event_generate("<<Logout>>")

    # Menangani proses save score card action pada PerformanceDashboard sambil menjaga state internal tetap konsisten.
    def _save_score_card_action(self):
        s          = self.session
        stats      = self.ml.get_summary()
        p_type     = stats.get("player_type", "Pemain")
        type_info  = stats.get("type_info", {})
        type_color = type_info.get("color", "#58A6FF")
        ai_msg     = self.recommended_reason or "Terus berlatih dan raih skor terbaik!"

        _data         = load_data()
        _pdata        = _data["players"].get(self.username, {})
        all_sessions  = _pdata.get("sessions", [])
        achievements  = _pdata.get("achievements", [])

        path = _save_score_card(
            username      = self.username,
            session       = s,
            player_type   = p_type,
            type_color    = type_color,
            ai_message    = ai_msg,
            all_sessions  = all_sessions,
            ml_summary    = stats,
            achievements  = achievements,
            parent_widget = self,
        )
        if path:
            self._show_save_toast(path)
        else:
            messagebox.showerror(
                "Gagal", "Gagal menyimpan kartu skor.\nPastikan Pillow terinstall.",
                parent=self)

    # Menampilkan save toast pada PerformanceDashboard dan mengaktifkan elemen pendukung yang diperlukan.
    def _show_save_toast(self, path):
        _sys = sys

        old = getattr(self, "_toast_frame", None)
        if old and old.winfo_exists():
            try: old.destroy()
            except Exception: pass

        toast = tk.Frame(self, bg=C_SURFACE,
                         highlightbackground=C_ACCENT, highlightthickness=1)
        toast.place(relx=0.5, rely=1.0, anchor="s", relwidth=0.96, height=52)
        self._toast_frame = toast

        txt_row = tk.Frame(toast, bg=C_SURFACE)
        txt_row.pack(side="left", fill="y", padx=(10, 0))
        chk_cv = tk.Canvas(txt_row, width=18, height=18,
                           bg=C_SURFACE, highlightthickness=0)
        chk_cv.pack(side="left", pady=16)
        chk_cv.create_text(9, 9, text="OK", font=("Segoe UI", 7, "bold"), fill=C_ACCENT)
        tk.Label(txt_row, text="Scorecard berhasil tersimpan!",
                 font=("Segoe UI", 10, "bold"),
                 bg=C_SURFACE, fg=C_ACCENT,
                 anchor="w").pack(side="left", padx=(4, 0), pady=0, fill="y")

        # Menangani proses open file pada PerformanceDashboard sambil menjaga state internal tetap konsisten.
        def _open_file():
            try:
                if _sys.platform == "win32":
                    os.startfile(path)
                elif _sys.platform == "darwin":
                    subprocess.run(["open", path], check=False)
                else:
                    subprocess.run(["xdg-open", path], check=False)
            except Exception:
                messagebox.showinfo("Lokasi File", f"File tersimpan di:\n{path}", parent=self)

        btn_lihat = tk.Button(
            toast, text="  Lihat  ",
            font=("Segoe UI", 9, "bold"),
            bg=C_ACCENT, fg=C_BG,
            relief="flat", cursor="hand2",
            activebackground=C_ACCENT2, activeforeground=C_BG,
            command=_open_file)
        btn_lihat.pack(side="right", padx=8, pady=8, ipady=2)

        btn_close = tk.Button(
            toast, text="X",
            font=("Segoe UI", 9),
            bg=C_SURFACE, fg=C_TEXT_DIM,
            relief="flat", cursor="hand2",
            activebackground=C_SURFACE2, activeforeground=C_TEXT,
            command=lambda: toast.destroy() if toast.winfo_exists() else None)
        btn_close.pack(side="right", padx=(0, 4), pady=8)

        # Menangani proses slide in pada PerformanceDashboard sambil menjaga state internal tetap konsisten.
        def _slide_in(step=0, total=10):
            if not toast.winfo_exists(): return
            rely = 1.0 - (step / total) * 0.005 * 52 / max(self.winfo_height(), 1)
            y_offset = int((1.0 - step / total) * 52)
            toast.place_configure(rely=1.0, y=-step * 52 // total)
            if step < total:
                self.after(16, lambda: _slide_in(step + 1, total))
        toast.place_configure(y=0)
        _slide_in()

        # Menangani proses dismiss pada PerformanceDashboard sambil menjaga state internal tetap konsisten.
        def _dismiss():
            if toast.winfo_exists():
                try: toast.destroy()
                except Exception: pass
        self.after(6000, _dismiss)

    # Menangani proses bind mousewheel pada PerformanceDashboard sambil menjaga state internal tetap konsisten.
    def _bind_mousewheel(self):
        if self._mousewheel_bound:
            return
        top = self.winfo_toplevel()
        top.bind_all("<MouseWheel>", self._on_mousewheel, add="+")
        top.bind_all("<Button-4>", self._on_mousewheel, add="+")
        top.bind_all("<Button-5>", self._on_mousewheel, add="+")
        self._mousewheel_bound = True

    # Menangani event mousewheel pada PerformanceDashboard dan memperbarui state yang terkait.
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

    # Menghancurkan widget pada PerformanceDashboard dan melepaskan resource yang masih aktif.
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

    # Menangani proses bar pada PerformanceDashboard sambil menjaga state internal tetap konsisten.
    def _bar(self, parent, name, val, color):
        row = tk.Frame(parent, bg=C_BG)
        row.pack(fill="x", pady=2)
        tk.Label(row, text=name, font=("Segoe UI", 9),
                 bg=C_BG, fg=C_TEXT_DIM, width=14, anchor="w").pack(side="left")
        bg = tk.Frame(row, bg=C_SURFACE2, height=12)
        bg.pack(side="left", fill="x", expand=True)
        bar = tk.Frame(bg, bg=color, height=12, width=4)
        bar.place(x=0, y=0)
        # Menangani proses grow pada PerformanceDashboard sambil menjaga state internal tetap konsisten.
        def grow():
            bg.update_idletasks()
            target = int(bg.winfo_width() * val / 100)
            # Menangani proses step pada PerformanceDashboard sambil menjaga state internal tetap konsisten.
            def step(cur):
                if cur < target:
                    nxt = min(cur + max(1, target//20), target)
                    bar.config(width=nxt)
                    bg.after(15, lambda: step(nxt))
            step(4)
        bg.after(250, grow)
        tk.Label(row, text=f"{val:.0f}%", font=("Segoe UI", 9, "bold"),
                 bg=C_BG, fg=color, width=5).pack(side="right")

# TUTORIAL OVERLAY - Pemain Baru
# Muncul otomatis pada game PERTAMA setiap pemain baru.
# Tiga langkah dasar cara bermain Sudoku ditampilkan dalam
# satu overlay card. Setelah ditutup, flag 'tutorial_done'
# disimpan ke sudoku_data.json sehingga tidak muncul lagi.
class TutorialOverlay(tk.Frame):
    """
    Overlay tutorial 3 langkah untuk pemain baru.
    Dipanggil dari GameScreen._show_tutorial() setelah puzzle pertama terbentuk.
    """
    # Menginisialisasi objek TutorialOverlay dan menyiapkan state awal, referensi penting, serta elemen yang dibutuhkan sebelum layar dipakai.
    def __init__(self, master, n, on_close):
        overlay_bg = "#080B10" if _CURRENT_THEME_NAME == "dark" else "#BCC8DC"
        super().__init__(master, bg=overlay_bg)
        self.on_close = on_close
        self.n = n
        self.place(relx=0, rely=0, relwidth=1, relheight=1)
        self._build()
        self.lift()

    # Membangun bagian antarmuka pada TutorialOverlay dan menyiapkan widget supaya state tampilan tetap konsisten.
    def _build(self):
        badge_fg = C_BG
        card = tk.Frame(self, bg=C_SURFACE,
                        highlightbackground=C_ACCENT, highlightthickness=2)
        card.place(relx=0.5, rely=0.5, anchor="center", width=600, height=350)

        title_row = tk.Frame(card, bg=C_SURFACE)
        title_row.pack(pady=(22, 2))
        title_bar = tk.Frame(title_row, bg=C_ACCENT, width=6, height=26)
        title_bar.pack(side="left", padx=(0, 10))
        title_bar.pack_propagate(False)
        tk.Label(title_row, text="Selamat Datang di Sudoku AI!",
                 font=("Segoe UI", 17, "bold"),
                 bg=C_SURFACE, fg=C_TEXT).pack(side="left")

        tk.Label(card,
                 text="Ikuti 3 langkah sederhana berikut untuk mulai bermain",
                 font=("Segoe UI", 10),
                 bg=C_SURFACE, fg=C_TEXT_DIM).pack(pady=(0, 16))

        row = tk.Frame(card, bg=C_SURFACE)
        row.pack(fill="x", padx=20, pady=(0, 16))

        steps = [
            ("1", "Klik Sel Kosong",
             "Sentuh salah satu kotak kosong di papan permainan",
             C_ACCENT),
            ("2", f"Masukkan Angka 1-{self.n}",
             "Ketik angka yang menurutmu tepat mengisi sel tersebut",
             C_ACCENT2),
            ("3", "Ingat Aturannya",
             "Tiap baris, kolom, dan kotak kecil harus berisi angka yang berbeda",
             C_WARN),
        ]

        for num, title, desc, color in steps:
            col = tk.Frame(row, bg=C_SURFACE2,
                           highlightbackground=color, highlightthickness=1)
            col.pack(side="left", fill="both", expand=True, padx=5)

            tk.Frame(col, bg=color, height=4).pack(fill="x")

            badge_wrap = tk.Frame(col, bg=C_SURFACE2)
            badge_wrap.pack(pady=(18, 0))
            badge = tk.Frame(badge_wrap, bg=color, width=44, height=44)
            badge.pack()
            badge.pack_propagate(False)
            tk.Label(badge, text=num, font=("Segoe UI", 20, "bold"),
                     bg=color, fg=badge_fg).pack(expand=True)

            tk.Label(col, text=title, font=("Segoe UI", 9, "bold"),
                     bg=C_SURFACE2, fg=C_TEXT,
                     wraplength=156, justify="center").pack(padx=8, pady=(12, 0))
            tk.Label(col, text=desc, font=("Segoe UI", 8),
                     bg=C_SURFACE2, fg=C_TEXT_DIM,
                     wraplength=156, justify="center").pack(padx=8, pady=(4, 18))

        tk.Button(card,
                  text="  Mengerti, mulai bermain!  ",
                  font=("Segoe UI", 12, "bold"),
                  bg=C_ACCENT, fg=badge_fg,
                  activebackground=C_ACCENT2, activeforeground=badge_fg,
                  relief="flat", cursor="hand2", pady=11,
                  command=self._close
                  ).pack(fill="x", padx=20, pady=(0, 20))

    # Menangani proses close pada TutorialOverlay sambil menjaga state internal tetap konsisten.
    def _close(self):
        if callable(self.on_close):
            self.on_close()
        try:
            self.destroy()
        except Exception:
            pass

# MAIN GAME SCREEN
# Draft Mode (Hard only):
#   - Setiap sel kosong punya Canvas yang bisa menampilkan
#     angka draft kecil di 9 posisi (seperti Sudoku pro)
#   - Toggle mode draft dengan tombol ✏ atau shortcut "D"
#   - Saat draft mode aktif, angka yang diketik masuk ke
#     draft corner kecil, bukan ke sel utama
#   - Tombol "✔ Konfirmasi" commit semua draft ke board
#   - Tombol "✗ Hapus Draft" bersihkan semua draft
class GameScreen(tk.Frame):
    """
    GameScreen - Layar permainan utama Sudoku dengan semua mekanik game.

    Deskripsi:
        Mengelola siklus satu sesi: generate puzzle, input pemain, validasi,
        hint adaptif, Draft Mode (pensil kandidat), timer, menang/kalah,
        dan penyimpanan sesi.

    Atribut:
        username (str): Username pemain.
        N (int): Ukuran papan (grid_size²).
        puzzle (list): Papan awal (0 = kosong).
        solution (list): Kunci jawaban.
        current_board (list): Papan yang sedang dimainkan.
        draft_board (dict): ``{(r,c): set}`` kandidat pensil.
        error_count (int): Jumlah kesalahan dalam sesi ini.
        hints_used (int): Jumlah hint yang digunakan.
        on_finish (callable): Callback dengan dict sesi saat selesai.
    """
    # Ukuran canvas per sel (pixel) - disesuaikan per grid
    CELL_PX_9 = 58   # untuk 9x9
    CELL_PX_4 = 90   # untuk 4x4

    # Menginisialisasi objek GameScreen dan menyiapkan state awal, referensi penting, serta elemen yang dibutuhkan sebelum layar dipakai.
    def __init__(self, master, username, grid_size, difficulty, on_finish,
                 resume_state=None):
        super().__init__(master, bg=C_BG)
        self.username   = username
        self.grid_size  = grid_size
        self.difficulty = difficulty
        self.on_finish  = on_finish
        self.N          = grid_size * grid_size
        self.BOX        = grid_size
        self.theme      = get_diff_theme(difficulty)

        self.CELL_PX = self.CELL_PX_4 if self.N == 4 else self.CELL_PX_9

        self.puzzle        = []
        self.solution      = []
        self.current_board = []
        self.selected      = None
        self.canvases      = {}

        self.draft_board   = {}
        self.draft_mode    = False
        self.numpad_btns   = {}

        self.timer_running    = False
        self.start_time       = 0
        self.elapsed          = 0
        self.game_over        = False
        self.error_count      = 0
        self.move_count       = 0
        self.hints_used       = 0
        self.auto_used_count  = 0
        self.last_action      = time.time()
        self.idle_after       = None
        self.hint_shown       = False

        self.max_hearts       = self.N
        self.hearts           = self.max_hearts
        self.cell_errors      = {}
        self.cell_last_time   = {}
        self.near_miss_count  = 0
        self.guessing_count   = 0
        self.empty_cells      = 0

        data     = load_data()
        sessions = [s for s in data["players"].get(username, {}).get("sessions", [])
                    if s.get("grid_size") == grid_size]
        self.ml  = PlayerMLEngine()
        self.ml.sessions = sessions

        self.max_errors = self._get_max_errors()
        player_data    = data.get("players", {}).get(username, {})
        has_sessions   = bool(player_data.get("sessions"))
        tutorial_done  = player_data.get("tutorial_done", False)
        self._tutorial_pending = (not has_sessions) and (not tutorial_done)

        self._ml_panel_visible  = False
        self._ml_panel_frame    = None
        self._ml_panel_job      = None
        self._ml_last_refresh   = 0.0
        self._ml_refresh_busy   = False
        self._ml_panel_pos      = None

        self._build()
        if resume_state:
            self._restore_state(resume_state)
            if getattr(self, "_tutorial_pending", False):
                self._tutorial_pending = False
                self.after(500, self._show_tutorial)
        else:
            self._start_new_game()

    # Membangun bagian antarmuka pada GameScreen dan menyiapkan widget supaya state tampilan tetap konsisten.
    def _build(self):
        self.place(relx=0, rely=0, relwidth=1, relheight=1)

        _sb_w = 275
        sb = tk.Frame(self, bg=C_SIDEBAR, width=_sb_w)
        sb.pack(side="left", fill="y")
        sb.pack_propagate(False)

        tk.Frame(sb, height=1, bg=C_BORDER).pack(fill="x")
        brand = tk.Frame(sb, bg=C_SIDEBAR, pady=4)
        brand.pack(fill="x")

        _sb_logo = None
        _sb_logo_path = IMAGE_LOGO
        if os.path.exists(_sb_logo_path):
            try:
                from PIL import Image as _SbImg, ImageTk as _SbITk
                _sb_pil  = _SbImg.open(_sb_logo_path).convert("RGBA")
                _sb_h    = 28
                _sb_logo_w = int(_sb_pil.width * _sb_h / _sb_pil.height)
                _sb_pil  = _sb_pil.resize((_sb_logo_w, _sb_h), _SbImg.LANCZOS)
                _sb_logo = _SbITk.PhotoImage(_sb_pil)
            except Exception:
                _sb_logo = None

        brand_row = tk.Frame(brand, bg=C_SIDEBAR)
        brand_row.pack()

        if _sb_logo:
            _sb_wrap  = tk.Frame(brand_row, bg=C_ACCENT, padx=1, pady=1)
            _sb_wrap.pack(side="left", padx=(0, 6))
            _sb_inner = tk.Frame(_sb_wrap, bg=C_SIDEBAR, padx=3, pady=2)
            _sb_inner.pack()
            _sb_lbl   = tk.Label(_sb_inner, image=_sb_logo, bg=C_SIDEBAR)
            _sb_lbl.image = _sb_logo
            _sb_lbl.pack()
        else:
            tk.Label(brand_row, text="⬛", font=("Segoe UI", 14),
                     bg=C_SIDEBAR, fg=C_ACCENT).pack(side="left", padx=(0, 4))

        tk.Label(brand_row, text="SUDOKU AI",
                 font=("Segoe UI", 13, "bold"), bg=C_SIDEBAR, fg=C_TEXT).pack(side="left")
        tk.Label(brand, text=f"@{self.username}  |  {self.N}×{self.N}",
                 font=FONT_SMALL, bg=C_SIDEBAR, fg=C_TEXT_DIM).pack()
        tk.Frame(sb, height=1, bg=C_BORDER).pack(fill="x", pady=(4, 0))

        df = tk.Frame(sb, bg=C_SIDEBAR, pady=3)
        df.pack(fill="x", padx=10)
        tk.Label(df, text="DIFFICULTY", font=("Segoe UI", 8, "bold"),
                 bg=C_SIDEBAR, fg=C_TEXT_DIM).pack(anchor="w", pady=(2, 1))
        for d in ["Easy", "Normal", "Hard"]:
            b = tk.Button(df, text=get_diff_theme(d)["emoji"] + "  " + d,
                          font=FONT_BTN_SM, relief="flat", cursor="hand2", pady=2,
                          command=lambda dd=d: self._change_difficulty(dd))
            b.pack(fill="x", pady=1)
            b.config(bg=get_diff_theme(d)["accent"] if d == self.difficulty else C_SURFACE2,
                     fg=C_BG if d == self.difficulty else C_TEXT_DIM)
            setattr(self, f"_dbtn_{d}", b)

        tk.Frame(sb, height=1, bg=C_BORDER).pack(fill="x", padx=10, pady=(2, 3))

        cf = tk.Frame(sb, bg=C_SIDEBAR)
        cf.pack(fill="x", padx=10)
        tk.Label(cf, text="KONTROL", font=("Segoe UI", 8, "bold"),
                 bg=C_SIDEBAR, fg=C_TEXT_DIM).pack(anchor="w", pady=(2, 1))
        self._sb_btn(cf, "🔄  New Game",    "#DA3633", self._new_game)
        self._sb_btn(cf, "🏆  Leaderboard",  C_GOLD,
                     lambda: LeaderboardWindow(self.master, load_data()))
        self._sb_btn(cf, "◀  Back",   C_ACCENT,  self._back_to_grid)

        tk.Frame(cf, height=1, bg=C_BORDER).pack(fill="x", pady=(5, 3))

        btn_gp = tk.Button(cf,
            text="🔙  Switch Player",
            font=FONT_BTN_SM,
            bg=C_SIDEBAR_PURPLE_BG, fg=C_PURPLE,
            activebackground=C_PURPLE, activeforeground=C_BG,
            relief="flat", cursor="hand2", pady=4, anchor="w",
            command=self._change_player)
        btn_gp.pack(fill="x", pady=1)

        btn_lo = tk.Button(cf,
            text="🚪  Logout",
            font=FONT_BTN_SM,
            bg=C_SIDEBAR_RED_BG, fg=C_ERROR,
            activebackground=C_ERROR, activeforeground=C_BG,
            relief="flat", cursor="hand2", pady=4, anchor="w",
            command=self._kiosk_reset)
        btn_lo.pack(fill="x", pady=(1, 2))
        Tooltip(btn_lo, "Back to login page (F5)")

        tk.Frame(sb, height=1, bg=C_BORDER).pack(fill="x", padx=10, pady=(5, 2))

        self.draft_panel = tk.Frame(sb, bg=C_SIDEBAR)
        self.draft_panel.pack(fill="x", padx=10)
        if self.difficulty == "Hard":
            self._build_draft_panel()

        tk.Frame(sb, height=1, bg=C_BORDER).pack(fill="x", padx=10, pady=(4, 2))

        af = tk.Frame(sb, bg=C_SIDEBAR)
        af.pack(fill="x", padx=10)
        tk.Label(af, text="AI FEATURES", font=("Segoe UI", 8, "bold"),
                 bg=C_SIDEBAR, fg=C_TEXT_DIM).pack(anchor="w", pady=(2, 1))
        btn_bt = self._sb_btn(af, "🤖  Solve", "#1ABC9C", self._run_backtrack)
        Tooltip(btn_bt, "Solve using Backtracking + MRV heuristic")

        self._ml_toggle_btn = tk.Button(
            af,
            text="🔬  Live Analysis",
            font=("Segoe UI", 8, "bold"),
            bg=C_SURFACE2, fg=C_PURPLE,
            activebackground=C_PURPLE, activeforeground=C_BG,
            relief="flat", cursor="hand2", pady=3, anchor="w",
            command=self._toggle_ml_panel,
        )
        Tooltip(self._ml_toggle_btn, "View live analysis from the ML engine (i)")
        self._ml_toggle_btn.pack(fill="x", pady=1)

        tk.Frame(sb, height=1, bg=C_BORDER).pack(fill="x", padx=10, pady=(4, 2))

        sf = tk.Frame(sb, bg=C_SIDEBAR)
        sf.pack(fill="x", padx=10)
        tk.Label(sf, text="STATISTIK SESI", font=("Segoe UI", 8, "bold"),
                 bg=C_SIDEBAR, fg=C_TEXT_DIM).pack(anchor="w", pady=(2, 1))
        self.lbl_moves  = self._stat_row(sf, "Moves",        "0")
        self.lbl_errors = self._stat_row(sf, "Errors / Batas", "0/0")
        self.lbl_hints  = self._stat_row(sf, "Hints",  "0")
        if self.difficulty == "Hard":
            self.lbl_auto = self._stat_row(sf, "Auto ⚡", "0", col="#BC8CFF")
        else:
            self.lbl_auto = None

        tk.Frame(sf, height=1, bg=C_BORDER).pack(fill="x", pady=(3,1))
        self.lbl_nearmiss = self._stat_row(sf, "Hampir Benar", "0",
                                           col="#F0883E")
        self.lbl_guessing = self._stat_row(sf, "Asal Tebak",   "0",
                                           col=C_ERROR)

        tk.Frame(sf, height=1, bg=C_BORDER).pack(fill="x", pady=(4, 2))
        heart_lbl_row = tk.Frame(sf, bg=C_SIDEBAR)
        heart_lbl_row.pack(fill="x")
        tk.Label(heart_lbl_row, text="HATI (HINT)",
                 font=("Segoe UI", 8, "bold"),
                 bg=C_SIDEBAR, fg=C_TEXT_DIM).pack(side="left")
        self.lbl_hearts_count = tk.Label(heart_lbl_row,
                 text=f"{self.hearts}/{self.max_hearts}",
                 font=("Segoe UI", 8, "bold"),
                 bg=C_SIDEBAR, fg=C_ERROR)
        self.lbl_hearts_count.pack(side="right")

        self.heart_row = tk.Frame(sf, bg=C_SIDEBAR)
        self.heart_row.pack(fill="x", pady=(1,0))
        self.heart_labels = []
        for _ in range(self.max_hearts):
            lbl = tk.Label(self.heart_row, text="♥",
                           font=("Segoe UI", 10),
                           bg=C_SIDEBAR, fg=C_ERROR)
            lbl.pack(side="left")
            self.heart_labels.append(lbl)
        self._update_hearts_ui()

        hint_wrap = tk.Frame(sf, bg=C_SIDEBAR)
        hint_wrap.pack(fill="x", pady=(6, 4))

        hint_btn = tk.Button(
            hint_wrap,
            text="💡  HINT",
            font=("Segoe UI", 10, "bold"),
            bg="#F97316",
            fg="#FFF7ED",
            activebackground="#FB923C",
            activeforeground="#FFF7ED",
            relief="flat",
            cursor="hand2",
            pady=6,
            padx=12,
            anchor="center",
            justify="center",
            command=self._give_hint,
        )
        hint_btn.pack(fill="x", pady=1)

        main = tk.Frame(self, bg=C_BG)
        main.pack(side="right", fill="both", expand=True)

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

        self.draft_badge = tk.Label(
            title_row, text="",
            font=("Segoe UI", 10, "bold"),
            bg=C_SIDEBAR_PURPLE_BG, fg=C_PURPLE, padx=8, pady=2)

        self.timer_var = tk.StringVar(value="00:00")
        self.timer_lbl = tk.Label(topbar, textvariable=self.timer_var,
                                  font=FONT_TIMER, bg=C_SURFACE, fg=C_ERROR)
        self.timer_lbl.pack()

        self.status_var = tk.StringVar(value="Pilih sel dan ketik angka untuk mulai...")
        self.status_lbl = tk.Label(topbar, textvariable=self.status_var,
                                   font=FONT_SMALL, bg=C_SURFACE, fg=C_TEXT_DIM)
        self.status_lbl.pack()

        board_wrap = tk.Frame(main, bg=C_BG)
        board_wrap.pack(expand=True, fill="both")
        self.grid_container = tk.Frame(board_wrap,
                                       bg=self.theme["grid_line"], padx=3, pady=3)
        self.grid_container.place(relx=0.5, rely=0.5, anchor="center")
        self._build_grid()

        numpad_outer = tk.Frame(main, bg=C_BG, pady=8)
        numpad_outer.pack()
        self.numpad_row = tk.Frame(numpad_outer, bg=C_BG)
        self.numpad_row.pack()
        self._build_numpad()

        self.master.bind("<Key>",    self._on_key)
        self._idle_check()

    # Sidebar helpers
    # Menangani proses sb btn pada GameScreen sambil menjaga state internal tetap konsisten.
    def _sb_btn(self, parent, text, color, cmd):
        # Menangani proses cmd with sfx pada GameScreen sambil menjaga state internal tetap konsisten.
        def _cmd_with_sfx():
            _play_sfx(_SFX_CLICK)
            cmd()
        b = tk.Button(parent, text=text, font=FONT_BTN_SM,
                      bg=C_SURFACE2, fg=color,
                      activebackground=color, activeforeground=C_BG,
                      relief="flat", cursor="hand2", pady=3, anchor="w",
                      command=_cmd_with_sfx)
        b.pack(fill="x", pady=1)
        return b

    # Menangani proses stat row pada GameScreen sambil menjaga state internal tetap konsisten.
    def _stat_row(self, parent, label, val, col=None):
        r = tk.Frame(parent, bg=C_SIDEBAR)
        r.pack(fill="x", pady=0)
        tk.Label(r, text=label, font=("Segoe UI", 9),
                 bg=C_SIDEBAR, fg=C_TEXT_DIM).pack(side="left")
        lbl = tk.Label(r, text=val, font=("Segoe UI", 9, "bold"),
                       bg=C_SIDEBAR, fg=col if col else C_TEXT)
        lbl.pack(side="right")
        return lbl

    # Draft Panel (Hard only)
    # Membangun draft panel pada GameScreen dan menyiapkan widget supaya state tampilan tetap konsisten.
    def _build_draft_panel(self):
        for w in self.draft_panel.winfo_children():
            w.destroy()
        if self.difficulty != "Hard":
            return

        title_row = tk.Frame(self.draft_panel, bg=C_SIDEBAR)
        title_row.pack(fill="x")
        tk.Label(title_row, text="✏ DRAFT MODE",
                 font=("Segoe UI", 8, "bold"),
                 bg=C_SIDEBAR, fg=C_PURPLE).pack(side="left")
        tk.Label(title_row, text="  [D]",
                 font=("Segoe UI", 8),
                 bg=C_SIDEBAR, fg=C_TEXT_DIM).pack(side="left")

        self.draft_toggle_btn = tk.Button(
            self.draft_panel,
            text="✏  Aktifkan Draft Mode",
            font=FONT_BTN_SM,
            bg=C_SURFACE2, fg="#BC8CFF",
            activebackground="#4A2060", activeforeground="#E8AAFF",
            relief="flat", cursor="hand2", pady=4,
            command=self._toggle_draft_mode)
        self.draft_toggle_btn.pack(fill="x", pady=(2, 1))

        action_row = tk.Frame(self.draft_panel, bg=C_SIDEBAR)
        action_row.pack(fill="x", pady=0)

        self.auto_cand_btn = tk.Button(
            action_row,
            text="⚡ Auto",
            font=("Segoe UI", 8, "bold"),
            bg=C_SURFACE2, fg=C_ACCENT,
            activebackground=C_ACCENT, activeforeground=C_BG,
            relief="flat", cursor="hand2", pady=3,
            command=self._auto_fill_candidates)
        self.auto_cand_btn.pack(side="left", fill="x", expand=True, padx=(0, 2))

        self.clear_draft_btn = tk.Button(
            action_row,
            text="✗ Hapus",
            font=("Segoe UI", 8, "bold"),
            bg=C_NUMPAD_DEL_BG, fg=C_ERROR,
            activebackground=C_NUMPAD_DEL_HOV, activeforeground=C_BG,
            relief="flat", cursor="hand2", pady=3,
            command=self._clear_all_drafts)
        self.clear_draft_btn.pack(side="right", fill="x", expand=True, padx=(2, 0))

        tip = tk.Label(self.draft_panel,
                       text="Klik angka = toggle  |  ⚡ Auto = isi semua",
                       font=("Segoe UI", 7),
                       bg=C_SIDEBAR, fg=C_TEXT_DIM,
                       justify="left")
        tip.pack(anchor="w", pady=(1, 0))

        self.confirm_btn = None

    # Grid (Canvas cells)
    # Membangun grid pada GameScreen dan menyiapkan widget supaya state tampilan tetap konsisten.
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
                               bg=self.theme["grid_line"], padx=2, pady=2)
                blk.grid(row=br, column=bc, padx=3, pady=3)
                self.blocks[(br, bc)] = blk

        for r in range(self.N):
            for c in range(self.N):
                parent = self.blocks[(r // self.BOX, c // self.BOX)]
                cv = tk.Canvas(parent, width=px, height=px,
                               bg=self.theme["cell_bg"],
                               highlightthickness=0, relief="flat",
                               cursor="hand2")
                cv.grid(row=r % self.BOX, column=c % self.BOX,
                        padx=1, pady=1)
                cv.bind("<Button-1>", lambda e, rr=r, cc=c: self._on_click(rr, cc))
                self.canvases[(r, c)] = cv

    # Numpad
    # Membangun numpad pada GameScreen dan menyiapkan widget supaya state tampilan tetap konsisten.
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
                            bg=C_NUMPAD_DEL_BG, fg=C_ERROR,
                            activebackground=C_NUMPAD_DEL_HOV, activeforeground=C_ERROR,
                            relief="flat", cursor="hand2",
                            width=3, height=1, pady=4,
                            command=self._delete_cell)
        del_btn.pack(side="left", padx=2)

    # Memperbarui numpad pada GameScreen agar data, status, dan tampilan tetap selaras.
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
                btn.config(state="disabled", bg=C_NUMPAD_DIS_BG, fg=C_NUMPAD_DIS_FG,
                           text=f"✓{n}", font=("Segoe UI", 10, "bold"),
                           cursor="arrow")
            else:
                btn.config(state="normal", bg=C_SURFACE2, fg=C_TEXT,
                           text=str(n), font=("Segoe UI", 13, "bold"),
                           cursor="hand2")

    # Draft Mode Logic
    # Mengalihkan draft mode pada GameScreen sambil menjaga state internal tetap sinkron.
    def _toggle_draft_mode(self):
        if self.difficulty != "Hard": return
        self.draft_mode = not self.draft_mode
        self._update_draft_ui()
        if self.draft_mode:
            self.status_var.set(
                "✏ Draft Mode aktif - ketik angka untuk toggle kandidat, Enter konfirmasi")
        else:
            self.status_var.set("📝 Draft Mode nonaktif - kembali ke mode normal")
        self._draw_board()

    # Memperbarui draft ui pada GameScreen agar data, status, dan tampilan tetap selaras.
    def _update_draft_ui(self):
        btn = getattr(self, "draft_toggle_btn", None)
        if btn is None:
            try:
                self.draft_badge.pack_forget()
            except Exception:
                pass
            self.draft_toggle_btn = None
            return
        if self.draft_mode:
            self.draft_toggle_btn.config(
                text="✏  Draft Mode: AKTIF",
                bg="#4A2060", fg="#E8AAFF",
                activebackground="#6A30A0")
            self.draft_badge.config(text=" ✏ DRAFT MODE ")
            self.draft_badge.pack(side="left", padx=4)
        else:
            self.draft_toggle_btn.config(
                text="✏  Aktifkan Draft Mode",
                bg=C_SURFACE2, fg="#BC8CFF",
                activebackground="#4A2060")
            self.draft_badge.pack_forget()

    # Menangani proses auto fill candidates pada GameScreen sambil menjaga state internal tetap konsisten.
    def _auto_fill_candidates(self):
        if self.difficulty != "Hard": return
        added = 0
        for r in range(self.N):
            for c in range(self.N):
                if self.puzzle[r][c] != 0: continue
                if self.current_board[r][c] != 0: continue
                candidates = {
                    n for n in range(1, self.N + 1)
                    if is_valid(self.current_board, n, (r, c), self.N, self.BOX)
                }
                if candidates:
                    self.draft_board[(r, c)] = candidates
                    added += len(candidates)
        self.auto_used_count += 1
        self._draw_board()
        penalty_info = f" | -50 poin (total: {self.auto_used_count}×)"
        self.status_var.set(
            f"⚡ {added} kandidat diisi di {len(self.draft_board)} sel{penalty_info}")

    # Menangani proses eliminate candidates pada GameScreen sambil menjaga state internal tetap konsisten.
    def _eliminate_candidates(self, confirmed_r, confirmed_c, confirmed_num):
        for c2 in range(self.N):
            if c2 == confirmed_c: continue
            cands = self.draft_board.get((confirmed_r, c2))
            if cands:
                cands.discard(confirmed_num)
                if not cands:
                    self.draft_board.pop((confirmed_r, c2), None)
        for r2 in range(self.N):
            if r2 == confirmed_r: continue
            cands = self.draft_board.get((r2, confirmed_c))
            if cands:
                cands.discard(confirmed_num)
                if not cands:
                    self.draft_board.pop((r2, confirmed_c), None)
        br, bc = confirmed_r // self.BOX, confirmed_c // self.BOX
        for ri in range(br * self.BOX, br * self.BOX + self.BOX):
            for ci in range(bc * self.BOX, bc * self.BOX + self.BOX):
                if ri == confirmed_r and ci == confirmed_c: continue
                cands = self.draft_board.get((ri, ci))
                if cands:
                    cands.discard(confirmed_num)
                    if not cands:
                        self.draft_board.pop((ri, ci), None)

    # Menangani proses add draft pada GameScreen sambil menjaga state internal tetap konsisten.
    def _add_draft(self, r, c, num):
        if (r, c) not in self.draft_board:
            self.draft_board[(r, c)] = set()
        cands = self.draft_board[(r, c)]
        if num in cands:
            cands.discard(num)
            if not cands:
                self.draft_board.pop((r, c), None)
            self.status_var.set(f"✏ Kandidat {num} dihapus dari ({r+1},{c+1})")
        else:
            cands.add(num)
            self.status_var.set(
                f"✏ Kandidat {sorted(cands)} di ({r+1},{c+1})")
        self._draw_board()

    # Menangani proses clear all drafts pada GameScreen sambil menjaga state internal tetap konsisten.
    def _clear_all_drafts(self):
        total = sum(len(v) for v in self.draft_board.values())
        self.draft_board.clear()
        self._draw_board()
        if total:
            self.status_var.set(f"✗ {total} kandidat dihapus dari board")
        else:
            self.status_var.set("ℹ Tidak ada kandidat untuk dihapus")

    # Menangani proses confirm single draft pada GameScreen sambil menjaga state internal tetap konsisten.
    def _confirm_single_draft(self, r, c):
        if self.puzzle[r][c] != 0: return
        if self.current_board[r][c] != 0: return
        cands = self.draft_board.get((r, c), set())
        if len(cands) == 0:
            self.status_var.set("ℹ Sel ini tidak punya kandidat")
            return
        if len(cands) > 1:
            self.status_var.set(
                f"⚠ {len(cands)} kandidat tersisa - hapus yang salah dulu")
            return
        num = next(iter(cands))
        ok  = is_valid(self.current_board, num, (r, c), self.N, self.BOX)
        self.current_board[r][c] = num
        self.draft_board.pop((r, c), None)
        self.move_count += 1
        if not ok:
            self.error_count += 1
        else:
            self._eliminate_candidates(r, c, num)
        if not self.timer_running: self._start_timer()
        self._update_stat_labels()
        self._update_numpad()
        self._draw_board()
        icon = "✔" if ok else "⚠"
        self.status_var.set(f"{icon} [{num}] dikonfirmasi di ({r+1},{c+1})")
        if not ok:
            self._check_lose()
        if not self.game_over:
            self._check_win()
        """
        Konfirmasi semua sel dengan tepat 1 kandidat (naked singles).
        Satu klik untuk menyelesaikan semua deduksi trivial sekaligus.
        """
        to_do = [
            (r, c)
            for r in range(self.N) for c in range(self.N)
            if self.puzzle[r][c] == 0
            and self.current_board[r][c] == 0
            and len(self.draft_board.get((r, c), set())) == 1
        ]
        committed = 0
        for r, c in to_do:
            if self.game_over:
                break
            num = next(iter(self.draft_board.get((r, c), {0})))
            if num == 0: continue
            ok  = is_valid(self.current_board, num, (r, c), self.N, self.BOX)
            self.current_board[r][c] = num
            self.draft_board.pop((r, c), None)
            self.move_count += 1
            if not ok:
                self.error_count += 1
            else:
                self._eliminate_candidates(r, c, num)
            committed += 1
        if committed:
            self._update_stat_labels()
            self._update_numpad()
            self._draw_board()
            if not self.game_over:
                self.status_var.set(f"✔ {committed} naked single dikonfirmasi")
                self._check_lose()
                self._check_win()
        else:
            self.status_var.set("ℹ Tidak ada sel dengan tepat 1 kandidat")
    # Game State
    # Menangani proses reset all canvas colors pada GameScreen sambil menjaga state internal tetap konsisten.
    def _reset_all_canvas_colors(self):
        t = self.theme
        for cv in self.canvases.values():
            cv.config(bg=t["cell_bg"])
            cv.delete("all")
        for blk in self.blocks.values():
            blk.config(bg=t["grid_line"])
        self.grid_container.config(bg=t["grid_line"])

    # Memulai new game pada GameScreen dan menyalakan mekanisme pendukung yang dibutuhkan.
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
        self.auto_used_count = 0
        self.last_action   = time.time()
        self.hint_shown    = False
        self.max_errors    = self._get_max_errors()

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
        self._reset_all_canvas_colors()
        self._update_draft_ui()
        self._update_stat_labels()
        self._update_hearts_ui()
        self._update_numpad()
        self.update_idletasks()
        self._draw_board()

        if getattr(self, "_tutorial_pending", False):
            self._tutorial_pending = False
            self.after(500, self._show_tutorial)

    # Menangani proses new game pada GameScreen sambil menjaga state internal tetap konsisten.
    def _new_game(self):
        self._stop_timer()
        self._start_new_game()

    # Menangani proses restore state pada GameScreen sambil menjaga state internal tetap konsisten.
    def _restore_state(self, s):
        self.puzzle        = [row[:] for row in s["puzzle"]]
        self.solution      = [row[:] for row in s["solution"]]
        self.current_board = [row[:] for row in s["current_board"]]
        self.draft_board   = {k: set(v) for k, v in s["draft_board"].items()}
        self.draft_mode    = s["draft_mode"]
        self.elapsed       = s["elapsed"]
        self.game_over     = s["game_over"]
        self.error_count   = s["error_count"]
        self.move_count    = s["move_count"]
        self.hints_used    = s["hints_used"]
        self.auto_used_count = s["auto_used_count"]
        self.hearts        = s["hearts"]
        self.cell_errors   = dict(s["cell_errors"])
        self.cell_last_time = dict(s["cell_last_time"])
        self.near_miss_count = s["near_miss_count"]
        self.guessing_count  = s["guessing_count"]
        self.empty_cells   = s["empty_cells"]
        self.hint_shown    = s["hint_shown"]
        self.selected      = None

        self._reset_all_canvas_colors()
        self._update_draft_ui()
        self._update_stat_labels()
        self._update_hearts_ui()
        self._update_numpad()
        self.update_idletasks()
        self._draw_board()

        if s["timer_running"] and not self.game_over:
            self.timer_running = True
            self.start_time = time.time() - self.elapsed
            self.timer_lbl.config(fg=C_ACCENT2)
            self._tick()

    # Menangani proses change difficulty pada GameScreen sambil menjaga state internal tetap konsisten.
    def _change_difficulty(self, diff):
        self._stop_timer()
        self.difficulty = diff
        self.theme      = get_diff_theme(diff)
        t = self.theme

        for d in ["Easy", "Normal", "Hard"]:
            b = getattr(self, f"_dbtn_{d}")
            b.config(
                bg=get_diff_theme(d)["accent"] if d == diff else C_SURFACE2,
                fg=C_BG if d == diff else C_TEXT_DIM
            )

        self.diff_badge.config(
            text=f" {t['emoji']} {diff}  {self.N}×{self.N} ",
            bg=t["accent"]
        )

        self.grid_container.config(bg=t["grid_line"])
        for blk in self.blocks.values():
            blk.config(bg=t["grid_line"])

        for cv in self.canvases.values():
            cv.config(bg=t["cell_bg"])
            cv.delete("all")

        self.draft_toggle_btn  = None
        self.confirm_btn       = None
        self.clear_draft_btn   = None
        self.auto_cand_btn     = None
        for w in self.draft_panel.winfo_children():
            w.destroy()
        if diff == "Hard":
            self._build_draft_panel()

        self._build_numpad()

        self._start_new_game()

    # Menangani proses change player pada GameScreen sambil menjaga state internal tetap konsisten.
    def _change_player(self):
        self._stop_timer()
        self.master.event_generate("<<ChangePlayer>>")

    # Menangani proses logout pada GameScreen sambil menjaga state internal tetap konsisten.
    def _logout(self):
        self._stop_timer()
        self.master.event_generate("<<Logout>>")

    # Menangani proses kiosk reset pada GameScreen sambil menjaga state internal tetap konsisten.
    def _kiosk_reset(self):
        self._stop_timer()
        self.master.event_generate("<<Logout>>")

    # Menangani proses back to grid pada GameScreen sambil menjaga state internal tetap konsisten.
    def _back_to_grid(self):
        self._stop_timer()
        self.master.event_generate("<<BackToGrid>>")

    # Tutorial overlay (pemain baru)
    # Menampilkan tutorial pada GameScreen dan mengaktifkan elemen pendukung yang diperlukan.
    def _show_tutorial(self):
        try:
            TutorialOverlay(self, n=self.N, on_close=self._on_tutorial_done)
        except Exception:
            pass

    # Menangani event tutorial done pada GameScreen dan memperbarui state yang terkait.
    def _on_tutorial_done(self):
        try:
            data = load_data()
            if self.username not in data.get("players", {}):
                data["players"][self.username] = {
                    "sessions": [], "created_at": time.time()
                }
            data["players"][self.username]["tutorial_done"] = True
            save_data(data)
        except Exception:
            pass

    # Event Handlers
    # Menangani event click pada GameScreen dan memperbarui state yang terkait.
    def _on_click(self, r, c):
        self.selected = (r, c)
        if not self.game_over:
            self.last_action = time.time()
        self._draw_board()

    # Menangani event key pada GameScreen dan memperbarui state yang terkait.
    def _on_key(self, event):
        if self.game_over: return

        if event.char.lower() == "i":
            self._toggle_ml_panel()
            return

        if event.char.lower() == "d" and self.difficulty == "Hard":
            self._toggle_draft_mode()
            return

        if event.keysym in ("Return", "space") and self.draft_mode and self.selected:
            self._confirm_single_draft(*self.selected)
            return

        if event.keysym in ("BackSpace", "Delete"):
            self._delete_cell()
        elif event.char in [str(i) for i in range(1, self.N + 1)]:
            self._input_number(int(event.char))
        elif event.keysym in ("Up", "Down", "Left", "Right"):
            self._move_sel(event.keysym)

    # Menangani event esc pada GameScreen dan memperbarui state yang terkait.
    def _on_esc(self, _=None):
        self._show_back_to_grid_confirm()

    # Menampilkan back to grid confirm pada GameScreen dan mengaktifkan elemen pendukung yang diperlukan.
    def _show_back_to_grid_confirm(self):
        if hasattr(self, "_exit_popup") and self._exit_popup and \
                self._exit_popup.winfo_exists():
            return

        try:
            _root_ref = self.winfo_toplevel()
        except Exception:
            _root_ref = None
        _blur_pre_game = _grab_blur_bg(_root_ref) if _root_ref else None

        _overlay_parent = _root_ref if _root_ref else self
        overlay = tk.Frame(_overlay_parent, bg="")
        overlay.place(relx=0, rely=0, relwidth=1, relheight=1)
        overlay.lift()
        self._exit_popup = overlay

        if _root_ref:
            dim = _place_blur_canvas(overlay, _root_ref, pre_captured=_blur_pre_game)
        else:
            dim = tk.Canvas(overlay, bg="#050810", highlightthickness=0)
        if not PIL_AVAILABLE or _root_ref is None:
            dim.place(relx=0, rely=0, relwidth=1, relheight=1)

        card_bg     = C_SURFACE
        card_border = C_ACCENT

        glow_exit = tk.Frame(overlay, bg=card_border)
        glow_exit.place(relx=0.5, rely=0.62, anchor="center", width=440, height=248)

        card = tk.Frame(glow_exit, bg=card_bg,
                        highlightbackground=card_border, highlightthickness=0)
        card.place(relx=0, rely=0, relwidth=1, relheight=1)

        SLIDE_STEPS = 12
        # Menangani proses slide exit pada GameScreen sambil menjaga state internal tetap konsisten.
        def _slide_exit(step, _glow=glow_exit, _ov=overlay):
            if not _ov.winfo_exists():
                return
            t_s    = step / SLIDE_STEPS
            t_ease = 1 - (1 - t_s) ** 3
            rely_now = 0.62 + (0.5 - 0.62) * t_ease
            try:
                _glow.place(rely=rely_now)
            except Exception:
                return
            if step < SLIDE_STEPS:
                _ov.after(16, lambda: _slide_exit(step + 1))
        _slide_exit(0)

        _stripe_c1 = C_ACCENT
        _stripe_c2 = C_PURPLE
        stripe_cv = tk.Canvas(card, height=6, bg=card_bg, highlightthickness=0)
        stripe_cv.pack(fill="x")
        # Menggambar stripe pada GameScreen sesuai state yang sedang aktif.
        def _draw_stripe(cv=stripe_cv, _c1=_stripe_c1, _c2=_stripe_c2):
            cv.update_idletasks()
            w = cv.winfo_width() or 434
            stops = [(0, _c1), (w // 2, _c2), (w, _c1)]
            steps = 40
            for i in range(len(stops) - 1):
                x1, c1 = stops[i]
                x2, c2 = stops[i + 1]
                for k in range(steps):
                    t_g = k / steps
                    r_  = int(int(c1[1:3], 16) * (1 - t_g) + int(c2[1:3], 16) * t_g)
                    g_  = int(int(c1[3:5], 16) * (1 - t_g) + int(c2[3:5], 16) * t_g)
                    b_  = int(int(c1[5:7], 16) * (1 - t_g) + int(c2[5:7], 16) * t_g)
                    xi  = x1 + k * (x2 - x1) // steps
                    xj  = x1 + (k + 1) * (x2 - x1) // steps
                    cv.create_rectangle(xi, 0, xj, 6,
                                        fill=f"#{r_:02x}{g_:02x}{b_:02x}",
                                        outline="")
        stripe_cv.after(60, _draw_stripe)

        icon_row = tk.Frame(card, bg=card_bg)
        icon_row.pack(pady=(18, 4))
        tk.Label(icon_row, text="\U0001f3e0",
                 font=("Segoe UI", 26), bg=card_bg, fg=C_ACCENT).pack(side="left", padx=(0, 8))
        title_col = tk.Frame(icon_row, bg=card_bg)
        title_col.pack(side="left")
        tk.Label(title_col, text="Kembali ke Pilih Grid",
                 font=("Segoe UI", 16, "bold"),
                 bg=card_bg, fg=C_TEXT, anchor="w").pack(anchor="w")
        tk.Label(title_col, text="Sesi ini akan dihentikan",
                 font=("Segoe UI", 9),
                 bg=card_bg, fg=C_TEXT_DIM, anchor="w").pack(anchor="w")

        tk.Frame(card, height=1, bg=C_BORDER).pack(fill="x", padx=24, pady=(6, 0))

        tk.Label(card,
                 text="Apakah Anda ingin keluar dari game ini\ndan kembali ke layar pemilihan grid?",
                 font=("Segoe UI", 10),
                 bg=card_bg, fg=C_TEXT_DIM,
                 justify="center").pack(pady=(12, 16))

        btn_row = tk.Frame(card, bg=card_bg)
        btn_row.pack(fill="x", padx=24, pady=(0, 20))

        _corner_icons_lower()

        # Menangani proses cancel pada GameScreen sambil menjaga state internal tetap konsisten.
        def _cancel():
            _corner_icons_restore()
            try:
                overlay.destroy()
            except Exception:
                pass
            self._exit_popup = None

        # Menangani proses confirm back pada GameScreen sambil menjaga state internal tetap konsisten.
        def _confirm_back():
            _corner_icons_restore()
            try:
                overlay.destroy()
            except Exception:
                pass
            self._exit_popup = None
            self._back_to_grid()

        cancel_btn = tk.Button(
            btn_row,
            text="Lanjut Bermain",
            font=("Segoe UI", 10, "bold"),
            bg=C_SURFACE2, fg=C_TEXT,
            activebackground=C_BORDER, activeforeground=C_TEXT,
            relief="flat", cursor="hand2", pady=10,
            command=_cancel)
        cancel_btn.pack(side="left", fill="x", expand=True, padx=(0, 6))

        back_btn = tk.Button(
            btn_row,
            text="Ya, Kembali",
            font=("Segoe UI", 10, "bold"),
            bg=C_ACCENT, fg="#FFFFFF",
            activebackground=C_PURPLE, activeforeground="#FFFFFF",
            relief="flat", cursor="hand2", pady=10,
            command=_confirm_back)
        back_btn.pack(side="left", fill="x", expand=True, padx=(6, 0))

        cancel_btn.bind("<Enter>",
            lambda _: cancel_btn.config(bg=C_BORDER, fg=C_TEXT))
        cancel_btn.bind("<Leave>",
            lambda _: cancel_btn.config(bg=C_SURFACE2, fg=C_TEXT))

        dim.bind("<Button-1>", lambda _: _cancel())
        overlay.bind("<Button-1>", lambda e: _cancel()
                     if e.widget is overlay else None)

        # Menangani event esc back confirm pada GameScreen dan memperbarui state yang terkait.
        def _on_esc_back_confirm(event=None):
            _cancel()
            return "break"
        overlay.bind("<Escape>", _on_esc_back_confirm)
        overlay.focus_set()
        self._exit_popup = overlay

    # Alias lama - dipertahankan agar tidak ada referensi yang putus
    # Menampilkan exit confirm pada GameScreen dan mengaktifkan elemen pendukung yang diperlukan.
    def _show_exit_confirm(self):
        self._show_back_to_grid_confirm()

    # Menangani proses move sel pada GameScreen sambil menjaga state internal tetap konsisten.
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

    # Menangani proses input number pada GameScreen sambil menjaga state internal tetap konsisten.
    def _input_number(self, num):
        if self.game_over or not self.selected: return
        r, c = self.selected
        if self.puzzle[r][c] != 0: return

        if self.draft_mode and self.difficulty == "Hard":
            if self.current_board[r][c] != 0: return
            if not self.timer_running: self._start_timer()
            self.last_action = time.time()
            self.hint_shown  = False
            self._add_draft(r, c, num)
            return

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
            _play_sfx(_SFX_ERROR)
            now  = time.time()
            prev = self.cell_errors.get((r, c), 0)
            last = self.cell_last_time.get((r, c), 0)

            if prev == 0:
                self.near_miss_count += 1
            else:
                self.guessing_count += 1
                if prev == 1:
                    self.near_miss_count = max(0, self.near_miss_count - 1)

            self.cell_errors[(r, c)] = prev + 1
            self.cell_last_time[(r, c)] = now
        else:
            self.cell_errors.pop((r, c), None)
            self.cell_last_time.pop((r, c), None)
            _play_sfx(_SFX_CORRECT)

        self.draft_board.pop((r, c), None)
        if num_valid:
            self._eliminate_candidates(r, c, num)

        self._update_stat_labels()
        self._update_numpad()
        self._draw_board()
        if not num_valid:
            self._check_lose()
        if not self.game_over:
            self._check_win()

    # Menangani proses delete cell pada GameScreen sambil menjaga state internal tetap konsisten.
    def _delete_cell(self):
        if not self.selected or self.game_over: return
        r, c = self.selected
        if self.puzzle[r][c] != 0: return

        if self.draft_mode and self.difficulty == "Hard":
            cands = self.draft_board.pop((r, c), set())
            self._draw_board()
            if cands:
                self.status_var.set(
                    f"✗ Kandidat {sorted(cands)} dihapus dari ({r+1},{c+1})")
            else:
                self.status_var.set(f"ℹ Tidak ada kandidat di ({r+1},{c+1})")
            return

        self.current_board[r][c] = 0
        self.draft_board.pop((r, c), None)
        self.last_action = time.time()
        self._update_numpad()
        self._draw_board()

    # Cross-line helpers
    # Menangani proses get cross cells pada GameScreen sambil menjaga state internal tetap konsisten.
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

    # Menangani proses is cell error pada GameScreen sambil menjaga state internal tetap konsisten.
    def _is_cell_error(self, r, c):
        val = self.current_board[r][c]
        if val == 0: return False
        self.current_board[r][c] = 0
        valid = is_valid(self.current_board, val, (r, c), self.N, self.BOX)
        self.current_board[r][c] = val
        return not valid

    # Canvas-based board drawing
    # Menggambar board pada GameScreen sesuai state yang sedang aktif.
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

                if is_fixed:
                    bg = t["cell_fixed_bg"]
                    fg = t["cell_fixed_fg"]
                else:
                    if val != 0:
                        is_err = self._is_cell_error(r, c)
                        bg = t["error_bg"] if is_err else t.get("cell_user_bg", t["cell_bg"])
                        fg = t["error_fg"] if is_err else t["cell_user_fg"]
                    else:
                        bg = t["cell_bg"]
                        fg = t["cell_user_fg"]

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

                draft_cands = (
                    self.draft_board.get((r, c), set())
                    if val == 0 else set()
                )
                has_draft = bool(draft_cands) and self.difficulty == "Hard"

                cell_bg = bg
                if has_draft and not is_sel:
                    cell_bg = self._blend_draft(bg)

                cv.config(bg=cell_bg)
                cv.delete("all")

                if val != 0:
                    cv.create_text(
                        px // 2, px // 2,
                        text=str(val),
                        fill=fg,
                        font=("Segoe UI", main_font_sz, "bold"),
                        anchor="center"
                    )

                elif has_draft:

                    BOX = self.BOX
                    sub_w = px / BOX
                    sub_h = px / BOX

                    if BOX == 3:
                        cand_font_sz = 7 if px < 55 else 8
                    else:
                        cand_font_sz = 11

                    is_light_theme = C_BG.upper() not in ("#0D1117", "#0D111700") and \
                        int(C_BG[1:3], 16) > 150 if len(C_BG) >= 7 else False
                    if is_light_theme:
                        cand_col_base   = "#6A3DBB" if is_sel else "#7B4FCF"
                        cand_col_single = "#CC6600" if is_sel else "#B85C00"
                    else:
                        cand_col_base   = "#C89EFF" if is_sel else "#9370DB"
                        cand_col_single = "#FFD700" if is_sel else "#F0883E"

                    is_naked_single = (len(draft_cands) == 1)

                    for digit in range(1, self.N + 1):
                        if digit not in draft_cands:
                            continue
                        di  = (digit - 1) // BOX
                        dj  = (digit - 1) % BOX
                        cx  = dj * sub_w + sub_w / 2
                        cy  = di * sub_h + sub_h / 2
                        col = cand_col_single if is_naked_single else cand_col_base
                        cv.create_text(
                            cx, cy,
                            text=str(digit),
                            fill=col,
                            font=("Segoe UI", cand_font_sz, "bold"),
                            anchor="center"
                        )

                    if is_naked_single:
                        border_col = "#F0883E" if not is_sel else "#FFD700"
                        cv.create_rectangle(
                            2, 2, px - 3, px - 3,
                            outline=border_col, width=1
                        )
                        if is_sel:
                            cv.create_text(
                                px - 3, px - 3,
                                text="↵",
                                fill="#7EE787",
                                font=("Segoe UI", 9, "bold"),
                                anchor="se"
                            )

                else:
                    if not is_fixed and self.draft_mode:
                        cv.create_text(
                            px // 2, px // 2,
                            text="·",
                            fill="#2A1A40",
                            font=("Segoe UI", 18),
                            anchor="center"
                        )

                if not is_fixed and val != 0 and not self._is_cell_error(r, c) and not is_sel:
                    acc = t["accent"]
                    cv.create_rectangle(
                        1, 1, px - 2, px - 2,
                        outline=acc, width=1
                    )
                    pip_w = max(8, px // 3)
                    pip_x1 = (px - pip_w) // 2
                    pip_x2 = pip_x1 + pip_w
                    cv.create_rectangle(
                        pip_x1, px - 5, pip_x2, px - 3,
                        fill=acc, outline=""
                    )

                if is_sel:
                    cv.create_rectangle(
                        1, 1, px-2, px-2,
                        outline=t["hl_sel_border"],
                        width=2
                    )

                if not is_fixed and val != 0 and self._is_cell_error(r, c):
                    err_col = t.get("error_indicator", t["error_fg"])
                    cv.create_rectangle(
                        1, 1, px - 2, px - 2,
                        outline=err_col,
                        width=2
                    )
                    cv.create_text(
                        px - 3, 3,
                        text="!",
                        fill=err_col,
                        font=("Segoe UI", 8, "bold"),
                        anchor="ne"
                    )

    # _draw_draft_numbers removed - draft is now always single value,
    # rendered directly in _draw_board as a large centered digit.

    # Menangani proses blend draft pada GameScreen sambil menjaga state internal tetap konsisten.
    def _blend_draft(self, hex_bg):
        try:
            r1,g1,b1 = int(hex_bg[1:3],16),int(hex_bg[3:5],16),int(hex_bg[5:7],16)
            is_light = (r1 + g1 + b1) / 3 > 180
            if is_light:
                r2,g2,b2 = 0xBB, 0x99, 0xEE
                rr = int(r1*0.65 + r2*0.35)
                gg = int(g1*0.65 + g2*0.35)
                bb = int(b1*0.65 + b2*0.35)
            else:
                r2,g2,b2 = 0x3A, 0x1A, 0x5A
                rr = int(r1*0.6 + r2*0.4)
                gg = int(g1*0.6 + g2*0.4)
                bb = int(b1*0.6 + b2*0.4)
            return f"#{rr:02x}{gg:02x}{bb:02x}"
        except Exception:
            return hex_bg

    # Menangani proses blend pada GameScreen sambil menjaga state internal tetap konsisten.
    def _blend(self, hex1, hex2):
        try:
            r1,g1,b1 = int(hex1[1:3],16),int(hex1[3:5],16),int(hex1[5:7],16)
            r2,g2,b2 = int(hex2[1:3],16),int(hex2[3:5],16),int(hex2[5:7],16)
            return f"#{(r1+r2)//2:02x}{(g1+g2)//2:02x}{(b1+b2)//2:02x}"
        except Exception:
            return hex1

    # Win check
    # Menangani proses is board valid and complete pada GameScreen sambil menjaga state internal tetap konsisten.
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

    # Menangani proses check win pada GameScreen sambil menjaga state internal tetap konsisten.
    def _check_win(self):
        if not self._is_board_valid_and_complete(): return
        self.game_over = True
        self._stop_timer()
        _play_sfx(_SFX_WIN)
        self.status_var.set("🎉 SELAMAT! Puzzle selesai!")
        self.timer_lbl.config(fg=C_ACCENT2)
        self._flash_win()
        session = self._build_session(completed=True)
        self._save_session(session)
        self.master.after(1600, lambda: self.on_finish(session, self.ml))

    # Menangani proses flash win pada GameScreen sambil menjaga state internal tetap konsisten.
    def _flash_win(self):
        t  = self.theme
        cs = list(self.canvases.values())
        pulse_cols = [t["highlight"], t["accent"], t["cell_bg"]]
        # Menangani proses step pada GameScreen sambil menjaga state internal tetap konsisten.
        def step(i):
            col = pulse_cols[i % len(pulse_cols)]
            for cv in cs:
                cv.config(bg=col)
            if i < 6:
                self.master.after(140, lambda: step(i+1))
            else:
                self._draw_board()
        step(0)

    # Lose condition
    # Menangani proses get max errors pada GameScreen sambil menjaga state internal tetap konsisten.
    def _get_max_errors(self):
        base = {"Easy": 5, "Normal": 5, "Hard": 3}.get(self.difficulty, 5)
        if self.ml.sessions:
            try:
                pt, _, _ = self.ml.classify_player_confidence()
                if pt == "Struggling":
                    base = base + 1
                elif pt == "Expert":
                    base = max(1, base - 1)
            except Exception:
                pass
        return base

    # Menangani proses check lose pada GameScreen sambil menjaga state internal tetap konsisten.
    def _check_lose(self):
        if self.game_over:
            return
        if self.error_count >= self.max_errors:
            self._trigger_lose()

    # Menangani proses trigger lose pada GameScreen sambil menjaga state internal tetap konsisten.
    def _trigger_lose(self):
        if self.game_over:
            return
        self.game_over = True
        self._stop_timer()
        _play_sfx(_SFX_ERROR)
        self.status_var.set(
            f"💀 GAME OVER!")
        self.timer_lbl.config(fg=C_ERROR)
        self._flash_lose()
        session = self._build_session(completed=False, lose_reason="max_errors")
        self._save_session(session)
        self.master.after(2000, lambda: self.on_finish(session, self.ml))

    # Menangani proses flash lose pada GameScreen sambil menjaga state internal tetap konsisten.
    def _flash_lose(self):
        cs      = list(self.canvases.values())
        t       = self.theme
        err_col = t.get("error_bg", "#3B1212")
        norm    = t["cell_bg"]
        cols    = [err_col, norm, err_col, norm, err_col, norm, err_col, norm]
        # Menangani proses step pada GameScreen sambil menjaga state internal tetap konsisten.
        def step(i):
            col = cols[i] if i < len(cols) else norm
            for cv in cs:
                cv.config(bg=col)
            if i < len(cols) - 1:
                self.master.after(120, lambda: step(i + 1))
            else:
                self._draw_board()
        step(0)

    # Menangani proses give hint pada GameScreen sambil menjaga state internal tetap konsisten.
    def _give_hint(self, auto=False):
        if self.game_over: return

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
                        self.cell_errors.pop((r, c), None)
                        self.cell_last_time.pop((r, c), None)
                        self.hints_used += 1
                        self.hearts = max(0, self.hearts - 1)
                        if not self.timer_running: self._start_timer()
                        src = "idle" if auto else "manual"
                        remaining = self.hearts
                        heart_str = "♥" * remaining + "♡" * (self.max_hearts - remaining)
                        self.status_var.set(
                            f"💡 Hint ({src}) - sisa hati: {heart_str}")
                        self._update_stat_labels()
                        self._update_hearts_ui()
                        self._update_numpad()
                        self._draw_board()
                        self._check_win()
                        return
        self.status_var.set("✅ Semua sel sudah terisi dengan benar!")

    # Menangani proses idle check pada GameScreen sambil menjaga state internal tetap konsisten.
    def _idle_check(self):
        if self.game_over:
            return
        if self.timer_running:
            idle = time.time() - self.last_action

            remaining_empty = sum(
                1 for r in range(self.N) for c in range(self.N)
                if self.current_board[r][c] == 0
                and self.puzzle[r][c] == 0
            )
            remaining_pct = remaining_empty / max(self.empty_cells, 1)

            give, reason = self.ml.should_give_hint(
                idle,
                self.error_count,
                self.move_count,
                grid_size=self.grid_size,
                difficulty=self.difficulty,
                remaining_pct=remaining_pct,
            )
            if give and not self.hint_shown:
                self.hint_shown = True
                if reason == "idle":
                    self._banner(
                        f"Sudah diam {idle:.0f} detik. Perlu hint? 💡",
                        auto_hint=True)
                elif reason == "errors":
                    if self.difficulty == "Easy":
                        self._banner(
                            "Banyak kesalahan! Coba gunakan hint untuk membantu. 💡",
                            auto_hint=True)
                    else:
                        self._banner(
                            "Banyak kesalahan! Coba turunkan level? 💡",
                            suggest_lower=True)
        self.idle_after = self.master.after(3000, self._idle_check)

    # Menangani proses banner pada GameScreen sambil menjaga state internal tetap konsisten.
    def _banner(self, msg, auto_hint=False, suggest_lower=False):
        _ban_bg = "#1C2530" if _CURRENT_THEME_NAME == "dark" else C_SURFACE
        ban = tk.Frame(self, bg=_ban_bg,
                       highlightbackground=C_WARN, highlightthickness=1)
        ban.place(relx=0.5, y=8, anchor="n")
        tk.Label(ban, text=msg, font=FONT_BODY,
                 bg=_ban_bg, fg=C_WARN, padx=12, pady=7).pack(side="left")
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
                  bg=_ban_bg, fg=C_TEXT_DIM, relief="flat", cursor="hand2",
                  command=ban.destroy).pack(side="left", padx=4)
        self.master.after(8000, lambda: ban.destroy() if ban.winfo_exists() else None)

    # Stats / Timer
    # Memperbarui stat labels pada GameScreen agar data, status, dan tampilan tetap selaras.
    def _update_stat_labels(self):
        self.lbl_moves.config(text=str(self.move_count))
        remaining = self.max_errors - self.error_count
        err_text  = f"{self.error_count}/{self.max_errors}"
        if self.error_count == 0:
            err_col = C_TEXT
        elif remaining <= 1:
            err_col = C_ERROR
        elif remaining <= 2:
            err_col = C_WARN
        else:
            err_col = C_ERROR
        self.lbl_errors.config(text=err_text, fg=err_col)
        self.lbl_hints.config(text=str(self.hints_used))
        if self.lbl_auto is not None:
            self.lbl_auto.config(text=str(self.auto_used_count),
                                  fg=C_PURPLE if self.auto_used_count > 0 else C_TEXT)
        self.lbl_nearmiss.config(text=str(self.near_miss_count),
                                  fg="#F0883E" if self.near_miss_count else C_TEXT)
        self.lbl_guessing.config(text=str(self.guessing_count),
                                  fg=C_ERROR if self.guessing_count else C_TEXT)
        if self._ml_panel_visible and (time.time() - self._ml_last_refresh > 1.0):
            self._schedule_ml_refresh()

    # Memperbarui hearts ui pada GameScreen agar data, status, dan tampilan tetap selaras.
    def _update_hearts_ui(self):
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

    # Memulai timer pada GameScreen dan menyalakan mekanisme pendukung yang dibutuhkan.
    def _start_timer(self):
        if not self.timer_running and not self.game_over:
            self.timer_running = True
            self.start_time    = time.time()
            self.timer_lbl.config(fg=C_ACCENT2)
            self._tick()

    # Menghentikan timer pada GameScreen dan membersihkan state yang masih aktif.
    def _stop_timer(self):
        self.timer_running = False
        if self.idle_after:
            try: self.master.after_cancel(self.idle_after)
            except Exception: pass

    # Menangani proses tick pada GameScreen sambil menjaga state internal tetap konsisten.
    def _tick(self):
        if self.timer_running:
            self.elapsed = time.time() - self.start_time
            m, s = int(self.elapsed // 60), int(self.elapsed % 60)
            self.timer_var.set(f"{m:02}:{s:02}")
            self.master.after(500, self._tick)

    # Membangun session pada GameScreen dan menyiapkan widget supaya state tampilan tetap konsisten.
    def _build_session(self, completed, lose_reason=None):
        tpc = self.elapsed / max(self.empty_cells, 1)
        s = {
            "username":       self.username,
            "difficulty":     self.difficulty,
            "grid_size":      self.grid_size,
            "total_time":     self.elapsed,
            "moves":          self.move_count,
            "errors":         self.error_count,
            "hints_used":     self.hints_used,
            "auto_used":      self.auto_used_count,
            "completed":      completed,
            "timestamp":      time.time(),
            "empty_cells":    self.empty_cells,
            "time_per_cell":  round(tpc, 3),
            "near_miss":      self.near_miss_count,
            "guessing":       self.guessing_count,
            "hearts_left":    self.hearts,
            "max_hearts":     self.max_hearts,
            "max_errors":     self.max_errors,
            "score":          calculate_score(
                self.difficulty, self.elapsed, self.empty_cells,
                self.error_count, self.hints_used, completed,
                self.near_miss_count, self.guessing_count, self.auto_used_count),
        }
        if lose_reason:
            s["lose_reason"] = lose_reason
        return s

    # Menangani proses save session pada GameScreen sambil menjaga state internal tetap konsisten.
    def _save_session(self, s):
        data = load_data()
        if self.username not in data["players"]:
            data["players"][self.username] = {"sessions": [], "created_at": time.time()}

        player_data = data["players"][self.username]
        sessions = _dedupe_sessions(player_data.get("sessions", []))
        fp = _session_fingerprint(s)

        existing_fps = {_session_fingerprint(x) for x in sessions}
        is_new = fp not in existing_fps
        if is_new:
            sessions.append(s)
            self.ml.add_session(s)

        player_data["sessions"] = sessions
        save_data(data)

        if is_new:
            _ml_schedule_retrain(self.ml)

    # AI Solver
    # Menangani proses animate pada GameScreen sambil menjaga state internal tetap konsisten.
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

        # Menangani proses step pada GameScreen sambil menjaga state internal tetap konsisten.
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

    # Menangani proses run backtrack pada GameScreen sambil menjaga state internal tetap konsisten.
    def _run_backtrack(self):
        self.current_board = [row[:] for row in self.puzzle]
        self.draft_board   = {}
        hist, exp, t = solve_backtracking_mrv(self.puzzle, self.N, self.BOX)
        self._animate(hist,
                      f"Backtracking MRV - {exp} nodes | {t*1000:.1f}ms")

    # ML TRANSPARENCY PANEL
    # Panel overlay yang menampilkan output ML secara real-time:
    #   • KNN Player Type + Confidence
    #   • HintTimer RFR threshold adaptif
    #   • RFR predicted next score
    #   • IsolationForest anomaly status
    #   • MultiOutput skill bars (speed/accuracy/consistency/independence)
    # Toggle: tombol "🔬 ML Panel [I]" di sidebar atau tekan I saat game.
    # Refresh: setiap 8 detik atau setelah setiap move (min 5 detik jeda).

    # Snapshot state sesi berjalan
    # Membangun live session pada GameScreen dan menyiapkan widget supaya state tampilan tetap konsisten.
    def _build_live_session(self):
        elapsed = (time.time() - self.start_time
                   if self.timer_running else self.elapsed)
        moves = max(self.move_count, 1)
        ec    = max(self.empty_cells, 1)
        return {
            "username":    self.username,
            "difficulty":  self.difficulty,
            "grid_size":   self.grid_size,
            "total_time":  elapsed,
            "moves":       self.move_count,
            "errors":      self.error_count,
            "hints_used":  self.hints_used,
            "auto_used":   self.auto_used_count,
            "completed":   False,
            "timestamp":   time.time(),
            "empty_cells": ec,
            "time_per_cell": elapsed / ec,
            "near_miss":   self.near_miss_count,
            "guessing":    self.guessing_count,
            "score":       0,
        }

    # Menangani proses remaining pct pada GameScreen sambil menjaga state internal tetap konsisten.
    def _remaining_pct(self):
        remaining = sum(
            1 for rr in range(self.N) for cc in range(self.N)
            if self.current_board[rr][cc] == 0
        )
        return remaining / max(self.empty_cells, 1)

    # Toggle visibilitas panel
    # Mengalihkan ml panel pada GameScreen sambil menjaga state internal tetap sinkron.
    def _toggle_ml_panel(self):
        if self._ml_panel_visible:
            self._hide_ml_panel()
        else:
            self._show_ml_panel()

    # Menampilkan ml panel pada GameScreen dan mengaktifkan elemen pendukung yang diperlukan.
    def _show_ml_panel(self):
        self._ml_panel_visible = True
        try:
            self._ml_toggle_btn.config(
                bg=C_PURPLE, fg=C_BG,
                activebackground=C_SURFACE2, activeforeground=C_PURPLE,
            )
        except Exception:
            pass
        self._build_ml_transparency_panel()
        # Refresh pertama dijadwalkan oleh _build_ml_transparency_panel via after(120,...)

    # Menyembunyikan ml panel pada GameScreen dan merapikan state tampilan yang terkait.
    def _hide_ml_panel(self):
        self._ml_panel_visible = False
        try:
            self._ml_toggle_btn.config(
                bg=C_SURFACE2, fg=C_PURPLE,
                activebackground=C_PURPLE, activeforeground=C_BG,
            )
        except Exception:
            pass
        if self._ml_panel_job:
            try:
                self.after_cancel(self._ml_panel_job)
            except Exception:
                pass
            self._ml_panel_job = None
        if self._ml_panel_frame and self._ml_panel_frame.winfo_exists():
            try:
                self._ml_panel_frame.destroy()
            except Exception:
                pass
        self._ml_panel_frame = None

    # Bangun overlay panel (widget tetap, data diisi kemudian)
    # Membangun ml transparency panel pada GameScreen dan menyiapkan widget supaya state tampilan tetap konsisten.
    def _build_ml_transparency_panel(self):
        if self._ml_panel_frame and self._ml_panel_frame.winfo_exists():
            try:
                self._ml_panel_frame.destroy()
            except Exception:
                pass

        is_light = (_CURRENT_THEME_NAME == "light")
        bg_panel  = C_SURFACE
        bg_card   = C_SURFACE2
        sep_col   = C_BORDER

        W_PANEL = 292

        self.update_idletasks()
        MARGIN_RIGHT = 40
        MARGIN_TOP   = 195

        if self._ml_panel_pos is not None:
            _px, _py = self._ml_panel_pos
        else:
            try:
                sw   = self.winfo_width()
                _px  = max(0, sw - W_PANEL - MARGIN_RIGHT)
                _py  = MARGIN_TOP
            except Exception:
                _px = 900
                _py = MARGIN_TOP

        pnl = tk.Frame(
            self,
            bg=bg_panel,
            highlightbackground=C_ACCENT,
            highlightthickness=1,
        )
        pnl.place(x=_px, y=_py, anchor="nw")
        self._ml_panel_frame = pnl

        hdr = tk.Frame(pnl, bg=C_PURPLE, width=W_PANEL, cursor="fleur")
        hdr.pack(fill="x")
        hdr_inner = tk.Frame(hdr, bg=C_PURPLE)
        hdr_inner.pack(fill="x", padx=10, pady=6)
        tk.Label(hdr_inner, text="🔬  LIVE ANALYSIS",
                 font=("Segoe UI", 9, "bold"),
                 bg=C_PURPLE, fg=C_BG).pack(side="left")
        close_btn = tk.Button(
            hdr_inner, text="✕",
            font=("Segoe UI", 9, "bold"),
            bg=C_PURPLE, fg=C_BG,
            activebackground="#9A6FDF", activeforeground=C_BG,
            relief="flat", cursor="hand2",
            command=self._hide_ml_panel,
        )
        close_btn.pack(side="right")

        self._ml_drag_x = 0
        self._ml_drag_y = 0
        # Menangani proses drag start pada GameScreen sambil menjaga state internal tetap konsisten.
        def _drag_start(e):
            self._ml_drag_x = e.x_root - pnl.winfo_x()
            self._ml_drag_y = e.y_root - pnl.winfo_y()
        # Menangani proses drag move pada GameScreen sambil menjaga state internal tetap konsisten.
        def _drag_move(e):
            nx = e.x_root - self._ml_drag_x
            ny = e.y_root - self._ml_drag_y
            try:
                sw = self.winfo_width()
                sh = self.winfo_height()
                nx = max(0, min(nx, sw - W_PANEL))
                ny = max(0, min(ny, sh - 60))
            except Exception:
                pass
            pnl.place(x=nx, y=ny, anchor="nw")
            self._ml_panel_pos = (nx, ny)
        # Menangani proses drag end pada GameScreen sambil menjaga state internal tetap konsisten.
        def _drag_end(e):
            try:
                self._ml_panel_pos = (pnl.winfo_x(), pnl.winfo_y())
            except Exception:
                pass
        for _w in (hdr, hdr_inner):
            _w.bind("<ButtonPress-1>",   _drag_start)
            _w.bind("<B1-Motion>",       _drag_move)
            _w.bind("<ButtonRelease-1>", _drag_end)
        self._ml_refresh_btn = tk.Button(
            hdr_inner, text="🔄",
            font=("Segoe UI", 9),
            bg=C_PURPLE, fg=C_BG,
            activebackground="#9A6FDF", activeforeground=C_BG,
            relief="flat", cursor="hand2",
            command=self._schedule_ml_refresh,
        )
        self._ml_refresh_btn.pack(side="right", padx=(0, 4))
        Tooltip(self._ml_refresh_btn, "Refresh data ML sekarang")

        body_outer = tk.Frame(pnl, bg=bg_panel, width=W_PANEL)
        body_outer.pack(fill="x", padx=0, pady=0)
        sc_body = tk.Canvas(body_outer, bg=bg_panel, highlightthickness=0, width=W_PANEL)
        sc_body.pack(side="left", fill="both", expand=True)
        body = tk.Frame(sc_body, bg=bg_panel, width=W_PANEL)
        sc_body.create_window((0, 0), window=body, anchor="nw")
        # Menangani event body configure pada GameScreen dan memperbarui state yang terkait.
        def _on_body_configure(e=None):
            sc_body.configure(scrollregion=sc_body.bbox("all"))
            h = min(body.winfo_reqheight(), 600)
            sc_body.configure(height=h)
        body.bind("<Configure>", _on_body_configure)
        sc_body.bind("<MouseWheel>", lambda e: sc_body.yview_scroll(int(-1*(e.delta/120)), "units"))
        body.bind("<MouseWheel>", lambda e: sc_body.yview_scroll(int(-1*(e.delta/120)), "units"))
        body_inner_pad = tk.Frame(body, bg=bg_panel)
        body_inner_pad.pack(fill="x", padx=8, pady=6)
        body = body_inner_pad

        # Menangani proses sep pada GameScreen sambil menjaga state internal tetap konsisten.
        def _sep():
            tk.Frame(body, height=1, bg=sep_col).pack(fill="x", pady=(4, 2))

        # Menangani proses section title pada GameScreen sambil menjaga state internal tetap konsisten.
        def _section_title(icon, title, color=C_TEXT_DIM):
            r = tk.Frame(body, bg=bg_panel)
            r.pack(fill="x", pady=(4, 1))
            tk.Label(r, text=f"{icon}  {title}",
                     font=("Segoe UI", 8, "bold"),
                     bg=bg_panel, fg=color).pack(side="left")

        _section_title("🎯", "PLAYER TYPE", C_ACCENT)
        knn_card = tk.Frame(body, bg=bg_card,
                            highlightbackground=C_ACCENT, highlightthickness=1)
        knn_card.pack(fill="x", pady=2)
        knn_inner = tk.Frame(knn_card, bg=bg_card)
        knn_inner.pack(fill="x", padx=8, pady=5)

        knn_row = tk.Frame(knn_inner, bg=bg_card)
        knn_row.pack(fill="x")
        self._ml_knn_emoji  = tk.Label(knn_row, text="?",
                                        font=("Segoe UI", 14),
                                        bg=bg_card, fg=C_ACCENT)
        self._ml_knn_emoji.pack(side="left")
        self._ml_knn_type   = tk.Label(knn_row, text="Loading…",
                                        font=("Segoe UI", 10, "bold"),
                                        bg=bg_card, fg=C_TEXT)
        self._ml_knn_type.pack(side="left", padx=(4, 0))
        self._ml_knn_conf   = tk.Label(knn_row, text="",
                                        font=("Segoe UI", 8),
                                        bg=bg_card, fg=C_TEXT_DIM)
        self._ml_knn_conf.pack(side="right")

        conf_track = tk.Frame(knn_inner, bg=C_BORDER, height=6)
        conf_track.pack(fill="x", pady=(4, 0))
        self._ml_knn_bar = tk.Frame(conf_track, bg=C_ACCENT, height=6, width=4)
        self._ml_knn_bar.place(x=0, y=0)

        _sep()
        _section_title("⏱", "HINT TIMER", C_WARN)
        hint_card = tk.Frame(body, bg=bg_card,
                             highlightbackground=C_WARN, highlightthickness=1)
        hint_card.pack(fill="x", pady=2)
        hint_inner = tk.Frame(hint_card, bg=bg_card)
        hint_inner.pack(fill="x", padx=8, pady=5)
        thresh_row = tk.Frame(hint_inner, bg=bg_card)
        thresh_row.pack(fill="x")
        tk.Label(thresh_row, text="Threshold:",
                 font=("Segoe UI", 8), bg=bg_card, fg=C_TEXT_DIM).pack(side="left")
        self._ml_hint_thresh = tk.Label(thresh_row, text="- det",
                                         font=("Segoe UI", 10, "bold"),
                                         bg=bg_card, fg=C_WARN)
        self._ml_hint_thresh.pack(side="left", padx=(4, 0))
        idle_row = tk.Frame(hint_inner, bg=bg_card)
        idle_row.pack(fill="x", pady=(2, 0))
        tk.Label(idle_row, text="Idle saat ini:",
                 font=("Segoe UI", 8), bg=bg_card, fg=C_TEXT_DIM).pack(side="left")
        self._ml_hint_idle = tk.Label(idle_row, text="- det",
                                       font=("Segoe UI", 9, "bold"),
                                       bg=bg_card, fg=C_TEXT)
        self._ml_hint_idle.pack(side="left", padx=(4, 0))

        _sep()
        _section_title("📈", "PREDICTED SCORE", C_ACCENT2)
        rfr_card = tk.Frame(body, bg=bg_card,
                            highlightbackground=C_ACCENT2, highlightthickness=1)
        rfr_card.pack(fill="x", pady=2)
        rfr_inner = tk.Frame(rfr_card, bg=bg_card)
        rfr_inner.pack(fill="x", padx=8, pady=5)
        rfr_row = tk.Frame(rfr_inner, bg=bg_card)
        rfr_row.pack(fill="x")
        tk.Label(rfr_row, text="Sesi berikutnya:",
                 font=("Segoe UI", 8), bg=bg_card, fg=C_TEXT_DIM).pack(side="left")
        self._ml_rfr_score = tk.Label(rfr_row, text="-",
                                       font=("Segoe UI", 11, "bold"),
                                       bg=bg_card, fg=C_ACCENT2)
        self._ml_rfr_score.pack(side="right")

        _sep()
        _section_title("🔍", "ANOMALY DETECTION", C_PINK)
        iso_card = tk.Frame(body, bg=bg_card,
                            highlightbackground=C_PINK, highlightthickness=1)
        iso_card.pack(fill="x", pady=2)
        iso_inner = tk.Frame(iso_card, bg=bg_card)
        iso_inner.pack(fill="x", padx=8, pady=5)
        self._ml_iso_status = tk.Label(iso_inner, text="Menunggu…",
                                        font=("Segoe UI", 9, "bold"),
                                        bg=bg_card, fg=C_TEXT)
        self._ml_iso_status.pack(anchor="w")
        self._ml_iso_reason = tk.Label(iso_inner, text="",
                                        font=("Segoe UI", 7),
                                        bg=bg_card, fg=C_TEXT_DIM,
                                        wraplength=250, justify="left")
        self._ml_iso_reason.pack(anchor="w")

        _sep()
        _section_title("📊", "PERFORMANCE BAR", C_GOLD)
        bars_frame = tk.Frame(body, bg=bg_panel)
        bars_frame.pack(fill="x", pady=(2, 4))

        self._ml_skill_bars = {}
        bar_specs = [
            ("speed",       "Kecepatan",     C_ACCENT),
            ("accuracy",    "Akurasi",       C_ACCENT2),
            ("consistency", "Konsistensi",   C_PURPLE),
            ("independence","Kemandirian",   C_GOLD),
        ]
        for key, label, color in bar_specs:
            row = tk.Frame(bars_frame, bg=bg_panel)
            row.pack(fill="x", pady=2)
            tk.Label(row, text=label,
                     font=("Segoe UI", 8), bg=bg_panel, fg=C_TEXT_DIM,
                     width=13, anchor="w").pack(side="left")
            track = tk.Frame(row, bg=C_BORDER, height=7)
            track.pack(side="left", fill="x", expand=True)
            fill = tk.Frame(track, bg=color, height=7, width=4)
            fill.place(x=0, y=0)
            pct_lbl = tk.Label(row, text=" -%",
                                font=("Segoe UI", 8, "bold"),
                                bg=bg_panel, fg=color, width=5)
            pct_lbl.pack(side="right")
            self._ml_skill_bars[key] = (track, fill, pct_lbl, color)

        tk.Frame(pnl, height=1, bg=sep_col).pack(fill="x")
        foot = tk.Frame(pnl, bg=bg_panel)
        foot.pack(fill="x", padx=8, pady=4)
        sklearn_txt = "sklearn AKTIF" if SKLEARN_AVAILABLE else "sklearn tidak tersedia"
        sklearn_col = C_ACCENT2 if SKLEARN_AVAILABLE else C_TEXT_DIM
        tk.Label(foot, text=sklearn_txt,
                 font=("Segoe UI", 7), bg=bg_panel, fg=sklearn_col).pack(side="left")
        self._ml_last_update_lbl = tk.Label(foot, text="",
                                             font=("Segoe UI", 7),
                                             bg=bg_panel, fg=C_TEXT_DIM)
        self._ml_last_update_lbl.pack(side="right")

        self.after(120, self._schedule_ml_refresh)

    # Refresh loop
    # Menangani proses schedule ml refresh pada GameScreen sambil menjaga state internal tetap konsisten.
    def _schedule_ml_refresh(self):
        if not self._ml_panel_visible:
            return
        if self._ml_refresh_busy:
            return

        self._ml_refresh_busy = True
        self._ml_last_refresh = time.time()

        try:
            self._ml_refresh_btn.config(text="⏳")
        except Exception:
            pass

        snap = self._build_live_session()
        rem  = self._remaining_pct()

        # Menangani proses bg pada GameScreen sambil menjaga state internal tetap konsisten.
        def _bg():
            result = {}
            try:
                pt, conf, _ = self.ml.classify_player_confidence()
                result["knn_type"] = pt
                result["knn_conf"] = conf
            except Exception:
                result["knn_type"] = "Unknown"
                result["knn_conf"] = 0.0

            try:
                thresh = self.ml.compute_hint_threshold(
                    self.grid_size, self.difficulty, rem)
                result["hint_thresh"] = thresh
            except Exception:
                result["hint_thresh"] = None

            try:
                pred, avail = self.ml.predict_next_score()
                result["rfr_score"] = pred
                result["rfr_avail"] = avail
            except Exception:
                result["rfr_score"] = None
                result["rfr_avail"] = False

            try:
                anom, reason = self.ml.detect_anomaly()
                result["iso_status"] = anom
                result["iso_reason"] = reason
            except Exception:
                result["iso_status"] = "unknown"
                result["iso_reason"] = "Tidak dapat dianalisis"

            try:
                profile = self.ml.predict_stat_profile(snap)
                result["speed"]        = profile.get("speed_index",       0.0)
                result["accuracy"]     = profile.get("accuracy_index",    0.0)
                result["consistency"]  = profile.get("consistency_index", 0.0)
                result["independence"] = profile.get("independence_index",0.0)
            except Exception:
                feat = self.ml.extract_features()
                result["speed"]       = max(0, min(100, 100 - feat.get("avg_time_per_cell",10)*8))
                result["accuracy"]    = max(0, min(100, 100 - feat.get("error_rate",0)*250))
                result["consistency"] = feat.get("completion_rate",0)*100
                result["independence"]= max(0, min(100, (1-feat.get("hint_rate",0))*100))

            result["idle_now"] = time.time() - self.last_action
            return result

        # Menangani proses done pada GameScreen sambil menjaga state internal tetap konsisten.
        def _done(result):
            try:
                self._apply_ml_data(result)
            except Exception:
                pass
            self._ml_refresh_busy = False
            try:
                self._ml_refresh_btn.config(text="🔄")
            except Exception:
                pass
            if self._ml_panel_visible:
                try:
                    if self._ml_panel_job:
                        self.after_cancel(self._ml_panel_job)
                except Exception:
                    pass
                self._ml_panel_job = self.after(
                    1000, self._schedule_ml_refresh)

        # Menangani proses thread target pada GameScreen sambil menjaga state internal tetap konsisten.
        def _thread_target():
            data = _bg()
            try:
                self.after(0, lambda: _done(data))
            except Exception:
                self._ml_refresh_busy = False

        threading.Thread(target=_thread_target, daemon=True).start()

    # Terapkan hasil ke widget panel
    # Menangani proses apply ml data pada GameScreen sambil menjaga state internal tetap konsisten.
    def _apply_ml_data(self, d):
        if not self._ml_panel_visible:
            return
        if not self._ml_panel_frame or not self._ml_panel_frame.winfo_exists():
            return
        if not hasattr(self, "_ml_knn_type") or self._ml_knn_type is None:
            return

        _TYPE_EMOJIS = {
            "Speedrunner":  "⚡",
            "Careful":      "🛡",
            "Learner":      "📚",
            "Struggling":   "💪",
            "Inconsistent": "🎲",
        }
        _TYPE_COLORS = {
            "Speedrunner":  C_ACCENT,
            "Careful":      C_ACCENT2,
            "Learner":      C_PURPLE,
            "Struggling":   C_ERROR,
            "Inconsistent": C_WARN,
        }

        pt   = d.get("knn_type", "?")
        conf = d.get("knn_conf", 0.0)
        emoji = _TYPE_EMOJIS.get(pt, "🎮")
        color = _TYPE_COLORS.get(pt, C_ACCENT)
        try:
            self._ml_knn_emoji.config(text=emoji, fg=color)
            self._ml_knn_type.config(text=pt, fg=color)
            conf_txt = (f"Conf: {conf:.0f}%" if conf > 0
                        else "Rule-based")
            self._ml_knn_conf.config(text=conf_txt)
            bar_w = 4
            try:
                self._ml_knn_bar.update_idletasks()
                track_w = self._ml_knn_bar.master.winfo_width()
                if track_w > 8:
                    bar_w = max(4, int(track_w * conf / 100))
            except Exception:
                pass
            self._ml_knn_bar.config(bg=color, width=bar_w)
        except Exception:
            pass

        try:
            thresh = d.get("hint_thresh")
            idle   = d.get("idle_now", 0.0)
            if thresh is not None:
                self._ml_hint_thresh.config(
                    text=f"{thresh:.0f} det",
                    fg=C_WARN if idle < thresh else C_ERROR)
            else:
                self._ml_hint_thresh.config(text="-", fg=C_TEXT_DIM)
            idle_col = (C_ERROR if thresh and idle > thresh
                        else C_TEXT)
            self._ml_hint_idle.config(
                text=f"{idle:.0f} det", fg=idle_col)
        except Exception:
            pass

        try:
            if d.get("rfr_avail") and d.get("rfr_score") is not None:
                score_txt = f"{d['rfr_score']:,} poin"
                self._ml_rfr_score.config(text=score_txt, fg=C_ACCENT2)
            elif not SKLEARN_AVAILABLE:
                self._ml_rfr_score.config(text="sklearn off", fg=C_TEXT_DIM)
            else:
                self._ml_rfr_score.config(
                    text=f"Butuh ≥3 sesi  ({len(self.ml.sessions)} ada)",
                    fg=C_TEXT_DIM)
        except Exception:
            pass

        try:
            status = d.get("iso_status", "unknown")
            reason = d.get("iso_reason", "")
            status_map = {
                "normal":  ("✅  Normal",      C_ACCENT2),
                "anomaly": ("⚠  Anomali",     C_ERROR),
                "unknown": ("⬜  Belum cukup", C_TEXT_DIM),
            }
            stxt, scol = status_map.get(status, ("-", C_TEXT_DIM))
            self._ml_iso_status.config(text=stxt, fg=scol)
            self._ml_iso_reason.config(text=reason)
        except Exception:
            pass

        bar_keys = {
            "speed":        d.get("speed",       0.0),
            "accuracy":     d.get("accuracy",    0.0),
            "consistency":  d.get("consistency", 0.0),
            "independence": d.get("independence",0.0),
        }
        for key, pct in bar_keys.items():
            try:
                track, fill, pct_lbl, color = self._ml_skill_bars[key]
                pct_lbl.config(text=f"{pct:4.0f}%")
                track.update_idletasks()
                tw = track.winfo_width()
                if tw > 8:
                    target_w = max(4, int(tw * pct / 100))
                    fill.config(width=target_w, bg=color)
            except Exception:
                pass

        try:
            ts = time.strftime("%H:%M:%S")
            self._ml_last_update_lbl.config(text=f"Update: {ts}")
        except Exception:
            pass

# SCREEN: PLAYER SELECT  (Ganti Pemain)
# SCREEN: GANTI PEMAIN  (2-panel: list + detail)
class PlayerSelectScreen(tk.Frame):
    """
    Layar Ganti Pemain - dua panel:
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

    # Menginisialisasi objek PlayerSelectScreen dan menyiapkan state awal, referensi penting, serta elemen yang dibutuhkan sebelum layar dipakai.
    def __init__(self, master, current_user, on_select, on_new_player,
                 initial_selected=None, on_highlight=None, on_back_to_login=None,
                 on_back_to_game=None):
        super().__init__(master, bg=C_BG)
        self.current_user     = current_user
        self.on_select        = on_select
        self.on_new_player    = on_new_player
        self.on_highlight     = on_highlight
        self.on_back_to_login = on_back_to_login
        self.on_back_to_game  = on_back_to_game
        self.data          = load_data()
        self._selected     = initial_selected if initial_selected is not None else current_user
        self._row_widgets  = {}
        self._detail_frame = None
        self._build()

    @staticmethod
    # Menangani proses fast classify pada PlayerSelectScreen sambil menjaga state internal tetap konsisten.
    def _fast_classify(sessions):
        if not sessions:
            return "Learner"
        n = len(sessions)
        moves_total = sum(max(s.get("moves", 1), 1) for s in sessions)
        tpc = sum(s.get("total_time", 0) / max(s.get("empty_cells", 1), 1)
                  for s in sessions) / n
        er  = sum(s.get("errors", 0) for s in sessions) / max(moves_total, 1)
        hr  = sum(s.get("hints_used", 0) for s in sessions) / max(moves_total, 1)
        cr  = sum(1 for s in sessions if s.get("completed")) / n
        nmr = sum(s.get("near_miss", 0) / max(s.get("errors", 1), 1)
                  for s in sessions) / n
        gur = sum(s.get("guessing", 0) / max(s.get("errors", 1), 1)
                  for s in sessions) / n
        return _ml_player_type_from_metrics(tpc, er, hr, cr, nmr, gur)

    # Stat computation
    # Menangani proses get stats pada PlayerSelectScreen sambil menjaga state internal tetap konsisten.
    def _get_stats(self, username):
        sessions  = self.data["players"].get(username, {}).get("sessions", [])
        completed = [s for s in sessions if s.get("completed", False)]
        scores = [
            s.get("score") or calculate_score(
                s.get("difficulty", "Normal"),
                s.get("total_time", 1),
                s.get("empty_cells", max(s.get("moves", 1), 1)),
                s.get("errors", 0), s.get("hints_used", 0),
                s.get("completed", False),
                s.get("near_miss", 0), s.get("guessing", 0))
            for s in sessions
        ]
        n = len(sessions)
        # Menangani proses avg pada PlayerSelectScreen sambil menjaga state internal tetap konsisten.
        def _avg(fn): return sum(fn(s) for s in sessions) / n if n else 0.0
        moves_total = sum(max(s.get("moves", 1), 1) for s in sessions)
        tpc  = _avg(lambda s: s.get("total_time", 0) / max(s.get("empty_cells", 1), 1))
        er   = sum(s.get("errors", 0) for s in sessions) / max(moves_total, 1)
        hr   = sum(s.get("hints_used", 0) for s in sessions) / max(moves_total, 1)
        cr   = len(completed) / n if n else 0.0
        nmr  = _avg(lambda s: s.get("near_miss", 0) / max(s.get("errors", 1), 1))
        gur  = _avg(lambda s: s.get("guessing", 0) / max(s.get("errors", 1), 1))

        p_type = self._fast_classify(sessions)
        if tpc < 8 and er < 0.08 and cr > 0.85:
            rec = "Hard"
        elif tpc > 25 or er > 0.25:
            rec = "Easy"
        else:
            rec = "Normal"

        feat = {
            "avg_time_per_cell": tpc, "error_rate": er,
            "hint_rate": hr,         "completion_rate": cr,
            "near_miss_rate": nmr,   "guessing_rate": gur,
            "sessions_count": n,
        }
        return {
            "sessions":        sessions,
            "n_sess":          n,
            "n_done":          len(completed),
            "best_score":      max(scores, default=0),
            "completion_rate": cr,
            "error_rate":      er,
            "hint_rate":       hr,
            "avg_time":        tpc,
            "total_playtime":  sum(s.get("total_time", 0) for s in sessions),
            "player_type":     p_type,
            "type_color":      self._TYPE_COLORS.get(p_type, C_ACCENT),
            "type_emoji":      self._TYPE_EMOJIS.get(p_type, "🎮"),
            "recommended":     rec,
            "feat":            feat,
        }

    # Menangani proses fmt time pada PlayerSelectScreen sambil menjaga state internal tetap konsisten.
    def _fmt_time(self, s):
        s = int(s)
        if s >= 3600: return f"{s//3600}j {(s%3600)//60}m"
        if s >= 60:   return f"{s//60}m {s%60}s"
        return f"{s}s"

    # Menangani proses initials pada PlayerSelectScreen sambil menjaga state internal tetap konsisten.
    def _initials(self, name):
        p = name.strip().split()
        if len(p) >= 2: return (p[0][0]+p[1][0]).upper()
        return name[:2].upper() if len(name)>=2 else name[0].upper()

    # gambar avatar lingkaran dengan inisial pemain
    @staticmethod
    # Menggambar avatar pada PlayerSelectScreen sesuai state yang sedang aktif.
    def _draw_avatar(canvas, size, color, initials):
        pad = 3
        canvas.create_oval(pad, pad, size-pad, size-pad,
                           outline=color, width=2, fill="")
        try:
            r_ = int(int(color[1:3],16)*0.22)
            g_ = int(int(color[3:5],16)*0.22)
            b_ = int(int(color[5:7],16)*0.22)
            inner = f"#{r_:02x}{g_:02x}{b_:02x}"
        except Exception:
            inner = C_BG
        canvas.create_oval(pad+3, pad+3, size-pad-3, size-pad-3,
                           fill=inner, outline="")
        canvas.create_text(size//2, size//2, text=initials,
                           fill=color,
                           font=("Segoe UI", int(size*0.38), "bold"),
                           anchor="center")

    # gambar progress bar skill dengan animasi
    # Menangani proses skill bar pada PlayerSelectScreen sambil menjaga state internal tetap konsisten.
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
        # Menangani proses grow pada PlayerSelectScreen sambil menjaga state internal tetap konsisten.
        def _grow():
            track.update_idletasks()
            target = int(track.winfo_width() * pct / 100)
            # Menangani proses step pada PlayerSelectScreen sambil menjaga state internal tetap konsisten.
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

    # Main build
    # Membangun bagian antarmuka pada PlayerSelectScreen dan menyiapkan widget supaya state tampilan tetap konsisten.
    def _build(self):
        self.place(relx=0, rely=0, relwidth=1, relheight=1)

        abg = AnimatedBG(self, bg=C_BG, highlightthickness=0)
        abg.place(relx=0, rely=0, relwidth=1, relheight=1)

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
        title_text = "PLAYER LOGIN" if not self.current_user else "SWITCH PLAYER"
        tk.Label(title_row, text=title_text,
                 font=("Segoe UI", 22, "bold"), bg=C_SURFACE, fg=C_TEXT).pack(side="left")

        subtitle = ("Select any player to log in and view their statistics"
                    if not self.current_user
                    else f"Currently logged in as  @{self.current_user}  ·  click a player to view their profile")
        tk.Label(hdr_inner, text=subtitle,
                 font=("Segoe UI", 9), bg=C_SURFACE, fg=C_TEXT_DIM).pack(pady=(4,0))

        body = tk.Frame(self, bg=C_BG)
        body.pack(fill="both", expand=True)

        left = tk.Frame(body, bg=C_SIDEBAR, width=360)
        left.pack(side="left", fill="y")
        left.pack_propagate(False)

        lhdr = tk.Frame(left, bg=C_SIDEBAR, pady=12)
        lhdr.pack(fill="x", padx=16)
        tk.Label(lhdr, text="PLAYER REGISTERED",
                 font=("Segoe UI", 9, "bold"),
                 bg=C_SIDEBAR, fg=C_TEXT_DIM).pack(side="left")
        all_players = self.data.get("players", {})
        tk.Label(lhdr, text=str(len(all_players)),
                 font=("Segoe UI", 9, "bold"),
                 bg=C_SIDEBAR, fg=C_PURPLE).pack(side="right")

        tk.Frame(left, height=1, bg=C_BORDER).pack(fill="x")

        sc = tk.Canvas(left, bg=C_SIDEBAR, highlightthickness=0)
        sc.pack(side="left", fill="both", expand=True)

        sb = tk.Scrollbar(left, orient="vertical", command=sc.yview)
        sc.configure(yscrollcommand=sb.set)

        list_frame = tk.Frame(sc, bg=C_SIDEBAR)
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
                     font=("Segoe UI",11), bg=C_SIDEBAR, fg=C_TEXT_DIM).pack(pady=30)
        else:
            # Menghasilkan kunci pengurutan untuk data pada PlayerSelectScreen agar daftar tampil dengan urutan yang stabil.
            def sort_key(n):
                nsess = len(all_players[n].get("sessions",[]))
                return (0 if n==self.current_user else 1, -nsess)
            for name in sorted(all_players, key=sort_key):
                self._player_row(list_frame, name, sc)
            if len(all_players) > 6:
                sb.pack(side="right", fill="y")

        tk.Frame(body, width=1, bg=C_BORDER).pack(side="left", fill="y")

        self._right_panel = tk.Frame(body, bg=C_BG)
        self._right_panel.pack(side="left", fill="both", expand=True)
        self._refresh_detail(self._selected)

        foot = tk.Frame(self, bg=C_SURFACE)
        foot.pack(fill="x", side="bottom")

        foot_bar = tk.Canvas(foot, height=3, bg=C_SURFACE, highlightthickness=0)
        foot_bar.pack(fill="x")
        foot_bar.after(120, lambda: draw_gradient_bar(foot_bar, height=3))

        foot_inner = tk.Frame(foot, bg=C_SURFACE)
        foot_inner.pack(pady=12)

        tk.Button(foot_inner,
                  text="➕  NEW PLAYER",
                  font=("Segoe UI", 10, "bold"),
                  bg=C_PURPLE, fg=C_WHITE,
                  activebackground="#9B5FEF", activeforeground=C_WHITE,
                  relief="flat", cursor="hand2", pady=10, padx=24,
                  command=lambda: (_play_sfx(_SFX_SELECT), self.on_new_player())
                  ).pack(side="left", padx=(0,10))

        if self.current_user:
            _back_cmd = (self.on_back_to_game
                         if callable(self.on_back_to_game)
                         else lambda: self.on_select(self.current_user))
            tk.Button(foot_inner,
                      text="←  Back",
                      font=("Segoe UI", 10),
                      bg=C_SURFACE2, fg=C_TEXT_DIM,
                      activebackground=C_BORDER, activeforeground=C_TEXT,
                      relief="flat", cursor="hand2", pady=10, padx=18,
                      command=lambda: (_play_sfx(_SFX_CLICK), _back_cmd())
                      ).pack(side="left")
        elif callable(self.on_back_to_login):
            tk.Button(foot_inner,
                      text="←  BACK",
                      font=("Segoe UI", 10),
                      bg=C_SURFACE2, fg=C_TEXT_DIM,
                      activebackground=C_BORDER, activeforeground=C_TEXT,
                      relief="flat", cursor="hand2", pady=10, padx=18,
                      command=lambda: (_play_sfx(_SFX_CLICK), self.on_back_to_login())
                      ).pack(side="left")

    # Left panel: compact player row
    # Menangani proses player row pada PlayerSelectScreen sambil menjaga state internal tetap konsisten.
    def _player_row(self, parent, name, scroll_canvas):
        sessions   = self.data["players"][name].get("sessions", [])
        n_sess     = len(sessions)
        p_type     = self._fast_classify(sessions)
        type_color = self._TYPE_COLORS.get(p_type, C_ACCENT)
        type_emoji = self._TYPE_EMOJIS.get(p_type, "🎮")
        is_me      = (name == self.current_user)
        is_sel     = (name == self._selected)

        BG_NORMAL   = C_SIDEBAR
        BG_ACTIVE   = C_SIDEBAR_ACTIVE
        BG_SELECTED = C_SIDEBAR_SELECTED

        row_bg = BG_SELECTED if is_sel else (BG_ACTIVE if is_me else BG_NORMAL)

        row = tk.Frame(parent, bg=row_bg, cursor="hand2")
        row.pack(fill="x")

        strip = tk.Frame(row, width=3, bg=C_PURPLE if is_me else (type_color if is_sel else C_SIDEBAR))
        strip.pack(side="left", fill="y")

        inner = tk.Frame(row, bg=row_bg, pady=10, padx=12)
        inner.pack(side="left", fill="x", expand=True)

        AV = 38
        av = tk.Canvas(inner, width=AV, height=AV,
                       bg=row_bg, highlightthickness=0)
        av.pack(side="left")
        av.after(80, lambda c=av, col=type_color, s=AV, n=name:
                 self._draw_avatar(c, s, col, self._initials(n)))

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
        tk.Label(sub, text=f"  ·  {n_sess} session",
                 font=("Segoe UI", 8),
                 bg=row_bg, fg=C_TEXT_DIM).pack(side="left")

        tk.Label(inner, text="›",
                 font=("Segoe UI", 16),
                 bg=row_bg, fg=type_color if is_sel else C_BORDER).pack(side="right")

        tk.Frame(parent, height=1, bg=C_SIDEBAR_SEP).pack(fill="x")

        self._row_widgets[name] = (row, strip, inner, text_col, name_row, sub, av)

        for w in [row, inner, text_col, name_row, sub, av]:
            w.bind("<MouseWheel>",
                   lambda e: scroll_canvas.yview_scroll(int(-1*(e.delta/120)),"units"))

        # Menangani interaksi click pada PlayerSelectScreen agar respons UI tetap konsisten.
        def _click(_, n=name):
            self._selected = n
            if callable(self.on_highlight):
                self.on_highlight(n)
            self._refresh_row_highlights()
            self._refresh_detail(n)

        for w in [row, inner, text_col, av, name_row, sub]:
            w.bind("<Button-1>", _click)

        # Menangani interaksi enter pada PlayerSelectScreen agar respons UI tetap konsisten.
        def _enter(_):
            if name != self._selected:
                row.config(bg=C_SIDEBAR_HOVER)
                inner.config(bg=C_SIDEBAR_HOVER)
        # Menangani interaksi leave pada PlayerSelectScreen agar respons UI tetap konsisten.
        def _leave(_):
            bg_ = BG_SELECTED if self._selected==name else (BG_ACTIVE if is_me else BG_NORMAL)
            row.config(bg=bg_)
            inner.config(bg=bg_)

        for w in [row, inner]:
            w.bind("<Enter>", _enter)
            w.bind("<Leave>", _leave)

    # Highlight selected row in list
    # Menyegarkan row highlights pada PlayerSelectScreen setelah data atau pilihan pengguna berubah.
    def _refresh_row_highlights(self):
        for name, widgets in self._row_widgets.items():
            row, strip, inner, text_col, name_row, sub, av = widgets
            is_sel = (name == self._selected)
            is_me  = (name == self.current_user)
            sessions = self.data["players"][name].get("sessions", [])
            p_type = self._fast_classify(sessions)
            tc = self._TYPE_COLORS.get(p_type, C_ACCENT)
            bg_ = C_SIDEBAR_SELECTED if is_sel else (C_SIDEBAR_ACTIVE if is_me else C_SIDEBAR)
            row.config(bg=bg_)
            inner.config(bg=bg_)
            strip.config(bg=C_PURPLE if is_me else (tc if is_sel else C_SIDEBAR))

    # Scroll helper - bind wheel to every descendant
    # Menangani proses bind scroll all pada PlayerSelectScreen sambil menjaga state internal tetap konsisten.
    def _bind_scroll_all(self, widget, scroll_fn):
        try:
            widget.bind("<MouseWheel>",
                        lambda e=None: scroll_fn(int(-1 * (e.delta / 120)), "units") if e else None,
                        add="+")
            widget.bind("<Button-4>",  lambda e=None: scroll_fn(-1, "units"), add="+")
            widget.bind("<Button-5>",  lambda e=None: scroll_fn(+1, "units"), add="+")
            for child in widget.winfo_children():
                self._bind_scroll_all(child, scroll_fn)
        except Exception:
            pass

    # Right panel: full player detail
    # Menyegarkan detail pada PlayerSelectScreen setelah data atau pilihan pengguna berubah.
    def _refresh_detail(self, username):
        for w in self._right_panel.winfo_children():
            w.destroy()

        if username not in self.data.get("players", {}):
            tk.Label(self._right_panel,
                     text="Select a player from the list to view their profile",
                     font=("Segoe UI", 12), bg=C_BG, fg=C_TEXT_DIM).pack(expand=True)
            return

        skel = tk.Frame(self._right_panel, bg=C_BG)
        skel.pack(fill="both", expand=True)
        tk.Label(skel, text="⏳  Memuat profil...",
                 font=("Segoe UI", 13), bg=C_BG, fg=C_TEXT_DIM
                 ).pack(expand=True)

        req_id = object()
        self._pending_req = req_id

        # Menangani proses compute pada PlayerSelectScreen sambil menjaga state internal tetap konsisten.
        def _compute():
            try:
                result = self._get_stats(username)
            except Exception:
                result = None
            try:
                self._right_panel.after(0, lambda: _render(result))
            except Exception:
                pass

        # Menangani proses render pada PlayerSelectScreen sambil menjaga state internal tetap konsisten.
        def _render(st):
            if getattr(self, "_pending_req", None) is not req_id:
                return
            for w in self._right_panel.winfo_children():
                try: w.destroy()
                except Exception: pass
            if st is None:
                tk.Label(self._right_panel, text="⚠  Gagal memuat profil.",
                         font=("Segoe UI", 12), bg=C_BG, fg=C_ERROR).pack(expand=True)
                return
            self._build_detail_ui(st, username)

        threading.Thread(target=_compute, daemon=True).start()

    # Membangun detail ui pada PlayerSelectScreen dan menyiapkan widget supaya state tampilan tetap konsisten.
    def _build_detail_ui(self, st, username):
        tc       = st["type_color"]
        is_me    = (username == self.current_user)
        PANEL_BG = C_BG

        if st["n_sess"] == 0:
            empty = tk.Frame(self._right_panel, bg=PANEL_BG)
            empty.pack(fill="both", expand=True)

            inner = tk.Frame(empty, bg=C_SURFACE,
                             highlightbackground=C_BORDER, highlightthickness=1)
            inner.place(relx=0.5, rely=0.5, anchor="center", width=380, height=290)

            tk.Frame(inner, bg=tc, height=4).pack(fill="x")

            AV = 52
            av_cv = tk.Canvas(inner, width=AV, height=AV,
                              bg=C_SURFACE, highlightthickness=0)
            av_cv.pack(pady=(22, 0))
            av_cv.after(80, lambda c=av_cv, col=tc, s=AV:
                        self._draw_avatar(c, s, col, self._initials(username)))

            tk.Label(inner, text=f"@{username}",
                     font=("Segoe UI", 14, "bold"),
                     bg=C_SURFACE, fg=C_TEXT).pack(pady=(8, 0))

            tk.Label(inner, text="🎮  Pemain Baru",
                     font=("Segoe UI", 10, "bold"),
                     bg=C_SURFACE, fg=tc).pack(pady=(4, 0))

            tk.Frame(inner, height=1, bg=C_BORDER).pack(fill="x", padx=24, pady=14)

            tk.Label(inner,
                     text="Belum ada data permainan.\nMainkan satu sesi untuk melihat statistik.",
                     font=("Segoe UI", 10),
                     bg=C_SURFACE, fg=C_TEXT_DIM,
                     justify="center").pack()

            tk.Button(inner,
                      text="▶  Mulai Bermain",
                      font=("Segoe UI", 10, "bold"),
                      bg=C_ACCENT, fg=C_BG,
                      activebackground=C_ACCENT2, activeforeground=C_BG,
                      relief="flat", cursor="hand2", pady=8,
                      command=lambda: (_play_sfx(_SFX_SELECT), self.on_select(username))
                      ).pack(fill="x", padx=24, pady=(14, 0))
            return

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

        sc.bind("<Configure>",
                lambda e=None: sc.itemconfig(wid, width=e.width) if e else None)
        detail.bind("<Configure>",
                    lambda e=None: sc.configure(scrollregion=sc.bbox("all")))

        # Menangani proses scroll pada PlayerSelectScreen sambil menjaga state internal tetap konsisten.
        def _scroll(amount, unit):
            try:
                sc.yview_scroll(amount, unit)
            except Exception:
                pass

        sc.bind("<MouseWheel>",
                lambda e=None: _scroll(int(-1 * (e.delta / 120)), "units") if e else None)
        sc.bind("<Button-4>",  lambda e=None: _scroll(-1, "units"))
        sc.bind("<Button-5>",  lambda e=None: _scroll(+1, "units"))

        # Menangani proses bind all later pada PlayerSelectScreen sambil menjaga state internal tetap konsisten.
        def _bind_all_later():
            try:
                self._bind_scroll_all(detail, _scroll)
            except Exception:
                pass
        detail.after(0, _bind_all_later)

        pad = tk.Frame(detail, bg=PANEL_BG)
        pad.pack(fill="x", padx=32, pady=20)

        hero = tk.Frame(pad, bg=C_SURFACE,
                        highlightbackground=tc, highlightthickness=2)
        hero.pack(fill="x", pady=(0,18))

        tk.Frame(hero, bg=tc, height=5).pack(fill="x")

        hero_inner = tk.Frame(hero, bg=C_SURFACE)
        hero_inner.pack(fill="x", padx=20, pady=16)

        AV = 72
        av_cv = tk.Canvas(hero_inner, width=AV, height=AV,
                          bg=C_SURFACE, highlightthickness=0)
        av_cv.pack(side="left", anchor="n", pady=(2,0))
        av_cv.after(80, lambda c=av_cv, col=tc, s=AV:
                    self._draw_avatar(c, s, col, self._initials(username)))

        _btn_fg        = C_BG
        try:
            _hr = min(255, int(tc[1:3], 16) + 35)
            _hg = min(255, int(tc[3:5], 16) + 35)
            _hb = min(255, int(tc[5:7], 16) + 35)
            _btn_hover_bg = f"#{_hr:02x}{_hg:02x}{_hb:02x}"
        except Exception:
            _btn_hover_bg = C_BORDER
        _btn_hover_fg  = C_BG if _CURRENT_THEME_NAME == "dark" else "#1C2330"

        hero_action = tk.Frame(hero_inner, bg=C_SURFACE)
        hero_action.pack(side="right", anchor="center", padx=(16, 0), pady=0)

        if is_me:
            _cont_bg  = C_SURFACE2
            _cont_fg  = C_TEXT
            _cont_hov = C_BORDER
            _cont_btn = tk.Button(
                hero_action,
                text="↩  LANJUTKAN",
                font=("Segoe UI", 10, "bold"),
                bg=_cont_bg, fg=_cont_fg,
                activebackground=_cont_hov, activeforeground=C_TEXT,
                relief="flat", cursor="hand2",
                padx=18, pady=10,
                command=lambda: self.on_select(username),
            )
            _cont_btn.pack()
            tk.Label(hero_action,
                     text="● Akun Aktif",
                     font=("Segoe UI", 8, "bold"),
                     bg=C_SURFACE, fg="#7EE787").pack(pady=(5, 0))
        else:
            _login_label = (
                f"▶  Login as @{username}"
                if not self.current_user
                else f"⇄  Switch to @{username}"
            )
            _login_btn = tk.Button(
                hero_action,
                text=_login_label,
                font=("Segoe UI", 10, "bold"),
                bg=tc, fg=_btn_fg,
                activebackground=_btn_hover_bg,
                activeforeground=_btn_hover_fg,
                relief="flat", cursor="hand2",
                padx=18, pady=10,
                command=lambda u=username: self.on_select(u),
            )
            _login_btn.pack()
            _hint_text = (
                "Klik untuk mulai bermain"
                if not self.current_user
                else f"Sesi @{self.current_user} akan dihentikan"
            )
            tk.Label(hero_action,
                     text=_hint_text,
                     font=("Segoe UI", 8),
                     bg=C_SURFACE,
                     fg=C_TEXT_DIM,
                     wraplength=160,
                     justify="center").pack(pady=(6, 0))

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
        try:
            rr = int(tc[1:3], 16)
            gg = int(tc[3:5], 16)
            bb = int(tc[5:7], 16)
            if _CURRENT_THEME_NAME == "light":
                rb = int(rr * 0.20 + 255 * 0.80)
                gb = int(gg * 0.20 + 255 * 0.80)
                bb2 = int(bb * 0.20 + 255 * 0.80)
            else:
                rb = int(rr * 0.15)
                gb = int(gg * 0.15)
                bb2 = int(bb * 0.15)
            badge_tint = f"#{rb:02x}{gb:02x}{bb2:02x}"
        except Exception:
            badge_tint = "#E8E8F0" if _CURRENT_THEME_NAME == "light" else "#1A1A2E"
        badge_bg = tk.Frame(type_r, bg=badge_tint)
        badge_bg.pack(side="left")
        tk.Label(badge_bg,
                 text=f"  {st['type_emoji']}  {st['player_type']}  ",
                 font=("Segoe UI", 10, "bold"),
                 bg=badge_tint, fg=tc, pady=3, padx=2).pack()

        rec_col = {"Easy":"#7EE787","Normal":"#58A6FF","Hard":"#FF7B7B"}.get(st["recommended"],C_ACCENT)
        rec_r = tk.Frame(hero_text, bg=C_SURFACE)
        rec_r.pack(anchor="w", pady=(6,0))
        tk.Label(rec_r, text="AI Rekomendasikan: ",
                 font=("Segoe UI", 9), bg=C_SURFACE, fg=C_TEXT_DIM).pack(side="left")
        tk.Label(rec_r, text=f"  {st['recommended']}  ",
                 font=("Segoe UI", 9, "bold"),
                 bg=rec_col, fg=C_BG, padx=4, pady=2).pack(side="left")

        tk.Label(rec_r, text=f"  ·  Total waktu: {self._fmt_time(st['total_playtime'])}",
                 font=("Segoe UI", 9),
                 bg=C_SURFACE, fg=C_TEXT_DIM).pack(side="left")

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
            ("Kemandirian",   max(0, min(100, (1 - feats["hint_rate"] - feats.get("auto_rate", 0) * 0.4) * 100)), "#BC8CFF"),
        ]
        for lbl, pct, col in bars:
            self._skill_bar(sk_body, lbl, pct, col, C_SURFACE)

        recent = st["sessions"][-8:][::-1]
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
                tk.Label(srow, text=f" {s.get('difficulty','?')} ",
                         font=("Segoe UI", 8, "bold"),
                         bg=dc, fg=C_BG, padx=3).pack(side="left")
                gs = s.get("grid_size",3)
                tk.Label(srow, text=f"  {gs*gs}×{gs*gs}",
                         font=("Segoe UI", 9), bg=C_SURFACE, fg=C_TEXT_DIM).pack(side="left")
                done = s.get("completed", False)
                tk.Label(srow,
                         text="  ✅ Selesai" if done else "  ⏸ Berhenti",
                         font=("Segoe UI", 9),
                         bg=C_SURFACE, fg="#7EE787" if done else C_TEXT_DIM).pack(side="left")
                t = int(s.get("total_time", 0))
                tk.Label(srow, text=f"  ⏱ {t//60:02}:{t%60:02}",
                         font=("Segoe UI", 9),
                         bg=C_SURFACE, fg=C_TEXT_DIM).pack(side="left")
                err = s.get("errors", 0)
                tk.Label(srow, text=f"  ❌ {err}",
                         font=("Segoe UI", 9),
                         bg=C_SURFACE, fg=C_ERROR if err else C_TEXT_DIM).pack(side="left")
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

        if not is_me:
            cta_outer = tk.Frame(pad, bg=C_SURFACE,
                                 highlightbackground=C_BORDER, highlightthickness=1)
            cta_outer.pack(fill="x", pady=(0, 8))

            tk.Frame(cta_outer, bg=C_WARN if self.current_user else tc, height=3).pack(fill="x")

            cta_inner = tk.Frame(cta_outer, bg=C_SURFACE)
            cta_inner.pack(fill="x", padx=18, pady=12)

            info_row = tk.Frame(cta_inner, bg=C_SURFACE)
            info_row.pack(fill="x")

            tk.Label(info_row, text="ℹ",
                     font=("Segoe UI", 13), bg=C_SURFACE,
                     fg=C_WARN if self.current_user else C_ACCENT).pack(side="left", anchor="n", pady=2)

            msg_col = tk.Frame(info_row, bg=C_SURFACE)
            msg_col.pack(side="left", padx=(8, 0))

            if self.current_user:
                tk.Label(msg_col,
                         text=f"Mengganti sesi aktif dari @{self.current_user}",
                         font=("Segoe UI", 9, "bold"),
                         bg=C_SURFACE, fg=C_TEXT, anchor="w").pack(anchor="w")
                tk.Label(msg_col,
                         text="Progres sesi yang sedang berjalan akan dihentikan setelah konfirmasi.",
                         font=("Segoe UI", 8),
                         bg=C_SURFACE, fg=C_TEXT_DIM,
                         wraplength=340, justify="left", anchor="w").pack(anchor="w", pady=(2,0))
            else:
                tk.Label(msg_col,
                         text=f"Masuk sebagai @{username}",
                         font=("Segoe UI", 9, "bold"),
                         bg=C_SURFACE, fg=C_TEXT, anchor="w").pack(anchor="w")
                tk.Label(msg_col,
                         text="Klik tombol Login di kanan atas kartu profil untuk mulai bermain.",
                         font=("Segoe UI", 8),
                         bg=C_SURFACE, fg=C_TEXT_DIM,
                         wraplength=340, justify="left", anchor="w").pack(anchor="w", pady=(2,0))


# =============================================================================
# BAGIAN 8: MAIN GAME CONTROLLER
# =============================================================================

class SudokuApp:
    """
    SudokuApp - Kontroler utama aplikasi Sudoku AI.

    Deskripsi:
        Mengelola siklus hidup aplikasi: inisialisasi tkinter fullscreen,
        navigasi antar layar, musik latar, toggle tema, dan ML engine.

    Atribut:
        root (tk.Tk): Jendela utama (fullscreen).
        username (str | None): Username pemain aktif.
        _ml (PlayerMLEngine | None): ML engine pemain aktif.
        _music_on (bool): Status musik latar.

    Contoh Penggunaan:
        app = SudokuApp()
        app.run()
    """
    # Menginisialisasi objek SudokuApp dan menyiapkan state awal, referensi penting, serta elemen yang dibutuhkan sebelum layar dipakai.
    def __init__(self):
        global _APP_INSTANCE
        _APP_INSTANCE   = self
        self._active_overlay = None
        startup_check()
        self.root = tk.Tk()
        self.root.title("Sudoku AI - ML Intelligence System")
        self.root.attributes("-fullscreen", True)
        self.root.configure(bg=C_BG)

        self.logo_image  = None
        self.logo_small  = None
        self.root._logo_tk = None
        _logo_path = IMAGE_LOGO
        if os.path.exists(_logo_path):
            try:
                from PIL import Image, ImageTk
                _pil = Image.open(_logo_path)

                _dpi  = self.root.winfo_fpixels('1i')
                _scale = max(1.0, _dpi / 96.0)
                _logo_sz  = int(32 * _scale)
                _header_sz = int(48 * _scale)

                _icon = _pil.resize((_logo_sz, _logo_sz), Image.LANCZOS)
                self.logo_image = ImageTk.PhotoImage(_icon)
                self.root.iconphoto(True, self.logo_image)

                _hdr = _pil.resize((_header_sz, _header_sz), Image.LANCZOS)
                self.logo_small = ImageTk.PhotoImage(_hdr)
                self.root._logo_tk = self.logo_small
            except ImportError:
                try:
                    raw = tk.PhotoImage(file=_logo_path)
                    factor = max(1, raw.width() // 48)
                    self.logo_image = raw.subsample(factor, factor)
                    self.root.iconphoto(True, self.logo_image)
                    self.logo_small = self.logo_image
                    self.root._logo_tk = self.logo_small
                except Exception:
                    pass
            except Exception:
                pass

        self.username   = None
        self.grid_size  = 3
        self.difficulty = "Normal"
        self.screen     = None

        self._ingame_saved_state      = None
        self._ingame_saved_username   = None
        self._ingame_saved_grid_size  = None
        self._ingame_saved_difficulty = None

        self.root.bind("<<PlayAgain>>",        self._play_again)
        self.root.bind("<<ExitGame>>",         self._exit)
        self.root.bind("<<ChangePlayer>>",     self._show_player_select)
        self.root.bind("<<Logout>>",           self._logout)
        self.root.bind("<<BackToGrid>>",       lambda _: self._back_to_grid_select())

        self.root.bind_all("<F5>",             self._kiosk_reset)
        self.root.bind_all("<Control-Shift-r>", self._kiosk_reset)
        self.root.bind_all("<Control-Shift-R>", self._kiosk_reset)

        self._last_screen_change = 0.0
        self.root.bind_all("<Escape>", self._on_esc_global)

        self._music_ready  = False
        self._music_on     = False
        self._music_paused = False
        self._init_music()

        self.root.bind_all("<m>", self._toggle_music)
        self.root.bind_all("<M>", self._toggle_music)

        self._music_btn = tk.Canvas(
            self.root, width=44, height=44,
            highlightthickness=0, bd=0, cursor="hand2",
        )
        self._music_btn.place(relx=1.0, rely=1.0, anchor="se", x=-14, y=-14)
        self._music_btn.bind("<Button-1>", self._toggle_music)
        self._music_btn.bind("<Enter>",    self._on_music_btn_enter)
        self._music_btn.bind("<Leave>",    self._on_music_btn_leave)
        self._draw_music_btn()

        self._rebuild_fn = None
        self._theme_btn  = tk.Canvas(
            self.root, width=48, height=48,
            highlightthickness=0, bd=0, cursor="hand2",
        )
        self._theme_btn.place(relx=1.0, rely=0.0, anchor="ne", x=-14, y=14)
        self._theme_btn.bind("<Button-1>", self._toggle_theme)
        self._theme_btn.bind("<Enter>",    self._on_theme_btn_enter)
        self._theme_btn.bind("<Leave>",    self._on_theme_btn_leave)
        self._draw_theme_btn()

        self._show_login()
        self._start_overlay_loop()
        threading.Thread(target=_warmup_pkl_cache, daemon=True).start()

    # Musik helpers
    # Menangani proses init music pada SudokuApp sambil menjaga state internal tetap konsisten.
    def _init_music(self):
        if not PYGAME_AVAILABLE:
            return
        try:
            pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
            _build_sfx()
        except Exception:
            return

        if os.path.exists(MUSIC_FILE):
            self._music_ready = True
            self._start_music()

    # Memulai music pada SudokuApp dan menyalakan mekanisme pendukung yang dibutuhkan.
    def _start_music(self):
        if not PYGAME_AVAILABLE or not self._music_ready:
            return
        if not os.path.exists(MUSIC_FILE):
            return
        try:
            pygame.mixer.music.load(MUSIC_FILE)
            pygame.mixer.music.set_volume(0.7)
            pygame.mixer.music.play(loops=-1)
            self._music_on     = True
            self._music_paused = False
            self._draw_music_btn()
        except Exception:
            pass

    # Mengalihkan music pada SudokuApp sambil menjaga state internal tetap sinkron.
    def _toggle_music(self, event=None):
        if not PYGAME_AVAILABLE:
            return

        is_mouse_click = (
            event is not None and
            str(getattr(event, 'type', '')).lower() in ('4', 'buttonpress')
        )
        if not is_mouse_click:
            try:
                focused = self.root.focus_get()
                if isinstance(focused, tk.Entry):
                    return
            except Exception:
                pass

        if self._music_on:
            try:
                pygame.mixer.music.pause()
            except Exception:
                pass
            self._music_on     = False
            self._music_paused = True
        else:
            if not self._music_ready:
                self._music_paused = False
                return
            try:
                if self._music_paused:
                    pygame.mixer.music.unpause()
                else:
                    self._start_music()
                    return
                self._music_on     = True
                self._music_paused = False
            except Exception:
                return

        self._draw_music_btn()

    # Menggambar music btn pada SudokuApp sesuai state yang sedang aktif.
    def _draw_music_btn(self, hover=False):
        c = self._music_btn
        c.delete("all")
        is_dark = (_CURRENT_THEME_NAME == "dark")
        on = self._music_on

        if is_dark:
            if on:
                bg_fill  = "#1A2840" if not hover else "#223050"
                ring_col = "#2A4060" if not hover else "#3A6090"
                note_col = "#58A6FF"
            else:
                bg_fill  = "#141418" if not hover else "#1C1C22"
                ring_col = "#2A2A32" if not hover else "#38383F"
                note_col = "#3A3A44"
        else:
            if on:
                bg_fill  = "#E0E8F4" if not hover else "#CCD6EC"
                ring_col = "#B0C0D8" if not hover else "#90A8C8"
                note_col = "#0969DA"
            else:
                bg_fill  = "#E8EDF5" if not hover else "#DDE3EE"
                ring_col = "#C5CFE0" if not hover else "#A8B8CC"
                note_col = "#9CA3AF"

        c.create_oval(2, 2, 42, 42, fill=bg_fill,
                      outline=ring_col, width=2)

        c.create_text(22, 23, text="♫",
                      font=("Segoe UI", 17, "bold"),
                      fill=note_col, anchor="center")

        if not on:
            strike = "#FF5555" if is_dark else "#CF222E"
            c.create_line(9, 9, 35, 35,
                          fill=strike, width=2.5, capstyle="round")

        c.config(bg=self._bg_behind(c))

    # Menangani event music btn enter pada SudokuApp dan memperbarui state yang terkait.
    def _on_music_btn_enter(self, _=None):
        self._draw_music_btn(hover=True)

    # Menangani event music btn leave pada SudokuApp dan memperbarui state yang terkait.
    def _on_music_btn_leave(self, _=None):
        self._draw_music_btn(hover=False)

    # Memperbarui music hint pada SudokuApp agar data, status, dan tampilan tetap selaras.
    def _update_music_hint(self):
        try:
            self._draw_music_btn()
        except Exception:
            pass

    # Menangani proses bg behind pada SudokuApp sambil menjaga state internal tetap konsisten.
    def _bg_behind(self, canvas_widget):
        try:
            if _APP_INSTANCE is not None:
                overlay = getattr(_APP_INSTANCE, "_active_overlay", None)
                if (overlay is not None
                        and hasattr(overlay, "_blur_pil")
                        and overlay._blur_pil is not None):
                    try:
                        rx = self.root.winfo_rootx()
                        ry = self.root.winfo_rooty()
                        bx = (canvas_widget.winfo_rootx()
                              + canvas_widget.winfo_width() // 2 - rx)
                        by = (canvas_widget.winfo_rooty()
                              + canvas_widget.winfo_height() // 2 - ry)
                        pil = overlay._blur_pil
                        px = max(0, min(bx, pil.width  - 1))
                        py = max(0, min(by, pil.height - 1))
                        pixel = pil.getpixel((px, py))
                        return f"#{pixel[0]:02x}{pixel[1]:02x}{pixel[2]:02x}"
                    except Exception:
                        pass

            cx = canvas_widget.winfo_rootx() + canvas_widget.winfo_width() // 2
            cy = canvas_widget.winfo_rooty() + canvas_widget.winfo_height() // 2
            best = C_BG

            # Menangani proses walk pada SudokuApp sambil menjaga state internal tetap konsisten.
            def _walk(w):
                nonlocal best
                try:
                    if not w.winfo_ismapped():
                        return
                    wx, wy = w.winfo_rootx(), w.winfo_rooty()
                    ww, wh = w.winfo_width(),  w.winfo_height()
                    if ww <= 0 or wh <= 0:
                        return
                    if not (wx <= cx < wx + ww and wy <= cy < wy + wh):
                        return
                    try:
                        bg = w.cget("bg")
                        if bg:
                            best = bg
                    except Exception:
                        pass
                    for child in w.winfo_children():
                        if child is canvas_widget:
                            continue
                        _walk(child)
                except Exception:
                    pass

            screen = getattr(self, "screen", None)
            if screen:
                _walk(screen)
            else:
                best = self.root.cget("bg")

            return best
        except Exception:
            return C_BG

    # Menangani proses sync overlay bg pada SudokuApp sambil menjaga state internal tetap konsisten.
    def _sync_overlay_bg(self):
        for btn, draw_fn in (
            (self._music_btn, self._draw_music_btn),
            (self._theme_btn, self._draw_theme_btn),
        ):
            try:
                new_bg = self._bg_behind(btn)
                if btn.cget("bg") != new_bg:
                    draw_fn()
            except Exception:
                pass

    # Overlay helpers (theme btn + music hint selalu di atas)
    # Menangani proses raise overlay pada SudokuApp sambil menjaga state internal tetap konsisten.
    def _raise_overlay(self):
        if getattr(self, "_corner_overlay_paused", False):
            return
        self._sync_overlay_bg()
        try:
            self.root.tk.call("raise", self._theme_btn._w)
        except Exception:
            pass
        try:
            self.root.tk.call("raise", self._music_btn._w)
        except Exception:
            pass

    # Memulai overlay loop pada SudokuApp dan menyalakan mekanisme pendukung yang dibutuhkan.
    def _start_overlay_loop(self):
        self._raise_overlay()
        self._overlay_loop_id = self.root.after(300, self._start_overlay_loop)

    # Menghentikan overlay loop pada SudokuApp dan membersihkan state yang masih aktif.
    def _stop_overlay_loop(self):
        _id = getattr(self, "_overlay_loop_id", None)
        if _id is not None:
            try:
                self.root.after_cancel(_id)
            except Exception:
                pass
            self._overlay_loop_id = None

    # Menangani proses clear pada SudokuApp sambil menjaga state internal tetap konsisten.
    def _clear(self):
        self._last_screen_change = time.time()
        if self.screen:
            try:
                self.screen.place_forget()
                self.screen.destroy()
            except Exception: pass
            self.screen = None
        self._raise_overlay()

    # Menampilkan login pada SudokuApp dan mengaktifkan elemen pendukung yang diperlukan.
    def _show_login(self):
        self._rebuild_fn = self._show_login
        self._clear()
        self.screen = LoginScreen(
            self.root,
            self._on_login,
            self._show_player_select_from_login,
            on_attractor=self._show_attractor,
        )
        self.root.after(50, self._raise_overlay)

    # Menampilkan attractor pada SudokuApp dan mengaktifkan elemen pendukung yang diperlukan.
    def _show_attractor(self):
        self._rebuild_fn = self._show_login
        self._clear()
        self.screen = AttractorScreen(self.root, on_dismiss=self._show_login)
        self.root.after(50, self._raise_overlay)

    # Menampilkan player select from login pada SudokuApp dan mengaktifkan elemen pendukung yang diperlukan.
    def _show_player_select_from_login(self, initial_selected=None):
        # Menangani event highlight pada SudokuApp dan memperbarui state yang terkait.
        def _on_highlight(name):
            self._rebuild_fn = lambda: self._show_player_select_from_login(name)

        self._rebuild_fn = lambda: self._show_player_select_from_login(initial_selected)
        self._clear()
        self.screen = PlayerSelectScreen(
            self.root,
            current_user=None,
            on_select=self._on_player_selected,
            on_new_player=self._show_login,
            initial_selected=initial_selected,
            on_highlight=_on_highlight,
            on_back_to_login=self._show_login,
        )
        self.root.after(50, self._raise_overlay)

    # Menangani event login pada SudokuApp dan memperbarui state yang terkait.
    def _on_login(self, username, is_new, greeting):
        self.username = username
        self._rebuild_fn = lambda: self._on_login(username, is_new, greeting)
        self._clear()
        self.screen = GridSizeScreen(
            self.root, username, greeting,
            on_select=self._on_grid_selected)
        self.root.after(50, self._raise_overlay)

    # Menangani event grid selected pada SudokuApp dan memperbarui state yang terkait.
    def _on_grid_selected(self, box):
        self.grid_size = box
        self._rebuild_fn = lambda: self._on_grid_selected(box)
        self._clear()
        self.screen = DifficultyScreen(
            self.root, self.username, self.grid_size,
            on_select=self._on_diff_selected,
            on_back=self._back_to_grid_select)
        self.root.after(50, self._raise_overlay)

    # Menangani event diff selected pada SudokuApp dan memperbarui state yang terkait.
    def _on_diff_selected(self, diff):
        self.difficulty = diff
        self._show_game()

    # Menampilkan game pada SudokuApp dan mengaktifkan elemen pendukung yang diperlukan.
    def _show_game(self):
        if isinstance(self.screen, GameScreen):
            self.difficulty = self.screen.difficulty

        saved_state = None
        if isinstance(self.screen, GameScreen):
            gs = self.screen
            was_running = gs.timer_running
            gs._stop_timer()
            saved_state = {
                "puzzle":        [row[:] for row in gs.puzzle],
                "solution":      [row[:] for row in gs.solution],
                "current_board": [row[:] for row in gs.current_board],
                "draft_board":   {k: set(v) for k, v in gs.draft_board.items()},
                "draft_mode":    gs.draft_mode,
                "elapsed":       gs.elapsed,
                "timer_running": was_running,
                "game_over":     gs.game_over,
                "error_count":   gs.error_count,
                "move_count":    gs.move_count,
                "hints_used":    gs.hints_used,
                "auto_used_count": gs.auto_used_count,
                "hearts":        gs.hearts,
                "cell_errors":   dict(gs.cell_errors),
                "cell_last_time": dict(gs.cell_last_time),
                "near_miss_count": gs.near_miss_count,
                "guessing_count":  gs.guessing_count,
                "empty_cells":   gs.empty_cells,
                "hint_shown":    gs.hint_shown,
            }

        self._rebuild_fn = self._show_game
        self._clear()
        self.screen = GameScreen(
            self.root,
            username=self.username,
            grid_size=self.grid_size,
            difficulty=self.difficulty,
            on_finish=self._on_finish,
            resume_state=saved_state)
        self.root.after(50, self._raise_overlay)

    # Menangani event finish pada SudokuApp dan memperbarui state yang terkait.
    def _on_finish(self, session, ml):
        self._rebuild_fn = lambda: self._on_finish(session, ml)
        self._clear()
        self.screen = PerformanceDashboard(
            self.root, self, self.username, session, ml)
        self.root.after(50, self._raise_overlay)

    # Memulai recommended grid pada SudokuApp dan menyalakan mekanisme pendukung yang dibutuhkan.
    def _start_recommended_grid(self, box, difficulty=None):
        self.grid_size = box
        if difficulty in {"Easy", "Normal", "Hard"}:
            self.difficulty = difficulty
        else:
            self.difficulty = "Easy" if box == 2 else "Normal"
        self._show_game()

    # Menangani proses play again pada SudokuApp sambil menjaga state internal tetap konsisten.
    def _play_again(self, _=None):
        self._rebuild_fn = self._play_again
        self._clear()
        self.screen = DifficultyScreen(
            self.root, self.username, self.grid_size,
            on_select=self._on_diff_selected,
            on_back=self._back_to_grid_select)
        self.root.after(50, self._raise_overlay)

    # Menangani proses back to grid select pada SudokuApp sambil menjaga state internal tetap konsisten.
    def _back_to_grid_select(self):
        greeting = "Halo kembali"
        self._rebuild_fn = lambda: self._back_to_grid_select()
        self._clear()
        self.screen = GridSizeScreen(
            self.root, self.username, greeting,
            on_select=self._on_grid_selected)
        self.root.after(50, self._raise_overlay)

    # Ganti Pemain → tampilkan PlayerSelectScreen
    # Menampilkan player select pada SudokuApp dan mengaktifkan elemen pendukung yang diperlukan.
    def _show_player_select(self, _=None, initial_selected=None):
        if isinstance(self.screen, GameScreen):
            gs = self.screen
            was_running = gs.timer_running
            gs._stop_timer()
            self._ingame_saved_username   = self.username
            self._ingame_saved_grid_size  = gs.grid_size
            self._ingame_saved_difficulty = gs.difficulty
            self._ingame_saved_state = {
                "puzzle":          [row[:] for row in gs.puzzle],
                "solution":        [row[:] for row in gs.solution],
                "current_board":   [row[:] for row in gs.current_board],
                "draft_board":     {k: set(v) for k, v in gs.draft_board.items()},
                "draft_mode":      gs.draft_mode,
                "elapsed":         gs.elapsed,
                "timer_running":   was_running,
                "game_over":       gs.game_over,
                "error_count":     gs.error_count,
                "move_count":      gs.move_count,
                "hints_used":      gs.hints_used,
                "auto_used_count": gs.auto_used_count,
                "hearts":          gs.hearts,
                "cell_errors":     dict(gs.cell_errors),
                "cell_last_time":  dict(gs.cell_last_time),
                "near_miss_count": gs.near_miss_count,
                "guessing_count":  gs.guessing_count,
                "empty_cells":     gs.empty_cells,
                "hint_shown":      gs.hint_shown,
            }

        # Menangani event highlight pada SudokuApp dan memperbarui state yang terkait.
        def _on_highlight(name):
            self._rebuild_fn = lambda: self._show_player_select(initial_selected=name)

        _on_back_game = (self._resume_game_from_player_select
                         if self._ingame_saved_state is not None else None)

        self._rebuild_fn = lambda: self._show_player_select(initial_selected=initial_selected)
        self._clear()
        self.screen = PlayerSelectScreen(
            self.root,
            current_user=self.username,
            on_select=self._on_player_selected,
            on_new_player=self._show_login,
            initial_selected=initial_selected,
            on_highlight=_on_highlight,
            on_back_to_game=_on_back_game,
        )
        self.root.after(50, self._raise_overlay)

    # Menangani proses resume game from player select pada SudokuApp sambil menjaga state internal tetap konsisten.
    def _resume_game_from_player_select(self):
        state = self._ingame_saved_state
        if not state:
            self._back_to_grid_select()
            return

        self.username   = self._ingame_saved_username
        self.grid_size  = self._ingame_saved_grid_size
        self.difficulty = self._ingame_saved_difficulty

        self._ingame_saved_state      = None
        self._ingame_saved_username   = None
        self._ingame_saved_grid_size  = None
        self._ingame_saved_difficulty = None

        self._rebuild_fn = self._show_game
        self._clear()
        self.screen = GameScreen(
            self.root,
            username=self.username,
            grid_size=self.grid_size,
            difficulty=self.difficulty,
            on_finish=self._on_finish,
            resume_state=state)
        self.root.after(50, self._raise_overlay)

    # Menangani event player selected pada SudokuApp dan memperbarui state yang terkait.
    def _on_player_selected(self, username):
        self.username = username
        self._rebuild_fn = lambda: self._on_player_selected(username)
        self._clear()
        data    = load_data()
        is_new  = username not in data.get("players", {})
        greeting = "Halo" if is_new else "Halo kembali"
        self.screen = GridSizeScreen(
            self.root, username, greeting,
            on_select=self._on_grid_selected)
        self.root.after(50, self._raise_overlay)

    # Logout → reset semua state lalu ke login
    # Menangani proses logout pada SudokuApp sambil menjaga state internal tetap konsisten.
    def _logout(self, _=None):
        self.username   = None
        self.grid_size  = 3
        self.difficulty = "Normal"
        self._show_login()

    # Menangani proses kiosk reset pada SudokuApp sambil menjaga state internal tetap konsisten.
    def _kiosk_reset(self, _=None):
        self._logout()

    # Menampilkan exit confirm global pada SudokuApp dan mengaktifkan elemen pendukung yang diperlukan.
    def _show_exit_confirm_global(self):
        _popup = getattr(self, "_global_exit_popup", None)
        if _popup is not None:
            try:
                if _popup.winfo_exists():
                    return
            except Exception:
                pass

        parent = self.root

        _blur_pre = _grab_blur_bg(self.root)

        overlay = tk.Frame(parent, bg="")
        overlay.place(relx=0, rely=0, relwidth=1, relheight=1)
        overlay.lift()
        self._global_exit_popup = overlay

        dim = _place_blur_canvas(overlay, self.root, pre_captured=_blur_pre)

        card_bg     = C_SURFACE
        card_border = C_ERROR

        glow_glob = tk.Frame(overlay, bg=card_border)
        glow_glob.place(relx=0.5, rely=0.62, anchor="center", width=426, height=236)

        card = tk.Frame(glow_glob, bg=card_bg,
                        highlightbackground=card_border, highlightthickness=0)
        card.place(relx=0, rely=0, relwidth=1, relheight=1)

        SLIDE_STEPS_G = 12
        # Menangani proses slide global pada SudokuApp sambil menjaga state internal tetap konsisten.
        def _slide_global(step, _glow=glow_glob, _ov=overlay):
            if not _ov.winfo_exists():
                return
            t      = step / SLIDE_STEPS_G
            t_ease = 1 - (1 - t) ** 3
            rely_now = 0.62 + (0.5 - 0.62) * t_ease
            try:
                _glow.place(rely=rely_now)
            except Exception:
                return
            if step < SLIDE_STEPS_G:
                _ov.after(16, lambda: _slide_global(step + 1))
        _slide_global(0)

        stripe_cv = tk.Canvas(card, height=6, bg=card_bg, highlightthickness=0)
        stripe_cv.pack(fill="x")
        # Menggambar stripe pada SudokuApp sesuai state yang sedang aktif.
        def _draw_stripe(cv=stripe_cv):
            cv.update_idletasks()
            w = cv.winfo_width() or 420
            stops = [(0, C_ERROR), (w // 2, C_WARN), (w, "#FFDD55")]
            steps = 40
            for i in range(len(stops) - 1):
                x1, c1 = stops[i]
                x2, c2 = stops[i + 1]
                for k in range(steps):
                    t_  = k / steps
                    r_  = int(int(c1[1:3], 16) * (1 - t_) + int(c2[1:3], 16) * t_)
                    g_  = int(int(c1[3:5], 16) * (1 - t_) + int(c2[3:5], 16) * t_)
                    b_  = int(int(c1[5:7], 16) * (1 - t_) + int(c2[5:7], 16) * t_)
                    xi  = x1 + k * (x2 - x1) // steps
                    xj  = x1 + (k + 1) * (x2 - x1) // steps
                    cv.create_rectangle(xi, 0, xj, 6,
                                        fill=f"#{r_:02x}{g_:02x}{b_:02x}",
                                        outline="")
        stripe_cv.after(60, _draw_stripe)

        icon_row = tk.Frame(card, bg=card_bg)
        icon_row.pack(pady=(20, 4))
        tk.Label(icon_row, text="\U0001f6aa",
                 font=("Segoe UI", 28), bg=card_bg, fg=C_ERROR).pack(side="left")
        tk.Label(icon_row, text="Exit Game",
                 font=("Segoe UI", 17, "bold"),
                 bg=card_bg, fg=C_TEXT).pack(side="left")

        tk.Frame(card, height=1, bg=C_BORDER).pack(fill="x", padx=24, pady=(4, 0))

        tk.Label(card,
                 text="Apakah Anda yakin ingin menutup program?",
                 font=("Segoe UI", 10),
                 bg=card_bg, fg=C_TEXT_DIM,
                 justify="center").pack(pady=(12, 16))

        btn_row = tk.Frame(card, bg=card_bg)
        btn_row.pack(fill="x", padx=24, pady=(0, 20))

        _corner_icons_lower()

        # Menangani proses cancel pada SudokuApp sambil menjaga state internal tetap konsisten.
        def _cancel():
            _corner_icons_restore()
            try:
                overlay.destroy()
            except Exception:
                pass
            self._global_exit_popup = None

        # Menangani proses confirm exit pada SudokuApp sambil menjaga state internal tetap konsisten.
        def _confirm_exit():
            try:
                overlay.destroy()
            except Exception:
                pass
            self._global_exit_popup = None
            self._stop_overlay_loop()
            try:
                self.root.destroy()
            except Exception:
                pass

        cancel_btn = tk.Button(
            btn_row,
            text="Batal",
            font=("Segoe UI", 10, "bold"),
            bg=C_SURFACE2, fg=C_TEXT,
            activebackground=C_BORDER, activeforeground=C_TEXT,
            relief="flat", cursor="hand2", pady=10,
            command=_cancel)
        cancel_btn.pack(side="left", fill="x", expand=True, padx=(0, 6))

        exit_btn = tk.Button(
            btn_row,
            text="Ya, Keluar",
            font=("Segoe UI", 10, "bold"),
            bg=C_ERROR, fg="#FFFFFF",
            activebackground="#FF9999", activeforeground="#FFFFFF",
            relief="flat", cursor="hand2", pady=10,
            command=_confirm_exit)
        exit_btn.pack(side="left", fill="x", expand=True, padx=(6, 0))

        cancel_btn.bind("<Enter>",
            lambda _: cancel_btn.config(bg=C_BORDER, fg=C_TEXT))
        cancel_btn.bind("<Leave>",
            lambda _: cancel_btn.config(bg=C_SURFACE2, fg=C_TEXT))

        dim.bind("<Button-1>", lambda _: _cancel())
        overlay.bind("<Button-1>", lambda e: _cancel() if e.widget is overlay else None)

        # Menangani event esc exit pada SudokuApp dan memperbarui state yang terkait.
        def _on_esc_exit(event=None):
            _cancel()
            return "break"
        overlay.bind("<Escape>", _on_esc_exit)
        overlay.focus_set()

    # Menangani event esc global pada SudokuApp dan memperbarui state yang terkait.
    def _on_esc_global(self, event=None):
        if (time.time() - getattr(self, "_last_screen_change", 0)) < 0.15:
            return

        overlay = getattr(self, "_active_overlay", None)
        if overlay is not None:
            try:
                overlay._close()
            except Exception:
                self._active_overlay = None
            self._last_screen_change = time.time()
            return "break"

        screen = self.screen
        if screen is None:
            return

        if isinstance(screen, GameScreen):
            screen._on_esc()

        elif isinstance(screen, AttractorScreen):
            self._show_login()

        elif isinstance(screen, PlayerSelectScreen):
            if self._ingame_saved_state is not None:
                self._resume_game_from_player_select()
            else:
                self._show_login()

        elif isinstance(screen, GridSizeScreen):
            self._show_login()

        elif isinstance(screen, DifficultyScreen):
            self._back_to_grid_select()

        elif isinstance(screen, PerformanceDashboard):
            self._back_to_grid_select()

        elif isinstance(screen, AchievementPopup):
            pass

        else:
            self._show_exit_confirm_global()

    # Menangani proses exit pada SudokuApp sambil menjaga state internal tetap konsisten.
    def _exit(self, _=None):
        self._show_exit_confirm_global()

    # Theme toggle helpers
    # Menggambar theme btn pada SudokuApp sesuai state yang sedang aktif.
    def _draw_theme_btn(self, hover=False):
        c      = self._theme_btn
        is_dark = _CURRENT_THEME_NAME == "dark"
        c.delete("all")

        if is_dark:
            ring_col  = "#2A3F5F"
            ring_hi   = "#3A5F8F"
            icon_col  = "#FFD700"
            bg_fill   = "#1A2840" if not hover else "#223050"
        else:
            ring_col  = "#C5CFE0"
            ring_hi   = "#A0B0CC"
            icon_col  = "#4A80C8"
            bg_fill   = "#E0E8F4" if not hover else "#CCD6EC"

        c.create_oval(2, 2, 46, 46, fill=bg_fill,
                      outline=ring_hi if hover else ring_col, width=2)

        if is_dark:
            cx, cy, r = 24, 24, 8
            c.create_oval(cx-r, cy-r, cx+r, cy+r,
                          fill=icon_col, outline="")
            for i in range(8):
                angle  = math.radians(i * 45)
                x1 = cx + (r + 3) * math.cos(angle)
                y1 = cy + (r + 3) * math.sin(angle)
                x2 = cx + (r + 7) * math.cos(angle)
                y2 = cy + (r + 7) * math.sin(angle)
                c.create_line(x1, y1, x2, y2,
                              fill=icon_col, width=2, capstyle="round")
        else:
            c.create_oval(12, 10, 32, 38,
                          fill=icon_col, outline="")
            c.create_oval(17, 10, 37, 34,
                          fill=bg_fill, outline="")

        c.config(bg=self._bg_behind(c))

    # Menangani event theme btn enter pada SudokuApp dan memperbarui state yang terkait.
    def _on_theme_btn_enter(self, _=None):
        self._draw_theme_btn(hover=True)

    # Menangani event theme btn leave pada SudokuApp dan memperbarui state yang terkait.
    def _on_theme_btn_leave(self, _=None):
        self._draw_theme_btn(hover=False)

    # Mengalihkan theme pada SudokuApp sambil menjaga state internal tetap sinkron.
    def _toggle_theme(self, _=None):
        new = "light" if _CURRENT_THEME_NAME == "dark" else "dark"
        apply_theme(new)

        self.root.configure(bg=C_BG)

        self._draw_theme_btn()
        self._draw_music_btn()

        self.root.tk.call("raise", self._theme_btn._w)
        self.root.tk.call("raise", self._music_btn._w)

        if callable(self._rebuild_fn):
            try:
                self._rebuild_fn()
                self.root.tk.call("raise", self._theme_btn._w)
                self.root.tk.call("raise", self._music_btn._w)
            except Exception:
                pass

    # Menangani proses run pada SudokuApp sambil menjaga state internal tetap konsisten.
    def run(self):
        self.root.mainloop()


# =============================================================================
# BAGIAN 9: ACHIEVEMENT, SCORE CARD, DAN MEDIA
# =============================================================================

# Menggambar ikon achievement pada kartu pop-up atau dashboard sesuai ukuran yang diminta.
def _draw_ach_icon(cv, badge_id, color, w=120, h=68, bg="#0D1117"):
    _m = math
    cv.delete("all")
    cx, cy = w // 2, h // 2

    # Fungsi bantu ini memecah logika draw ach icon agar langkah kecilnya lebih rapi dan mudah dipakai ulang di bagian ini.
    def _star(px, py, r_out=22, r_in=9, pts=5):
        coords = []
        for i in range(pts * 2):
            ang = _m.pi / pts * i - _m.pi / 2
            r   = r_out if i % 2 == 0 else r_in
            coords.extend([px + r * _m.cos(ang), py + r * _m.sin(ang)])
        cv.create_polygon(coords, fill=color, outline="")

    # Fungsi bantu ini memecah logika draw ach icon agar langkah kecilnya lebih rapi dan mudah dipakai ulang di bagian ini.
    def _shield(px, py, s=22):
        pts = [
            px - s, py - s * 0.5,
            px,     py - s,
            px + s, py - s * 0.5,
            px + s, py + s * 0.3,
            px,     py + s,
            px - s, py + s * 0.3,
        ]
        cv.create_polygon(pts, fill=color, outline="")
        cv.create_line(px - s * 0.45, py + s * 0.05,
                       px - s * 0.1,  py + s * 0.45,
                       fill=bg, width=3, capstyle="round")
        cv.create_line(px - s * 0.1,  py + s * 0.45,
                       px + s * 0.5,  py - s * 0.3,
                       fill=bg, width=3, capstyle="round")

    # Fungsi bantu ini memecah logika draw ach icon agar langkah kecilnya lebih rapi dan mudah dipakai ulang di bagian ini.
    def _lightning(px, py, s=22):
        pts = [
            px + s * 0.22,  py - s,
            px - s * 0.38,  py - s * 0.04,
            px + s * 0.08,  py - s * 0.04,
            px - s * 0.22,  py + s,
            px + s * 0.38,  py + s * 0.04,
            px - s * 0.08,  py + s * 0.04,
        ]
        cv.create_polygon(pts, fill=color, outline="")

    # Fungsi bantu ini memecah logika draw ach icon agar langkah kecilnya lebih rapi dan mudah dipakai ulang di bagian ini.
    def _double_lightning(px, py, s=22):
        _lightning(px - s * 0.32, py, s * 0.82)
        _lightning(px + s * 0.38, py, s * 0.62)

    # Fungsi bantu ini memecah logika draw ach icon agar langkah kecilnya lebih rapi dan mudah dipakai ulang di bagian ini.
    def _target(px, py):
        for r, col in [(22, color), (15, bg), (9, color), (4, bg)]:
            cv.create_oval(px - r, py - r, px + r, py + r, fill=col, outline="")

    # Fungsi bantu ini memecah logika draw ach icon agar langkah kecilnya lebih rapi dan mudah dipakai ulang di bagian ini.
    def _crown(px, py, s=22):
        bot = py + s * 0.52
        pts = [
            px - s,        bot,
            px - s,        py + s * 0.02,
            px - s * 0.48, py - s * 0.62,
            px - s * 0.14, py + s * 0.02,
            px,            py - s,
            px + s * 0.14, py + s * 0.02,
            px + s * 0.48, py - s * 0.62,
            px + s,        py + s * 0.02,
            px + s,        bot,
        ]
        cv.create_polygon(pts, fill=color, outline="")
        for dx, dy in [(-s * 0.48, -s * 0.62), (0, -s), (s * 0.48, -s * 0.62)]:
            cv.create_oval(px + dx - 4, py + dy - 4,
                           px + dx + 4, py + dy + 4, fill=bg, outline="")

    # Fungsi bantu ini memecah logika draw ach icon agar langkah kecilnya lebih rapi dan mudah dipakai ulang di bagian ini.
    def _medal(px, py, s=17):
        cv.create_rectangle(px - 7, py - s - 13, px - 2, py - s + 2, fill=color, outline="")
        cv.create_rectangle(px + 2, py - s - 13, px + 7, py - s + 2, fill=color, outline="")
        cv.create_oval(px - s, py - s, px + s, py + s, fill=color, outline="")
        cv.create_oval(px - s + 5, py - s + 5,
                       px + s - 5, py + s - 5, fill=bg, outline="")

    # Fungsi bantu ini memecah logika draw ach icon agar langkah kecilnya lebih rapi dan mudah dipakai ulang di bagian ini.
    def _flame(px, py, s=22):
        pts = [
            px,             py - s,
            px + s * 0.55,  py - s * 0.32,
            px + s * 0.65,  py + s * 0.42,
            px,             py + s * 0.55,
            px - s * 0.65,  py + s * 0.42,
            px - s * 0.55,  py - s * 0.32,
        ]
        cv.create_polygon(pts, fill=color, outline="", smooth=True)
        cv.create_oval(px - s * 0.23, py + s * 0.02,
                       px + s * 0.23, py + s * 0.42, fill=bg, outline="")

    # Fungsi bantu ini memecah logika draw ach icon agar langkah kecilnya lebih rapi dan mudah dipakai ulang di bagian ini.
    def _lightbulb(px, py, s=20, cross=False):
        cv.create_oval(px - s, py - s * 1.12, px + s, py + s * 0.38, fill=color, outline="")
        cv.create_rectangle(px - s * 0.48, py + s * 0.32,
                            px + s * 0.48, py + s * 0.62, fill=color, outline="")
        cv.create_rectangle(px - s * 0.34, py + s * 0.60,
                            px + s * 0.34, py + s * 0.82, fill=color, outline="")
        if cross:
            r = s * 0.38
            cv.create_line(px - r, py - s * 0.5, px + r, py + s * 0.08,
                           fill=bg, width=3, capstyle="round")
            cv.create_line(px + r, py - s * 0.5, px - r, py + s * 0.08,
                           fill=bg, width=3, capstyle="round")

    # Fungsi bantu ini memecah logika draw ach icon agar langkah kecilnya lebih rapi dan mudah dipakai ulang di bagian ini.
    def _stopwatch(px, py, r=19):
        cv.create_rectangle(px - 5, py - r - 7, px + 5, py - r, fill=color, outline="")
        cv.create_oval(px - 4, py - r - 13, px + 4, py - r - 5, fill=color, outline="")
        cv.create_oval(px - r, py - r * 0.8, px + r, py + r * 1.2, fill=color, outline="")
        cv.create_line(px, py + r * 0.2,
                       px + int(r * 0.62), py - int(r * 0.38),
                       fill=bg, width=3, capstyle="round")

    # Fungsi bantu ini memecah logika draw ach icon agar langkah kecilnya lebih rapi dan mudah dipakai ulang di bagian ini.
    def _grid3(px, py, cell=7, gap=3):
        step = cell + gap
        for row in range(3):
            for col in range(3):
                x0 = px - step + col * step
                y0 = py - step + row * step
                cv.create_rectangle(x0, y0, x0 + cell, y0 + cell,
                                    fill=color, outline="")

    # Fungsi bantu ini memecah logika draw ach icon agar langkah kecilnya lebih rapi dan mudah dipakai ulang di bagian ini.
    def _compass(px, py, r=22):
        cv.create_oval(px - r, py - r, px + r, py + r, outline=color, width=2)
        for i in range(4):
            ang   = _m.pi / 2 * i - _m.pi / 2
            tip_x = px + r * 0.82 * _m.cos(ang)
            tip_y = py + r * 0.82 * _m.sin(ang)
            l_x   = px + r * 0.22 * _m.cos(ang + _m.pi / 2)
            l_y   = py + r * 0.22 * _m.sin(ang + _m.pi / 2)
            r_x   = px + r * 0.22 * _m.cos(ang - _m.pi / 2)
            r_y   = py + r * 0.22 * _m.sin(ang - _m.pi / 2)
            fill_c = color if i % 2 == 0 else bg
            cv.create_polygon([tip_x, tip_y, l_x, l_y, px, py, r_x, r_y],
                              fill=fill_c, outline=color, width=1)

    # Fungsi bantu ini memecah logika draw ach icon agar langkah kecilnya lebih rapi dan mudah dipakai ulang di bagian ini.
    def _diamond(px, py, s=22):
        pts = [px, py - s, px + s * 0.72, py, px, py + s, px - s * 0.72, py]
        cv.create_polygon(pts, fill=color, outline="")
        inner = [px, py - s * 0.46, px + s * 0.36, py, px, py + s * 0.46, px - s * 0.36, py]
        cv.create_polygon(inner, fill=bg, outline="")

    # Fungsi bantu ini memecah logika draw ach icon agar langkah kecilnya lebih rapi dan mudah dipakai ulang di bagian ini.
    def _comeback_arrow(px, py, r=20):
        cv.create_arc(px - r, py - r, px + r, py + r,
                      start=50, extent=260, style="arc", outline=color, width=4)
        ang = _m.radians(50)
        ex  = px + r * _m.cos(ang)
        ey  = py - r * _m.sin(ang)
        perp = ang - _m.pi / 2
        pts  = [
            ex, ey,
            ex + 10 * _m.cos(perp + 0.45), ey + 10 * _m.sin(perp + 0.45),
            ex + 10 * _m.cos(perp - 0.45), ey + 10 * _m.sin(perp - 0.45),
        ]
        cv.create_polygon(pts, fill=color, outline="")

    _ICON_MAP = {
        "pemula_berhasil":     lambda: _star(cx, cy),
        "maraton":             lambda: _medal(cx, cy + 2),
        "veteran":             lambda: _shield(cx, cy),
        "konsisten":           lambda: _lightning(cx, cy),
        "tak_terkalahkan":     lambda: _flame(cx, cy),
        "serial_winner":       lambda: _crown(cx, cy + 5),
        "comeback":            lambda: _comeback_arrow(cx, cy),
        "kilat":               lambda: _lightning(cx, cy),
        "cepat_kilat":         lambda: _double_lightning(cx, cy),
        "speed_demon":         lambda: _stopwatch(cx, cy + 2),
        "tanpa_petunjuk":      lambda: _lightbulb(cx, cy, cross=True),
        "tanpa_cela":          lambda: _shield(cx, cy),
        "sempurna":            lambda: _star(cx, cy),
        "efisien":             lambda: _target(cx, cy),
        "ahli_hard":           lambda: _flame(cx, cy),
        "tanpa_menyerah_hard": lambda: _flame(cx, cy),
        "master_9x9":          lambda: _grid3(cx, cy),
        "explorer":            lambda: _compass(cx, cy),
        "jenius":              lambda: _lightbulb(cx, cy),
        "pakar":               lambda: _diamond(cx, cy),
    }
    fn = _ICON_MAP.get(badge_id)
    if fn:
        fn()
    else:
        cv.create_oval(cx - 22, cy - 22, cx + 22, cy + 22, fill=color, outline="")

class AchievementPopup(tk.Frame):
    """
    AchievementPopup - Overlay animasi kartu achievement pasca-game.

    Deskripsi:
        Menampilkan satu per satu kartu achievement yang diraih pemain.
        Setiap kartu: ikon vektor berwarna + nama + deskripsi.
        Mendukung auto-advance (timer), skip satu, dan skip semua.

    Atribut:
        badges (list[dict]): Achievement yang diraih.
        on_done (callable): Callback setelah semua kartu selesai.
        _idx (int): Indeks kartu yang sedang ditampilkan.
    """
    # Menginisialisasi objek AchievementPopup dan menyiapkan state awal, referensi penting, serta elemen yang dibutuhkan sebelum layar dipakai.
    def __init__(self, master, badges, on_done, initial_idx=0,
                 pre_blur_photo=None):
        super().__init__(master, bg="")
        self.on_done         = on_done
        self.badges          = badges
        self._idx            = initial_idx
        self._after_id       = None
        self._skip_bound     = False
        self._pre_blur_photo = pre_blur_photo
        self.place(relx=0, rely=0, relwidth=1, relheight=1)
        self.lift()
        _corner_icons_lower()
        self._bind_skip()
        self._show_next()

    # Input skip
    # Menangani proses bind skip pada AchievementPopup sambil menjaga state internal tetap konsisten.
    def _bind_skip(self):
        if self._skip_bound:
            return
        try:
            root = self.winfo_toplevel()
            root.bind("<KeyPress>",   self._on_skip, add="+")
            root.bind("<Button-1>",   self._on_skip, add="+")
            root.bind("<Button-3>",   self._on_skip, add="+")
            root.bind("<space>",      self._on_skip, add="+")
            root.bind("<Return>",     self._on_skip, add="+")
        except Exception:
            pass
        self._skip_bound = True

    # Menangani proses unbind skip pada AchievementPopup sambil menjaga state internal tetap konsisten.
    def _unbind_skip(self):
        try:
            root = self.winfo_toplevel()
            for ev in ("<KeyPress>", "<Button-1>", "<Button-3>",
                       "<space>", "<Return>"):
                try: root.unbind(ev)
                except Exception: pass
        except Exception:
            pass
        self._skip_bound = False

    # Menangani event skip pada AchievementPopup dan memperbarui state yang terkait.
    def _on_skip(self, event=None):
        if not self.winfo_exists():
            return
        if self._after_id:
            try: self.after_cancel(self._after_id)
            except Exception: pass
            self._after_id = None
        self._show_next()

    # Menangani event skip all pada AchievementPopup dan memperbarui state yang terkait.
    def _on_skip_all(self, event=None):
        if not self.winfo_exists():
            return
        self._idx = len(self.badges)
        self._on_skip()

    # Tampilan
    # Menampilkan next pada AchievementPopup dan mengaktifkan elemen pendukung yang diperlukan.
    def _show_next(self):
        for w in self.winfo_children():
            try: w.destroy()
            except Exception: pass
        if self._idx >= len(self.badges):
            self._finish()
            return
        badge = self.badges[self._idx]
        self._idx += 1
        color    = badge.get("warna", "#58A6FF")
        total    = len(self.badges)
        current  = self._idx

        self.config(bg=C_BG)

        _play_sfx(_SFX_ACHIEVEMENT)

        try:
            _root_ref = self.winfo_toplevel()
        except Exception:
            _root_ref = None

        blur_lbl = _place_blur_canvas(
            self, _root_ref,
            radius=16, darken=0.32,
            pre_captured=self._pre_blur_photo
        )

        blur_lbl.bind("<Button-1>", self._on_skip)

        glow = tk.Frame(self, bg=color,
                        highlightbackground=color, highlightthickness=0)
        glow.place(relx=0.5, rely=0.5, anchor="center", width=428, height=256)

        card = tk.Frame(glow, bg=C_SURFACE,
                        highlightbackground=color, highlightthickness=2)
        card.place(relx=0, rely=0, relwidth=1, relheight=1)

        SLIDE_STEPS = 12
        _rely_start = 0.62
        _rely_end   = 0.5
        # Menangani proses slide pada AchievementPopup sambil menjaga state internal tetap konsisten.
        def _slide(step):
            if not self.winfo_exists():
                return
            t = step / SLIDE_STEPS
            t_ease = 1 - (1 - t) ** 3
            rely_now = _rely_start + (_rely_end - _rely_start) * t_ease
            try:
                glow.place(rely=rely_now)
            except Exception:
                return
            if step < SLIDE_STEPS:
                self.after(16, lambda: _slide(step + 1))
        _slide(0)

        tk.Frame(card, bg=color, height=6).pack(fill="x")

        hdr = tk.Frame(card, bg=C_SURFACE)
        hdr.pack(fill="x", padx=16, pady=(10, 0))
        tk.Label(hdr, text="🏆  ACHIEVEMENT UNLOCKED",
                 font=("Segoe UI", 9, "bold"), bg=C_SURFACE, fg=color).pack(side="left")
        if total > 1:
            tk.Label(hdr, text=f"{current}/{total}",
                     font=("Segoe UI", 8), bg=C_SURFACE, fg=C_TEXT_DIM).pack(side="right")

        icon_cv = tk.Canvas(card, width=120, height=68,
                            bg=C_SURFACE, highlightthickness=0)
        icon_cv.pack(pady=(4, 0))
        _draw_ach_icon(icon_cv, badge.get("id", ""), color, w=120, h=68, bg=C_SURFACE)
        tk.Label(card, text=badge["nama"], font=("Segoe UI", 17, "bold"),
                 bg=C_SURFACE, fg=C_TEXT).pack()
        tk.Label(card, text=badge["desc"], font=("Segoe UI", 10),
                 bg=C_SURFACE, fg=C_TEXT_DIM).pack(pady=(4, 0))

        tk.Label(card, text="Tekan tombol apapun untuk lanjut",
                 font=("Segoe UI", 8), bg=C_SURFACE, fg=C_TEXT_DIM).pack(pady=(6, 0))

        bar_host = tk.Frame(card, bg=C_SURFACE2, height=5)
        bar_host.pack(fill="x", padx=24, pady=(8, 0))
        bar = tk.Frame(bar_host, bg=color, height=5)
        bar.place(relx=0, rely=0, relwidth=1.0, relheight=1.0)

        STEPS = 60
        # Menangani proses shrink pada AchievementPopup sambil menjaga state internal tetap konsisten.
        def _shrink(step):
            if not self.winfo_exists(): return
            try: bar.place(relwidth=max(0.0, 1.0 - step / STEPS))
            except Exception: pass
            if step < STEPS:
                self._after_id = self.after(50, lambda: _shrink(step + 1))
            else:
                self._show_next()
        _shrink(0)

    # Menangani proses finish pada AchievementPopup sambil menjaga state internal tetap konsisten.
    def _finish(self):
        self._unbind_skip()
        _corner_icons_restore()
        try:
            if self._after_id: self.after_cancel(self._after_id)
        except Exception: pass
        try: self.destroy()
        except Exception: pass
        if callable(self.on_done): self.on_done()

# SCORE CARD PNG
# [DEMO-POINT] Export score card PNG berdesain dengan logo + statistik
# Menyimpan kartu skor ke file gambar dan menangani lokasi output yang dipilih.
def _save_score_card(username, session, player_type, type_color, ai_message,
                     all_sessions=None, ml_summary=None, achievements=None,
                     parent_widget=None):
    if not PIL_AVAILABLE:
        if parent_widget:
            messagebox.showinfo("Info",
                "Pillow belum terinstall.\nJalankan: pip install Pillow",
                parent=parent_widget)
        return None

    try:
        _math, _rnd, _dt = math, random, datetime

        W = 860

        # Fungsi bantu ini memecah logika save score card agar langkah kecilnya lebih rapi dan mudah dipakai ulang di bagian ini.
        def _hex(h):
            h = h.lstrip("#")
            return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
        # Fungsi bantu ini memecah logika save score card agar langkah kecilnya lebih rapi dan mudah dipakai ulang di bagian ini.
        def _lerp(c1, c2, t):
            return tuple(max(0,min(255,int(c1[i]+(c2[i]-c1[i])*t))) for i in range(3))
        # Fungsi bantu ini memecah logika save score card agar langkah kecilnya lebih rapi dan mudah dipakai ulang di bagian ini.
        def _blend(b, o, a):
            return tuple(max(0,min(255,int(b[i]*(1-a)+o[i]*a))) for i in range(3))
        # Fungsi bantu ini memecah logika save score card agar langkah kecilnya lebih rapi dan mudah dipakai ulang di bagian ini.
        def _lighten(c, f):
            return tuple(max(0,min(255,int(v+(255-v)*f))) for v in c)
        # Fungsi bantu ini memecah logika save score card agar langkah kecilnya lebih rapi dan mudah dipakai ulang di bagian ini.
        def _darken(c, f):
            return tuple(max(0,min(255,int(v*(1-f)))) for v in c)
        # Fungsi bantu ini memecah logika save score card agar langkah kecilnya lebih rapi dan mudah dipakai ulang di bagian ini.
        def _hex_rgba(h, a=255):
            r,g,b = _hex(h)
            return (r,g,b,a)

        BG      = _hex("#0B0F1E")
        BG2     = _hex("#111827")
        CARD    = _hex("#1A2235")
        CARD2   = _hex("#1E2A3A")
        BORDER  = _hex("#2D3A52")
        BORDER2 = _hex("#3D5070")
        TXT     = _hex("#EFF6FF")
        DIM     = _hex("#7A8BA8")
        DIM2    = _hex("#A8BACE")

        ACC_RAW = type_color if type_color else "#00D4FF"
        ACC     = _hex(ACC_RAW)
        br      = sum(ACC)/3
        if br < 150:
            ACC = tuple(min(255,int(v*2.0)) for v in ACC)

        ACC2    = _lighten(ACC, 0.42)

        CYAN    = _hex("#00D4FF")
        MAGENTA = _hex("#FF3A8C")
        GOLD    = _hex("#FFD24C")
        GREEN   = _hex("#00F0A0")
        PURPLE  = _hex("#C678FF")
        ORANGE  = _hex("#FF8C42")
        WHITE   = _hex("#FFFFFF")

        diff      = session.get("difficulty","Normal")
        gs        = session.get("grid_size",3)
        N         = gs*gs
        score     = int(session.get("score",0) or 0)
        t_sec     = int(session.get("total_time",0))
        errors    = int(session.get("errors",0))
        hints     = int(session.get("hints_used",0))
        moves     = int(session.get("moves",0))
        completed = session.get("completed",False)
        near_miss = int(session.get("near_miss",0))
        guessing  = int(session.get("guessing",0))
        t_str     = f"{t_sec//60:02d}:{t_sec%60:02d}"
        grid_s    = f"{N}\u00d7{N}"
        diff_col  = MAGENTA if diff=="Hard" else (GREEN if diff=="Easy" else CYAN)

        ml        = ml_summary or {}
        ml_conf   = float(ml.get("ml_confidence",0) or 0)
        pred_sc   = ml.get("predicted_next_score")
        anomaly   = ml.get("anomaly_status","unknown")
        feat      = ml.get("features",{})
        err_rate  = float(feat.get("error_rate", errors/max(moves,1)))
        hint_rate = float(feat.get("hint_rate",  hints /max(moves,1)))

        hist_scores = []
        if all_sessions:
            done = [s for s in all_sessions if s.get("completed")][-12:]
            hist_scores = [int(s.get("score",0) or 0) for s in done]
        if not hist_scores:
            hist_scores = [score]
        total_sess = len(all_sessions or [])

        ach_list = achievements or []
        rng = _rnd.Random(username+str(score)+str(t_sec))

        # Fungsi bantu ini memecah logika save score card agar langkah kecilnya lebih rapi dan mudah dipakai ulang di bagian ini.
        def _f(nms,sz):
            for n in (nms or []):
                try: return _PilFont.truetype(n,sz)
                except: pass
            return _PilFont.load_default()
        REG  = ["segoeui.ttf","arial.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]
        BOLD = ["segoeuib.ttf","arialbd.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"]
        F = {
            "xl":    _f(BOLD,52), "title": _f(BOLD,30), "h1": _f(BOLD,22),
            "h2":    _f(BOLD,17), "h3":    _f(BOLD,13), "body":_f(REG,13),
            "sm":    _f(REG,12),  "xs":    _f(REG,10),
        }

        HDR_H    = 272
        TILE_H   = 58; TILE_GAP = 7
        TILE_W   = 183
        STAT_W   = 390
        RIGHT_X  = STAT_W+22
        RIGHT_W  = W-RIGHT_X-16
        PAD      = 16

        ACC_BAR_H = 52
        RADAR_H   = 270
        LEFT_H    = 3*(TILE_H+TILE_GAP)+ACC_BAR_H+RADAR_H+12

        DONUT_AREA_H = 160
        BAR_CHART_H  = 195
        ML_BAR_H     = 44
        RIGHT_H      = DONUT_AREA_H+BAR_CHART_H+ML_BAR_H+8

        CELL_S_VIZ   = min(18, 144//N)
        GRID_VIZ_H   = max(160, 48 + CELL_S_VIZ*N + 24)
        MAX_ACH_PER_ROW = 5
        n_ach    = len(ach_list)
        ach_rows = max(1, (n_ach+MAX_ACH_PER_ROW-1)//MAX_ACH_PER_ROW) if n_ach else 1
        BADGE_ROW_H = 32
        ACH_H    = 44+ach_rows*BADGE_ROW_H+10 if n_ach else 0
        AI_H     = 82
        PRED_H   = 0
        FOOT_H   = 72

        BODY_Y   = HDR_H+8
        BODY_H   = max(LEFT_H, RIGHT_H) + GRID_VIZ_H+ACH_H+AI_H+PRED_H+FOOT_H+80
        H        = HDR_H+BODY_H+FOOT_H

        img  = _PilImage.new("RGBA", (W,H), BG+(255,))
        draw = _PilDraw.Draw(img)

        # Fungsi bantu ini memecah logika save score card agar langkah kecilnya lebih rapi dan mudah dipakai ulang di bagian ini.
        def _fill(c):   return c+(255,)
        # Fungsi bantu ini memecah logika save score card agar langkah kecilnya lebih rapi dan mudah dipakai ulang di bagian ini.
        def _fillA(c,a):return c+(a,)

        blob_layer = _PilImage.new("RGBA",(W,H),(0,0,0,0))
        bd         = _PilDraw.Draw(blob_layer)
        blobs = [
            (ACC,    -60, -40, 340, 28),
            (PURPLE,  W+40, H//3, 280, 20),
            (CYAN,    W//2,-80,  220, 14),
            (MAGENTA,-40,  H-60, 200, 11),
        ]
        for col,bx,by,br2,ba in blobs:
            for dr2 in range(br2,0,-8):
                a2 = int(ba*(1-dr2/br2)**2)
                bd.ellipse([(bx-dr2,by-dr2),(bx+dr2,by+dr2)],
                           fill=col+(a2,))
        img = _PilImage.alpha_composite(img, blob_layer)
        draw = _PilDraw.Draw(img)

        for _ in range(300):
            dx,dy = rng.randint(0,W), rng.randint(0,H)
            dr2   = rng.randint(1,2)
            dc    = rng.choice([ACC,CYAN,PURPLE,GOLD])
            da    = rng.randint(5,18)
            draw.ellipse([(dx-dr2,dy-dr2),(dx+dr2,dy+dr2)], fill=dc+(da,))

        for y in range(HDR_H):
            t = y/HDR_H
            rc = _blend(BG2, _blend(BG2,ACC,0.30), 1-(1-t)**0.5)
            draw.rectangle([(0,y),(W,y+1)], fill=_fill(rc))

        for i in range(7):
            lx = W-24-i*24
            la = 18-i*2
            draw.line([(lx,0),(lx-180,HDR_H)], fill=ACC+(la,), width=1)

        draw.rectangle([(0,0),(W,7)],            fill=_fill(ACC))
        draw.rectangle([(0,7),(int(W*0.56),12)], fill=_fill(_lighten(ACC,0.3)))
        draw.rectangle([(int(W*0.56),7),(W,12)], fill=_fill(ACC2))
        draw.ellipse([(int(W*0.56)-6,3),(int(W*0.56)+6,15)], fill=_fill(WHITE))

        AV_R=56; AV_CX,AV_CY=108,140
        for gr in range(22,0,-1):
            ga=int(55*(1-gr/22)**1.5)
            draw.ellipse([(AV_CX-AV_R-gr,AV_CY-AV_R-gr),(AV_CX+AV_R+gr,AV_CY+AV_R+gr)],
                         fill=ACC+(ga,))
        draw.ellipse([(AV_CX-AV_R-5,AV_CY-AV_R-5),(AV_CX+AV_R+5,AV_CY+AV_R+5)],
                     fill=_fill(ACC2))
        draw.ellipse([(AV_CX-AV_R-1,AV_CY-AV_R-1),(AV_CX+AV_R+1,AV_CY+AV_R+1)],
                     fill=_fill(CARD))
        for yr in range(-AV_R,AV_R):
            hlf = int((AV_R**2-yr**2)**0.5)
            t   = (yr+AV_R)/(2*AV_R)
            rc  = _lerp(_lighten(ACC,0.30), _darken(ACC,0.15), t)
            draw.rectangle([(AV_CX-hlf,AV_CY+yr),(AV_CX+hlf,AV_CY+yr+1)], fill=_fill(rc))
        initials = "".join(w[0].upper() for w in username.split()[:2]) or username[:2].upper()
        draw.text((AV_CX,AV_CY), initials, font=F["h1"], fill=_fill(WHITE), anchor="mm")

        draw.text((208,92),  f"@{username}",  font=F["title"], fill=_fill(TXT),  anchor="lm")
        draw.text((208,124), player_type,     font=F["h2"],    fill=_fill(ACC),  anchor="lm")

        st_txt  = "COMPLETED" if completed else "INCOMPLETE"
        st_col  = GREEN if completed else ORANGE
        spw     = int(draw.textlength(f"  {st_txt}  ",font=F["sm"]))+2
        pill_bg = _fill(_blend(CARD, st_col, 0.28))
        pill_ol = _fill(_blend(st_col, WHITE, 0.15))
        draw.rounded_rectangle([(208,140),(208+spw,162)], radius=8,
                                fill=pill_bg, outline=pill_ol, width=2)
        draw.text((208+spw//2,151), f"{st_txt}",
                  font=F["sm"], fill=_fill(_lighten(st_col,0.2)), anchor="mm")

        SC_X = W-40
        draw.text((SC_X,78),  str(score),    font=F["xl"], fill=_fill(ACC2), anchor="rm")
        draw.text((SC_X,128), "SKOR AKHIR",  font=F["sm"], fill=_fill(DIM),  anchor="rm")
        if pred_sc is not None:
            draw.text((SC_X,150), f">> Prediksi: {int(pred_sc)}",
                      font=F["xs"], fill=_fill(PURPLE), anchor="rm")
        draw.text((SC_X,172), f"{grid_s}  \u2022  {diff}",
                  font=F["h3"], fill=_fill(diff_col), anchor="rm")
        draw.text((SC_X,195), t_str,
                  font=F["sm"], fill=_fill(DIM2),     anchor="rm")

        for x in range(W):
            wy = int(3*_math.sin(x*_math.pi*2/W))
            draw.rectangle([(x,HDR_H+wy-2),(x+1,HDR_H+wy+1)], fill=ACC+(90,))
        draw.rectangle([(0,HDR_H+3),(W,HDR_H+4)], fill=_fill(BORDER))

        # Fungsi bantu ini memecah logika save score card agar langkah kecilnya lebih rapi dan mudah dipakai ulang di bagian ini.
        def _card(x,y,w,h,fill=None,outline=None,radius=11):
            fill    = _fill(fill    or CARD)
            outline = _fill(outline or BORDER)
            draw.rounded_rectangle([(x,y),(x+w,y+h)], radius=radius,
                                   fill=fill, outline=outline, width=1)

        # Fungsi bantu ini memecah logika save score card agar langkah kecilnya lebih rapi dan mudah dipakai ulang di bagian ini.
        def _tile(x,y,w,h,icon,lbl,val,col):
            _card(x,y,w,h,fill=CARD2)
            draw.rounded_rectangle([(x+2,y+10),(x+5,y+h-10)], radius=2, fill=_fill(col))
            ic_cx,ic_cy = x+21,y+h//2-7
            draw.ellipse([(ic_cx-11,ic_cy-11),(ic_cx+11,ic_cy+11)], fill=col+(45,))
            draw.text((ic_cx,ic_cy), icon, font=F["xs"], fill=_fill(col), anchor="mm")
            draw.text((x+38,y+13),   lbl,  font=F["xs"], fill=_fill(DIM))
            val_str = str(val)[:8]
            draw.text((x+w-10,y+h//2+4), val_str, font=F["h2"],
                      fill=_fill(col), anchor="rm")

        TW=184; TH=TILE_H; GAP=TILE_GAP
        tiles = [
            ("T","WAKTU",   t_str,      CYAN),
            ("X","ERROR",   str(errors), MAGENTA if errors>0 else GREEN),
            ("?","HINT",    str(hints),  PURPLE  if hints>0  else GREEN),
            ("M","GERAKAN", str(moves),  ACC),
            ("L","LEVEL",   diff[:8],    diff_col),
            ("#","GRID",    grid_s,      GOLD),
        ]
        for i,(icon,lbl,val,col) in enumerate(tiles):
            ci=i%2; ri=i//2
            tx = PAD+ci*(TW+GAP)
            ty = BODY_Y+ri*(TH+GAP)
            _tile(tx,ty,TW,TH,icon,lbl,val,col)

        accuracy = max(0,min(100,int(100-err_rate*100)))
        acc_col  = GREEN if accuracy>=80 else (GOLD if accuracy>=55 else MAGENTA)
        TL_BTM   = BODY_Y+3*(TH+GAP)
        _card(PAD,TL_BTM,STAT_W-PAD,ACC_BAR_H,fill=CARD2)
        draw.rounded_rectangle([(PAD+2,TL_BTM+10),(PAD+5,TL_BTM+ACC_BAR_H-10)],
                                radius=2, fill=_fill(acc_col))
        draw.text((PAD+12,TL_BTM+9),  "AKURASI", font=F["h3"], fill=_fill(TXT))
        draw.text((STAT_W-22,TL_BTM+ACC_BAR_H//2+4), f"{accuracy}%",
                  font=F["h2"], fill=_fill(_lighten(acc_col,0.15)), anchor="rm")
        BX0,BX1 = PAD+14,STAT_W-14
        BY = TL_BTM+40
        draw.rounded_rectangle([(BX0,BY),(BX1,BY+8)], radius=4, fill=acc_col+(40,))
        fx = BX0+int((BX1-BX0)*accuracy/100)
        if fx>BX0:
            for xi in range(BX0,fx):
                t3=(xi-BX0)/(BX1-BX0)
                draw.rectangle([(xi,BY),(xi+1,BY+8)],
                               fill=_fill(_lerp(_lighten(acc_col,0.2),acc_col,t3)))
            draw.rounded_rectangle([(BX0,BY),(fx,BY+8)],radius=4,outline=_fill(acc_col),width=1)

        RC_Y0   = TL_BTM+ACC_BAR_H+8
        RC_CARD_H = RADAR_H
        _card(PAD,RC_Y0,STAT_W-PAD,RC_CARD_H)
        draw.text((PAD+12,RC_Y0+10), "RADAR PERFORMA",       font=F["h3"], fill=_fill(TXT))
        draw.rectangle([(PAD+8,RC_Y0+30),(STAT_W-PAD+8-10,RC_Y0+31)], fill=_fill(BORDER2))

        RCX    = PAD+(STAT_W-PAD)//2
        RCY    = RC_Y0+RC_CARD_H//2+12
        R_MAX  = 76
        LABEL_OFFSET = 28
        RDIMS  = ["Akurasi","Kecepatan","Konsistensi","Fokus","Efisiensi","Gaya"]
        consist = min(100,max(0, len([s for s in (all_sessions or []) if s.get("completed")])*18))
        effic   = min(100,int(score/12))
        style_v = min(100,int(ml_conf))
        speed_v = max(0,min(100,int(100-(t_sec/max(N*N,1))*3)))
        focus_v = max(0,min(100,int(100-hint_rate*200)))
        R_VALS  = [accuracy,speed_v,consist,focus_v,effic,style_v]

        for lvl in [0.25,0.5,0.75,1.0]:
            pts = []
            for k in range(6):
                ang=_math.radians(k*60-90)
                pts.append((RCX+R_MAX*lvl*_math.cos(ang), RCY+R_MAX*lvl*_math.sin(ang)))
            lc = _blend(CARD,BORDER2,lvl)
            for i in range(6):
                draw.line([pts[i],pts[(i+1)%6]], fill=_fill(lc), width=1)
        for k in range(6):
            ang=_math.radians(k*60-90)
            draw.line([(RCX,RCY),(RCX+R_MAX*_math.cos(ang),RCY+R_MAX*_math.sin(ang))],
                      fill=_fill(BORDER2), width=1)

        rpts = []
        for k,v in enumerate(R_VALS):
            ang=_math.radians(k*60-90)
            r  =R_MAX*v/100
            rpts.append((RCX+r*_math.cos(ang), RCY+r*_math.sin(ang)))

        radar_ov = _PilImage.new("RGBA",(W,H),(0,0,0,0))
        rod = _PilDraw.Draw(radar_ov)
        rod.polygon(rpts, fill=ACC+(85,))
        img = _PilImage.alpha_composite(img, radar_ov)
        draw = _PilDraw.Draw(img)

        for i in range(6):
            draw.line([rpts[i],rpts[(i+1)%6]], fill=_fill(_lighten(ACC,0.3)), width=2)
        for pt in rpts:
            draw.ellipse([(pt[0]-5,pt[1]-5),(pt[0]+5,pt[1]+5)], fill=_fill(WHITE))
            draw.ellipse([(pt[0]-3,pt[1]-3),(pt[0]+3,pt[1]+3)], fill=_fill(ACC))

        for k,(lbl,v) in enumerate(zip(RDIMS,R_VALS)):
            ang=_math.radians(k*60-90)
            lx = RCX+(R_MAX+LABEL_OFFSET)*_math.cos(ang)
            ly = RCY+(R_MAX+LABEL_OFFSET)*_math.sin(ang)
            lx = max(PAD+18, min(STAT_W-PAD-18, lx))
            ly = max(RC_Y0+30, min(RC_Y0+RC_CARD_H-12, ly))
            draw.text((lx,ly-7),  lbl,    font=F["xs"], fill=_fill(DIM2),  anchor="mm")
            draw.text((lx,ly+7),  f"{v}%",font=F["xs"], fill=_fill(_lighten(ACC,0.3)), anchor="mm")

        donuts = [
            ("AKURASI",   accuracy, GREEN   if accuracy>=80  else (GOLD if accuracy>=55 else MAGENTA)),
            ("KECEPATAN", speed_v,  CYAN    if speed_v>=70   else (GOLD if speed_v>=40  else ORANGE)),
            ("FOKUS",     focus_v,  PURPLE  if focus_v>=70   else (GOLD if focus_v>=40  else MAGENTA)),
        ]

        # Fungsi bantu ini memecah logika save score card agar langkah kecilnya lebih rapi dan mudah dipakai ulang di bagian ini.
        def _donut(cx,cy,ro,ri,pct,col,label,val):
            mid_r = (ro + ri) / 2

            draw.ellipse([(cx-ro,cy-ro),(cx+ro,cy+ro)], fill=_fill(BORDER2))
            draw.ellipse([(cx-ri,cy-ri),(cx+ri,cy+ri)], fill=_fill(CARD))

            if pct >= 100:
                draw.ellipse([(cx-ro,cy-ro),(cx+ro,cy+ro)], fill=_fill(col))
                draw.ellipse([(cx-ri,cy-ri),(cx+ri,cy+ri)], fill=_fill(CARD))
            elif pct > 0:
                ang_end = -90 + 360*pct/100
                draw.pieslice([(cx-ro,cy-ro),(cx+ro,cy+ro)],
                              start=-90, end=ang_end, fill=_fill(col))
                draw.ellipse([(cx-ri,cy-ri),(cx+ri,cy+ri)], fill=_fill(CARD))

                ang_r = _math.radians(ang_end)
                ex = cx + mid_r * _math.cos(ang_r)
                ey = cy + mid_r * _math.sin(ang_r)
                draw.ellipse([(ex-5, ey-5), (ex+5, ey+5)], fill=_fill(WHITE))

                sx = cx + mid_r * _math.cos(_math.radians(-90))
                sy = cy + mid_r * _math.sin(_math.radians(-90))
                draw.ellipse([(sx-4, sy-4), (sx+4, sy+4)], fill=_fill(col))

            for dg in range(4,0,-1):
                draw.ellipse([(cx-ro-dg,cy-ro-dg),(cx+ro+dg,cy+ro+dg)],
                             outline=col+(int(40*dg/4),), width=1)

            draw.text((cx,cy-8),  f"{val}%", font=F["h2"], fill=_fill(_lighten(col,0.25)), anchor="mm")
            draw.text((cx,cy+11), label,     font=F["xs"], fill=_fill(DIM2), anchor="mm")

        DR=50; DI=33
        D_Y = BODY_Y+44
        DSP = RIGHT_W//3
        draw.text((RIGHT_X+8, BODY_Y+9), "STATISTIK PERFORMA",
                  font=F["h3"], fill=_fill(TXT))
        draw.rectangle([(RIGHT_X+4,BODY_Y+28),(RIGHT_X+RIGHT_W-4,BODY_Y+29)],
                       fill=_fill(BORDER2))
        for di,(lbl,val,col) in enumerate(donuts):
            dcx = RIGHT_X+DSP*di+DSP//2
            _donut(dcx, D_Y+DR+4, DR, DI, val, col, lbl, val)

        CH_Y0 = BODY_Y+DONUT_AREA_H
        CH_H  = BAR_CHART_H
        CHX0  = RIGHT_X; CHX1 = RIGHT_X+RIGHT_W
        CHY0  = CH_Y0;   CHY1 = CH_Y0+CH_H
        TITLE_H = 42
        _card(CHX0,CHY0,RIGHT_W,CH_H+ML_BAR_H+6)
        draw.text((CHX0+12,CHY0+10), "RIWAYAT SKOR",
                  font=F["h3"], fill=_fill(TXT))
        draw.rectangle([(CHX0+8,CHY0+TITLE_H-4),(CHX1-8,CHY0+TITLE_H-3)],
                       fill=_fill(BORDER2))

        max_s = max(hist_scores) if hist_scores else 1
        n_bar = len(hist_scores)
        B_PAD = 14
        BAR_W = max(8,(RIGHT_W-B_PAD*2)//(n_bar)-4)
        LABEL_SPACE = 18
        BAR_AREA_H = CH_H - TITLE_H - 14 - LABEL_SPACE
        for i,sv in enumerate(hist_scores):
            frac    = sv/max(max_s,1)
            is_last = (i==n_bar-1)
            bx      = CHX0+B_PAD+i*(BAR_W+4)
            by1     = CHY1-4
            by0     = max(CHY0+TITLE_H+4+LABEL_SPACE, by1-max(6,int(BAR_AREA_H*frac)))
            bar_col = ACC if is_last else _blend(CARD2,ACC,0.55)
            bar_top = _lighten(bar_col,0.25)
            for yi in range(by0,by1):
                t2 = (yi-by0)/max(by1-by0,1)
                draw.rectangle([(bx,yi),(bx+BAR_W,yi+1)],
                               fill=_fill(_lerp(bar_top,bar_col,t2)))
            draw.rounded_rectangle([(bx,by0),(bx+BAR_W,by1)],
                                   radius=3, outline=_fill(_lighten(bar_col,0.15)), width=1)
            if is_last:
                for gl in range(3,0,-1):
                    draw.rounded_rectangle([(bx-gl,by0-gl),(bx+BAR_W+gl,by1)],
                                           radius=3+gl, outline=ACC+(int(35*gl/3),), width=1)
            if frac > 0.05:
                lbl_y   = by0 - 9
                lbl_col = WHITE if is_last else DIM2
                draw.text((bx+BAR_W//2, lbl_y), str(sv),
                          font=F["xs"], fill=_fill(lbl_col), anchor="mm")

        draw.rectangle([(CHX0+B_PAD,CHY1-5),(CHX1-B_PAD,CHY1-4)], fill=_fill(BORDER2))

        ML_Y  = CHY1+4
        ML_H  = ML_BAR_H
        conf_col = GREEN if ml_conf>=70 else (GOLD if ml_conf>=40 else MAGENTA)
        draw.text((CHX0+12,ML_Y+6), "ML CONFIDENCE",
                  font=F["h3"], fill=_fill(TXT))
        draw.text((CHX1-10,ML_Y+7), f"{int(ml_conf)}%  |  {anomaly.title()}",
                  font=F["xs"], fill=_fill(conf_col), anchor="rm")
        CW0,CW1 = CHX0+10,CHX1-10
        CY2 = ML_Y+24
        draw.rounded_rectangle([(CW0,CY2),(CW1,CY2+8)], radius=4, fill=conf_col+(40,))
        cx_end = CW0+int((CW1-CW0)*ml_conf/100)
        if cx_end>CW0:
            for xi in range(CW0,cx_end):
                t4 = (xi-CW0)/(CW1-CW0)
                draw.rectangle([(xi,CY2),(xi+1,CY2+8)],
                               fill=_fill(_lerp(_lighten(conf_col,0.3),conf_col,t4)))

        GRID_Y = BODY_Y+max(LEFT_H,RIGHT_H)+8
        _card(PAD,GRID_Y,W-PAD*2,GRID_VIZ_H)
        draw.text((PAD+12,GRID_Y+10), "VISUALISASI PUZZLE",
                  font=F["h3"], fill=_fill(TXT))
        draw.rectangle([(PAD+8,GRID_Y+29),(W-PAD*2-8,GRID_Y+30)], fill=_fill(BORDER2))

        GX0   = PAD+18; GY0 = GRID_Y+44
        CELL_S= CELL_S_VIZ
        GW    = CELL_S*N; GH = CELL_S*N
        GX1   = GX0+GW;   GY1 = GY0+GH

        draw.rectangle([(GX0,GY0),(GX1,GY1)], fill=_fill(_darken(CARD2, 0.15)))
        total_cells = N*N
        filled_ratio = min(1.0, moves/(total_cells*1.5))
        seeded = _rnd.Random(username+str(gs)+str(score))

        COL_FILLED  = _blend(BG, GOLD, 0.72)
        COL_INITIAL = _blend(BG, ACC,  0.78)
        COL_EMPTY   = _blend(BG, CARD2, 0.60)

        for cr in range(N):
            for cc in range(N):
                roll = seeded.random()
                if roll < filled_ratio*0.85:
                    base = COL_FILLED
                    bright = _lighten(base, 0.30)
                elif roll < 0.92:
                    base = COL_INITIAL
                    bright = _lighten(base, 0.30)
                else:
                    base = COL_EMPTY
                    bright = _lighten(base, 0.10)
                cx2,cy2 = GX0+cc*CELL_S, GY0+cr*CELL_S
                cs = CELL_S - 1
                for gy in range(cs):
                    tg = gy / max(cs-1, 1)
                    rc2 = _lerp(bright, base, tg**0.6)
                    draw.rectangle([(cx2+1, cy2+1+gy), (cx2+cs, cy2+1+gy+1)], fill=_fill(rc2))
                draw.line([(cx2+1, cy2+1), (cx2+cs-1, cy2+1)],
                          fill=bright+(160,), width=1)
                draw.line([(cx2+1, cy2+1), (cx2+1, cy2+cs-1)],
                          fill=bright+(80,),  width=1)

        for ci in range(N+1):
            lw  = 2 if ci % gs == 0 else 1
            lcol = _lighten(BORDER2, 0.25) if ci % gs == 0 else BORDER2
            draw.line([(GX0+ci*CELL_S, GY0), (GX0+ci*CELL_S, GY1)], fill=_fill(lcol), width=lw)
            draw.line([(GX0, GY0+ci*CELL_S), (GX1, GY0+ci*CELL_S)], fill=_fill(lcol), width=lw)

        for gd in range(3, 0, -1):
            draw.rectangle([(GX0-gd, GY0-gd), (GX1+gd, GY1+gd)],
                           outline=ACC+(int(40*gd/3),), width=1)

        QX = GX1+32; QY = GY0
        qs_data = [
            (f"Near-miss: {near_miss}",    ORANGE),
            (f"Guessing:  {guessing}",     MAGENTA),
            (f"Sesi total: {total_sess}",  CYAN),
            (f"Moves/menit: {int(moves/(max(t_sec,1)/60)*10)/10}", PURPLE),
        ]
        for qi,(qtxt,qcol) in enumerate(qs_data):
            col2i = qi%2; row2i = qi//2
            qx2 = QX+col2i*180
            qy2 = QY+row2i*22
            draw.text((qx2,qy2), qtxt, font=F["xs"], fill=_fill(qcol))

        LEG_X = QX; LEG_Y = QY+50
        for lc2,ltxt in [(COL_FILLED, "Sel terisi"),
                          (COL_INITIAL,"Angka awal"),
                          (COL_EMPTY,  "Kosong")]:
            draw.rounded_rectangle([(LEG_X,LEG_Y),(LEG_X+12,LEG_Y+12)],
                                   radius=2, fill=_fill(lc2), outline=_fill(BORDER2))
            draw.text((LEG_X+18,LEG_Y+6), ltxt, font=F["xs"], fill=_fill(DIM2), anchor="lm")
            LEG_X += 100

        ACH_Y = GRID_Y+GRID_VIZ_H+8

        if ach_list:
            ach_card_h = 44+ach_rows*BADGE_ROW_H+10
            _card(PAD,ACH_Y,W-PAD*2,ach_card_h)

            draw.text((PAD+12,ACH_Y+10), "ACHIEVEMENT UNLOCKED",
                      font=F["h3"], fill=_fill(GOLD))
            draw.text((W-PAD-12,ACH_Y+10), f"{n_ach} badge",
                      font=F["sm"], fill=_fill(DIM2), anchor="rm")
            draw.rectangle([(PAD+8,ACH_Y+29),(W-PAD*2-8,ACH_Y+30)], fill=_fill(BORDER2))

            BADGE_PAD  = 12
            avail_w    = W-PAD*2-BADGE_PAD*2
            badge_w    = avail_w//MAX_ACH_PER_ROW - 8
            badge_h    = BADGE_ROW_H - 6

            for ai,aid in enumerate(ach_list):
                info  = ACHIEVEMENTS.get(aid, {"emoji":"\u2b50","nama":aid,"warna":"#FFD700"})
                emoji = info.get("emoji","\u2b50")
                nama  = info.get("nama", aid)
                ac2   = _hex(info.get("warna","#FFD700"))
                if sum(ac2)/3 < 130: ac2 = _lighten(ac2, 0.5)
                ac2_l = _lighten(ac2, 0.30)
                badge_bg = _fill(_blend(CARD2, ac2, 0.32))

                col_i = ai % MAX_ACH_PER_ROW
                row_i = ai // MAX_ACH_PER_ROW
                bx2   = PAD+BADGE_PAD+col_i*(badge_w+8)
                by2   = ACH_Y+42+row_i*(badge_h+6)

                draw.rounded_rectangle([(bx2,by2),(bx2+badge_w,by2+badge_h)],
                                       radius=7, fill=badge_bg, outline=_fill(ac2_l), width=1)
                ind_x = bx2+8; ind_cy = by2+badge_h//2
                draw.rounded_rectangle([(ind_x,ind_cy-5),(ind_x+10,ind_cy+5)],
                                       radius=3, fill=_fill(ac2))
                badge_txt = nama
                max_w = badge_w-28
                while draw.textlength(badge_txt, font=F["sm"]) > max_w and len(badge_txt)>3:
                    badge_txt = badge_txt[:-1]+"..."
                draw.text((bx2+badge_w//2+6, by2+badge_h//2),
                          badge_txt, font=F["sm"], fill=_fill(WHITE), anchor="mm")
            ACH_Y = ACH_Y+ach_card_h+8
        else:
            _card(PAD,ACH_Y,W-PAD*2,36,fill=CARD2)
            draw.text((W//2,ACH_Y+18), "Belum ada achievement - mainkan terus untuk membuka badge!",
                      font=F["xs"], fill=_fill(DIM), anchor="mm")
            ACH_Y = ACH_Y+44

        AI_Y2 = ACH_Y
        _card(PAD,AI_Y2,W-PAD*2,AI_H)
        for yi in range(AI_H-2):
            t5 = yi/(AI_H-2)
            sc5 = _lerp(PURPLE,CYAN,t5)
            draw.rectangle([(PAD+2,AI_Y2+1+yi),(PAD+6,AI_Y2+2+yi)], fill=_fill(sc5))
        draw.text((PAD+12,AI_Y2+10), "AI INSIGHT  -  "+player_type,
                  font=F["h3"], fill=_fill(_lighten(PURPLE,0.5)))
        draw.rectangle([(PAD+8,AI_Y2+28),(W-PAD*2-8,AI_Y2+29)], fill=_fill(_blend(CARD,PURPLE,0.4)))
        msg = (ai_message or "Terus berlatih dan raih skor terbaik!")[:200]
        words = msg.split(); lines2,cur2 = [],""
        for w in words:
            test2 = cur2+(" " if cur2 else "")+w
            if draw.textlength(test2,font=F["sm"]) > W-PAD*2-28:
                if cur2: lines2.append(cur2)
                cur2 = w
            else: cur2 = test2
        if cur2: lines2.append(cur2)
        for li,ln in enumerate(lines2[:2]):
            draw.text((PAD+14,AI_Y2+40+li*18), ln, font=F["sm"], fill=_fill(TXT))

        FOOT_START = AI_Y2 + AI_H + 10

        FOOT_Y = FOOT_START
        ACTUAL_H = FOOT_Y+FOOT_H

        img = img.crop((0,0,W,ACTUAL_H))
        draw = _PilDraw.Draw(img)

        for x in range(W):
            wy2 = int(2*_math.sin(x*_math.pi*3/W+0.8))
            draw.rectangle([(x,FOOT_Y+wy2),(x+1,FOOT_Y+wy2+2)], fill=PURPLE+(70,))
        draw.rectangle([(0,FOOT_Y+3),(W,FOOT_Y+4)], fill=_fill(BORDER))

        for y in range(FOOT_Y+4,ACTUAL_H):
            t6 = (y-FOOT_Y-4)/FOOT_H
            draw.rectangle([(0,y),(W,y+1)],
                           fill=_fill(_lerp(BG2,_darken(BG,0.25),t6)))

        BOX_S   = 58
        QX2     = W - BOX_S - 12
        QY2     = FOOT_Y + (FOOT_H - 8 - BOX_S) // 2
        _logo_path_sc = IMAGE_LOGO
        _logo_pil_sc  = None
        try:
            _logo_pil_sc = _PilImage.open(_logo_path_sc).convert("RGBA")
        except Exception:
            pass

        if _logo_pil_sc is not None:
            draw.rounded_rectangle(
                [(QX2-2, QY2-2), (QX2+BOX_S+2, QY2+BOX_S+2)],
                radius=12, outline=ACC+(45,), width=1)

            _bg_logo = _blend(CARD2, ACC, 0.18)
            draw.rounded_rectangle([(QX2, QY2), (QX2+BOX_S, QY2+BOX_S)],
                                    radius=10, fill=_fill(_bg_logo))

            _stripe_h = 4
            draw.rounded_rectangle(
                [(QX2+2, QY2+1), (QX2+BOX_S-2, QY2+_stripe_h+1)],
                radius=6, fill=_fill(ACC))

            draw.rounded_rectangle([(QX2, QY2), (QX2+BOX_S, QY2+BOX_S)],
                                    radius=10, outline=_fill(ACC), width=2)

            _inner_border_col = _lighten(BORDER2, 0.25)
            draw.rounded_rectangle(
                [(QX2+3, QY2+3), (QX2+BOX_S-3, QY2+BOX_S-3)],
                radius=7, outline=_fill(_inner_border_col), width=1)

            _LP       = 11
            _fit      = BOX_S - _LP * 2
            _lw, _lh  = _logo_pil_sc.size
            _scale    = min(_fit / max(_lw, 1), _fit / max(_lh, 1))
            _nw       = max(1, int(_lw * _scale))
            _nh       = max(1, int(_lh * _scale))
            _logo_rsz = _logo_pil_sc.resize((_nw, _nh), _PilImage.LANCZOS)
            _lx       = QX2 + (BOX_S - _nw) // 2
            _ly       = QY2 + _stripe_h + ((BOX_S - _stripe_h - _nh) // 2) + 1
            img.paste(_logo_rsz, (_lx, _ly), _logo_rsz)
            draw = _PilDraw.Draw(img)

        else:
            draw.rounded_rectangle(
                [(QX2-2, QY2-2), (QX2+BOX_S+2, QY2+BOX_S+2)],
                radius=10, outline=ACC+(45,), width=1)
            draw.rounded_rectangle([(QX2, QY2), (QX2+BOX_S, QY2+BOX_S)],
                                    radius=8, fill=_fill(CARD2), outline=_fill(ACC), width=2)
            draw.text((QX2+BOX_S//2, QY2+BOX_S//2), "AI",
                      font=F["h2"], fill=_fill(ACC), anchor="mm")

        draw.text((PAD+4,FOOT_Y+18), "Sudoku AI", font=F["h2"], fill=_fill(TXT))
        draw.text((PAD+4,FOOT_Y+42), "Powered by Machine Learning",
                  font=F["xs"], fill=_fill(DIM))
        date_str = _dt.datetime.now().strftime("%d %b %Y  %H:%M")
        draw.text((QX2-14,FOOT_Y+50), date_str, font=F["xs"], fill=_fill(DIM), anchor="rm")

        draw.rectangle([(0,ACTUAL_H-8),(W//3,ACTUAL_H)],      fill=_fill(ACC))
        draw.rectangle([(W//3,ACTUAL_H-8),(2*W//3,ACTUAL_H)], fill=_fill(PURPLE))
        draw.rectangle([(2*W//3,ACTUAL_H-8),(W,ACTUAL_H)],    fill=_fill(ACC2))
        draw.ellipse([(W//3-5,ACTUAL_H-13),(W//3+5,ACTUAL_H-3)],   fill=_fill(WHITE))
        draw.ellipse([(2*W//3-5,ACTUAL_H-13),(2*W//3+5,ACTUAL_H-3)], fill=_fill(WHITE))

        final = _PilImage.new("RGB",(W,ACTUAL_H), BG)
        final.paste(img.convert("RGB"), (0,0))

        ts    = int(time.time())
        fname = f"SudokuAI_{username}_{ts}.png"
        _card_dir = CARD_DIR
        os.makedirs(_card_dir, exist_ok=True)
        path = os.path.join(_card_dir, fname)
        final.save(path, "PNG")
        return path

    except Exception as e:
        print("Score card error:", e)
        traceback.print_exc()
    return None

# EASTER EGG - Title Click (7x)
# easter egg aktif jika logo diklik 7 kali dalam 4 detik di LoginScreen
# Jika easter_egg.mp4 ada di Assets → video diputar via cv2 + PIL.
# Jika easter_egg.mp3 ada di Assets → audio diputar via pygame.
# Fallback: animasi partikel kembang api bawaan (tanpa dependensi tambahan).

_EASTER_EGG_AUDIO = AUDIO_EASTER_EGG  # alias ke konstanta bagian 2

# Global guard - satu instance saja agar tidak tumpang-tindih
_EASTER_EGG_ACTIVE = False

# Referensi ke instance SudokuApp - diset saat SudokuApp.__init__ berjalan.
# Dipakai oleh EasterEggOverlay._close() untuk me-restore musik tanpa callback.
_APP_INSTANCE = None

class EasterEggOverlay(tk.Toplevel):
    """
    Full-screen overlay untuk Easter Egg.

    Alur:
    1. Tampil sebagai Toplevel transparan di atas semua widget.
    2. Jika cv2 + PIL tersedia dan easter_egg.mp4 ada  → putar video.
    3. Jika tidak  → animasi partikel kembang api + teks.
    4. Audio: putar easter_egg.mp3 jika ada, fallback _SFX_WIN.
    5. Tutup via ESC, klik, atau setelah video selesai.
    """

    _PARTICLE_COLORS = [
        "#FFD700", "#FF7B7B", "#58A6FF", "#7EE787",
        "#BC8CFF", "#FF7EDB", "#F0883E", "#FFFFFF",
    ]

    # Menginisialisasi objek EasterEggOverlay dan menyiapkan state awal, referensi penting, serta elemen yang dibutuhkan sebelum layar dipakai.
    def __init__(self, master, on_close_callback=None):
        global _EASTER_EGG_ACTIVE
        super().__init__(master)
        _EASTER_EGG_ACTIVE = True
        self._on_close_callback = on_close_callback
        self._running   = True
        self._cap       = None
        self._photo_ref = None
        self._frame_job = None
        self._anim_job  = None
        self._tick      = 0
        self._particles = []
        self._stars     = []

        sw = master.winfo_screenwidth()
        sh = master.winfo_screenheight()
        self.geometry(f"{sw}x{sh}+0+0")
        self.overrideredirect(True)
        self.attributes("-topmost", True)
        self.configure(bg=self._egg_bg if hasattr(self, "_egg_bg") else "#000000")

        self._sw = sw
        self._sh = sh

        self._egg_bg = "#000000" if _CURRENT_THEME_NAME == "dark" else C_BG

        self._build_ui()
        self._start_audio()
        self._start_content()

        self.bind("<Escape>", self._close)
        self.bind("<Key>",    self._on_key)
        self.focus_force()

    # UI skeleton
    # Membangun ui pada EasterEggOverlay dan menyiapkan widget supaya state tampilan tetap konsisten.
    def _build_ui(self):
        sw, sh = self._sw, self._sh
        _egg_bg = getattr(self, "_egg_bg", "#000000")

        self.canvas = tk.Canvas(
            self, bg=_egg_bg, highlightthickness=0,
            width=sw, height=sh,
        )
        self.canvas.pack(fill="both", expand=True)

    # Audio
    # Memulai audio pada EasterEggOverlay dan menyalakan mekanisme pendukung yang dibutuhkan.
    def _start_audio(self):
        if not PYGAME_AVAILABLE:
            return
        self._egg_sound_obj     = None
        self._music_was_replaced = False

        try:
            pygame.mixer.music.pause()
        except Exception:
            pass

        if os.path.exists(_EASTER_EGG_AUDIO):
            try:
                snd = pygame.mixer.Sound(_EASTER_EGG_AUDIO)
                snd.set_volume(0.3)
                snd.play()
                self._egg_sound_obj = snd
                return
            except Exception:
                pass
            try:
                pygame.mixer.music.load(_EASTER_EGG_AUDIO)
                pygame.mixer.music.set_volume(0.3)
                pygame.mixer.music.play(loops=0)
                self._music_was_replaced = True
            except Exception:
                pass
        else:
            _play_sfx(_SFX_WIN)

    # Content router
    # Memulai content pada EasterEggOverlay dan menyalakan mekanisme pendukung yang dibutuhkan.
    def _start_content(self):
        cv2_ok = CV2_AVAILABLE
        pil_ok = PIL_AVAILABLE

        if cv2_ok and pil_ok and os.path.exists(VIDEO_EASTER_EGG):
            self._start_video()
        else:
            self._start_particle_show()

    # Video mode
    # Memulai video pada EasterEggOverlay dan menyalakan mekanisme pendukung yang dibutuhkan.
    def _start_video(self):
        self._cap = cv2.VideoCapture(VIDEO_EASTER_EGG)
        fps = self._cap.get(cv2.CAP_PROP_FPS) or 30
        self._frame_delay = max(16, int(1000 / fps))
        self._draw_banner()
        self._play_video_frame()

    # Menangani proses play video frame pada EasterEggOverlay sambil menjaga state internal tetap konsisten.
    def _play_video_frame(self):
        if not self._running or self._cap is None:
            return
        _PIL_Image, _PIL_ImageTk = _PilImage, _PilImageTk

        ret, frame = self._cap.read()
        if not ret:
            self._close()
            return

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame_rgb.shape[:2]
        sw, sh = self._sw, self._sh

        scale = min(sw / w, sh / h)
        nw, nh = int(w * scale), int(h * scale)
        frame_resized = cv2.resize(frame_rgb, (nw, nh))

        img = _PIL_Image.fromarray(frame_resized)
        self._photo_ref = _PIL_ImageTk.PhotoImage(img)

        x0 = (sw - nw) // 2
        y0 = (sh - nh) // 2
        self.canvas.delete("video_frame")
        self.canvas.create_image(x0, y0, anchor="nw",
                                 image=self._photo_ref, tags="video_frame")
        self.canvas.tag_raise("banner")

        if self._running:
            self._frame_job = self.after(self._frame_delay, self._play_video_frame)

    # Menggambar banner pada EasterEggOverlay sesuai state yang sedang aktif.
    def _draw_banner(self):
        sw = self._sw
        _banner_bg = "#0D0D0D" if _CURRENT_THEME_NAME == "dark" else C_SURFACE
        self.canvas.create_rectangle(
            0, 0, sw, 70,
            fill=_banner_bg, outline="", tags="banner",
        )
        seg_w = sw // 4
        cols = ["#BC8CFF", "#58A6FF", "#7EE787", "#F0883E"]
        for i, col in enumerate(cols):
            self.canvas.create_rectangle(
                i * seg_w, 67, (i + 1) * seg_w + 2, 70,
                fill=col, outline="", tags="banner",
            )
        self.canvas.create_text(
            sw // 2, 35,
            text="🥚  YOU'VE FOUND THE EASTER EGG!",
            font=("Segoe UI", 16, "bold"),
            fill="#FFD700", anchor="center", tags="banner",
        )

    # Particle show (fallback)
    # Memulai particle show pada EasterEggOverlay dan menyalakan mekanisme pendukung yang dibutuhkan.
    def _start_particle_show(self):
        sw, sh = self._sw, self._sh

        random.seed(42)
        _star_palette = (
            ["#222244", "#112222", "#221122"]
            if _CURRENT_THEME_NAME == "dark"
            else ["#C8D9F0", "#C8EDD8", "#DDD0F0"]
        )
        self._stars = [
            (random.randint(0, sw), random.randint(0, sh),
             random.uniform(0.5, 2.0),
             random.choice(_star_palette))
            for _ in range(120)
        ]
        random.seed()
        self._anim_tick()

    # Menangani proses anim tick pada EasterEggOverlay sambil menjaga state internal tetap konsisten.
    def _anim_tick(self):
        if not self._running:
            return
        self._tick += 1
        t = self._tick
        sw, sh = self._sw, self._sh
        cx, cy = sw // 2, sh // 2
        self.canvas.delete("fx")

        for sx, sy, sr, sc in self._stars:
            self.canvas.create_oval(
                sx - sr, sy - sr, sx + sr, sy + sr,
                fill=sc, outline="", tags="fx",
            )

        if t % 2 == 0:
            origins = [
                (cx, cy),
                (random.randint(sw // 5, 4 * sw // 5),
                 random.randint(sh // 5, 4 * sh // 5)),
            ]
            origin = origins[t // 6 % len(origins)]
            for _ in range(12):
                speed = random.uniform(4, 15)
                angle = random.uniform(0, 6.2832)
                self._particles.append({
                    "x": origin[0], "y": origin[1],
                    "vx": speed * math.cos(angle),
                    "vy": speed * math.sin(angle) - random.uniform(2, 6),
                    "color": random.choice(self._PARTICLE_COLORS),
                    "life": 1.0,
                    "decay": random.uniform(0.012, 0.022),
                    "size": random.uniform(3, 10),
                    "trail": [],
                })

        alive = []
        for p in self._particles:
            p["x"]  += p["vx"]
            p["y"]  += p["vy"]
            p["vy"] += 0.35
            p["vx"] *= 0.98
            p["life"] -= p["decay"]
            if p["life"] <= 0:
                continue
            alive.append(p)
            sz = max(1, p["size"] * p["life"])
            x, y = int(p["x"]), int(p["y"])
            self.canvas.create_oval(
                x - sz, y - sz, x + sz, y + sz,
                fill=p["color"], outline="", tags="fx",
            )
        self._particles = alive

        glow_cols = self._PARTICLE_COLORS
        gc = glow_cols[(t // 8) % len(glow_cols)]

        _shadow_col = "#111111" if _CURRENT_THEME_NAME == "dark" else "#CCCCCC"
        self.canvas.create_text(
            cx + 2, cy - 78,
            text="🥚  EASTER EGG!",
            font=("Segoe UI", 52, "bold"),
            fill=_shadow_col, anchor="center", tags="fx",
        )
        self.canvas.create_text(
            cx, cy - 80,
            text="🥚  EASTER EGG!",
            font=("Segoe UI", 52, "bold"),
            fill=gc, anchor="center", tags="fx",
        )

        sub_col = glow_cols[(t // 12 + 3) % len(glow_cols)]
        self.canvas.create_text(
            cx, cy,
            text="KAMU MENEMUKAN RAHASIA TERSEMBUNYI!",
            font=("Segoe UI", 26, "bold"),
            fill=sub_col, anchor="center", tags="fx",
        )
        self.canvas.create_text(
            cx, cy + 52,
            text="Sudoku AI  ·  Machine Learning Intelligence System",
            font=("Segoe UI", 15),
            fill="#BC8CFF", anchor="center", tags="fx",
        )

        if (t // 30) % 2 == 0:
            _tip_col = "#555555" if _CURRENT_THEME_NAME == "dark" else C_TEXT_DIM
            self.canvas.create_text(
                cx, sh - 50,
                text="[ Klik di mana saja atau tekan ESC untuk menutup ]",
                font=("Segoe UI", 13),
                fill=_tip_col, anchor="center", tags="fx",
            )

        if self._running:
            self._anim_job = self.after(30, self._anim_tick)

    # Key handler (ESC ditangani binding, selain itu abaikan Modifier)
    # Menangani event key pada EasterEggOverlay dan memperbarui state yang terkait.
    def _on_key(self, event=None):
        if event and event.keysym in (
            "Shift_L", "Shift_R", "Control_L", "Control_R",
            "Alt_L", "Alt_R", "Super_L", "Super_R",
            "Caps_Lock", "Num_Lock",
        ):
            return
        self._close()

    # Cleanup
    # Menangani proses close pada EasterEggOverlay sambil menjaga state internal tetap konsisten.
    def _close(self, event=None):
        global _EASTER_EGG_ACTIVE
        if not self._running:
            return
        self._running = False
        _EASTER_EGG_ACTIVE = False

        for job in (self._frame_job, self._anim_job):
            if job:
                try:
                    self.after_cancel(job)
                except Exception:
                    pass

        if self._cap:
            try:
                self._cap.release()
            except Exception:
                pass

        if PYGAME_AVAILABLE:
            if getattr(self, '_egg_sound_obj', None) is not None:
                try:
                    self._egg_sound_obj.stop()
                except Exception:
                    pass
            else:
                try:
                    pygame.mixer.music.stop()
                except Exception:
                    pass

        if callable(self._on_close_callback):
            try:
                self._on_close_callback()
            except Exception:
                pass
        elif PYGAME_AVAILABLE and _APP_INSTANCE is not None:
            try:
                app = _APP_INSTANCE
                if app._music_ready and not app._music_paused:
                    if not getattr(self, '_music_was_replaced', False):
                        pygame.mixer.music.unpause()
                    else:
                        pygame.mixer.music.load(MUSIC_FILE)
                        pygame.mixer.music.set_volume(0.7)
                        pygame.mixer.music.play(loops=-1)
                    app._music_on = True
                    app._draw_music_btn()
            except Exception:
                pass

        self.destroy()

# Mengaktifkan urutan easter egg ketika kondisi rahasia terpenuhi.
def _trigger_easter_egg(root, on_close_cb=None):
    global _EASTER_EGG_ACTIVE
    if _EASTER_EGG_ACTIVE:
        return
    try:
        EasterEggOverlay(root, on_close_callback=on_close_cb)
    except Exception as e:
        print("Easter egg error:", e)

# Menjalankan hook tambahan setelah sesi selesai agar alur penyimpanan dan notifikasi tetap terpanggil.
def _patched_on_finish(self, session, ml):
    no_activity = (
        session.get("moves", 0) == 0 and
        session.get("total_time", 0) < 5
    )
    if no_activity:
        self._back_to_grid_select()
        return

    # Fungsi bantu ini memecah logika patched on finish agar langkah kecilnya lebih rapi dan mudah dipakai ulang di bagian ini.
    def _show_dashboard():
        self._clear()
        dashboard_session = _ml_dashboard_session(session, ml)
        self._rebuild_fn = lambda: _patched_on_finish(self, session, ml)
        self.screen = PerformanceDashboard(
            self.root, self, self.username, dashboard_session, ml)
        self.root.after(50, self._raise_overlay)

    data         = load_data()
    all_sessions = data["players"].get(self.username, {}).get("sessions", [])
    earned_ids   = _evaluate_achievements(self.username, session, all_sessions)
    earned_badges = [ACHIEVEMENTS[i] for i in earned_ids if i in ACHIEVEMENTS]

    if earned_badges:
        _blur_photo = _grab_blur_bg(self.root, radius=16, darken=0.32)
        # Fungsi bantu ini memecah logika patched on finish agar langkah kecilnya lebih rapi dan mudah dipakai ulang di bagian ini.
        def _reshowpopup():
            resume = 0
            cur = getattr(self, "screen", None)
            if isinstance(cur, AchievementPopup):
                resume = max(0, cur._idx - 1)
            self._clear()
            self.screen = AchievementPopup(self.root, earned_badges,
                                           on_done=_show_dashboard,
                                           initial_idx=resume,
                                           pre_blur_photo=_blur_photo)
            self.root.after(50, self._raise_overlay)
        self._clear()
        self._rebuild_fn = _reshowpopup
        self.screen = AchievementPopup(self.root, earned_badges,
                                       on_done=_show_dashboard,
                                       pre_blur_photo=_blur_photo)
        self.root.after(50, self._raise_overlay)
    else:
        _show_dashboard()

SudokuApp._on_finish = _patched_on_finish

# =============================================================================
# MAIN PROGRAM
# =============================================================================
if __name__ == "__main__":
    app = SudokuApp()
    app.run()