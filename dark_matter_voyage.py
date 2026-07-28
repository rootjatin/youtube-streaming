from __future__ import annotations

"""
DARK MATTER: THE SILENT ARCHITECTURE
====================================
Result : https://youtu.be/_uCr-5y_DDg?si=OlOPRDZRT8GCrdbY
A full-length, widescreen cinematic galaxy film rendered entirely with Python.
The camera drifts like a slow drone through a luminous disk and an inferred
three-dimensional dark-matter halo.

This is not a YouTube Short. The default render is a 10-minute 1920x1080 film
at 24 fps, with a procedurally generated stereo ambient soundtrack.

REAL-DATA FOUNDATION
--------------------
The program downloads the public SPARC galaxy database and, by default, uses
NGC 3198. SPARC combines Spitzer 3.6-micron photometry with high-quality
H I + H-alpha rotation curves for nearby disk galaxies.

For the selected galaxy, the renderer uses:
- measured radius and observed circular velocity,
- measured gas, stellar-disk, and bulge velocity contributions,
- SPARC distance, inclination, luminosity, disk scale length, H I mass,
  H I radius, flat velocity, and quality flag when available.

The invisible component is estimated from the rotation-curve residual:

    v_dark^2 = max(v_observed^2 - v_baryonic^2, 0)
    M_dark(<r) = v_dark^2 r / G

The resulting enclosed-mass profile is used as a probability distribution for
placing the cinematic halo tracers. This is a data-constrained visualization,
not a direct image of dark matter and not a unique halo fit.

Scientific honesty
------------------
Dark matter does not emit or reflect enough light to be directly photographed.
The blue-violet halo, filaments, lensing arcs, and particles in this film are a
visual metaphor for an inferred gravitational mass distribution. Their colors,
particle sizes, camera motion, and time evolution are artistic choices.

Primary data and paper:
- SPARC data: https://astroweb.case.edu/SPARC/
- Lelli, McGaugh & Schombert (2016), AJ 152, 157
- VizieR catalog J/AJ/152/157

INSTALL
-------
    pip install numpy pandas pillow imageio imageio-ffmpeg requests tqdm

FULL RENDER
-----------
    python dark_matter_the_silent_architecture.py

FAST TEST
---------
    DARK_MATTER_QUICK=1 python dark_matter_the_silent_architecture.py

PREVIEWS ONLY
-------------
    DARK_MATTER_PREVIEW_ONLY=1 python dark_matter_the_silent_architecture.py

SELECT ANOTHER SPARC GALAXY
---------------------------
    DARK_MATTER_GALAXY=NGC2403 python dark_matter_the_silent_architecture.py

CUSTOM DURATION / RESOLUTION
----------------------------
    DARK_MATTER_DURATION=900 python dark_matter_the_silent_architecture.py
    DARK_MATTER_4K=1 python dark_matter_the_silent_architecture.py

USE YOUR OWN MUSIC INSTEAD OF THE GENERATED SOUNDTRACK
------------------------------------------------------
    #use wav music
    DARK_MATTER_AUDIO=/path/to/music.wav python dark_matter_the_silent_architecture.py

Force a fresh SPARC download:
    DARK_MATTER_REFRESH=1 python dark_matter_the_silent_architecture.py
"""

import hashlib
import io
import json
import math
import os
import shutil
import subprocess
import wave
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import imageio.v2 as iio
import numpy as np
import pandas as pd
import requests
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont
from requests.adapters import HTTPAdapter
from tqdm.auto import tqdm
from urllib3.util.retry import Retry


# =============================================================================
# Configuration
# =============================================================================

QUICK_MODE = os.getenv("DARK_MATTER_QUICK", "0") == "1"
PREVIEW_ONLY = os.getenv("DARK_MATTER_PREVIEW_ONLY", "0") == "1"
FORCE_REFRESH = os.getenv("DARK_MATTER_REFRESH", "0") == "1"
FOUR_K = os.getenv("DARK_MATTER_4K", "0") == "1" and not QUICK_MODE
SELECTED_GALAXY = os.getenv("DARK_MATTER_GALAXY", "NGC3198").strip()
EXTERNAL_AUDIO = os.getenv("DARK_MATTER_AUDIO", "").strip() or None

DEFAULT_DURATION = 24.0 if QUICK_MODE else 600.0
DURATION = float(os.getenv("DARK_MATTER_DURATION", str(DEFAULT_DURATION)))
FPS = int(os.getenv("DARK_MATTER_FPS", "10" if QUICK_MODE else "24"))

if QUICK_MODE:
    WIDTH, HEIGHT = 960, 540
elif FOUR_K:
    WIDTH, HEIGHT = 3840, 2160
else:
    WIDTH, HEIGHT = 1920, 1080

OUT_SIZE = (WIDTH, HEIGHT)
SCALE = WIDTH / 1920.0

OUTPUT_ROOT = Path("dark_matter_the_silent_architecture_output")
DATA_ROOT = OUTPUT_ROOT / "data"
PREVIEW_DIR = OUTPUT_ROOT / "previews"
AUDIO_DIR = OUTPUT_ROOT / "audio"
for directory in (OUTPUT_ROOT, DATA_ROOT, PREVIEW_DIR, AUDIO_DIR):
    directory.mkdir(parents=True, exist_ok=True)

CONFIG: Dict[str, object] = {
    "title": "DARK MATTER",
    "subtitle": "THE SILENT ARCHITECTURE",
    "youtube_title": "DARK MATTER: The Silent Architecture of a Galaxy",
    "output_basename": "dark_matter_the_silent_architecture",
    "width": WIDTH,
    "height": HEIGHT,
    "fps": FPS,
    "duration_s": DURATION,
    "selected_galaxy": SELECTED_GALAXY,
    "render_scale": 0.46 if FOUR_K else (0.62 if QUICK_MODE else 0.54),
    "disk_particles": 14_000 if QUICK_MODE else (150_000 if FOUR_K else 105_000),
    "bulge_particles": 3_000 if QUICK_MODE else (32_000 if FOUR_K else 22_000),
    "halo_particles": 22_000 if QUICK_MODE else (230_000 if FOUR_K else 155_000),
    "dust_particles": 5_000 if QUICK_MODE else (50_000 if FOUR_K else 34_000),
    "background_stars": 1_200 if QUICK_MODE else (5_000 if FOUR_K else 3_200),
    "audio_sample_rate": 48_000,
    "master_data_url": "https://astroweb.case.edu/SPARC/SPARC_Lelli2016c.mrt",
    "rotation_zip_url": "https://astroweb.case.edu/SPARC/Rotmod_LTG.zip",
    "vizier_fallback_url": (
        "https://vizier.cds.unistra.fr/viz-bin/asu-tsv?"
        "-source=J/AJ/152/157/table1&-out.all=1&-out.max=unlimited"
    ),
    "request_timeout": (20, 180),
    "request_retries": 5,
    "retry_backoff_s": 1.4,
    "disk_mass_to_light": 0.5,
    "bulge_mass_to_light": 0.7,
    "halo_radius_multiplier": 2.8,
    "show_science_cards": True,
    "write_subtitle_sidecar": True,
    "burn_subtitles": False,
    "audio_path": EXTERNAL_AUDIO,
}

SPARC_MASTER_CACHE = DATA_ROOT / "SPARC_Lelli2016c.mrt"
SPARC_ROTATION_CACHE = DATA_ROOT / "Rotmod_LTG.zip"

G_KPC_KMS2_MSUN = 4.30091e-6

YOUTUBE_DESCRIPTION_TEMPLATE = """A slow voyage through the gravity we cannot see.

“DARK MATTER: The Silent Architecture of a Galaxy” is a full-length cinematic journey through a spiral galaxy and the vast invisible halo inferred from its motion. The camera moves from the outer gravitational envelope, descends toward the stellar disk, crosses the galactic plane, and returns to the quiet halo beyond the visible light.

The scientific foundation comes from the public SPARC database. This render is calibrated with real rotation-curve and photometric measurements for {galaxy}: observed circular speed, gas contribution, stellar-disk contribution, bulge contribution, disk scale length, distance, inclination, and H I properties where available.

Dark matter cannot be directly photographed. The blue-violet halo, filaments, lensing arcs, and luminous particles are an artistic visualization of an inferred mass distribution. The halo sampling is constrained by the difference between the observed rotation curve and the baryonic contribution; it is not a unique physical reconstruction.

Selected galaxy: {galaxy}
SPARC distance: {distance_mpc:.2f} Mpc
SPARC disk scale length: {rdisk_kpc:.2f} kpc
SPARC flat rotation speed: {vflat_kms:.1f} km/s
Rotation-curve quality flag: {quality}

Data: SPARC — Spitzer Photometry & Accurate Rotation Curves
Reference: Lelli, McGaugh & Schombert, Astronomical Journal 152, 157 (2016)

Best experienced in a dark room with headphones.
"""

CHAPTERS = [
    (0.00, "The invisible shore"),
    (0.13, "Falling through the halo"),
    (0.27, "The luminous disk"),
    (0.43, "Where the curve stays flat"),
    (0.59, "Crossing the galactic plane"),
    (0.75, "The gravity beyond light"),
    (0.90, "Return to silence"),
]


# =============================================================================
# General helpers
# =============================================================================


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def smoothstep(t: float) -> float:
    t = clamp(t)
    return t * t * (3.0 - 2.0 * t)


def smootherstep(t: float) -> float:
    t = clamp(t)
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def ease_in_out_sine(t: float) -> float:
    t = clamp(t)
    return -(math.cos(math.pi * t) - 1.0) / 2.0


def deterministic_unit(text: str, salt: str = "") -> float:
    digest = hashlib.sha256(f"{text}|{salt}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64 - 1)


def safe_float(value, default=np.nan) -> float:
    try:
        if value is None or str(value).strip() in {"", "...", "--", "nan"}:
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def safe_int(value, default=0) -> int:
    try:
        return int(float(value))
    except Exception:
        return int(default)


def find_ffmpeg() -> Optional[str]:
    executable = shutil.which("ffmpeg")
    if executable:
        return executable
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def run_ffmpeg(command: Sequence[str]) -> None:
    print("Running:", " ".join(str(item) for item in command))
    subprocess.run(list(command), check=True)


def get_font(size: int, bold: bool = False, serif: bool = False):
    size = max(8, int(size))
    names: List[str] = []
    if serif:
        names += [
            "DejaVuSerif-Bold.ttf" if bold else "DejaVuSerif.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf"
            if bold
            else "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
        ]
    names += [
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/Library/Fonts/Arial Bold.ttf" if bold else "/Library/Fonts/Arial.ttf",
        "arialbd.ttf" if bold else "arial.ttf",
    ]
    for name in names:
        try:
            return ImageFont.truetype(name, size=size)
        except Exception:
            pass
    return ImageFont.load_default()


def draw_text(
    image: Image.Image,
    text: str,
    xy: Tuple[float, float],
    size: int,
    fill=(255, 255, 255, 255),
    bold: bool = False,
    serif: bool = False,
    anchor: str = "la",
    stroke: int = 1,
    spacing: int = 4,
) -> None:
    draw = ImageDraw.Draw(image)
    draw.text(
        xy,
        text,
        font=get_font(int(size * SCALE), bold=bold, serif=serif),
        fill=fill,
        anchor=anchor,
        stroke_width=max(0, int(stroke * SCALE)),
        stroke_fill=(0, 0, 0, min(220, fill[3] if len(fill) > 3 else 220)),
        spacing=int(spacing * SCALE),
    )


def draw_wrapped_text(
    image: Image.Image,
    text: str,
    xy: Tuple[int, int],
    max_width: int,
    size: int,
    fill=(255, 255, 255, 245),
    bold: bool = False,
    serif: bool = False,
    line_spacing: int = 8,
) -> int:
    draw = ImageDraw.Draw(image)
    font = get_font(int(size * SCALE), bold=bold, serif=serif)
    words = text.split()
    lines: List[str] = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        box = draw.textbbox((0, 0), candidate, font=font, stroke_width=max(0, int(SCALE)))
        if box[2] - box[0] <= max_width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    x, y = xy
    for line in lines:
        draw.text(
            (x, y),
            line,
            font=font,
            fill=fill,
            stroke_width=max(0, int(SCALE)),
            stroke_fill=(0, 0, 0, 210),
        )
        box = draw.textbbox((x, y), line, font=font, stroke_width=max(0, int(SCALE)))
        y += box[3] - box[1] + int(line_spacing * SCALE)
    return y


def format_srt_time(seconds: float) -> str:
    milliseconds = int(round(seconds * 1000.0))
    hours, milliseconds = divmod(milliseconds, 3_600_000)
    minutes, milliseconds = divmod(milliseconds, 60_000)
    secs, milliseconds = divmod(milliseconds, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"


def make_vignette(width: int, height: int, strength: float = 0.34) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width]
    nx = (xx - width / 2.0) / (width / 2.0)
    ny = (yy - height / 2.0) / (height / 2.0)
    radius = np.sqrt(nx * nx + ny * ny)
    return np.clip(1.0 - strength * radius**1.7, 0.0, 1.0).astype(np.float32)


VIGNETTE = make_vignette(WIDTH, HEIGHT)


# =============================================================================
# SPARC download and parsing
# =============================================================================


def build_retry_session() -> requests.Session:
    retries = Retry(
        total=int(CONFIG["request_retries"]),
        connect=int(CONFIG["request_retries"]),
        read=int(CONFIG["request_retries"]),
        status=int(CONFIG["request_retries"]),
        backoff_factor=float(CONFIG["retry_backoff_s"]),
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"]),
        raise_on_status=False,
    )
    session = requests.Session()
    adapter = HTTPAdapter(max_retries=retries, pool_connections=2, pool_maxsize=2)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update(
        {"User-Agent": "dark-matter-silent-architecture/1.0 educational-visualization"}
    )
    return session


def download_with_cache(
    session: requests.Session,
    url: str,
    path: Path,
    force_refresh: bool = False,
) -> Path:
    if path.exists() and path.stat().st_size > 100 and not force_refresh:
        return path
    response = session.get(url, timeout=CONFIG["request_timeout"])
    if response.status_code >= 400:
        raise RuntimeError(f"HTTP {response.status_code} while downloading {url}")
    path.write_bytes(response.content)
    return path


def parse_sparc_master_fixed_width(text: str) -> pd.DataFrame:
    columns = [
        "Galaxy",
        "T",
        "D",
        "e_D",
        "f_D",
        "Inc",
        "e_Inc",
        "L36",
        "e_L36",
        "Reff",
        "SBeff",
        "Rdisk",
        "SBdisk",
        "MHI",
        "RHI",
        "Vflat",
        "e_Vflat",
        "Q",
        "Ref",
    ]
    slices = [
        (0, 11),
        (11, 13),
        (13, 19),
        (19, 24),
        (24, 26),
        (26, 30),
        (30, 34),
        (34, 41),
        (41, 48),
        (48, 53),
        (53, 61),
        (61, 66),
        (66, 74),
        (74, 81),
        (81, 86),
        (86, 91),
        (91, 96),
        (96, 99),
        (99, 113),
    ]
    rows: List[dict] = []
    for raw in text.splitlines():
        if len(raw) < 99:
            continue
        fields = [raw[start:end].strip() for start, end in slices]
        galaxy = fields[0]
        # Data rows have a valid numerical Hubble type and distance.
        try:
            hubble_type = int(fields[1])
            distance = float(fields[2])
        except Exception:
            continue
        if not galaxy or not (0 <= hubble_type <= 20) or distance <= 0:
            continue
        record = dict(zip(columns, fields))
        rows.append(record)
    if not rows:
        raise RuntimeError("Could not parse any rows from the SPARC master table.")
    frame = pd.DataFrame(rows)
    numeric = [name for name in columns if name not in {"Galaxy", "Ref"}]
    for name in numeric:
        frame[name] = pd.to_numeric(frame[name], errors="coerce")
    frame["Galaxy_key"] = frame["Galaxy"].str.replace(" ", "", regex=False).str.upper()
    return frame


def parse_vizier_tsv(text: str) -> pd.DataFrame:
    useful_lines = [line for line in text.splitlines() if line and not line.startswith("#")]
    if len(useful_lines) < 3:
        raise RuntimeError("VizieR fallback returned no usable table rows.")
    frame = pd.read_csv(io.StringIO("\n".join(useful_lines)), sep="\t")
    rename = {
        "Name": "Galaxy",
        "Type": "T",
        "Dist": "D",
        "e_Dist": "e_D",
        "f_Dist": "f_D",
        "i": "Inc",
        "e_i": "e_Inc",
        "L3.6": "L36",
        "e_L3.6": "e_L36",
        "Reff": "Reff",
        "SBeff": "SBeff",
        "Rdisk": "Rdisk",
        "SBdisk": "SBdisk",
        "MHI": "MHI",
        "RHI": "RHI",
        "Vflat": "Vflat",
        "e_Vflat": "e_Vflat",
        "Qual": "Q",
        "Ref": "Ref",
    }
    frame = frame.rename(columns=rename)
    needed = list(rename.values())
    for name in needed:
        if name not in frame.columns:
            frame[name] = np.nan if name not in {"Galaxy", "Ref"} else ""
    for name in needed:
        if name not in {"Galaxy", "Ref"}:
            frame[name] = pd.to_numeric(frame[name], errors="coerce")
    frame["Galaxy"] = frame["Galaxy"].astype(str).str.strip()
    frame["Galaxy_key"] = frame["Galaxy"].str.replace(" ", "", regex=False).str.upper()
    return frame


def fetch_sparc_master(force_refresh: bool = False) -> Tuple[pd.DataFrame, str]:
    session = build_retry_session()
    try:
        path = download_with_cache(
            session,
            str(CONFIG["master_data_url"]),
            SPARC_MASTER_CACHE,
            force_refresh,
        )
        return parse_sparc_master_fixed_width(path.read_text(encoding="utf-8", errors="replace")), "SPARC master table"
    except Exception as primary_error:
        print("Primary SPARC table download/parse failed:", primary_error)
        try:
            response = session.get(
                str(CONFIG["vizier_fallback_url"]), timeout=CONFIG["request_timeout"]
            )
            response.raise_for_status()
            return parse_vizier_tsv(response.text), "VizieR J/AJ/152/157"
        except Exception as fallback_error:
            print("VizieR fallback failed:", fallback_error)
            return fallback_master_table(), "built-in fallback"


def fallback_master_table() -> pd.DataFrame:
    # Approximate values are used only when the public catalog cannot be reached.
    # They are deliberately marked as a fallback in the generated manifest.
    rows = [
        {
            "Galaxy": "NGC3198",
            "T": 5,
            "D": 13.8,
            "e_D": np.nan,
            "f_D": 1,
            "Inc": 72.0,
            "e_Inc": np.nan,
            "L36": 31.0,
            "e_L36": np.nan,
            "Reff": 5.2,
            "SBeff": np.nan,
            "Rdisk": 3.14,
            "SBdisk": np.nan,
            "MHI": 10.1,
            "RHI": 30.0,
            "Vflat": 150.0,
            "e_Vflat": 5.0,
            "Q": 1,
            "Ref": "offline fallback",
        },
        {
            "Galaxy": "NGC2403",
            "T": 6,
            "D": 3.2,
            "e_D": np.nan,
            "f_D": 2,
            "Inc": 63.0,
            "e_Inc": np.nan,
            "L36": 10.0,
            "e_L36": np.nan,
            "Reff": 3.0,
            "SBeff": np.nan,
            "Rdisk": 1.8,
            "SBdisk": np.nan,
            "MHI": 3.2,
            "RHI": 18.0,
            "Vflat": 130.0,
            "e_Vflat": 5.0,
            "Q": 1,
            "Ref": "offline fallback",
        },
    ]
    frame = pd.DataFrame(rows)
    frame["Galaxy_key"] = frame["Galaxy"].str.upper()
    return frame


def normalize_galaxy_key(name: str) -> str:
    return "".join(character for character in name.upper() if character.isalnum())


def choose_galaxy_row(frame: pd.DataFrame, requested: str) -> pd.Series:
    wanted = normalize_galaxy_key(requested)
    keys = frame["Galaxy"].astype(str).map(normalize_galaxy_key)
    exact = frame[keys.eq(wanted)]
    if len(exact):
        return exact.iloc[0]
    contains = frame[keys.str.contains(wanted, regex=False)] if wanted else frame.iloc[0:0]
    if len(contains):
        return contains.iloc[0]
    print(f"Galaxy '{requested}' was not found. Falling back to NGC3198 or the first usable row.")
    fallback = frame[keys.eq("NGC3198")]
    return fallback.iloc[0] if len(fallback) else frame.iloc[0]


def download_rotation_zip(force_refresh: bool = False) -> Optional[Path]:
    session = build_retry_session()
    try:
        return download_with_cache(
            session,
            str(CONFIG["rotation_zip_url"]),
            SPARC_ROTATION_CACHE,
            force_refresh,
        )
    except Exception as error:
        print("SPARC rotation-curve archive unavailable:", error)
        return None


def parse_rotation_curve_bytes(data: bytes) -> pd.DataFrame:
    rows: List[List[float]] = []
    text = data.decode("utf-8", errors="replace")
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("-"):
            continue
        values: List[float] = []
        for item in line.replace(",", " ").split():
            try:
                values.append(float(item))
            except Exception:
                pass
        if len(values) >= 6 and values[0] >= 0:
            values += [0.0] * (8 - len(values))
            rows.append(values[:8])
    if not rows:
        raise RuntimeError("No numerical rotation-curve rows were found.")
    columns = ["R", "Vobs", "e_Vobs", "Vgas", "Vdisk", "Vbul", "SBdisk", "SBbul"]
    frame = pd.DataFrame(rows, columns=columns)
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=["R", "Vobs"])
    frame = frame[(frame["R"] > 0) & (frame["Vobs"] > 0)].sort_values("R")
    return frame.drop_duplicates("R").reset_index(drop=True)


def extract_rotation_curve(zip_path: Optional[Path], galaxy: str) -> Optional[pd.DataFrame]:
    if zip_path is None or not zip_path.exists():
        return None
    wanted = normalize_galaxy_key(galaxy)
    try:
        with zipfile.ZipFile(zip_path, "r") as archive:
            names = archive.namelist()
            candidates = [
                name
                for name in names
                if wanted in normalize_galaxy_key(Path(name).stem)
                and name.lower().endswith((".dat", ".txt"))
            ]
            if not candidates:
                return None
            candidates.sort(key=len)
            return parse_rotation_curve_bytes(archive.read(candidates[0]))
    except Exception as error:
        print("Could not extract rotation curve:", error)
        return None


def fallback_rotation_curve(row: pd.Series) -> pd.DataFrame:
    rdisk = max(safe_float(row.get("Rdisk"), 3.0), 0.5)
    vflat = max(safe_float(row.get("Vflat"), 150.0), 30.0)
    radius = np.geomspace(max(0.12 * rdisk, 0.25), 12.0 * rdisk, 72)
    vobs = vflat * (1.0 - np.exp(-radius / (1.45 * rdisk)))
    vdisk = 0.78 * vflat * (radius / rdisk) * np.exp(1.0 - radius / (1.9 * rdisk))
    vdisk = np.minimum(vdisk, 0.86 * vobs)
    vgas = 0.32 * vflat * (1.0 - np.exp(-radius / (3.4 * rdisk)))
    vbul = 0.20 * vflat * np.exp(-radius / max(0.65 * rdisk, 0.3))
    return pd.DataFrame(
        {
            "R": radius,
            "Vobs": vobs,
            "e_Vobs": np.full_like(radius, 5.0),
            "Vgas": vgas,
            "Vdisk": vdisk,
            "Vbul": vbul,
            "SBdisk": np.exp(-radius / rdisk),
            "SBbul": np.exp(-radius / max(0.45 * rdisk, 0.2)),
        }
    )


# =============================================================================
# Physical profile and particle model
# =============================================================================


@dataclass
class GalaxyProfile:
    galaxy: str
    data_source: str
    rotation_curve_source: str
    distance_mpc: float
    inclination_deg: float
    luminosity_36_1e9_lsun: float
    rdisk_kpc: float
    reff_kpc: float
    mhi_1e9_msun: float
    rhi_kpc: float
    vflat_kms: float
    quality: int
    radius_kpc: np.ndarray
    vobs_kms: np.ndarray
    vbar_kms: np.ndarray
    vdark_kms: np.ndarray
    mdark_msun: np.ndarray
    halo_radius_kpc: float
    spatial_scale_kpc: float

    def serializable(self) -> dict:
        output = asdict(self)
        for key in ["radius_kpc", "vobs_kms", "vbar_kms", "vdark_kms", "mdark_msun"]:
            output[key] = [float(value) for value in output[key]]
        return output


def build_galaxy_profile(
    row: pd.Series,
    rotation_curve: pd.DataFrame,
    data_source: str,
    rotation_curve_source: str,
) -> GalaxyProfile:
    curve = rotation_curve.copy()
    upsilon_disk = float(CONFIG["disk_mass_to_light"])
    upsilon_bulge = float(CONFIG["bulge_mass_to_light"])

    gas_term = np.sign(curve["Vgas"].to_numpy(float)) * np.square(
        curve["Vgas"].to_numpy(float)
    )
    disk_term = upsilon_disk * np.square(curve["Vdisk"].to_numpy(float))
    bulge_term = upsilon_bulge * np.square(curve["Vbul"].to_numpy(float))
    vbar2 = np.maximum(gas_term + disk_term + bulge_term, 0.0)
    vbar = np.sqrt(vbar2)
    vobs = curve["Vobs"].to_numpy(float)
    vdark = np.sqrt(np.maximum(vobs * vobs - vbar2, 0.0))
    radius = curve["R"].to_numpy(float)
    mdark = np.maximum(vdark * vdark * radius / G_KPC_KMS2_MSUN, 0.0)
    mdark = np.maximum.accumulate(mdark)

    rdisk = max(safe_float(row.get("Rdisk"), np.nan), 0.5)
    if not np.isfinite(rdisk):
        rdisk = max(float(np.nanmedian(radius)) / 4.0, 1.0)
    rhi = safe_float(row.get("RHI"), np.nan)
    measured_extent = float(np.nanmax(radius))
    halo_radius = max(
        measured_extent * float(CONFIG["halo_radius_multiplier"]),
        (rhi if np.isfinite(rhi) and rhi > 0 else 0.0) * 1.8,
        rdisk * 18.0,
    )
    spatial_scale = max(measured_extent, rdisk * 7.0)

    return GalaxyProfile(
        galaxy=str(row.get("Galaxy", SELECTED_GALAXY)).strip(),
        data_source=data_source,
        rotation_curve_source=rotation_curve_source,
        distance_mpc=safe_float(row.get("D"), 13.8),
        inclination_deg=safe_float(row.get("Inc"), 70.0),
        luminosity_36_1e9_lsun=safe_float(row.get("L36"), np.nan),
        rdisk_kpc=rdisk,
        reff_kpc=safe_float(row.get("Reff"), 1.678 * rdisk),
        mhi_1e9_msun=safe_float(row.get("MHI"), np.nan),
        rhi_kpc=safe_float(row.get("RHI"), np.nan),
        vflat_kms=safe_float(row.get("Vflat"), float(np.nanmedian(vobs[-10:]))),
        quality=safe_int(row.get("Q"), 0),
        radius_kpc=radius,
        vobs_kms=vobs,
        vbar_kms=vbar,
        vdark_kms=vdark,
        mdark_msun=mdark,
        halo_radius_kpc=halo_radius,
        spatial_scale_kpc=spatial_scale,
    )


def inverse_cdf_sample(x: np.ndarray, cdf: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    cdf = np.asarray(cdf, dtype=float)
    x = np.asarray(x, dtype=float)
    keep = np.isfinite(x) & np.isfinite(cdf)
    x, cdf = x[keep], cdf[keep]
    if len(x) < 2 or cdf[-1] <= cdf[0]:
        return rng.uniform(float(x[0]) if len(x) else 0.0, float(x[-1]) if len(x) else 1.0, count)
    cdf = np.maximum.accumulate(cdf)
    cdf = (cdf - cdf[0]) / max(cdf[-1] - cdf[0], 1e-12)
    unique = np.concatenate(([True], np.diff(cdf) > 1e-10))
    return np.interp(rng.random(count), cdf[unique], x[unique])


@dataclass
class ParticleCloud:
    xyz: np.ndarray
    luminosity: np.ndarray
    temperature: np.ndarray
    radius_norm: np.ndarray
    phase: np.ndarray


@dataclass
class GalaxyParticles:
    disk: ParticleCloud
    bulge: ParticleCloud
    dust: ParticleCloud
    halo: ParticleCloud
    background_xyz: np.ndarray
    background_luminosity: np.ndarray


class ParticleFactory:
    def __init__(self, profile: GalaxyProfile, seed: int = 3198):
        self.profile = profile
        self.rng = np.random.default_rng(seed)
        self.scale = profile.spatial_scale_kpc

    def _cloud(self, xyz, luminosity, temperature, phase=None) -> ParticleCloud:
        xyz = np.asarray(xyz, dtype=np.float32)
        radius = np.linalg.norm(xyz, axis=1)
        if phase is None:
            phase = self.rng.uniform(0.0, 2.0 * np.pi, len(xyz))
        return ParticleCloud(
            xyz=xyz,
            luminosity=np.asarray(luminosity, dtype=np.float32),
            temperature=np.asarray(temperature, dtype=np.float32),
            radius_norm=np.asarray(radius / max(self.profile.halo_radius_kpc, 1e-6), dtype=np.float32),
            phase=np.asarray(phase, dtype=np.float32),
        )

    def make_disk(self, count: int) -> ParticleCloud:
        rd = self.profile.rdisk_kpc
        # Gamma(k=2) gives the radial probability density of an exponential disk.
        radius = self.rng.gamma(shape=2.0, scale=rd, size=count)
        radius = np.clip(radius, 0.03 * rd, 7.6 * rd)
        arms = 4
        arm_index = self.rng.integers(0, arms, count)
        pitch = math.radians(18.5)
        arm_theta = (
            arm_index * (2.0 * np.pi / arms)
            + np.log(np.maximum(radius / rd, 0.05)) / math.tan(pitch)
        )
        interarm = self.rng.random(count) < 0.30
        theta = arm_theta + self.rng.normal(0.0, 0.17 + 0.025 * radius / rd, count)
        theta[interarm] = self.rng.uniform(0.0, 2.0 * np.pi, interarm.sum())
        z_sigma = 0.065 * rd * (1.0 + 0.10 * radius / rd)
        z = self.rng.normal(0.0, z_sigma)
        xyz = np.column_stack((radius * np.cos(theta), radius * np.sin(theta), z))
        young = (~interarm) & (self.rng.random(count) < 0.22)
        temperature = np.where(young, self.rng.uniform(0.72, 1.0, count), self.rng.uniform(0.15, 0.68, count))
        luminosity = self.rng.lognormal(mean=-0.18, sigma=0.62, size=count)
        luminosity *= np.exp(-0.055 * radius / rd)
        luminosity[young] *= self.rng.uniform(1.8, 4.4, young.sum())
        return self._cloud(xyz, luminosity, temperature, phase=theta)

    def make_bulge(self, count: int) -> ParticleCloud:
        a = max(0.38 * self.profile.rdisk_kpc, 0.25)
        u = np.clip(self.rng.random(count), 1e-7, 1.0 - 1e-7)
        radius = a * np.sqrt(u) / (1.0 - np.sqrt(u))
        radius = np.clip(radius, 0.0, 3.8 * self.profile.rdisk_kpc)
        cos_theta = self.rng.uniform(-1.0, 1.0, count)
        sin_theta = np.sqrt(np.maximum(1.0 - cos_theta * cos_theta, 0.0))
        phi = self.rng.uniform(0.0, 2.0 * np.pi, count)
        flatten = 0.72
        xyz = np.column_stack(
            (
                radius * sin_theta * np.cos(phi),
                radius * sin_theta * np.sin(phi),
                flatten * radius * cos_theta,
            )
        )
        luminosity = self.rng.lognormal(mean=0.35, sigma=0.55, size=count)
        temperature = self.rng.uniform(0.18, 0.54, count)
        return self._cloud(xyz, luminosity, temperature, phase=phi)

    def make_dust(self, count: int) -> ParticleCloud:
        rd = self.profile.rdisk_kpc
        radius = self.rng.gamma(shape=2.1, scale=1.25 * rd, size=count)
        radius = np.clip(radius, 0.5 * rd, 8.0 * rd)
        arm = self.rng.integers(0, 4, count)
        pitch = math.radians(17.0)
        theta = arm * (np.pi / 2.0) + np.log(np.maximum(radius / rd, 0.08)) / math.tan(pitch)
        theta += self.rng.normal(0.24, 0.28, count)
        z = self.rng.normal(0.0, 0.12 * rd, count)
        xyz = np.column_stack((radius * np.cos(theta), radius * np.sin(theta), z))
        luminosity = self.rng.lognormal(mean=-0.60, sigma=0.55, size=count)
        temperature = self.rng.uniform(0.0, 1.0, count)
        return self._cloud(xyz, luminosity, temperature, phase=theta)

    def make_halo(self, count: int) -> ParticleCloud:
        profile = self.profile
        measured_r = profile.radius_kpc
        measured_m = profile.mdark_msun
        outer_r = np.geomspace(max(measured_r[-1] * 1.001, 0.1), profile.halo_radius_kpc, 160)
        outer_v = max(profile.vdark_kms[-1], 0.72 * profile.vflat_kms, 20.0)
        outer_m = outer_v * outer_v * outer_r / G_KPC_KMS2_MSUN
        radius_grid = np.concatenate((measured_r, outer_r))
        mass_grid = np.concatenate((measured_m, outer_m))
        mass_grid = np.maximum.accumulate(mass_grid)
        radius = inverse_cdf_sample(radius_grid, mass_grid, count, self.rng)

        cos_theta = self.rng.uniform(-1.0, 1.0, count)
        sin_theta = np.sqrt(np.maximum(1.0 - cos_theta * cos_theta, 0.0))
        phi = self.rng.uniform(0.0, 2.0 * np.pi, count)
        # A mildly triaxial halo makes the drone movement more legible.
        x = 1.05 * radius * sin_theta * np.cos(phi)
        y = 0.92 * radius * sin_theta * np.sin(phi)
        z = 0.80 * radius * cos_theta
        xyz = np.column_stack((x, y, z))
        concentration = 1.0 - np.clip(radius / profile.halo_radius_kpc, 0.0, 1.0)
        luminosity = self.rng.lognormal(mean=-0.78, sigma=0.62, size=count)
        luminosity *= 0.32 + 1.75 * concentration**1.5
        temperature = self.rng.uniform(0.0, 1.0, count)
        return self._cloud(xyz, luminosity, temperature, phase=phi)

    def make_background(self, count: int) -> Tuple[np.ndarray, np.ndarray]:
        cos_theta = self.rng.uniform(-1.0, 1.0, count)
        sin_theta = np.sqrt(np.maximum(1.0 - cos_theta * cos_theta, 0.0))
        phi = self.rng.uniform(0.0, 2.0 * np.pi, count)
        radius = self.rng.uniform(45.0, 65.0, count)
        xyz = np.column_stack(
            (
                radius * sin_theta * np.cos(phi),
                radius * sin_theta * np.sin(phi),
                radius * cos_theta,
            )
        ).astype(np.float32)
        luminosity = self.rng.lognormal(mean=-0.45, sigma=0.90, size=count).astype(np.float32)
        return xyz, luminosity

    def build(self) -> GalaxyParticles:
        background_xyz, background_luminosity = self.make_background(int(CONFIG["background_stars"]))
        return GalaxyParticles(
            disk=self.make_disk(int(CONFIG["disk_particles"])),
            bulge=self.make_bulge(int(CONFIG["bulge_particles"])),
            dust=self.make_dust(int(CONFIG["dust_particles"])),
            halo=self.make_halo(int(CONFIG["halo_particles"])),
            background_xyz=background_xyz,
            background_luminosity=background_luminosity,
        )


# =============================================================================
# Camera and projection
# =============================================================================


@dataclass
class CameraKeyframe:
    fraction: float
    position: Tuple[float, float, float]
    target: Tuple[float, float, float]
    fov_deg: float
    roll_deg: float = 0.0


CAMERA_KEYS = [
    CameraKeyframe(0.00, (0.4, -5.8, 2.7), (0.0, 0.0, 0.0), 44.0, -1.2),
    CameraKeyframe(0.10, (2.6, -4.2, 1.9), (0.0, 0.0, 0.0), 40.0, 1.0),
    CameraKeyframe(0.20, (-1.9, -3.2, 1.1), (0.0, 0.0, 0.0), 38.0, -1.6),
    CameraKeyframe(0.31, (-2.2, -1.8, 0.50), (0.15, 0.05, 0.0), 42.0, 0.8),
    CameraKeyframe(0.43, (0.5, -1.18, 0.24), (0.0, 0.15, 0.0), 47.0, 0.0),
    CameraKeyframe(0.54, (1.10, -0.45, 0.12), (-0.15, 0.22, 0.0), 52.0, 1.6),
    CameraKeyframe(0.65, (-0.65, 0.25, -0.10), (0.05, 0.68, 0.03), 50.0, -1.0),
    CameraKeyframe(0.76, (-1.90, 1.55, 0.72), (0.0, 0.0, 0.0), 43.0, 1.1),
    CameraKeyframe(0.87, (1.30, 3.65, 1.65), (0.0, 0.0, 0.0), 39.0, -1.2),
    CameraKeyframe(1.00, (0.2, 6.6, 3.2), (0.0, 0.0, 0.0), 45.0, 0.0),
]


def catmull_rom(p0, p1, p2, p3, t: float):
    t2 = t * t
    t3 = t2 * t
    return 0.5 * (
        2.0 * p1
        + (-p0 + p2) * t
        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
        + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
    )


def interpolate_camera(fraction: float) -> CameraKeyframe:
    fraction = clamp(fraction)
    for index in range(len(CAMERA_KEYS) - 1):
        left = CAMERA_KEYS[index]
        right = CAMERA_KEYS[index + 1]
        if left.fraction <= fraction <= right.fraction:
            local = (fraction - left.fraction) / max(right.fraction - left.fraction, 1e-8)
            local = smootherstep(local)
            p0 = np.array(CAMERA_KEYS[max(0, index - 1)].position, dtype=float)
            p1 = np.array(left.position, dtype=float)
            p2 = np.array(right.position, dtype=float)
            p3 = np.array(CAMERA_KEYS[min(len(CAMERA_KEYS) - 1, index + 2)].position, dtype=float)
            q0 = np.array(CAMERA_KEYS[max(0, index - 1)].target, dtype=float)
            q1 = np.array(left.target, dtype=float)
            q2 = np.array(right.target, dtype=float)
            q3 = np.array(CAMERA_KEYS[min(len(CAMERA_KEYS) - 1, index + 2)].target, dtype=float)
            position = catmull_rom(p0, p1, p2, p3, local)
            target = catmull_rom(q0, q1, q2, q3, local)
            return CameraKeyframe(
                fraction=fraction,
                position=tuple(position),
                target=tuple(target),
                fov_deg=lerp(left.fov_deg, right.fov_deg, local),
                roll_deg=lerp(left.roll_deg, right.roll_deg, local),
            )
    return CAMERA_KEYS[-1]


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-12:
        return np.zeros_like(vector)
    return vector / norm


def camera_basis(position: np.ndarray, target: np.ndarray, roll_deg: float):
    forward = normalize_vector(target - position)
    world_up = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(forward, world_up))) > 0.97:
        world_up = np.array([0.0, 1.0, 0.0])
    right = normalize_vector(np.cross(forward, world_up))
    up = normalize_vector(np.cross(right, forward))
    roll = math.radians(roll_deg)
    cr, sr = math.cos(roll), math.sin(roll)
    rolled_right = cr * right + sr * up
    rolled_up = -sr * right + cr * up
    return rolled_right, rolled_up, forward


def rotate_z(xyz: np.ndarray, angle: np.ndarray | float) -> np.ndarray:
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    c = np.cos(angle)
    s = np.sin(angle)
    return np.column_stack((c * x - s * y, s * x + c * y, z))


def rotate_xyz_global(xyz: np.ndarray, ax: float, ay: float, az: float) -> np.ndarray:
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    cx, sx = math.cos(ax), math.sin(ax)
    cy, sy = math.cos(ay), math.sin(ay)
    cz, sz = math.cos(az), math.sin(az)
    y, z = cx * y - sx * z, sx * y + cx * z
    x, z = cy * x + sy * z, -sy * x + cy * z
    x, y = cz * x - sz * y, sz * x + cz * y
    return np.column_stack((x, y, z))


# =============================================================================
# Cinematic scene renderer
# =============================================================================


class DarkMatterScene:
    def __init__(self, profile: GalaxyProfile, particles: GalaxyParticles):
        self.profile = profile
        self.particles = particles
        self.render_scale = float(CONFIG["render_scale"])
        self.rw = max(320, int(WIDTH * self.render_scale))
        self.rh = max(180, int(HEIGHT * self.render_scale))
        self.scale_kpc = profile.spatial_scale_kpc
        self.rng = np.random.default_rng(7741)
        self.background_nebula = self._make_background_nebula()
        self.rotation_curve_card = self._make_rotation_curve_card()

    def _make_background_nebula(self) -> Image.Image:
        base = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(base)
        blobs = [
            (0.14, 0.30, 0.34, (32, 20, 82), 54),
            (0.78, 0.26, 0.30, (8, 58, 92), 42),
            (0.66, 0.80, 0.38, (26, 10, 65), 48),
            (0.34, 0.72, 0.44, (4, 46, 74), 36),
        ]
        for fx, fy, fr, color, alpha in blobs:
            cx, cy = WIDTH * fx, HEIGHT * fy
            radius = WIDTH * fr
            draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=(*color, alpha))
        return base.filter(ImageFilter.GaussianBlur(max(18, int(150 * SCALE))))

    def _make_rotation_curve_card(self) -> Image.Image:
        card_w = int(660 * SCALE)
        card_h = int(390 * SCALE)
        card = Image.new("RGBA", (card_w, card_h), (2, 5, 14, 220))
        draw = ImageDraw.Draw(card)
        margin = int(54 * SCALE)
        left, top = margin, int(112 * SCALE)
        right, bottom = card_w - margin, card_h - int(66 * SCALE)
        draw.rounded_rectangle(
            (0, 0, card_w - 1, card_h - 1),
            radius=int(26 * SCALE),
            fill=(2, 5, 14, 216),
            outline=(105, 150, 235, 92),
            width=max(1, int(2 * SCALE)),
        )
        draw_text(card, "THE CURVE THAT DOES NOT FALL", (margin, int(34 * SCALE)), 25, (236, 240, 250, 250), True, False, "la", 1)
        draw_text(card, f"{self.profile.galaxy}  //  SPARC", (margin, int(76 * SCALE)), 17, (135, 195, 245, 235), True, False, "la", 1)
        draw.line((left, bottom, right, bottom), fill=(115, 145, 180, 110), width=max(1, int(SCALE)))
        draw.line((left, top, left, bottom), fill=(115, 145, 180, 110), width=max(1, int(SCALE)))
        radius = self.profile.radius_kpc
        xmax = max(float(radius.max()), 1.0)
        ymax = max(float(self.profile.vobs_kms.max()) * 1.08, 1.0)

        def points(values: np.ndarray):
            return [
                (
                    left + float(r / xmax) * (right - left),
                    bottom - float(v / ymax) * (bottom - top),
                )
                for r, v in zip(radius, values)
            ]

        draw.line(points(self.profile.vobs_kms), fill=(244, 246, 252, 235), width=max(2, int(3 * SCALE)))
        draw.line(points(self.profile.vbar_kms), fill=(245, 174, 96, 220), width=max(1, int(2 * SCALE)))
        draw.line(points(self.profile.vdark_kms), fill=(105, 170, 255, 225), width=max(1, int(2 * SCALE)))
        draw_text(card, "observed", (right, top - int(10 * SCALE)), 14, (245, 247, 252, 235), False, False, "ra", 0)
        draw_text(card, "baryonic", (left, bottom + int(27 * SCALE)), 14, (245, 174, 96, 225), False, False, "la", 0)
        draw_text(card, "inferred dark component", (right, bottom + int(27 * SCALE)), 14, (105, 170, 255, 225), False, False, "ra", 0)
        return card

    def _camera(self, t: float):
        fraction = clamp(t / max(DURATION, 1e-9))
        key = interpolate_camera(fraction)
        # Very slow breathing prevents the path from feeling mechanically perfect.
        position = np.array(key.position, dtype=float)
        target = np.array(key.target, dtype=float)
        position += np.array(
            [
                0.035 * math.sin(t * 0.031),
                0.028 * math.sin(t * 0.023 + 1.2),
                0.024 * math.sin(t * 0.027 + 2.1),
            ]
        )
        target += np.array(
            [
                0.015 * math.sin(t * 0.019 + 0.5),
                0.018 * math.sin(t * 0.017 + 1.4),
                0.010 * math.sin(t * 0.021 + 2.8),
            ]
        )
        right, up, forward = camera_basis(position, target, key.roll_deg)
        return position, right, up, forward, key.fov_deg

    def _animate_disk(self, cloud: ParticleCloud, t: float, factor: float = 1.0) -> np.ndarray:
        radius_kpc = np.linalg.norm(cloud.xyz[:, :2], axis=1)
        v = np.interp(
            np.clip(radius_kpc, self.profile.radius_kpc[0], self.profile.radius_kpc[-1]),
            self.profile.radius_kpc,
            self.profile.vobs_kms,
        )
        omega = np.divide(v, np.maximum(radius_kpc, 0.12), out=np.zeros_like(v), where=radius_kpc > 0)
        omega /= max(float(np.nanpercentile(omega, 75)), 1e-6)
        angle = factor * (0.085 * t / max(DURATION / 600.0, 0.10)) * omega
        return rotate_z(cloud.xyz, angle)

    def _animate_halo(self, cloud: ParticleCloud, t: float) -> np.ndarray:
        outer = np.clip(cloud.radius_norm, 0.0, 1.0)
        angle = (0.030 + 0.024 * (1.0 - outer)) * t / max(DURATION / 600.0, 0.10)
        moved = rotate_z(cloud.xyz, angle)
        moved = rotate_xyz_global(
            moved,
            0.025 * math.sin(t * 0.011),
            0.032 * math.sin(t * 0.009 + 1.4),
            0.0,
        )
        breathing = 1.0 + 0.008 * np.sin(t * 0.025 + cloud.phase)
        return moved * breathing[:, None]

    def _project(self, xyz_kpc: np.ndarray, camera_state):
        position, right, up, forward, fov_deg = camera_state
        xyz = xyz_kpc / self.scale_kpc
        relative = xyz - position[None, :]
        camera_x = relative @ right
        camera_y = relative @ up
        depth = relative @ forward
        valid = depth > 0.035
        focal = 0.5 * self.rw / math.tan(math.radians(fov_deg) / 2.0)
        sx = self.rw * 0.5 + focal * camera_x / np.maximum(depth, 1e-6)
        sy = self.rh * 0.5 - focal * camera_y / np.maximum(depth, 1e-6)
        valid &= (sx >= 0.0) & (sx < self.rw) & (sy >= 0.0) & (sy < self.rh)
        return sx, sy, depth, valid

    def _rasterize(
        self,
        sx: np.ndarray,
        sy: np.ndarray,
        depth: np.ndarray,
        valid: np.ndarray,
        luminosity: np.ndarray,
        color_a: Tuple[float, float, float],
        color_b: Tuple[float, float, float],
        mix: np.ndarray,
        alpha_scale: float,
        exposure: float,
        blur_px: float,
    ) -> Image.Image:
        if not np.any(valid):
            return Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        xi = np.floor(sx[valid]).astype(np.int32)
        yi = np.floor(sy[valid]).astype(np.int32)
        idx = yi * self.rw + xi
        lum = luminosity[valid].astype(np.float32)
        depth_fade = 1.0 / np.maximum(depth[valid], 0.18) ** 0.42
        weight = np.clip(lum * depth_fade, 0.0, 14.0)
        local_mix = np.clip(mix[valid], 0.0, 1.0).astype(np.float32)
        count = self.rw * self.rh
        energy = np.bincount(idx, weights=weight, minlength=count).reshape(self.rh, self.rw)
        mix_energy = np.bincount(idx, weights=weight * local_mix, minlength=count).reshape(self.rh, self.rw)
        mean_mix = np.divide(mix_energy, energy, out=np.zeros_like(energy), where=energy > 0)
        intensity = 1.0 - np.exp(-energy * exposure)
        color_a_array = np.array(color_a, dtype=np.float32)
        color_b_array = np.array(color_b, dtype=np.float32)
        rgb = color_a_array[None, None, :] * (1.0 - mean_mix[..., None]) + color_b_array[None, None, :] * mean_mix[..., None]
        array = np.zeros((self.rh, self.rw, 4), dtype=np.uint8)
        array[..., :3] = np.clip(rgb * intensity[..., None], 0.0, 255.0).astype(np.uint8)
        array[..., 3] = np.clip(255.0 * intensity * alpha_scale, 0.0, 255.0).astype(np.uint8)
        image = Image.fromarray(array, "RGBA")
        if blur_px > 0:
            image = image.filter(ImageFilter.GaussianBlur(max(0.2, blur_px * self.render_scale)))
        return image.resize(OUT_SIZE, Image.Resampling.BILINEAR)

    def _draw_background(self, t: float, camera_state) -> Image.Image:
        canvas = Image.new("RGBA", OUT_SIZE, (0, 1, 7, 255))
        canvas.alpha_composite(self.background_nebula)
        xyz = self.particles.background_xyz
        # Background sphere is centered on the camera so it behaves like a skybox.
        position = camera_state[0]
        sky = xyz + position[None, :] * self.scale_kpc
        sx, sy, depth, valid = self._project(sky, camera_state)
        layer = self._rasterize(
            sx,
            sy,
            depth,
            valid,
            self.particles.background_luminosity,
            (135, 165, 220),
            (245, 245, 255),
            np.clip(self.particles.background_luminosity / 4.0, 0.0, 1.0),
            alpha_scale=0.92,
            exposure=1.4,
            blur_px=0.05,
        )
        canvas.alpha_composite(layer)
        return canvas

    def _draw_halo(self, canvas: Image.Image, t: float, camera_state) -> None:
        cloud = self.particles.halo
        xyz = self._animate_halo(cloud, t)
        sx, sy, depth, valid = self._project(xyz, camera_state)
        inner = 1.0 - np.clip(cloud.radius_norm, 0.0, 1.0)
        pulse = 0.72 + 0.28 * np.sin(0.025 * t + cloud.phase)
        lum = cloud.luminosity * (0.76 + 0.24 * pulse)
        glow = self._rasterize(
            sx,
            sy,
            depth,
            valid,
            lum,
            (55, 44, 145),
            (70, 205, 255),
            np.clip(0.22 + 0.78 * inner + 0.10 * cloud.temperature, 0.0, 1.0),
            alpha_scale=0.27,
            exposure=0.40,
            blur_px=5.0,
        )
        points = self._rasterize(
            sx,
            sy,
            depth,
            valid,
            lum,
            (70, 58, 165),
            (105, 215, 255),
            np.clip(0.18 + 0.72 * inner + 0.12 * cloud.temperature, 0.0, 1.0),
            alpha_scale=0.33,
            exposure=0.84,
            blur_px=0.45,
        )
        canvas.alpha_composite(glow)
        canvas.alpha_composite(points)

    def _draw_disk(self, canvas: Image.Image, t: float, camera_state) -> None:
        disk = self.particles.disk
        disk_xyz = self._animate_disk(disk, t, factor=1.0)
        sx, sy, depth, valid = self._project(disk_xyz, camera_state)
        disk_glow = self._rasterize(
            sx,
            sy,
            depth,
            valid,
            disk.luminosity,
            (255, 154, 74),
            (120, 205, 255),
            disk.temperature,
            alpha_scale=0.72,
            exposure=0.62,
            blur_px=3.8,
        )
        disk_points = self._rasterize(
            sx,
            sy,
            depth,
            valid,
            disk.luminosity,
            (255, 190, 105),
            (185, 225, 255),
            disk.temperature,
            alpha_scale=0.94,
            exposure=1.25,
            blur_px=0.18,
        )
        canvas.alpha_composite(disk_glow)
        canvas.alpha_composite(disk_points)

        bulge = self.particles.bulge
        bulge_xyz = self._animate_disk(bulge, t, factor=0.68)
        sx, sy, depth, valid = self._project(bulge_xyz, camera_state)
        bulge_glow = self._rasterize(
            sx,
            sy,
            depth,
            valid,
            bulge.luminosity,
            (255, 126, 54),
            (255, 235, 185),
            bulge.temperature,
            alpha_scale=0.90,
            exposure=0.76,
            blur_px=6.5,
        )
        bulge_points = self._rasterize(
            sx,
            sy,
            depth,
            valid,
            bulge.luminosity,
            (255, 154, 67),
            (255, 242, 202),
            bulge.temperature,
            alpha_scale=0.95,
            exposure=1.22,
            blur_px=0.30,
        )
        canvas.alpha_composite(bulge_glow)
        canvas.alpha_composite(bulge_points)

    def _draw_dust(self, canvas: Image.Image, t: float, camera_state) -> None:
        cloud = self.particles.dust
        xyz = self._animate_disk(cloud, t, factor=0.92)
        sx, sy, depth, valid = self._project(xyz, camera_state)
        layer = self._rasterize(
            sx,
            sy,
            depth,
            valid,
            cloud.luminosity,
            (62, 8, 90),
            (4, 75, 108),
            cloud.temperature,
            alpha_scale=0.34,
            exposure=0.48,
            blur_px=7.0,
        )
        canvas.alpha_composite(layer)

    def _draw_filaments(self, canvas: Image.Image, t: float, camera_state) -> None:
        overlay = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        halo_r = self.profile.halo_radius_kpc
        for filament in range(7):
            points_3d = []
            count = 180
            phase = filament * (2.0 * np.pi / 7.0) + 0.015 * t
            for index in range(count):
                u = index / (count - 1)
                radius = halo_r * (0.18 + 0.78 * u)
                theta = phase + 4.2 * u + 0.20 * math.sin(5.0 * u + filament)
                z = halo_r * 0.22 * math.sin(2.3 * theta + filament * 0.8) * (0.25 + 0.75 * u)
                points_3d.append((radius * math.cos(theta), radius * math.sin(theta), z))
            xyz = np.asarray(points_3d, dtype=float)
            sx, sy, depth, valid = self._project(xyz, camera_state)
            segment: List[Tuple[float, float]] = []
            for x, y, ok in zip(sx, sy, valid):
                if ok:
                    segment.append((float(x / self.render_scale), float(y / self.render_scale)))
                elif len(segment) > 1:
                    draw.line(segment, fill=(80, 132, 255, 22), width=max(1, int(SCALE)))
                    segment = []
            if len(segment) > 1:
                draw.line(segment, fill=(80, 132, 255, 22), width=max(1, int(SCALE)))
        overlay = overlay.filter(ImageFilter.GaussianBlur(max(0.6, 1.4 * SCALE)))
        canvas.alpha_composite(overlay)

    def _draw_lensing_arcs(self, canvas: Image.Image, t: float, camera_state) -> None:
        center = np.array([[0.0, 0.0, 0.0]])
        sx, sy, depth, valid = self._project(center, camera_state)
        if not valid[0]:
            return
        x = float(sx[0] / self.render_scale)
        y = float(sy[0] / self.render_scale)
        focal = 0.5 * WIDTH / math.tan(math.radians(camera_state[4]) / 2.0)
        radius = clamp(focal * 0.055 / max(float(depth[0]), 0.2), 18 * SCALE, 170 * SCALE)
        overlay = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        phase = 18.0 * math.sin(t * 0.012)
        for multiplier, alpha, width in [(1.0, 74, 2), (1.34, 34, 1), (1.72, 18, 1)]:
            r = radius * multiplier
            draw.arc(
                (x - r, y - 0.63 * r, x + r, y + 0.63 * r),
                start=24 + phase,
                end=145 + phase,
                fill=(115, 195, 255, alpha),
                width=max(1, int(width * SCALE)),
            )
            draw.arc(
                (x - r, y - 0.63 * r, x + r, y + 0.63 * r),
                start=202 + phase,
                end=324 + phase,
                fill=(145, 102, 255, alpha),
                width=max(1, int(width * SCALE)),
            )
        overlay = overlay.filter(ImageFilter.GaussianBlur(max(0.5, 1.0 * SCALE)))
        canvas.alpha_composite(overlay)

    def _chapter(self, fraction: float) -> Tuple[int, str, float]:
        for index in range(len(CHAPTERS) - 1):
            start, title = CHAPTERS[index]
            end = CHAPTERS[index + 1][0]
            if start <= fraction < end:
                return index, title, (fraction - start) / max(end - start, 1e-8)
        index = len(CHAPTERS) - 1
        return index, CHAPTERS[-1][1], 1.0

    def _draw_titles_and_cards(self, canvas: Image.Image, t: float) -> None:
        fraction = clamp(t / max(DURATION, 1e-8))
        intro_length = min(18.0, DURATION * 0.09)
        if t < intro_length:
            fade_in = smoothstep(t / max(3.2, intro_length * 0.24))
            fade_out = 1.0 - smoothstep((t - intro_length * 0.72) / max(intro_length * 0.28, 1e-6))
            alpha = int(255 * fade_in * fade_out)
            draw_text(
                canvas,
                str(CONFIG["title"]),
                (WIDTH * 0.5, HEIGHT * 0.43),
                88,
                (244, 246, 252, alpha),
                True,
                True,
                "mm",
                2,
            )
            draw_text(
                canvas,
                str(CONFIG["subtitle"]),
                (WIDTH * 0.5, HEIGHT * 0.53),
                31,
                (122, 188, 246, int(alpha * 0.92)),
                True,
                False,
                "mm",
                1,
            )
            draw_text(
                canvas,
                f"A data-constrained voyage through {self.profile.galaxy}",
                (WIDTH * 0.5, HEIGHT * 0.61),
                18,
                (195, 208, 226, int(alpha * 0.78)),
                False,
                False,
                "mm",
                0,
            )

        chapter_index, chapter_title, chapter_u = self._chapter(fraction)
        chapter_alpha = int(210 * (1.0 - smoothstep((chapter_u - 0.02) / 0.14)))
        if chapter_index > 0 and chapter_alpha > 0:
            draw_text(
                canvas,
                f"0{chapter_index + 1}",
                (76 * SCALE, 76 * SCALE),
                15,
                (115, 176, 235, chapter_alpha),
                True,
                False,
                "la",
                0,
            )
            draw_text(
                canvas,
                chapter_title.upper(),
                (76 * SCALE, 108 * SCALE),
                22,
                (227, 234, 246, chapter_alpha),
                True,
                False,
                "la",
                1,
            )

        if bool(CONFIG["show_science_cards"]):
            card_center = 0.455
            card_half = min(0.055, 14.0 / max(DURATION, 1.0))
            if abs(fraction - card_center) < card_half:
                u = 1.0 - abs(fraction - card_center) / card_half
                alpha = smoothstep(u)
                card = self.rotation_curve_card.copy()
                card.putalpha(card.getchannel("A").point(lambda p: int(p * alpha)))
                x = int(WIDTH * 0.06)
                y = int(HEIGHT * 0.50 - card.height * 0.5)
                canvas.alpha_composite(card, (x, y))

            truth_center = 0.785
            truth_half = min(0.052, 13.0 / max(DURATION, 1.0))
            if abs(fraction - truth_center) < truth_half:
                u = 1.0 - abs(fraction - truth_center) / truth_half
                alpha = int(235 * smoothstep(u))
                panel = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
                draw = ImageDraw.Draw(panel)
                x0, y0 = int(WIDTH * 0.57), int(HEIGHT * 0.58)
                w, h = int(650 * SCALE), int(205 * SCALE)
                draw.rounded_rectangle(
                    (x0, y0, x0 + w, y0 + h),
                    radius=int(24 * SCALE),
                    fill=(2, 5, 14, int(alpha * 0.80)),
                    outline=(105, 165, 240, int(alpha * 0.34)),
                    width=max(1, int(2 * SCALE)),
                )
                draw_text(panel, "WHAT YOU SEE IS A METAPHOR", (x0 + 28 * SCALE, y0 + 30 * SCALE), 21, (235, 240, 250, alpha), True, False, "la", 1)
                draw_wrapped_text(
                    panel,
                    "Dark matter emits no visible light. This halo is sampled from the gravitational mass implied by the measured rotation curve.",
                    (int(x0 + 28 * SCALE), int(y0 + 78 * SCALE)),
                    int(w - 56 * SCALE),
                    18,
                    (178, 205, 230, alpha),
                )
                canvas.alpha_composite(panel)

        if fraction > 0.93:
            u = smoothstep((fraction - 0.93) / 0.04) * (1.0 - smoothstep((fraction - 0.985) / 0.015))
            alpha = int(245 * u)
            draw_text(canvas, "THE LIGHT ENDS.", (WIDTH * 0.5, HEIGHT * 0.45), 44, (240, 243, 250, alpha), True, True, "mm", 2)
            draw_text(canvas, "THE GRAVITY DOES NOT.", (WIDTH * 0.5, HEIGHT * 0.53), 44, (132, 188, 246, alpha), True, True, "mm", 2)
            draw_text(canvas, "Data: SPARC  •  Lelli, McGaugh & Schombert (2016)", (WIDTH * 0.5, HEIGHT * 0.63), 16, (190, 204, 224, int(alpha * 0.75)), False, False, "mm", 0)

    def _post_process(self, canvas: Image.Image, t: float) -> np.ndarray:
        image = canvas.convert("RGB")
        image = ImageEnhance.Contrast(image).enhance(1.10)
        image = ImageEnhance.Color(image).enhance(1.06)
        array = np.asarray(image).astype(np.float32)
        array *= VIGNETTE[..., None]

        # Fine deterministic film grain. It is subtle enough not to damage compression.
        frame_number = int(round(t * FPS))
        rng = np.random.default_rng(frame_number + 99017)
        gh, gw = max(45, HEIGHT // 10), max(80, WIDTH // 10)
        grain_small = rng.normal(0.0, 1.8, (gh, gw)).astype(np.float32)
        grain = np.asarray(
            Image.fromarray(grain_small, mode="F").resize(OUT_SIZE, Image.Resampling.BILINEAR),
            dtype=np.float32,
        )
        array += grain[..., None]

        fade_in = smoothstep(t / max(min(4.0, DURATION * 0.03), 1e-6))
        fade_out = 1.0 - smoothstep((t - (DURATION - min(5.0, DURATION * 0.04))) / max(min(5.0, DURATION * 0.04), 1e-6))
        array *= fade_in * fade_out
        return np.clip(array, 0.0, 255.0).astype(np.uint8)

    def render_frame(self, t: float) -> np.ndarray:
        camera_state = self._camera(t)
        canvas = self._draw_background(t, camera_state)
        self._draw_halo(canvas, t, camera_state)
        self._draw_filaments(canvas, t, camera_state)
        self._draw_dust(canvas, t, camera_state)
        self._draw_disk(canvas, t, camera_state)
        self._draw_lensing_arcs(canvas, t, camera_state)
        self._draw_titles_and_cards(canvas, t)
        return self._post_process(canvas, t)


# =============================================================================
# Ambient soundtrack
# =============================================================================


@dataclass
class ChimeEvent:
    start: float
    frequency: float
    duration: float
    amplitude: float
    pan: float


def make_chime_events(duration: float) -> List[ChimeEvent]:
    rng = np.random.default_rng(12031998)
    events: List[ChimeEvent] = []
    cursor = 18.0
    scale = np.array([1.0, 9 / 8, 6 / 5, 4 / 3, 3 / 2, 8 / 5, 16 / 9, 2.0])
    while cursor < duration - 8.0:
        cursor += float(rng.uniform(10.0, 28.0))
        fundamental = float(rng.choice([41.203, 55.0, 61.735, 73.416]))
        ratio = float(rng.choice(scale))
        events.append(
            ChimeEvent(
                start=cursor,
                frequency=fundamental * ratio * float(rng.choice([1.0, 2.0, 4.0])),
                duration=float(rng.uniform(4.0, 10.0)),
                amplitude=float(rng.uniform(0.018, 0.050)),
                pan=float(rng.uniform(-0.85, 0.85)),
            )
        )
    return events


def shaped_noise(rng: np.random.Generator, count: int, sample_rate: int) -> np.ndarray:
    white = rng.normal(0.0, 1.0, count)
    spectrum = np.fft.rfft(white)
    frequencies = np.fft.rfftfreq(count, d=1.0 / sample_rate)
    shape = 1.0 / np.sqrt(1.0 + (frequencies / 95.0) ** 2.2)
    shape *= 1.0 - np.exp(-frequencies / 7.0)
    noise = np.fft.irfft(spectrum * shape, n=count)
    rms = float(np.sqrt(np.mean(noise * noise)))
    return noise / max(rms, 1e-9)


def render_ambient_audio(path: Path, duration: float, sample_rate: int = 48_000) -> Path:
    print("Generating procedural ambient soundtrack:", path)
    rng = np.random.default_rng(441122)
    events = make_chime_events(duration)
    block_seconds = 8.0
    total_samples = int(round(duration * sample_rate))
    fade_seconds = min(8.0, duration * 0.08)

    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        for start_sample in tqdm(range(0, total_samples, int(block_seconds * sample_rate)), desc="Ambient audio"):
            count = min(int(block_seconds * sample_rate), total_samples - start_sample)
            absolute_t = (start_sample + np.arange(count)) / sample_rate
            slow = absolute_t / max(duration, 1.0)

            left = np.zeros(count, dtype=np.float64)
            right = np.zeros(count, dtype=np.float64)

            fundamentals = [27.5, 41.203, 55.0, 73.416]
            for index, frequency in enumerate(fundamentals):
                phase = 0.7 * index + 0.16 * np.sin(2.0 * np.pi * absolute_t / (47.0 + 11.0 * index))
                drift = 1.0 + 0.0018 * np.sin(2.0 * np.pi * absolute_t / (31.0 + 7.0 * index) + index)
                signal = np.sin(2.0 * np.pi * frequency * drift * absolute_t + phase)
                signal += 0.31 * np.sin(2.0 * np.pi * frequency * 2.0 * absolute_t + phase * 1.7)
                signal += 0.12 * np.sin(2.0 * np.pi * frequency * 3.0 * absolute_t + phase * 2.2)
                envelope = 0.038 + 0.020 * np.sin(2.0 * np.pi * absolute_t / (69.0 + index * 13.0) + index * 1.1)
                pan = math.sin(index * 1.7) * 0.48
                left += signal * envelope * math.sqrt((1.0 - pan) * 0.5)
                right += signal * envelope * math.sqrt((1.0 + pan) * 0.5)

            noise = shaped_noise(rng, count, sample_rate)
            wind_env = 0.013 + 0.008 * np.square(np.sin(2.0 * np.pi * absolute_t / 83.0 + 0.4))
            left += noise * wind_env
            right += np.roll(noise, min(421, count - 1)) * wind_env * 0.96

            # A very low sub pulse suggests gravitational scale without becoming musical percussion.
            pulse_phase = np.mod(absolute_t, 17.0)
            pulse_env = np.exp(-pulse_phase / 2.8) * (pulse_phase < 8.0)
            sub = np.sin(2.0 * np.pi * 22.0 * absolute_t) * pulse_env * 0.018
            left += sub
            right += sub

            block_start = start_sample / sample_rate
            block_end = (start_sample + count) / sample_rate
            for event in events:
                if event.start + event.duration < block_start or event.start > block_end:
                    continue
                local = absolute_t - event.start
                active = (local >= 0.0) & (local <= event.duration)
                if not np.any(active):
                    continue
                x = local[active]
                attack = np.clip(x / 0.22, 0.0, 1.0)
                decay = np.exp(-x / max(event.duration * 0.34, 0.8))
                shimmer = (
                    np.sin(2.0 * np.pi * event.frequency * x)
                    + 0.48 * np.sin(2.0 * np.pi * event.frequency * 2.01 * x + 0.2)
                    + 0.22 * np.sin(2.0 * np.pi * event.frequency * 3.98 * x + 0.8)
                )
                signal = shimmer * attack * decay * event.amplitude
                left[active] += signal * math.sqrt((1.0 - event.pan) * 0.5)
                right[active] += signal * math.sqrt((1.0 + event.pan) * 0.5)

            fade_in = np.clip(absolute_t / max(fade_seconds, 1e-6), 0.0, 1.0)
            fade_out = np.clip((duration - absolute_t) / max(fade_seconds, 1e-6), 0.0, 1.0)
            master = np.minimum(fade_in, fade_out)
            # Slight chapter swell in the middle of the voyage.
            master *= 0.82 + 0.18 * np.sin(np.pi * np.clip(slow, 0.0, 1.0))
            stereo = np.column_stack((left, right)) * master[:, None]
            stereo = np.tanh(stereo * 1.34) * 0.82
            pcm = np.clip(stereo * 32767.0, -32768, 32767).astype("<i2")
            wav.writeframes(pcm.tobytes())
    return path


# =============================================================================
# Metadata, subtitles, previews, and final rendering
# =============================================================================


def write_chapter_srt(path: Path) -> Path:
    lines: List[str] = []
    for index, (fraction, title) in enumerate(CHAPTERS, 1):
        start = fraction * DURATION
        end_fraction = CHAPTERS[index][0] if index < len(CHAPTERS) else 1.0
        end = min(end_fraction * DURATION, start + max(6.0, DURATION * 0.045))
        lines += [str(index), f"{format_srt_time(start)} --> {format_srt_time(end)}", title, ""]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_metadata(profile: GalaxyProfile) -> Tuple[Path, Path]:
    description = YOUTUBE_DESCRIPTION_TEMPLATE.format(
        galaxy=profile.galaxy,
        distance_mpc=profile.distance_mpc,
        rdisk_kpc=profile.rdisk_kpc,
        vflat_kms=profile.vflat_kms,
        quality=profile.quality,
    )
    metadata_path = OUTPUT_ROOT / "youtube_title_and_description.txt"
    metadata_path.write_text(
        f"TITLE\n{CONFIG['youtube_title']}\n\nDESCRIPTION\n{description}", encoding="utf-8"
    )
    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "film_title": CONFIG["youtube_title"],
        "render": {
            "width": WIDTH,
            "height": HEIGHT,
            "fps": FPS,
            "duration_seconds": DURATION,
            "quick_mode": QUICK_MODE,
            "preview_only": PREVIEW_ONLY,
        },
        "galaxy_profile": profile.serializable(),
        "scientific_note": (
            "Dark matter is not directly imaged. Halo particles are a visual metaphor "
            "sampled from the dark mass implied by the SPARC rotation-curve residual."
        ),
        "data_urls": {
            "SPARC": CONFIG["master_data_url"],
            "SPARC_rotation_archive": CONFIG["rotation_zip_url"],
            "VizieR": "J/AJ/152/157",
        },
    }
    manifest_path = DATA_ROOT / "render_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return metadata_path, manifest_path


def create_contact_sheet(paths: Sequence[Path]) -> Optional[Path]:
    if not paths:
        return None
    images = [Image.open(path).convert("RGB") for path in paths]
    thumb_w = 480 if WIDTH >= 1920 else 320
    thumb_h = int(thumb_w * HEIGHT / WIDTH)
    margin = 24
    sheet = Image.new("RGB", (thumb_w * 3 + margin * 4, thumb_h * 2 + margin * 3), (2, 3, 9))
    for index, image in enumerate(images[:6]):
        thumb = image.resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)
        x = margin + (index % 3) * (thumb_w + margin)
        y = margin + (index // 3) * (thumb_h + margin)
        sheet.paste(thumb, (x, y))
    path = PREVIEW_DIR / "dark_matter_contact_sheet.jpg"
    sheet.save(path, quality=93)
    return path


def render_previews(scene: DarkMatterScene) -> List[Path]:
    fractions = [0.035, 0.18, 0.34, 0.51, 0.72, 0.94]
    paths: List[Path] = []
    for index, fraction in enumerate(tqdm(fractions, desc="Preview frames"), 1):
        t = fraction * DURATION
        frame = scene.render_frame(t)
        path = PREVIEW_DIR / f"preview_{index:02d}_{t:07.2f}s.png"
        Image.fromarray(frame).save(path)
        paths.append(path)
    return paths


def render_video(scene: DarkMatterScene) -> Path:
    basename = str(CONFIG["output_basename"])
    raw_path = OUTPUT_ROOT / f"{basename}_silent.mp4"
    final_path = OUTPUT_ROOT / f"{basename}_final.mp4"
    audio_path = AUDIO_DIR / f"{basename}_ambient.wav"
    srt_path = OUTPUT_ROOT / f"{basename}_chapters.srt"

    if bool(CONFIG["write_subtitle_sidecar"]):
        write_chapter_srt(srt_path)

    frame_count = int(round(DURATION * FPS))
    times = np.arange(frame_count, dtype=float) / FPS
    print(f"Rendering {frame_count:,} frames at {WIDTH}x{HEIGHT}, {FPS} fps...")
    with iio.get_writer(
        raw_path,
        fps=FPS,
        codec="libx264",
        pixelformat="yuv420p",
        macro_block_size=None,
        ffmpeg_params=["-crf", "18", "-preset", "medium", "-movflags", "+faststart"],
    ) as writer:
        for t in tqdm(times, desc="Rendering dark-matter voyage"):
            writer.append_data(scene.render_frame(float(t)))

    ffmpeg = find_ffmpeg()
    requested_audio = CONFIG.get("audio_path")
    if requested_audio and Path(str(requested_audio)).exists():
        soundtrack = Path(str(requested_audio))
    else:
        soundtrack = render_ambient_audio(audio_path, DURATION, int(CONFIG["audio_sample_rate"]))

    if ffmpeg:
        candidate = raw_path
        if bool(CONFIG["burn_subtitles"]) and srt_path.exists():
            subbed_path = OUTPUT_ROOT / f"{basename}_subbed.mp4"
            escaped = str(srt_path).replace("\\", "\\\\").replace(":", "\\:")
            run_ffmpeg(
                [
                    ffmpeg,
                    "-y",
                    "-i",
                    str(candidate),
                    "-vf",
                    f"subtitles={escaped}:force_style=Fontname=DejaVu Sans,Fontsize=19,Outline=1.0,BorderStyle=3,MarginV=46",
                    "-c:v",
                    "libx264",
                    "-crf",
                    "18",
                    "-preset",
                    "medium",
                    "-an",
                    str(subbed_path),
                ]
            )
            candidate = subbed_path
        run_ffmpeg(
            [
                ffmpeg,
                "-y",
                "-i",
                str(candidate),
                "-i",
                str(soundtrack),
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-b:a",
                "256k",
                "-shortest",
                "-movflags",
                "+faststart",
                str(final_path),
            ]
        )
    else:
        print("ffmpeg was not found; the silent video is being copied as the final file.")
        shutil.copyfile(raw_path, final_path)
    return final_path


def load_profile() -> GalaxyProfile:
    master, data_source = fetch_sparc_master(FORCE_REFRESH)
    row = choose_galaxy_row(master, SELECTED_GALAXY)
    zip_path = download_rotation_zip(FORCE_REFRESH)
    curve = extract_rotation_curve(zip_path, str(row.get("Galaxy", SELECTED_GALAXY)))
    if curve is None:
        curve = fallback_rotation_curve(row)
        curve_source = "analytic fallback shaped by SPARC master parameters"
    else:
        curve_source = "SPARC Rotmod_LTG observed rotation curve"
    return build_galaxy_profile(row, curve, data_source, curve_source)


def main() -> None:
    print("Starting DARK MATTER: THE SILENT ARCHITECTURE")
    print("Quick mode:", QUICK_MODE)
    print("Preview only:", PREVIEW_ONLY)
    print("Requested galaxy:", SELECTED_GALAXY)

    profile = load_profile()
    print(f"Selected galaxy: {profile.galaxy}")
    print(f"Master data source: {profile.data_source}")
    print(f"Rotation curve: {profile.rotation_curve_source}")
    print(f"Distance: {profile.distance_mpc:.2f} Mpc")
    print(f"Disk scale length: {profile.rdisk_kpc:.2f} kpc")
    print(f"Flat velocity: {profile.vflat_kms:.1f} km/s")
    print(f"Halo visualization radius: {profile.halo_radius_kpc:.1f} kpc")

    metadata_path, manifest_path = write_metadata(profile)
    print("YouTube title and description:", metadata_path.resolve())
    print("Render manifest:", manifest_path.resolve())

    particles = ParticleFactory(profile).build()
    scene = DarkMatterScene(profile, particles)
    preview_paths = render_previews(scene)
    contact_sheet = create_contact_sheet(preview_paths)
    if contact_sheet:
        print("Contact sheet:", contact_sheet.resolve())

    if not PREVIEW_ONLY:
        final_path = render_video(scene)
        print("Final film:", final_path.resolve())
    else:
        print("Preview-only mode complete; no movie was encoded.")

    print("Output directory:", OUTPUT_ROOT.resolve())
    for path in sorted(OUTPUT_ROOT.rglob("*")):
        if path.is_file():
            print("-", path.relative_to(OUTPUT_ROOT))


if __name__ == "__main__":
    main()
