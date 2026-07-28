from __future__ import annotations

"""
DARK MATTER: THE SILENT ARCHITECTURE
====================================

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
