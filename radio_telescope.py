from __future__ import annotations

"""
RADIO TELESCOPE + FOURIER TRANSFORM — A Very Easy Visual Explainer
===================================================================

A self-contained, cinematic Python renderer inspired by a shot-based astronomy
short workflow. It explains, visually and with beginner-friendly captions:

1) Radio light is just light with a long wavelength.
2) A dish collects weak radio waves and focuses them into a receiver.
3) A single dish has limited angular resolution: theta ~ 1.22 lambda / D.
4) Two dishes receive the same wave at slightly different times/phases.
5) A correlator compares those signals and produces a "visibility".
6) Each antenna pair (baseline) measures one piece of the sky's 2-D Fourier transform.
7) Many baselines fill the u-v plane (spatial-frequency plane).
8) An inverse Fourier transform turns the measured visibilities into an image.
9) Earth rotation changes the projected baselines, filling in more Fourier samples.

This is intentionally an educational visualization, not a telescope simulator.
Real radio interferometry also requires calibration, bandpass corrections,
atmospheric/ionospheric corrections, weighting, gridding, deconvolution (e.g.
CLEAN), primary-beam correction, and many other engineering/science steps.

Official background references:
- ALMA Interferometry:
  https://www.almaobservatory.org/en/about-alma/how-alma-works/technologies/interferometry/
- ALMA How It Works:
  https://www.almaobservatory.org/en/about-alma/how-alma-works/
- NRAO Interferometry Explained:
  https://public.nrao.edu/interferometry-explained/
  
Usage
-----
Normal 720p render:
    python radio_telescope_fourier_explainer.py

Fast validation preview:
    RADIO_TELESCOPE_QUICK=1 python radio_telescope_fourier_explainer.py

1080p render:
    RADIO_TELESCOPE_FULLHD=1 python radio_telescope_fourier_explainer.py

Dependencies:
    pip install numpy pillow imageio imageio-ffmpeg

The script will create:
    radio_telescope_fourier_output/
        radio_telescope_fourier_explainer.mp4
        radio_telescope_fourier_explainer.srt
        radio_telescope_fourier_narration.txt
        radio_telescope_fourier_summary.json
        previews/contact_sheet.jpg

Optional voice-over:
The .srt and narration .txt are generated so you can record or synthesize a
voice-over separately and mux it later. A restrained procedural soundtrack is
created automatically.
"""

import json
import math
import os
import shutil
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import imageio.v2 as iio
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont


# =============================================================================
# Configuration
# =============================================================================

QUICK_MODE = os.environ.get("RADIO_TELESCOPE_QUICK", "0") == "1"
FULLHD_MODE = os.environ.get("RADIO_TELESCOPE_FULLHD", "0") == "1" and not QUICK_MODE

OUTPUT_ROOT = Path("radio_telescope_fourier_output")
PREVIEW_DIR = OUTPUT_ROOT / "previews"
for directory in (OUTPUT_ROOT, PREVIEW_DIR):
    directory.mkdir(parents=True, exist_ok=True)

if QUICK_MODE:
    OUT_W, OUT_H, FPS, DURATION = 640, 360, 8, 26.0
elif FULLHD_MODE:
    OUT_W, OUT_H, FPS, DURATION = 1920, 1080, 24, 110.0
else:
    OUT_W, OUT_H, FPS, DURATION = 1280, 720, 24, 110.0

FRAME_COUNT = int(round(FPS * DURATION))
AUDIO_RATE = 44100

COLORS = {
    "bg0": (2, 5, 14),
    "bg1": (8, 15, 32),
    "white": (244, 248, 255),
    "muted": (165, 181, 198),
    "cyan": (90, 225, 245),
    "blue": (110, 160, 255),
    "gold": (246, 194, 92),
    "violet": (173, 130, 238),
    "pink": (246, 132, 189),
    "green": (125, 229, 177),
    "red": (244, 113, 118),
    "dish": (188, 201, 215),
    "grid": (80, 100, 125),
}

# Full-length chapter plan. Quick mode rescales this timeline automatically.
SHOT_PLAN_FULL: List[Tuple[str, float, float]] = [
    ("hook", 0.0, 7.0),
    ("radio_is_light", 7.0, 16.0),
    ("dish", 16.0, 27.0),
    ("single_dish_resolution", 27.0, 37.0),
    ("two_dishes", 37.0, 49.0),
    ("phase_and_correlation", 49.0, 61.0),
    ("fourier_intuition", 61.0, 74.0),
    ("uv_sampling", 74.0, 87.0),
    ("inverse_fourier", 87.0, 100.0),
    ("earth_rotation", 100.0, 106.0),
    ("finale", 106.0, 110.0),
]

NARRATION_FULL: List[Tuple[float, float, str]] = [
    (0.2, 6.7, "A radio telescope is a camera for invisible light. But instead of directly taking a picture, it first measures waves."),
    (7.2, 15.7, "Radio waves are the same electromagnetic family as visible light. The big difference is wavelength: radio wavelengths can be millimeters, centimeters, or even meters long."),
    (16.2, 26.7, "A parabolic dish catches extremely weak radio waves over a large area. Its curved surface redirects those waves toward a receiver, where electronics turn the signal into numbers."),
    (27.2, 36.7, "One dish can make a sky map, but its sharpness is limited. Roughly, angular resolution is wavelength divided by dish diameter. Longer waves need a much larger telescope for the same detail."),
    (37.2, 48.7, "So astronomers spread several dishes apart. The same wave reaches each dish at a slightly different time. The distance between two antennas is called a baseline."),
    (49.2, 60.7, "That tiny arrival-time difference appears as a phase shift in the wave. A correlator compares the two signals. The result stores how strongly that baseline responds, and the phase of that response."),
    (61.2, 73.7, "Now comes the Fourier idea. Any picture can be described as a mixture of smooth large-scale patterns and fine repeating patterns. Fourier space tells us how much of each spatial pattern exists."),
    (74.2, 86.7, "Each antenna pair samples one spatial frequency in a plane astronomers call the u-v plane. Short baselines measure broad structure. Long baselines measure fine detail. Many pairs give many Fourier samples."),
    (87.2, 99.7, "We place the measured complex visibilities into Fourier space. Because we only sampled some points, the first image is imperfect. An inverse Fourier transform converts those measurements back into an image of the sky."),
    (100.2, 105.7, "As Earth rotates, the projected baselines change direction. That traces extra paths through the u-v plane, filling missing information and improving the final reconstruction."),
    (106.1, 109.8, "So the pipeline is simple: collect waves, compare antennas, sample Fourier space, then transform those samples back into a radio image."),
]

if QUICK_MODE:
    SCALE = DURATION / 110.0
    SHOT_PLAN = [(name, a * SCALE, b * SCALE) for name, a, b in SHOT_PLAN_FULL]
    NARRATION = [(a * SCALE, b * SCALE, text) for a, b, text in NARRATION_FULL]
else:
    SHOT_PLAN = SHOT_PLAN_FULL
    NARRATION = NARRATION_FULL


# =============================================================================
# General helpers
# =============================================================================

def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(value)))


def smoothstep(value: float) -> float:
    x = clamp(value)
    return x * x * (3.0 - 2.0 * x)


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def ease_out(value: float) -> float:
    x = clamp(value)
    return 1.0 - (1.0 - x) ** 3


def get_shot(t: float) -> Tuple[str, float, float]:
    for shot in SHOT_PLAN:
        if shot[1] <= t < shot[2]:
            return shot
    return SHOT_PLAN[-1]


def narration_at(t: float) -> Optional[str]:
    for start, end, text in NARRATION:
        if start <= t < end:
            return text
    return None


def format_srt_time(seconds: float) -> str:
    ms = int(round(seconds * 1000.0))
    hours = ms // 3_600_000
    ms %= 3_600_000
    minutes = ms // 60_000
    ms %= 60_000
    secs = ms // 1000
    ms %= 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def write_srt(captions: Sequence[Tuple[float, float, str]], path: Path):
    rows: List[str] = []
    for index, (start, end, text) in enumerate(captions, 1):
        rows.extend([str(index), f"{format_srt_time(start)} --> {format_srt_time(end)}", text, ""])
    path.write_text("\n".join(rows), encoding="utf-8")


def get_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
        "arialbd.ttf" if bold else "arial.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=max(8, int(size)))
        except Exception:
            continue
    return ImageFont.load_default()


def text(image: Image.Image, value: str, xy: Tuple[int, int], size: int, fill, *, bold=False, anchor="la", stroke=2):
    ImageDraw.Draw(image).text(
        xy,
        value,
        font=get_font(size, bold=bold),
        fill=fill,
        anchor=anchor,
        stroke_width=stroke,
        stroke_fill=(0, 0, 0, 210),
    )


def wrapped_text(image: Image.Image, value: str, box: Tuple[int, int, int, int], size: int, fill, *, bold=False, line_gap=0.22):
    draw = ImageDraw.Draw(image)
    font = get_font(size, bold=bold)
    x0, y0, x1, y1 = box
    words = value.split()
    lines: List[str] = []
    current = ""
    for word in words:
        candidate = word if not current else current + " " + word
        bbox = draw.textbbox((0, 0), candidate, font=font, stroke_width=2)
        if bbox[2] - bbox[0] <= x1 - x0:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)

    y = y0
    for line in lines:
        draw.text((x0, y), line, font=font, fill=fill, stroke_width=2, stroke_fill=(0, 0, 0, 210))
        bbox = draw.textbbox((x0, y), line, font=font, stroke_width=2)
        y += (bbox[3] - bbox[1]) + int(size * line_gap)
        if y > y1:
            break


def rounded_panel(image: Image.Image, box: Tuple[int, int, int, int], alpha: int = 140, outline_alpha: int = 40):
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    radius = max(10, int(min(box[2] - box[0], box[3] - box[1]) * 0.06))
    draw.rounded_rectangle(box, radius=radius, fill=(4, 8, 18, alpha), outline=COLORS["cyan"] + (outline_alpha,), width=max(1, OUT_W // 1000))
    image.alpha_composite(overlay)


def draw_arrow(draw: ImageDraw.ImageDraw, p0: Tuple[float, float], p1: Tuple[float, float], fill, width: int = 3, head: float = 10.0):
    draw.line((p0[0], p0[1], p1[0], p1[1]), fill=fill, width=width)
    angle = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
    left = (p1[0] - head * math.cos(angle - 0.55), p1[1] - head * math.sin(angle - 0.55))
    right = (p1[0] - head * math.cos(angle + 0.55), p1[1] - head * math.sin(angle + 0.55))
    draw.polygon([p1, left, right], fill=fill)


def make_vignette(width: int, height: int, strength: float = 0.24) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width]
    nx = (xx - width / 2) / (width / 2)
    ny = (yy - height / 2) / (height / 2)
    r = np.sqrt(nx * nx + ny * ny)
    return np.clip(1.0 - strength * r ** 1.7, 0.0, 1.0).astype(np.float32)


VIGNETTE = make_vignette(OUT_W, OUT_H)


@dataclass
class Star:
    x: float
    y: float
    radius: float
    alpha: int
    phase: float


# =============================================================================
# Scientific / procedural data
# =============================================================================

def make_sky_model(size: int = 192) -> np.ndarray:
    """Synthetic radio sky brightness image in arbitrary units."""
    yy, xx = np.mgrid[0:size, 0:size]
    x = (xx - size / 2) / size
    y = (yy - size / 2) / size

    def gaussian(cx, cy, sx, sy, amp):
        return amp * np.exp(-0.5 * (((x - cx) / sx) ** 2 + ((y - cy) / sy) ** 2))

    sky = np.zeros((size, size), dtype=np.float32)
    sky += gaussian(-0.16, -0.08, 0.038, 0.038, 1.0)
    sky += gaussian(0.16, 0.08, 0.055, 0.035, 0.78)
    sky += gaussian(0.03, -0.18, 0.10, 0.055, 0.36)

    # A faint ring / shell structure to make fine detail useful.
    r = np.sqrt(((x + 0.02) / 0.22) ** 2 + ((y - 0.02) / 0.16) ** 2)
    sky += 0.30 * np.exp(-((r - 1.0) / 0.11) ** 2)

    # A smooth diffuse component.
    sky += gaussian(0.03, 0.02, 0.28, 0.20, 0.20)
    sky = np.clip(sky, 0.0, None)
    sky /= max(float(sky.max()), 1e-9)
    return sky


def make_antenna_positions() -> np.ndarray:
    """Toy 2-D antenna layout, normalized to a unit array radius."""
    return np.array([
        [-0.86, -0.02], [-0.68, 0.17], [-0.52, -0.21], [-0.34, 0.05],
        [-0.17, 0.23], [0.00, 0.00], [0.18, -0.23], [0.35, 0.07],
        [0.51, 0.22], [0.69, -0.16], [0.86, 0.03], [-0.12, -0.48],
        [0.09, 0.50], [0.42, -0.43],
    ], dtype=np.float32)


def baseline_vectors(antennas: np.ndarray) -> np.ndarray:
    vectors = []
    for i in range(len(antennas)):
        for j in range(i + 1, len(antennas)):
            vectors.append(antennas[j] - antennas[i])
    return np.array(vectors, dtype=np.float32)


def uv_tracks(antennas: np.ndarray, rotation_steps: int = 32) -> np.ndarray:
    """Toy Earth-rotation synthesis: rotate baseline vectors in the u-v plane."""
    base = baseline_vectors(antennas)
    points: List[Tuple[float, float]] = []
    angles = np.linspace(-0.88, 0.88, rotation_steps)
    for angle in angles:
        c, s = math.cos(angle), math.sin(angle)
        rot = np.array([[c, -s], [s, c]], dtype=np.float32)
        rotated = base @ rot.T
        for u, v in rotated:
            points.append((float(u), float(v)))
            points.append((-float(u), -float(v)))  # conjugate point for a real sky
    points_arr = np.array(points, dtype=np.float32)
    max_abs = max(float(np.abs(points_arr).max()), 1e-6)
    points_arr /= max_abs
    return points_arr


def add_disk(mask: np.ndarray, cx: int, cy: int, radius: int = 1):
    h, w = mask.shape
    y0, y1 = max(0, cy - radius), min(h, cy + radius + 1)
    x0, x1 = max(0, cx - radius), min(w, cx + radius + 1)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    disk = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius * radius
    mask[y0:y1, x0:x1][disk] = 1.0


def build_uv_mask(size: int, points: np.ndarray, fraction: float) -> np.ndarray:
    mask = np.zeros((size, size), dtype=np.float32)
    count = max(1, int(len(points) * clamp(fraction)))
    selected = points[:count]
    scale = size * 0.43
    c = size // 2
    for u, v in selected:
        x = int(round(c + u * scale))
        y = int(round(c - v * scale))
        if 0 <= x < size and 0 <= y < size:
            add_disk(mask, x, y, radius=1)
    # Always preserve a tiny central patch so broad flux is visible in the demo.
    add_disk(mask, c, c, radius=2)
    return mask


def reconstruct_from_uv(sky: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    vis = np.fft.fftshift(np.fft.fft2(sky))
    sampled = vis * mask
    dirty = np.fft.ifft2(np.fft.ifftshift(sampled)).real
    dirty -= dirty.min()
    dirty /= max(float(dirty.max()), 1e-9)
    amp = np.log1p(np.abs(sampled))
    amp /= max(float(amp.max()), 1e-9)
    return dirty.astype(np.float32), amp.astype(np.float32)


def scalar_image(arr: np.ndarray, tint: Tuple[int, int, int] = COLORS["cyan"]) -> Image.Image:
    a = np.clip(arr, 0.0, 1.0)
    rgb = np.zeros((*a.shape, 3), dtype=np.uint8)
    rgb[..., 0] = np.clip(a * tint[0], 0, 255).astype(np.uint8)
    rgb[..., 1] = np.clip(a * tint[1], 0, 255).astype(np.uint8)
    rgb[..., 2] = np.clip(a * tint[2], 0, 255).astype(np.uint8)
    # small floor so the image is visible on black
    rgb = np.clip(rgb + (a[..., None] * 26).astype(np.uint8), 0, 255)
    return Image.fromarray(rgb, mode="RGB")


# =============================================================================
# Main renderer
# =============================================================================

class RadioTelescopeExplainer:
    def __init__(self):
        self.rng = np.random.default_rng(42)
        self.stars = self._make_stars(180 if QUICK_MODE else 460)
        self.sky = make_sky_model(192)
        self.sky_img = scalar_image(self.sky, COLORS["gold"])
        self.antennas = make_antenna_positions()
        self.baselines = baseline_vectors(self.antennas)
        self.uv_points = uv_tracks(self.antennas, rotation_steps=12 if QUICK_MODE else 36)

        # Cache several reconstruction stages so frame rendering remains fast.
        self.recon_cache: Dict[int, Tuple[Image.Image, Image.Image, np.ndarray]] = {}
        cache_steps = 8 if QUICK_MODE else 24
        for idx in range(cache_steps + 1):
            frac = max(0.012, idx / cache_steps)
            mask = build_uv_mask(self.sky.shape[0], self.uv_points, frac)
            dirty, amp = reconstruct_from_uv(self.sky, mask)
            self.recon_cache[idx] = (
                scalar_image(dirty, COLORS["cyan"]),
                scalar_image(amp, COLORS["violet"]),
                mask,
            )
        self.cache_steps = cache_steps

    def _make_stars(self, count: int) -> List[Star]:
        stars = []
        for _ in range(count):
            stars.append(Star(
                x=float(self.rng.uniform(0, OUT_W)),
                y=float(self.rng.uniform(0, OUT_H)),
                radius=float(self.rng.uniform(0.4, 1.7) * OUT_W / 1280),
                alpha=int(self.rng.uniform(35, 170)),
                phase=float(self.rng.uniform(0, 2 * math.pi)),
            ))
        return stars

    def background(self, t: float) -> Image.Image:
        image = Image.new("RGBA", (OUT_W, OUT_H), COLORS["bg0"] + (255,))
        draw = ImageDraw.Draw(image)
        for y in range(OUT_H):
            p = y / max(OUT_H - 1, 1)
            r = int(lerp(COLORS["bg0"][0], COLORS["bg1"][0], p))
            g = int(lerp(COLORS["bg0"][1], COLORS["bg1"][1], p))
            b = int(lerp(COLORS["bg0"][2], COLORS["bg1"][2], p))
            draw.line((0, y, OUT_W, y), fill=(r, g, b, 255))
        for star in self.stars:
            alpha = int(star.alpha * (0.78 + 0.22 * math.sin(t * 0.8 + star.phase)))
            r = star.radius
            draw.ellipse((star.x - r, star.y - r, star.x + r, star.y + r), fill=COLORS["white"] + (alpha,))
        return image

    def draw_header(self, image: Image.Image, title: str, subtitle: Optional[str] = None, accent="cyan"):
        text(image, title, (int(OUT_W * 0.05), int(OUT_H * 0.08)), int(38 * OUT_W / 1280), COLORS["white"] + (245,), bold=True, anchor="la")
        if subtitle:
            text(image, subtitle, (int(OUT_W * 0.05), int(OUT_H * 0.135)), int(20 * OUT_W / 1280), COLORS[accent] + (230,), anchor="la", stroke=1)

    def draw_caption(self, image: Image.Image, caption: str, t: float):
        start = end = 0.0
        for a, b, value in NARRATION:
            if value == caption and a <= t < b:
                start, end = a, b
                break
        fade = min(clamp((t - start) / (0.4 if not QUICK_MODE else 0.08)), clamp((end - t) / (0.55 if not QUICK_MODE else 0.10)))
        alpha = int(232 * fade)
        if alpha <= 0:
            return
        box = (int(OUT_W * 0.075), int(OUT_H * 0.825), int(OUT_W * 0.925), int(OUT_H * 0.955))
        rounded_panel(image, box, alpha=min(150, int(alpha * 0.58)), outline_alpha=28)
        wrapped_text(
            image,
            caption,
            (box[0] + int(OUT_W * 0.025), box[1] + int(OUT_H * 0.022), box[2] - int(OUT_W * 0.025), box[3] - int(OUT_H * 0.015)),
            int(23 * OUT_W / 1280),
            COLORS["white"] + (alpha,),
            line_gap=0.18,
        )

    def draw_source_star(self, image: Image.Image, pos: Tuple[int, int], radius: float, pulse: float = 1.0):
        layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)
        x, y = pos
        for mult, alpha in [(4.0, 22), (2.3, 55), (1.4, 130)]:
            rr = radius * mult * pulse
            draw.ellipse((x - rr, y - rr, x + rr, y + rr), fill=COLORS["gold"] + (alpha,))
        image.alpha_composite(layer.filter(ImageFilter.GaussianBlur(max(1, int(radius * 0.75)))))
        draw = ImageDraw.Draw(image)
        rr = radius * pulse
        draw.ellipse((x - rr, y - rr, x + rr, y + rr), fill=COLORS["white"] + (255,))

    def draw_radio_wave(self, image: Image.Image, p0: Tuple[float, float], p1: Tuple[float, float], phase: float, amplitude: float, cycles: float, color, width=3, alpha=230):
        layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)
        x0, y0 = p0
        x1, y1 = p1
        dx, dy = x1 - x0, y1 - y0
        length = math.hypot(dx, dy)
        if length < 1:
            return
        nx, ny = -dy / length, dx / length
        points = []
        n = max(32, int(length / 5))
        for i in range(n + 1):
            q = i / n
            sx, sy = x0 + dx * q, y0 + dy * q
            offset = math.sin(2 * math.pi * cycles * q + phase) * amplitude
            points.append((sx + nx * offset, sy + ny * offset))
        draw.line(points, fill=color + (alpha,), width=max(1, int(width * OUT_W / 1280)), joint="curve")
        image.alpha_composite(layer)

    def draw_dish(self, image: Image.Image, center: Tuple[float, float], scale: float = 1.0, angle_deg: float = 0.0, label: Optional[str] = None):
        layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)
        cx, cy = center
        s = scale * OUT_W / 1280
        angle = math.radians(angle_deg)
        ca, sa = math.cos(angle), math.sin(angle)

        def tr(px: float, py: float) -> Tuple[float, float]:
            x = px * ca - py * sa
            y = px * sa + py * ca
            return cx + x * s, cy + y * s

        # Parabolic bowl y = 0.0065 x^2, opening upward in local coordinates.
        bowl = []
        for x in np.linspace(-110, 110, 70):
            y = 0.0060 * x * x
            bowl.append(tr(float(x), float(y)))
        draw.line(bowl, fill=COLORS["dish"] + (245,), width=max(2, int(5 * s)))
        # dish rim thickness / inner line
        bowl2 = []
        for x in np.linspace(-104, 104, 70):
            y = 0.0060 * x * x + 8
            bowl2.append(tr(float(x), float(y)))
        draw.line(bowl2, fill=COLORS["grid"] + (220,), width=max(1, int(2 * s)))

        # support and receiver near focus
        support_bottom = tr(0, 105)
        support_top = tr(0, 180)
        draw.line((*support_bottom, *support_top), fill=COLORS["dish"] + (230,), width=max(2, int(7 * s)))
        feed = tr(0, 72)
        fr = 9 * s
        draw.ellipse((feed[0] - fr, feed[1] - fr, feed[0] + fr, feed[1] + fr), fill=COLORS["gold"] + (255,))
        # feed arms
        for bx in (-70, 70):
            p = tr(bx, 30)
            draw.line((*p, *feed), fill=COLORS["grid"] + (220,), width=max(1, int(3 * s)))
        # base foot
        p0 = tr(-38, 180)
        p1 = tr(38, 180)
        draw.line((*p0, *p1), fill=COLORS["dish"] + (230,), width=max(2, int(7 * s)))

        image.alpha_composite(layer)
        if label:
            text(image, label, (int(cx), int(cy + 205 * s)), int(17 * OUT_W / 1280), COLORS["muted"] + (220,), anchor="ma", stroke=1)

    def draw_small_spectrum(self, image: Image.Image, reveal: float):
        box = (int(OUT_W * 0.10), int(OUT_H * 0.55), int(OUT_W * 0.90), int(OUT_H * 0.72))
        rounded_panel(image, box, alpha=95, outline_alpha=30)
        draw = ImageDraw.Draw(image)
        x0, y0, x1, y1 = box
        left = x0 + int(OUT_W * 0.045)
        right = x1 - int(OUT_W * 0.045)
        y = int((y0 + y1) / 2)
        draw.line((left, y, right, y), fill=COLORS["white"] + (130,), width=max(1, OUT_W // 640))
        bands = [
            ("GAMMA", 0.00, COLORS["violet"]),
            ("X-RAY", 0.14, COLORS["blue"]),
            ("VISIBLE", 0.31, COLORS["green"]),
            ("IR", 0.48, COLORS["gold"]),
            ("MICROWAVE", 0.66, COLORS["pink"]),
            ("RADIO", 0.86, COLORS["cyan"]),
        ]
        for label, p, color in bands:
            xx = int(lerp(left, right, p))
            draw.ellipse((xx - 5, y - 5, xx + 5, y + 5), fill=color + (220,))
            text(image, label, (xx, y + int(OUT_H * 0.043)), int(12 * OUT_W / 1280), color + (235,), bold=label == "RADIO", anchor="ma", stroke=1)
        radio_x = int(lerp(left, right, 0.86))
        rr = int(18 * OUT_W / 1280 * (0.7 + 0.3 * reveal))
        glow = Image.new("RGBA", image.size, (0, 0, 0, 0))
        gd = ImageDraw.Draw(glow)
        gd.ellipse((radio_x - rr, y - rr, radio_x + rr, y + rr), outline=COLORS["cyan"] + (int(220 * reveal),), width=max(2, OUT_W // 500))
        image.alpha_composite(glow.filter(ImageFilter.GaussianBlur(max(1, int(6 * OUT_W / 1280)))))
        image.alpha_composite(glow)

    def draw_resolution_diagram(self, image: Image.Image, local: float):
        draw = ImageDraw.Draw(image)
        dish_center = (int(OUT_W * 0.24), int(OUT_H * 0.55))
        self.draw_dish(image, dish_center, scale=0.78, label="single dish")

        # Two close sky sources.
        sx = int(OUT_W * 0.77)
        ymid = int(OUT_H * 0.36)
        sep = int(OUT_H * lerp(0.11, 0.055, local))
        self.draw_source_star(image, (sx, ymid - sep // 2), 7 * OUT_W / 1280, 1.0)
        self.draw_source_star(image, (sx, ymid + sep // 2), 7 * OUT_W / 1280, 1.0)
        text(image, "two nearby sources", (sx, ymid + sep // 2 + int(OUT_H * 0.07)), int(17 * OUT_W / 1280), COLORS["muted"] + (220,), anchor="ma", stroke=1)

        # Beam cone from dish toward sky.
        apex = (dish_center[0], dish_center[1] - int(120 * OUT_W / 1280))
        top1 = (sx - int(OUT_W * lerp(0.12, 0.055, local)), int(OUT_H * 0.24))
        top2 = (sx + int(OUT_W * lerp(0.12, 0.055, local)), int(OUT_H * 0.24))
        draw.line((*apex, *top1), fill=COLORS["cyan"] + (110,), width=max(1, OUT_W // 700))
        draw.line((*apex, *top2), fill=COLORS["cyan"] + (110,), width=max(1, OUT_W // 700))
        draw.arc((sx - int(OUT_W * 0.08), int(OUT_H * 0.21), sx + int(OUT_W * 0.08), int(OUT_H * 0.37)), 205, 335, fill=COLORS["gold"] + (210,), width=max(2, OUT_W // 500))
        text(image, "θ", (sx, int(OUT_H * 0.205)), int(28 * OUT_W / 1280), COLORS["gold"] + (245,), bold=True, anchor="ma")

        formula_box = (int(OUT_W * 0.47), int(OUT_H * 0.51), int(OUT_W * 0.92), int(OUT_H * 0.70))
        rounded_panel(image, formula_box, alpha=115)
        text(image, "θ  ≈  1.22 λ / D", (int((formula_box[0] + formula_box[2]) / 2), int(OUT_H * 0.585)), int(38 * OUT_W / 1280), COLORS["white"] + (245,), bold=True, anchor="mm")
        text(image, "longer λ → blurrier     larger D → sharper", (int((formula_box[0] + formula_box[2]) / 2), int(OUT_H * 0.66)), int(17 * OUT_W / 1280), COLORS["gold"] + (230,), anchor="mm", stroke=1)

    def draw_two_dish_delay(self, image: Image.Image, local: float):
        draw = ImageDraw.Draw(image)
        d1 = (int(OUT_W * 0.25), int(OUT_H * 0.66))
        d2 = (int(OUT_W * 0.73), int(OUT_H * 0.66))
        self.draw_dish(image, d1, scale=0.62, label="antenna A")
        self.draw_dish(image, d2, scale=0.62, label="antenna B")

        # Baseline
        yb = int(OUT_H * 0.77)
        draw_arrow(draw, (d1[0], yb), (d2[0], yb), COLORS["gold"] + (225,), width=max(2, OUT_W // 600), head=12 * OUT_W / 1280)
        draw_arrow(draw, (d2[0], yb), (d1[0], yb), COLORS["gold"] + (225,), width=max(2, OUT_W // 600), head=12 * OUT_W / 1280)
        text(image, "baseline  B", (int((d1[0] + d2[0]) / 2), yb - int(OUT_H * 0.025)), int(20 * OUT_W / 1280), COLORS["gold"] + (240,), bold=True, anchor="ms", stroke=1)

        # Distant radio source
        source = (int(OUT_W * 0.50), int(OUT_H * 0.17))
        self.draw_source_star(image, source, 8 * OUT_W / 1280, 1.0 + 0.06 * math.sin(local * math.pi * 4))
        text(image, "radio source", (source[0], source[1] - int(OUT_H * 0.045)), int(17 * OUT_W / 1280), COLORS["muted"] + (225,), anchor="ma", stroke=1)

        # Oblique wave fronts: reach left slightly earlier.
        alpha = math.radians(-13)
        for k in range(6):
            y = int(OUT_H * (0.28 + k * 0.065) + math.sin(local * math.pi * 2) * 4)
            x_shift = int((y - OUT_H * 0.28) * math.tan(alpha))
            draw.line((int(OUT_W * 0.09) + x_shift, y, int(OUT_W * 0.91) + x_shift, y - int(OUT_W * 0.18 * math.tan(alpha))), fill=COLORS["cyan"] + (100,), width=max(1, OUT_W // 900))

        # Path difference marker.
        marker_x = int(OUT_W * 0.70)
        marker_y0 = int(OUT_H * 0.39)
        marker_y1 = int(OUT_H * 0.47)
        draw_arrow(draw, (marker_x, marker_y0), (marker_x, marker_y1), COLORS["pink"] + (230,), width=max(2, OUT_W // 650), head=10 * OUT_W / 1280)
        text(image, "extra path  ΔL", (marker_x + int(OUT_W * 0.02), int((marker_y0 + marker_y1) / 2)), int(17 * OUT_W / 1280), COLORS["pink"] + (240,), anchor="lm", stroke=1)

        formula_box = (int(OUT_W * 0.09), int(OUT_H * 0.19), int(OUT_W * 0.35), int(OUT_H * 0.31))
        rounded_panel(image, formula_box, alpha=105)
        text(image, "ΔL ≈ B sin θ", (int((formula_box[0] + formula_box[2]) / 2), int((formula_box[1] + formula_box[3]) / 2)), int(25 * OUT_W / 1280), COLORS["white"] + (240,), bold=True, anchor="mm")

    def draw_signal_plot(self, image: Image.Image, box: Tuple[int, int, int, int], phase: float, color, label: str, amplitude=1.0):
        x0, y0, x1, y1 = box
        rounded_panel(image, box, alpha=85, outline_alpha=20)
        draw = ImageDraw.Draw(image)
        mid = (y0 + y1) / 2
        draw.line((x0 + 15, mid, x1 - 15, mid), fill=COLORS["grid"] + (120,), width=1)
        pts = []
        n = 220
        for i in range(n + 1):
            q = i / n
            x = lerp(x0 + 15, x1 - 15, q)
            y = mid - math.sin(2 * math.pi * 3.0 * q + phase) * amplitude * (y1 - y0) * 0.28
            pts.append((x, y))
        draw.line(pts, fill=color + (235,), width=max(2, OUT_W // 700))
        text(image, label, (x0 + 18, y0 + int((y1 - y0) * 0.17)), int(16 * OUT_W / 1280), color + (235,), bold=True, anchor="la", stroke=1)

    def draw_phase_correlation(self, image: Image.Image, local: float):
        phase = lerp(2.1, 0.0, smoothstep(local * 1.25))
        box_a = (int(OUT_W * 0.07), int(OUT_H * 0.25), int(OUT_W * 0.43), int(OUT_H * 0.43))
        box_b = (int(OUT_W * 0.07), int(OUT_H * 0.49), int(OUT_W * 0.43), int(OUT_H * 0.67))
        self.draw_signal_plot(image, box_a, 0.0, COLORS["cyan"], "signal A")
        self.draw_signal_plot(image, box_b, phase, COLORS["pink"], "signal B")

        draw = ImageDraw.Draw(image)
        if local < 0.52:
            text(image, "same wave, shifted in phase", (int(OUT_W * 0.25), int(OUT_H * 0.72)), int(18 * OUT_W / 1280), COLORS["muted"] + (225,), anchor="ma", stroke=1)
        else:
            draw_arrow(draw, (int(OUT_W * 0.46), int(OUT_H * 0.46)), (int(OUT_W * 0.56), int(OUT_H * 0.46)), COLORS["gold"] + (230,), width=max(2, OUT_W // 600), head=11 * OUT_W / 1280)
            text(image, "CORRELATOR", (int(OUT_W * 0.63), int(OUT_H * 0.36)), int(25 * OUT_W / 1280), COLORS["gold"] + (245,), bold=True, anchor="ma")
            result_box = (int(OUT_W * 0.55), int(OUT_H * 0.42), int(OUT_W * 0.92), int(OUT_H * 0.67))
            rounded_panel(image, result_box, alpha=115)
            text(image, "visibility  V(u,v)", (int((result_box[0] + result_box[2]) / 2), int(OUT_H * 0.49)), int(30 * OUT_W / 1280), COLORS["white"] + (245,), bold=True, anchor="mm")
            text(image, "amplitude  +  phase", (int((result_box[0] + result_box[2]) / 2), int(OUT_H * 0.56)), int(20 * OUT_W / 1280), COLORS["cyan"] + (235,), anchor="mm", stroke=1)
            text(image, "one complex Fourier sample", (int((result_box[0] + result_box[2]) / 2), int(OUT_H * 0.625)), int(18 * OUT_W / 1280), COLORS["muted"] + (225,), anchor="mm", stroke=1)

        # phase formula
        text(image, "Δφ = 2π ΔL / λ", (int(OUT_W * 0.70), int(OUT_H * 0.76)), int(24 * OUT_W / 1280), COLORS["pink"] + (240,), bold=True, anchor="ma", stroke=1)

    def paste_scalar(self, image: Image.Image, scalar_img: Image.Image, box: Tuple[int, int, int, int], label: Optional[str] = None, border=True):
        x0, y0, x1, y1 = box
        img = scalar_img.resize((x1 - x0, y1 - y0), Image.Resampling.LANCZOS).convert("RGBA")
        image.alpha_composite(img, (x0, y0))
        if border:
            draw = ImageDraw.Draw(image)
            draw.rounded_rectangle(box, radius=max(6, int((x1 - x0) * 0.025)), outline=COLORS["cyan"] + (70,), width=max(1, OUT_W // 1000))
        if label:
            text(image, label, ((x0 + x1) // 2, y1 + int(OUT_H * 0.03)), int(16 * OUT_W / 1280), COLORS["muted"] + (225,), anchor="ma", stroke=1)

    def make_stripe_image(self, freq: float, angle: float, size: int = 220) -> Image.Image:
        yy, xx = np.mgrid[0:size, 0:size]
        x = (xx - size / 2) / size
        y = (yy - size / 2) / size
        coord = x * math.cos(angle) + y * math.sin(angle)
        arr = 0.5 + 0.5 * np.cos(2 * math.pi * freq * coord)
        arr *= np.exp(-((x / 0.63) ** 8 + (y / 0.63) ** 8))
        return scalar_image(arr.astype(np.float32), COLORS["violet"])

    def draw_fourier_intuition(self, image: Image.Image, local: float):
        left = (int(OUT_W * 0.06), int(OUT_H * 0.22), int(OUT_W * 0.36), int(OUT_H * 0.72))
        self.paste_scalar(image, self.sky_img, left, label="sky brightness  I(x,y)")

        draw = ImageDraw.Draw(image)
        draw_arrow(draw, (int(OUT_W * 0.39), int(OUT_H * 0.47)), (int(OUT_W * 0.47), int(OUT_H * 0.47)), COLORS["gold"] + (230,), width=max(2, OUT_W // 650), head=12 * OUT_W / 1280)
        text(image, "FOURIER", (int(OUT_W * 0.43), int(OUT_H * 0.42)), int(17 * OUT_W / 1280), COLORS["gold"] + (240,), bold=True, anchor="ma", stroke=1)

        boxes = [
            (int(OUT_W * 0.50), int(OUT_H * 0.22), int(OUT_W * 0.63), int(OUT_H * 0.46)),
            (int(OUT_W * 0.67), int(OUT_H * 0.22), int(OUT_W * 0.80), int(OUT_H * 0.46)),
            (int(OUT_W * 0.84), int(OUT_H * 0.22), int(OUT_W * 0.97), int(OUT_H * 0.46)),
        ]
        freqs = [1.5, 4.0, 8.0]
        labels = ["broad", "medium", "fine"]
        for i, (box, freq, label) in enumerate(zip(boxes, freqs, labels)):
            reveal = clamp(local * 2.0 - i * 0.28)
            if reveal <= 0:
                continue
            stripes = self.make_stripe_image(freq, angle=0.25 + i * 0.65, size=220)
            self.paste_scalar(image, stripes, box, border=True)
            text(image, label, ((box[0] + box[2]) // 2, box[3] + int(OUT_H * 0.03)), int(15 * OUT_W / 1280), COLORS["muted"] + (225,), anchor="ma", stroke=1)

        formula = (int(OUT_W * 0.50), int(OUT_H * 0.54), int(OUT_W * 0.97), int(OUT_H * 0.72))
        rounded_panel(image, formula, alpha=105)
        text(image, "I(x,y)  ⇄  V(u,v)", (int((formula[0] + formula[2]) / 2), int(OUT_H * 0.60)), int(34 * OUT_W / 1280), COLORS["white"] + (245,), bold=True, anchor="mm")
        text(image, "image space        spatial-frequency space", (int((formula[0] + formula[2]) / 2), int(OUT_H * 0.67)), int(17 * OUT_W / 1280), COLORS["cyan"] + (230,), anchor="mm", stroke=1)

    def draw_uv_axes(self, image: Image.Image, box: Tuple[int, int, int, int], alpha=180):
        x0, y0, x1, y1 = box
        draw = ImageDraw.Draw(image)
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        draw.line((x0, cy, x1, cy), fill=COLORS["grid"] + (alpha,), width=1)
        draw.line((cx, y0, cx, y1), fill=COLORS["grid"] + (alpha,), width=1)
        text(image, "u", (x1 - 7, cy - 9), int(14 * OUT_W / 1280), COLORS["muted"] + (220,), anchor="rs", stroke=1)
        text(image, "v", (cx + 8, y0 + 8), int(14 * OUT_W / 1280), COLORS["muted"] + (220,), anchor="la", stroke=1)

    def draw_array_layout(self, image: Image.Image, box: Tuple[int, int, int, int], reveal: float):
        x0, y0, x1, y1 = box
        rounded_panel(image, box, alpha=75, outline_alpha=20)
        draw = ImageDraw.Draw(image)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        scale = min(x1 - x0, y1 - y0) * 0.40
        # A few baseline lines appear first.
        pair_indices = [(0, 5), (2, 9), (4, 12), (6, 13), (1, 10)]
        for idx, (i, j) in enumerate(pair_indices):
            r = clamp(reveal * 1.6 - idx * 0.14)
            if r <= 0:
                continue
            p0 = (cx + self.antennas[i, 0] * scale, cy - self.antennas[i, 1] * scale)
            p1 = (cx + self.antennas[j, 0] * scale, cy - self.antennas[j, 1] * scale)
            draw.line((*p0, *p1), fill=COLORS["gold"] + (int(90 * r),), width=max(1, OUT_W // 900))
        for ax, ay in self.antennas:
            x = cx + ax * scale
            y = cy - ay * scale
            rr = 5 * OUT_W / 1280
            draw.ellipse((x - rr, y - rr, x + rr, y + rr), fill=COLORS["white"] + (235,), outline=COLORS["cyan"] + (230,), width=max(1, OUT_W // 900))
        text(image, "antenna positions", (int(cx), y1 + int(OUT_H * 0.026)), int(15 * OUT_W / 1280), COLORS["muted"] + (225,), anchor="ma", stroke=1)

    def draw_uv_points(self, image: Image.Image, box: Tuple[int, int, int, int], fraction: float, point_color=None):
        if point_color is None:
            point_color = COLORS["violet"]
        rounded_panel(image, box, alpha=75, outline_alpha=20)
        self.draw_uv_axes(image, box)
        draw = ImageDraw.Draw(image)
        x0, y0, x1, y1 = box
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        scale = min(x1 - x0, y1 - y0) * 0.44
        count = max(1, int(len(self.uv_points) * clamp(fraction)))
        step = max(1, count // (900 if not QUICK_MODE else 250))
        for u, v in self.uv_points[:count:step]:
            x = cx + u * scale
            y = cy - v * scale
            rr = max(1.1, 2.3 * OUT_W / 1280)
            draw.ellipse((x - rr, y - rr, x + rr, y + rr), fill=point_color + (205,))
        text(image, "u-v plane", (int(cx), y1 + int(OUT_H * 0.026)), int(15 * OUT_W / 1280), COLORS["muted"] + (225,), anchor="ma", stroke=1)

    def draw_uv_sampling(self, image: Image.Image, local: float):
        left = (int(OUT_W * 0.05), int(OUT_H * 0.22), int(OUT_W * 0.44), int(OUT_H * 0.70))
        right = (int(OUT_W * 0.56), int(OUT_H * 0.22), int(OUT_W * 0.95), int(OUT_H * 0.70))
        self.draw_array_layout(image, left, local)
        self.draw_uv_points(image, right, fraction=max(0.015, local))
        draw = ImageDraw.Draw(image)
        draw_arrow(draw, (int(OUT_W * 0.46), int(OUT_H * 0.46)), (int(OUT_W * 0.54), int(OUT_H * 0.46)), COLORS["gold"] + (230,), width=max(2, OUT_W // 650), head=11 * OUT_W / 1280)
        text(image, "every pair → one Fourier sample", (int(OUT_W * 0.50), int(OUT_H * 0.76)), int(21 * OUT_W / 1280), COLORS["gold"] + (240,), bold=True, anchor="ma", stroke=1)
        text(image, "short baseline = broad structure      long baseline = fine detail", (int(OUT_W * 0.50), int(OUT_H * 0.80)), int(17 * OUT_W / 1280), COLORS["cyan"] + (225,), anchor="ma", stroke=1)

    def draw_inverse_fourier(self, image: Image.Image, local: float):
        # Step through cached sampling stages.
        stage = int(round(clamp(local) * self.cache_steps))
        dirty_img, amp_img, _ = self.recon_cache[stage]

        box_uv = (int(OUT_W * 0.05), int(OUT_H * 0.24), int(OUT_W * 0.30), int(OUT_H * 0.70))
        box_dirty = (int(OUT_W * 0.375), int(OUT_H * 0.24), int(OUT_W * 0.625), int(OUT_H * 0.70))
        box_true = (int(OUT_W * 0.70), int(OUT_H * 0.24), int(OUT_W * 0.95), int(OUT_H * 0.70))
        self.paste_scalar(image, amp_img, box_uv, label="measured Fourier samples")
        self.paste_scalar(image, dirty_img, box_dirty, label="inverse FFT: reconstructed image")
        self.paste_scalar(image, self.sky_img, box_true, label="original toy sky")

        draw = ImageDraw.Draw(image)
        draw_arrow(draw, (int(OUT_W * 0.315), int(OUT_H * 0.47)), (int(OUT_W * 0.36), int(OUT_H * 0.47)), COLORS["gold"] + (230,), width=max(2, OUT_W // 650), head=10 * OUT_W / 1280)
        draw_arrow(draw, (int(OUT_W * 0.64), int(OUT_H * 0.47)), (int(OUT_W * 0.685), int(OUT_H * 0.47)), COLORS["gold"] + (120,), width=max(2, OUT_W // 650), head=10 * OUT_W / 1280)

        pct = int(100 * max(0.012, stage / max(self.cache_steps, 1)))
        text(image, f"u-v coverage: {pct}% of this toy sample list", (int(OUT_W * 0.50), int(OUT_H * 0.77)), int(19 * OUT_W / 1280), COLORS["cyan"] + (235,), bold=True, anchor="ma", stroke=1)
        text(image, "Real pipelines also calibrate, weight, grid, and deconvolve the data.", (int(OUT_W * 0.50), int(OUT_H * 0.81)), int(15 * OUT_W / 1280), COLORS["muted"] + (215,), anchor="ma", stroke=1)

    def draw_earth_rotation(self, image: Image.Image, local: float):
        # Left: Earth + source + two antennas. Right: uv tracks accumulate.
        left = (int(OUT_W * 0.05), int(OUT_H * 0.20), int(OUT_W * 0.45), int(OUT_H * 0.72))
        right = (int(OUT_W * 0.55), int(OUT_H * 0.20), int(OUT_W * 0.95), int(OUT_H * 0.72))
        rounded_panel(image, left, alpha=75, outline_alpha=20)
        draw = ImageDraw.Draw(image)
        cx, cy = int(OUT_W * 0.25), int(OUT_H * 0.50)
        r = int(OUT_H * 0.17)
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(25, 65, 105, 255), outline=COLORS["cyan"] + (120,), width=max(1, OUT_W // 700))
        # latitude-like arcs
        draw.arc((cx - r, cy - int(r * 0.45), cx + r, cy + int(r * 0.45)), 0, 360, fill=COLORS["grid"] + (110,), width=1)
        draw.line((cx - r, cy, cx + r, cy), fill=COLORS["grid"] + (110,), width=1)
        # rotating antenna pair on top of Earth
        angle = lerp(-0.8, 0.8, local)
        p1 = (cx + math.cos(angle) * r * 0.78, cy - math.sin(angle) * r * 0.32 - r * 0.72)
        p2 = (cx + math.cos(angle + 0.75) * r * 0.72, cy - math.sin(angle + 0.75) * r * 0.28 - r * 0.68)
        for p in (p1, p2):
            rr = 5 * OUT_W / 1280
            draw.ellipse((p[0] - rr, p[1] - rr, p[0] + rr, p[1] + rr), fill=COLORS["white"] + (240,), outline=COLORS["gold"] + (230,), width=max(1, OUT_W // 800))
        draw.line((*p1, *p2), fill=COLORS["gold"] + (210,), width=max(2, OUT_W // 700))
        draw_arrow(draw, (cx - r * 0.9, cy + r * 1.16), (cx + r * 0.9, cy + r * 1.16), COLORS["cyan"] + (200,), width=max(2, OUT_W // 700), head=10 * OUT_W / 1280)
        text(image, "Earth rotates", (cx, cy + int(r * 1.38)), int(17 * OUT_W / 1280), COLORS["cyan"] + (235,), bold=True, anchor="ma", stroke=1)

        self.draw_uv_points(image, right, fraction=clamp(0.08 + 0.92 * local), point_color=COLORS["green"])
        text(image, "projected baseline changes → more u-v coverage", (int(OUT_W * 0.75), int(OUT_H * 0.77)), int(18 * OUT_W / 1280), COLORS["green"] + (235,), bold=True, anchor="ma", stroke=1)

    def draw_final_pipeline(self, image: Image.Image, local: float):
        labels = ["RADIO WAVES", "DISHES", "CORRELATE", "u-v SAMPLES", "INVERSE FFT", "IMAGE"]
        colors = [COLORS["cyan"], COLORS["white"], COLORS["pink"], COLORS["violet"], COLORS["gold"], COLORS["green"]]
        xs = np.linspace(0.10, 0.90, len(labels))
        y = int(OUT_H * 0.48)
        draw = ImageDraw.Draw(image)
        for i, (label, x, color) in enumerate(zip(labels, xs, colors)):
            reveal = clamp(local * 2.2 - i * 0.13)
            if reveal <= 0:
                continue
            xx = int(OUT_W * x)
            rr = int(28 * OUT_W / 1280)
            glow = Image.new("RGBA", image.size, (0, 0, 0, 0))
            gd = ImageDraw.Draw(glow)
            gd.ellipse((xx - rr, y - rr, xx + rr, y + rr), fill=color + (int(45 * reveal),))
            image.alpha_composite(glow.filter(ImageFilter.GaussianBlur(max(2, int(10 * OUT_W / 1280)))))
            draw.ellipse((xx - rr * 0.55, y - rr * 0.55, xx + rr * 0.55, y + rr * 0.55), fill=color + (230,))
            text(image, label, (xx, y + int(OUT_H * 0.075)), int(14 * OUT_W / 1280), COLORS["white"] + (235,), bold=True, anchor="ma", stroke=1)
            if i < len(labels) - 1:
                nx = int(OUT_W * xs[i + 1])
                draw_arrow(draw, (xx + rr * 0.75, y), (nx - rr * 0.75, y), COLORS["muted"] + (160,), width=max(1, OUT_W // 900), head=8 * OUT_W / 1280)

        text(image, "A RADIO INTERFEROMETER IS A FOURIER-MEASURING CAMERA", (OUT_W // 2, int(OUT_H * 0.23)), int(34 * OUT_W / 1280), COLORS["white"] + (245,), bold=True, anchor="ma")
        text(image, "maximum baseline sets the finest detail:  θ ~ λ / Bmax", (OUT_W // 2, int(OUT_H * 0.68)), int(20 * OUT_W / 1280), COLORS["gold"] + (235,), anchor="ma", stroke=1)

    def frame(self, t: float) -> np.ndarray:
        shot, start, end = get_shot(t)
        local = smoothstep((t - start) / max(end - start, 1e-9))
        image = self.background(t)

        if shot == "hook":
            self.draw_header(image, "HOW DOES A RADIO TELESCOPE MAKE AN IMAGE?", "The answer is hidden inside waves + Fourier transforms", "gold")
            self.draw_source_star(image, (int(OUT_W * 0.74), int(OUT_H * 0.28)), 10 * OUT_W / 1280, 1.0 + 0.06 * math.sin(t * 3.0))
            self.draw_dish(image, (int(OUT_W * 0.29), int(OUT_H * 0.63)), scale=1.0, angle_deg=-6)
            self.draw_radio_wave(image, (int(OUT_W * 0.69), int(OUT_H * 0.32)), (int(OUT_W * 0.34), int(OUT_H * 0.52)), phase=-t * 3.0, amplitude=7 * OUT_W / 1280, cycles=8, color=COLORS["cyan"], width=3)
            text(image, "not a normal photo", (int(OUT_W * 0.50), int(OUT_H * 0.70)), int(22 * OUT_W / 1280), COLORS["muted"] + (220,), anchor="ma", stroke=1)

        elif shot == "radio_is_light":
            self.draw_header(image, "1  RADIO IS LIGHT", "Same physics. Much longer wavelength.")
            # visible wave vs radio wave
            self.draw_radio_wave(image, (int(OUT_W * 0.12), int(OUT_H * 0.32)), (int(OUT_W * 0.88), int(OUT_H * 0.32)), phase=t * 2.0, amplitude=8 * OUT_W / 1280, cycles=17, color=COLORS["green"], width=2)
            self.draw_radio_wave(image, (int(OUT_W * 0.12), int(OUT_H * 0.44)), (int(OUT_W * 0.88), int(OUT_H * 0.44)), phase=t * 1.1, amplitude=18 * OUT_W / 1280, cycles=4.0, color=COLORS["cyan"], width=3)
            text(image, "visible: short λ", (int(OUT_W * 0.12), int(OUT_H * 0.25)), int(18 * OUT_W / 1280), COLORS["green"] + (230,), anchor="la", stroke=1)
            text(image, "radio: long λ", (int(OUT_W * 0.12), int(OUT_H * 0.51)), int(18 * OUT_W / 1280), COLORS["cyan"] + (235,), bold=True, anchor="la", stroke=1)
            self.draw_small_spectrum(image, local)

        elif shot == "dish":
            self.draw_header(image, "2  THE DISH COLLECTS WEAK WAVES", "Curved metal redirects the incoming wave toward a receiver")
            dish_center = (int(OUT_W * 0.50), int(OUT_H * 0.62))
            self.draw_dish(image, dish_center, scale=1.25)
            # plane-ish incoming waves
            for i in range(6):
                yy = int(OUT_H * (0.23 + i * 0.055))
                self.draw_radio_wave(image, (int(OUT_W * 0.18), yy), (int(OUT_W * 0.82), yy), phase=-t * 3.2 + i * 0.8, amplitude=5 * OUT_W / 1280, cycles=6.0, color=COLORS["cyan"], width=2, alpha=140)
            # focus arrows toward receiver
            draw = ImageDraw.Draw(image)
            focus = (dish_center[0], int(dish_center[1] + 72 * 1.25 * OUT_W / 1280))
            for px in [dish_center[0] - int(110 * 1.25 * OUT_W / 1280), dish_center[0] - int(55 * 1.25 * OUT_W / 1280), dish_center[0] + int(55 * 1.25 * OUT_W / 1280), dish_center[0] + int(110 * 1.25 * OUT_W / 1280)]:
                py = int(dish_center[1] + 0.0060 * ((px - dish_center[0]) / (1.25 * OUT_W / 1280)) ** 2 * 1.25 * OUT_W / 1280)
                draw_arrow(draw, (px, py), focus, COLORS["gold"] + (150,), width=max(1, OUT_W // 900), head=7 * OUT_W / 1280)
            text(image, "receiver", (focus[0] + int(OUT_W * 0.035), focus[1]), int(17 * OUT_W / 1280), COLORS["gold"] + (240,), anchor="lm", stroke=1)
            text(image, "wave → voltage → digital numbers", (OUT_W // 2, int(OUT_H * 0.76)), int(21 * OUT_W / 1280), COLORS["white"] + (235,), bold=True, anchor="ma", stroke=1)

        elif shot == "single_dish_resolution":
            self.draw_header(image, "3  ONE DISH HAS LIMITED SHARPNESS", "The longer the wavelength, the harder it is to see tiny angles", "gold")
            self.draw_resolution_diagram(image, local)

        elif shot == "two_dishes":
            self.draw_header(image, "4  USE TWO DISHES FAR APART", "Now the same wave arrives at two places")
            self.draw_two_dish_delay(image, local)

        elif shot == "phase_and_correlation":
            self.draw_header(image, "5  MEASURE THE PHASE DIFFERENCE", "A correlator compares the two time-varying signals", "pink")
            self.draw_phase_correlation(image, local)

        elif shot == "fourier_intuition":
            self.draw_header(image, "6  FOURIER TRANSFORM = A PATTERN RECIPE", "A picture can be rebuilt from spatial patterns at many scales", "violet")
            self.draw_fourier_intuition(image, local)

        elif shot == "uv_sampling":
            self.draw_header(image, "7  EACH BASELINE MEASURES ONE FOURIER PIECE", "Antenna geometry becomes sampling geometry in the u-v plane", "violet")
            self.draw_uv_sampling(image, local)

        elif shot == "inverse_fourier":
            self.draw_header(image, "8  INVERSE FOURIER TRANSFORM → IMAGE", "More u-v coverage means a more faithful reconstruction", "green")
            self.draw_inverse_fourier(image, local)

        elif shot == "earth_rotation":
            self.draw_header(image, "9  EARTH ROTATION FILLS THE GAPS", "The same antennas sample new projected baselines as the sky moves", "green")
            self.draw_earth_rotation(image, local)

        elif shot == "finale":
            self.draw_final_pipeline(image, local)

        caption = narration_at(t)
        if caption:
            self.draw_caption(image, caption, t)

        # tiny science disclaimer / signature
        text(image, "educational visualization • real interferometric imaging includes calibration + deconvolution", (int(OUT_W * 0.018), int(OUT_H * 0.985)), int(10 * OUT_W / 1280), COLORS["muted"] + (105,), anchor="ls", stroke=1)

        arr = np.asarray(image.convert("RGB"), dtype=np.float32)
        arr = np.clip(arr * VIGNETTE[..., None], 0, 255).astype(np.uint8)
        graded = Image.fromarray(arr, mode="RGB")
        graded = ImageEnhance.Contrast(graded).enhance(1.06)
        graded = ImageEnhance.Color(graded).enhance(0.97)
        return np.asarray(graded)


# =============================================================================
# Procedural soundtrack
# =============================================================================

def create_soundtrack(path: Path, duration_s: float):
    sr = AUDIO_RATE
    n = int(duration_s * sr)
    tt = np.arange(n, dtype=np.float32) / sr
    rng = np.random.default_rng(31415)

    # Quiet documentary pad.
    audio = (
        0.11 * np.sin(2 * math.pi * 46.0 * tt)
        + 0.060 * np.sin(2 * math.pi * 69.0 * tt + 0.4)
        + 0.025 * np.sin(2 * math.pi * 138.0 * tt + 1.1)
        + 0.008 * rng.normal(0.0, 1.0, n).astype(np.float32)
    )

    # Chapter transition thumps + soft whooshes.
    for name, start, _ in SHOT_PLAN[1:]:
        i0 = int(start * sr)
        length = min(int((0.45 if QUICK_MODE else 0.85) * sr), n - i0)
        if length <= 0:
            continue
        x = np.arange(length, dtype=np.float32) / sr
        hit = 0.075 * np.sin(2 * math.pi * 88.0 * x) * np.exp(-x * 4.5)
        whoosh = 0.015 * rng.normal(0.0, 1.0, length).astype(np.float32) * np.exp(-x * 2.8)
        if name in ("fourier_intuition", "inverse_fourier"):
            hit += 0.035 * np.sin(2 * math.pi * 440.0 * x) * np.exp(-x * 1.7)
        audio[i0:i0 + length] += hit + whoosh

    # Small "data" chimes during Fourier chapters.
    for start in [SHOT_PLAN_FULL[6][1], SHOT_PLAN_FULL[7][1], SHOT_PLAN_FULL[8][1]]:
        st = start * (DURATION / 110.0 if QUICK_MODE else 1.0)
        i0 = int(st * sr)
        length = min(int(1.4 * sr), n - i0)
        if length <= 0:
            continue
        x = np.arange(length, dtype=np.float32) / sr
        chime = (
            0.030 * np.sin(2 * math.pi * 523.25 * x)
            + 0.022 * np.sin(2 * math.pi * 659.25 * x)
            + 0.016 * np.sin(2 * math.pi * 783.99 * x)
        ) * np.exp(-x * 1.5)
        audio[i0:i0 + length] += chime

    audio /= max(float(np.max(np.abs(audio))), 1e-9)
    audio *= 0.62
    left = audio * (0.98 + 0.02 * np.sin(2 * math.pi * 0.026 * tt))
    right = audio * (0.98 + 0.02 * np.cos(2 * math.pi * 0.026 * tt + 0.7))
    stereo = np.stack([left, right], axis=1)
    pcm = np.int16(np.clip(stereo, -1.0, 1.0) * 32767)

    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(sr)
        wav.writeframes(pcm.tobytes())


# =============================================================================
# Output helpers
# =============================================================================

def render_video(scene: RadioTelescopeExplainer, output_path: Path):
    with iio.get_writer(
        str(output_path),
        fps=FPS,
        codec="libx264",
        quality=8,
        pixelformat="yuv420p",
        macro_block_size=1,
    ) as writer:
        for frame_index in range(FRAME_COUNT):
            if frame_index % max(1, FPS * 5) == 0:
                print(f"Rendering {frame_index}/{FRAME_COUNT} frames...")
            writer.append_data(scene.frame(frame_index / FPS))


def mux_audio(video_path: Path, audio_path: Path, final_path: Path) -> bool:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        shutil.copy2(video_path, final_path)
        return False
    command = [
        ffmpeg, "-y",
        "-i", str(video_path),
        "-i", str(audio_path),
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        str(final_path),
    ]
    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except Exception as exc:
        print("Audio mux failed:", exc)
        shutil.copy2(video_path, final_path)
        return False


def save_contact_sheet(scene: RadioTelescopeExplainer, path: Path):
    sample_times = []
    for _, start, end in SHOT_PLAN:
        sample_times.append(start + (end - start) * 0.55)
    thumbs = []
    tw, th = 320, 180
    for sample in sample_times:
        img = Image.fromarray(scene.frame(sample)).resize((tw, th), Image.Resampling.LANCZOS)
        thumbs.append(img)
    cols = 3
    rows = int(math.ceil(len(thumbs) / cols))
    sheet = Image.new("RGB", (tw * cols, th * rows), COLORS["bg0"])
    for i, thumb in enumerate(thumbs):
        sheet.paste(thumb, ((i % cols) * tw, (i // cols) * th))
    sheet.save(path, quality=92)


def write_narration(path: Path):
    lines = ["RADIO TELESCOPE + FOURIER TRANSFORM — NARRATION\n"]
    for index, (start, end, text_value) in enumerate(NARRATION, 1):
        lines.append(f"{index:02d}. {start:6.1f}s–{end:6.1f}s\n{text_value}\n")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_summary(path: Path, audio_muxed: bool):
    payload: Dict[str, Any] = {
        "title": "How Radio Telescopes Make Images — Fourier Transform Explained Simply",
        "format": "16:9 full explainer",
        "resolution": [OUT_W, OUT_H],
        "fps": FPS,
        "duration_s": DURATION,
        "chapters": [name for name, _, _ in SHOT_PLAN],
        "key_math": [
            "Single-dish angular resolution (approx.): theta ≈ 1.22 lambda / D",
            "Geometric path difference (simple 2-D illustration): Delta L ≈ B sin(theta)",
            "Phase difference: Delta phi = 2 pi Delta L / lambda",
            "Interferometric visibility: V(u,v) is a complex Fourier-domain measurement",
            "Image formation concept: I(x,y) <-> V(u,v), with inverse Fourier transform returning image space",
            "Interferometer finest angular scale (rule of thumb): theta ~ lambda / B_max",
        ],
        "important_caveats": [
            "The u-v sampling and Earth-rotation geometry are pedagogical toy models.",
            "Real visibility sampling depends on projected baselines in wavelengths, source declination, hour angle, array latitude, and frequency.",
            "Real imaging normally includes calibration, flagging, weighting, gridding, inverse transforms, deconvolution such as CLEAN, and primary-beam corrections.",
            "A single dish measures total power / convolved sky brightness; an interferometer is insensitive to some spatial scales unless short spacings / total-power information are included.",
        ],
        "official_background_sources": [
            "https://www.almaobservatory.org/en/about-alma/how-alma-works/technologies/interferometry/",
            "https://www.almaobservatory.org/en/about-alma/how-alma-works/",
            "https://public.nrao.edu/interferometry-explained/",
        ],
        "audio_muxed": audio_muxed,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

def main():
    scene = RadioTelescopeExplainer()

    silent_video = OUTPUT_ROOT / "radio_telescope_fourier_explainer_silent.mp4"
    soundtrack = OUTPUT_ROOT / "radio_telescope_fourier_soundtrack.wav"
    final_video = OUTPUT_ROOT / "radio_telescope_fourier_explainer.mp4"
    subtitles = OUTPUT_ROOT / "radio_telescope_fourier_explainer.srt"
    narration_txt = OUTPUT_ROOT / "radio_telescope_fourier_narration.txt"
    summary_json = OUTPUT_ROOT / "radio_telescope_fourier_summary.json"
    contact_sheet = PREVIEW_DIR / "contact_sheet.jpg"

    print(f"Mode: {'QUICK' if QUICK_MODE else ('1080p' if FULLHD_MODE else '720p')}")
    print(f"Video: {OUT_W}x{OUT_H}, {FPS} fps, {DURATION:.1f} s")

    write_srt(NARRATION, subtitles)
    write_narration(narration_txt)
    save_contact_sheet(scene, contact_sheet)
    render_video(scene, silent_video)
    create_soundtrack(soundtrack, DURATION)
    audio_muxed = mux_audio(silent_video, soundtrack, final_video)
    write_summary(summary_json, audio_muxed)

    print("\nRender complete")
    print(f"Final video:   {final_video.resolve()}")
    print(f"Subtitles:     {subtitles.resolve()}")
    print(f"Narration:     {narration_txt.resolve()}")
    print(f"Contact sheet: {contact_sheet.resolve()}")
    print(f"Summary:       {summary_json.resolve()}")


if __name__ == "__main__":
    main()

