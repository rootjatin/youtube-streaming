from __future__ import annotations

"""
HOW RADIO TELESCOPES RENDER IMAGES — Part 2
============================================

A beginner-friendly cinematic Python renderer explaining how an interferometric
radio telescope array turns weak voltages into a scientifically useful image.

This episode intentionally does NOT teach Fourier-transform / FFT mathematics.
It only shows where the transform sits in the real imaging pipeline. A separate
short can explain why Fourier transforms are the right mathematics.

Scientific story
----------------
1) Antennas do not directly record a photograph; they record changing voltages.
2) A correlator compares signals from pairs of antennas.
3) Calibration removes instrumental/atmospheric amplitude and phase errors.
4) Each antenna pair contributes a sample of spatial information (u-v data).
5) Earth rotation and many baselines fill more of the u-v plane.
6) An imaging transform converts the sampled measurements into a DIRTY IMAGE.
7) Sparse sampling creates a DIRTY BEAM / sidelobe pattern.
8) Deconvolution (illustrated with a simplified CLEAN-like process) removes much
   of that point-spread response and restores a cleaner radio image.
9) Pixel values represent measured radio brightness / flux-related quantities;
   colors in published images are usually a visualization choice.
10) Repeating imaging for many frequency channels creates a spectral cube:
        sky X × sky Y × frequency
    which can reveal gas motion through Doppler shifts.

The synthetic galaxy and measurements in this renderer are pedagogical, not a
real calibrated observation. The numerical FFT/IFFT and simplified Högbom-like
CLEAN are genuine mathematical operations used here to demonstrate the pipeline.


Subtitle behavior
-----------------
Exactly ONE subtitle layer is burned into the MP4 by default. An optional SRT is
written into a separate folder with a non-matching basename to avoid accidental
double subtitles in common media players.

Usage
-----
Normal 720p:
    python how_radio_telescope_renders_images.py

Fast validation:
    RADIO_IMAGE_QUICK=1 python how_radio_telescope_renders_images.py

1080p:
    RADIO_IMAGE_FULLHD=1 python how_radio_telescope_renders_images.py

No burned subtitles:
    RADIO_IMAGE_BURN_SUBTITLES=0 python how_radio_telescope_renders_images.py

Dependencies:
    pip install numpy pillow imageio imageio-ffmpeg
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

QUICK_MODE = os.environ.get("RADIO_IMAGE_QUICK", "0") == "1"
FULLHD_MODE = os.environ.get("RADIO_IMAGE_FULLHD", "0") == "1" and not QUICK_MODE
BURN_SUBTITLES = os.environ.get("RADIO_IMAGE_BURN_SUBTITLES", "1") == "1"

OUTPUT_ROOT = Path("how_radio_telescope_renders_images_output")
PREVIEW_DIR = OUTPUT_ROOT / "previews"
SUBTITLE_DIR = OUTPUT_ROOT / "optional_subtitles"
for directory in (OUTPUT_ROOT, PREVIEW_DIR, SUBTITLE_DIR):
    directory.mkdir(parents=True, exist_ok=True)

FULL_DURATION = 112.0
if QUICK_MODE:
    OUT_W, OUT_H, FPS, DURATION = 640, 360, 8, 30.0
elif FULLHD_MODE:
    OUT_W, OUT_H, FPS, DURATION = 1920, 1080, 24, FULL_DURATION
else:
    OUT_W, OUT_H, FPS, DURATION = 1280, 720, 24, FULL_DURATION

FRAME_COUNT = int(round(FPS * DURATION))
AUDIO_RATE = 44100

COLORS = {
    "bg0": (2, 5, 14),
    "bg1": (8, 15, 30),
    "white": (244, 248, 255),
    "muted": (164, 181, 199),
    "cyan": (89, 224, 245),
    "blue": (105, 158, 255),
    "gold": (246, 194, 92),
    "violet": (172, 132, 240),
    "pink": (246, 132, 188),
    "green": (124, 228, 176),
    "red": (244, 113, 118),
    "dish": (191, 204, 218),
    "grid": (77, 97, 122),
    "ground": (26, 31, 40),
}

SHOT_PLAN_FULL: List[Tuple[str, float, float]] = [
    ("hook", 0.0, 8.0),
    ("voltages", 8.0, 19.0),
    ("correlator", 19.0, 31.0),
    ("calibration", 31.0, 43.0),
    ("uv_sampling", 43.0, 57.0),
    ("dirty_image", 57.0, 69.0),
    ("clean", 69.0, 83.0),
    ("final_image", 83.0, 93.0),
    ("spectral_cube", 93.0, 105.0),
    ("finale", 105.0, 112.0),
]

NARRATION_FULL: List[Tuple[float, float, str]] = [
    (0.3, 7.7, "A radio telescope array does not receive a ready-made photograph. It receives waves, and records numbers."),
    (8.3, 18.7, "At each antenna, the receiver turns the arriving radio field into a tiny changing electrical voltage, sampled many times every second."),
    (19.3, 30.7, "A correlator compares the signals from pairs of antennas. Each pair measures one piece of information about structure on the sky."),
    (31.3, 42.7, "Before imaging, astronomers calibrate the data to correct timing, electronics, atmosphere, amplitude, and phase errors as accurately as possible."),
    (43.3, 56.7, "Every antenna pair gives a point in a sampling plane called the u-v plane. Many baselines and Earth rotation provide many different samples."),
    (57.3, 68.7, "The calibrated samples are transformed into an image. But because many spatial measurements are still missing, the first result is called a dirty image."),
    (69.3, 82.7, "The dirty image contains sidelobes from the array's sampling pattern. Deconvolution methods such as CLEAN model the real emission and remove much of that response."),
    (83.3, 92.7, "The restored result is a radio brightness image. The displayed colors are usually chosen by us; the underlying pixel values are scientific measurements."),
    (93.3, 104.7, "Do this separately across many frequencies and you get a data cube: two sky directions plus frequency. Spectral lines can then reveal gas and its motion."),
    (105.3, 111.7, "So the final image is not a camera snapshot. It is a calibrated reconstruction built from many precisely compared radio-wave measurements."),
]

if QUICK_MODE:
    SCALE = DURATION / FULL_DURATION
    SHOT_PLAN = [(name, a * SCALE, b * SCALE) for name, a, b in SHOT_PLAN_FULL]
    NARRATION = [(a * SCALE, b * SCALE, txt) for a, b, txt in NARRATION_FULL]
else:
    SHOT_PLAN = SHOT_PLAN_FULL
    NARRATION = NARRATION_FULL


# =============================================================================
# Helpers
# =============================================================================

def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(value)))


def smoothstep(value: float) -> float:
    x = clamp(value)
    return x * x * (3.0 - 2.0 * x)


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def get_shot(t: float) -> Tuple[str, float, float]:
    for shot in SHOT_PLAN:
        if shot[1] <= t < shot[2]:
            return shot
    return SHOT_PLAN[-1]


def narration_at(t: float) -> Optional[Tuple[float, float, str]]:
    for start, end, value in NARRATION:
        if start <= t < end:
            return start, end, value
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
    for index, (start, end, value) in enumerate(captions, 1):
        rows.extend([str(index), f"{format_srt_time(start)} --> {format_srt_time(end)}", value, ""])
    path.write_text("\n".join(rows), encoding="utf-8")


def get_font(size: int, bold: bool = False):
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
            pass
    return ImageFont.load_default()


def draw_text(image: Image.Image, text: str, xy: Tuple[int, int], size: int, fill, bold=False, anchor="la", stroke=2):
    ImageDraw.Draw(image).text(
        xy,
        text,
        font=get_font(size, bold),
        fill=fill,
        anchor=anchor,
        stroke_width=stroke,
        stroke_fill=(0, 0, 0, 210),
    )


def wrapped_text(image: Image.Image, text: str, box: Tuple[int, int, int, int], size: int, fill, bold=False, anchor="la"):
    x0, y0, x1, y1 = box
    draw = ImageDraw.Draw(image)
    font = get_font(size, bold)
    words = text.split()
    lines: List[str] = []
    cur = ""
    for word in words:
        trial = word if not cur else cur + " " + word
        b = draw.textbbox((0, 0), trial, font=font, stroke_width=2)
        if b[2] - b[0] <= x1 - x0:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    line_h = int(size * 1.25)
    total_h = line_h * len(lines)
    y = y0 + max(0, (y1 - y0 - total_h) // 2)
    for line in lines:
        if anchor == "ma":
            xx = (x0 + x1) // 2
        else:
            xx = x0
        draw.text((xx, y), line, font=font, fill=fill, anchor=anchor, stroke_width=2, stroke_fill=(0, 0, 0, 210))
        y += line_h


def rounded_panel(image: Image.Image, box: Tuple[int, int, int, int], alpha=140, outline_alpha=44):
    layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    r = max(8, int((box[3] - box[1]) * 0.12))
    d.rounded_rectangle(box, radius=r, fill=(2, 6, 16, alpha), outline=COLORS["cyan"] + (outline_alpha,), width=max(1, OUT_W // 900))
    image.alpha_composite(layer)


def draw_arrow(draw: ImageDraw.ImageDraw, p0, p1, fill, width=3, head=10.0):
    draw.line((p0, p1), fill=fill, width=width)
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    a = math.atan2(dy, dx)
    for off in (+2.55, -2.55):
        q = (p1[0] + math.cos(a + off) * head, p1[1] + math.sin(a + off) * head)
        draw.line((p1, q), fill=fill, width=width)


def make_vignette(width: int, height: int, strength=0.24) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width]
    nx = (xx - width / 2) / (width / 2)
    ny = (yy - height / 2) / (height / 2)
    r = np.sqrt(nx * nx + ny * ny)
    return np.clip(1.0 - strength * r ** 1.8, 0, 1).astype(np.float32)


VIGNETTE = make_vignette(OUT_W, OUT_H)


# =============================================================================
# Synthetic scientific dataset
# =============================================================================

def gaussian2d(xx, yy, cx, cy, sx, sy, amp=1.0, angle=0.0):
    c, s = math.cos(angle), math.sin(angle)
    xr = c * (xx - cx) + s * (yy - cy)
    yr = -s * (xx - cx) + c * (yy - cy)
    return amp * np.exp(-0.5 * ((xr / sx) ** 2 + (yr / sy) ** 2))


def make_true_sky(n: int = 256) -> np.ndarray:
    y, x = np.mgrid[-1:1:complex(n), -1:1:complex(n)]
    sky = np.zeros((n, n), dtype=np.float64)
    # Core + two radio lobes + faint curved-ish jet knots + compact sources.
    sky += gaussian2d(x, y, 0.0, 0.02, 0.045, 0.045, 1.00)
    sky += gaussian2d(x, y, -0.34, 0.03, 0.16, 0.10, 0.62, angle=0.08)
    sky += gaussian2d(x, y, 0.36, -0.01, 0.18, 0.11, 0.56, angle=-0.06)
    for i in range(6):
        f = i / 5
        sky += gaussian2d(x, y, 0.07 + 0.25 * f, 0.01 - 0.06 * f + 0.018 * math.sin(i), 0.025, 0.018, 0.22 * (1 - 0.08 * i))
    sky += gaussian2d(x, y, -0.68, -0.46, 0.025, 0.025, 0.19)
    sky += gaussian2d(x, y, 0.62, 0.52, 0.018, 0.018, 0.15)
    sky += gaussian2d(x, y, -0.05, 0.58, 0.022, 0.022, 0.10)
    sky /= max(float(sky.max()), 1e-12)
    return sky


def make_uv_mask(n: int = 256, antennas: int = 13, hours: int = 48, max_r: float = 0.44) -> np.ndarray:
    rng = np.random.default_rng(20260807)
    # Compact but irregular antenna placement.
    radii = np.sqrt(rng.uniform(0.03, 1.0, antennas))
    angles = rng.uniform(0, 2 * math.pi, antennas)
    ant = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    mask = np.zeros((n, n), dtype=np.float64)
    c = n // 2
    for h in np.linspace(-1.05, 1.05, hours):
        rot = np.array([[math.cos(h), -math.sin(h)], [math.sin(h), math.cos(h)]])
        pos = ant @ rot.T
        for i in range(antennas):
            for j in range(i + 1, antennas):
                uv = (pos[j] - pos[i]) * max_r
                for sign in (-1, 1):
                    u, v = sign * uv[0], sign * uv[1]
                    ix = int(round(c + u * n))
                    iy = int(round(c + v * n))
                    if 1 <= ix < n - 1 and 1 <= iy < n - 1:
                        # small gridding kernel rather than isolated pixels
                        mask[iy - 1:iy + 2, ix - 1:ix + 2] = np.maximum(mask[iy - 1:iy + 2, ix - 1:ix + 2], np.array([[0.2,0.5,0.2],[0.5,1.0,0.5],[0.2,0.5,0.2]]))
    mask[c, c] = 0.0  # interferometer lacks true zero-spacing measurement
    return np.clip(mask, 0, 1)


def normalize_signed(arr: np.ndarray) -> np.ndarray:
    m = max(float(np.max(np.abs(arr))), 1e-12)
    return arr / m


def hogbom_clean(dirty: np.ndarray, psf: np.ndarray, niter: int = 220, gain: float = 0.12, threshold: float = 0.015):
    residual = dirty.copy().astype(np.float64)
    model = np.zeros_like(residual)
    cy, cx = np.unravel_index(np.argmax(np.abs(psf)), psf.shape)
    psf_peak = psf[cy, cx]
    if abs(psf_peak) < 1e-12:
        return model, residual, dirty
    psf_norm = psf / psf_peak
    snapshots: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    for it in range(niter):
        iy, ix = np.unravel_index(np.argmax(np.abs(residual)), residual.shape)
        peak = residual[iy, ix]
        if abs(peak) < threshold:
            break
        comp = gain * peak
        model[iy, ix] += comp
        shifted = np.roll(np.roll(psf_norm, iy - cy, axis=0), ix - cx, axis=1)
        residual -= comp * shifted
        if it in (0, 12, 35, 80, 150, niter - 1):
            snapshots[it] = (model.copy(), residual.copy())
    # Restore model with a simple Gaussian "clean beam".
    n = dirty.shape[0]
    yy, xx = np.mgrid[:n, :n]
    sigma = max(1.4, n * 0.012)
    beam = np.exp(-0.5 * (((xx - n//2)/sigma)**2 + ((yy - n//2)/sigma)**2))
    beam /= beam.sum()
    restored_model = np.fft.ifft2(np.fft.fft2(model) * np.fft.fft2(np.fft.ifftshift(beam))).real
    restored = restored_model + residual
    return model, residual, restored, snapshots


def make_science_data(n: int = 256) -> Dict[str, np.ndarray]:
    sky = make_true_sky(n)
    ft = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(sky)))
    mask = make_uv_mask(n)
    sampled = ft * mask
    dirty = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(sampled))).real
    psf = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(mask))).real
    dirty_n = normalize_signed(dirty)
    psf_n = normalize_signed(psf)
    model, residual, restored, snapshots = hogbom_clean(dirty_n, psf_n)
    restored = np.clip(restored, -0.08, None)
    restored /= max(float(restored.max()), 1e-12)
    return {
        "sky": sky,
        "ft": ft,
        "mask": mask,
        "dirty": dirty_n,
        "psf": psf_n,
        "clean_model": model,
        "residual": residual,
        "restored": restored,
        "snapshots": snapshots,
    }


def heatmap_image(arr: np.ndarray, signed: bool = False, gamma: float = 0.65) -> Image.Image:
    a = np.asarray(arr, dtype=np.float64)
    if signed:
        m = max(float(np.max(np.abs(a))), 1e-12)
        z = np.clip(a / m, -1, 1)
        r = np.where(z >= 0, 0.15 + 0.85 * z, 0.08 + 0.12 * (1 + z))
        g = np.where(z >= 0, 0.18 + 0.75 * z, 0.16 + 0.25 * (1 + z))
        b = np.where(z >= 0, 0.28 + 0.40 * z, 0.38 + 0.62 * (-z))
    else:
        z = a - np.nanmin(a)
        z /= max(float(np.nanmax(z)), 1e-12)
        z = np.clip(z, 0, 1) ** gamma
        r = np.clip(0.05 + 1.10 * z, 0, 1)
        g = np.clip(0.08 + 1.35 * np.maximum(z - 0.18, 0), 0, 1)
        b = np.clip(0.20 + 1.55 * np.maximum(z - 0.48, 0), 0, 1)
    rgb = np.dstack([r, g, b])
    return Image.fromarray(np.uint8(np.clip(rgb, 0, 1) * 255), mode="RGB")


SCI = make_science_data(192 if QUICK_MODE else 256)
MAPS = {
    "sky": heatmap_image(SCI["sky"]),
    "dirty": heatmap_image(SCI["dirty"], signed=True),
    "psf": heatmap_image(SCI["psf"], signed=True),
    "restored": heatmap_image(SCI["restored"]),
    "residual": heatmap_image(SCI["residual"], signed=True),
}


# =============================================================================
# Scene
# =============================================================================

@dataclass
class Star:
    x: float
    y: float
    r: float
    a: int
    phase: float


class RadioImagingExplainer:
    def __init__(self):
        rng = np.random.default_rng(2026)
        self.stars: List[Star] = []
        for _ in range(220 if QUICK_MODE else 700):
            self.stars.append(Star(float(rng.uniform(0, OUT_W)), float(rng.uniform(0, OUT_H)), float(rng.uniform(0.25, 1.8) * OUT_W / 1280), int(rng.uniform(20, 130)), float(rng.uniform(0, 2*math.pi))))

    def background(self, t: float) -> Image.Image:
        im = Image.new("RGBA", (OUT_W, OUT_H), COLORS["bg0"] + (255,))
        d = ImageDraw.Draw(im)
        for y in range(OUT_H):
            p = y / max(1, OUT_H - 1)
            d.line((0, y, OUT_W, y), fill=(int(lerp(2, 8, p)), int(lerp(5, 14, p)), int(lerp(14, 30, p)), 255))
        for s in self.stars:
            a = int(s.a * (0.75 + 0.25 * math.sin(t * 0.9 + s.phase)))
            d.ellipse((s.x-s.r, s.y-s.r, s.x+s.r, s.y+s.r), fill=COLORS["white"] + (a,))
        return im

    def draw_header(self, im: Image.Image, title: str, subtitle: Optional[str] = None, accent="cyan"):
        draw_text(im, title, (OUT_W//2, int(OUT_H*0.085)), max(20, int(48*OUT_W/1280)), COLORS["white"] + (245,), True, "ma", 2)
        if subtitle:
            draw_text(im, subtitle, (OUT_W//2, int(OUT_H*0.135)), max(11, int(23*OUT_W/1280)), COLORS[accent] + (230,), False, "ma", 1)

    def draw_subtitle(self, im: Image.Image, start: float, end: float, caption: str, t: float):
        fi = clamp((t-start)/0.28) if not QUICK_MODE else clamp((t-start)/0.08)
        fo = clamp((end-t)/0.35) if not QUICK_MODE else clamp((end-t)/0.10)
        alpha = int(235 * min(fi, fo, 1.0))
        if alpha <= 0:
            return
        box = (int(OUT_W*0.075), int(OUT_H*0.805), int(OUT_W*0.925), int(OUT_H*0.925))
        rounded_panel(im, box, alpha=min(155, int(alpha*0.62)), outline_alpha=28)
        wrapped_text(im, caption, (box[0]+int(OUT_W*.025), box[1]+4, box[2]-int(OUT_W*.025), box[3]-4), max(12, int(27*OUT_W/1280)), COLORS["white"] + (alpha,), False, "ma")

    def draw_dish(self, im: Image.Image, center: Tuple[float,float], scale=1.0, angle_deg=0.0):
        layer = Image.new("RGBA", im.size, (0,0,0,0))
        d = ImageDraw.Draw(layer)
        cx, cy = center
        s = scale * OUT_W / 1280
        # pedestal
        d.polygon([(cx-18*s,cy+45*s),(cx+18*s,cy+45*s),(cx+12*s,cy+95*s),(cx-12*s,cy+95*s)], fill=COLORS["dish"]+(180,))
        # dish parabola represented by rotated arc polyline
        pts=[]
        a=math.radians(angle_deg)
        for q in np.linspace(-1,1,50):
            x=q*62*s; y=(q*q)*34*s
            xr=x*math.cos(a)-y*math.sin(a); yr=x*math.sin(a)+y*math.cos(a)
            pts.append((cx+xr,cy+yr))
        d.line(pts, fill=COLORS["dish"]+(245,), width=max(2,int(5*s)))
        # feed
        fx = cx + math.sin(a)*(-8*s); fy = cy + math.cos(a)*(-8*s)
        d.ellipse((fx-5*s,fy-5*s,fx+5*s,fy+5*s), fill=COLORS["gold"]+(245,))
        im.alpha_composite(layer)

    def paste_map(self, im: Image.Image, source: Image.Image, box: Tuple[int,int,int,int], alpha=1.0):
        x0,y0,x1,y1 = box
        p = source.resize((x1-x0,y1-y0), Image.Resampling.LANCZOS).convert("RGBA")
        if alpha < 1:
            a = p.getchannel("A").point(lambda v: int(v*alpha))
            p.putalpha(a)
        mask = Image.new("L", p.size, 0)
        md = ImageDraw.Draw(mask)
        md.rounded_rectangle((0,0,p.width-1,p.height-1), radius=max(8,int(min(p.size)*.035)), fill=255)
        p.putalpha(mask)
        im.alpha_composite(p,(x0,y0))
        fr=Image.new("RGBA",im.size,(0,0,0,0)); fd=ImageDraw.Draw(fr)
        fd.rounded_rectangle(box, radius=max(8,int(min(x1-x0,y1-y0)*.035)), outline=COLORS["cyan"]+(70,), width=max(1,OUT_W//900))
        im.alpha_composite(fr)

    def draw_waveform(self, im: Image.Image, box, phase=0.0, noisy=False, corrected=False, color=None):
        x0,y0,x1,y1=box
        layer=Image.new("RGBA",im.size,(0,0,0,0)); d=ImageDraw.Draw(layer)
        d.rounded_rectangle(box, radius=12, fill=(3,8,20,150), outline=COLORS["grid"]+(80,), width=1)
        mid=(y0+y1)/2
        d.line((x0+10,mid,x1-10,mid), fill=COLORS["grid"]+(80,), width=1)
        rng=np.random.default_rng(4)
        pts=[]
        for k in range(160):
            f=k/159; x=lerp(x0+14,x1-14,f)
            y=math.sin(2*math.pi*(3.4*f)+phase)
            y += 0.18*math.sin(2*math.pi*9.1*f+0.3)
            if noisy:
                y += 0.18*rng.normal()
                if not corrected:
                    y *= 0.72 + 0.28*math.sin(2*math.pi*.7*f+1.1)
            yy=mid-y*(y1-y0)*0.25
            pts.append((x,yy))
        d.line(pts, fill=(color or COLORS["cyan"])+(230,), width=max(1,OUT_W//500))
        im.alpha_composite(layer)

    def draw_uv_plane(self, im: Image.Image, box, reveal=1.0):
        x0,y0,x1,y1=box; w=x1-x0; h=y1-y0
        layer=Image.new("RGBA",im.size,(0,0,0,0)); d=ImageDraw.Draw(layer)
        d.rounded_rectangle(box,radius=14,fill=(3,8,20,165),outline=COLORS["grid"]+(80,),width=1)
        cx=(x0+x1)/2; cy=(y0+y1)/2
        d.line((x0+10,cy,x1-10,cy),fill=COLORS["grid"]+(90,),width=1); d.line((cx,y0+10,cx,y1-10),fill=COLORS["grid"]+(90,),width=1)
        yy,xx=np.nonzero(SCI["mask"]>.7)
        total=len(xx); lim=int(total*clamp(reveal))
        stride=max(1,total//3500)
        for ix,iy in zip(xx[:lim:stride],yy[:lim:stride]):
            px=x0+(ix/(SCI["mask"].shape[1]-1))*w; py=y0+(iy/(SCI["mask"].shape[0]-1))*h
            r=max(1,OUT_W//900)
            d.ellipse((px-r,py-r,px+r,py+r),fill=COLORS["cyan"]+(150,))
        draw_text(layer,"u",(x1-18,cy-8),max(10,int(20*OUT_W/1280)),COLORS["muted"]+(210,),False,"ma",1)
        draw_text(layer,"v",(cx+12,y0+18),max(10,int(20*OUT_W/1280)),COLORS["muted"]+(210,),False,"ma",1)
        im.alpha_composite(layer)

    def draw_colorbar(self, im: Image.Image, x: int, y: int, w: int, h: int):
        d=ImageDraw.Draw(im)
        for i in range(w):
            z=i/max(1,w-1)
            r=int(255*min(1,0.05+1.1*z)); g=int(255*min(1,0.08+1.35*max(z-.18,0))); b=int(255*min(1,0.20+1.55*max(z-.48,0)))
            d.line((x+i,y,x+i,y+h),fill=(r,g,b,255))
        draw_text(im,"faint",(x,y+h+16),max(8,int(16*OUT_W/1280)),COLORS["muted"]+(210,),False,"ma",1)
        draw_text(im,"bright",(x+w,y+h+16),max(8,int(16*OUT_W/1280)),COLORS["muted"]+(210,),False,"ma",1)

    def draw_spectral_cube(self, im: Image.Image, local: float):
        self.draw_header(im,"A RADIO IMAGE CAN BECOME A DATA CUBE","x × y × frequency",accent="gold")
        cx,cy=OUT_W*.50,OUT_H*.47
        size=int(OUT_H*.40)
        # Create channel slices where left/right side peaks shift with frequency.
        n=128; yy,xx=np.mgrid[-1:1:complex(n),-1:1:complex(n)]
        for k in range(5):
            z=(k-2)/2
            amp_l=math.exp(-0.5*((z+0.55)/0.55)**2); amp_r=math.exp(-0.5*((z-0.55)/0.55)**2)
            arr=gaussian2d(xx,yy,-.28,0,.20,.12,amp_l)+gaussian2d(xx,yy,.28,0,.20,.12,amp_r)+0.25*gaussian2d(xx,yy,0,0,.10,.10,1)
            ch=heatmap_image(arr)
            off=int((k-2)*size*.11); yoff=int((2-k)*size*.05)
            box=(int(cx-size*.42+off),int(cy-size*.42+yoff),int(cx+size*.42+off),int(cy+size*.42+yoff))
            self.paste_map(im,ch,box,alpha=0.40+0.12*k)
        # frequency axis
        layer=Image.new("RGBA",im.size,(0,0,0,0));d=ImageDraw.Draw(layer)
        p0=(int(OUT_W*.25),int(OUT_H*.73));p1=(int(OUT_W*.75),int(OUT_H*.73));draw_arrow(d,p0,p1,COLORS["gold"]+(220,),max(2,OUT_W//500),12*OUT_W/1280)
        im.alpha_composite(layer)
        draw_text(im,"frequency →",(OUT_W//2,int(OUT_H*.765)),max(11,int(24*OUT_W/1280)),COLORS["gold"]+(230,),True,"ma",1)
        draw_text(im,"different channels can trace different Doppler velocities",(OUT_W//2,int(OUT_H*.80)),max(10,int(20*OUT_W/1280)),COLORS["white"]+(220,),False,"ma",1)

    def frame(self, t: float) -> np.ndarray:
        shot,start,end=get_shot(t); local=smoothstep((t-start)/max(end-start,1e-9))
        im=self.background(t)

        if shot=="hook":
            self.draw_header(im,"HOW RADIO TELESCOPES RENDER IMAGES","Part 2 • from voltages to a scientific map",accent="gold")
            # Split: raw waves -> final image
            self.draw_dish(im,(OUT_W*.20,OUT_H*.52),1.25,-14)
            for i in range(5):
                x0=OUT_W*.29+i*OUT_W*.035
                layer=Image.new("RGBA",im.size,(0,0,0,0));d=ImageDraw.Draw(layer)
                d.arc((x0-50,x0*0+OUT_H*.37,x0+50,OUT_H*.67),-75,75,fill=COLORS["cyan"]+(80+i*22,),width=max(1,OUT_W//600));im.alpha_composite(layer)
            self.paste_map(im,MAPS["restored"],(int(OUT_W*.56),int(OUT_H*.25),int(OUT_W*.88),int(OUT_H*.69)))
            draw_text(im,"NOT A CAMERA SNAPSHOT",(OUT_W//2,int(OUT_H*.735)),max(14,int(29*OUT_W/1280)),COLORS["gold"]+(240,),True,"ma",1)

        elif shot=="voltages":
            self.draw_header(im,"STEP 1 — RECORD THE WAVE","the receiver outputs a changing voltage")
            self.draw_dish(im,(OUT_W*.22,OUT_H*.50),1.2,-12)
            self.draw_waveform(im,(int(OUT_W*.43),int(OUT_H*.32),int(OUT_W*.88),int(OUT_H*.56)),phase=t*6.5,noisy=True)
            draw_text(im,"voltage",(int(OUT_W*.655),int(OUT_H*.29)),max(11,int(22*OUT_W/1280)),COLORS["cyan"]+(225,),True,"ma",1)
            draw_text(im,"sample → number → sample → number ...",(int(OUT_W*.655),int(OUT_H*.62)),max(10,int(21*OUT_W/1280)),COLORS["muted"]+(220,),False,"ma",1)

        elif shot=="correlator":
            self.draw_header(im,"STEP 2 — COMPARE ANTENNA PAIRS","the correlator measures how similar the signals are",accent="violet")
            self.draw_dish(im,(OUT_W*.18,OUT_H*.51),.86,-12); self.draw_dish(im,(OUT_W*.40,OUT_H*.51),.86,-8); self.draw_dish(im,(OUT_W*.62,OUT_H*.51),.86,-4)
            self.draw_waveform(im,(int(OUT_W*.70),int(OUT_H*.25),int(OUT_W*.94),int(OUT_H*.39)),phase=0.1+t*4,color=COLORS["cyan"])
            self.draw_waveform(im,(int(OUT_W*.70),int(OUT_H*.43),int(OUT_W*.94),int(OUT_H*.57)),phase=0.8+t*4,color=COLORS["gold"])
            rounded_panel(im,(int(OUT_W*.72),int(OUT_H*.62),int(OUT_W*.92),int(OUT_H*.71)),120,35)
            draw_text(im,"CORRELATE",(int(OUT_W*.82),int(OUT_H*.665)),max(11,int(24*OUT_W/1280)),COLORS["white"]+(240,),True,"mm",1)
            draw_text(im,"each pair → one measurement",(OUT_W//2,int(OUT_H*.755)),max(11,int(24*OUT_W/1280)),COLORS["violet"]+(230,),True,"ma",1)

        elif shot=="calibration":
            self.draw_header(im,"STEP 3 — CALIBRATE THE DATA","correct the instrument + atmosphere before trusting the image",accent="green")
            self.draw_waveform(im,(int(OUT_W*.08),int(OUT_H*.27),int(OUT_W*.46),int(OUT_H*.56)),phase=t*3,noisy=True,corrected=False,color=COLORS["red"])
            self.draw_waveform(im,(int(OUT_W*.54),int(OUT_H*.27),int(OUT_W*.92),int(OUT_H*.56)),phase=t*3,noisy=True,corrected=True,color=COLORS["green"])
            draw_text(im,"RAW / DISTORTED",(int(OUT_W*.27),int(OUT_H*.61)),max(11,int(23*OUT_W/1280)),COLORS["red"]+(230,),True,"ma",1)
            draw_text(im,"CALIBRATED",(int(OUT_W*.73),int(OUT_H*.61)),max(11,int(23*OUT_W/1280)),COLORS["green"]+(230,),True,"ma",1)
            draw_text(im,"timing • gain • phase • bandpass • atmosphere",(OUT_W//2,int(OUT_H*.71)),max(10,int(21*OUT_W/1280)),COLORS["muted"]+(220,),False,"ma",1)

        elif shot=="uv_sampling":
            self.draw_header(im,"STEP 4 — COLLECT SPATIAL INFORMATION","each baseline samples one place in the u-v plane",accent="cyan")
            # mini array left; uv right
            for k,(x,y) in enumerate([(0.13,.51),(.23,.39),(.30,.58),(.39,.46),(.18,.66)]): self.draw_dish(im,(OUT_W*x,OUT_H*y),.58,-8+k*2)
            self.draw_uv_plane(im,(int(OUT_W*.50),int(OUT_H*.22),int(OUT_W*.90),int(OUT_H*.70)),reveal=local)
            draw_text(im,"more baselines + time = better sampling",(OUT_W//2,int(OUT_H*.75)),max(11,int(23*OUT_W/1280)),COLORS["cyan"]+(230,),True,"ma",1)
            draw_text(im,"(Fourier mathematics explained in a separate short)",(OUT_W//2,int(OUT_H*.785)),max(9,int(17*OUT_W/1280)),COLORS["muted"]+(190,),False,"ma",1)

        elif shot=="dirty_image":
            self.draw_header(im,"STEP 5 — MAKE THE FIRST IMAGE","incomplete sampling leaves a characteristic blur + sidelobes",accent="gold")
            self.paste_map(im,MAPS["dirty"],(int(OUT_W*.08),int(OUT_H*.22),int(OUT_W*.46),int(OUT_H*.70)))
            self.paste_map(im,MAPS["psf"],(int(OUT_W*.54),int(OUT_H*.22),int(OUT_W*.92),int(OUT_H*.70)))
            draw_text(im,"DIRTY IMAGE",(int(OUT_W*.27),int(OUT_H*.735)),max(11,int(23*OUT_W/1280)),COLORS["white"]+(240,),True,"ma",1)
            draw_text(im,"DIRTY BEAM / PSF",(int(OUT_W*.73),int(OUT_H*.735)),max(11,int(23*OUT_W/1280)),COLORS["gold"]+(235,),True,"ma",1)

        elif shot=="clean":
            self.draw_header(im,"STEP 6 — DECONVOLVE","CLEAN-like reconstruction: model sources, subtract the beam response",accent="green")
            # animate blend dirty -> restored + residual shrinking
            mix=local
            a=np.asarray(MAPS["dirty"].resize(MAPS["restored"].size),dtype=np.float32)
            b=np.asarray(MAPS["restored"],dtype=np.float32)
            blended=Image.fromarray(np.uint8(np.clip(a*(1-mix)+b*mix,0,255)))
            self.paste_map(im,blended,(int(OUT_W*.09),int(OUT_H*.22),int(OUT_W*.57),int(OUT_H*.70)))
            # right conceptual loop
            x=int(OUT_W*.70); y0=int(OUT_H*.29); gap=int(OUT_H*.105)
            labels=[("1  find peak",COLORS["cyan"]),("2  subtract PSF",COLORS["gold"]),("3  save component",COLORS["violet"]),("4  repeat",COLORS["green"])]
            for i,(lab,col) in enumerate(labels):
                rounded_panel(im,(x-int(OUT_W*.09),y0+i*gap,x+int(OUT_W*.16),y0+i*gap+int(OUT_H*.065)),110,25)
                draw_text(im,lab,(x+int(OUT_W*.035),y0+i*gap+int(OUT_H*.032)),max(10,int(20*OUT_W/1280)),col+(235,),True,"mm",1)
            draw_text(im,"simplified visualization",(int(OUT_W*.73),int(OUT_H*.74)),max(9,int(17*OUT_W/1280)),COLORS["muted"]+(190,),False,"ma",1)

        elif shot=="final_image":
            self.draw_header(im,"THE RESTORED RADIO IMAGE","pixels contain measured brightness information",accent="gold")
            self.paste_map(im,MAPS["restored"],(int(OUT_W*.22),int(OUT_H*.19),int(OUT_W*.78),int(OUT_H*.72)))
            self.draw_colorbar(im,int(OUT_W*.30),int(OUT_H*.755),int(OUT_W*.40),max(5,int(OUT_H*.018)))
            draw_text(im,"color is visualization — the numbers are the science",(OUT_W//2,int(OUT_H*.82)),max(10,int(21*OUT_W/1280)),COLORS["white"]+(215,),False,"ma",1)

        elif shot=="spectral_cube":
            self.draw_spectral_cube(im,local)

        elif shot=="finale":
            self.draw_header(im,"FROM RADIO WAVES TO A SCIENTIFIC IMAGE","not photographed — reconstructed from calibrated measurements",accent="gold")
            steps=["VOLTAGES","CORRELATE","CALIBRATE","SAMPLE","IMAGE","DECONVOLVE"]
            y=int(OUT_H*.43); xs=np.linspace(OUT_W*.12,OUT_W*.88,len(steps))
            layer=Image.new("RGBA",im.size,(0,0,0,0));d=ImageDraw.Draw(layer)
            for i in range(len(xs)-1): draw_arrow(d,(xs[i]+OUT_W*.035,y),(xs[i+1]-OUT_W*.035,y),COLORS["grid"]+(180,),max(1,OUT_W//700),9*OUT_W/1280)
            im.alpha_composite(layer)
            for x,lab in zip(xs,steps):
                rr=max(18,int(30*OUT_W/1280)); ImageDraw.Draw(im).ellipse((x-rr,y-rr,x+rr,y+rr),fill=(8,18,32,230),outline=COLORS["cyan"]+(120,),width=max(1,OUT_W//700))
                draw_text(im,lab,(int(x),int(y+OUT_H*.075)),max(8,int(16*OUT_W/1280)),COLORS["white"]+(230,),True,"ma",1)
            self.paste_map(im,MAPS["restored"],(int(OUT_W*.40),int(OUT_H*.57),int(OUT_W*.60),int(OUT_H*.78)))

        narr=narration_at(t)
        if BURN_SUBTITLES and narr:
            self.draw_subtitle(im,*narr,t)

        draw_text(im,"Synthetic pedagogical radio source • imaging math is real; pipeline is simplified",(int(OUT_W*.018),int(OUT_H*.985)),max(7,int(11*OUT_W/1280)),COLORS["muted"]+(105,),False,"ls",1)
        arr=np.asarray(im.convert("RGB"),dtype=np.float32)
        arr=np.clip(arr*VIGNETTE[...,None],0,255).astype(np.uint8)
        out=Image.fromarray(arr)
        out=ImageEnhance.Contrast(out).enhance(1.08)
        return np.asarray(out)


# =============================================================================
# Audio and output
# =============================================================================

def create_soundtrack(path: Path, duration_s: float):
    sr=AUDIO_RATE; n=int(duration_s*sr); t=np.arange(n,dtype=np.float32)/sr
    rng=np.random.default_rng(91)
    audio=(0.11*np.sin(2*math.pi*46*t)+0.055*np.sin(2*math.pi*69*t+.5)+0.025*np.sin(2*math.pi*138*t+1.1)+0.008*rng.normal(0,1,n).astype(np.float32))
    for _,start,_ in SHOT_PLAN[1:]:
        i0=int(start*sr); L=min(int(.45*sr),n-i0)
        if L<=0: continue
        tt=np.arange(L,dtype=np.float32)/sr
        audio[i0:i0+L]+=0.08*np.sin(2*math.pi*92*tt)*np.exp(-tt*6)+0.014*rng.normal(0,1,L).astype(np.float32)*np.exp(-tt*4)
    # light digital ticks in correlator/calibration region
    for sec in np.arange(SHOT_PLAN[2][1],SHOT_PLAN[4][1],0.72 if not QUICK_MODE else .24):
        i0=int(sec*sr);L=min(int(.09*sr),n-i0)
        if L>0:
            tt=np.arange(L,dtype=np.float32)/sr
            audio[i0:i0+L]+=0.025*np.sin(2*math.pi*520*tt)*np.exp(-tt*28)
    audio/=max(float(np.max(np.abs(audio))),1e-9); audio*=.68
    left=audio*(.98+.02*np.sin(2*math.pi*.04*t)); right=audio*(.98+.02*np.cos(2*math.pi*.04*t+.6))
    pcm=np.int16(np.clip(np.stack([left,right],axis=1),-1,1)*32767)
    with wave.open(str(path),"wb") as wav:
        wav.setnchannels(2);wav.setsampwidth(2);wav.setframerate(sr);wav.writeframes(pcm.tobytes())


def render_video(scene: RadioImagingExplainer, path: Path):
    with iio.get_writer(str(path),fps=FPS,codec="libx264",quality=8,pixelformat="yuv420p") as writer:
        for i in range(FRAME_COUNT):
            writer.append_data(scene.frame(i/FPS))


def mux_audio(video: Path, audio: Path, final: Path) -> bool:
    ff=shutil.which("ffmpeg")
    if not ff:
        shutil.copy2(video,final);return False
    cmd=[ff,"-y","-i",str(video),"-i",str(audio),"-c:v","copy","-c:a","aac","-b:a","192k","-shortest",str(final)]
    try:
        subprocess.run(cmd,check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE);return True
    except Exception:
        shutil.copy2(video,final);return False


def save_contact_sheet(scene: RadioImagingExplainer, path: Path):
    samples=[]
    for _,a,b in SHOT_PLAN:
        samples.append(a+(b-a)*.55)
    thumb_w=320 if not QUICK_MODE else 240; thumb_h=int(thumb_w*OUT_H/OUT_W)
    thumbs=[Image.fromarray(scene.frame(t)).resize((thumb_w,thumb_h),Image.Resampling.LANCZOS) for t in samples]
    cols=5;rows=2
    sheet=Image.new("RGB",(thumb_w*cols,thumb_h*rows),(4,7,14))
    for i,th in enumerate(thumbs): sheet.paste(th,((i%cols)*thumb_w,(i//cols)*thumb_h))
    sheet.save(path,quality=92)


def write_narration(path: Path):
    rows=["HOW RADIO TELESCOPES RENDER IMAGES — PART 2","","Narration / subtitle script",""]
    for a,b,txt in NARRATION_FULL: rows.append(f"[{a:05.1f}–{b:05.1f}] {txt}")
    path.write_text("\n".join(rows),encoding="utf-8")


def write_summary(path: Path, audio_muxed: bool):
    payload: Dict[str,Any]={
        "title":"How Radio Telescopes Render Images",
        "series_part":2,
        "duration_seconds":DURATION,
        "quick_mode":QUICK_MODE,
        "burned_subtitles":BURN_SUBTITLES,
        "audio_muxed":audio_muxed,
        "scientific_pipeline":[
            "receiver voltage samples",
            "pairwise correlation -> complex visibility measurements",
            "calibration of instrumental/atmospheric effects",
            "u-v sampling by baselines over time",
            "Fourier-related imaging transform -> dirty image",
            "dirty beam / point spread function from incomplete sampling",
            "deconvolution such as CLEAN",
            "restored radio brightness image",
            "multi-channel imaging -> spectral cube",
        ],
        "scientific_caveats":[
            "The sky source is synthetic and is not a real galaxy observation.",
            "The CLEAN implementation is deliberately simplified for education.",
            "Real pipelines may include flagging, weighting, gridding, primary-beam corrections, wide-field corrections, self-calibration, mosaicking, polarization calibration, and more.",
            "An interferometer does not measure the true zero-spacing flux unless additional short-spacing/single-dish information is supplied.",
            "Displayed image colors are visualization choices; calibrated pixel values carry the physical measurement.",
            "Fourier/FFT mathematics is mentioned but intentionally not derived in this episode.",
        ],
        "authoritative_background_sources":[
            "https://www.almaobservatory.org/en/about-alma/how-alma-works/technologies/interferometry/",
            "https://science.nrao.edu/facilities/vla/data-processing/end-to-end-recipes/continuum",
            "https://science.nrao.edu/facilities/vla/docs/manuals/oss2026a/performance/limitations-on-imaging-performance",
        ],
    }
    path.write_text(json.dumps(payload,indent=2),encoding="utf-8")

def main():
    scene=RadioImagingExplainer()
    silent=OUTPUT_ROOT/"how_radio_telescope_renders_images_silent.mp4"
    audio=OUTPUT_ROOT/"how_radio_telescope_renders_images_soundtrack.wav"
    final=OUTPUT_ROOT/"how_radio_telescope_renders_images.mp4"
    srt=SUBTITLE_DIR/"part2_optional_subtitles.srt"
    narration=OUTPUT_ROOT/"how_radio_telescope_renders_images_narration.txt"
    summary=OUTPUT_ROOT/"how_radio_telescope_renders_images_summary.json"
    contact=PREVIEW_DIR/"contact_sheet.jpg"

    write_srt(NARRATION,srt)
    write_narration(narration)
    save_contact_sheet(scene,contact)
    render_video(scene,silent)
    create_soundtrack(audio,DURATION)
    muxed=mux_audio(silent,audio,final)
    write_summary(summary,muxed)

    print("Render complete")
    print("Video:",final.resolve())
    print("Contact sheet:",contact.resolve())
    print("Narration:",narration.resolve())
    print("Optional SRT:",srt.resolve())
    print("Summary:",summary.resolve())


if __name__=="__main__":
    main()
