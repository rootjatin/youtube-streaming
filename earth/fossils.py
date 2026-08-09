from __future__ import annotations

"""
Result : https://youtu.be/z-5ZsOBPhGM
FOSSILS: BEHAVIOUR WRITTEN IN STONE
===================================

A 4.5-minute widescreen cinematic palaeontology documentary rendered entirely
with Python. The film moves through a quarry, follows dinosaur trackways,
reveals a brooding nest, enters the evidence preserved in coprolites and bite
marks, examines healed combat injuries, descends into a fossil burrow, and ends
with iridescent feathers and the limits of behavioural inference.

There are NO graphs, plots, axes, dashboards, or classroom-style data cards.
Information appears only through restrained documentary captions, specimen
labels, cinematic reconstructions, and brief evidence statements.

REAL-EVIDENCE FOUNDATION
------------------------
The renderer uses a curated set of measurements and conclusions from primary
palaeontological research:

- Fossil trackways provide direct evidence of gait, speed, and distal limb
  motion, although substrate and preservation complicate interpretation.
- An oviraptorid was preserved over a nest of eggs in a posture closely
  resembling brooding birds.
- A large tyrannosaur coprolite contained approximately 30–50% bone fragments,
  preserving direct evidence of diet and digestive processing.
- A Triceratops pathology study found a non-random distribution of healed
  cranial injuries consistent with intraspecific horn combat; the authors
  explicitly cautioned that individual lesions cannot be assigned a precise
  cause.
- An adult Oryctodromeus and two juveniles were preserved in the expanded end
  chamber of a sediment-filled burrow, providing direct evidence of burrowing
  and an association interpreted as extensive parental care.
- Fossil melanosomes indicate that Microraptor possessed black, iridescent
  plumage. A display function is plausible, but remains an inference.

PRIMARY REFERENCES
------------------
- Falkingham, P. L. (2025), Reconstructing dinosaur locomotion.
  https://pmc.ncbi.nlm.nih.gov/articles/PMC11732409/
- Norell et al. (1995), A nesting dinosaur, Nature 378, 774–776.
  https://doi.org/10.1038/378774a0
- Chin et al. (1998), A king-sized theropod coprolite, Nature 393, 680–682.
  https://doi.org/10.1038/31461
- Farke, Wolff & Tanke (2009), Evidence of Combat in Triceratops.
  https://doi.org/10.1371/journal.pone.0004252
- Varricchio, Martin & Katsura (2007), First trace and body fossil evidence of
  a burrowing, denning dinosaur.
  https://doi.org/10.1098/rspb.2006.0443
- Li et al. (2012), Reconstruction of Microraptor and the evolution of
  iridescent plumage, Science 335, 1215–1219.
  https://doi.org/10.1126/science.1213780

SCIENTIFIC HONESTY
------------------
The numerical records and published interpretations are real. The animals,
landscapes, colours, poses, camera movements, sediment, lighting, and sequence
of events are procedural artistic visualisations. A fossil preserves evidence;
it rarely preserves a complete explanation. The film distinguishes direct
observations from interpretations whenever possible.

INSTALL
-------
    pip install numpy pillow imageio imageio-ffmpeg tqdm

FULL 4.5-MINUTE RENDER
----------------------
    python fossils_behaviour_written_in_stone.py

FAST TEST
---------
    FOSSIL_BEHAVIOUR_QUICK=1 python fossils_behaviour_written_in_stone.py

PREVIEWS ONLY
-------------
    FOSSIL_BEHAVIOUR_PREVIEW_ONLY=1 python fossils_behaviour_written_in_stone.py

CUSTOM DURATION / RESOLUTION
----------------------------
    FOSSIL_BEHAVIOUR_DURATION=240 python fossils_behaviour_written_in_stone.py
    FOSSIL_BEHAVIOUR_DURATION=300 python fossils_behaviour_written_in_stone.py
    FOSSIL_BEHAVIOUR_4K=1 python fossils_behaviour_written_in_stone.py

USE YOUR OWN MUSIC
------------------
    FOSSIL_BEHAVIOUR_AUDIO=/path/to/music.wav python fossils_behaviour_written_in_stone.py
"""

import hashlib
import json
import math
import os
import shutil
import subprocess
import wave
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import imageio.v2 as iio
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont
from tqdm.auto import tqdm


# =============================================================================
# Configuration
# =============================================================================

QUICK_MODE = os.getenv("FOSSIL_BEHAVIOUR_QUICK", "0") == "1"
PREVIEW_ONLY = os.getenv("FOSSIL_BEHAVIOUR_PREVIEW_ONLY", "0") == "1"
FOUR_K = os.getenv("FOSSIL_BEHAVIOUR_4K", "0") == "1" and not QUICK_MODE
EXTERNAL_AUDIO = os.getenv("FOSSIL_BEHAVIOUR_AUDIO", "").strip() or None

DEFAULT_DURATION = 24.0 if QUICK_MODE else 270.0
DURATION = float(os.getenv("FOSSIL_BEHAVIOUR_DURATION", str(DEFAULT_DURATION)))
FPS = int(os.getenv("FOSSIL_BEHAVIOUR_FPS", "10" if QUICK_MODE else "24"))

if QUICK_MODE:
    WIDTH, HEIGHT = 960, 540
elif FOUR_K:
    WIDTH, HEIGHT = 3840, 2160
else:
    WIDTH, HEIGHT = 1920, 1080

OUT_SIZE = (WIDTH, HEIGHT)
SCALE = WIDTH / 1920.0
RENDER_SCALE = 0.58 if QUICK_MODE else (0.50 if FOUR_K else 0.56)
RW, RH = max(480, int(WIDTH * RENDER_SCALE)), max(270, int(HEIGHT * RENDER_SCALE))

OUTPUT_ROOT = Path("fossils_behaviour_written_in_stone_output")
DATA_ROOT = OUTPUT_ROOT / "data"
PREVIEW_DIR = OUTPUT_ROOT / "previews"
AUDIO_DIR = OUTPUT_ROOT / "audio"
for directory in (OUTPUT_ROOT, DATA_ROOT, PREVIEW_DIR, AUDIO_DIR):
    directory.mkdir(parents=True, exist_ok=True)

CONFIG: Dict[str, object] = {
    "title": "FOSSILS",
    "subtitle": "BEHAVIOUR WRITTEN IN STONE",
    "youtube_title": "How Fossils Reveal the Behaviour of Extinct Animals — Cinematic Documentary",
    "output_basename": "fossils_behaviour_written_in_stone",
    "width": WIDTH,
    "height": HEIGHT,
    "fps": FPS,
    "duration_s": DURATION,
    "audio_sample_rate": 48_000,
    "write_subtitle_sidecar": True,
    "audio_path": EXTERNAL_AUDIO,
}

CHAPTERS = [
    (0.000, "A moment becomes stone"),
    (0.125, "Footsteps without bodies"),
    (0.285, "A parent over the nest"),
    (0.440, "The evidence of feeding"),
    (0.595, "Scars that healed"),
    (0.740, "A family beneath the ground"),
    (0.880, "Colour, display, uncertainty"),
]


# =============================================================================
# Evidence records
# =============================================================================

@dataclass(frozen=True)
class FossilEvidence:
    key: str
    specimen: str
    age_ma: float
    observation: str
    interpretation: str
    confidence: str
    source: str
    numeric_value: Optional[float] = None
    numeric_unit: str = ""


EVIDENCE: Dict[str, FossilEvidence] = {
    "trackways": FossilEvidence(
        key="trackways",
        specimen="Fossil trackways",
        age_ma=0.0,
        observation="Sequences of footprints preserve stride, foot placement, direction and changes in pace.",
        interpretation="They directly constrain gait and speed, while substrate and preservation introduce uncertainty.",
        confidence="DIRECT TRACE / MODELLED MOTION",
        source="Falkingham (2025), Reconstructing dinosaur locomotion",
    ),
    "nest": FossilEvidence(
        key="nest",
        specimen="Oviraptorid over an egg clutch",
        age_ma=75.0,
        observation="An adult skeleton was preserved directly above a nest in a bird-like posture.",
        interpretation="The association provides strong evidence for brooding behaviour.",
        confidence="DIRECT ASSOCIATION",
        source="Norell et al. (1995), Nature 378",
    ),
    "coprolite": FossilEvidence(
        key="coprolite",
        specimen="Large tyrannosaur coprolite",
        age_ma=66.0,
        observation="The fossilised dropping contains a high proportion of fragmented bone.",
        interpretation="It records carnivorous diet, bone consumption and digestive processing.",
        confidence="DIRECT DIETARY EVIDENCE",
        source="Chin et al. (1998), Nature 393",
        numeric_value=40.0,
        numeric_unit="% BONE FRAGMENTS (REPORTED RANGE 30–50%)",
    ),
    "combat": FossilEvidence(
        key="combat",
        specimen="Triceratops cranial pathologies",
        age_ma=67.0,
        observation="Healed lesions occur in a non-random pattern across the horns, face and frill.",
        interpretation="The population-level pattern is consistent with horn-to-horn combat between individuals.",
        confidence="STATISTICAL PATTERN / CAUTIOUS INFERENCE",
        source="Farke, Wolff & Tanke (2009), PLOS ONE",
        numeric_value=0.002,
        numeric_unit="P VALUE FOR SQUAMOSAL DIFFERENCE",
    ),
    "burrow": FossilEvidence(
        key="burrow",
        specimen="Oryctodromeus burrow assemblage",
        age_ma=95.0,
        observation="One adult and two juveniles were found in the expanded terminal chamber of a burrow.",
        interpretation="The geometry and skeletons provide direct evidence of burrowing and prolonged parental association.",
        confidence="TRACE + BODY FOSSILS",
        source="Varricchio, Martin & Katsura (2007), Proc. R. Soc. B",
        numeric_value=3.0,
        numeric_unit="INDIVIDUALS IN TERMINAL CHAMBER",
    ),
    "feathers": FossilEvidence(
        key="feathers",
        specimen="Microraptor plumage",
        age_ma=120.0,
        observation="Fossilised melanosomes indicate glossy black, iridescent feathers.",
        interpretation="Iridescence may have contributed to visual display, but colour alone does not preserve intent.",
        confidence="COLOUR DIRECT / FUNCTION INFERRED",
        source="Li et al. (2012), Science 335",
    ),
}


# =============================================================================
# Helpers
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


def deterministic_unit(text: str, salt: str = "") -> float:
    digest = hashlib.sha256(f"{text}|{salt}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64 - 1)


def get_font(size: int, bold: bool = False, serif: bool = False):
    size = max(8, int(size))
    candidates: List[str] = []
    if serif:
        candidates += [
            "DejaVuSerif-Bold.ttf" if bold else "DejaVuSerif.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
        ]
    candidates += [
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "arialbd.ttf" if bold else "arial.ttf",
    ]
    for name in candidates:
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
    )


def draw_wrapped_text(
    image: Image.Image,
    text: str,
    xy: Tuple[int, int],
    max_width: int,
    size: int,
    fill=(255, 255, 255, 245),
    bold: bool = False,
    line_spacing: int = 8,
) -> int:
    draw = ImageDraw.Draw(image)
    font = get_font(int(size * SCALE), bold=bold)
    words = text.split()
    lines: List[str] = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        box = draw.textbbox((0, 0), candidate, font=font)
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
        box = draw.textbbox((x, y), line, font=font)
        y += box[3] - box[1] + int(line_spacing * SCALE)
    return y


def format_srt_time(seconds: float) -> str:
    milliseconds = int(round(seconds * 1000.0))
    hours, milliseconds = divmod(milliseconds, 3_600_000)
    minutes, milliseconds = divmod(milliseconds, 60_000)
    secs, milliseconds = divmod(milliseconds, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"


def chapter_at(fraction: float) -> Tuple[int, str, float]:
    for index in range(len(CHAPTERS) - 1):
        start, title = CHAPTERS[index]
        end = CHAPTERS[index + 1][0]
        if start <= fraction < end:
            return index, title, (fraction - start) / max(end - start, 1e-8)
    return len(CHAPTERS) - 1, CHAPTERS[-1][1], 1.0


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


# =============================================================================
# Narration
# =============================================================================

@dataclass(frozen=True)
class CaptionCue:
    start_fraction: float
    end_fraction: float
    text: str
    evidence_key: Optional[str] = None


NARRATION: List[CaptionCue] = [
    CaptionCue(0.010, 0.075, "Bones tell us what an extinct animal was. Behaviour survives only when an action leaves a physical trace."),
    CaptionCue(0.085, 0.135, "A footprint pressed into wet sediment can outlive the foot that made it by more than one hundred million years."),
    CaptionCue(0.145, 0.215, "A sequence of prints preserves direction, stride and changes in pace—the geometry of a moving body.", "trackways"),
    CaptionCue(0.225, 0.280, "But mud deforms. Tracks are direct evidence of contact, not a perfect recording of the animal above them."),
    CaptionCue(0.300, 0.370, "In Mongolia, an oviraptorid was fossilised over a ring of eggs, its limbs spread around the clutch."),
    CaptionCue(0.375, 0.430, "The posture resembles a brooding bird. Here, skeleton and nest preserve one behaviour in the same instant.", "nest"),
    CaptionCue(0.450, 0.505, "Sometimes behaviour survives after passing through the body. Coprolites are fossilised droppings."),
    CaptionCue(0.510, 0.565, "One enormous tyrannosaur coprolite contained roughly thirty to fifty percent fragmented bone.", "coprolite"),
    CaptionCue(0.570, 0.610, "Bite marks add another layer: where teeth entered, how often they returned, and whether the victim healed."),
    CaptionCue(0.620, 0.690, "Healed injuries on Triceratops skulls occur in a pattern consistent with horns striking the frill during combat.", "combat"),
    CaptionCue(0.695, 0.735, "The pattern is evidence across a population—not proof of one exact fight preserved in stone."),
    CaptionCue(0.750, 0.815, "A sediment-filled burrow in Montana held an adult Oryctodromeus and two juveniles in its terminal chamber.", "burrow"),
    CaptionCue(0.820, 0.870, "The tunnel records digging. The family-sized association suggests that the shelter was also used to rear young."),
    CaptionCue(0.885, 0.930, "Microscopic pigment structures reveal that Microraptor carried glossy black, iridescent feathers.", "feathers"),
    CaptionCue(0.935, 0.985, "Colour can suggest camouflage or display. The fossil preserves the signal—but not the intention. Behaviour is reconstructed by testing which story best fits every trace."),
]


# =============================================================================
# Cinematic renderer
# =============================================================================

class FossilBehaviourDocumentary:
    def __init__(self) -> None:
        self.rng = np.random.default_rng(20260806)
        self.grain_rng = np.random.default_rng(9403)
        self.vignette = self._make_vignette(RW, RH)
        self.dust = self._make_dust(1300 if QUICK_MODE else 4200)
        self.cracks = self._make_cracks(90 if QUICK_MODE else 230)
        self.sediment_noise = self.rng.normal(0.0, 1.0, (RH, RW)).astype(np.float32)

    @staticmethod
    def _make_vignette(width: int, height: int) -> np.ndarray:
        yy, xx = np.mgrid[0:height, 0:width]
        nx = (xx - width / 2.0) / (width / 2.0)
        ny = (yy - height / 2.0) / (height / 2.0)
        radius = np.sqrt(nx * nx + ny * ny)
        return np.clip(1.0 - 0.43 * radius**1.72, 0.0, 1.0).astype(np.float32)

    def _make_dust(self, count: int) -> np.ndarray:
        data = np.zeros((count, 7), dtype=np.float32)
        data[:, 0] = self.rng.uniform(0.0, 1.0, count)
        data[:, 1] = self.rng.uniform(0.0, 1.0, count)
        data[:, 2] = self.rng.uniform(0.15, 1.0, count)
        data[:, 3] = self.rng.uniform(0.3, 1.4, count)
        data[:, 4] = self.rng.uniform(0.0, 2.0 * np.pi, count)
        data[:, 5] = self.rng.uniform(-1.0, 1.0, count)
        data[:, 6] = self.rng.uniform(0.25, 1.0, count)
        return data

    def _make_cracks(self, count: int) -> List[List[Tuple[float, float]]]:
        cracks: List[List[Tuple[float, float]]] = []
        for i in range(count):
            x = self.rng.uniform(0.0, 1.0)
            y = self.rng.uniform(0.0, 1.0)
            points = [(x, y)]
            for j in range(self.rng.integers(3, 8)):
                x += self.rng.normal(0.0, 0.018)
                y += abs(self.rng.normal(0.018, 0.012))
                points.append((x, y))
            cracks.append(points)
        return cracks

    def _base_rock(self, fraction: float, t: float) -> Image.Image:
        chapter, _, u = chapter_at(fraction)
        palettes = [
            ((47, 34, 26), (13, 10, 9)),
            ((70, 48, 30), (20, 13, 10)),
            ((77, 43, 25), (23, 12, 8)),
            ((55, 35, 24), (15, 10, 8)),
            ((38, 28, 24), (10, 8, 8)),
            ((61, 39, 25), (17, 11, 8)),
            ((29, 28, 35), (7, 7, 10)),
        ]
        top, bottom = palettes[min(chapter, len(palettes) - 1)]
        yy, xx = np.mgrid[0:RH, 0:RW]
        y = yy / max(RH - 1, 1)
        x = xx / max(RW - 1, 1)
        top_arr = np.array(top, dtype=np.float32)
        bot_arr = np.array(bottom, dtype=np.float32)
        image = top_arr[None, None, :] * (1.0 - y[..., None]) + bot_arr[None, None, :] * y[..., None]
        strata = 4.0 * np.sin(y * 42.0 + 1.2 * np.sin(x * 9.0) + t * 0.005)
        rough = self.sediment_noise * 4.2
        image += strata[..., None] + rough[..., None]
        warm = 4.5 * math.sin(fraction * math.pi)
        image[..., 0] += warm
        image *= self.vignette[..., None]
        return Image.fromarray(np.clip(image, 0, 255).astype(np.uint8), "RGB").convert("RGBA")

    def _draw_cracks_and_strata(self, canvas: Image.Image, t: float, strength: float = 1.0) -> None:
        overlay = Image.new("RGBA", (RW, RH), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        for index, points in enumerate(self.cracks):
            shifted = []
            for x, y in points:
                px = (x + 0.004 * math.sin(t * 0.006 + index)) * RW
                py = y * RH
                shifted.append((px, py))
            draw.line(shifted, fill=(5, 4, 3, int(30 * strength)), width=max(1, int(RENDER_SCALE)))
        for k in range(12):
            y = int(RH * (0.08 + 0.075 * k + 0.004 * math.sin(t * 0.01 + k)))
            draw.line((0, y, RW, y + int(6 * math.sin(k * 1.8))), fill=(155, 116, 76, int(10 * strength)), width=1)
        canvas.alpha_composite(overlay)

    def _draw_dust(self, canvas: Image.Image, t: float, strength: float = 1.0) -> None:
        overlay = Image.new("RGBA", (RW, RH), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        for p in self.dust:
            x0, y0, depth, size, phase, drift, brightness = p
            x = (x0 + drift * t * (0.0005 + 0.0013 * depth) + 0.015 * math.sin(phase + t * 0.018)) % 1.0
            y = (y0 + t * (0.0007 + 0.0015 * depth)) % 1.0
            px, py = int(x * RW), int(y * RH)
            r = max(1, int(size * depth * 1.7 * RENDER_SCALE))
            a = int(70 * brightness * depth * strength)
            draw.ellipse((px - r, py - r, px + r, py + r), fill=(225, 194, 148, a))
        canvas.alpha_composite(overlay.filter(ImageFilter.GaussianBlur(max(0.15, 0.45 * RENDER_SCALE))))

    @staticmethod
    def _three_toed_print(draw: ImageDraw.ImageDraw, cx: float, cy: float, size: float, angle: float, fill) -> None:
        c, s = math.cos(angle), math.sin(angle)
        def rot(x: float, y: float) -> Tuple[float, float]:
            return (cx + c * x - s * y, cy + s * x + c * y)
        heel = [rot(-0.34 * size, 0.25 * size), rot(0.34 * size, 0.25 * size), rot(0.22 * size, -0.22 * size), rot(-0.22 * size, -0.22 * size)]
        draw.polygon(heel, fill=fill)
        for toe_angle, length, width in [(-0.48, 1.0, 0.18), (0.0, 1.18, 0.20), (0.48, 1.0, 0.18)]:
            local = angle + toe_angle
            tc, ts = math.cos(local), math.sin(local)
            base_x = cx + tc * (-0.08 * size)
            base_y = cy + ts * (-0.08 * size)
            tip_x = base_x + tc * length * size
            tip_y = base_y + ts * length * size
            nx, ny = -ts * width * size, tc * width * size
            draw.polygon([(base_x + nx, base_y + ny), (base_x - nx, base_y - ny), (tip_x, tip_y)], fill=fill)

    def _draw_quarry(self, canvas: Image.Image, fraction: float, t: float) -> None:
        chapter, _, u = chapter_at(fraction)
        if chapter != 0:
            return
        overlay = Image.new("RGBA", (RW, RH), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        # Excavation trench and a footprint emerging under a moving brush shadow.
        draw.polygon([(0, RH * 0.62), (RW * 0.35, RH * 0.48), (RW, RH * 0.58), (RW, RH), (0, RH)], fill=(18, 12, 9, 180))
        reveal = smoothstep((u - 0.18) / 0.52)
        self._three_toed_print(draw, RW * 0.53, RH * 0.57, 55 * RENDER_SCALE, -1.42, (7, 5, 4, int(190 * reveal)))
        beam_x = RW * lerp(0.18, 0.82, smootherstep(u))
        draw.polygon([(beam_x - RW * 0.06, 0), (beam_x + RW * 0.04, 0), (beam_x + RW * 0.18, RH), (beam_x - RW * 0.22, RH)], fill=(255, 208, 135, 22))
        # Brush bristles.
        bx = RW * lerp(0.78, 0.47, smoothstep((u - 0.20) / 0.55))
        by = RH * lerp(0.35, 0.56, smoothstep((u - 0.20) / 0.55))
        draw.line((bx - 50, by - 110, bx + 25, by), fill=(27, 19, 14, 210), width=max(3, int(8 * RENDER_SCALE)))
        for k in range(16):
            off = (k - 7.5) * 3.2 * RENDER_SCALE
            draw.line((bx + off, by, bx + off * 1.6, by + 32 * RENDER_SCALE), fill=(115, 79, 48, 155), width=max(1, int(RENDER_SCALE)))
        canvas.alpha_composite(overlay.filter(ImageFilter.GaussianBlur(max(0.1, 0.35 * RENDER_SCALE))))

    def _draw_trackway(self, canvas: Image.Image, fraction: float, t: float) -> None:
        chapter, _, u = chapter_at(fraction)
        if chapter != 1:
            return
        overlay = Image.new("RGBA", (RW, RH), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        # Low camera gliding along a trackway. Prints grow toward the foreground.
        for i in range(12):
            z = ((i / 11.0 + u * 0.42) % 1.15)
            perspective = z ** 1.75
            y = RH * lerp(0.20, 1.08, perspective)
            x = RW * (0.50 + (0.10 if i % 2 == 0 else -0.10) * perspective + 0.03 * math.sin(i * 1.7))
            size = lerp(7, 70, perspective) * RENDER_SCALE
            alpha = int(lerp(35, 220, perspective))
            self._three_toed_print(draw, x, y, size, -1.55 + 0.08 * math.sin(i), (8, 5, 4, alpha))
            # Rim light in the compressed mud.
            self._three_toed_print(draw, x - 1.5, y - 2.0, size * 0.94, -1.55 + 0.08 * math.sin(i), (128, 86, 48, int(alpha * 0.15)))
        # Distant animal silhouette, kept deliberately indistinct.
        animal_a = smoothstep((u - 0.42) / 0.25) * (1.0 - smoothstep((u - 0.82) / 0.15))
        ax = RW * (0.64 + 0.03 * math.sin(t * 0.025))
        ay = RH * 0.23
        body = 28 * RENDER_SCALE
        draw.ellipse((ax - body, ay - body * 0.45, ax + body, ay + body * 0.45), fill=(5, 4, 4, int(130 * animal_a)))
        draw.line((ax + body * 0.7, ay, ax + body * 2.4, ay - body * 0.6), fill=(5, 4, 4, int(130 * animal_a)), width=max(2, int(5 * RENDER_SCALE)))
        draw.line((ax - body * 0.4, ay + body * 0.2, ax - body * 0.55, ay + body * 1.55), fill=(5, 4, 4, int(130 * animal_a)), width=max(2, int(5 * RENDER_SCALE)))
        draw.line((ax + body * 0.25, ay + body * 0.2, ax + body * 0.1, ay + body * 1.55), fill=(5, 4, 4, int(130 * animal_a)), width=max(2, int(5 * RENDER_SCALE)))
        canvas.alpha_composite(overlay.filter(ImageFilter.GaussianBlur(max(0.1, 0.5 * RENDER_SCALE))))

    def _draw_nest(self, canvas: Image.Image, fraction: float, t: float) -> None:
        chapter, _, u = chapter_at(fraction)
        if chapter != 2:
            return
        overlay = Image.new("RGBA", (RW, RH), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        cx, cy = RW * 0.52, RH * 0.64
        ring_r = min(RW, RH) * 0.25
        for i in range(18):
            a = i * 2.0 * math.pi / 18.0 + 0.09 * math.sin(t * 0.01)
            x = cx + math.cos(a) * ring_r
            y = cy + math.sin(a) * ring_r * 0.44
            egg_w = 13 * RENDER_SCALE
            egg_h = 34 * RENDER_SCALE
            draw.ellipse((x - egg_w, y - egg_h, x + egg_w, y + egg_h), fill=(172, 139, 95, 210), outline=(230, 195, 137, 85), width=max(1, int(RENDER_SCALE)))
        # Brooding oviraptorid silhouette: body, neck, head, tail and feathered arms.
        reveal = smoothstep((u - 0.18) / 0.38)
        alpha = int(210 * reveal)
        body_w, body_h = 112 * RENDER_SCALE, 54 * RENDER_SCALE
        draw.ellipse((cx - body_w, cy - 78 * RENDER_SCALE - body_h, cx + body_w, cy - 78 * RENDER_SCALE + body_h), fill=(10, 7, 6, alpha))
        draw.line((cx + body_w * 0.60, cy - 82 * RENDER_SCALE, cx + body_w * 1.15, cy - 160 * RENDER_SCALE), fill=(10, 7, 6, alpha), width=max(3, int(22 * RENDER_SCALE)))
        draw.ellipse((cx + body_w * 1.00, cy - 176 * RENDER_SCALE, cx + body_w * 1.30, cy - 146 * RENDER_SCALE), fill=(10, 7, 6, alpha))
        draw.line((cx - body_w * 0.8, cy - 78 * RENDER_SCALE, cx - body_w * 2.0, cy - 48 * RENDER_SCALE), fill=(10, 7, 6, alpha), width=max(3, int(18 * RENDER_SCALE)))
        for side in (-1, 1):
            base_x = cx + side * body_w * 0.45
            base_y = cy - 75 * RENDER_SCALE
            for k in range(9):
                ang = side * (0.55 + 0.08 * k)
                length = (70 + 7 * k) * RENDER_SCALE
                ex = base_x + math.sin(ang) * length
                ey = base_y + math.cos(ang) * length * 0.58
                draw.line((base_x, base_y, ex, ey), fill=(24, 16, 12, int(alpha * 0.86)), width=max(1, int(5 * RENDER_SCALE)))
        # Windblown sand, creating a preserved-in-an-instant feeling.
        for k in range(90 if QUICK_MODE else 220):
            phase = deterministic_unit(str(k), "nest") * 2 * math.pi
            x = (deterministic_unit(str(k), "nx") + t * 0.0024) % 1.0 * RW
            y = RH * (0.35 + 0.55 * deterministic_unit(str(k), "ny")) + math.sin(phase + t * 0.08) * 5
            draw.line((x, y, x + 28 * RENDER_SCALE, y - 3), fill=(210, 164, 105, 28), width=1)
        canvas.alpha_composite(overlay.filter(ImageFilter.GaussianBlur(max(0.1, 0.45 * RENDER_SCALE))))

    def _draw_coprolite_and_bites(self, canvas: Image.Image, fraction: float, t: float) -> None:
        chapter, _, u = chapter_at(fraction)
        if chapter != 3:
            return
        overlay = Image.new("RGBA", (RW, RH), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        split = smoothstep((u - 0.48) / 0.18)
        # Macro cross-section of a coprolite.
        cop_a = 1.0 - split
        cx, cy = RW * 0.50, RH * 0.54
        rx, ry = RW * 0.24, RH * 0.20
        outline_pts = []
        for i in range(120):
            a = 2 * math.pi * i / 120
            jitter = 1.0 + 0.06 * math.sin(7 * a + 0.3) + 0.04 * math.sin(13 * a)
            outline_pts.append((cx + math.cos(a) * rx * jitter, cy + math.sin(a) * ry * jitter))
        draw.polygon(outline_pts, fill=(41, 28, 20, int(245 * cop_a)), outline=(120, 83, 48, int(130 * cop_a)))
        for k in range(85 if QUICK_MODE else 210):
            a = deterministic_unit(str(k), "ca") * 2 * math.pi
            r = math.sqrt(deterministic_unit(str(k), "cr"))
            x = cx + math.cos(a) * rx * 0.82 * r
            y = cy + math.sin(a) * ry * 0.78 * r
            if k % 3 == 0:
                length = (8 + 26 * deterministic_unit(str(k), "cl")) * RENDER_SCALE
                ang = deterministic_unit(str(k), "ct") * math.pi
                dx, dy = math.cos(ang) * length, math.sin(ang) * length
                draw.line((x - dx, y - dy, x + dx, y + dy), fill=(215, 190, 145, int(170 * cop_a)), width=max(1, int(3 * RENDER_SCALE)))
            else:
                rr = max(1, int((1 + 4 * deterministic_unit(str(k), "cs")) * RENDER_SCALE))
                draw.ellipse((x - rr, y - rr, x + rr, y + rr), fill=(105, 69, 42, int(125 * cop_a)))
        # Bite-marked bone, replacing the coprolite during the second half.
        bone_a = split
        bx0, by0 = RW * 0.19, RH * 0.59
        bx1, by1 = RW * 0.82, RH * 0.48
        width = 40 * RENDER_SCALE
        dx, dy = bx1 - bx0, by1 - by0
        length = max(math.hypot(dx, dy), 1.0)
        nx, ny = -dy / length * width, dx / length * width
        draw.polygon([(bx0 + nx, by0 + ny), (bx1 + nx, by1 + ny), (bx1 - nx, by1 - ny), (bx0 - nx, by0 - ny)], fill=(190, 166, 121, int(230 * bone_a)))
        for ex, ey in [(bx0, by0), (bx1, by1)]:
            draw.ellipse((ex - width * 1.45, ey - width * 1.2, ex + width * 1.45, ey + width * 1.2), fill=(204, 181, 134, int(235 * bone_a)))
        for k in range(11):
            p = 0.28 + 0.045 * k
            x, y = bx0 + dx * p, by0 + dy * p
            tooth = 7 + 6 * (k % 3)
            draw.arc((x - tooth, y - tooth, x + tooth, y + tooth), 10, 170, fill=(60, 36, 22, int(210 * bone_a)), width=max(1, int(3 * RENDER_SCALE)))
        # Predator shadow passes over the evidence.
        shadow_x = RW * lerp(-0.25, 1.25, u)
        draw.ellipse((shadow_x - 180 * RENDER_SCALE, RH * 0.12, shadow_x + 180 * RENDER_SCALE, RH * 0.72), fill=(0, 0, 0, 22))
        canvas.alpha_composite(overlay.filter(ImageFilter.GaussianBlur(max(0.1, 0.55 * RENDER_SCALE))))

    def _draw_triceratops(self, canvas: Image.Image, fraction: float, t: float) -> None:
        chapter, _, u = chapter_at(fraction)
        if chapter != 4:
            return
        overlay = Image.new("RGBA", (RW, RH), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        cx, cy = RW * 0.52, RH * 0.53
        frill_r = 132 * RENDER_SCALE
        # Frontal skull silhouette.
        draw.ellipse((cx - frill_r, cy - frill_r, cx + frill_r, cy + frill_r * 0.82), fill=(20, 14, 11, 230), outline=(157, 116, 76, 105), width=max(1, int(2 * RENDER_SCALE)))
        draw.polygon([(cx - 58 * RENDER_SCALE, cy - 30 * RENDER_SCALE), (cx + 58 * RENDER_SCALE, cy - 30 * RENDER_SCALE), (cx + 35 * RENDER_SCALE, cy + 112 * RENDER_SCALE), (cx, cy + 155 * RENDER_SCALE), (cx - 35 * RENDER_SCALE, cy + 112 * RENDER_SCALE)], fill=(35, 24, 17, 240))
        # Brow horns.
        for side in (-1, 1):
            base_x = cx + side * 58 * RENDER_SCALE
            base_y = cy - 25 * RENDER_SCALE
            tip_x = cx + side * 175 * RENDER_SCALE
            tip_y = cy - 145 * RENDER_SCALE
            draw.polygon([(base_x - side * 13 * RENDER_SCALE, base_y + 12 * RENDER_SCALE), (base_x + side * 20 * RENDER_SCALE, base_y - 8 * RENDER_SCALE), (tip_x, tip_y)], fill=(172, 143, 101, 235))
        draw.polygon([(cx - 13 * RENDER_SCALE, cy + 8 * RENDER_SCALE), (cx + 13 * RENDER_SCALE, cy + 8 * RENDER_SCALE), (cx, cy - 68 * RENDER_SCALE)], fill=(169, 140, 98, 230))
        # Healed lesion ridges across frill.
        lesion_points = [(-0.64, -0.18), (-0.50, 0.18), (0.55, -0.08), (0.68, 0.19), (0.30, -0.55)]
        pulse = 0.55 + 0.45 * math.sin(t * 0.15)
        for i, (fx, fy) in enumerate(lesion_points):
            x, y = cx + fx * frill_r, cy + fy * frill_r
            rr = (7 + 4 * (i % 2)) * RENDER_SCALE
            draw.arc((x - rr * 2, y - rr, x + rr * 2, y + rr), 190, 350, fill=(210, 92, 48, int(150 + 70 * pulse)), width=max(1, int(4 * RENDER_SCALE)))
        # Ghosted opposing skulls suggest contact without claiming a preserved event.
        ghost = smoothstep((u - 0.48) / 0.28)
        for side in (-1, 1):
            gx = RW * (0.10 if side < 0 else 0.90) + side * (-1) * smoothstep((u - 0.40) / 0.35) * RW * 0.12
            gy = RH * 0.55
            draw.ellipse((gx - 58 * RENDER_SCALE, gy - 48 * RENDER_SCALE, gx + 58 * RENDER_SCALE, gy + 48 * RENDER_SCALE), outline=(166, 123, 83, int(70 * ghost)), width=max(1, int(3 * RENDER_SCALE)))
            draw.line((gx - side * 30 * RENDER_SCALE, gy - 20 * RENDER_SCALE, gx - side * 115 * RENDER_SCALE, gy - 62 * RENDER_SCALE), fill=(166, 123, 83, int(70 * ghost)), width=max(1, int(4 * RENDER_SCALE)))
        canvas.alpha_composite(overlay.filter(ImageFilter.GaussianBlur(max(0.1, 0.45 * RENDER_SCALE))))

    @staticmethod
    def _draw_small_dinosaur(draw: ImageDraw.ImageDraw, x: float, y: float, scale: float, direction: float, fill) -> None:
        draw.ellipse((x - 30 * scale, y - 12 * scale, x + 30 * scale, y + 12 * scale), fill=fill)
        draw.line((x + direction * 22 * scale, y - 2 * scale, x + direction * 46 * scale, y - 22 * scale), fill=fill, width=max(2, int(10 * scale)))
        hx0 = x + direction * 40 * scale
        hx1 = x + direction * 55 * scale
        draw.ellipse((min(hx0, hx1), y - 30 * scale, max(hx0, hx1), y - 17 * scale), fill=fill)
        draw.line((x - direction * 26 * scale, y, x - direction * 80 * scale, y + 8 * scale), fill=fill, width=max(2, int(8 * scale)))
        draw.line((x - 8 * scale, y + 8 * scale, x - 16 * scale, y + 42 * scale), fill=fill, width=max(2, int(7 * scale)))
        draw.line((x + 11 * scale, y + 8 * scale, x + 18 * scale, y + 42 * scale), fill=fill, width=max(2, int(7 * scale)))

    def _draw_burrow(self, canvas: Image.Image, fraction: float, t: float) -> None:
        chapter, _, u = chapter_at(fraction)
        if chapter != 5:
            return
        overlay = Image.new("RGBA", (RW, RH), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        # Cross-section of ground and a descending tunnel.
        draw.rectangle((0, 0, RW, RH), fill=(72, 45, 27, 90))
        surface_y = RH * 0.20
        draw.line((0, surface_y, RW, surface_y), fill=(204, 153, 91, 90), width=max(1, int(2 * RENDER_SCALE)))
        tunnel_pts = []
        for i in range(65):
            p = i / 64.0
            x = RW * lerp(0.16, 0.62, p)
            y = surface_y + RH * (0.06 + 0.50 * p + 0.05 * math.sin(p * 5.2))
            tunnel_pts.append((x, y))
        draw.line(tunnel_pts, fill=(8, 6, 5, 245), width=max(12, int(62 * RENDER_SCALE)))
        chamber_x, chamber_y = RW * 0.70, RH * 0.72
        draw.ellipse((chamber_x - 150 * RENDER_SCALE, chamber_y - 92 * RENDER_SCALE, chamber_x + 150 * RENDER_SCALE, chamber_y + 92 * RENDER_SCALE), fill=(8, 6, 5, 248))
        # Adult and two juveniles in the terminal chamber.
        reveal = smoothstep((u - 0.22) / 0.35)
        self._draw_small_dinosaur(draw, chamber_x + 12 * RENDER_SCALE, chamber_y + 4 * RENDER_SCALE, 1.15 * RENDER_SCALE, 1, (88, 68, 48, int(215 * reveal)))
        self._draw_small_dinosaur(draw, chamber_x - 72 * RENDER_SCALE, chamber_y + 32 * RENDER_SCALE, 0.48 * RENDER_SCALE, -1, (135, 102, 65, int(220 * reveal)))
        self._draw_small_dinosaur(draw, chamber_x + 78 * RENDER_SCALE, chamber_y + 42 * RENDER_SCALE, 0.43 * RENDER_SCALE, 1, (135, 102, 65, int(220 * reveal)))
        # Fine soil falls through the chamber.
        for k in range(75 if QUICK_MODE else 190):
            x = RW * deterministic_unit(str(k), "bx")
            y = (RH * deterministic_unit(str(k), "by") + t * (0.7 + 1.5 * deterministic_unit(str(k), "bs"))) % RH
            draw.line((x, y, x + 2, y + 8), fill=(215, 164, 104, 22), width=1)
        canvas.alpha_composite(overlay.filter(ImageFilter.GaussianBlur(max(0.1, 0.6 * RENDER_SCALE))))

    def _draw_feather(self, canvas: Image.Image, fraction: float, t: float) -> None:
        chapter, _, u = chapter_at(fraction)
        if chapter != 6:
            return
        overlay = Image.new("RGBA", (RW, RH), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        cx, cy = RW * 0.50, RH * 0.56
        length = RH * 0.68
        angle = -0.32 + 0.025 * math.sin(t * 0.013)
        c, s = math.cos(angle), math.sin(angle)
        def point(along: float, across: float) -> Tuple[float, float]:
            return (cx + c * along - s * across, cy + s * along + c * across)
        start = point(-length * 0.5, 0)
        end = point(length * 0.5, 0)
        draw.line((*start, *end), fill=(188, 170, 155, 220), width=max(2, int(7 * RENDER_SCALE)))
        shimmer = 0.5 + 0.5 * math.sin(t * 0.10)
        for i in range(90):
            p = i / 89.0
            along = lerp(-length * 0.44, length * 0.46, p)
            span = math.sin(math.pi * p) ** 0.65 * length * 0.20
            for side in (-1, 1):
                base = point(along, 0)
                tip = point(along + side * length * 0.015, side * span)
                hue = (p * 2.2 + shimmer + (0.2 if side > 0 else 0.0)) % 1.0
                if hue < 0.33:
                    color = (32, 70, 94)
                elif hue < 0.66:
                    color = (62, 30, 92)
                else:
                    color = (12, 76, 70)
                draw.line((*base, *tip), fill=(*color, 185), width=max(1, int(2 * RENDER_SCALE)))
        # Moving grazing light reveals structural colour.
        light_x = RW * lerp(0.12, 0.88, 0.5 + 0.5 * math.sin(t * 0.025))
        draw.polygon([(light_x - RW * 0.06, 0), (light_x + RW * 0.04, 0), (light_x + RW * 0.18, RH), (light_x - RW * 0.14, RH)], fill=(105, 190, 220, 18))
        # A dissolving fossil outline sits behind the colour reconstruction.
        fossil_a = int(110 * (1.0 - smoothstep((u - 0.62) / 0.25)))
        draw.ellipse((RW * 0.16, RH * 0.22, RW * 0.84, RH * 0.87), outline=(205, 183, 145, fossil_a), width=max(1, int(2 * RENDER_SCALE)))
        canvas.alpha_composite(overlay.filter(ImageFilter.GaussianBlur(max(0.1, 0.45 * RENDER_SCALE))))

    def _draw_documentary_text(self, canvas: Image.Image, fraction: float, t: float) -> None:
        intro_length = min(20.0, DURATION * 0.095)
        if t < intro_length:
            fade_in = smoothstep(t / max(3.0, intro_length * 0.22))
            fade_out = 1.0 - smoothstep((t - intro_length * 0.72) / max(intro_length * 0.28, 1e-6))
            alpha = int(255 * fade_in * fade_out)
            draw_text(canvas, str(CONFIG["title"]), (WIDTH * 0.5, HEIGHT * 0.42), 88, (244, 235, 219, alpha), True, True, "mm", 2)
            draw_text(canvas, str(CONFIG["subtitle"]), (WIDTH * 0.5, HEIGHT * 0.52), 29, (226, 164, 92, int(alpha * 0.94)), True, False, "mm", 1)
            draw_text(canvas, "A cinematic investigation of traces, nests, scars and colour", (WIDTH * 0.5, HEIGHT * 0.61), 18, (222, 207, 188, int(alpha * 0.76)), False, False, "mm", 0)

        chapter_index, chapter_title, chapter_u = chapter_at(fraction)
        chapter_alpha = int(205 * (1.0 - smoothstep((chapter_u - 0.02) / 0.14)))
        if chapter_index > 0 and chapter_alpha > 0:
            draw_text(canvas, f"0{chapter_index + 1}", (74 * SCALE, 70 * SCALE), 15, (221, 151, 80, chapter_alpha), True, False, "la", 0)
            draw_text(canvas, chapter_title.upper(), (74 * SCALE, 103 * SCALE), 22, (237, 225, 207, chapter_alpha), True, False, "la", 1)

        for cue in NARRATION:
            if cue.start_fraction <= fraction <= cue.end_fraction:
                local = (fraction - cue.start_fraction) / max(cue.end_fraction - cue.start_fraction, 1e-8)
                alpha = smoothstep(local / 0.14) * (1.0 - smoothstep((local - 0.82) / 0.18))
                x = int(95 * SCALE)
                y = int(HEIGHT * 0.79)
                max_width = int(WIDTH * 0.72)
                panel = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
                pd = ImageDraw.Draw(panel)
                pd.rounded_rectangle((x - 28 * SCALE, y - 25 * SCALE, x + max_width + 25 * SCALE, y + 145 * SCALE), radius=int(22 * SCALE), fill=(3, 3, 3, int(112 * alpha)))
                draw_wrapped_text(panel, cue.text, (x, y), max_width, 24, (244, 238, 228, int(245 * alpha)), False, 9)
                if cue.evidence_key:
                    record = EVIDENCE[cue.evidence_key]
                    draw_text(panel, record.confidence, (x, y - 18 * SCALE), 13, (224, 151, 77, int(235 * alpha)), True, False, "ls", 0)
                    label = record.specimen.upper()
                    if record.numeric_value is not None:
                        if cue.evidence_key == "coprolite":
                            label += "  •  30–50% BONE FRAGMENTS"
                        elif cue.evidence_key == "burrow":
                            label += "  •  1 ADULT + 2 JUVENILES"
                    draw_text(panel, label, (x + max_width, y + 119 * SCALE), 13, (197, 177, 151, int(200 * alpha)), True, False, "rs", 0)
                canvas.alpha_composite(panel)
                break

        # A subtle evidence-mode label, not a graph or dashboard.
        evidence_map = {1: "TRACE FOSSIL", 2: "BODY + NEST ASSOCIATION", 3: "FEEDING TRACE", 4: "HEALED PATHOLOGY", 5: "BURROW + SKELETONS", 6: "MICROSTRUCTURE"}
        if chapter_index in evidence_map:
            draw_text(canvas, evidence_map[chapter_index], (WIDTH - 72 * SCALE, 64 * SCALE), 12, (205, 176, 137, 145), True, False, "ra", 0)

    def _finish(self, canvas: Image.Image, t: float) -> np.ndarray:
        image = canvas.resize(OUT_SIZE, Image.Resampling.LANCZOS).convert("RGB")
        array = np.asarray(image, dtype=np.float32)
        grain = self.grain_rng.normal(0.0, 2.7 if QUICK_MODE else 2.1, array.shape[:2]).astype(np.float32)
        array += grain[..., None]
        # Letterbox for a documentary-cinema frame.
        bar = int(HEIGHT * 0.043)
        array[:bar, :, :] *= 0.04
        array[-bar:, :, :] *= 0.04
        image = Image.fromarray(np.clip(array, 0, 255).astype(np.uint8), "RGB")
        image = ImageEnhance.Contrast(image).enhance(1.06)
        return np.asarray(image)

    def render_frame(self, t: float) -> np.ndarray:
        fraction = clamp(t / max(DURATION, 1e-8))
        canvas = self._base_rock(fraction, t)
        self._draw_cracks_and_strata(canvas, t, 1.0)
        self._draw_quarry(canvas, fraction, t)
        self._draw_trackway(canvas, fraction, t)
        self._draw_nest(canvas, fraction, t)
        self._draw_coprolite_and_bites(canvas, fraction, t)
        self._draw_triceratops(canvas, fraction, t)
        self._draw_burrow(canvas, fraction, t)
        self._draw_feather(canvas, fraction, t)
        self._draw_dust(canvas, t, 0.85)
        full = canvas.resize(OUT_SIZE, Image.Resampling.LANCZOS)
        self._draw_documentary_text(full, fraction, t)
        return self._finish(full.resize((RW, RH), Image.Resampling.LANCZOS), t)


# =============================================================================
# Procedural soundtrack
# =============================================================================

@dataclass(frozen=True)
class ToneEvent:
    start: float
    duration: float
    frequency: float
    amplitude: float
    pan: float


def render_ambient_audio(path: Path, duration: float, sample_rate: int = 48_000) -> Path:
    rng = np.random.default_rng(731)
    events: List[ToneEvent] = []
    scale = [36.7, 43.65, 55.0, 65.41, 73.42, 87.31, 110.0]
    for i in range(max(12, int(duration / 10))):
        start = i * duration / max(12, int(duration / 10)) + rng.uniform(-1.2, 1.2)
        events.append(ToneEvent(max(0.0, start), rng.uniform(6.0, 15.0), float(rng.choice(scale)), rng.uniform(0.012, 0.032), rng.uniform(-0.7, 0.7)))

    path.parent.mkdir(parents=True, exist_ok=True)
    total = int(round(duration * sample_rate))
    block = 65536
    fade_seconds = min(5.0, duration * 0.08)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        for start_idx in tqdm(range(0, total, block), desc="Rendering ambient soundtrack"):
            count = min(block, total - start_idx)
            absolute_t = (start_idx + np.arange(count, dtype=np.float64)) / sample_rate
            fraction = absolute_t / max(duration, 1e-8)
            left = np.zeros(count, dtype=np.float64)
            right = np.zeros(count, dtype=np.float64)

            # Subterranean air and low bowed-stone resonance.
            drone = (
                0.026 * np.sin(2.0 * np.pi * 36.7 * absolute_t)
                + 0.018 * np.sin(2.0 * np.pi * 55.0 * absolute_t + 1.1)
                + 0.010 * np.sin(2.0 * np.pi * 73.42 * absolute_t + 2.0)
            )
            left += drone
            right += np.roll(drone, int(0.007 * sample_rate))

            noise = rng.normal(0.0, 1.0, count).astype(np.float64)
            window = min(1800, max(8, count // 4))
            padded = np.pad(noise, (window // 2, window - window // 2), mode="edge")
            cumulative = np.cumsum(np.concatenate(([0.0], padded)), dtype=np.float64)
            filtered = ((cumulative[window:] - cumulative[:-window]) / window)[:count]
            left += filtered * 0.055
            right += np.roll(filtered, int(0.019 * sample_rate)) * 0.055

            # Sparse excavation taps and brush-like noise early in the film.
            early = absolute_t < duration * 0.14
            tap_phase = np.mod(absolute_t, 3.8)
            active_tap = early & (tap_phase < 0.18)
            x = tap_phase[active_tap]
            tap = np.sin(2.0 * np.pi * (220.0 - 80.0 * x) * x) * np.exp(-x * 25.0) * 0.025
            left[active_tap] += tap
            right[active_tap] += tap * 0.7

            for event in events:
                local = absolute_t - event.start
                active = (local >= 0.0) & (local <= event.duration)
                if not np.any(active):
                    continue
                x = local[active]
                env = np.clip(x / 0.7, 0.0, 1.0) * np.exp(-x / max(event.duration * 0.52, 1.0))
                tone = (
                    np.sin(2.0 * np.pi * event.frequency * x)
                    + 0.27 * np.sin(2.0 * np.pi * event.frequency * 2.01 * x + 0.4)
                    + 0.10 * np.sin(2.0 * np.pi * event.frequency * 3.98 * x + 1.2)
                ) * env * event.amplitude
                left[active] += tone * math.sqrt((1.0 - event.pan) * 0.5)
                right[active] += tone * math.sqrt((1.0 + event.pan) * 0.5)

            fade_in = np.clip(absolute_t / max(fade_seconds, 1e-6), 0.0, 1.0)
            fade_out = np.clip((duration - absolute_t) / max(fade_seconds, 1e-6), 0.0, 1.0)
            master = np.minimum(fade_in, fade_out)
            master *= 0.78 + 0.22 * np.sin(np.pi * np.clip(fraction, 0.0, 1.0))
            stereo = np.column_stack((left, right)) * master[:, None]
            stereo = np.tanh(stereo * 1.55) * 0.82
            pcm = np.clip(stereo * 32767.0, -32768, 32767).astype("<i2")
            wav.writeframes(pcm.tobytes())
    return path


# =============================================================================
# Metadata, previews and video
# =============================================================================

def write_subtitles(path: Path) -> Path:
    lines: List[str] = []
    for index, cue in enumerate(NARRATION, 1):
        lines.extend([
            str(index),
            f"{format_srt_time(cue.start_fraction * DURATION)} --> {format_srt_time(cue.end_fraction * DURATION)}",
            cue.text,
            "",
        ])
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_metadata() -> Tuple[Path, Path]:
    description = f"""A cinematic investigation into how palaeontologists reconstruct behaviour from fossils.

This {DURATION / 60.0:.1f}-minute film moves from footprint trackways to nests, coprolites, bite marks, healed injuries, burrows and fossil colour. It contains no graphs or dashboard-style graphics. Published evidence is presented through atmospheric procedural scenes and restrained captions.

Evidence highlighted:
- Trackways preserve stride, foot placement and direction, constraining gait and speed.
- An oviraptorid was preserved in a bird-like brooding posture over its eggs.
- A giant tyrannosaur coprolite contained approximately 30–50% bone fragments.
- Population-level patterns of healed Triceratops frill injuries are consistent with horn combat.
- An Oryctodromeus burrow preserved one adult and two juveniles.
- Microraptor melanosomes indicate glossy black, iridescent plumage.

Scientific honesty:
The measurements and published interpretations are real. The landscapes, animals, colours, poses, lighting and camera path are artistic visualisations. The film separates direct observations from behavioural inference and does not claim that every fossil trace has one unique explanation.
"""
    metadata_path = OUTPUT_ROOT / "youtube_title_and_description.txt"
    metadata_path.write_text(f"TITLE\n{CONFIG['youtube_title']}\n\nDESCRIPTION\n{description}", encoding="utf-8")
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
            "contains_graphs": False,
        },
        "evidence_records": {key: asdict(value) for key, value in EVIDENCE.items()},
        "primary_references": [
            "https://pmc.ncbi.nlm.nih.gov/articles/PMC11732409/",
            "https://doi.org/10.1038/378774a0",
            "https://doi.org/10.1038/31461",
            "https://doi.org/10.1371/journal.pone.0004252",
            "https://doi.org/10.1098/rspb.2006.0443",
            "https://doi.org/10.1126/science.1213780",
        ],
        "scientific_note": "Published evidence combined with artistic procedural cinematics. Direct observation and inference are labelled separately.",
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
    sheet = Image.new("RGB", (thumb_w * 3 + margin * 4, thumb_h * 2 + margin * 3), (8, 6, 5))
    for index, image in enumerate(images[:6]):
        thumb = image.resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)
        x = margin + (index % 3) * (thumb_w + margin)
        y = margin + (index // 3) * (thumb_h + margin)
        sheet.paste(thumb, (x, y))
    path = PREVIEW_DIR / "fossils_behaviour_contact_sheet.jpg"
    sheet.save(path, quality=93)
    return path


def render_previews(scene: FossilBehaviourDocumentary) -> List[Path]:
    fractions = [0.20, 0.355, 0.515, 0.675, 0.81, 0.93]
    paths: List[Path] = []
    for index, fraction in enumerate(tqdm(fractions, desc="Preview frames"), 1):
        t = fraction * DURATION
        frame = scene.render_frame(t)
        path = PREVIEW_DIR / f"preview_{index:02d}_{t:07.2f}s.png"
        Image.fromarray(frame).save(path)
        paths.append(path)
    return paths


def render_video(scene: FossilBehaviourDocumentary) -> Path:
    basename = str(CONFIG["output_basename"])
    raw_path = OUTPUT_ROOT / f"{basename}_silent.mp4"
    final_path = OUTPUT_ROOT / f"{basename}_final.mp4"
    audio_path = AUDIO_DIR / f"{basename}_ambient.wav"
    srt_path = OUTPUT_ROOT / f"{basename}_narration.srt"

    if bool(CONFIG["write_subtitle_sidecar"]):
        write_subtitles(srt_path)

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
        for t in tqdm(times, desc="Rendering fossil-behaviour documentary"):
            writer.append_data(scene.render_frame(float(t)))

    ffmpeg = find_ffmpeg()
    requested_audio = CONFIG.get("audio_path")
    if requested_audio and Path(str(requested_audio)).exists():
        soundtrack = Path(str(requested_audio))
    else:
        soundtrack = render_ambient_audio(audio_path, DURATION, int(CONFIG["audio_sample_rate"]))

    if ffmpeg:
        run_ffmpeg([
            ffmpeg, "-y",
            "-i", str(raw_path),
            "-i", str(soundtrack),
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "copy", "-c:a", "aac", "-b:a", "256k",
            "-shortest", "-movflags", "+faststart",
            str(final_path),
        ])
    else:
        print("ffmpeg was not found; copying the silent render as the final file.")
        shutil.copyfile(raw_path, final_path)
    return final_path


def main() -> None:
    print("Starting FOSSILS: BEHAVIOUR WRITTEN IN STONE")
    print("Quick mode:", QUICK_MODE)
    print("Preview only:", PREVIEW_ONLY)
    print(f"Duration: {DURATION:.1f} seconds")
    print("Graphs included: no")
    print("Evidence chapters:", ", ".join(record.specimen for record in EVIDENCE.values()))
    metadata_path, manifest_path = write_metadata()
    print("YouTube metadata:", metadata_path.resolve())
    print("Render manifest:", manifest_path.resolve())
    scene = FossilBehaviourDocumentary()
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
