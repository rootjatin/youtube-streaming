#!/usr/bin/env python3
"""
Live Satellite Tracker over an Equal Earth projected SVG map.

What this script does
- Renders an Equal Earth-projected SVG world map to a raster image (RGBA).
- Loads TLEs (satellite orbital elements) via Skyfield (download + cache) OR from a local file.
- Computes satellite subpoints (lat/lon/alt) periodically.
- Projects lat/lon to Equal Earth X/Y coordinates, maps those to pixels, and draws markers + labels.
- Optionally previews in an OpenCV window.
- Optionally records to MP4 or streams to RTMP using FFmpeg by piping raw frames.

Important assumptions
- The input SVG uses the Equal Earth projection and includes a `viewBox`.
- Coordinate mapping relies on that SVG `viewBox` and the Equal Earth math below.
"""

import argparse
import io
import math
import time
import shutil
import subprocess
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import cairosvg
from xml.etree import ElementTree as ET

from skyfield.api import EarthSatellite, Loader


# ============================================================
# Equal Earth projection (matches your SVG map projection)
# ============================================================
# Coefficients for Equal Earth projection (standard values).
A1 = 1.340264
A2 = -0.081106
A3 = 0.000893
A4 = 0.003796
M = math.sqrt(3) / 2.0

# The math below computes a scaling constant so that the projected x-range matches degrees in your SVG's
# coordinate system. This keeps projected coordinates consistent with an Equal Earth world map.
_x_equator_rad = (2.0 * math.sqrt(3) * math.pi) / (3.0 * A1)
_x_equator_deg = _x_equator_rad * 180.0 / math.pi
EE_SCALE = 180.0 / _x_equator_deg


def equal_earth_project(lat_deg: float, lon_deg: float) -> Tuple[float, float]:
    """
    Convert (lat, lon) in degrees to Equal Earth projected coordinates (x_vb, y_vb).

    Returns:
        x_vb, y_vb: projected coordinates in a "viewBox-like" coordinate space (not pixels yet).

    Notes:
    - Equal Earth projection produces x/y in radians-based units.
    - We convert to degrees-like units and apply EE_SCALE so the result aligns with typical map SVGs.
    """
    phi = math.radians(lat_deg)
    lam = math.radians(lon_deg)

    # Equal Earth uses an auxiliary latitude theta.
    theta = math.asin(M * math.sin(phi))
    t2 = theta * theta
    t6 = t2 * t2 * t2
    t8 = t6 * t2

    # Denominator for x
    d = (A1 + 3.0 * A2 * t2 + 7.0 * A3 * t6 + 9.0 * A4 * t8)

    # Projected x/y in Equal Earth
    x = (2.0 * math.sqrt(3) * lam * math.cos(theta)) / (3.0 * d)
    y = (A1 * theta + A2 * theta**3 + A3 * theta**7 + A4 * theta**9)

    # Convert to a scaled "degrees-like" coordinate space.
    x_vb = (x * 180.0 / math.pi) * EE_SCALE
    y_vb = (y * 180.0 / math.pi) * EE_SCALE
    return x_vb, y_vb


# ============================================================
# SVG helpers
# ============================================================
@dataclass
class SvgViewBox:
    """Simple container for an SVG viewBox."""
    minx: float
    miny: float
    width: float
    height: float


def parse_svg_viewbox(svg_path: str) -> SvgViewBox:
    """
    Read the root <svg> viewBox attribute and parse it into numbers.

    The viewBox is crucial because we map Equal Earth x/y into that coordinate space,
    then into pixels.
    """
    root = ET.parse(svg_path).getroot()
    vb = root.attrib.get("viewBox")
    if not vb:
        raise ValueError("SVG is missing viewBox attribute (needed for mapping).")
    minx, miny, w, h = map(float, vb.split())
    return SvgViewBox(minx=minx, miny=miny, width=w, height=h)


def render_svg_to_rgba(svg_path: str, out_w: int, out_h: int) -> Image.Image:
    """
    Rasterize the SVG into an RGBA Pillow image at the given output resolution.

    CairoSVG converts SVG -> PNG bytes, then Pillow loads it.
    """
    png_bytes = cairosvg.svg2png(url=svg_path, output_width=out_w, output_height=out_h)
    return Image.open(io.BytesIO(png_bytes)).convert("RGBA")


def viewbox_to_pixels(
    vb: SvgViewBox, img_w: int, img_h: int, x_vb: float, y_vb: float
) -> Tuple[float, float]:
    """
    Convert SVG viewBox coordinates (x_vb, y_vb) into pixel coordinates in the output raster.

    This assumes a linear mapping from viewBox space to pixel space.
    """
    px = (x_vb - vb.minx) / vb.width * img_w
    py = (y_vb - vb.miny) / vb.height * img_h
    return px, py


# ============================================================
# Colors (auto distinct)
# ============================================================
def make_color(i: int, n: int) -> Tuple[int, int, int, int]:
    """
    Generate a visually distinct RGBA color for index i out of n.

    Uses HSV hue cycling so each satellite has a different color.
    """
    import colorsys

    h = (i / max(1, n)) % 1.0
    s = 0.85
    v = 0.95
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return (int(r * 255), int(g * 255), int(b * 255), 255)


# ============================================================
# FFmpeg output (optional)
# ============================================================
def ffmpeg_available() -> bool:
    """Return True if ffmpeg is installed and on PATH."""
    return shutil.which("ffmpeg") is not None


def start_ffmpeg_sink(width: int, height: int, fps: int, output_url_or_file: str) -> subprocess.Popen:
    """
    Start an ffmpeg process that reads raw BGR frames from stdin and encodes them.

    - If output is rtmp://..., we use FLV container for RTMP streaming.
    - Otherwise we produce an MP4 file.

    The script will continuously write raw frame bytes into ffmpeg's stdin.
    """
    is_rtmp = output_url_or_file.lower().startswith("rtmp://")
    out_fmt = "flv" if is_rtmp else "mp4"

    cmd = [
        "ffmpeg",
        "-loglevel", "warning",
        "-re",  # read input at native frame rate (useful for live-feel)
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}",
        "-r", str(fps),
        "-i", "-",  # input from stdin
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-tune", "zerolatency",
        "-pix_fmt", "yuv420p",
        "-g", str(max(1, fps * 2)),  # GOP size (keyframe interval)
        "-f", out_fmt,
        output_url_or_file,
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


# ============================================================
# TLE loading via Skyfield downloader/cache
# ============================================================
def load_tles_active_set(loader: Loader, force_reload: bool) -> Dict[int, EarthSatellite]:
    """
    Load TLEs from CelesTrak "active" group via Skyfield's Loader cache.

    Skyfield will cache downloads in the loader directory (here ~/.sat_tracker_cache).
    """
    # Using gp.php query (active.txt may 404). Skyfield handles caching + conditional reload.
    url = "https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=tle"
    sats = loader.tle_file(url, reload=force_reload)

    by_id: Dict[int, EarthSatellite] = {}
    for s in sats:
        by_id[int(s.model.satnum)] = s
    return by_id


def load_tles_from_file(loader: Loader, filepath: str) -> Dict[int, EarthSatellite]:
    """
    Load TLEs from a local file (offline mode).

    The file must contain TLEs in standard 2-line format (optionally with name lines).
    """
    sats = loader.tle_file(filepath, reload=True)
    by_id: Dict[int, EarthSatellite] = {}
    for s in sats:
        by_id[int(s.model.satnum)] = s
    return by_id


# ============================================================
# Main
# ============================================================
def main():
    # CLI arguments so you can control output size, refresh frequency, preview/recording, and TLE sources.
    ap = argparse.ArgumentParser(description="Live Satellite Tracker over an Equal Earth SVG map.")
    ap.add_argument("--svg", required=True, help="Path to your SVG map file (Equal Earth projection)")
    ap.add_argument("--width", type=int, default=1920, help="Output width in pixels")
    ap.add_argument("--height", type=int, default=960, help="Output height in pixels")
    ap.add_argument("--fps", type=int, default=30, help="Preview/video FPS")
    ap.add_argument("--update-seconds", type=int, default=10, help="Recompute satellite position every N seconds")
    ap.add_argument("--tle-refresh-seconds", type=int, default=6 * 3600, help="Refetch TLEs every N seconds")
    ap.add_argument("--tle-file", default="", help="Optional: use local TLE file instead of downloading active group")
    ap.add_argument("--no-preview", action="store_true", help="Disable OpenCV preview window")
    ap.add_argument("--record", default="", help="Optional: record to file via ffmpeg (e.g. out.mp4)")
    ap.add_argument("--rtmp", default="", help="Optional: RTMP URL for streaming")
    args = ap.parse_args()

    # A small curated set of useful / well-known satellites (NORAD catalog IDs).
    # You can add/remove items here easily.
    sat_catnr = {
        "ISS (ZARYA)": 25544,
        "Hubble (HST)": 20580,
        "Terra (EOS AM-1)": 25994,
        "Aqua (EOS PM-1)": 27424,
        "Sentinel-1A": 39634,
        "Sentinel-2A": 40697,
        "Landsat 9": 49260,
        "GOES 16": 41866,
        "GOES 18": 51850,
        "Himawari-9": 41836,
    }

    # Used for stable per-satellite colors.
    label_list = list(sat_catnr.keys())
    label_index = {name: i for i, name in enumerate(label_list)}

    # Skyfield Loader handles caching of downloads (TLEs, etc).
    loader = Loader("~/.sat_tracker_cache")
    ts = loader.timescale()

    # Load SVG map and rasterize it once; we draw on top per-frame.
    print(f"[INFO] Loading SVG: {args.svg}")
    vb = parse_svg_viewbox(args.svg)
    base = render_svg_to_rgba(args.svg, args.width, args.height)

    # Choose a font for labels. Falls back to default if DejaVu isn't available.
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", size=max(14, args.width // 120))
    except Exception:
        font = ImageFont.load_default()

    # Optional ffmpeg sinks: one for recording to file, one for RTMP streaming.
    ffmpeg_record = None
    ffmpeg_rtmp = None

    if args.record:
        if not ffmpeg_available():
            raise RuntimeError("ffmpeg not found. Install it: sudo apt-get install -y ffmpeg")
        ffmpeg_record = start_ffmpeg_sink(args.width, args.height, args.fps, args.record)
        print(f"[INFO] Recording to: {args.record}")

    if args.rtmp:
        if not ffmpeg_available():
            raise RuntimeError("ffmpeg not found. Install it: sudo apt-get install -y ffmpeg")
        ffmpeg_rtmp = start_ffmpeg_sink(args.width, args.height, args.fps, args.rtmp)
        print("[INFO] Streaming via RTMP.")

    # Optional on-screen preview using OpenCV.
    if not args.no_preview:
        cv2.namedWindow("Live Satellite Tracker", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Live Satellite Tracker", args.width, args.height)
        print("[INFO] Preview window opened. Press 'q' to quit.")

    # Satellite objects and TLE refresh logic
    sats: Dict[str, EarthSatellite] = {}
    last_tle_fetch = 0.0

    def refresh_tles(force_reload: bool):
        """
        Refresh TLE data either from local file or from CelesTrak.

        `force_reload=True` forces Skyfield to re-download (otherwise it may use cache).
        """
        nonlocal last_tle_fetch, sats

        if args.tle_file:
            print(f"[INFO] Loading TLEs from local file: {args.tle_file}")
            by_id = load_tles_from_file(loader, args.tle_file)
        else:
            print("[INFO] Loading TLEs from CelesTrak Active group (cached by Skyfield)...")
            by_id = load_tles_active_set(loader, force_reload=force_reload)

        # Only keep satellites we care about; report missing ones.
        new_sats: Dict[str, EarthSatellite] = {}
        missing = []
        for label, catnr in sat_catnr.items():
            s = by_id.get(catnr)
            if s is None:
                missing.append(f"{label}({catnr})")
            else:
                new_sats[label] = s

        if missing:
            print("[WARN] Missing from current TLE set (won't be drawn):")
            for m in missing:
                print("  -", m)

        sats = new_sats
        last_tle_fetch = time.time()
        print(f"[INFO] TLEs ready. Tracking {len(sats)}/{len(sat_catnr)} satellites.")

    # Initial TLE load.
    refresh_tles(force_reload=True)

    # last_positions holds the most recent computed lat/lon/alt for each satellite.
    # We update these every --update-seconds (NOT every frame) to reduce compute cost.
    last_positions: Dict[str, Tuple[float, float, float]] = {}
    last_update = 0.0

    # Frame pacing for display/recording.
    frame_interval = 1.0 / max(1, args.fps)

    try:
        while True:
            now = time.time()

            # Periodically refresh the TLE set (e.g., every 6 hours).
            if now - last_tle_fetch >= args.tle_refresh_seconds:
                try:
                    refresh_tles(force_reload=True)
                except Exception as e:
                    # We keep running with old TLEs if refresh fails.
                    print(f"[WARN] TLE refresh failed: {e}")

            # Periodically recompute satellite subpoints (lat/lon) using Skyfield.
            if now - last_update >= args.update_seconds:
                t = ts.now()
                for label, sat in sats.items():
                    try:
                        # Subpoint is the point on Earth directly beneath the satellite.
                        sp = sat.at(t).subpoint()
                        last_positions[label] = (
                            sp.latitude.degrees,
                            sp.longitude.degrees,
                            sp.elevation.km,
                        )
                    except Exception as e:
                        print(f"[WARN] Position failed for {label}: {e}")

                # Log positions so you can see numeric output even if not previewing.
                for label, (lat, lon, alt) in last_positions.items():
                    print(f"[POS] {label}: lat={lat:+.2f} lon={lon:+.2f} alt={alt:.0f}km")

                last_update = now

            # ----------------------------
            # Draw a single frame
            # ----------------------------
            img = base.copy()
            draw = ImageDraw.Draw(img)

            # Header overlay with current UTC time.
            utc_txt = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
            header_h = max(34, args.height // 30)
            draw.rectangle([(0, 0), (args.width, header_h)], fill=(0, 0, 0, 150))
            draw.text(
                (10, 6),
                f"Live Satellite Tracker — {utc_txt}",
                font=font,
                fill=(255, 255, 255, 255),
            )

            # Marker radius scales with output size.
            r = max(6, args.width // 260)

            # Draw each satellite marker + label box.
            for label, (lat, lon, alt_km) in last_positions.items():
                # Project lat/lon to Equal Earth coordinates.
                x_vb, y_vb = equal_earth_project(lat, lon)

                # SVG coordinate systems are typically y-down; math projections are y-up.
                # If your SVG appears vertically flipped, this negation is usually the fix.
                y_vb = -y_vb

                # Convert projected viewBox coordinates to pixels.
                px, py = viewbox_to_pixels(vb, args.width, args.height, x_vb, y_vb)

                # Skip satellites that would be far off-screen (padding avoids popping at edges).
                if px < -200 or px > args.width + 200 or py < -200 or py > args.height + 200:
                    continue

                # Assign a consistent distinct color per satellite.
                c = make_color(label_index.get(label, 0), len(label_list))

                # Satellite marker dot.
                draw.ellipse(
                    [(px - r, py - r), (px + r, py + r)],
                    fill=c,
                    outline=(0, 0, 0, 255),
                    width=2,
                )

                # Label text with position and altitude.
                text = f"{label}: {lat:+.2f}°, {lon:+.2f}°  alt {alt_km:.0f} km"

                # Measure text so we can draw a box behind it.
                bbox = draw.textbbox((0, 0), text, font=font)
                tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
                pad = 6

                # Default label placement to the right of the marker.
                bx1, by1 = px + r + 10, py - th / 2 - pad
                bx2, by2 = bx1 + tw + 2 * pad, by1 + th + 2 * pad

                # If it would overflow to the right, place on the left instead.
                if bx2 > args.width:
                    bx1 = px - r - 10 - (tw + 2 * pad)
                    bx2 = bx1 + tw + 2 * pad

                # If it would overlap the header, nudge it down below the header.
                if by1 < header_h:
                    by1 = header_h + 4
                    by2 = by1 + th + 2 * pad

                # Semi-transparent label background.
                draw.rectangle(
                    [(bx1, by1), (bx2, by2)],
                    fill=(0, 0, 0, 160),
                    outline=(255, 255, 255, 160),
                )
                draw.text((bx1 + pad, by1 + pad), text, font=font, fill=(255, 255, 255, 255))

            # Convert Pillow RGBA image to OpenCV BGR for display/ffmpeg.
            frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGBA2BGR)

            # Preview window
            if not args.no_preview:
                cv2.imshow("Live Satellite Tracker", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    break

            # Record to file via ffmpeg pipe
            if ffmpeg_record and ffmpeg_record.stdin:
                try:
                    ffmpeg_record.stdin.write(frame.tobytes())
                except BrokenPipeError:
                    print("[ERROR] ffmpeg recording pipe closed.")
                    break

            # Stream to RTMP via ffmpeg pipe
            if ffmpeg_rtmp and ffmpeg_rtmp.stdin:
                try:
                    ffmpeg_rtmp.stdin.write(frame.tobytes())
                except BrokenPipeError:
                    print("[ERROR] ffmpeg RTMP pipe closed.")
                    break

            # Control frame rate
            time.sleep(frame_interval)

    finally:
        # Always clean up windows and ffmpeg processes even if an exception occurs.
        if not args.no_preview:
            cv2.destroyAllWindows()

        for proc in (ffmpeg_record, ffmpeg_rtmp):
            if proc:
                try:
                    if proc.stdin:
                        proc.stdin.close()
                except Exception:
                    pass
                proc.terminate()


if __name__ == "__main__":
    main()
