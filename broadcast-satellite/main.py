
#!/usr/bin/env python3
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
A1 = 1.340264
A2 = -0.081106
A3 = 0.000893
A4 = 0.003796
M = math.sqrt(3) / 2.0

_x_equator_rad = (2.0 * math.sqrt(3) * math.pi) / (3.0 * A1)
_x_equator_deg = _x_equator_rad * 180.0 / math.pi
EE_SCALE = 180.0 / _x_equator_deg


def equal_earth_project(lat_deg: float, lon_deg: float) -> Tuple[float, float]:
    phi = math.radians(lat_deg)
    lam = math.radians(lon_deg)

    theta = math.asin(M * math.sin(phi))
    t2 = theta * theta
    t6 = t2 * t2 * t2
    t8 = t6 * t2

    d = (A1 + 3.0 * A2 * t2 + 7.0 * A3 * t6 + 9.0 * A4 * t8)

    x = (2.0 * math.sqrt(3) * lam * math.cos(theta)) / (3.0 * d)
    y = (A1 * theta + A2 * theta**3 + A3 * theta**7 + A4 * theta**9)

    x_vb = (x * 180.0 / math.pi) * EE_SCALE
    y_vb = (y * 180.0 / math.pi) * EE_SCALE
    return x_vb, y_vb


# ============================================================
# SVG helpers
# ============================================================
@dataclass
class SvgViewBox:
    minx: float
    miny: float
    width: float
    height: float


def parse_svg_viewbox(svg_path: str) -> SvgViewBox:
    root = ET.parse(svg_path).getroot()
    vb = root.attrib.get("viewBox")
    if not vb:
        raise ValueError("SVG is missing viewBox attribute (needed for mapping).")
    minx, miny, w, h = map(float, vb.split())
    return SvgViewBox(minx=minx, miny=miny, width=w, height=h)


def render_svg_to_rgba(svg_path: str, out_w: int, out_h: int) -> Image.Image:
    png_bytes = cairosvg.svg2png(url=svg_path, output_width=out_w, output_height=out_h)
    return Image.open(io.BytesIO(png_bytes)).convert("RGBA")


def viewbox_to_pixels(vb: SvgViewBox, img_w: int, img_h: int, x_vb: float, y_vb: float) -> Tuple[float, float]:
    px = (x_vb - vb.minx) / vb.width * img_w
    py = (y_vb - vb.miny) / vb.height * img_h
    return px, py


# ============================================================
# Colors (auto distinct)
# ============================================================
def make_color(i: int, n: int) -> Tuple[int, int, int, int]:
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
    return shutil.which("ffmpeg") is not None


def start_ffmpeg_sink(width: int, height: int, fps: int, output_url_or_file: str) -> subprocess.Popen:
    is_rtmp = output_url_or_file.lower().startswith("rtmp://")
    out_fmt = "flv" if is_rtmp else "mp4"

    cmd = [
        "ffmpeg",
        "-loglevel", "warning",
        "-re",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}",
        "-r", str(fps),
        "-i", "-",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-tune", "zerolatency",
        "-pix_fmt", "yuv420p",
        "-g", str(max(1, fps * 2)),
        "-f", out_fmt,
        output_url_or_file,
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


# ============================================================
# TLE loading via Skyfield downloader/cache (NO direct requests)
# ============================================================
def load_tles_active_set(loader: Loader, force_reload: bool) -> Dict[int, EarthSatellite]:
    # IMPORTANT: use gp.php group query (active.txt can 404)
    url = "https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=tle"
    sats = loader.tle_file(url, reload=force_reload)

    by_id: Dict[int, EarthSatellite] = {}
    for s in sats:
        by_id[int(s.model.satnum)] = s
    return by_id


def load_tles_from_file(loader: Loader, filepath: str) -> Dict[int, EarthSatellite]:
    sats = loader.tle_file(filepath, reload=True)
    by_id: Dict[int, EarthSatellite] = {}
    for s in sats:
        by_id[int(s.model.satnum)] = s
    return by_id


# ============================================================
# Main
# ============================================================
def main():
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

    # ~10 useful/important satellites (NORAD IDs)
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

    label_list = list(sat_catnr.keys())
    label_index = {name: i for i, name in enumerate(label_list)}

    loader = Loader("~/.sat_tracker_cache")
    ts = loader.timescale()

    print(f"[INFO] Loading SVG: {args.svg}")
    vb = parse_svg_viewbox(args.svg)
    base = render_svg_to_rgba(args.svg, args.width, args.height)

    # Font (PIL) - best effort
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", size=max(14, args.width // 120))
    except Exception:
        font = ImageFont.load_default()

    # ffmpeg outputs (optional)
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

    if not args.no_preview:
        cv2.namedWindow("Live Satellite Tracker", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Live Satellite Tracker", args.width, args.height)
        print("[INFO] Preview window opened. Press 'q' to quit.")

    # Satellites + refresh logic
    sats: Dict[str, EarthSatellite] = {}
    last_tle_fetch = 0.0

    def refresh_tles(force_reload: bool):
        nonlocal last_tle_fetch, sats

        if args.tle_file:
            print(f"[INFO] Loading TLEs from local file: {args.tle_file}")
            by_id = load_tles_from_file(loader, args.tle_file)
        else:
            print("[INFO] Loading TLEs from CelesTrak Active group (cached by Skyfield)...")
            by_id = load_tles_active_set(loader, force_reload=force_reload)

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

    refresh_tles(force_reload=True)

    last_positions: Dict[str, Tuple[float, float, float]] = {}
    last_update = 0.0
    frame_interval = 1.0 / max(1, args.fps)

    try:
        while True:
            now = time.time()

            if now - last_tle_fetch >= args.tle_refresh_seconds:
                try:
                    refresh_tles(force_reload=True)
                except Exception as e:
                    print(f"[WARN] TLE refresh failed: {e}")

            if now - last_update >= args.update_seconds:
                t = ts.now()
                for label, sat in sats.items():
                    try:
                        sp = sat.at(t).subpoint()
                        last_positions[label] = (
                            sp.latitude.degrees,
                            sp.longitude.degrees,
                            sp.elevation.km,
                        )
                    except Exception as e:
                        print(f"[WARN] Position failed for {label}: {e}")

                for label, (lat, lon, alt) in last_positions.items():
                    print(f"[POS] {label}: lat={lat:+.2f} lon={lon:+.2f} alt={alt:.0f}km")

                last_update = now

            # Draw frame
            img = base.copy()
            draw = ImageDraw.Draw(img)

            utc_txt = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
            header_h = max(34, args.height // 30)
            draw.rectangle([(0, 0), (args.width, header_h)], fill=(0, 0, 0, 150))
            draw.text((10, 6), f"Live Satellite Tracker — {utc_txt}", font=font, fill=(255, 255, 255, 255))

            r = max(6, args.width // 260)

            for label, (lat, lon, alt_km) in last_positions.items():
                x_vb, y_vb = equal_earth_project(lat, lon)
                y_vb = -y_vb  # flip for SVG y-down

                px, py = viewbox_to_pixels(vb, args.width, args.height, x_vb, y_vb)

                if px < -200 or px > args.width + 200 or py < -200 or py > args.height + 200:
                    continue

                c = make_color(label_index.get(label, 0), len(label_list))

                draw.ellipse([(px - r, py - r), (px + r, py + r)], fill=c, outline=(0, 0, 0, 255), width=2)

                text = f"{label}: {lat:+.2f}°, {lon:+.2f}°  alt {alt_km:.0f} km"
                bbox = draw.textbbox((0, 0), text, font=font)
                tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
                pad = 6

                bx1, by1 = px + r + 10, py - th / 2 - pad
                bx2, by2 = bx1 + tw + 2 * pad, by1 + th + 2 * pad

                if bx2 > args.width:
                    bx1 = px - r - 10 - (tw + 2 * pad)
                    bx2 = bx1 + tw + 2 * pad
                if by1 < header_h:
                    by1 = header_h + 4
                    by2 = by1 + th + 2 * pad

                draw.rectangle([(bx1, by1), (bx2, by2)], fill=(0, 0, 0, 160), outline=(255, 255, 255, 160))
                draw.text((bx1 + pad, by1 + pad), text, font=font, fill=(255, 255, 255, 255))

            frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGBA2BGR)

            if not args.no_preview:
                cv2.imshow("Live Satellite Tracker", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    break

            if ffmpeg_record and ffmpeg_record.stdin:
                try:
                    ffmpeg_record.stdin.write(frame.tobytes())
                except BrokenPipeError:
                    print("[ERROR] ffmpeg recording pipe closed.")
                    break

            if ffmpeg_rtmp and ffmpeg_rtmp.stdin:
                try:
                    ffmpeg_rtmp.stdin.write(frame.tobytes())
                except BrokenPipeError:
                    print("[ERROR] ffmpeg RTMP pipe closed.")
                    break

            time.sleep(frame_interval)

    finally:
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
