import base64
import io
import time
import threading
import zipfile
from pathlib import Path
from typing import List, Optional

import cv2
import requests
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from PIL.ExifTags import GPSTAGS
from starlette.concurrency import run_in_threadpool
from ultralytics import YOLO

BASE = Path(__file__).parent
MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_ZIP_BYTES = 100 * 1024 * 1024
MAX_ZIP_UNCOMPRESSED = 300 * 1024 * 1024
MAX_FILES = 30
IMG_EXT = {".jpg", ".jpeg", ".png"}
# Nominatim policy: yahan apna real email daalein
UA = {"User-Agent": "PotholeAI/1.0 (student project; contact: your-email@example.com)"}

AUTHORITIES = ["nhai", "cg_pwd", "pmgsy", "urban", "other"]
URBAN_WORDS = ("nagar", "municipal", "nigam", "palika", "smart city", "urban")

app = FastAPI(title="PotholeAI")
model = YOLO(str(BASE / "best.pt"))
model_lock = threading.Lock()
app.mount("/static", StaticFiles(directory=BASE / "static"), name="static")

GEO_CACHE = {}
ROAD_CACHE = {}
_last = {"nominatim": 0.0, "overpass": 0.0}


def _throttle(key, gap):
    wait = gap - (time.time() - _last[key])
    if wait > 0:
        time.sleep(wait)
    _last[key] = time.time()


def get_gps(data: bytes):
    try:
        exif = Image.open(io.BytesIO(data)).getexif()
        gps = exif.get_ifd(0x8825)
        if not gps:
            return None
        g = {GPSTAGS.get(k, k): v for k, v in gps.items()}

        def conv(vals, ref):
            d, m, s = [float(x) for x in vals]
            val = d + m / 60 + s / 3600
            return -val if ref in ("S", "W") else val

        return (conv(g["GPSLatitude"], g["GPSLatitudeRef"]),
                conv(g["GPSLongitude"], g["GPSLongitudeRef"]))
    except Exception:
        return None


def reverse_geo(lat, lon):
    key = (round(lat, 4), round(lon, 4))
    if key in GEO_CACHE:
        return GEO_CACHE[key]
    empty = {"address": None, "state": None, "district": None}
    try:
        _throttle("nominatim", 1.1)  # Nominatim: ~1 request/second
        r = requests.get("https://nominatim.openstreetmap.org/reverse",
                         params={"lat": lat, "lon": lon, "format": "json",
                                 "zoom": 18, "addressdetails": 1},
                         headers=UA, timeout=10)
        j = r.json()
        a = j.get("address", {})
        info = {"address": j.get("display_name"), "state": a.get("state"),
                "district": a.get("state_district") or a.get("county")}
        GEO_CACHE[key] = info
        return info
    except Exception:
        return empty


def nearby_road(lat, lon):
    key = (round(lat, 4), round(lon, 4))
    if key in ROAD_CACHE:
        return ROAD_CACHE[key]
    road = None
    try:
        _throttle("overpass", 1.0)
        q = f'[out:json][timeout:15];way(around:30,{lat},{lon})["highway"];out tags 8;'
        r = requests.post("https://overpass-api.de/api/interpreter",
                          data={"data": q}, headers=UA, timeout=20)
        elements = r.json().get("elements", [])
        named = [w["tags"] for w in elements
                 if w.get("tags", {}).get("name") or w.get("tags", {}).get("ref")]
        pick = (named or [w.get("tags", {}) for w in elements] or [None])[0]
        if pick:
            road = {"name": pick.get("name"), "ref": pick.get("ref"),
                    "highway": pick.get("highway"), "operator": pick.get("operator")}
        ROAD_CACHE[key] = road
    except Exception:
        pass
    return road


def suggest_authority(road, geo):
    """Sirf anumaan. Road ka type aur address dekh kar; pakki jaankari nahi."""
    road = road or {}
    ref = (road.get("ref") or "").upper()
    hw = road.get("highway") or ""
    addr = (geo.get("address") or "").lower()
    if ref.startswith("NH") or hw in ("motorway", "trunk"):
        return "nhai"
    if ref.startswith(("SH", "MDR")) or hw in ("primary", "secondary"):
        return "cg_pwd"
    if any(w in addr for w in URBAN_WORDS):
        return "urban"
    if hw in ("residential", "unclassified", "tertiary", "track", "service", "living_street"):
        return "pmgsy"
    return "other"


def severity(area_ratio):
    """Heuristic: box ka photo mein kitna hissa hai. Standard nahi hai."""
    if area_ratio < 0.02:
        return "Low"
    if area_ratio < 0.08:
        return "Medium"
    return "High"


def is_image_name(name):
    return Path(name).suffix.lower() in IMG_EXT


def extract_zip(raw: bytes, errors: list):
    out, total = [], 0
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        for info in z.infolist():
            if info.is_dir() or info.filename.startswith("__MACOSX"):
                continue
            name = Path(info.filename).name  # sirf naam lete hain, disk par kuch nahi likhte
            if name.startswith(".") or not is_image_name(name):
                continue
            if info.file_size > MAX_IMAGE_BYTES:
                errors.append(f"{info.filename}: file 10 MB se badi hai")
                continue
            total += info.file_size
            if total > MAX_ZIP_UNCOMPRESSED:
                raise HTTPException(413, "ZIP ke andar ka data bahut bada hai")
            out.append((name, z.read(info)))
            if len(out) > MAX_FILES:
                raise HTTPException(400, f"ZIP mein {MAX_FILES} se zyada images hain")
    return out


def process_image(name, data, conf, browser):
    try:
        Image.open(io.BytesIO(data)).verify()
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        return None, f"{name}: invalid image file"

    with model_lock:
        t0 = time.perf_counter()
        result = model.predict(img, conf=conf, imgsz=640, verbose=False)[0]
        inf = time.perf_counter() - t0

    h, w = result.orig_shape
    dets = []
    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        ratio = (x2 - x1) * (y2 - y1) / (w * h)
        dets.append({
            "class_name": result.names[int(box.cls)],
            "confidence": round(float(box.conf), 3),
            "severity": severity(ratio),
            "bbox": {"x1": round(x1), "y1": round(y1), "x2": round(x2), "y2": round(y2)},
        })

    coords, source = None, None
    exif_gps = get_gps(data)
    if exif_gps:
        coords, source = exif_gps, "photo"
    elif browser:
        coords, source = browser, "browser"

    location = None
    if coords:
        clat, clon = coords
        geo = reverse_geo(clat, clon)
        road = nearby_road(clat, clon)
        location = {
            "lat": round(clat, 6), "lon": round(clon, 6), "source": source,
            "address": geo.get("address"), "state": geo.get("state"),
            "district": geo.get("district"),
            "in_chhattisgarh": geo.get("state") == "Chhattisgarh",
            "road": road,
            "suggested_authority": suggest_authority(road, geo),
        }

    ok, buf = cv2.imencode(".jpg", result.plot())
    confs = [d["confidence"] for d in dets]
    return {
        "file_name": name,
        "total_potholes": len(dets),
        "average_confidence": round(sum(confs) / len(confs), 3) if confs else 0,
        "inference_time": round(inf, 3),
        "detections": dets,
        "location": location,
        "processed_image": "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode(),
    }, None


def run_all(items, conf, browser):
    results, errors = [], []
    for name, data in items:
        if len(data) > MAX_IMAGE_BYTES:
            errors.append(f"{name}: file 10 MB se badi hai")
            continue
        r, err = process_image(name, data, conf, browser)
        if err:
            errors.append(err)
        else:
            results.append(r)
    return results, errors


@app.get("/")
def index():
    return FileResponse(BASE / "static" / "index.html")


@app.get("/health")
def health():
    return {"status": "ok", "classes": model.names}


@app.get("/api/meta")
def meta():
    return {"authorities": AUTHORITIES, "state": "Chhattisgarh"}


@app.post("/api/detect-batch")
async def detect_batch(
    files: List[UploadFile] = File(...),
    conf: float = Form(0.25),
    lat: Optional[float] = Form(None),
    lon: Optional[float] = Form(None),
):
    if not 0.05 <= conf <= 0.95:
        raise HTTPException(400, "conf 0.05 se 0.95 ke beech hona chahiye")
    if lat is not None and not -90 <= lat <= 90:
        raise HTTPException(400, "Invalid latitude")
    if lon is not None and not -180 <= lon <= 180:
        raise HTTPException(400, "Invalid longitude")

    items, errors = [], []
    for f in files:
        name = Path(f.filename or "image").name  # path traversal se bachav
        raw = await f.read()
        ext = Path(name).suffix.lower()
        if ext == ".zip":
            if len(raw) > MAX_ZIP_BYTES:
                errors.append(f"{name}: ZIP 100 MB se badi hai")
                continue
            try:
                items.extend(extract_zip(raw, errors))
            except zipfile.BadZipFile:
                errors.append(f"{name}: ZIP file kharab hai")
        elif ext in IMG_EXT:
            items.append((name, raw))
        else:
            errors.append(f"{name}: sirf JPG, PNG ya ZIP allowed hai")

    if not items:
        raise HTTPException(400, "Koi valid image nahi mili" + (": " + "; ".join(errors))[:400] if errors else "Koi valid image nahi mili")
    if len(items) > MAX_FILES:
        raise HTTPException(400, f"Ek baar mein {MAX_FILES} se zyada images nahi")

    browser = (lat, lon) if lat is not None and lon is not None else None
    results, more_errors = await run_in_threadpool(run_all, items, conf, browser)
    errors.extend(more_errors)

    return {
        "success": True,
        "count": len(results),
        "total_potholes": sum(r["total_potholes"] for r in results),
        "results": results,
        "errors": errors,
    }