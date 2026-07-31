"""
ACR -> MP3 -> Transcript API
============================
שרת קל-משקל שמקבל הקלטת שיחה (ACR/AMR/3GP/M4A/OPUS/WAV/MP3),
ממיר ל-MP3 דחוס עם ffmpeg, ומתמלל אותה (עברית) דרך Groq Whisper (חינם)
או faster-whisper מקומי כגיבוי.

תוכנן לרוץ על Hugging Face Spaces (חינם, ללא כרטיס אשראי),
Google Cloud Run, Fly.io או כל VPS.
"""

import os
import re
import shutil
import asyncio
import logging
import tempfile
import subprocess
import uuid
from pathlib import Path
from typing import Optional, Literal

import httpx
from fastapi import FastAPI, UploadFile, File, Form, Header, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ----------------------------------------------------------------------------
# הגדרות (משתני סביבה)
# ----------------------------------------------------------------------------
API_KEY          = os.getenv("API_KEY", "")                  # מפתח לאבטחת ה-API. ריק = פתוח
GEMINI_API_KEY   = os.getenv("GEMINI_API_KEY", "")           # מפתח מ-Google AI Studio (חינם)
# gemini-2.5-flash חסום למפתחות חדשים ("no longer available to new users").
GEMINI_MODEL     = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
GROQ_API_KEY     = os.getenv("GROQ_API_KEY", "")             # https://console.groq.com  (חינם)
GROQ_MODEL       = os.getenv("GROQ_MODEL", "whisper-large-v3-turbo")
LOCAL_MODEL      = os.getenv("LOCAL_MODEL", "small")         # tiny/base/small/medium
DEFAULT_LANGUAGE = os.getenv("DEFAULT_LANGUAGE", "he")
MP3_BITRATE      = os.getenv("MP3_BITRATE", "24k")           # 24k = מרווח איכות מעל AMR-NB (12.8k)
GEMINI_CHUNK_MIN = int(os.getenv("GEMINI_CHUNK_MIN", "15"))  # פיצול שיחות ארוכות (מגבלת טוקני פלט)
MAX_UPLOAD_MB    = int(os.getenv("MAX_UPLOAD_MB", "200"))
KEEP_MP3_MINUTES = int(os.getenv("KEEP_MP3_MINUTES", "60"))  # כמה זמן לשמור MP3 להורדה
DATA_DIR         = Path(os.getenv("DATA_DIR", tempfile.gettempdir())) / "acr_api"
PUBLIC_BASE_URL  = os.getenv("PUBLIC_BASE_URL", "").rstrip("/")

DATA_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("acr-api")

# httpx מתעד כל URL מלא ברמת INFO. מכיוון שספקים מסוימים מקבלים מפתח
# בשורת השאילתה, זה עלול לכתוב סודות ללוג. משתיקים לרמת אזהרה.
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

app = FastAPI(
    title="ACR Transcribe API",
    version="1.0.0",
    description="המרת הקלטות שיחה ל-MP3 + תמלול עברית. חינמי.",
)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

JOBS: dict[str, dict] = {}
_local_model = None


# ----------------------------------------------------------------------------
# עזרים
# ----------------------------------------------------------------------------
def check_key(x_api_key: Optional[str]):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(401, "Invalid or missing X-API-Key")


def run(cmd: list, timeout: int = 900) -> subprocess.CompletedProcess:
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if p.returncode != 0:
        raise HTTPException(500, f"ffmpeg failed: {p.stderr[-800:]}")
    return p


def probe_duration(path: Path) -> float:
    try:
        p = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", str(path)],
            capture_output=True, text=True, timeout=60,
        )
        return round(float(p.stdout.strip()), 2)
    except Exception:
        return 0.0


def probe_sample_rate(path: Path) -> int:
    try:
        p = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a:0",
             "-show_entries", "stream=sample_rate", "-of", "default=nw=1:nk=1", str(path)],
            capture_output=True, text=True, timeout=60,
        )
        return int(p.stdout.strip())
    except Exception:
        return 16000


def to_mp3(src: Path, dst: Path, bitrate: str = MP3_BITRATE) -> Path:
    """
    המרה אוניברסלית ל-MP3 מונו.
    ffmpeg מזהה לבד AMR-NB/AMR-WB/3GP/M4A/OPUS/WAV גם אם הסיומת היא .acr

    שומר על קצב הדגימה של המקור (עד 16kHz): הקלטות טלפון הן 8kHz,
    ודגימה מחדש ל-16kHz רק מנפחת את הקובץ בלי להוסיף מידע.
    """
    rate = min(probe_sample_rate(src), 16000)
    run([
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(src),
        "-vn", "-ac", "1", "-ar", str(rate),
        "-codec:a", "libmp3lame", "-b:a", bitrate,
        str(dst),
    ])
    if not dst.exists() or dst.stat().st_size == 0:
        raise HTTPException(500, "Conversion produced an empty file")
    return dst


def split_audio(src: Path, chunk_seconds: int = 900) -> list:
    """פיצול קבצים ארוכים (מגבלת 25MB של Groq / זיכרון מוגבל)."""
    out_dir = src.parent / f"{src.stem}_chunks"
    out_dir.mkdir(exist_ok=True)
    run([
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(src), "-f", "segment",
        "-segment_time", str(chunk_seconds),
        "-c", "copy", str(out_dir / "part_%03d.mp3"),
    ])
    return sorted(out_dir.glob("part_*.mp3"))


def parse_drive_id(url: str) -> Optional[str]:
    """מחלץ file-id מקישור שיתוף / קישור פתיחה / id גולמי."""
    m = (re.search(r"/file/d/([A-Za-z0-9_-]{20,})", url)
         or re.search(r"[?&]id=([A-Za-z0-9_-]{20,})", url)
         or re.search(r"/d/([A-Za-z0-9_-]{20,})", url))
    if m:
        return m.group(1)
    if re.fullmatch(r"[A-Za-z0-9_-]{20,}", url.strip()):
        return url.strip()
    return None


async def download_to(url: str, dst: Path, drive_token: str = "") -> Path:
    """
    הורדה מ-URL כללי או מ-Google Drive.
    drive_token: OAuth access token — מאפשר למשוך קבצים *פרטיים* מהדרייב
                 (ה-CRM שלך כבר מחזיק טוקן כזה).
    """
    headers = {}
    drive_id = parse_drive_id(url)
    if drive_id:
        if drive_token:
            url = f"https://www.googleapis.com/drive/v3/files/{drive_id}?alt=media&supportsAllDrives=true"
            headers["Authorization"] = f"Bearer {drive_token}"
        else:
            url = f"https://drive.google.com/uc?export=download&id={drive_id}"

    limit = MAX_UPLOAD_MB * 1024 * 1024
    total = 0
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=300) as client:
            async with client.stream("GET", url, headers=headers) as r:
                if r.status_code in (401, 403):
                    raise HTTPException(
                        403,
                        "Access denied. For a private Drive file pass 'drive_access_token', "
                        "or share the file as 'anyone with the link'.",
                    )
                if r.status_code >= 400:
                    raise HTTPException(400, f"Download failed ({r.status_code}) for {url[:120]}")
                ctype = r.headers.get("content-type", "")
                with open(dst, "wb") as f:
                    async for chunk in r.aiter_bytes(1 << 16):
                        total += len(chunk)
                        if total > limit:
                            raise HTTPException(413, f"File exceeds {MAX_UPLOAD_MB}MB")
                        f.write(chunk)
    except httpx.HTTPError as e:
        raise HTTPException(502, f"Network error while downloading: {type(e).__name__}: {e}")

    if total == 0:
        raise HTTPException(400, "Downloaded 0 bytes — file is empty or inaccessible")
    if "text/html" in ctype:
        raise HTTPException(
            400,
            "Got an HTML page instead of audio — the Drive file is private or hit the "
            "virus-scan interstitial. Pass 'drive_access_token' instead.",
        )
    return dst


# ----------------------------------------------------------------------------
# מנועי תמלול
# ----------------------------------------------------------------------------
GEMINI_BASE = "https://generativelanguage.googleapis.com"


def gemini_auth() -> dict:
    """
    המפתח נשלח בכותרת ולא ב-?key= בשורת השאילתה.
    בשורת השאילתה הוא היה נכתב לכל לוג גישה — של השרת, של הפרוקסי,
    ושל כל שירות תיעוד באמצע.
    """
    return {"x-goog-api-key": GEMINI_API_KEY}


async def _gemini_upload(path: Path) -> str:
    """העלאת קובץ ל-Files API של Gemini (חינם, נשמר 48 שעות)."""
    size = path.stat().st_size
    async with httpx.AsyncClient(timeout=600) as c:
        start = await c.post(
            f"{GEMINI_BASE}/upload/v1beta/files",
            headers={
                **gemini_auth(),
                "X-Goog-Upload-Protocol": "resumable",
                "X-Goog-Upload-Command": "start",
                "X-Goog-Upload-Header-Content-Length": str(size),
                "X-Goog-Upload-Header-Content-Type": "audio/mpeg",
                "Content-Type": "application/json",
            },
            json={"file": {"display_name": path.name}},
        )
        if start.status_code >= 400:
            raise HTTPException(502, f"Gemini upload init failed: {start.text[:400]}")
        upload_url = start.headers.get("x-goog-upload-url")
        if not upload_url:
            raise HTTPException(502, "Gemini did not return an upload URL")

        up = await c.post(
            upload_url,
            headers={
                "Content-Length": str(size),
                "X-Goog-Upload-Offset": "0",
                "X-Goog-Upload-Command": "upload, finalize",
            },
            content=path.read_bytes(),
        )
        if up.status_code >= 400:
            raise HTTPException(502, f"Gemini upload failed: {up.text[:400]}")
        info = up.json()["file"]
        name, uri = info["name"], info["uri"]

        # המתנה עד שהקובץ מוכן
        for _ in range(60):
            if info.get("state") == "ACTIVE":
                return uri
            await asyncio.sleep(1.5)
            st = await c.get(f"{GEMINI_BASE}/v1beta/{name}", headers=gemini_auth())
            info = st.json()
        raise HTTPException(504, "Gemini file stayed in PROCESSING state")


async def transcribe_gemini(path: Path, language: str, prompt: str = "") -> dict:
    """
    תמלול דרך Gemini (Google AI Studio) — אותו מפתח שכבר יש לך.
    יתרון על Whisper: מבין הקשר, מפריד דוברים, ומחזיר JSON מובנה.
    """
    lang_name = {"he": "Hebrew", "en": "English", "ar": "Arabic", "ru": "Russian"}.get(language, language)
    instruction = (
        f"You are a precise call-recording transcriber. Transcribe this phone call verbatim in "
        f"{'the original spoken language' if language == 'auto' else lang_name}. "
        "Do not translate, summarize, censor or add commentary. Keep filler words. "
        "Split into segments by speaker turn, label speakers as 'דובר א' / 'דובר ב' "
        "(or Speaker A / Speaker B for non-Hebrew). Give start/end times in seconds. "
        "If audio is silent or unintelligible, return an empty segments array."
    )
    if prompt:
        instruction += f"\nDomain context (names, products, jargon that may appear): {prompt}"

    file_uri = await _gemini_upload(path)

    body = {
        "contents": [{
            "role": "user",
            "parts": [
                {"text": instruction},
                {"file_data": {"mime_type": "audio/mpeg", "file_uri": file_uri}},
            ],
        }],
        "generationConfig": {
            "temperature": 0,
            "response_mime_type": "application/json",
            "response_schema": {
                "type": "OBJECT",
                "properties": {
                    "language": {"type": "STRING"},
                    "segments": {
                        "type": "ARRAY",
                        "items": {
                            "type": "OBJECT",
                            "properties": {
                                "start": {"type": "NUMBER"},
                                "end": {"type": "NUMBER"},
                                "speaker": {"type": "STRING"},
                                "text": {"type": "STRING"},
                            },
                            "required": ["start", "end", "speaker", "text"],
                        },
                    },
                },
                "required": ["language", "segments"],
            },
        },
    }

    async with httpx.AsyncClient(timeout=900) as c:
        r = await c.post(
            f"{GEMINI_BASE}/v1beta/models/{GEMINI_MODEL}:generateContent",
            headers=gemini_auth(),
            json=body,
        )
    if r.status_code >= 400:
        raise HTTPException(502, f"Gemini error {r.status_code}: {r.text[:500]}")

    import json as _json
    try:
        raw = r.json()["candidates"][0]["content"]["parts"][0]["text"]
        parsed = _json.loads(raw)
    except Exception as e:
        raise HTTPException(502, f"Could not parse Gemini response: {e}")

    segs = [
        {"start": round(float(s.get("start") or 0), 2),
         "end": round(float(s.get("end") or 0), 2),
         "speaker": s.get("speaker", ""),
         "text": (s.get("text") or "").strip()}
        for s in parsed.get("segments", [])
    ]
    return {
        "text": " ".join(s["text"] for s in segs).strip(),
        "language": parsed.get("language", language),
        "segments": segs,
        "engine": f"gemini:{GEMINI_MODEL}",
    }


async def transcribe_groq(path: Path, language: str, prompt: str = "") -> dict:
    """Groq Whisper large-v3-turbo — מהיר מאוד, tier חינמי נדיב, עברית מצוינת."""
    async with httpx.AsyncClient(timeout=600) as client:
        with open(path, "rb") as f:
            data = {"model": GROQ_MODEL, "response_format": "verbose_json", "temperature": "0"}
            if language and language != "auto":
                data["language"] = language
            if prompt:
                data["prompt"] = prompt
            r = await client.post(
                "https://api.groq.com/openai/v1/audio/transcriptions",
                headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                files={"file": (path.name, f, "audio/mpeg")},
                data=data,
            )
    if r.status_code >= 400:
        raise HTTPException(502, f"Groq error {r.status_code}: {r.text[:500]}")
    j = r.json()
    return {
        "text": (j.get("text") or "").strip(),
        "language": j.get("language", language),
        "segments": [
            {"start": s.get("start"), "end": s.get("end"), "text": (s.get("text") or "").strip()}
            for s in j.get("segments", [])
        ],
        "engine": f"groq:{GROQ_MODEL}",
    }


def transcribe_local(path: Path, language: str) -> dict:
    """faster-whisper על CPU — עובד לגמרי אופליין, בלי מפתחות."""
    global _local_model
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        raise HTTPException(
            501,
            "Local engine unavailable: faster-whisper is not installed. "
            "Uncomment it in requirements.txt and redeploy with >=2Gi memory, "
            "or use engine='gemini' / 'groq'.",
        )

    if _local_model is None:
        log.info("Loading faster-whisper '%s' ...", LOCAL_MODEL)
        _local_model = WhisperModel(LOCAL_MODEL, device="cpu", compute_type="int8")

    segments, info = _local_model.transcribe(
        str(path),
        language=None if language == "auto" else language,
        vad_filter=True,
        beam_size=1,
    )
    segs, parts = [], []
    for s in segments:
        segs.append({"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip()})
        parts.append(s.text.strip())
    return {
        "text": " ".join(parts).strip(),
        "language": info.language,
        "segments": segs,
        "engine": f"faster-whisper:{LOCAL_MODEL}",
    }


def pick_engine(engine: str) -> str:
    """auto -> gemini אם יש מפתח, אחרת groq, אחרת מקומי."""
    if engine != "auto":
        return engine
    if GEMINI_API_KEY:
        return "gemini"
    if GROQ_API_KEY:
        return "groq"
    return "local"


async def transcribe_file(mp3: Path, language: str, engine: str, prompt: str = "") -> dict:
    chosen = pick_engine(engine)
    size_mb = mp3.stat().st_size / 1024 / 1024

    if chosen == "gemini":
        if not GEMINI_API_KEY:
            raise HTTPException(400, "GEMINI_API_KEY is not configured")

        duration = probe_duration(mp3)
        # שיחות ארוכות: תמלול מלא בבקשה אחת נחתך במגבלת טוקני הפלט.
        # פיצול לקטעים הוא מה שמונע תמלול קטוע באמצע המשפט.
        if duration > GEMINI_CHUNK_MIN * 60:
            parts_files = split_audio(mp3, GEMINI_CHUNK_MIN * 60)
            log.info("Long call (%.1f min) -> %d chunks", duration / 60, len(parts_files))
            merged, all_segs, offset = [], [], 0.0
            for i, part in enumerate(parts_files, 1):
                res = await transcribe_gemini(part, language, prompt)
                log.info("chunk %d/%d done (%d chars)", i, len(parts_files), len(res["text"]))
                merged.append(res["text"])
                for s in res["segments"]:
                    s["start"] = round((s["start"] or 0) + offset, 2)
                    s["end"] = round((s["end"] or 0) + offset, 2)
                    all_segs.append(s)
                offset += probe_duration(part)
                part.unlink(missing_ok=True)
            return {
                "text": " ".join(x for x in merged if x).strip(),
                "language": language,
                "segments": all_segs,
                "engine": f"gemini:{GEMINI_MODEL}(x{len(parts_files)} chunks)",
            }

        return await transcribe_gemini(mp3, language, prompt)

    use_groq = chosen == "groq"
    if use_groq and not GROQ_API_KEY:
        raise HTTPException(400, "GROQ_API_KEY is not configured")

    if use_groq and size_mb > 24:
        merged, all_segs, offset = [], [], 0.0
        for part in split_audio(mp3):
            res = await transcribe_groq(part, language, prompt)
            merged.append(res["text"])
            for s in res["segments"]:
                s["start"] = round((s["start"] or 0) + offset, 2)
                s["end"] = round((s["end"] or 0) + offset, 2)
                all_segs.append(s)
            offset += probe_duration(part)
        return {"text": " ".join(merged).strip(), "language": language,
                "segments": all_segs, "engine": f"groq:{GROQ_MODEL}(chunked)"}

    if use_groq:
        return await transcribe_groq(mp3, language, prompt)

    return await asyncio.to_thread(transcribe_local, mp3, language)


# ----------------------------------------------------------------------------
# הליבה
# ----------------------------------------------------------------------------
async def process(src: Path, job_id: str, language: str, engine: str,
                  prompt: str, want_mp3: bool, want_segments: bool) -> dict:
    mp3_path = DATA_DIR / f"{job_id}.mp3"
    original_size = src.stat().st_size

    to_mp3(src, mp3_path)
    duration = probe_duration(mp3_path)
    result = await transcribe_file(mp3_path, language, engine, prompt)

    out = {
        "job_id": job_id,
        "status": "done",
        "text": result["text"],
        "language": result["language"],
        "engine": result["engine"],
        "duration_sec": duration,
        "original_size_bytes": original_size,
        "mp3_size_bytes": mp3_path.stat().st_size,
        "compression_ratio": round(original_size / max(mp3_path.stat().st_size, 1), 2),
    }
    if want_segments:
        out["segments"] = result["segments"]
    if want_mp3:
        out["mp3_url"] = f"{PUBLIC_BASE_URL}/download/{job_id}.mp3"
        out["mp3_expires_in_minutes"] = KEEP_MP3_MINUTES
    else:
        mp3_path.unlink(missing_ok=True)
    return out


class JobRequest(BaseModel):
    url: str = Field(..., description="URL ישיר, קישור שיתוף של Google Drive, או file-id של Drive")
    drive_access_token: str = Field("", description="OAuth token של Drive — למשיכת קבצים פרטיים")
    language: str = Field(DEFAULT_LANGUAGE, description="he / en / auto")
    engine: Literal["auto", "gemini", "groq", "local"] = "auto"
    prompt: str = Field("", description="רמז הקשר לשיפור דיוק, למשל שמות מוצרים")
    return_mp3: bool = True
    return_segments: bool = True
    callback_url: str = Field("", description="Webhook שיקבל POST עם התוצאה בסיום")


# ----------------------------------------------------------------------------
# Endpoints
# ----------------------------------------------------------------------------
@app.get("/")
def root():
    return {
        "service": "ACR Transcribe API",
        "status": "ok",
        "engine_default": pick_engine("auto"),
        "engines_available": {
            "gemini": bool(GEMINI_API_KEY),
            "groq": bool(GROQ_API_KEY),
            "local": True,
        },
        "auth": "X-API-Key header" if API_KEY else "open",
        "endpoints": {
            "POST /transcribe": "multipart file upload -> transcript (sync)",
            "POST /transcribe-url": "JSON {url} (Drive/HTTP) -> transcript (sync)",
            "POST /jobs": "JSON {url, callback_url} -> async job id",
            "GET  /jobs/{id}": "async job status/result",
            "POST /convert": "multipart file upload -> MP3 only",
            "GET  /download/{id}.mp3": "fetch converted MP3",
            "GET  /health": "healthcheck",
        },
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {
        "ok": True,
        "ffmpeg": bool(shutil.which("ffmpeg")),
        "engine": pick_engine("auto"),
        "gemini": bool(GEMINI_API_KEY),
        "groq": bool(GROQ_API_KEY),
    }


@app.post("/transcribe")
async def transcribe_upload(
    file: UploadFile = File(...),
    language: str = Form(DEFAULT_LANGUAGE),
    engine: str = Form("auto"),
    prompt: str = Form(""),
    return_mp3: bool = Form(True),
    return_segments: bool = Form(True),
    x_api_key: Optional[str] = Header(None),
):
    check_key(x_api_key)
    job_id = uuid.uuid4().hex[:16]
    suffix = Path(file.filename or "audio.acr").suffix or ".acr"
    src = DATA_DIR / f"{job_id}_src{suffix}"

    size = 0
    with open(src, "wb") as f:
        while True:
            chunk = await file.read(1 << 20)
            if not chunk:
                break
            size += len(chunk)
            if size > MAX_UPLOAD_MB * 1024 * 1024:
                src.unlink(missing_ok=True)
                raise HTTPException(413, f"File exceeds {MAX_UPLOAD_MB}MB")
            f.write(chunk)

    try:
        return await process(src, job_id, language, engine, prompt, return_mp3, return_segments)
    finally:
        src.unlink(missing_ok=True)


@app.post("/convert")
async def convert_only(
    file: UploadFile = File(...),
    bitrate: str = Form(MP3_BITRATE),
    x_api_key: Optional[str] = Header(None),
):
    """המרה בלבד — מהיר וזול, בלי תמלול."""
    check_key(x_api_key)
    job_id = uuid.uuid4().hex[:16]
    src = DATA_DIR / f"{job_id}_src{Path(file.filename or 'a.acr').suffix}"
    src.write_bytes(await file.read())
    mp3 = to_mp3(src, DATA_DIR / f"{job_id}.mp3", bitrate)
    src.unlink(missing_ok=True)
    return FileResponse(mp3, media_type="audio/mpeg",
                        filename=f"{Path(file.filename or 'audio').stem}.mp3")


@app.post("/transcribe-url")
async def transcribe_url(req: JobRequest, x_api_key: Optional[str] = Header(None)):
    check_key(x_api_key)
    job_id = uuid.uuid4().hex[:16]
    src = DATA_DIR / f"{job_id}_src.bin"
    await download_to(req.url, src, req.drive_access_token)
    try:
        return await process(src, job_id, req.language, req.engine,
                             req.prompt, req.return_mp3, req.return_segments)
    finally:
        src.unlink(missing_ok=True)


async def _run_job(job_id: str, req: JobRequest):
    src = DATA_DIR / f"{job_id}_src.bin"
    try:
        JOBS[job_id]["status"] = "processing"
        await download_to(req.url, src, req.drive_access_token)
        JOBS[job_id] = await process(src, job_id, req.language, req.engine,
                                     req.prompt, req.return_mp3, req.return_segments)
    except Exception as e:
        log.exception("job %s failed", job_id)
        JOBS[job_id] = {"job_id": job_id, "status": "error", "error": str(e)}
    finally:
        src.unlink(missing_ok=True)

    if req.callback_url:
        try:
            async with httpx.AsyncClient(timeout=60) as c:
                await c.post(req.callback_url, json=JOBS[job_id])
        except Exception as e:
            log.warning("callback failed: %s", e)


@app.post("/jobs", status_code=202)
async def create_job(req: JobRequest, bg: BackgroundTasks, x_api_key: Optional[str] = Header(None)):
    """מצב אסינכרוני — מומלץ ל-CRM: מחזיר מיד job_id ושולח webhook בסיום."""
    check_key(x_api_key)
    job_id = uuid.uuid4().hex[:16]
    JOBS[job_id] = {"job_id": job_id, "status": "queued"}
    bg.add_task(_run_job, job_id, req)
    return {"job_id": job_id, "status": "queued", "poll": f"/jobs/{job_id}"}


@app.get("/jobs/{job_id}")
async def get_job(job_id: str, x_api_key: Optional[str] = Header(None)):
    check_key(x_api_key)
    if job_id not in JOBS:
        raise HTTPException(404, "Unknown job id")
    return JOBS[job_id]


@app.get("/download/{name}")
async def download(name: str, x_api_key: Optional[str] = Header(None), key: str = ""):
    # הקלטות שיחה הן מידע רגיש — ההורדה מוגנת כמו כל שאר הנתיבים.
    # אפשר להעביר את המפתח בכותרת X-API-Key או ב-?key= (לשימוש ב-<audio src>).
    check_key(x_api_key or key)
    if ".." in name or "/" in name:
        raise HTTPException(400, "Bad name")
    p = DATA_DIR / name
    if not p.exists():
        raise HTTPException(404, "Not found or expired")
    return FileResponse(p, media_type="audio/mpeg", filename=name)


# ----------------------------------------------------------------------------
# ניקוי קבצים ישנים (חוסך מקום דיסק)
# ----------------------------------------------------------------------------
async def cleaner():
    import time
    while True:
        await asyncio.sleep(300)
        cutoff = time.time() - KEEP_MP3_MINUTES * 60
        for p in DATA_DIR.rglob("*"):
            try:
                if p.is_file() and p.stat().st_mtime < cutoff:
                    p.unlink()
            except Exception:
                pass
        done = [k for k, v in JOBS.items() if v.get("status") in ("done", "error")]
        for jid in done[:-500]:
            JOBS.pop(jid, None)


@app.on_event("startup")
async def startup():
    if not shutil.which("ffmpeg"):
        log.error("ffmpeg NOT FOUND — conversion will fail")
    asyncio.create_task(cleaner())
    engine = pick_engine("auto")
    log.info("Ready. Engine=%s%s", engine,
             f" ({GEMINI_MODEL})" if engine == "gemini" else "")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "7860")))
