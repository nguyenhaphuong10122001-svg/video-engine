import os, re, shutil, subprocess, textwrap, threading, time, uuid, urllib.parse
from concurrent.futures import ThreadPoolExecutor
from typing import Literal

import requests
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
from gtts import gTTS
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel, Field

TOKEN = os.environ.get("VIDEO_RENDER_TOKEN", "")
FFMPEG = os.environ.get("FFMPEG_BIN", "ffmpeg")
FONT = os.environ.get("FONT_FILE", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
DATA = os.environ.get("DATA_DIR", "/tmp/jobs")
SIZES = {"9:16": (720, 1280), "16:9": (1280, 720), "1:1": (960, 960)}
os.makedirs(DATA, exist_ok=True)

app = FastAPI(title="n8n Cloud external FFmpeg video engine", version="2.0.0")
pool = ThreadPoolExecutor(max_workers=1)
jobs: dict = {}
lock = threading.Lock()


class Scene(BaseModel):
    visual_prompt: str = Field(min_length=5, max_length=1800)
    narration: str = Field(min_length=4, max_length=1200)
    caption: str = Field(min_length=1, max_length=140)
    seconds: int = Field(ge=4, le=25)


class RenderRequest(BaseModel):
    title: str = Field(min_length=1, max_length=160)
    language: str = Field(default="ru", max_length=12)
    ratio: Literal["9:16", "16:9", "1:1"] = "9:16"
    target_seconds: int = Field(default=30, ge=12, le=75)
    scenes: list[Scene] = Field(min_length=2, max_length=6)


def check_token(authorization, x_render_token):
    got = x_render_token or ""
    if not got and authorization:
        got = re.sub(r"^Bearer\s+", "", authorization, flags=re.I)
    if not TOKEN or got != TOKEN:
        raise HTTPException(status_code=401, detail="Invalid renderer token")


def base_url():
    return (os.environ.get("PUBLIC_BASE_URL") or os.environ.get("RENDER_EXTERNAL_URL") or "").rstrip("/")


def run(cmd):
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError((p.stderr or "")[-600:])
    return p.stdout


def duration_of(path):
    p = subprocess.run([FFMPEG, "-i", path], capture_output=True, text=True)
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.?\d*)", p.stderr)
    if not m:
        raise RuntimeError("cannot read duration")
    return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))


def make_image(prompt, size, path, seed):
    w, h = size
    url = ("https://image.pollinations.ai/prompt/" + urllib.parse.quote(prompt[:900]) +
           f"?width={w}&height={h}&nologo=true&seed={seed}")
    for attempt in range(3):
        try:
            r = requests.get(url, timeout=90)
            if r.status_code == 200 and r.headers.get("content-type", "").startswith("image"):
                with open(path, "wb") as f:
                    f.write(r.content)
                Image.open(path).verify()
                return "ai"
        except Exception:
            pass
        time.sleep(4 * (attempt + 1))
    img = Image.new("RGB", size, (20, 30, 60))
    d = ImageDraw.Draw(img)
    for y in range(h):
        c = int(20 + 90 * y / h)
        d.line([(0, y), (w, y)], fill=(c, c // 2 + 30, 90 + c // 2))
    img.save(path, "JPEG")
    return "fallback"


def burn_caption(path, text, size):
    w, h = size
    im = Image.open(path).convert("RGB").resize(size)
    d = ImageDraw.Draw(im)
    fs = max(28, w // 18)
    try:
        font = ImageFont.truetype(FONT, fs)
    except Exception:
        font = ImageFont.load_default()
    lines = textwrap.wrap(text, 22 if w < h else 40) or [text]
    lh = fs + 10
    y = int(h * 0.72)
    for ln in lines:
        tw = d.textlength(ln, font=font)
        d.text(((w - tw) / 2, y), ln, font=font, fill="white", stroke_width=3, stroke_fill="black")
        y += lh
    im.save(path, "JPEG", quality=90)


def make_voice(text, lang, path):
    last = None
    for attempt in range(3):
        try:
            gTTS(text=text, lang=lang if lang in ("ru", "vi", "en") else "en").save(path)
            return
        except Exception as e:
            last = e
            time.sleep(3)
    raise RuntimeError(f"TTS failed: {last}")


def render(job_id, req: RenderRequest):
    job = jobs[job_id]
    job["status"] = "running"
    work = os.path.join(DATA, job_id)
    os.makedirs(work, exist_ok=True)
    try:
        size = SIZES[req.ratio]
        w, h = size
        clips = []
        total = len(req.scenes)
        notes = []
        for i, s in enumerate(req.scenes):
            job["progress"] = f"scene {i + 1}/{total}"
            img, mp3 = f"{work}/s{i}.jpg", f"{work}/s{i}.mp3"
            notes.append(make_image(s.visual_prompt, size, img, seed=i + 7))
            make_voice(s.narration, req.language, mp3)
            dur = max(float(s.seconds), duration_of(mp3) + 0.4)
            burn_caption(img, s.caption, size)
            vf = f"scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h},format=yuv420p"
            out = f"{work}/clip{i}.mp4"
            run([FFMPEG, "-y", "-loop", "1", "-framerate", "25", "-i", img, "-i", mp3,
                 "-t", f"{dur:.2f}", "-vf", vf, "-af", "apad,aresample=44100", "-ac", "2",
                 "-c:v", "libx264", "-preset", "veryfast", "-crf", "26", "-r", "25",
                 "-c:a", "aac", "-b:a", "128k", out])
            clips.append(out)
        job["progress"] = "joining"
        lst = f"{work}/list.txt"
        with open(lst, "w") as f:
            for c in clips:
                f.write(f"file '{c}'\n")
        final = f"{work}/video.mp4"
        run([FFMPEG, "-y", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy",
             "-movflags", "+faststart", final])
        job["duration"] = round(duration_of(final), 1)
        job["video_url"] = f"{base_url()}/videos/{job_id}/video.mp4"
        job["image_source"] = notes
        job["status"] = "succeeded"
        job["progress"] = "done"
        for fn in os.listdir(work):
            if fn != "video.mp4":
                os.remove(os.path.join(work, fn))
    except Exception as e:
        job["status"] = "failed"
        job["error"] = str(e)[:800]


def cleanup_old():
    now = time.time()
    for name in os.listdir(DATA):
        p = os.path.join(DATA, name)
        if os.path.isdir(p) and now - os.path.getmtime(p) > 6 * 3600:
            shutil.rmtree(p, ignore_errors=True)
            jobs.pop(name, None)


@app.get("/health")
def health():
    return {"status": "ok", "ffmpeg": shutil.which(FFMPEG) is not None, "token_set": bool(TOKEN)}


@app.post("/jobs", status_code=202)
def create_job(req: RenderRequest, authorization: str | None = Header(default=None),
               x_render_token: str | None = Header(default=None)):
    check_token(authorization, x_render_token)
    cleanup_old()
    job_id = uuid.uuid4().hex
    with lock:
        jobs[job_id] = {"job_id": job_id, "status": "queued", "progress": "queued",
                        "title": req.title, "video_url": None, "duration": None, "error": None}
    pool.submit(render, job_id, req)
    return jobs[job_id]


@app.get("/jobs/{job_id}")
def status(job_id: str, authorization: str | None = Header(default=None),
           x_render_token: str | None = Header(default=None)):
    check_token(authorization, x_render_token)
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found (server may have restarted)")
    return jobs[job_id]


@app.get("/videos/{job_id}/{filename}")
def download(job_id: str, filename: str):
    if not re.fullmatch(r"[0-9a-f]{32}", job_id) or filename != "video.mp4":
        raise HTTPException(status_code=404, detail="Not found")
    path = os.path.join(DATA, job_id, filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Not found")
    return FileResponse(path, media_type="video/mp4", filename=f"{job_id}.mp4")
