"""
Fish S2 Pro — Local WebUI Backend
"""
import asyncio
import json
import os
import shutil
import subprocess
import sys
import tempfile
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

# Force offline mode to prevent HuggingFace Hub from making slow network checks that hang the server
os.environ["HF_HUB_OFFLINE"] = "1"

import numpy as np
import soundfile as sf
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

# ── Paths ────────────────────────────────────────────────────────────────────
BASE     = Path(__file__).parent
OUTPUTS  = BASE / "outputs"
VOICES   = BASE / "voices"
OUTPUTS.mkdir(exist_ok=True)
VOICES.mkdir(exist_ok=True)

# ── Model state ───────────────────────────────────────────────────────────────
model        = None
model_status = "loading"   # "loading" | "ready" | "error"
model_error  = None
_model_task  = None        # Strong reference to prevent garbage collection


async def _load_model_bg():
    """Load model in background so the server binds to the port immediately."""
    global model, model_status, model_error
    loop = asyncio.get_event_loop()
    try:
        def _load():
            import mlx.core as mx
            if hasattr(mx, "set_cache_limit"):
                # Enable a healthy 4GB allocator cache for GPU buffer reuse while preventing unbounded growth
                mx.set_cache_limit(4 * 1024 * 1024 * 1024)
            import mlx_speech
            m = mlx_speech.tts.load("fish-s2-pro")
            mx.clear_cache()
            return m
        model = await loop.run_in_executor(None, _load)
        model_status = "ready"
        print("\n✅  Fish S2 Pro loaded and ready")
    except Exception as e:
        model_status = "error"
        model_error  = str(e)
        print(f"\n❌  Model load failed: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model_task
    # Fire-and-forget: model loads while server is already accepting requests
    _model_task = asyncio.create_task(_load_model_bg())
    yield


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="Fish S2 Pro WebUI", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _require_model():
    if model_status == "loading":
        raise HTTPException(503, "Model is still loading — please wait")
    if model_status == "error":
        raise HTTPException(503, f"Model failed to load: {model_error}")


def _postprocess(waveform, sample_rate, trim: bool, normalize: bool):
    if trim:
        frame = int(sample_rate * 0.010)
        frames = [waveform[i:i+frame] for i in range(0, len(waveform)-frame, frame)]
        energies = [np.mean(np.abs(f)) for f in frames]
        threshold = 0.01
        start = next((i * frame for i, e in enumerate(energies) if e > threshold), 0)
        end   = next(((i+1)*frame for i, e in reversed(list(enumerate(energies))) if e > threshold), len(waveform))
        waveform = waveform[start:end]
    if normalize:
        peak = max(abs(waveform.max()), abs(waveform.min()))
        if peak > 0:
            waveform = waveform * (0.95 / peak)
    return waveform


def _save_output(waveform, sample_rate) -> dict:
    filename = f"{uuid.uuid4().hex[:12]}.wav"
    path = OUTPUTS / filename
    sf.write(str(path), waveform, sample_rate)
    duration = round(len(waveform) / sample_rate, 2)
    return {"filename": filename, "url": f"/outputs/{filename}", "duration": duration, "sample_rate": sample_rate}


def _read_voice_meta(voice_dir: Path) -> dict | None:
    meta_path = voice_dir / "meta.json"
    if not meta_path.exists():
        return None
    try:
        return json.loads(meta_path.read_text())
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Status
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/status")
async def get_status():
    return {
        "status":       model_status,
        "error":        model_error,
        "output_count": len(list(OUTPUTS.glob("*.wav"))),
        "voice_count":  len([d for d in VOICES.iterdir() if d.is_dir()]),
    }


@app.post("/api/shutdown")
async def shutdown():
    import signal
    
    async def _kill_later():
        await asyncio.sleep(0.5)
        print("\n🛑 Process terminated.")
        os.kill(os.getpid(), signal.SIGINT)
        
    asyncio.create_task(_kill_later())
    print("\n🛑 Shutdown requested via WebUI. Stopping server in 0.5 seconds...")
    return {"status": "shutting down"}


# ─────────────────────────────────────────────────────────────────────────────
# TTS — Generate
# ─────────────────────────────────────────────────────────────────────────────

def _generate_with_chunks(text: str, reference_audio: str | None, reference_text: str | None, max_new_tokens: int) -> tuple[np.ndarray, int]:
    """Highly optimized sentence-chunked generation using MLX GPU allocator cache controls."""
    import mlx.core as mx
    import re
    if hasattr(mx, "set_cache_limit"):
        # Limit cache to 2GB to prevent excessive RAM/VRAM consumption on lower-end Macs
        mx.set_cache_limit(2 * 1024 * 1024 * 1024)
        
    # Split text into sentences to prevent token overflow & memory spikes
    # Split by sentence endings (. ! ? \n) but keep the punctuation attached
    raw_sentences = re.split(r'([.!?\n]+)', text)
    chunks = []
    current_chunk = ""
    for item in raw_sentences:
        if not item:
            continue
        if item in (".", "!", "?", "\n"):
            current_chunk += item
            chunks.append(current_chunk.strip())
            current_chunk = ""
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = item
    if current_chunk:
        chunks.append(current_chunk.strip())
        
    chunks = [c for c in chunks if c]
    
    # If the text is short enough or there's only one chunk, generate directly
    if len(text) < 120 or len(chunks) <= 1:
        res = model.generate(
            text,
            reference_audio=reference_audio,
            reference_text=reference_text,
            max_new_tokens=max_new_tokens
        )
        mx.eval(res.waveform)
        waveform_np = np.array(res.waveform)
        mx.clear_cache()
        return waveform_np, res.sample_rate

    # Generate each chunk sequentially and concatenate them
    waveforms = []
    sample_rate = 44100
    for idx, chunk in enumerate(chunks):
        print(f"🎙 Generating chunk {idx+1}/{len(chunks)}: {chunk[:40]}...")
        # Keep token limit bounded per sentence chunk for maximum speed
        chunk_tokens = min(512, max_new_tokens)
        res = model.generate(
            chunk,
            reference_audio=reference_audio,
            reference_text=reference_text,
            max_new_tokens=chunk_tokens
        )
        mx.eval(res.waveform)
        waveforms.append(np.array(res.waveform))
        mx.clear_cache()
        
    if not waveforms:
        raise Exception("No speech was generated by the model")
        
    waveform_np = np.concatenate(waveforms)
    return waveform_np, sample_rate


@app.post("/api/generate")
async def generate(
    text:           str        = Form(...),
    max_new_tokens: int        = Form(1024),
    trim_silence:   bool       = Form(False),
    normalize:      bool       = Form(False),
    voice_id:       str | None = Form(None),
):
    _require_model()

    reference_audio = None
    reference_text = None
    if voice_id and voice_id not in ("none", "default", "custom"):
        voice_dir = VOICES / voice_id
        if not voice_dir.exists():
            raise HTTPException(404, f"Voice preset {voice_id} not found")
        meta = _read_voice_meta(voice_dir) or {}
        reference_audio = str(voice_dir / "audio.wav")
        reference_text = meta.get("reference_text", "")

    loop = asyncio.get_event_loop()

    try:
        waveform_np, sample_rate = await loop.run_in_executor(
            None, _generate_with_chunks, text, reference_audio, reference_text, max_new_tokens
        )
    except Exception as e:
        raise HTTPException(500, f"Generation error: {e}")

    waveform = _postprocess(waveform_np, sample_rate, trim_silence, normalize)
    return _save_output(waveform, sample_rate)


# ─────────────────────────────────────────────────────────────────────────────
# Voice Clone — Generate with reference
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/api/clone")
async def clone(
    text:            str        = Form(...),
    reference_text:  str        = Form(...),
    voice_id:        str | None = Form(None),      # saved voice from library
    reference_audio: UploadFile | None = File(None),  # or new upload
    max_new_tokens:  int        = Form(1024),
    trim_silence:    bool       = Form(False),
    normalize:       bool       = Form(False),
):
    _require_model()

    # Resolve reference audio path
    if voice_id:
        voice_dir = VOICES / voice_id
        audio_path = voice_dir / "audio.wav"
        if not audio_path.exists():
            raise HTTPException(404, "Voice not found in library")
        # Use saved reference_text from meta if caller didn't provide one
        if not reference_text:
            meta = _read_voice_meta(voice_dir)
            reference_text = (meta or {}).get("reference_text", "")
        tmp_path = None
    elif reference_audio:
        suffix = Path(reference_audio.filename).suffix or ".wav"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(await reference_audio.read())
            tmp_path = tmp.name
        audio_path = Path(tmp_path)
    else:
        raise HTTPException(400, "Provide either voice_id or reference_audio")

    loop = asyncio.get_event_loop()

    try:
        waveform_np, sample_rate = await loop.run_in_executor(
            None, _generate_with_chunks, text, str(audio_path), reference_text, max_new_tokens
        )
    except Exception as e:
        raise HTTPException(500, f"Clone error: {e}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

    waveform = _postprocess(waveform_np, sample_rate, trim_silence, normalize)
    return _save_output(waveform, sample_rate)


# ─────────────────────────────────────────────────────────────────────────────
# Transcription
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/api/transcribe")
async def transcribe(
    audio: UploadFile = File(None),
    voice_id: str = Form(None)
):
    if voice_id:
        audio_path = str(VOICES / voice_id / "audio.wav")
        if not os.path.exists(audio_path):
            raise HTTPException(404, "Voice audio not found")
        tmp_path = None
    elif audio:
        suffix = Path(audio.filename).suffix or ".wav"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(await audio.read())
            tmp_path = tmp.name
        audio_path = tmp_path
    else:
        raise HTTPException(400, "Provide either voice_id or audio")

    script = (
        "import os, mlx_whisper, json, sys, mlx.core as mx; "
        "os.environ['HF_HUB_OFFLINE'] = '1'; "
        "r = mlx_whisper.transcribe(sys.argv[1], "
        "path_or_hf_repo='mlx-community/whisper-tiny'); "
        "mx.clear_cache(); "
        "print(json.dumps(r['text'].strip()))"
    )
    try:
        loop = asyncio.get_event_loop()
        def _run():
            python_path = "/Users/asifgondal/.gemini/antigravity/scratch/Voice-Clone-Studio/venv/bin/python"
            return subprocess.run(
                [python_path, "-c", script, audio_path],
                capture_output=True, text=True, timeout=120
            )
        proc = await loop.run_in_executor(None, _run)
        if proc.returncode != 0:
            err = proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else "unknown error"
            print(f"❌ Transcription subprocess error: {proc.stderr}")
            raise HTTPException(500, f"Transcription error: {err}")
        text = json.loads(proc.stdout.strip())
        return {"text": text}
    except subprocess.TimeoutExpired:
        raise HTTPException(504, "Transcription timed out (>120s)")
    except Exception as e:
        print(f"❌ Transcription error: {e}")
        raise HTTPException(500, f"Transcription error: {e}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


# ─────────────────────────────────────────────────────────────────────────────
# Voice Library Auto-Importer & Backend
# ─────────────────────────────────────────────────────────────────────────────

def _auto_import_voices():
    """Scan both VOICES and the external Downloads folder for new raw WAV/MP3 presets."""
    # 1. Gather all existing imported original filenames to avoid duplicates
    existing_imported = set()
    for d in VOICES.iterdir():
        if d.is_dir():
            meta = _read_voice_meta(d)
            if meta and meta.get("original_file"):
                existing_imported.add(meta["original_file"])

    # 2. Paths to scan
    paths_to_scan = [
        VOICES,
        Path("/Users/asifgondal/Downloads/voice presets speech")
    ]

    for scan_dir in paths_to_scan:
        if not scan_dir.exists() or not scan_dir.is_dir():
            continue
        
        for file_path in scan_dir.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in (".wav", ".mp3"):
                # Avoid duplicates
                if file_path.name in existing_imported:
                    continue
                
                # Clean name: if it looks like a hex string, truncate it for neatness
                stem = file_path.stem
                is_hex = len(stem) > 12 and all(c in "0123456789abcdefABCDEF" for c in stem[:12])
                if is_hex:
                    name = f"Preset {stem[:6].upper()}"
                else:
                    name = stem.replace("_", " ").replace("-", " ").title()
                
                slug = stem.lower().replace(" ", "_").replace("-", "_")
                voice_id = f"preset_{slug}"
                voice_dir = VOICES / voice_id
                
                # Ensure unique directory name
                counter = 1
                while voice_dir.exists():
                    voice_dir = VOICES / f"preset_{slug}_{counter}"
                    counter += 1
                    
                voice_dir.mkdir(exist_ok=True)
                dest_audio = voice_dir / "audio.wav"
                
                try:
                    # Read original audio and save as clean WAV
                    data, sr = sf.read(str(file_path))
                    sf.write(str(dest_audio), data, sr)
                    duration = round(len(data) / sr, 2)
                    
                    # If it was in the local VOICES folder, we can delete the raw file.
                    # If it was in the external Downloads folder, we LEAVE it alone (safe copy)!
                    if scan_dir == VOICES:
                        file_path.unlink()
                    
                    # Create structured meta.json
                    meta = {
                        "name": name,
                        "reference_text": "Transcribing audio...",
                        "notes": f"Auto-imported preset from {scan_dir.name}",
                        "duration": duration,
                        "sample_rate": sr,
                        "created_at": datetime.now().isoformat(),
                        "original_file": file_path.name
                    }
                    
                    meta_path = voice_dir / "meta.json"
                    with open(meta_path, "w", encoding="utf-8") as f:
                        json.dump(meta, f, indent=4)
                        
                    # Add to our local set to prevent duplicate imports in the same loop
                    existing_imported.add(file_path.name)

                    # Schedule auto-transcription in the background so it doesn't block the API
                    asyncio.create_task(_transcribe_preset_background(voice_dir.name, str(dest_audio)))
                    print(f"📥 Successfully imported preset: {name} ({file_path.name})")
                except Exception as e:
                    print(f"❌ Failed to auto-import voice {file_path.name}: {e}")

async def _transcribe_preset_background(voice_id: str, audio_path: str):
    """Run Whisper to transcribe the auto-imported voice preset in the background."""
    script = (
        "import os, mlx_whisper, json, sys, mlx.core as mx; "
        "os.environ['HF_HUB_OFFLINE'] = '1'; "
        "r = mlx_whisper.transcribe(sys.argv[1], "
        "path_or_hf_repo='mlx-community/whisper-tiny'); "
        "mx.clear_cache(); "
        "print(json.dumps(r['text'].strip()))"
    )
    python_path = "/Users/asifgondal/.gemini/antigravity/scratch/Voice-Clone-Studio/venv/bin/python"
    try:
        loop = asyncio.get_event_loop()
        def _run():
            return subprocess.run(
                [python_path, "-c", script, audio_path],
                capture_output=True, text=True, timeout=120
            )
        proc = await loop.run_in_executor(None, _run)
        if proc.returncode == 0:
            text = json.loads(proc.stdout.strip())
            meta_path = VOICES / voice_id / "meta.json"
            if meta_path.exists():
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                meta["reference_text"] = text
                with open(meta_path, "w", encoding="utf-8") as f:
                    json.dump(meta, f, indent=4)
                print(f"📝 Auto-transcribed voice preset '{meta['name']}': {text}")
        else:
            print(f"❌ Auto-transcription failed for {voice_id}: {proc.stderr}")
    except Exception as e:
        print(f"❌ Background transcription error for {voice_id}: {e}")


@app.get("/api/voices")
async def list_voices():
    # Scan and import any newly dropped presets before listing
    _auto_import_voices()
    
    voices = []
    for d in sorted(VOICES.iterdir()):
        if not d.is_dir():
            continue
        meta = _read_voice_meta(d)
        if meta is None:
            continue
        audio_path = d / "audio.wav"
        meta["id"]        = d.name
        meta["audio_url"] = f"/voice-audio/{d.name}"
        meta["has_audio"] = audio_path.exists()
        voices.append(meta)
    voices.sort(key=lambda v: v.get("created_at", ""), reverse=True)
    return voices


@app.post("/api/voices/save-from-output")
async def save_voice_from_output(
    name:           str  = Form(...),
    filename:       str  = Form(...),
    reference_text: str  = Form(...),
    notes:          str  = Form(""),
):
    output_path = OUTPUTS / filename
    if not output_path.exists():
        raise HTTPException(404, "Generated audio file not found")
        
    voice_id  = uuid.uuid4().hex[:12]
    voice_dir = VOICES / voice_id
    voice_dir.mkdir()
    
    dest_path = voice_dir / "audio.wav"
    try:
        shutil.copy(str(output_path), str(dest_path))
        data, sr = sf.read(str(dest_path))
        duration = round(len(data) / sr, 2)
    except Exception as e:
        shutil.rmtree(voice_dir, ignore_errors=True)
        raise HTTPException(400, f"Could not save voice file: {e}")
        
    meta = {
        "name":           name,
        "reference_text": reference_text,
        "notes":          notes,
        "duration":       duration,
        "sample_rate":    sr,
        "created_at":     datetime.now().isoformat(),
        "original_file":  filename,
    }
    
    meta_path = voice_dir / "meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=4)
        
    return {"status": "success", "voice_id": voice_id}


@app.post("/api/voices")
async def save_voice(
    name:           str        = Form(...),
    reference_text: str        = Form(""),
    notes:          str        = Form(""),
    audio:          UploadFile = File(...),
):
    voice_id  = uuid.uuid4().hex[:12]
    voice_dir = VOICES / voice_id
    voice_dir.mkdir()

    # Convert uploaded audio to WAV and save
    suffix = Path(audio.filename).suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await audio.read())
        tmp_path = tmp.name

    try:
        data, sr = sf.read(tmp_path)
        out_path = voice_dir / "audio.wav"
        sf.write(str(out_path), data, sr)
        duration = round(len(data) / sr, 2)
    except Exception as e:
        shutil.rmtree(voice_dir, ignore_errors=True)
        raise HTTPException(400, f"Could not read audio file: {e}")
    finally:
        os.unlink(tmp_path)

    meta = {
        "name":           name,
        "reference_text": reference_text,
        "notes":          notes,
        "duration":       duration,
        "sample_rate":    sr,
        "created_at":     datetime.now().isoformat(),
        "original_file":  audio.filename,
    }
    (voice_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    return {"id": voice_id, "audio_url": f"/voice-audio/{voice_id}", **meta}


@app.patch("/api/voices/{voice_id}")
async def update_voice(
    voice_id:       str,
    name:           str = Form(None),
    reference_text: str = Form(None),
    notes:          str = Form(None),
):
    voice_dir = VOICES / voice_id
    meta = _read_voice_meta(voice_dir)
    if meta is None:
        raise HTTPException(404, "Voice not found")
    if name           is not None: meta["name"]           = name
    if reference_text is not None: meta["reference_text"] = reference_text
    if notes          is not None: meta["notes"]          = notes
    meta["updated_at"] = datetime.now().isoformat()
    (voice_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return {**meta, "id": voice_id}


@app.delete("/api/voices/{voice_id}")
async def delete_voice(voice_id: str):
    voice_dir = VOICES / voice_id
    if not voice_dir.exists():
        raise HTTPException(404, "Voice not found")
    shutil.rmtree(voice_dir)
    return {"deleted": voice_id}


@app.get("/voice-audio/{voice_id}")
async def get_voice_audio(voice_id: str):
    path = VOICES / voice_id / "audio.wav"
    if not path.exists():
        raise HTTPException(404, "Voice audio not found")
    return FileResponse(str(path), media_type="audio/wav")


# ─────────────────────────────────────────────────────────────────────────────
# Outputs management
# ─────────────────────────────────────────────────────────────────────────────

@app.delete("/api/outputs/{filename}")
async def delete_output(filename: str):
    path = OUTPUTS / filename
    if not path.exists() or path.parent != OUTPUTS:
        raise HTTPException(404, "File not found")
    path.unlink()
    return {"deleted": filename}


@app.delete("/api/outputs")
async def clear_outputs():
    count = 0
    for f in OUTPUTS.glob("*.wav"):
        f.unlink()
        count += 1
    return {"deleted": count}


# ─────────────────────────────────────────────────────────────────────────────
# Frontend
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root():
    return HTMLResponse((BASE / "index.html").read_text())

app.mount("/outputs", StaticFiles(directory=str(OUTPUTS)), name="outputs")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8765, log_level="warning")
