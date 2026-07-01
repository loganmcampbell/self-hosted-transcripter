#!/usr/bin/env python3
"""
Record mic until ENTER or transcribe an existing audio file with Whisper.
ALL artifacts go into: <audio_name>-<uuid>

Artifacts:
- <audio>-<uuid>.log        sentence-level log (timestamps)
- <audio>-<uuid>.txt        full transcript (no speakers)
- <audio>-<uuid>.json       metadata + whisper segments
- <audio>-<uuid>.wav        only when recording from mic
- (optional diarization)
  - <audio>-<uuid>-diarization.json   diarization turns + mapping
  - <audio>-<uuid>-diarized.txt       speaker-tagged plain text
  - <audio>-<uuid>-diarized.srt       speaker-tagged subtitles
  - <audio>-<uuid>-diarized.vtt       speaker-tagged subtitles

Deps:
  pip install sounddevice soundfile numpy torch
  pip install git+https://github.com/openai/whisper.git
  pip install pyannote.audio==3.* typing_extensions
Env:
  set HUGGINGFACE_TOKEN (required for diarization)
"""

import json
import logging
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
import warnings

# Suppress known harmless warnings from pyannote and torchaudio
warnings.filterwarnings("ignore", message="std\\(\\).*degrees of freedom")
warnings.filterwarnings("ignore", message=".*MPEG_LAYER_III subtype is unknown")
from datetime import datetime, UTC
from pathlib import Path
from typing import Any

import numpy as np
import sounddevice as sd
import soundfile as sf
import torch
import torch.version  # explicit submodule import so PyCharm resolves torch.version.cuda
import whisper
from colorama import Fore, Style, init as colorama_init

colorama_init(autoreset=True)


def c(text: str, color: str) -> str:
    return f"{color}{text}{Style.RESET_ALL}"


def info(text: str) -> None:
    print(c(text, Fore.CYAN))


def ok(text: str) -> None:
    print(c(text, Fore.GREEN))


def warn(text: str) -> None:
    print(c(text, Fore.YELLOW))


def err(text: str) -> None:
    print(c(text, Fore.RED))


def banner(title: str, color: str = Fore.CYAN, width: int = 60) -> None:
    line = "=" * width
    print(c(line, color))
    print(c(title.center(width), color + Style.BRIGHT))
    print(c(line, color))


# Put your Hugging Face token here if you don't want to use env vars (do NOT commit a real token):
HUGGINGFACE_TOKEN = "hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"  # <-- replace with your token, or leave and set HUGGINGFACE_TOKEN/HF_TOKEN env var instead

# Optional (speaker diarization)
ENABLE_DIARIZATION = True
DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"

# -------------- Config --------------
DEFAULT_AUDIO = "example.mp3"
AUDIO_ONLY_EXTENSIONS = (".mp3", ".wav", ".flac", ".m4a", ".ogg", ".aac", ".wma")
# Containers whose audio track gets extracted to WAV before transcription/diarization,
# since pyannote's loader doesn't reliably read video containers the way ffmpeg/Whisper do.
VIDEO_EXTENSIONS = (".mp4", ".flv", ".webm", ".mkv", ".mov", ".avi", ".wmv", ".m4v", ".ts")
AUDIO_EXTENSIONS = AUDIO_ONLY_EXTENSIONS + VIDEO_EXTENSIONS
EXTRACT_SAMPLE_RATE = 16000  # matches Whisper/pyannote's internal sample rate
PREF_SAMPLE_RATES = [16000, 48000, 44100, 32000, 22050]
CHANNELS = 1
SUBTYPE = "PCM_16"
MODEL_NAME = "large-v3"  # tiny/base/small/medium/large
FORCE_LANGUAGE = "en"  # e.g., "en" or None to auto-detect
USE_SENTENCE_SPLIT = True
USE_TEMPERATURE_FALLBACK = True
# Whisper anti-hallucination: conditioning on previous text causes repetition loops.
# Set CONDITION_ON_PREVIOUS_TEXT=False to fully break the feedback loop.
# dedup_transcript() strips any remaining repeated phrases from the saved text.
CONDITION_ON_PREVIOUS_TEXT = False
# ------------------------------------
PROCESSED_DB = ".processed.json"


# -------- Helpers --------
def _processed_db_path(base_dir: Path) -> Path:
    return base_dir / PROCESSED_DB


def _load_processed_db(base_dir: Path) -> dict:
    p = _processed_db_path(base_dir)
    if p.exists():
        try:
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError, ValueError):
            return {}
    return {}


def _save_processed_db(base_dir: Path, data: dict) -> None:
    p = _processed_db_path(base_dir)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _has_prior_outputs(base_dir: Path, audio_stem: str) -> bool:
    """Return True if any output dir for this audio stem already contains a transcript."""
    for entry in base_dir.iterdir():
        if not entry.is_dir() or not entry.name.startswith(f"{audio_stem}-"):
            continue
        for f in entry.glob("*.txt"):
            if f.name.startswith(audio_stem + "-"):
                return True
    return False


def _list_unprocessed_audio_candidates(base_dir: Path) -> list[str]:
    """Run cleanup then return audio files not yet in the processed manifest."""
    cleanup_old_runs(base_dir)

    all_audio = [f for f in os.listdir(base_dir) if f.lower().endswith(AUDIO_EXTENSIONS)]
    if not all_audio:
        return []

    db = _load_processed_db(base_dir)
    unprocessed = []
    for name in sorted(all_audio):
        abs_path = str((base_dir / name).resolve())
        stem = Path(name).stem
        already_marked = db.get(abs_path, {}).get("processed", False)
        if not (already_marked or _has_prior_outputs(base_dir, stem)):
            unprocessed.append(name)
    return unprocessed


def _mark_processed(base_dir: Path, audio_path: str, run_id: str) -> None:
    db = _load_processed_db(base_dir)
    db[str(Path(audio_path).resolve())] = {
        "processed": True,
        "run_id": run_id,
        "timestamp": datetime.now(UTC).isoformat(),
    }
    _save_processed_db(base_dir, db)


def cleanup_old_runs(base_dir: Path, log_name: str = "cleanup.log"):
    """Delete run directories (e.g. <audio>-<uuid>) older than the current ISO week."""
    now = datetime.now()
    current_year, current_week, _ = now.isocalendar()
    log_path = base_dir / log_name

    def log(message: str):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(log_path, "a", encoding="utf-8") as lf:
            lf.write(f"[{ts}] {message}\n")
        print(message)

    log(f"Starting cleanup (current week={current_week}, year={current_year})")

    for entry in base_dir.iterdir():
        if not entry.is_dir() or not re.match(r".+-[0-9a-fA-F\-]{8,}", entry.name):
            continue
        try:
            mtime = datetime.fromtimestamp(entry.stat().st_mtime)
            year, week, _ = mtime.isocalendar()
            if (year, week) < (current_year, current_week):
                log(f"Removing old run: {entry.name} (week={week}, year={year})")
                shutil.rmtree(entry, ignore_errors=True)
                log(f"Removed: {entry}")
        except OSError as e:
            log(f"Skipped {entry}: {e}")

    log("Cleanup complete.\n")


def _fmt_duration(seconds: float) -> str:
    """Return a human-readable duration string, using the largest sensible unit."""
    s = int(seconds)
    ms = round((seconds - s) * 1000)
    if s < 60:
        return f"{seconds:.2f}s"
    if s < 3600:
        m, sec = divmod(s, 60)
        return f"{m}m {sec:02d}s"
    if s < 86400:
        h, rem = divmod(s, 3600)
        m, sec = divmod(rem, 60)
        return f"{h}h {m:02d}m {sec:02d}s"
    d, rem = divmod(s, 86400)
    h, rem = divmod(rem, 3600)
    m, sec = divmod(rem, 60)
    return f"{d}d {h:02d}h {m:02d}m {sec:02d}s"


def sentence_split(text: str) -> list[str]:
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]


def dedup_transcript(text: str) -> str:
    """Remove consecutive repeated phrases that Whisper hallucinates.

    Handles:
    - Repeated sentences:  "Thank you. Thank you." → "Thank you."
    - Comma-separated runs: "oh, oh, oh" → "oh"
    - Space-separated runs: "you you you" → "you"
    - Mid-sentence loops:   "that's why it's not going to work [x3]" → once
    """
    # Pass 1: collapse repeated whole sentences
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    deduped: list[str] = []
    for s in sentences:
        if not s:
            continue
        if deduped and s.strip().lower() == deduped[-1].strip().lower():
            continue
        deduped.append(s)

    # Pass 2: within each chunk collapse intra-sentence repetitions
    result_parts = []
    for s in deduped:
        # comma-separated single word: "oh, oh, oh" → "oh"
        s = re.sub(r'\b(\w+)(,\s*\1)+\b', r'\1', s, flags=re.IGNORECASE)
        # space-separated single word: "you you you" → "you"
        s = re.sub(r'\b(\w+)(\s+\1)+\b', r'\1', s, flags=re.IGNORECASE)
        # multi-word phrase loops, longest first to avoid partial matches
        for n in range(10, 1, -1):
            wp = r"(?:\w+(?:'\w+)?[,]?\s+){" + str(n - 1) + r"}\w+(?:'\w+)?"
            s = re.sub(r'\b(' + wp + r')\s+(?:\1\s*)+', r'\1 ', s, flags=re.IGNORECASE)
        result_parts.append(s.strip())

    return ' '.join(result_parts)


def list_input_devices() -> list[int]:
    try:
        devices = sd.query_devices()
    except Exception as e:
        err(f"Could not query audio devices: {e}")
        return []
    idxs = []
    info("\nAvailable input devices:")
    for i, d in enumerate(devices):
        if d.get("max_input_channels", 0) > 0:
            idxs.append(i)
            name = d.get("name", f"Device {i}")
            host = d.get("hostapi", "")
            ch = d.get("max_input_channels", 0)
            default_sr = d.get("default_samplerate")
            dsr = f"{int(default_sr)} Hz" if default_sr else "n/a"
            print(f"  {c(f'[{i}]', Fore.MAGENTA)} {name}  | hostapi={host}  | in_ch={ch}  | default_sr={dsr}")
    if not idxs:
        warn("  (No input/mic devices found)")
    print()
    return idxs


def _get_default_input_device_index() -> int | None:
    dev = sd.default.device
    if isinstance(dev, (list, tuple)) and len(dev) >= 1:
        return dev[0] if dev[0] is not None and dev[0] >= 0 else None
    if isinstance(dev, int) and dev >= 0:
        return dev
    return None


def _pick_working_samplerate(device: int | None, channels: int) -> int:
    candidates = []
    try:
        query_dev = device if device is not None else _get_default_input_device_index()
        devinfo = sd.query_devices(query_dev, "input")
        if devinfo and devinfo.get("default_samplerate"):
            candidates.append(int(devinfo["default_samplerate"]))
    except Exception:
        pass
    for r in PREF_SAMPLE_RATES:
        if r not in candidates:
            candidates.append(r)
    last_err = None
    for sr in candidates:
        try:
            sd.check_input_settings(device=device, samplerate=sr, channels=channels, dtype="float32")
            return sr
        except Exception as e:
            last_err = e
    if last_err:
        raise last_err
    raise RuntimeError("Could not determine a working input samplerate")


def _json_safe_segment(seg: dict) -> dict:
    get = seg.get

    def f(x):
        return None if x is None else float(x)

    def i(x):
        try:
            return int(x)
        except Exception:
            return None

    out: dict[str, Any] = {
        "id": i(get("id")),
        "start": f(get("start", 0.0)),
        "end": f(get("end", 0.0)),
        "text": (get("text") or "").strip(),
        "avg_logprob": f(get("avg_logprob")),
        "compression_ratio": f(get("compression_ratio")),
        "no_speech_prob": f(get("no_speech_prob")),
        "temperature": f(get("temperature")),
    }
    words = get("words")
    if isinstance(words, list):
        out["words"] = [
            {
                "word": (w.get("word") or "").strip(),
                "start": f(w.get("start")),
                "end": f(w.get("end")),
                "probability": f(w.get("probability")),
            }
            for w in words if isinstance(w, dict)
        ]
    return out


# -------- Video/audio extraction --------
def _extract_audio_to_wav(src_path: Path, out_dir: Path, stem: str) -> Path:
    """Extract the audio track from a video (or re-encode any audio file) to 16kHz mono PCM16 WAV.

    WAV is used instead of MP3 because Whisper/pyannote resample to 16kHz mono internally
    anyway, and the source is often already lossy-compressed (e.g. FLV/WEBM) — re-encoding
    to MP3 would compress it a second time for no benefit.
    """
    out_path = out_dir / f"{stem}-extracted.wav"
    cmd = [
        "ffmpeg", "-y", "-i", str(src_path),
        "-vn", "-ac", "1", "-ar", str(EXTRACT_SAMPLE_RATE), "-acodec", "pcm_s16le",
        str(out_path),
    ]
    info(f"Extracting audio: {src_path.name} -> {out_path.name} ...")
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError:
        err("ffmpeg not found on PATH. Install ffmpeg (https://ffmpeg.org/) to convert video files.")
        raise SystemExit(2)
    if proc.returncode != 0 or not out_path.exists():
        err(f"ffmpeg failed to extract audio (exit {proc.returncode}):\n{proc.stderr[-1500:]}")
        raise SystemExit(2)
    ok(f"Extracted audio to: {out_path}")
    return out_path


# -------- Recording --------
def record_mic_until_enter(out_path: str | Path | None = None,
                           device: int | None = None,
                           channels: int = CHANNELS,
                           subtype: str = SUBTYPE,
                           out_dir: Path | None = None,
                           audio_name: str = "mic",
                           run_id: str | None = None) -> str:
    if out_dir is None:
        out_dir = Path.cwd()
    out_dir.mkdir(parents=True, exist_ok=True)
    if run_id is None:
        run_id = str(uuid.uuid4())
    if out_path is None:
        out_path = out_dir / f"{audio_name}-{run_id}.wav"
    else:
        out_path = out_dir / Path(out_path).name
    out_path = Path(out_path)

    try:
        samplerate = _pick_working_samplerate(device, channels)
    except Exception as e:
        err("Could not find a supported sample rate for this mic device.")
        err(f"Original error: {e!r}")
        warn("Tips: In Windows Sound Settings > Recording > [Your Mic] > Advanced, check the default format.")
        raise SystemExit(2)

    stop = threading.Event()
    q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=64)

    def waiter():
        try:
            input("Recording… press ENTER to stop.\n")
        except EOFError:
            pass
        stop.set()

    def callback(indata, _frames, _time_info, status):
        if status:
            print(status, file=sys.stderr)
        if indata.ndim == 2 and indata.shape[1] > 1:
            data = np.mean(indata, axis=1, keepdims=True).astype("float32", copy=False)
        else:
            data = indata.astype("float32", copy=False)
        try:
            q.put_nowait(data.copy())
        except queue.Full:
            pass

    info(f"Using sample rate: {samplerate} Hz (device={device if device is not None else 'default'})")
    with sf.SoundFile(str(out_path), mode="w", samplerate=samplerate,
                      channels=channels, subtype=subtype) as wav:
        try:
            stream = sd.InputStream(samplerate=samplerate, channels=channels,
                                    dtype="float32", device=device, callback=callback)
            stream.__enter__()
        except Exception as e:
            err(f"Failed to open audio input stream: {e!r}")
            warn("Try a different device index or change the mic's default format in OS settings.")
            raise SystemExit(2)

        waiter_thread = threading.Thread(target=waiter, daemon=False)
        waiter_thread.start()

        try:
            while not stop.is_set():
                try:
                    wav.write(q.get(timeout=0.2))
                except queue.Empty:
                    pass
        finally:
            try:
                stream.__exit__(None, None, None)
            except Exception:
                pass
            while not q.empty():
                try:
                    wav.write(q.get_nowait())
                except queue.Empty:
                    break
            waiter_thread.join(timeout=1.0)

    ok(f"Saved recording to: {out_path}\n")
    return str(out_path.resolve())


# -------- Diarization utils --------
def _load_diarization_pipeline():
    if not ENABLE_DIARIZATION:
        warn("Diarization disabled (ENABLE_DIARIZATION=False).")
        return None

    token = (
            HUGGINGFACE_TOKEN
            or os.environ.get("HUGGINGFACE_TOKEN")
            or os.environ.get("HF_TOKEN")
            or os.environ.get("HUGGING_FACE_TOKEN")
    )

    if not token or not token.startswith("hf_"):
        warn("No valid Hugging Face token found. Please set HUGGINGFACE_TOKEN or HF_TOKEN in environment.")
        return None

    try:
        from pyannote.audio import Pipeline
    except ImportError:
        err("pyannote.audio not installed. Install with:\n  pip install 'pyannote.audio>=3.1'")
        return None

    # PyTorch Lightning passes weights_only=None to torch.load, which PyTorch 2.6 treats
    # as True. Patch torch.load to convert None → False so pyannote's trusted checkpoint
    # globals are allowed, then restore the original immediately after.
    import functools
    _orig_load = torch.load

    @functools.wraps(_orig_load)
    def _permissive_load(*args, **kwargs):
        if kwargs.get("weights_only") is None:
            kwargs["weights_only"] = False
        return _orig_load(*args, **kwargs)

    torch.load = _permissive_load
    try:
        # Try new huggingface_hub auth style first, fall back to legacy use_auth_token=
        try:
            pipeline = Pipeline.from_pretrained(DIARIZATION_MODEL, token=token)
            ok(f"Loaded diarization model '{DIARIZATION_MODEL}'.")
            return pipeline
        except TypeError:
            try:
                pipeline = Pipeline.from_pretrained(DIARIZATION_MODEL, use_auth_token=token)
                ok(f"Loaded diarization model '{DIARIZATION_MODEL}' (legacy auth).")
                return pipeline
            except Exception as e:
                err(f"Could not load diarization model ({DIARIZATION_MODEL}): {e}")
                return None
        except Exception as e:
            err(f"Could not load diarization model ({DIARIZATION_MODEL}): {e}")
            return None
    finally:
        torch.load = _orig_load


def run_diarization(audio_path: str, pipeline) -> list[dict]:
    """Run diarization; return [{speaker, start, end}, ...] sorted by start."""
    if pipeline is None:
        return []
    try:
        diarization = pipeline(audio_path)
    except Exception as e:
        err(f"Diarization failed: {e}")
        return []

    turns = [
        {"speaker": str(speaker), "start": float(turn.start), "end": float(turn.end)}
        for turn, _, speaker in diarization.itertracks(yield_label=True)
    ]
    turns.sort(key=lambda x: x["start"])

    speaker_map: dict[str, str] = {}
    for t in turns:
        spk = t["speaker"]
        if spk not in speaker_map:
            speaker_map[spk] = f"SPEAKER_{len(speaker_map):02d}"
        t["speaker"] = speaker_map[spk]
    return turns


def _overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def map_segments_to_speakers(segments: list[dict], turns: list[dict]) -> list[dict]:
    """Assign each Whisper segment to the speaker turn with the greatest time overlap."""
    if not turns:
        return [{
            "start": float(s.get("start", 0.0)),
            "end": float(s.get("end", 0.0)),
            "text": (s.get("text") or "").strip(),
            "speaker": None,
        } for s in segments]

    out = []
    for s in segments:
        s_start = float(s.get("start", 0.0))
        s_end = float(s.get("end", s_start))
        best_speaker = None
        best_ov = 0.0
        for t in turns:
            ov = _overlap(s_start, s_end, t["start"], t["end"])
            if ov > best_ov:
                best_ov = ov
                best_speaker = t["speaker"]
        out.append({
            "start": s_start,
            "end": s_end,
            "text": (s.get("text") or "").strip(),
            "speaker": best_speaker,
        })
    return out


def _fmt_ts(sec: float, sep: str) -> str:
    total_ms = round(sec * 1000)
    ms = total_ms % 1000
    total_s = total_ms // 1000
    s = total_s % 60
    m = (total_s // 60) % 60
    h = total_s // 3600
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def _fmt_ts_srt(sec: float) -> str:
    return _fmt_ts(sec, ",")


def _fmt_ts_vtt(sec: float) -> str:
    return _fmt_ts(sec, ".")


def save_diarized_txt(items: list[dict], out_path: Path):
    with open(out_path, "w", encoding="utf-8") as f:
        for it in items:
            spk = it["speaker"] or "SPEAKER_??"
            f.write(f"[{spk}] {it['text']}\n")


def save_srt(items: list[dict], out_path: Path):
    with open(out_path, "w", encoding="utf-8") as f:
        for i, it in enumerate(items, 1):
            f.write(f"{i}\n")
            f.write(f"{_fmt_ts_srt(it['start'])} --> {_fmt_ts_srt(it['end'])}\n")
            spk = it["speaker"] or "SPEAKER_??"
            f.write(f"{spk}: {it['text']}\n\n")


def save_vtt(items: list[dict], out_path: Path):
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        for it in items:
            f.write(f"{_fmt_ts_vtt(it['start'])} --> {_fmt_ts_vtt(it['end'])}\n")
            spk = it["speaker"] or "SPEAKER_??"
            f.write(f"{spk}: {it['text']}\n\n")


# -------- Main helpers --------
def _print_cuda_info():
    info(f"torch: {torch.__version__} | CUDA build: {torch.version.cuda}")
    info(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    if torch.cuda.is_available():
        ok(f"CUDA available — {torch.cuda.device_count()} device(s):")
        for i in range(torch.cuda.device_count()):
            alloc = torch.cuda.memory_allocated(i) / 1024 ** 2
            reserved = torch.cuda.memory_reserved(i) / 1024 ** 2
            print(f"  [{i}] {torch.cuda.get_device_name(i)}  alloc={alloc:.1f} MB  reserved={reserved:.1f} MB")
    else:
        warn("CUDA not available.")


def _setup_logging(out_dir: Path, audio_name: str, run_id: str) -> logging.Logger:
    log_path = out_dir / f"{audio_name}-{run_id}.log"
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    info(f"Logging to {log_path}")
    return logging.getLogger("whisper")


def _select_audio(base_dir: Path) -> str:
    """Prompt the user to pick an audio file or mic, preferring unprocessed files."""
    candidates = _list_unprocessed_audio_candidates(base_dir)
    if candidates:
        pool = candidates
        info("\nAvailable (unprocessed) audio files:")
    else:
        pool = sorted(f for f in os.listdir(base_dir) if f.lower().endswith(AUDIO_EXTENSIONS))
        if pool:
            warn("\nAll audio files (already processed):")
        else:
            warn("No audio files found in current folder.")
            return "mic"
    for i, file in enumerate(pool, start=1):
        print(f"  {c(f'[{i}]', Fore.MAGENTA)} {file}")

    raw = input(c(f"\nSelect file by number or filename ('mic' to record) [{pool[0]}]: ", Fore.CYAN)).strip()
    if not raw:
        return pool[0]
    if raw.lower() == "mic":
        return "mic"
    if raw.isdigit():
        idx = int(raw)
        if 1 <= idx <= len(pool):
            return pool[idx - 1]
    return raw


def _resolve_audio_path(choice: str, base_dir: Path) -> tuple[str, str, str, Path]:
    """Resolve a filename choice to (audio_path, audio_name, run_id, out_dir)."""
    p = Path(choice) if Path(choice).is_absolute() else base_dir / choice
    if not p.exists():
        print(c(f"Audio file not found: {p}", Fore.RED), file=sys.stderr)
        raise SystemExit(1)
    audio_name = p.stem
    run_id = str(uuid.uuid4())
    out_dir = base_dir / f"{audio_name}-{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return str(p.resolve()), audio_name, run_id, out_dir


def _log_segments(logger: logging.Logger, segments: list, use_sentence_split: bool):
    sent_id = 0
    for seg in segments:
        seg_text = (seg.get("text") or "").strip()
        if not seg_text:
            continue
        if use_sentence_split:
            for s in sentence_split(seg_text):
                logger.info("SENT %04d [%7.2f-%7.2f] %s", sent_id, seg.get("start", 0.0), seg.get("end", 0.0), s)
                print(c(">>> ", Fore.BLUE) + s)
                sent_id += 1
        else:
            logger.info("SEG  %04d [%7.2f-%7.2f] %s", seg.get("id", sent_id), seg.get("start", 0.0),
                        seg.get("end", 0.0), seg_text)
            print(c(">>> ", Fore.BLUE) + seg_text)
            sent_id += 1


# -------- Main flow --------
def main():
    base_dir = Path.cwd()
    banner("Speaker Diarization & Transcription")
    _print_cuda_info()

    # Pre-load diarization pipeline before audio selection so failures are caught early
    diarization_pipeline = None
    if ENABLE_DIARIZATION:
        info("Checking diarization pipeline...")
        diarization_pipeline = _load_diarization_pipeline()
        if diarization_pipeline is None:
            ans = input(
                c("Diarization unavailable. Continue with transcription only? [y/N]: ", Fore.YELLOW)).strip().lower()
            if ans != "y":
                sys.exit(1)

    banner("Select Audio Source", Fore.MAGENTA)
    choice = _select_audio(base_dir)

    if choice.lower() == "mic":
        device = None
        banner("Select Input Device", Fore.MAGENTA)
        avail = list_input_devices()
        if avail:
            pick = input(c("Pick device index (blank=default): ", Fore.CYAN)).strip()
            if pick:
                try:
                    device = int(pick)
                except ValueError:
                    warn("Invalid device index; using default device.")
        audio_name = "mic"
        run_id = str(uuid.uuid4())
        out_dir = base_dir / f"{audio_name}-{run_id}"
        out_dir.mkdir(parents=True, exist_ok=True)
        logger = _setup_logging(out_dir, audio_name, run_id)
        audio_path = record_mic_until_enter(
            device=device,
            out_dir=out_dir,
            audio_name=audio_name,
            run_id=run_id,
        )
        source_path = audio_path
    else:
        source_path, audio_name, run_id, out_dir = _resolve_audio_path(choice, base_dir)
        logger = _setup_logging(out_dir, audio_name, run_id)
        audio_path = source_path
        if Path(source_path).suffix.lower() in VIDEO_EXTENSIONS:
            banner("Extracting Audio", Fore.MAGENTA)
            audio_path = str(_extract_audio_to_wav(Path(source_path), out_dir, audio_name))

    t_total = time.perf_counter()

    # Load model
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    info(f"Loading Whisper model '{MODEL_NAME}' on {device_type.upper()}...")
    t_load = time.perf_counter()
    model = whisper.load_model(MODEL_NAME, device=device_type)
    dt_load = time.perf_counter() - t_load
    fp16 = device_type == "cuda"
    logger.info("Model loaded: %s on %s (fp16=%s) in %s", MODEL_NAME, device_type, fp16, _fmt_duration(dt_load))
    ok(f"Model loaded in {_fmt_duration(dt_load)}")

    # Transcribe
    banner("Transcribing", Fore.GREEN)
    t0 = time.perf_counter()
    if USE_TEMPERATURE_FALLBACK:
        result = model.transcribe(
            audio_path, fp16=fp16, language=FORCE_LANGUAGE, verbose=True,
            temperature=[0.0, 0.2, 0.4, 0.6], best_of=5, beam_size=5,
            condition_on_previous_text=CONDITION_ON_PREVIOUS_TEXT,
        )
    else:
        result = model.transcribe(
            audio_path, fp16=fp16, language=FORCE_LANGUAGE, verbose=True,
            condition_on_previous_text=CONDITION_ON_PREVIOUS_TEXT,
        )

    dt = time.perf_counter() - t0
    logger.info("Transcription finished in %s", _fmt_duration(dt))
    ok(f"Done in {_fmt_duration(dt)}\n")

    # Save transcript
    full_text = dedup_transcript((result.get("text") or "").strip())
    transcript_file = out_dir / f"{audio_name}-{run_id}.txt"
    transcript_file.write_text(full_text, encoding="utf-8")
    ok(f"Transcript saved to: {transcript_file}")

    # Log sentence-by-sentence
    segments = result.get("segments") or []
    if segments:
        _log_segments(logger, segments, USE_SENTENCE_SPLIT)

    # Save JSON
    json_file = out_dir / f"{audio_name}-{run_id}.json"
    payload: dict[str, Any] = {
        "meta": {
            "audio_name": audio_name,
            "run_id": run_id,
            "created_at": datetime.now(UTC).isoformat(),
            "source_path": str(source_path),
            "input_path": str(audio_path),
            "output_dir": str(out_dir),
            "model": MODEL_NAME,
            "device": device_type,
            "fp16": fp16,
            "duration_seconds": float(dt),
            "whisper_params": {
                "language_forced": FORCE_LANGUAGE,
                "sentence_split": USE_SENTENCE_SPLIT,
                "task": result.get("task"),
            },
            "language": {
                "detected": result.get("language"),
                "probability": result.get("language_probability") or result.get("detected_language_probability"),
            },
        },
        "text": full_text,
        "segments": [_json_safe_segment(s) for s in segments],
    }
    json_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    ok(f"JSON saved to: {json_file}")

    try:
        _mark_processed(base_dir, source_path, run_id)
    except OSError as e:
        err(f"Could not update processed manifest: {e}")

    # Speaker diarization (optional)
    diar_turns: list = []
    dt_diar = 0.0
    if ENABLE_DIARIZATION and diarization_pipeline is not None:
        banner("Diarization", Fore.GREEN)
        t_diar = time.perf_counter()
        diar_turns = run_diarization(audio_path, diarization_pipeline)
        dt_diar = time.perf_counter() - t_diar
        logger.info("Diarization finished in %s", _fmt_duration(dt_diar))
        ok(f"Diarization done in {_fmt_duration(dt_diar)}")
        if diar_turns:
            diar_items = map_segments_to_speakers(segments, diar_turns)

            diar_json_file = out_dir / f"{audio_name}-{run_id}-diarization.json"
            diar_json_file.write_text(json.dumps({
                "meta": {
                    "audio_name": audio_name,
                    "run_id": run_id,
                    "created_at": datetime.now(UTC).isoformat(),
                    "model": DIARIZATION_MODEL,
                },
                "turns": diar_turns,
                "whisper_segments_mapped": diar_items,
            }, ensure_ascii=False, indent=2), encoding="utf-8")
            ok(f"Diarization JSON saved to: {diar_json_file}")

            save_diarized_txt(diar_items, out_dir / f"{audio_name}-{run_id}-diarized.txt")
            save_srt(diar_items, out_dir / f"{audio_name}-{run_id}-diarized.srt")
            save_vtt(diar_items, out_dir / f"{audio_name}-{run_id}-diarized.vtt")
            ok(f"Diarized TXT/SRT/VTT saved to: {out_dir}")
        else:
            warn("Diarization skipped or returned no turns.")
    else:
        warn("Diarization disabled (ENABLE_DIARIZATION=False).")

    dt_total = time.perf_counter() - t_total

    banner("All Set", Fore.GREEN)
    print(f"  Output dir:  {out_dir}")
    print(f"  Log:         {out_dir / f'{audio_name}-{run_id}.log'}")
    print(f"  Transcript:  {transcript_file}")
    print(f"  JSON:        {json_file}")
    if diar_turns:
        print(f"  Diarization: {out_dir / f'{audio_name}-{run_id}-diarization.json'}")
        print(f"  SRT:         {out_dir / f'{audio_name}-{run_id}-diarized.srt'}")
        print(f"  VTT:         {out_dir / f'{audio_name}-{run_id}-diarized.vtt'}")
    print(c("\nTiming:", Fore.CYAN + Style.BRIGHT))
    print(f"  Model load:    {_fmt_duration(dt_load)}")
    print(f"  Transcription: {_fmt_duration(dt)}")
    if ENABLE_DIARIZATION:
        print(f"  Diarization:   {_fmt_duration(dt_diar)}")
    print(f"  Total:         {_fmt_duration(dt_total)}")


if __name__ == "__main__":
    main()
