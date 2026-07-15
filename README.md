# Python🐍 💬Audio Transcriber 🗣️

## Whisper 🤫 + Speaker Diarization 🙋🏿

## Background

Tired of capturing, writing down notes, and figuring out what was said in a meeting. Playing the game, what he said, she
said, and/or they said. Behold Python Audio Transcriber, this script/application will help you either post-process audio
files OR use an existing audio device to record audio, then post-process it into log files for transcriptions/record
keeping. This application will allow you to transcribe audio so that it will generate log files for what was captured,
so you're able to allow additional operations, such as summarizing via LLM, to capture meeting notes. The kicker? It's
all local/self-hosted!

## Requirements 📋

1. Python environment - (Docker, PyCharm, etc.)
2. Python 3.11 or 3.12
3. Requirements.txt
4. CUDA Toolkit (optional and highly recommended for people with a GPU)
5. Huggingface account and Huggingface token
6. Speaker-Diarization approval via HuggingFace.com
7. (Optional) Third Party LLM - OpenAI, Claude, Grok, DeepSeek, or whatever

### Walkthrough and Initial Steps

### Python Environment (IntelliJ's PyCharm)

At this rate, you should be able to set this up; however, what is a guide without showing you an example? I used
IntelliJ's PyCharm to run this application on my Windows 11 CPU. Now, I do have a 3080 GPU that I'm using with CUDA, but
we'll eventually get there after going through this first.

When setting up the Python interpreter, use Python 3.11 or 3.12 — this project's own `.venv` runs 3.12.10 and
installs/imports `torch`, `whisper`, and `pyannote.audio` cleanly. Python 3.10 is untested; very new releases (3.13/3.14
as of writing) aren't supported yet since `torch==2.6.0` and the other pinned dependencies in `requirements.txt` don't
have prebuilt wheels for them.

### ffmpeg

Whisper decodes audio via `ffmpeg`, and this script also uses it to pull the audio track out of video files. Install
`ffmpeg` and make sure it's on your `PATH` before running anything (`ffmpeg -version` should work from a terminal).

- Windows: `choco install ffmpeg` (Chocolatey) or download a build from https://ffmpeg.org/download.html and add it to
  `PATH`.
- macOS: `brew install ffmpeg`.
- Linux: `sudo apt install ffmpeg` (Debian/Ubuntu) or your distro's equivalent.

### Installing dependencies (cross-platform)

```
pip install -r requirements.txt
```

This installs everything on Windows, macOS, and Linux, including a CPU/MPS build of `torch`. If you're on **Windows or
Linux with an NVIDIA GPU** and want CUDA-accelerated transcription/diarization, follow up with:

```
pip install -r requirements-cuda.txt
```

which reinstalls `torch`/`torchaudio`/`torchvision` as CUDA 12.4 builds (requires a matching NVIDIA driver). This step
doesn't apply to macOS — there's no CUDA support there, so the plain `requirements.txt` install is all you need (the
script falls back to CPU automatically either way, per `torch.cuda.is_available()`).

### Hugging Face token (for diarization)

Speaker diarization uses a gated pyannote model, so you'll need a Hugging Face account and token:

1. Create a token at https://huggingface.co/settings/tokens (read access is enough).
2. Accept the model terms for `pyannote/speaker-diarization-3.1` and `pyannote/segmentation-3.0` on huggingface.co —
   diarization will fail to load without this.
3. Set the token, either via a `.env` file or an environment variable — never hardcode it in the script:
    - `.env` file (recommended): copy `.env.example` to `.env` and fill in your token. The script loads it automatically
      on startup. `.env` is gitignored, so it stays local.
    - PowerShell: `$env:HUGGINGFACE_TOKEN = "hf_..."`
    - Bash: `export HUGGINGFACE_TOKEN="hf_..."`

   (`HF_TOKEN` and `HUGGING_FACE_TOKEN` also work.) **Never commit a real token** — it's a live credential.

If no valid token is found, or `ENABLE_DIARIZATION = False`, the script still runs and produces a plain transcript, just
without speaker labels.

## Features 🚀

- Transcribes one or more existing audio/video files in a batch, or records straight from a mic until you press ENTER.
- Automatically extracts audio from video containers (`.mp4`, `.flv`, `.webm`, `.mkv`, `.mov`, `.avi`, `.wmv`, `.m4v`,
  `.ts`) to 16kHz mono WAV before transcription/diarization, since the diarization pipeline doesn't reliably read video
  containers directly.
- Speaker diarization via `pyannote.audio`, producing speaker-tagged `.txt`, `.srt`, and `.vtt` output alongside the raw
  transcript.
- De-duplicates Whisper's hallucinated repeated phrases/sentences (a known Whisper quirk on silence or noisy audio).
- Tracks already-processed files in `.processed.json` so re-running the script won't reprocess the same audio twice.
- Weekly cleanup of old per-run output folders (anything older than the current ISO week gets removed automatically,
  logged to `cleanup.log`).
- Color-coded terminal output (status/success/warning/error) for easier reading of long transcription runs.

## Usage

```
python best-speaker-diarization.py
```

- You'll be prompted to pick audio/video files from the current folder. Enter one number for a single file or
  comma-separated numbers such as `1,2,3` for a batch. The full batch is shown for confirmation before transcription.
  Alternatively, type `mic` to record from a microphone.
- For mic recording, pick an input device (or leave blank for default), then press ENTER when you're done talking.
- Everything for that run — logs, transcript, JSON, and diarized outputs — lands in a new `<audio_name>-<uuid>/` folder.

## Output files

For each run, `<audio_name>-<uuid>/` contains:
| File | Contents | |---|---| | `<audio>-<uuid>.log` | Sentence-level log with timestamps | | `<audio>-<uuid>.txt` |
Full transcript, no speaker labels | | `<audio>-<uuid>.json` | Run metadata + raw Whisper segments | |
`<audio>-<uuid>.wav` | Only present when recording from mic | | `<audio>-<uuid>-diarization.json` | Diarization turns +
speaker mapping (if diarization ran) | | `<audio>-<uuid>-diarized.txt` / `.srt` / `.vtt` | Speaker-tagged transcript /
subtitles (if diarization ran) |

## Configuration

The main knobs live as constants near the top of `best-speaker-diarization.py`:
| Setting | Purpose | |---|---| | `MODEL_NAME` | Whisper model size (`tiny`/`base`/`small`/`medium`/`large-v3`) — bigger
is more accurate but slower | | `FORCE_LANGUAGE` | Force a language code (e.g. `"en"`), or `None` to auto-detect | |
`ENABLE_DIARIZATION` | Turn speaker diarization on/off | | `DIARIZATION_MODEL` | Which pyannote model to use | |
`CONDITION_ON_PREVIOUS_TEXT` | Whisper anti-hallucination — `False` breaks repetition feedback loops | |
`USE_SENTENCE_SPLIT` | Log per-sentence instead of per-Whisper-segment | | `USE_TEMPERATURE_FALLBACK` | Retry
transcription at higher temperatures when quality is low |

## Notes

- This project's `.gitignore` excludes audio/video files, `.processed.json`, logs, and local editor config — recordings
  and transcripts of anything you process are meant to stay local, not get committed.
- CUDA is optional but strongly recommended; without a GPU, `large-v3` transcription will be noticeably slower.
