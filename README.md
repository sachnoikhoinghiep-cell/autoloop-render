# AutoLoop Render

Upload a short video and losslessly stream-loop it to any target duration
(e.g. an 8-second clip -> 10 hours) using FFmpeg's `-stream_loop -1 -t <seconds> -c copy`.
Because `-c copy` remuxes packets instead of re-encoding, even very long
outputs typically render in seconds, bound by disk I/O rather than playback time.

## Architecture

This is a single self-contained Streamlit app — no separate API server.
The UI (`app.py`) calls straight into `core_engine.py`, which shells out to
`ffmpeg`/`ffprobe` via `subprocess` (list-form args, no `shell=True`).
A FastAPI layer was deliberately left out: nothing here needs to be called
remotely, and running two processes for a local tool adds operational
overhead without benefit. If you later want a REST API (e.g. for batch jobs),
the functions in `core_engine.py` are already separated from the UI and can
be wrapped in FastAPI directly.

## Requirements

- Python 3.9+
- FFmpeg and ffprobe on your PATH. The app checks for both on startup and
  refuses to launch with a clear error if either is missing. Verify with:
  ```
  ffmpeg -version
  ffprobe -version
  ```

### Installing FFmpeg

**Windows** — already installed and on PATH on this machine via
`C:\ProgramData\chocolatey\bin` (`choco install ffmpeg`). To set it up
manually instead: download a build from the FFmpeg site, unzip it, and add
its `bin` folder to the `Path` system environment variable, then open a new
terminal.

**macOS**
```bash
brew install ffmpeg
```

**Linux (Debian/Ubuntu)**
```bash
sudo apt install ffmpeg
```

## Setup

```bash
pip install -r requirements.txt
```

Only Streamlit is required — this build doesn't use FastAPI or the
`ffmpeg-python` package (see [Architecture](#architecture) above for why).

## Run

```bash
streamlit run app.py
```

This opens the app at `http://localhost:8501`.

## Usage

1. Upload a short MP4/MOV/MKV/AVI/WEBM clip.
2. Enter the target output duration in hours/minutes/seconds.
3. Review the estimated output size and disk space check. For very large
   estimates (>20 GB) you'll be asked to explicitly confirm before rendering.
4. Click **Render Looped Video**.
5. Once done, you'll get the absolute output file path. Files under 1 GB
   also get an in-browser download button — larger files are left on disk
   and should be grabbed directly, since pushing many-GB files through a
   browser download is unreliable and memory-heavy.

Uploaded files are saved to `uploads/`, rendered files to `outputs/`. Both
are created automatically and are safe to clear out manually between runs.

## Edge cases handled

- **0 second target**: render button stays disabled with an inline warning.
- **Invalid/non-video files**: rejected both by the file picker's type
  filter and by an `ffprobe` validation pass server-side.
- **Missing ffmpeg/ffprobe**: the app refuses to start and shows install
  instructions instead of failing deep inside a render.
- **Stream-copy incompatible input**: if `-c copy` fails (e.g. a container
  that can't hold the source codec), the app surfaces ffmpeg's stderr
  directly rather than silently falling back to re-encoding.
- **Disk space**: output size is estimated from the source bitrate before
  rendering, and the render is blocked if free disk space is insufficient.
- **Runaway durations**: capped at 48 hours by default
  (`MAX_TARGET_HOURS` in `core_engine.py`) to guard against typos.

## Notes / limitations

- `-c copy` requires the source codec to be compatible with the chosen
  output container. If a clip fails to render, try re-exporting it as a
  standard H.264/AAC MP4 first.
- Very large outputs (e.g. 10 hours of 1080p video) can be tens to
  hundreds of GB — make sure you have the disk space the app estimates
  before confirming.
