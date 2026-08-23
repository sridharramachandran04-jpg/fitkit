# FitKit — AI Fitness Coach (Streamlit + MediaPipe)

FitKit is a Streamlit app that uses MediaPipe Pose to give live form feedback
and rep counts for 25 exercises, plus a Gemini-powered chat coach.

## What was broken and what I fixed

I extracted your zip, set up a clean environment, actually **installed the
dependencies and ran the app** (not just read the code), and fixed every
error that surfaced:

1. **Crash-on-launch: `AttributeError: module 'mediapipe' has no attribute
   'solutions'`.** MediaPipe `1.0+` removed the legacy `mp.solutions.*` API
   (Pose, drawing_utils, etc.) that every file in `exercises/` and `app.py`
   depends on. Your environment had auto-installed mediapipe `1.0.1`, which
   is incompatible with this codebase.
   **Fix:** pinned `mediapipe==0.10.21`, the last line that still ships the
   `solutions` API this project is built on.

2. **Dependency conflicts.** Once mediapipe was pinned, the newest
   `streamlit` wanted `protobuf>=5.26`, which fights mediapipe's `protobuf 4.x`
   requirement — and the newest `opencv-python-headless` wanted `numpy>=2`,
   which fights mediapipe's `numpy 1.x` requirement.
   **Fix:** pinned compatible versions of all four packages together and
   verified `pip check` reports zero conflicts (see `requirements.txt`).

3. **Real runtime crash: `NameError: name 'accuracy_bar' is not defined`.**
   In `exercises/pushups.py`, `exercises/shoulder_press.py`, and
   `exercises/jumping_jacks.py`, the realtime-camera loop referenced three
   Streamlit placeholder widgets (`accuracy_bar`, `counter_disp`,
   `feedback_disp`) that were never created — only `dashboard_panel` was.
   This means **Push-ups, Shoulder Press, and Jumping Jacks crashed the
   instant you clicked "Start Camera."** I compared against the 20+ other
   exercise files (which don't have this bug) and applied the same working
   pattern to all three.

4. **Stale files.** Removed `exercises/hammer_curls.py.bak` (an old,
   incomplete draft of the file sitting next to the real one — harmless but
   confusing) and all `__pycache__/` folders.

5. **Deprecated Gemini model.** `chatbot.py` was calling
   `gemini-3-flash-preview`, which Google has deprecated. Updated it to
   `gemini-3.5-flash`, its stable successor, so the chatbot won't start
   silently failing.

After these fixes I ran `pyflakes` across every file (no undefined names or
syntax errors remain) and did a full `streamlit run` smoke test — the app
boots clean with a `200 OK` and passing health check.

### Note on remaining minor lint warnings
`pyflakes` also flags a handful of unused imports/variables (e.g. unused
`PIL.Image` imports, a few assigned-but-unused locals in `app.py` and some
exercise files). These are harmless — cosmetic leftovers from development,
not bugs — so I left them alone rather than risk touching working detection
logic. Say the word if you'd like those tidied up too.

## Round 2 — after you added the demo videos

6. **Every demo video except two was silently invisible.** Each exercise
   page's "Demo" tab looks for a video at an exact, hardcoded path — e.g.
   `demo/Deadlift/deadlift_demo.mp4`. The 25 files you added were real,
   correctly-placed videos, but **23 of them had the wrong filename**
   (e.g. `demo/Deadlift/Deadlift.mp4`, `demo/BenchPress/bench press.mp4`
   with a space in it). Since the code does an `os.path.exists()` check
   before playing the video, every mismatched one just silently fell
   through to the "Add demo video to ..." placeholder message instead of
   erroring — easy to miss unless you clicked through all 25 pages.
   **Fix:** renamed all 23 files to the exact name each exercise file
   expects (full mapping below). I verified every single path the code
   references now resolves to a real file with a script, not just by eye.
   `Bicep/bicep_demo.mp4` and `Pushup/pushup_demo.mp4` were already correct
   and untouched.

   | Folder | Old filename | New filename |
   |---|---|---|
   | ArnoldPress | `ArnoldPress.mp4` | `arnold_press_demo.mp4` |
   | BenchPress | `bench press.mp4` | `bench_press_demo.mp4` |
   | BoxStepUp | `BoxStepUp.mp4` | `box_step_ups_demo.mp4` |
   | Burpee | `Burpee.mp4` | `burpee_demo.mp4` |
   | CableFly | `CableFly.mp4` | `cable_fly_demo.mp4` |
   | CalfRaise | `CalfRaise.mp4` | `calf_raise_demo.mp4` |
   | Deadlift | `Deadlift.mp4` | `deadlift_demo.mp4` |
   | FacePull | `FacePull.mp4` | `face_pull_demo.mp4` |
   | GluteBridge | `GluteBridge.mp4` | `glute_bridge_demo.mp4` |
   | HammerCurl | `HammerCurl.mp4` | `hammer_curl_demo.mp4` |
   | HighKnees | `HighKnees.mp4` | `high_knees_demo.mp4` |
   | HipThrust | `Hip Thrust.mp4` | `hip_thrust_demo.mp4` |
   | JumpingJack | `JumpingJack.mp4` | `jumping_jack_demo.mp4` |
   | LateralRaise | `LateralRaise.mp4` | `lateral_raise_demo.mp4` |
   | LegPress | `LegPress.mp4` | `leg_press_demo.mp4` |
   | Lunge | `Lunge.mp4` | `lunge_demo.mp4` |
   | MountainClimber | `MountainClimber.mp4` | `mc_demo.mp4` |
   | OverheadPress | `OverheadPress.mp4` | `ohp_demo.mp4` |
   | PullUp | `PullUp.mp4` | `pull_up_demo.mp4` |
   | Row | `Row.mp4` | `row_demo.mp4` |
   | ShoulderPress | `ShoulderPress.mp4` | `shoulder_press_demo.mp4` |
   | SumoSquat | `SumoSquat.mp4` | `sumo_squat_demo.mp4` |
   | TricepDip | `TricepDip.mp4` | `tricep_dip_demo.mp4` |

7. **`demo/Plank`, `demo/Squat`, `demo/wall_sit` videos aren't wired up —
   and that's fine.** Plank, Wall Sit, and Squat Hold are handled by a
   different code path (`_posture_exercise()` in `app.py`) that shows a
   static GIF/WebP from `assets/` instead of a demo video, so the extra
   `.mp4`/`.jpg`/`.png` files you added in those three `demo/` folders
   aren't referenced by any code and are just sitting there unused. I left
   them as-is rather than guess at wiring them in — say the word if you'd
   like the posture-exercise pages switched over to use those videos
   instead of the static images.

8. Removed leftover `__pycache__/` folders.

## Round 3 — deploy failure on Render: `No matching distribution found for mediapipe==0.10.21`

9. **Render's default Python version (3.14.3) doesn't have a `mediapipe==0.10.21` wheel.**
   Your build log showed pip successfully resolving `streamlit` and `opencv-python-headless`,
   then failing on `mediapipe==0.10.21` with "Could not find a version that satisfies the
   requirement" — even though that exact version exists on PyPI. I confirmed why:
   `mediapipe` only publishes prebuilt wheels for Python 3.10–3.12; Render creates new
   services on Python **3.14.3** by default as of Feb 2026, and there's no 3.14 (or 3.13)
   wheel for `mediapipe==0.10.21`, so pip has nothing to install.

   **I did *not* just bump the mediapipe version to one Render could find** (0.10.30+), even
   though those do have wheels for newer Python — I checked, and Google removed the
   `mp.solutions` API (the one this entire codebase is built on: `mp.solutions.pose`,
   `mp.solutions.drawing_utils`, etc.) starting around mediapipe 0.10.30. Multiple people
   hit `AttributeError: module 'mediapipe' has no attribute 'solutions'` on exactly those
   versions. Doing that would have "fixed" the install and reintroduced the original crash
   at runtime instead.

   **Fix:** added a `.python-version` file (containing `3.12`) to the project root. This is
   Render's documented way to pin your service's Python version — it takes priority over
   the platform default. I verified the fix directly: built a fresh Python 3.12 venv,
   installed the exact `requirements.txt`, confirmed `pip check` reports zero conflicts,
   confirmed `mediapipe.solutions.pose` exists and imports correctly, and ran a full
   `streamlit run` smoke test under that same Python 3.12 environment — `200 OK`, no
   traceback.

   No changes were needed to `requirements.txt` itself; the pins in there were already
   correct for Python 3.10–3.12, which is what the README always recommended running
   locally. Render was just defaulting to a newer Python than the pinned packages support.

   If you deploy anywhere other than Render and it also lets you pick a Python version,
   make sure it's 3.10, 3.11, or 3.12 — not 3.13+.

After these changes I reinstalled the pinned `requirements.txt` into a
clean venv (`pip check` → zero conflicts), imported all 27 Python modules
individually (all clean), byte-compiled every file (no syntax errors), ran
a script that confirms every single demo-video path referenced anywhere in
the code now resolves to a real file, and did a fresh `streamlit run`
smoke test — `200 OK`, no traceback in the log.

---

## Project structure

```
fitkit_output/
├── app.py                 # Main Streamlit app (auth, dashboard, routing)
├── chatbot.py              # Gemini-powered AI coach chat page
├── realtime_feedback.py    # Shared overlay/dashboard/TTS helpers
├── exercises/               # One detection module per exercise
├── assets/                  # Logos, demo gifs
├── demo/                    # Reference video clips per exercise
├── fitkit.db                 # SQLite DB (users, logs) — auto-created if missing
├── .streamlit/secrets.toml   # Your Gemini API key (keep this private!)
├── requirements.txt          # Pinned, tested dependencies
└── .gitignore
```

---

## Running it locally

**Requirements:** Python 3.10–3.12, a webcam if you want realtime mode.

```bash
# 1. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install the pinned dependencies
pip install -r requirements.txt

# 3. Make sure your Gemini key is set (already present in .streamlit/secrets.toml)
#    GEMINI_API_KEY = "your-real-key"

# 4. Run it
streamlit run app.py
```

Streamlit opens at `http://localhost:8501`. Your webcam is only accessed
from your own machine — that's expected and fine for local use.

---

## Hosting it — the one thing to know first

**Realtime camera mode will not work on a normal cloud host as-is.**

The code calls `cv2.VideoCapture(0)`, which opens a **physical camera
attached to the machine running the Python process.** On your laptop, that's
your webcam — perfect. On a cloud server (Streamlit Community Cloud,
Render, Railway, an EC2 box, etc.), there is no physical camera, so
`cv2.VideoCapture(0)` just fails silently and the video panel stays blank.

The **Upload Image** and **Upload Video** modes in every exercise page are
unaffected — they work identically wherever you host, since the file comes
from the visitor's browser.

You have two realistic paths:

- **Host it for upload-based use only** (simplest — no code changes). Realtime
  camera won't work for visitors, but everything else (auth, dashboard,
  chatbot, image/video analysis, exercise logging) works fine remotely.
- **Rebuild realtime mode with `streamlit-webrtc`**, which streams frames
  from the *visitor's* browser webcam to your server for processing. This is
  the standard way to do live webcam CV in a hosted Streamlit app, but it's
  a real code change (a new component per exercise), not a config tweak —
  happy to help with that if you want it.

### Option A — Streamlit Community Cloud (free, easiest)
1. Push this folder to a **public or private GitHub repo** (the `.gitignore`
   already keeps `secrets.toml` and `fitkit.db` out of git).
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with
   GitHub, click **New app**, pick the repo/branch, and set **Main file
   path** to `app.py`.
3. In **Advanced settings → Secrets**, paste:
   ```toml
   GEMINI_API_KEY = "your-real-key"
   ```
4. Deploy. Note: `fitkit.db` (SQLite) resets whenever the app restarts/redeploys
   on this free tier, since the filesystem isn't persistent — fine for a demo,
   not for real user data retention.

### Option B — A VPS / your own server (e.g. a small cloud box)
1. `git clone` your repo onto the server, `cd` into it.
2. `python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt`
3. Create `.streamlit/secrets.toml` with your `GEMINI_API_KEY` directly on
   the server (never commit it).
4. Run persistently, e.g. with `tmux`/`screen`, or as a systemd service:
   ```bash
   streamlit run app.py --server.port 80 --server.address 0.0.0.0 --server.headless true
   ```
5. Point a domain at the server's IP if you want a real URL, and put it
   behind Nginx + HTTPS (Let's Encrypt) for production use.
6. SQLite persists here since it's a real disk, unlike Option A.

### Option C — Docker (portable, works on most cloud hosts)
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
```
Build and run: `docker build -t fitkit . && docker run -p 8501:8501 fitkit`
(mount `.streamlit/secrets.toml` as a volume or set `GEMINI_API_KEY` as an
env var and read it in code, rather than baking the key into the image).

---

## Quick sanity checklist before you deploy
- [ ] `.streamlit/secrets.toml` is **not** committed to git (already gitignored)
- [ ] `pip install -r requirements.txt` succeeds with `pip check` showing no conflicts
- [ ] `streamlit run app.py` boots with no traceback in the terminal
- [ ] You've decided which hosting path (A/B/C) matches whether you need realtime camera for remote visitors
