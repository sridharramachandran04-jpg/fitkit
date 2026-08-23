"""
Guards against the #1 support issue in this project: a mediapipe version
mismatch. mediapipe 1.0+ removed the legacy `mp.solutions.*` API (Pose,
drawing_utils, etc.) that every file in exercises/ and app.py depends on.

This project needs the exact version pinned in requirements.txt. If that's
not what's installed, fail immediately with a clear fix instead of the
cryptic "AttributeError: module 'mediapipe' has no attribute 'solutions'"
three files deep in a stack trace.

Imported as the very first line of app.py.
"""
import sys

REQUIRED_MEDIAPIPE = "0.10.21"


def check():
    try:
        import mediapipe as mp
    except ImportError:
        _fail(
            "mediapipe is not installed at all.",
            installed="not installed",
        )
        return

    installed = getattr(mp, "__version__", "unknown")
    has_solutions = hasattr(mp, "solutions")

    if not has_solutions or installed != REQUIRED_MEDIAPIPE:
        _fail(
            f"mediapipe {installed} is installed, but this project needs "
            f"exactly {REQUIRED_MEDIAPIPE} (mediapipe 1.0+ removed the "
            f"mp.solutions API this app is built on).",
            installed=installed,
        )


def _fail(reason, installed):
    py = sys.executable
    msg = f"""
{'=' * 70}
  FitKit can't start: wrong mediapipe version
{'=' * 70}

  {reason}

  Currently running Python: {py}

  FIX — run this exact command (copy/paste it):

      "{py}" -m pip install "mediapipe==0.10.21" --force-reinstall

  Then verify it took effect:

      "{py}" -c "import mediapipe as mp; print(mp.__version__)"

  It must print 0.10.21. If it still doesn't after --force-reinstall,
  something else in your environment (a second Python install, a global
  site-packages conflict, etc.) is overriding it — use setup.bat in this
  folder instead, which builds an isolated virtual environment from
  scratch and can't be overridden by anything outside it.

{'=' * 70}
"""
    print(msg, file=sys.stderr)
    sys.exit(1)


check()
