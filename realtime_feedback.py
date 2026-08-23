"""
realtime_feedback.py
────────────────────
Shared real-time feedback utilities used by ALL exercise modules.

Provides:
  - draw_feedback_overlay()  → draws rich overlay onto the CV2 frame
  - render_dashboard()       → Streamlit live dashboard (rep counter, accuracy ring, stage badge)
  - speak_js()               → browser-side TTS via st.components (no pyttsx3 needed)
  - FeedbackThrottler        → prevents feedback spam (only speaks every N seconds)
"""

import cv2
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import time


# ── Colour palette ────────────────────────────────────────────────────────────
GREEN  = (0,  220,  80)
RED    = (0,   60, 220)
YELLOW = (0,  200, 255)
WHITE  = (255, 255, 255)
BLACK  = (0,    0,   0)
GOLD   = (0,  180, 215)


class FeedbackThrottler:
    """Only allows a new spoken/displayed feedback every `interval` seconds."""
    def __init__(self, interval: float = 3.0):
        self._interval  = interval
        self._last_time = 0.0
        self._last_text = ""

    def should_speak(self, text: str) -> bool:
        now = time.time()
        if text != self._last_text or (now - self._last_time) >= self._interval:
            self._last_time = now
            self._last_text = text
            return True
        return False


def speak_js(text: str, throttler: FeedbackThrottler | None = None):
    """
    Trigger browser-native TTS using the Web Speech API.
    Works on Chrome/Edge/Firefox without any Python audio dependency.
    Completely silent if the browser doesn't support it.
    """
    if throttler and not throttler.should_speak(text):
        return
    # Sanitise: escape quotes so they don't break the JS string
    safe = text.replace("'", "\\'").replace('"', '\\"').replace("\n", " ")
    components.html(
        f"""
        <script>
        (function() {{
            if (!window.speechSynthesis) return;
            window.speechSynthesis.cancel();
            var u = new SpeechSynthesisUtterance('{safe}');
            u.rate   = 1.1;
            u.pitch  = 1.0;
            u.volume = 1.0;
            window.speechSynthesis.speak(u);
        }})();
        </script>
        """,
        height=0,
        scrolling=False,
    )


def draw_feedback_overlay(
    frame: np.ndarray,
    feedback_text: str,
    is_correct: bool,
    counter: int,
    stage: str,
    accuracy: float,
    exercise_name: str,
) -> np.ndarray:
    """
    Draw a rich real-time overlay onto the video frame:
      • Top bar     : exercise name + rep counter
      • Side badge  : current stage (UP / DOWN / HOLD etc.)
      • Bottom bar  : feedback message with colour coding
      • Accuracy    : live % shown in corner
      • Border      : green when correct form, red when wrong
    """
    h, w = frame.shape[:2]

    # ── Border glow ─────────────────────────────────────────────────────────
    border_color = GREEN if is_correct else RED
    cv2.rectangle(frame, (0, 0), (w-1, h-1), border_color, 6)

    # ── Top bar ──────────────────────────────────────────────────────────────
    cv2.rectangle(frame, (0, 0), (w, 60), (20, 20, 20), -1)
    cv2.putText(frame, exercise_name.upper(),
                (12, 40), cv2.FONT_HERSHEY_DUPLEX, 0.9, GOLD, 2, cv2.LINE_AA)

    # Rep counter — right side of top bar
    rep_text = f"REPS: {counter}"
    (tw, _), _ = cv2.getTextSize(rep_text, cv2.FONT_HERSHEY_DUPLEX, 0.9, 2)
    cv2.putText(frame, rep_text,
                (w - tw - 12, 40), cv2.FONT_HERSHEY_DUPLEX, 0.9, WHITE, 2, cv2.LINE_AA)

    # ── Stage badge (top-left below bar) ────────────────────────────────────
    if stage:
        badge_col = GREEN if stage.upper() in ("UP", "OPEN", "TOP_POSITION") else YELLOW
        badge_txt = stage.upper().replace("_", " ")
        cv2.rectangle(frame, (10, 68), (10 + len(badge_txt)*14 + 10, 98), badge_col, -1)
        cv2.putText(frame, badge_txt,
                    (15, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.6, BLACK, 2, cv2.LINE_AA)

    # ── Accuracy pill (top-right below bar) ──────────────────────────────────
    acc_txt   = f"FORM: {accuracy:.0f}%"
    acc_color = GREEN if accuracy >= 70 else (YELLOW if accuracy >= 40 else RED)
    (aw, _), _ = cv2.getTextSize(acc_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    cv2.rectangle(frame, (w - aw - 20, 66), (w - 6, 98), acc_color, -1)
    cv2.putText(frame, acc_txt,
                (w - aw - 14, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.65, BLACK, 2, cv2.LINE_AA)

    # ── Bottom feedback bar ──────────────────────────────────────────────────
    fb_bg_color = (0, 80, 0) if is_correct else (0, 0, 100)
    cv2.rectangle(frame, (0, h - 65), (w, h), fb_bg_color, -1)

    # Icon
    icon = "✓" if is_correct else "!"
    icon_color = GREEN if is_correct else RED
    cv2.putText(frame, icon,
                (12, h - 22), cv2.FONT_HERSHEY_DUPLEX, 1.4, icon_color, 3, cv2.LINE_AA)

    # Feedback text — word-wrap if too long
    max_chars = w // 14
    words     = feedback_text.split()
    lines     = []
    line      = ""
    for word in words:
        if len(line) + len(word) + 1 <= max_chars:
            line = (line + " " + word).strip()
        else:
            if line: lines.append(line)
            line = word
    if line: lines.append(line)

    fb_color = GREEN if is_correct else YELLOW
    for i, ln in enumerate(lines[:2]):   # max 2 lines
        cv2.putText(frame, ln,
                    (50, h - 38 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, fb_color, 2, cv2.LINE_AA)

    return frame


def render_dashboard(
    counter: int,
    accuracy: float,
    stage: str,
    feedback: str,
    is_correct: bool,
    exercise_name: str,
    elapsed: float,
):
    """
    Render a live Streamlit dashboard panel alongside the video feed.
    Call this inside a st.empty().container() that gets replaced each frame.
    """
    # ── Rep counter + accuracy ────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    c1.metric("🔢 Reps",     counter)
    c2.metric("🎯 Accuracy", f"{accuracy:.0f}%")
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)
    c3.metric("⏱️ Time",   f"{minutes:02d}:{seconds:02d}")

    # ── Accuracy progress bar ─────────────────────────────────────────────
    bar_val = min(int(accuracy), 100)
    st.progress(bar_val / 100,
                text=f"Form accuracy: {bar_val}%  {'✅ Great form!' if bar_val >= 70 else '⚠️ Fix your form'}")

    # ── Stage badge ───────────────────────────────────────────────────────
    stage_colors = {
        "up":            ("#1a7a1a", "white"),
        "down":          ("#8b0000", "white"),
        "open":          ("#1a7a1a", "white"),
        "closed":        ("#555555", "white"),
        "top_position":  ("#1a7a1a", "white"),
        "bottom_position": ("#8b0000", "white"),
        "going_up":      ("#2255aa", "white"),
        "going_down":    ("#aa5500", "white"),
    }
    bg, fg = stage_colors.get(stage.lower() if stage else "", ("#333333", "white"))
    st.markdown(
        f"<div style='background:{bg}; color:{fg}; padding:10px 18px; border-radius:8px;"
        f"text-align:center; font-size:18px; font-weight:bold; margin:6px 0;'>"
        f"⚡ Stage: {stage.upper().replace('_',' ') if stage else 'WAITING'}</div>",
        unsafe_allow_html=True
    )

    # ── Live feedback card ────────────────────────────────────────────────
    if is_correct:
        st.success(f"✅ {feedback}")
    else:
        st.warning(f"⚠️ {feedback}")

    # ── Calories estimate ─────────────────────────────────────────────────
    MET = {"Bicep Curl": 3.5, "Push-up": 8.0, "Jumping Jack": 8.0,
           "Shoulder Press": 5.0, "Lateral Raise": 3.5, "Lunge": 5.0,
           "Tricep Dip": 5.0, "Deadlift": 6.0, "Plank": 4.0,
           "Wall Sit": 4.0, "Squat Hold": 5.0}
    met_val  = MET.get(exercise_name, 5.0)
    weight   = st.session_state.get("user_data", {}).get("weight", 70)
    cal_burn = round((met_val * weight * (elapsed / 3600)), 2)
    st.caption(f"🔥 Est. calories burned this session: **{cal_burn} kcal**")
