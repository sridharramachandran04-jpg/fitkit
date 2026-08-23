import cv2, mediapipe as mp, numpy as np, time, os, tempfile
from PIL import Image
import streamlit as st
from realtime_feedback import draw_feedback_overlay, render_dashboard, speak_js, FeedbackThrottler
import os as _os

_BASE_DIR = _os.path.dirname(_os.path.abspath(__file__))
_APP_DIR  = _os.path.dirname(_BASE_DIR)

mp_drawing = mp.solutions.drawing_utils
mp_pose    = mp.solutions.pose
drawing_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=1)

confidence_threshold = 0.5


def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))


def process_high_knees_frame(frame, state: dict) -> tuple:
    feedback_duration = 2
    frame_height, frame_width = frame.shape[:2]
    current_time = time.time()

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=1) as pose:
        results = pose.process(image)
    image.flags.writeable = True
    frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    correct_form  = False
    feedback_text = state.get("feedback_text", "Position yourself in frame")

    try:
        if results.pose_landmarks is None:
            raise Exception("No pose detected")

        lm = results.pose_landmarks.landmark

        rh  = lm[mp_pose.PoseLandmark.RIGHT_HIP.value]
        rk  = lm[mp_pose.PoseLandmark.RIGHT_KNEE.value]
        lh  = lm[mp_pose.PoseLandmark.LEFT_HIP.value]
        lk  = lm[mp_pose.PoseLandmark.LEFT_KNEE.value]

        right_ok = all(x.visibility > confidence_threshold for x in [rh, rk])
        left_ok  = all(x.visibility > confidence_threshold for x in [lh, lk])

        if not right_ok and not left_ok:
            raise Exception("Hips/knees not visible")

        stage = state.get("stage", None)
        lft   = state.get("last_feedback_time", 0)
        last_side = state.get("last_side", None)

        def px(lm_p): return (int(lm_p.x * frame_width), int(lm_p.y * frame_height))

        # Track whichever knee just came up (alternating)
        knee_raised = None
        if right_ok and rk.y < rh.y:
            knee_raised = "right"
            cv2.circle(frame, px(rk), 12, (0, 255, 0), -1)
            cv2.circle(frame, px(rh), 8,  (255, 255, 0), -1)
        if left_ok and lk.y < lh.y:
            knee_raised = "left"
            cv2.circle(frame, px(lk), 12, (0, 255, 0), -1)
            cv2.circle(frame, px(lh), 8,  (255, 255, 0), -1)

        if knee_raised and knee_raised != last_side:
            state["counter"]   = state.get("counter", 0) + 1
            state["last_side"] = knee_raised
            feedback_text      = f"Rep {state['counter']}! Switch knees"
            state["last_feedback_time"] = current_time
            correct_form = True
        elif not knee_raised and current_time - lft > feedback_duration:
            feedback_text = "Drive your knees up to hip level"
            state["last_feedback_time"] = current_time

        state["feedback_text"] = feedback_text

        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                   landmark_drawing_spec=drawing_spec,
                                   connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 180), thickness=2))
    except Exception:
        if current_time - state.get("last_feedback_time", 0) > 2:
            state["feedback_text"]      = "Position yourself in frame"
            state["last_feedback_time"] = current_time

    cv2.rectangle(frame, (0, frame_height-70), (frame_width, frame_height), (0, 0, 0), -1)
    cv2.putText(frame, f"High Knees: {state.get('counter', 0)}",
                (10, frame_height-40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, state.get("feedback_text", ""),
                (10, frame_height-15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    return frame, state, state.get("feedback_text", ""), state.get("counter", 0), correct_form


def high_knees_detection(user_id, log_exercise_fn):
    """Full Streamlit UI for High Knees detection."""
    col_left, col_right = st.columns(2)

    with col_right:
        st.subheader("🏃 High Knees — Correct Form")
        demo_tab, tips_tab = st.tabs(["📺 Demo Video", "📝 Form Tips"])
        with demo_tab:
            demo_path = _os.path.join(_APP_DIR, "demo/HighKnees/high_knees_demo.mp4")
            if os.path.exists(demo_path):
                with open(demo_path, "rb") as vf:
                    st.video(vf.read())
            else:
                st.info("Demo video not found.")
        with tips_tab:
            st.markdown("""
**Do's ✅**
- Drive knees up to at least hip height
- Pump your arms in sync with your legs
- Land softly on the balls of your feet
- Keep your core engaged throughout

**Don'ts ❌**
- Don't lean too far back
- Don't look down — keep eyes forward
- Don't let your arms go limp
- Don't slow down mid-set
            """)
            st.info("📐 **Rep counted** each time a knee crosses above hip level (alternating)")

    with col_left:
        st.subheader("📹 Your Input")
        input_type = st.radio("Choose Input Method",
                              ["📷 Realtime Camera", "🖼️ Upload Image", "🎬 Upload Video"],
                              horizontal=True, key="highknees_input_type")

        if input_type == "📷 Realtime Camera":
            run             = st.checkbox("▶️ Start Camera", key="highknees_camera_run")
            frame_window    = st.empty()
            dashboard_panel = st.empty()
            if run:
                cap       = cv2.VideoCapture(0)
                throttler = FeedbackThrottler(interval=3.0)
                state = {"counter": 0, "stage": None, "last_side": None,
                         "feedback_text": "Position yourself in frame", "last_feedback_time": time.time()}
                start_time = time.time()
                correct_frames = total_frames = 0
                while st.session_state.get("highknees_camera_run", False):
                    ret, frame = cap.read()
                    if not ret: break
                    frame = cv2.flip(frame, 1)
                    processed, state, feedback, counter, correct = process_high_knees_frame(frame, state)
                    elapsed = time.time() - start_time
                    total_frames += 1
                    if correct: correct_frames += 1
                    accuracy  = (correct_frames / total_frames * 100) if total_frames > 0 else 0
                    stage_val = str(state.get("stage", ""))
                    processed = draw_feedback_overlay(processed, feedback, correct, counter, stage_val, accuracy, "High Knees")
                    frame_window.image(processed, channels="BGR", use_container_width=True)
                    with dashboard_panel.container():
                        render_dashboard(counter, accuracy, stage_val, feedback, correct, "High Knees", elapsed)
                    speak_js(feedback, throttler)
                    time.sleep(0.05)
                duration = int(time.time() - start_time)
                cap.release()
                if duration > 5:
                    log_exercise_fn(user_id, "High Knees", duration, state["counter"] * 4, accuracy)
                    st.success(f"✅ Session saved — {state['counter']} reps, {accuracy:.0f}% accuracy")

        elif input_type == "🖼️ Upload Image":
            uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], key="highknees_img")
            if uploaded:
                from PIL import Image as PILImage
                img   = PILImage.open(uploaded)
                frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                state = {"counter": 0, "stage": None, "last_side": None, "feedback_text": "", "last_feedback_time": 0}
                processed, state, feedback, counter, correct = process_high_knees_frame(frame, state)
                st.image(processed, channels="BGR", use_container_width=True)
                if correct: st.success(f"✅ {feedback}")
                else:       st.warning(f"⚠️ {feedback}")

        else:
            uploaded = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"], key="highknees_vid")
            if uploaded:
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                tfile.write(uploaded.read()); tfile.close()
                cap   = cv2.VideoCapture(tfile.name)
                state = {"counter": 0, "stage": None, "last_side": None,
                         "feedback_text": "Analysing...", "last_feedback_time": 0}
                frame_ph = st.empty(); feedback_ph = st.empty()
                correct_frames = total_frames = 0
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret: break
                    processed, state, feedback, counter, correct = process_high_knees_frame(frame, state)
                    frame_ph.image(processed, channels="BGR", use_container_width=True)
                    total_frames += 1
                    if correct: correct_frames += 1
                    if any(w in feedback.lower() for w in ["rep", "good", "switch"]):
                        feedback_ph.success(f"✅ {feedback}")
                    else:
                        feedback_ph.warning(f"⚠️ {feedback}")
                    time.sleep(0.04)
                accuracy = (correct_frames / total_frames * 100) if total_frames > 0 else 0
                cap.release(); os.unlink(tfile.name)
                st.info(f"📊 Done — {state['counter']} reps, {accuracy:.0f}% accuracy")
