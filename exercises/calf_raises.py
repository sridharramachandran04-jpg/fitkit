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


def process_calf_raise_frame(frame, state: dict) -> tuple:
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
    feedback_text = state.get("feedback_text", "Stand sideways — full body in frame")

    try:
        if results.pose_landmarks is None:
            raise Exception("No pose detected")

        lm = results.pose_landmarks.landmark

        # Use both ankles + heels; pick the more visible side
        ra = lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
        rh = lm[mp_pose.PoseLandmark.RIGHT_HEEL.value]
        rk = lm[mp_pose.PoseLandmark.RIGHT_KNEE.value]
        la = lm[mp_pose.PoseLandmark.LEFT_ANKLE.value]
        lh = lm[mp_pose.PoseLandmark.LEFT_HEEL.value]
        lk = lm[mp_pose.PoseLandmark.LEFT_KNEE.value]

        right_ok = all(x.visibility > confidence_threshold for x in [ra, rh, rk])
        left_ok  = all(x.visibility > confidence_threshold for x in [la, lh, lk])

        if right_ok and (not left_ok or ra.visibility >= la.visibility):
            ankle_y, heel_y, knee_y, side = ra.y, rh.y, rk.y, "Right"
            ankle_x, heel_x = ra.x, rh.x
        elif left_ok:
            ankle_y, heel_y, knee_y, side = la.y, lh.y, lk.y, "Left"
            ankle_x, heel_x = la.x, lh.x
        else:
            raise Exception("Feet not visible")

        # In MediaPipe, smaller y = higher on screen = foot raised
        rise = heel_y - ankle_y          # positive when heel is lower than ankle (flat)
        stage = state.get("stage", None)
        lft   = state.get("last_feedback_time", 0)

        # Draw ankle + heel dots
        def px(x, y): return (int(x * frame_width), int(y * frame_height))
        ap = px(ankle_x, ankle_y)
        hp = px(heel_x,  heel_y)
        cv2.circle(frame, ap, 8, (0, 255, 128), -1)
        cv2.circle(frame, hp, 8, (0, 128, 255), -1)
        cv2.line(frame, ap, hp, (255, 255, 0), 3)
        cv2.putText(frame, side, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if rise < -0.02:          # heel is higher than ankle → raised
            if stage != "up":
                state["stage"] = "up"
                feedback_text  = "Good rise! Hold briefly"
                state["last_feedback_time"] = current_time
            correct_form = True
        else:                     # heel flat / down
            if stage == "up":
                state["stage"]   = "down"
                state["counter"] = state.get("counter", 0) + 1
                feedback_text    = f"Rep {state['counter']} complete! Rise again"
                state["last_feedback_time"] = current_time
            elif stage is None:
                if current_time - lft > feedback_duration:
                    feedback_text = "Rise up onto your toes"
                    state["last_feedback_time"] = current_time

        state["feedback_text"] = feedback_text

        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                   landmark_drawing_spec=drawing_spec,
                                   connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 100), thickness=2))
    except Exception:
        if current_time - state.get("last_feedback_time", 0) > 2:
            state["feedback_text"]      = "Stand sideways — full body in frame"
            state["last_feedback_time"] = current_time

    cv2.rectangle(frame, (0, frame_height-70), (frame_width, frame_height), (0, 0, 0), -1)
    cv2.putText(frame, f"Calf Raises: {state.get('counter', 0)}",
                (10, frame_height-40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, state.get("feedback_text", ""),
                (10, frame_height-15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    return frame, state, state.get("feedback_text", ""), state.get("counter", 0), correct_form


def calf_raise_detection(user_id, log_exercise_fn):
    """Full Streamlit UI for Calf Raise detection."""
    col_left, col_right = st.columns(2)

    with col_right:
        st.subheader("🦵 Calf Raise — Correct Form")
        demo_tab, tips_tab = st.tabs(["📺 Demo Video", "📝 Form Tips"])
        with demo_tab:
            demo_path = _os.path.join(_APP_DIR, "demo/CalfRaise/calf_raise_demo.mp4")
            if os.path.exists(demo_path):
                with open(demo_path, "rb") as vf:
                    st.video(vf.read())
            else:
                st.info("Demo video not found.")
        with tips_tab:
            st.markdown("""
**Do's ✅**
- Stand with feet hip-width apart, toes pointing forward
- Rise as high as possible onto your toes
- Hold briefly at the top for full contraction
- Lower slowly and with control

**Don'ts ❌**
- Don't bounce at the bottom
- Don't let your ankles roll outward
- Don't rush — slow, controlled reps are better
- Don't lean forward — stay upright
            """)
            st.info("📐 **Rep counted** when heel rises above ankle level then returns down")

    with col_left:
        st.subheader("📹 Your Input")
        input_type = st.radio("Choose Input Method",
                              ["📷 Realtime Camera", "🖼️ Upload Image", "🎬 Upload Video"],
                              horizontal=True, key="calf_input_type")

        if input_type == "📷 Realtime Camera":
            run             = st.checkbox("▶️ Start Camera", key="calf_camera_run")
            frame_window    = st.empty()
            dashboard_panel = st.empty()
            if run:
                cap       = cv2.VideoCapture(0)
                throttler = FeedbackThrottler(interval=3.0)
                state = {"counter": 0, "stage": None, "feedback_text": "Stand sideways — full body in frame",
                         "last_feedback_time": time.time()}
                start_time = time.time()
                correct_frames = total_frames = 0
                while st.session_state.get("calf_camera_run", False):
                    ret, frame = cap.read()
                    if not ret: break
                    frame = cv2.flip(frame, 1)
                    processed, state, feedback, counter, correct = process_calf_raise_frame(frame, state)
                    elapsed = time.time() - start_time
                    total_frames += 1
                    if correct: correct_frames += 1
                    accuracy  = (correct_frames / total_frames * 100) if total_frames > 0 else 0
                    stage_val = str(state.get("stage", ""))
                    processed = draw_feedback_overlay(processed, feedback, correct, counter, stage_val, accuracy, "Calf Raise")
                    frame_window.image(processed, channels="BGR", use_container_width=True)
                    with dashboard_panel.container():
                        render_dashboard(counter, accuracy, stage_val, feedback, correct, "Calf Raise", elapsed)
                    speak_js(feedback, throttler)
                    time.sleep(0.05)
                duration = int(time.time() - start_time)
                cap.release()
                if duration > 5:
                    log_exercise_fn(user_id, "Calf Raise", duration, state["counter"] * 4, accuracy)
                    st.success(f"✅ Session saved — {state['counter']} reps, {accuracy:.0f}% accuracy")

        elif input_type == "🖼️ Upload Image":
            uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], key="calf_img")
            if uploaded:
                from PIL import Image as PILImage
                img   = PILImage.open(uploaded)
                frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                state = {"counter": 0, "stage": None, "feedback_text": "", "last_feedback_time": 0}
                processed, state, feedback, counter, correct = process_calf_raise_frame(frame, state)
                st.image(processed, channels="BGR", use_container_width=True)
                if correct: st.success(f"✅ {feedback}")
                else:       st.warning(f"⚠️ {feedback}")

        else:
            uploaded = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"], key="calf_vid")
            if uploaded:
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                tfile.write(uploaded.read()); tfile.close()
                cap   = cv2.VideoCapture(tfile.name)
                state = {"counter": 0, "stage": None, "feedback_text": "Analysing...", "last_feedback_time": 0}
                frame_ph = st.empty(); feedback_ph = st.empty()
                correct_frames = total_frames = 0
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret: break
                    processed, state, feedback, counter, correct = process_calf_raise_frame(frame, state)
                    frame_ph.image(processed, channels="BGR", use_container_width=True)
                    total_frames += 1
                    if correct: correct_frames += 1
                    if any(w in feedback.lower() for w in ["complete", "good", "rep"]):
                        feedback_ph.success(f"✅ {feedback}")
                    else:
                        feedback_ph.warning(f"⚠️ {feedback}")
                    time.sleep(0.04)
                accuracy = (correct_frames / total_frames * 100) if total_frames > 0 else 0
                cap.release(); os.unlink(tfile.name)
                st.info(f"📊 Done — {state['counter']} reps, {accuracy:.0f}% accuracy")
