import cv2
import mediapipe as mp
import numpy as np
import time
import os
import tempfile
from PIL import Image
import streamlit as st
from collections import deque
from realtime_feedback import draw_feedback_overlay, render_dashboard, speak_js, FeedbackThrottler
import os as _os
_BASE_DIR = _os.path.dirname(_os.path.abspath(__file__))
_APP_DIR  = _os.path.dirname(_BASE_DIR)

mp_drawing = mp.solutions.drawing_utils
mp_pose    = mp.solutions.pose
drawing_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=1)

confidence_threshold = 0.5
max_smoothing_values = 5
max_curl_angle       = 160


def calculate_angle(a, b, c):
    a = np.array(a); b = np.array(b); c = np.array(c)
    ba = a - b;      bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))


def process_hammer_curl_frame(frame, state: dict) -> tuple:
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
        ls = lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        le = lm[mp_pose.PoseLandmark.LEFT_ELBOW.value]
        lw = lm[mp_pose.PoseLandmark.LEFT_WRIST.value]
        rs = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        re = lm[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
        rw = lm[mp_pose.PoseLandmark.RIGHT_WRIST.value]

        left_ok  = all(x.visibility > confidence_threshold for x in [ls, le, lw])
        right_ok = all(x.visibility > confidence_threshold for x in [rs, re, rw])

        if left_ok and (not right_ok or le.visibility >= re.visibility):
            shoulder, elbow, wrist, arm_text = [ls.x, ls.y], [le.x, le.y], [lw.x, lw.y], "Left Arm"
        elif right_ok:
            shoulder, elbow, wrist, arm_text = [rs.x, rs.y], [re.x, re.y], [rw.x, rw.y], "Right Arm"
        else:
            raise Exception("Arms not visible")

        angle = calculate_angle(shoulder, elbow, wrist)

        win = state.get("smoothing_window", [])
        win.append(angle)
        if len(win) > max_smoothing_values: win.pop(0)
        state["smoothing_window"] = win
        smoothed = sum(win) / len(win)

        last_angle   = state.get("last_angle", smoothed)
        angle_change = abs(smoothed - last_angle)
        state["last_angle"] = smoothed

        def px(lm_coord):
            return (int(lm_coord[0] * frame_width), int(lm_coord[1] * frame_height))

        sp, ep, wp = px(shoulder), px(elbow), px(wrist)
        cv2.line(frame, sp, ep, (0, 255, 255), 3)
        cv2.line(frame, ep, wp, (0, 255, 255), 3)
        for pt in [sp, ep, wp]:
            cv2.circle(frame, pt, 8, (0, 128, 255), -1)
        cv2.putText(frame, f"Angle: {smoothed:.1f}", (ep[0]-50, ep[1]-20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, arm_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        stage = state.get("stage", None)
        lft   = state.get("last_feedback_time", 0)

        if smoothed > 140:
            if stage != "down":
                state["stage"] = "down"
                feedback_text  = "Now curl up — keep wrist neutral"
                state["last_feedback_time"] = current_time
            if smoothed > max_curl_angle and current_time - lft > feedback_duration:
                feedback_text = "Don't lock your elbow completely"
                state["last_feedback_time"] = current_time
        elif smoothed < 50 and stage == "down":
            state["stage"]   = "up"
            state["counter"] = state.get("counter", 0) + 1
            feedback_text    = f"Rep {state['counter']} complete! Lower slowly"
            state["last_feedback_time"] = current_time
        elif stage == "down" and 80 < smoothed < 110 and current_time - lft > feedback_duration:
            if angle_change < 2:
                feedback_text = "Keep moving — don't pause halfway"
                state["last_feedback_time"] = current_time
        elif angle_change > 15 and current_time - lft > feedback_duration:
            feedback_text = "Slow down for better muscle engagement"
            state["last_feedback_time"] = current_time

        correct_form = (state.get("stage") == "up")
        state["feedback_text"] = feedback_text

        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                   landmark_drawing_spec=drawing_spec,
                                   connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 128), thickness=2))

    except Exception:
        if current_time - state.get("last_feedback_time", 0) > 2:
            state["feedback_text"]      = "Position yourself in frame"
            state["last_feedback_time"] = current_time

    cv2.rectangle(frame, (0, frame_height-70), (frame_width, frame_height), (0, 0, 0), -1)
    cv2.putText(frame, f"Hammer Curls: {state.get('counter', 0)}",
                (10, frame_height-40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, state.get("feedback_text", ""),
                (10, frame_height-15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    return frame, state, state.get("feedback_text", ""), state.get("counter", 0), correct_form


def hammer_curls_detection(user_id, log_exercise_fn):
    """Full Streamlit UI for hammer curl detection."""
    col_left, col_right = st.columns(2)

    with col_right:
        st.subheader("🔨 Hammer Curl — Correct Form")

        demo_tab, tips_tab = st.tabs(["📺 Demo Video", "📝 Form Tips"])

        with demo_tab:
            demo_path = _os.path.join(_APP_DIR, "demo/HammerCurl/hammer_curl_demo.mp4")
            if os.path.exists(demo_path):
                with open(demo_path, "rb") as vf:
                    st.video(vf.read())
            else:
                st.info("Demo video not found.")

        with tips_tab:
            st.markdown("""
**Do's ✅**
- Keep your upper arm still and tucked at your side
- Use a neutral (hammer) grip — thumb pointing up
- Curl all the way up until the dumbbell reaches shoulder level
- Lower the weight slowly and with control

**Don'ts ❌**
- Don't swing your body or use momentum
- Don't rotate your wrist during the movement
- Don't rush — slow reps build more strength
- Don't let your elbows drift forward or flare out
            """)
            st.info("📐 **Rep counted** when elbow goes from >140° (down) to <50° (up)")
            st.markdown("""
**Form Points:**
- ✅ Stand straight, feet shoulder-width apart
- ✅ Keep wrist straight and neutral throughout
- ✅ Exhale as you curl up, inhale as you lower
- ❌ Don't supinate (rotate) the wrist
- ❌ Don't lean back to complete the rep
            """)

    with col_left:
        st.subheader("📹 Your Input")
        input_type = st.radio("Choose Input Method",
                              ["📷 Realtime Camera", "🖼️ Upload Image", "🎬 Upload Video"],
                              horizontal=True, key="hammer_input_type")

        if input_type == "📷 Realtime Camera":
            run             = st.checkbox("▶️ Start Camera", key="hammer_camera_run")
            frame_window    = st.empty()
            dashboard_panel = st.empty()

            if run:
                cap       = cv2.VideoCapture(0)
                throttler = FeedbackThrottler(interval=3.0)
                state = {"counter": 0, "stage": None, "smoothing_window": [], "last_angle": 0,
                         "feedback_text": "Position yourself in frame", "last_feedback_time": time.time()}
                start_time = time.time()
                correct_frames = total_frames = 0

                while st.session_state.get("hammer_camera_run", False):
                    ret, frame = cap.read()
                    if not ret: break
                    frame = cv2.flip(frame, 1)
                    processed, state, feedback, counter, correct = process_hammer_curl_frame(frame, state)
                    elapsed = time.time() - start_time
                    total_frames += 1
                    if correct: correct_frames += 1
                    accuracy  = (correct_frames / total_frames * 100) if total_frames > 0 else 0
                    stage_val = str(state.get("stage", ""))
                    processed = draw_feedback_overlay(
                        processed, feedback, correct, counter,
                        stage_val, accuracy, "Hammer Curl"
                    )
                    frame_window.image(processed, channels="BGR", use_container_width=True)
                    with dashboard_panel.container():
                        render_dashboard(counter, accuracy, stage_val,
                                         feedback, correct, "Hammer Curl", elapsed)
                    speak_js(feedback, throttler)
                    time.sleep(0.05)

                duration = int(time.time() - start_time)
                cap.release()
                if duration > 5:
                    log_exercise_fn(user_id, "Hammer Curl", duration, state["counter"] * 4, accuracy)
                    st.success(f"✅ Session saved — {state['counter']} reps, {accuracy:.0f}% accuracy")

        elif input_type == "🖼️ Upload Image":
            uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], key="hammer_img")
            if uploaded:
                img   = Image.open(uploaded)
                frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                state = {"counter": 0, "stage": None, "smoothing_window": [], "last_angle": 0,
                         "feedback_text": "", "last_feedback_time": 0}
                processed, state, feedback, counter, correct = process_hammer_curl_frame(frame, state)
                st.image(processed, channels="BGR", use_container_width=True)
                if correct: st.success(f"✅ {feedback}")
                else:       st.warning(f"⚠️ {feedback}")

        else:
            uploaded = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"], key="hammer_vid")
            if uploaded:
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                tfile.write(uploaded.read()); tfile.close()
                cap   = cv2.VideoCapture(tfile.name)
                state = {"counter": 0, "stage": None, "smoothing_window": [], "last_angle": 0,
                         "feedback_text": "Analysing...", "last_feedback_time": 0}
                frame_ph = st.empty(); feedback_ph = st.empty()
                correct_frames = total_frames = 0

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret: break
                    processed, state, feedback, counter, correct = process_hammer_curl_frame(frame, state)
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
