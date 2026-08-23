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
max_smoothing_values = 5


def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))


def process_glute_bridge_frame(frame, state: dict) -> tuple:
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
    feedback_text = state.get("feedback_text", "Lie on your back — side view preferred")

    try:
        if results.pose_landmarks is None:
            raise Exception("No pose detected")

        lm = results.pose_landmarks.landmark

        rs = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        rh = lm[mp_pose.PoseLandmark.RIGHT_HIP.value]
        rk = lm[mp_pose.PoseLandmark.RIGHT_KNEE.value]
        ls = lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        lh = lm[mp_pose.PoseLandmark.LEFT_HIP.value]
        lk = lm[mp_pose.PoseLandmark.LEFT_KNEE.value]

        right_ok = all(x.visibility > confidence_threshold for x in [rs, rh, rk])
        left_ok  = all(x.visibility > confidence_threshold for x in [ls, lh, lk])

        if right_ok and (not left_ok or rh.visibility >= lh.visibility):
            shoulder, hip, knee = [rs.x, rs.y], [rh.x, rh.y], [rk.x, rk.y]
            sh_lm, hi_lm, kn_lm = rs, rh, rk
        elif left_ok:
            shoulder, hip, knee = [ls.x, ls.y], [lh.x, lh.y], [lk.x, lk.y]
            sh_lm, hi_lm, kn_lm = ls, lh, lk
        else:
            raise Exception("Key joints not visible")

        angle = calculate_angle(shoulder, hip, knee)

        win = state.get("smoothing_window", [])
        win.append(angle)
        if len(win) > max_smoothing_values: win.pop(0)
        state["smoothing_window"] = win
        smoothed = sum(win) / len(win)
        state["last_angle"] = smoothed

        def px(c): return (int(c[0] * frame_width), int(c[1] * frame_height))
        sp, hp2, kp = px(shoulder), px(hip), px(knee)
        cv2.line(frame, sp, hp2, (255, 200, 0), 3)
        cv2.line(frame, hp2, kp,  (255, 200, 0), 3)
        for pt in [sp, hp2, kp]:
            cv2.circle(frame, pt, 8, (0, 100, 255), -1)
        cv2.putText(frame, f"Angle: {smoothed:.1f}", (hp2[0]-50, hp2[1]-20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        stage = state.get("stage", None)
        lft   = state.get("last_feedback_time", 0)

        # Glute bridge: hip angle <120° = down (bent at hip), >160° = bridge up (hip extended)
        if smoothed < 120:
            if stage != "down":
                state["stage"] = "down"
                feedback_text  = "Push hips up — squeeze your glutes"
                state["last_feedback_time"] = current_time
        elif smoothed > 160 and stage == "down":
            state["stage"]   = "up"
            state["counter"] = state.get("counter", 0) + 1
            feedback_text    = f"Rep {state['counter']} complete! Hold & squeeze"
            state["last_feedback_time"] = current_time
            correct_form = True
        elif stage == "up" and current_time - lft > feedback_duration:
            feedback_text = "Lower slowly, then drive back up"
            state["last_feedback_time"] = current_time

        correct_form = (state.get("stage") == "up")
        state["feedback_text"] = feedback_text

        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                   landmark_drawing_spec=drawing_spec,
                                   connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 180, 0), thickness=2))
    except Exception:
        if current_time - state.get("last_feedback_time", 0) > 2:
            state["feedback_text"]      = "Lie on your back — side view preferred"
            state["last_feedback_time"] = current_time

    cv2.rectangle(frame, (0, frame_height-70), (frame_width, frame_height), (0, 0, 0), -1)
    cv2.putText(frame, f"Glute Bridge: {state.get('counter', 0)}",
                (10, frame_height-40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, state.get("feedback_text", ""),
                (10, frame_height-15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    return frame, state, state.get("feedback_text", ""), state.get("counter", 0), correct_form


def glute_bridge_detection(user_id, log_exercise_fn):
    """Full Streamlit UI for Glute Bridge detection."""
    col_left, col_right = st.columns(2)

    with col_right:
        st.subheader("🍑 Glute Bridge — Correct Form")
        demo_tab, tips_tab = st.tabs(["📺 Demo Video", "📝 Form Tips"])
        with demo_tab:
            demo_path = _os.path.join(_APP_DIR, "demo/GluteBridge/glute_bridge_demo.mp4")
            if os.path.exists(demo_path):
                with open(demo_path, "rb") as vf:
                    st.video(vf.read())
            else:
                st.info("Demo video not found.")
        with tips_tab:
            st.markdown("""
**Do's ✅**
- Lie flat on your back, knees bent, feet flat on the floor
- Drive hips up until body forms a straight line shoulder→hip→knee
- Squeeze glutes hard at the top and hold briefly
- Lower slowly under control

**Don'ts ❌**
- Don't hyperextend your lower back at the top
- Don't let your knees cave inward
- Don't use your lower back — it's a glute exercise
- Don't drop your hips quickly
            """)
            st.info("📐 **Rep counted** when hip angle goes from <120° (down) to >160° (bridge up)")

    with col_left:
        st.subheader("📹 Your Input")
        input_type = st.radio("Choose Input Method",
                              ["📷 Realtime Camera", "🖼️ Upload Image", "🎬 Upload Video"],
                              horizontal=True, key="glute_input_type")

        if input_type == "📷 Realtime Camera":
            run             = st.checkbox("▶️ Start Camera", key="glute_camera_run")
            frame_window    = st.empty()
            dashboard_panel = st.empty()
            if run:
                cap       = cv2.VideoCapture(0)
                throttler = FeedbackThrottler(interval=3.0)
                state = {"counter": 0, "stage": None, "smoothing_window": [], "last_angle": 0,
                         "feedback_text": "Lie on your back — side view preferred", "last_feedback_time": time.time()}
                start_time = time.time()
                correct_frames = total_frames = 0
                while st.session_state.get("glute_camera_run", False):
                    ret, frame = cap.read()
                    if not ret: break
                    frame = cv2.flip(frame, 1)
                    processed, state, feedback, counter, correct = process_glute_bridge_frame(frame, state)
                    elapsed = time.time() - start_time
                    total_frames += 1
                    if correct: correct_frames += 1
                    accuracy  = (correct_frames / total_frames * 100) if total_frames > 0 else 0
                    stage_val = str(state.get("stage", ""))
                    processed = draw_feedback_overlay(processed, feedback, correct, counter, stage_val, accuracy, "Glute Bridge")
                    frame_window.image(processed, channels="BGR", use_container_width=True)
                    with dashboard_panel.container():
                        render_dashboard(counter, accuracy, stage_val, feedback, correct, "Glute Bridge", elapsed)
                    speak_js(feedback, throttler)
                    time.sleep(0.05)
                duration = int(time.time() - start_time)
                cap.release()
                if duration > 5:
                    log_exercise_fn(user_id, "Glute Bridge", duration, state["counter"] * 4, accuracy)
                    st.success(f"✅ Session saved — {state['counter']} reps, {accuracy:.0f}% accuracy")

        elif input_type == "🖼️ Upload Image":
            uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], key="glute_img")
            if uploaded:
                from PIL import Image as PILImage
                img   = PILImage.open(uploaded)
                frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                state = {"counter": 0, "stage": None, "smoothing_window": [], "last_angle": 0,
                         "feedback_text": "", "last_feedback_time": 0}
                processed, state, feedback, counter, correct = process_glute_bridge_frame(frame, state)
                st.image(processed, channels="BGR", use_container_width=True)
                if correct: st.success(f"✅ {feedback}")
                else:       st.warning(f"⚠️ {feedback}")

        else:
            uploaded = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"], key="glute_vid")
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
                    processed, state, feedback, counter, correct = process_glute_bridge_frame(frame, state)
                    frame_ph.image(processed, channels="BGR", use_container_width=True)
                    total_frames += 1
                    if correct: correct_frames += 1
                    if any(w in feedback.lower() for w in ["complete", "good", "rep", "squeeze"]):
                        feedback_ph.success(f"✅ {feedback}")
                    else:
                        feedback_ph.warning(f"⚠️ {feedback}")
                    time.sleep(0.04)
                accuracy = (correct_frames / total_frames * 100) if total_frames > 0 else 0
                cap.release(); os.unlink(tfile.name)
                st.info(f"📊 Done — {state['counter']} reps, {accuracy:.0f}% accuracy")
