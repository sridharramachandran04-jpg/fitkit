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
_APP_DIR  = _os.path.dirname(_BASE_DIR)  # project root (one level up from exercises/)

mp_drawing = mp.solutions.drawing_utils
mp_pose    = mp.solutions.pose
drawing_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=1)

confidence_threshold = 0.5


def calculate_angle(a, b, c):
    a = np.array(a); b = np.array(b); c = np.array(c)
    ba = a - b; bc = c - b
    cos_a = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0)))


def make_sp_state():
    return {
        "counter": 0,
        "stage": None,           # "down" or "up"
        "feedback_text": "Get in position — arms at shoulder height",
        "last_feedback_time": time.time(),
        "left_buf":  deque(maxlen=6),
        "right_buf": deque(maxlen=6),
    }


def process_sp_frame(frame, state: dict) -> tuple:
    """
    Shoulder Press detection.
    DOWN: elbow angle ~90° (elbows bent, weights at shoulder level).
    UP  : elbow angle ~170° (arms fully extended overhead).
    Rep counted: DOWN → UP → back to DOWN.
    """
    feedback_duration = 2
    frame_height, frame_width = frame.shape[:2]
    current_time = time.time()

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=1) as pose:
        results = pose.process(image)
    image.flags.writeable = True
    frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    correct_form = False
    lft = state.get("last_feedback_time", 0)

    try:
        if results.pose_landmarks is None:
            raise Exception("No pose")

        lm = results.pose_landmarks.landmark
        def get(part): return lm[mp_pose.PoseLandmark[part].value]

        ls = get("LEFT_SHOULDER");   rs = get("RIGHT_SHOULDER")
        le = get("LEFT_ELBOW");      re = get("RIGHT_ELBOW")
        lw = get("LEFT_WRIST");      rw = get("RIGHT_WRIST")

        left_ok  = all(x.visibility > confidence_threshold for x in [ls, le, lw])
        right_ok = all(x.visibility > confidence_threshold for x in [rs, re, rw])

        if not (left_ok or right_ok):
            raise Exception("Arms not visible")

        angles = []
        if left_ok:
            angles.append(calculate_angle([ls.x,ls.y],[le.x,le.y],[lw.x,lw.y]))
        if right_ok:
            angles.append(calculate_angle([rs.x,rs.y],[re.x,re.y],[rw.x,rw.y]))

        avg_angle = sum(angles) / len(angles)

        if left_ok:
            state["left_buf"].append(calculate_angle([ls.x,ls.y],[le.x,le.y],[lw.x,lw.y]))
        if right_ok:
            state["right_buf"].append(calculate_angle([rs.x,rs.y],[re.x,re.y],[rw.x,rw.y]))

        bufs = list(state["left_buf"]) + list(state["right_buf"])
        smoothed = sum(bufs) / len(bufs) if bufs else avg_angle

        stage = state.get("stage")

        # DOWN — arms bent at ~90° (start position)
        if smoothed < 110:
            if stage != "down":
                state["stage"] = "down"
                if current_time - lft > feedback_duration:
                    state["feedback_text"] = "Good start — press up!"
                    state["last_feedback_time"] = current_time

        # UP — arms extended overhead
        elif smoothed > 155:
            if stage == "down":
                state["counter"] += 1
                state["stage"] = "up"
                state["feedback_text"] = f"Rep {state['counter']}! Lower back down"
                state["last_feedback_time"] = current_time
            elif stage != "up":
                state["stage"] = "up"
            correct_form = True

        else:
            # Mid-range feedback
            if current_time - lft > feedback_duration:
                if smoothed < 130:
                    state["feedback_text"] = "Push all the way up — extend fully!"
                else:
                    state["feedback_text"] = "Lower back to shoulder level"
                state["last_feedback_time"] = current_time

        # Back alignment: check wrists above shoulders (correct press position)
        if left_ok and right_ok:
            avg_wrist_y  = (lw.y + rw.y) / 2
            avg_shoulder_y = (ls.y + rs.y) / 2
            if smoothed > 155 and avg_wrist_y > avg_shoulder_y and current_time - lft > feedback_duration:
                state["feedback_text"] = "Push wrists above your head!"
                state["last_feedback_time"] = current_time

        # Draw
        def px(lm_obj): return (int(lm_obj.x * frame_width), int(lm_obj.y * frame_height))
        pairs = []
        if left_ok:  pairs += [(px(ls),px(le)),(px(le),px(lw))]
        if right_ok: pairs += [(px(rs),px(re)),(px(re),px(rw))]
        for a, b in pairs:
            cv2.line(frame, a, b, (255,255,0), 3)

        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                   landmark_drawing_spec=drawing_spec,
                                   connection_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0), thickness=2))

        cv2.putText(frame, f"Elbow angle: {smoothed:.0f}°", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.putText(frame, f"Stage: {state.get('stage','?')}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    except Exception:
        if current_time - state.get("last_feedback_time", 0) > 2:
            state["feedback_text"]      = "Stand in frame — show your arms"
            state["last_feedback_time"] = current_time

    # Bottom HUD
    cv2.rectangle(frame, (0, frame_height-70), (frame_width, frame_height), (0,0,0), -1)
    cv2.putText(frame, f"Shoulder Press: {state.get('counter',0)}",
                (10, frame_height-40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    cv2.putText(frame, state.get("feedback_text",""),
                (10, frame_height-15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    return frame, state, state.get("feedback_text",""), state.get("counter",0), correct_form


def shoulder_press_detection(user_id, log_exercise_fn):
    """Full Streamlit UI for Shoulder Press detection."""
    col_left, col_right = st.columns(2)

    with col_right:
        st.subheader("🎯 Shoulder Press — Correct Form")
        demo_tab, tips_tab = st.tabs(["📺 Demo", "📝 Form Tips"])

        with demo_tab:
            demo_path = _os.path.join(_APP_DIR, "demo/ShoulderPress/shoulder_press_demo.mp4")
            if os.path.exists(demo_path):
                with open(demo_path, "rb") as vf:
                    st.video(vf.read())
            else:
                st.info("🏋️ No demo video yet — add one to `demo/ShoulderPress/shoulder_press_demo.mp4`")
                st.markdown("""
**How a Shoulder Press looks:**
- Start: elbows bent ~90°, weights at shoulder height
- Press: push weights straight up, fully extend arms
- Lower: bring back to shoulder height with control
- That's **1 rep!**
                """)

        with tips_tab:
            st.markdown("""
**Do's ✅**
- Keep your core tight and spine neutral
- Press directly overhead — not forward
- Lower weights slowly and with control
- Keep feet flat on the ground if seated/standing

**Don'ts ❌**
- Don't arch your lower back
- Don't lock elbows at the top abruptly
- Don't shrug your shoulders up to your ears
- Don't let the weights drift forward
            """)
            st.info("📐 **Rep counted** when elbow goes from <110° (down) → >155° (fully extended overhead)")

    with col_left:
        st.subheader("📹 Your Input")
        input_type = st.radio("Choose Input Method",
                              ["📷 Realtime Camera", "🖼️ Upload Image", "🎬 Upload Video"],
                              horizontal=True, key="sp_input_type")

        if input_type == "📷 Realtime Camera":
            run             = st.checkbox("▶️ Start Camera", key="sp_camera_run")
            frame_window    = st.empty(); counter_disp = st.empty()
            feedback_disp   = st.empty(); accuracy_bar = st.empty()

            if run:
                cap        = cv2.VideoCapture(0)
                throttler  = FeedbackThrottler(interval=3.0)
                state = make_sp_state()
                start_time = time.time()
                correct_frames = total_frames = 0

                while st.session_state.get("sp_camera_run", False):
                    ret, frame = cap.read()
                    if not ret: break
                    frame = cv2.flip(frame, 1)
                    processed, state, feedback, counter, correct = process_sp_frame(frame, state)
                    frame_window.image(processed, channels="BGR", use_container_width=True)
                    total_frames += 1
                    if correct: correct_frames += 1
                    accuracy = (correct_frames / total_frames * 100) if total_frames > 0 else 0
                    accuracy_bar.progress(int(accuracy)/100, f"Form Accuracy: {accuracy:.0f}%")
                    counter_disp.metric("Shoulder Presses", counter)
                    if any(w in feedback.lower() for w in ["rep","great","good"]):
                        feedback_disp.success(f"✅ {feedback}")
                    else:
                        feedback_disp.warning(f"⚠️ {feedback}")
                    time.sleep(0.05)

                duration = int(time.time() - start_time)
                cap.release()
                if duration > 5:
                    log_exercise_fn(user_id, "Shoulder Press", duration, state["counter"] * 5, accuracy)
                    st.success(f"✅ Session saved — {state['counter']} reps, {accuracy:.0f}% accuracy")

        elif input_type == "🖼️ Upload Image":
            uploaded = st.file_uploader("Upload an image", type=["jpg","jpeg","png"], key="sp_img")
            if uploaded:
                img   = Image.open(uploaded)
                frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                state = make_sp_state()
                processed, state, feedback, counter, correct = process_sp_frame(frame, state)
                st.image(processed, channels="BGR", use_container_width=True)
                if correct: st.success(f"✅ {feedback}")
                else:       st.warning(f"⚠️ {feedback}")

        else:
            uploaded = st.file_uploader("Upload a video", type=["mp4","avi","mov"], key="sp_vid")
            if uploaded:
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                tfile.write(uploaded.read()); tfile.close()
                cap   = cv2.VideoCapture(tfile.name)
                state = make_sp_state()
                frame_ph = st.empty(); feedback_ph = st.empty()
                correct_frames = total_frames = 0

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret: break
                    processed, state, feedback, counter, correct = process_sp_frame(frame, state)
                    frame_ph.image(processed, channels="BGR", use_container_width=True)
                    total_frames += 1
                    if correct: correct_frames += 1
                    if any(w in feedback.lower() for w in ["rep","great","good"]):
                        feedback_ph.success(f"✅ {feedback}")
                    else:
                        feedback_ph.warning(f"⚠️ {feedback}")
                    time.sleep(0.04)

                accuracy = (correct_frames / total_frames * 100) if total_frames > 0 else 0
                cap.release(); os.unlink(tfile.name)
                st.info(f"📊 Done — {state['counter']} reps, {accuracy:.0f}% accuracy")
