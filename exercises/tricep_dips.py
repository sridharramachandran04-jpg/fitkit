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


def make_dip_state():
    return {
        "counter": 0,
        "stage": None,          # "up" or "down"
        "feedback_text": "Position hands on chair/bench, arms extended",
        "last_feedback_time": time.time(),
        "left_buf":  deque(maxlen=6),
        "right_buf": deque(maxlen=6),
    }


def process_dip_frame(frame, state: dict) -> tuple:
    """
    Tricep Dip detection.
    Tracks shoulder→elbow→wrist angle.
    UP  : arms extended — angle > 155°
    DOWN: dipped       — angle < 90°
    Rep: UP → DOWN → UP
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

        ls = get("LEFT_SHOULDER");  rs = get("RIGHT_SHOULDER")
        le = get("LEFT_ELBOW");     re = get("RIGHT_ELBOW")
        lw = get("LEFT_WRIST");     rw = get("RIGHT_WRIST")

        left_ok  = all(x.visibility > confidence_threshold for x in [ls, le, lw])
        right_ok = all(x.visibility > confidence_threshold for x in [rs, re, rw])
        if not (left_ok or right_ok):
            raise Exception("Arms not visible")

        if left_ok:
            a = calculate_angle([ls.x,ls.y],[le.x,le.y],[lw.x,lw.y])
            state["left_buf"].append(a)
        if right_ok:
            a = calculate_angle([rs.x,rs.y],[re.x,re.y],[rw.x,rw.y])
            state["right_buf"].append(a)

        all_buf = list(state["left_buf"]) + list(state["right_buf"])
        smoothed = sum(all_buf) / len(all_buf)

        stage = state.get("stage")

        # UP — arms extended
        if smoothed > 155:
            if stage != "up":
                state["stage"] = "up"
                if current_time - lft > feedback_duration:
                    state["feedback_text"] = "Arms extended — now dip down!"
                    state["last_feedback_time"] = current_time

        # DOWN — arms bent at elbow
        elif smoothed < 90:
            if stage == "up":
                state["counter"] += 1
                state["stage"] = "down"
                state["feedback_text"] = f"Rep {state['counter']}! Push back up"
                state["last_feedback_time"] = current_time
            elif stage != "down":
                state["stage"] = "down"
            correct_form = True

        else:
            if current_time - lft > feedback_duration:
                if smoothed > 120:
                    state["feedback_text"] = "Dip lower — bend elbows to 90°"
                else:
                    state["feedback_text"] = "Push all the way up — extend arms fully"
                state["last_feedback_time"] = current_time

        # Elbow flare check: elbows should stay close to body
        if left_ok and right_ok:
            elbow_width    = abs(le.x - re.x)
            shoulder_width = abs(ls.x - rs.x)
            if elbow_width > shoulder_width * 1.4 and current_time - lft > feedback_duration:
                state["feedback_text"] = "Keep elbows tucked — don't flare them out!"
                state["last_feedback_time"] = current_time

        def px(o): return (int(o.x * frame_width), int(o.y * frame_height))
        if left_ok:
            cv2.line(frame, px(ls), px(le), (255,255,0), 3)
            cv2.line(frame, px(le), px(lw), (255,255,0), 3)
        if right_ok:
            cv2.line(frame, px(rs), px(re), (255,255,0), 3)
            cv2.line(frame, px(re), px(rw), (255,255,0), 3)

        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                   landmark_drawing_spec=drawing_spec,
                                   connection_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0), thickness=2))
        cv2.putText(frame, f"Elbow angle: {smoothed:.0f}°", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.putText(frame, f"Stage: {state.get('stage','?')}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    except Exception:
        if current_time - state.get("last_feedback_time", 0) > 2:
            state["feedback_text"]      = "Show arms in frame clearly"
            state["last_feedback_time"] = current_time

    cv2.rectangle(frame, (0, frame_height-70), (frame_width, frame_height), (0,0,0), -1)
    cv2.putText(frame, f"Tricep Dips: {state.get('counter',0)}",
                (10, frame_height-40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    cv2.putText(frame, state.get("feedback_text",""),
                (10, frame_height-15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    return frame, state, state.get("feedback_text",""), state.get("counter",0), correct_form


def tricep_dip_detection(user_id, log_exercise_fn):
    col_left, col_right = st.columns(2)

    with col_right:
        st.subheader("🎯 Tricep Dip — Correct Form")
        demo_tab, tips_tab = st.tabs(["📺 Demo", "📝 Form Tips"])
        with demo_tab:
            demo_path = _os.path.join(_APP_DIR, "demo/TricepDip/tricep_dip_demo.mp4")
            if os.path.exists(demo_path):
                with open(demo_path, "rb") as vf: st.video(vf.read())
            else:
                st.info("🏋️ Add demo video to `demo/TricepDip/tricep_dip_demo.mp4`")
                st.markdown("""
**How a Tricep Dip looks:**
- Sit on edge of chair/bench, hands gripping the edge
- Slide hips off, arms extended, legs out
- Lower body by bending elbows to ~90°
- Push back up to full arm extension — That's **1 rep!**
                """)
        with tips_tab:
            st.markdown("""
**Do's ✅**
- Keep your back close to the chair/bench
- Lower until elbows are at 90° angle
- Keep elbows pointing straight back — not flaring
- Fully extend arms at the top of each rep

**Don'ts ❌**
- Don't let elbows flare outward
- Don't shrug shoulders toward your ears
- Don't partially dip — go to 90° each time
- Don't rush — control both phases
            """)
            st.info("📐 **Rep counted** when elbow angle goes from >155° (up) → <90° (dipped) and back")

    with col_left:
        st.subheader("📹 Your Input")
        input_type = st.radio("Choose Input Method",
                              ["📷 Realtime Camera", "🖼️ Upload Image", "🎬 Upload Video"],
                              horizontal=True, key="dip_input_type")

        if input_type == "📷 Realtime Camera":
            run = st.checkbox("▶️ Start Camera", key="dip_camera_run")
            frame_window = st.empty(); counter_disp = st.empty()
            feedback_disp = st.empty(); accuracy_bar = st.empty()
            if run:
                cap = cv2.VideoCapture(0); state = make_dip_state()
                start_time = time.time(); correct_frames = total_frames = 0
                while st.session_state.get("dip_camera_run", False):
                    ret, frame = cap.read()
                    if not ret: break
                    frame = cv2.flip(frame, 1)
                    processed, state, feedback, counter, correct = process_dip_frame(frame, state)
                    frame_window.image(processed, channels="BGR", use_container_width=True)
                    total_frames += 1
                    if correct: correct_frames += 1
                    accuracy = (correct_frames / total_frames * 100) if total_frames > 0 else 0
                    accuracy_bar.progress(int(accuracy)/100, f"Form Accuracy: {accuracy:.0f}%")
                    counter_disp.metric("Tricep Dips", counter)
                    if any(w in feedback.lower() for w in ["rep","push","good"]):
                        feedback_disp.success(f"✅ {feedback}")
                    else:
                        feedback_disp.warning(f"⚠️ {feedback}")
                    time.sleep(0.05)
                duration = int(time.time() - start_time); cap.release()
                if duration > 5:
                    log_exercise_fn(user_id, "Tricep Dip", duration, state["counter"] * 5, accuracy)
                    st.success(f"✅ Session saved — {state['counter']} reps, {accuracy:.0f}% accuracy")

        elif input_type == "🖼️ Upload Image":
            uploaded = st.file_uploader("Upload an image", type=["jpg","jpeg","png"], key="dip_img")
            if uploaded:
                frame = cv2.cvtColor(np.array(Image.open(uploaded)), cv2.COLOR_RGB2BGR)
                processed, state, feedback, counter, correct = process_dip_frame(frame, make_dip_state())
                st.image(processed, channels="BGR", use_container_width=True)
                st.success(f"✅ {feedback}") if correct else st.warning(f"⚠️ {feedback}")

        else:
            uploaded = st.file_uploader("Upload a video", type=["mp4","avi","mov"], key="dip_vid")
            if uploaded:
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                tfile.write(uploaded.read()); tfile.close()
                cap = cv2.VideoCapture(tfile.name); state = make_dip_state()
                frame_ph = st.empty(); feedback_ph = st.empty()
                correct_frames = total_frames = 0
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret: break
                    processed, state, feedback, counter, correct = process_dip_frame(frame, state)
                    frame_ph.image(processed, channels="BGR", use_container_width=True)
                    total_frames += 1
                    if correct: correct_frames += 1
                    if any(w in feedback.lower() for w in ["rep","push","good"]):
                        feedback_ph.success(f"✅ {feedback}")
                    else:
                        feedback_ph.warning(f"⚠️ {feedback}")
                    time.sleep(0.04)
                accuracy = (correct_frames / total_frames * 100) if total_frames > 0 else 0
                cap.release(); os.unlink(tfile.name)
                st.info(f"📊 Done — {state['counter']} reps, {accuracy:.0f}% accuracy")
