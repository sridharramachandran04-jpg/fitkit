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


def make_dl_state():
    return {
        "counter": 0,
        "stage": None,          # "up" (standing) or "down" (hinge)
        "feedback_text": "Stand tall, bar over mid-foot",
        "last_feedback_time": time.time(),
        "hip_buf":  deque(maxlen=6),
        "back_buf": deque(maxlen=6),
    }


def process_dl_frame(frame, state: dict) -> tuple:
    """
    Deadlift detection.
    Tracks the hip hinge angle: shoulder→hip→knee.
    UP   (lockout)  : angle > 160°  (standing tall, hips extended)
    DOWN (hinge)    : angle < 100°  (hips pushed back, back parallel or near)
    Rep: UP → DOWN → UP

    Additionally checks back angle (shoulder→hip vs vertical) for rounding.
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
        lh = get("LEFT_HIP");       rh = get("RIGHT_HIP")
        lk = get("LEFT_KNEE");      rk = get("RIGHT_KNEE")
        la = get("LEFT_ANKLE");     ra = get("RIGHT_ANKLE")

        left_ok  = all(x.visibility > confidence_threshold for x in [ls, lh, lk, la])
        right_ok = all(x.visibility > confidence_threshold for x in [rs, rh, rk, ra])
        if not (left_ok or right_ok):
            raise Exception("Not enough landmarks")

        # Hip angle: shoulder→hip→knee (measures hinge)
        hip_angles = []
        if left_ok:
            hip_angles.append(calculate_angle([ls.x,ls.y],[lh.x,lh.y],[lk.x,lk.y]))
        if right_ok:
            hip_angles.append(calculate_angle([rs.x,rs.y],[rh.x,rh.y],[rk.x,rk.y]))
        avg_hip = sum(hip_angles) / len(hip_angles)
        state["hip_buf"].append(avg_hip)
        smoothed_hip = sum(state["hip_buf"]) / len(state["hip_buf"])

        # Back angle: how horizontal is the spine (shoulder y vs hip y)
        avg_shoulder_y = ((ls.y if left_ok else 0) + (rs.y if right_ok else 0)) / max(1, left_ok + right_ok)
        avg_hip_y      = ((lh.y if left_ok else 0) + (rh.y if right_ok else 0)) / max(1, left_ok + right_ok)
        back_angle_deg = abs(avg_shoulder_y - avg_hip_y) * 180  # rough vertical deviation

        stage = state.get("stage")

        # STANDING / LOCKOUT
        if smoothed_hip > 160:
            if stage != "up":
                state["stage"] = "up"
                if current_time - lft > feedback_duration:
                    state["feedback_text"] = "Good lockout! Now hinge and lower"
                    state["last_feedback_time"] = current_time

        # HINGED / BOTTOM
        elif smoothed_hip < 100:
            if stage == "up":
                state["counter"] += 1
                state["stage"] = "down"
                state["feedback_text"] = f"Rep {state['counter']}! Drive hips forward to stand"
                state["last_feedback_time"] = current_time
            elif stage != "down":
                state["stage"] = "down"
            correct_form = True

        else:
            if current_time - lft > feedback_duration:
                if smoothed_hip > 130:
                    state["feedback_text"] = "Hinge hips further back — push them behind you"
                else:
                    state["feedback_text"] = "Drive hips forward — stand tall and squeeze glutes"
                state["last_feedback_time"] = current_time

        # Back rounding check
        if stage == "down" and back_angle_deg < 15 and current_time - lft > feedback_duration:
            state["feedback_text"] = "Keep your back flat — don't round the spine!"
            state["last_feedback_time"] = current_time

        def px(o): return (int(o.x * frame_width), int(o.y * frame_height))
        if left_ok:
            for a, b in [(px(ls),px(lh)),(px(lh),px(lk)),(px(lk),px(la))]:
                cv2.line(frame, a, b, (255,255,0), 3)
        if right_ok:
            for a, b in [(px(rs),px(rh)),(px(rh),px(rk)),(px(rk),px(ra))]:
                cv2.line(frame, a, b, (255,255,0), 3)

        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                   landmark_drawing_spec=drawing_spec,
                                   connection_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0), thickness=2))
        cv2.putText(frame, f"Hip angle: {smoothed_hip:.0f}°", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.putText(frame, f"Stage: {state.get('stage','?')}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    except Exception:
        if current_time - state.get("last_feedback_time", 0) > 2:
            state["feedback_text"]      = "Stand sideways to camera — show full body"
            state["last_feedback_time"] = current_time

    cv2.rectangle(frame, (0, frame_height-70), (frame_width, frame_height), (0,0,0), -1)
    cv2.putText(frame, f"Deadlifts: {state.get('counter',0)}",
                (10, frame_height-40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    cv2.putText(frame, state.get("feedback_text",""),
                (10, frame_height-15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    return frame, state, state.get("feedback_text",""), state.get("counter",0), correct_form


def deadlift_detection(user_id, log_exercise_fn):
    col_left, col_right = st.columns(2)

    with col_right:
        st.subheader("🎯 Deadlift — Correct Form")
        demo_tab, tips_tab = st.tabs(["📺 Demo", "📝 Form Tips"])
        with demo_tab:
            demo_path = _os.path.join(_APP_DIR, "demo/Deadlift/deadlift_demo.mp4")
            if os.path.exists(demo_path):
                with open(demo_path, "rb") as vf: st.video(vf.read())
            else:
                st.info("🏋️ Add demo video to `demo/Deadlift/deadlift_demo.mp4`")
                st.markdown("""
**How a Deadlift looks:**
- Start: stand tall, bar over mid-foot, hips extended
- Hinge: push hips back, lower bar along legs
- Bottom: back parallel/near-parallel to floor
- Drive: push hips forward to stand tall — That's **1 rep!**
- 💡 **Tip:** Stand sideways to camera for best detection
                """)
        with tips_tab:
            st.markdown("""
**Do's ✅**
- Keep the bar close to your legs throughout
- Push hips back (not just bending knees)
- Keep your chest up and back flat
- Drive hips forward powerfully to stand up

**Don'ts ❌**
- Don't round your lower back — this is the #1 injury risk
- Don't let the bar drift away from your body
- Don't hyperextend at the top — just stand tall
- Don't jerk the bar — engage lats and pull slack first
            """)
            st.info("📐 **Rep counted** when hip angle goes from >160° (standing) → <100° (hinged) and back")
            st.warning("⚠️ **Best detected from a side view** — stand sideways to the camera")

    with col_left:
        st.subheader("📹 Your Input")
        input_type = st.radio("Choose Input Method",
                              ["📷 Realtime Camera", "🖼️ Upload Image", "🎬 Upload Video"],
                              horizontal=True, key="dl_input_type")

        if input_type == "📷 Realtime Camera":
            run = st.checkbox("▶️ Start Camera", key="dl_camera_run")
            frame_window = st.empty(); counter_disp = st.empty()
            feedback_disp = st.empty(); accuracy_bar = st.empty()
            if run:
                cap = cv2.VideoCapture(0); state = make_dl_state()
                start_time = time.time(); correct_frames = total_frames = 0
                while st.session_state.get("dl_camera_run", False):
                    ret, frame = cap.read()
                    if not ret: break
                    frame = cv2.flip(frame, 1)
                    processed, state, feedback, counter, correct = process_dl_frame(frame, state)
                    frame_window.image(processed, channels="BGR", use_container_width=True)
                    total_frames += 1
                    if correct: correct_frames += 1
                    accuracy = (correct_frames / total_frames * 100) if total_frames > 0 else 0
                    accuracy_bar.progress(int(accuracy)/100, f"Form Accuracy: {accuracy:.0f}%")
                    counter_disp.metric("Deadlifts", counter)
                    if any(w in feedback.lower() for w in ["rep","lockout","drive"]):
                        feedback_disp.success(f"✅ {feedback}")
                    else:
                        feedback_disp.warning(f"⚠️ {feedback}")
                    time.sleep(0.05)
                duration = int(time.time() - start_time); cap.release()
                if duration > 5:
                    log_exercise_fn(user_id, "Deadlift", duration, state["counter"] * 8, accuracy)
                    st.success(f"✅ Session saved — {state['counter']} reps, {accuracy:.0f}% accuracy")

        elif input_type == "🖼️ Upload Image":
            uploaded = st.file_uploader("Upload an image", type=["jpg","jpeg","png"], key="dl_img")
            if uploaded:
                frame = cv2.cvtColor(np.array(Image.open(uploaded)), cv2.COLOR_RGB2BGR)
                processed, state, feedback, counter, correct = process_dl_frame(frame, make_dl_state())
                st.image(processed, channels="BGR", use_container_width=True)
                st.success(f"✅ {feedback}") if correct else st.warning(f"⚠️ {feedback}")

        else:
            uploaded = st.file_uploader("Upload a video", type=["mp4","avi","mov"], key="dl_vid")
            if uploaded:
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                tfile.write(uploaded.read()); tfile.close()
                cap = cv2.VideoCapture(tfile.name); state = make_dl_state()
                frame_ph = st.empty(); feedback_ph = st.empty()
                correct_frames = total_frames = 0
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret: break
                    processed, state, feedback, counter, correct = process_dl_frame(frame, state)
                    frame_ph.image(processed, channels="BGR", use_container_width=True)
                    total_frames += 1
                    if correct: correct_frames += 1
                    if any(w in feedback.lower() for w in ["rep","lockout","drive"]):
                        feedback_ph.success(f"✅ {feedback}")
                    else:
                        feedback_ph.warning(f"⚠️ {feedback}")
                    time.sleep(0.04)
                accuracy = (correct_frames / total_frames * 100) if total_frames > 0 else 0
                cap.release(); os.unlink(tfile.name)
                st.info(f"📊 Done — {state['counter']} reps, {accuracy:.0f}% accuracy")
