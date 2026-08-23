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


def make_jj_state():
    return {
        "counter": 0,
        "stage": None,          # "open" or "closed"
        "feedback_text": "Get in position — stand straight",
        "last_feedback_time": time.time(),
        "arm_buf": deque(maxlen=5),
        "leg_buf": deque(maxlen=5),
    }


def process_jj_frame(frame, state: dict) -> tuple:
    """
    Jumping Jack detection.
    Rep counted: arms go from down (< 30°) → up (> 130°) and back.
    Simultaneously legs spread out (hip-knee angle > 20°) and return.
    """
    feedback_duration = 2
    frame_height, frame_width = frame.shape[:2]
    current_time = time.time()

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
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
        lw = get("LEFT_WRIST");     rw = get("RIGHT_WRIST")
        lh = get("LEFT_HIP");       rh = get("RIGHT_HIP")
        lk = get("LEFT_KNEE");      rk = get("RIGHT_KNEE")

        vis_ok = all(x.visibility > confidence_threshold for x in [ls, rs, lw, rw, lh, rh])
        if not vis_ok:
            raise Exception("Not enough visibility")

        # Arm angle: shoulder-hip line vs wrist (measures arm raise)
        left_arm_angle  = calculate_angle([lw.x, lw.y], [ls.x, ls.y], [lh.x, lh.y])
        right_arm_angle = calculate_angle([rw.x, rw.y], [rs.x, rs.y], [rh.x, rh.y])
        avg_arm = (left_arm_angle + right_arm_angle) / 2

        # Leg spread: distance between ankles relative to hip width
        try:
            la = get("LEFT_ANKLE"); ra = get("RIGHT_ANKLE")
            leg_spread = abs(la.x - ra.x)
            hip_width  = abs(lh.x  - rh.x)
            leg_ratio  = leg_spread / (hip_width + 0.001)
        except Exception:
            leg_ratio = 1.0

        state["arm_buf"].append(avg_arm)
        state["leg_buf"].append(leg_ratio)
        sm_arm = sum(state["arm_buf"]) / len(state["arm_buf"])
        sm_leg = sum(state["leg_buf"]) / len(state["leg_buf"])

        stage = state.get("stage")

        # CLOSED position: arms down (<40°) and legs together (ratio < 1.5)
        if sm_arm < 40 and sm_leg < 1.5:
            if stage != "closed":
                state["stage"] = "closed"
                if current_time - lft > feedback_duration:
                    state["feedback_text"] = "Arms down — jump out!"
                    state["last_feedback_time"] = current_time

        # OPEN position: arms up (>120°) and legs spread (ratio > 1.8)
        elif sm_arm > 120 and sm_leg > 1.8:
            if stage == "closed":
                state["counter"] += 1
                state["stage"] = "open"
                state["feedback_text"] = f"Rep {state['counter']}! Back together!"
                state["last_feedback_time"] = current_time
            elif stage != "open":
                state["stage"] = "open"

            correct_form = True

        else:
            # Mid-position feedback
            if current_time - lft > feedback_duration:
                if sm_arm < 90:
                    state["feedback_text"] = "Raise arms higher!"
                else:
                    state["feedback_text"] = "Spread legs wider!"
                state["last_feedback_time"] = current_time

        # Draw landmarks
        def px(lm_obj): return (int(lm_obj.x * frame_width), int(lm_obj.y * frame_height))
        for a, b in [(px(ls), px(lw)), (px(rs), px(rw)),
                     (px(ls), px(lh)), (px(rs), px(rh)),
                     (px(lh), px(lk)), (px(rh), px(rk))]:
            cv2.line(frame, a, b, (255, 255, 0), 3)

        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                   landmark_drawing_spec=drawing_spec,
                                   connection_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0), thickness=2))

        cv2.putText(frame, f"Arm angle: {sm_arm:.0f}°", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.putText(frame, f"Leg spread: {sm_leg:.2f}x", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    except Exception:
        if current_time - state.get("last_feedback_time", 0) > 2:
            state["feedback_text"]      = "Stand fully in frame"
            state["last_feedback_time"] = current_time

    # Bottom HUD
    cv2.rectangle(frame, (0, frame_height-70), (frame_width, frame_height), (0,0,0), -1)
    cv2.putText(frame, f"Jumping Jacks: {state.get('counter', 0)}",
                (10, frame_height-40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    cv2.putText(frame, state.get("feedback_text",""),
                (10, frame_height-15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    return frame, state, state.get("feedback_text",""), state.get("counter",0), correct_form


def jumping_jack_detection(user_id, log_exercise_fn):
    """Full Streamlit UI for Jumping Jack detection."""
    col_left, col_right = st.columns(2)

    with col_right:
        st.subheader("🎯 Jumping Jack — Correct Form")
        demo_tab, tips_tab = st.tabs(["📺 Demo", "📝 Form Tips"])

        with demo_tab:
            demo_path = _os.path.join(_APP_DIR, "demo/JumpingJack/jumping_jack_demo.mp4")
            if os.path.exists(demo_path):
                with open(demo_path, "rb") as vf:
                    st.video(vf.read())
            else:
                st.info("🏃 No demo video yet — add one to `demo/JumpingJack/jumping_jack_demo.mp4`")
                st.markdown("""
**How a Jumping Jack looks:**
- Start: feet together, arms at sides
- Jump: feet spread wide, arms raise overhead
- Return: feet together, arms back down
- That's **1 rep!**
                """)

        with tips_tab:
            st.markdown("""
**Do's ✅**
- Keep your core engaged throughout
- Land softly on the balls of your feet
- Fully extend arms overhead on each rep
- Maintain a steady, consistent rhythm

**Don'ts ❌**
- Don't lock your knees when landing
- Don't hunch your shoulders
- Don't look down — keep head up
- Don't rush to the point of losing form
            """)
            st.info("📐 **Rep counted** when arms go from <40° → >120° with legs spread simultaneously")

    with col_left:
        st.subheader("📹 Your Input")
        input_type = st.radio("Choose Input Method",
                              ["📷 Realtime Camera", "🖼️ Upload Image", "🎬 Upload Video"],
                              horizontal=True, key="jj_input_type")

        if input_type == "📷 Realtime Camera":
            run             = st.checkbox("▶️ Start Camera", key="jj_camera_run")
            frame_window    = st.empty(); counter_disp = st.empty()
            feedback_disp   = st.empty(); accuracy_bar = st.empty()

            if run:
                cap        = cv2.VideoCapture(0)
                throttler  = FeedbackThrottler(interval=3.0)
                state = make_jj_state()
                start_time = time.time()
                correct_frames = total_frames = 0

                while st.session_state.get("jj_camera_run", False):
                    ret, frame = cap.read()
                    if not ret: break
                    frame = cv2.flip(frame, 1)
                    processed, state, feedback, counter, correct = process_jj_frame(frame, state)
                    frame_window.image(processed, channels="BGR", use_container_width=True)
                    total_frames += 1
                    if correct: correct_frames += 1
                    accuracy = (correct_frames / total_frames * 100) if total_frames > 0 else 0
                    accuracy_bar.progress(int(accuracy)/100, f"Form Accuracy: {accuracy:.0f}%")
                    counter_disp.metric("Jumping Jacks", counter)
                    if any(w in feedback.lower() for w in ["rep", "great", "good"]):
                        feedback_disp.success(f"✅ {feedback}")
                    else:
                        feedback_disp.warning(f"⚠️ {feedback}")
                    time.sleep(0.05)

                duration = int(time.time() - start_time)
                cap.release()
                if duration > 5:
                    log_exercise_fn(user_id, "Jumping Jack", duration, state["counter"] * 8, accuracy)
                    st.success(f"✅ Session saved — {state['counter']} reps, {accuracy:.0f}% accuracy")

        elif input_type == "🖼️ Upload Image":
            uploaded = st.file_uploader("Upload an image", type=["jpg","jpeg","png"], key="jj_img")
            if uploaded:
                img   = Image.open(uploaded)
                frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                state = make_jj_state()
                processed, state, feedback, counter, correct = process_jj_frame(frame, state)
                st.image(processed, channels="BGR", use_container_width=True)
                if correct: st.success(f"✅ {feedback}")
                else:       st.warning(f"⚠️ {feedback}")

        else:
            uploaded = st.file_uploader("Upload a video", type=["mp4","avi","mov"], key="jj_vid")
            if uploaded:
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                tfile.write(uploaded.read()); tfile.close()
                cap   = cv2.VideoCapture(tfile.name)
                state = make_jj_state()
                frame_ph = st.empty(); feedback_ph = st.empty()
                correct_frames = total_frames = 0

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret: break
                    processed, state, feedback, counter, correct = process_jj_frame(frame, state)
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
