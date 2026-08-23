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

confidence_threshold     = 0.5
TOP_POSITION_THRESHOLD   = 150
BOTTOM_POSITION_THRESHOLD= 90
MIN_ANGLE_CHANGE         = 3
MIN_REP_DEPTH            = 50
MIN_REP_INTERVAL         = 0.5

pushup_states = {'IDLE':0,'TOP_POSITION':1,'GOING_DOWN':2,'BOTTOM_POSITION':3,'GOING_UP':4}
state_names   = {v: k for k, v in pushup_states.items()}


def calculate_angle(a, b, c):
    a = np.array(a); b = np.array(b); c = np.array(c)
    ba = a - b;      bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))


def make_pushup_state():
    return {
        "counter": 0, "stage": None,
        "feedback_text": "Position yourself in frame",
        "last_feedback_time": time.time(),
        "elbow_buf": deque(maxlen=10),
        "shoulder_buf": deque(maxlen=10),
        "hip_buf": deque(maxlen=10),
        "current_state": pushup_states['IDLE'],
        "frames_in_state": 0,
        "last_rep_time": 0,
        "max_angle_in_rep": 0,
        "min_angle_in_rep": 180,
        "angle_history": deque(maxlen=30),
        "movement_direction": "none",
    }


def process_pushup_frame(frame, state: dict) -> tuple:
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

        def get(part): return lm[mp_pose.PoseLandmark[part].value]

        ls  = get("LEFT_SHOULDER");  rs  = get("RIGHT_SHOULDER")
        le  = get("LEFT_ELBOW");     re  = get("RIGHT_ELBOW")
        lw  = get("LEFT_WRIST");     rw  = get("RIGHT_WRIST")
        lh  = get("LEFT_HIP");       rh  = get("RIGHT_HIP")
        lk  = get("LEFT_KNEE");      rk  = get("RIGHT_KNEE")

        enough_visibility = all(
            x.visibility > confidence_threshold
            for x in [ls, le, lw, lh, rs, re, rw, rh]
        )

        if not enough_visibility:
            state["current_state"]   = pushup_states['IDLE']
            state["frames_in_state"] = 0
            if current_time - state.get("last_feedback_time", 0) > feedback_duration:
                state["feedback_text"]      = "Position your body so we can see you fully"
                state["last_feedback_time"] = current_time
            raise Exception("Not enough landmarks visible")

        def px(lm_obj):
            return (int(lm_obj.x * frame_width), int(lm_obj.y * frame_height))

        lsp, lep, lwp = px(ls), px(le), px(lw)
        lhp, lkp       = px(lh), px(lk)
        rsp, rep_, rwp = px(rs), px(re), px(rw)
        rhp, rkp       = px(rh), px(rk)

        # Angles
        left_elbow_a  = calculate_angle([ls.x,ls.y],[le.x,le.y],[lw.x,lw.y])
        right_elbow_a = calculate_angle([rs.x,rs.y],[re.x,re.y],[rw.x,rw.y])
        primary_elbow = left_elbow_a if le.visibility > re.visibility else right_elbow_a

        left_hip_a    = calculate_angle([ls.x,ls.y],[lh.x,lh.y],[lk.x,lk.y])
        right_hip_a   = calculate_angle([rs.x,rs.y],[rh.x,rh.y],[rk.x,rk.y])
        avg_hip       = (left_hip_a + right_hip_a) / 2

        left_shoulder_a  = calculate_angle([le.x,le.y],[ls.x,ls.y],[lh.x,lh.y])
        right_shoulder_a = calculate_angle([re.x,re.y],[rs.x,rs.y],[rh.x,rh.y])

        # Smoothing
        state["elbow_buf"].append(primary_elbow)
        state["hip_buf"].append(avg_hip)
        smoothed_elbow = sum(state["elbow_buf"]) / len(state["elbow_buf"])
        smoothed_hip   = sum(state["hip_buf"])   / len(state["hip_buf"])

        state["angle_history"].append(smoothed_elbow)
        if len(state["angle_history"]) >= 5:
            recent = list(state["angle_history"])[-5:]
            if   recent[0] - recent[-1] > MIN_ANGLE_CHANGE: state["movement_direction"] = "down"
            elif recent[-1] - recent[0] > MIN_ANGLE_CHANGE: state["movement_direction"] = "up"
            else:                                            state["movement_direction"] = "stable"

        # Draw skeleton
        for a, b in [(lsp,lep),(lep,lwp),(rsp,rep_),(rep_,rwp),
                     (lsp,lhp),(rsp,rhp),(lhp,lkp),(rhp,rkp)]:
            cv2.line(frame, a, b, (255,255,0), 3)
        for pt in [lsp,lep,lwp,lhp,lkp,rsp,rep_,rwp,rhp,rkp]:
            cv2.circle(frame, pt, 8, (0,0,255), -1)

        cv2.putText(frame, f"Elbow: {smoothed_elbow:.1f}", (10,60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.putText(frame, f"Back:  {smoothed_hip:.1f}",  (10,90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.putText(frame, f"State: {state_names[state['current_state']]}", (10,120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.putText(frame, f"Dir:   {state['movement_direction']}", (10,150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        # State machine
        cs  = state["current_state"]
        lft = state.get("last_feedback_time", 0)
        md  = state["movement_direction"]

        if cs == pushup_states['IDLE']:
            if smoothed_elbow > TOP_POSITION_THRESHOLD:
                state["current_state"]   = pushup_states['TOP_POSITION']
                state["frames_in_state"] = 1
                state["max_angle_in_rep"]= smoothed_elbow
                feedback_text            = "Good starting position"
                state["last_feedback_time"] = current_time
            else:
                state["frames_in_state"] = 0

        elif cs == pushup_states['TOP_POSITION']:
            state["frames_in_state"] += 1
            state["max_angle_in_rep"] = max(state["max_angle_in_rep"], smoothed_elbow)
            if md == "down" and smoothed_elbow < TOP_POSITION_THRESHOLD:
                state["current_state"]   = pushup_states['GOING_DOWN']
                state["frames_in_state"] = 1
                feedback_text            = "Going down, good form"
                state["last_feedback_time"] = current_time
            elif state["frames_in_state"] > 30 and current_time - lft > feedback_duration:
                feedback_text = "Lower yourself to start a push-up"
                state["last_feedback_time"] = current_time

        elif cs == pushup_states['GOING_DOWN']:
            state["frames_in_state"] += 1
            if smoothed_elbow < BOTTOM_POSITION_THRESHOLD:
                state["current_state"]    = pushup_states['BOTTOM_POSITION']
                state["frames_in_state"]  = 1
                state["min_angle_in_rep"] = smoothed_elbow
                feedback_text             = "Good depth!"
                state["last_feedback_time"] = current_time
            elif md == "up" and state["frames_in_state"] > 5:
                state["current_state"]    = pushup_states['GOING_UP']
                state["frames_in_state"]  = 1
                state["min_angle_in_rep"] = min(state["min_angle_in_rep"], smoothed_elbow)
                feedback_text             = "Try to go deeper next time"
                state["last_feedback_time"] = current_time
            elif state["frames_in_state"] > 20 and current_time - lft > feedback_duration:
                feedback_text = "Lower to complete the rep"
                state["last_feedback_time"] = current_time

        elif cs == pushup_states['BOTTOM_POSITION']:
            state["frames_in_state"] += 1
            state["min_angle_in_rep"] = min(state["min_angle_in_rep"], smoothed_elbow)
            if md == "up" and smoothed_elbow > BOTTOM_POSITION_THRESHOLD:
                state["current_state"]   = pushup_states['GOING_UP']
                state["frames_in_state"] = 1
                feedback_text            = "Push up, you got this!"
                state["last_feedback_time"] = current_time
            elif state["frames_in_state"] > 15 and current_time - lft > feedback_duration:
                feedback_text = "Push back up"
                state["last_feedback_time"] = current_time

        elif cs == pushup_states['GOING_UP']:
            state["frames_in_state"] += 1
            if smoothed_elbow > TOP_POSITION_THRESHOLD:
                rep_depth = state["max_angle_in_rep"] - state["min_angle_in_rep"]
                if rep_depth > MIN_REP_DEPTH and current_time - state["last_rep_time"] > MIN_REP_INTERVAL:
                    state["counter"]    += 1
                    feedback_text        = f"Rep {state['counter']} complete! Great job"
                    state["last_rep_time"] = current_time
                elif rep_depth <= MIN_REP_DEPTH:
                    feedback_text = "Not deep enough — go lower next rep"
                state["last_feedback_time"] = current_time
                state["current_state"]      = pushup_states['TOP_POSITION']
                state["frames_in_state"]    = 1
                state["max_angle_in_rep"]   = smoothed_elbow
                state["min_angle_in_rep"]   = 180
            elif state["frames_in_state"] > 20 and current_time - lft > feedback_duration:
                feedback_text = "Push all the way up"
                state["last_feedback_time"] = current_time

        # Back alignment check
        if cs != pushup_states['IDLE'] and smoothed_hip < 160 and current_time - state.get("last_feedback_time", 0) > feedback_duration:
            feedback_text = "Keep your back straight — don't sag!"
            state["last_feedback_time"] = current_time

        correct_form       = (state["current_state"] in [pushup_states['TOP_POSITION'], pushup_states['GOING_UP']])
        state["feedback_text"] = feedback_text

        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                   landmark_drawing_spec=drawing_spec,
                                   connection_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0), thickness=2))

    except Exception:
        if current_time - state.get("last_feedback_time", 0) > 2:
            state["feedback_text"]      = "Position yourself in frame"
            state["last_feedback_time"] = current_time

    # Bottom HUD
    stage_label = state_names.get(state.get("current_state", 0), "")
    cv2.rectangle(frame, (0, frame_height-70), (frame_width, frame_height), (0,0,0), -1)
    cv2.putText(frame, f"Push-ups: {state.get('counter',0)}",
                (10, frame_height-40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    cv2.putText(frame, state.get("feedback_text",""),
                (10, frame_height-15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
    col = (0,255,0) if "UP" in stage_label else (0,0,255) if "DOWN" in stage_label else (255,255,255)
    cv2.putText(frame, stage_label, (frame_width-220, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)

    return frame, state, state.get("feedback_text",""), state.get("counter",0), correct_form


def pushup_detection(user_id, log_exercise_fn):
    """Full Streamlit UI for push-up detection."""
    col_left, col_right = st.columns(2)

    with col_right:
        st.subheader("🎯 Push-up — Correct Form")

        demo_tab, tips_tab = st.tabs(["📺 Demo Video", "📝 Form Tips"])

        with demo_tab:
            demo_path = _os.path.join(_APP_DIR, "demo/Pushup/pushup_demo.mp4")
            if os.path.exists(demo_path):
                with open(demo_path, "rb") as vf:
                    st.video(vf.read())
            else:
                st.info("Demo video not found.")

        with tips_tab:
            st.markdown("""
**Do's ✅**
- Keep your body in a straight line from head to heels
- Lower your chest until it nearly touches the floor
- Keep elbows at ~45° from your torso
- Push all the way back up to full arm extension

**Don'ts ❌**
- Don't let your hips sag or pike up
- Don't flare your elbows out wide
- Don't only do partial reps
- Don't hold your breath
            """)
            st.info("📐 **Rep counted** when elbow goes below 90° (bottom) and back above 150° (top) with sufficient depth")
            st.markdown("""
**Form Points:**
- ✅ Hands slightly wider than shoulder-width
- ✅ Core tight throughout the movement
- ✅ Look slightly ahead, not straight down
- ❌ Don't lock elbows at the top
- ❌ Don't rush — control the descent
            """)

    with col_left:
        st.subheader("📹 Your Input")
        input_type = st.radio("Choose Input Method",
                              ["📷 Realtime Camera", "🖼️ Upload Image", "🎬 Upload Video"],
                              horizontal=True, key="pushup_input_type")

        if input_type == "📷 Realtime Camera":
            run             = st.checkbox("▶️ Start Camera", key="pushup_camera_run")
            frame_window    = st.empty(); counter_disp = st.empty()
            feedback_disp   = st.empty(); accuracy_bar = st.empty()

            if run:
                cap        = cv2.VideoCapture(0)
                throttler  = FeedbackThrottler(interval=3.0)
                state = make_pushup_state()
                start_time = time.time()
                correct_frames = total_frames = 0

                while st.session_state.get("pushup_camera_run", False):
                    ret, frame = cap.read()
                    if not ret: break
                    frame = cv2.flip(frame, 1)
                    processed, state, feedback, counter, correct = process_pushup_frame(frame, state)
                    frame_window.image(processed, channels="BGR", use_container_width=True)
                    total_frames += 1
                    if correct: correct_frames += 1
                    accuracy = (correct_frames / total_frames * 100) if total_frames > 0 else 0
                    accuracy_bar.progress(int(accuracy)/100, f"Form Accuracy: {accuracy:.0f}%")
                    counter_disp.metric("Reps Completed", counter)
                    if any(w in feedback.lower() for w in ["complete","good","great","depth"]):
                        feedback_disp.success(f"✅ {feedback}")
                    else:
                        feedback_disp.warning(f"⚠️ {feedback}")
                    time.sleep(0.05)

                duration = int(time.time() - start_time)
                cap.release()
                if duration > 5:
                    log_exercise_fn(user_id, "Push-up", duration, state["counter"]*5, accuracy)
                    st.success(f"✅ Session saved — {state['counter']} reps, {accuracy:.0f}% accuracy")

        elif input_type == "🖼️ Upload Image":
            uploaded = st.file_uploader("Upload an image", type=["jpg","jpeg","png"], key="pushup_img")
            if uploaded:
                img   = Image.open(uploaded)
                frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                state = make_pushup_state()
                processed, state, feedback, counter, correct = process_pushup_frame(frame, state)
                st.image(processed, channels="BGR", use_container_width=True)
                if correct: st.success(f"✅ {feedback}")
                else:       st.warning(f"⚠️ {feedback}")

        else:
            uploaded = st.file_uploader("Upload a video", type=["mp4","avi","mov"], key="pushup_vid")
            if uploaded:
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                tfile.write(uploaded.read()); tfile.close()
                cap   = cv2.VideoCapture(tfile.name)
                state = make_pushup_state()
                frame_ph = st.empty(); feedback_ph = st.empty()
                correct_frames = total_frames = 0

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret: break
                    processed, state, feedback, counter, correct = process_pushup_frame(frame, state)
                    frame_ph.image(processed, channels="BGR", use_container_width=True)
                    total_frames += 1
                    if correct: correct_frames += 1
                    if any(w in feedback.lower() for w in ["complete","good","great","depth"]):
                        feedback_ph.success(f"✅ {feedback}")
                    else:
                        feedback_ph.warning(f"⚠️ {feedback}")
                    time.sleep(0.04)

                accuracy = (correct_frames / total_frames * 100) if total_frames > 0 else 0
                cap.release(); os.unlink(tfile.name)
                st.info(f"📊 Done — {state['counter']} reps, {accuracy:.0f}% accuracy")
