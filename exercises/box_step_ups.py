import cv2, mediapipe as mp, numpy as np, time, os, tempfile
from PIL import Image
import streamlit as st
from collections import deque
from realtime_feedback import draw_feedback_overlay, render_dashboard, speak_js, FeedbackThrottler
import os as _os
_BASE_DIR=_os.path.dirname(_os.path.abspath(__file__)); _APP_DIR=_os.path.dirname(_BASE_DIR)
mp_drawing=mp.solutions.drawing_utils; mp_pose=mp.solutions.pose
drawing_spec=mp_drawing.DrawingSpec(thickness=2,circle_radius=1); confidence_threshold=0.5

def calculate_angle(a,b,c):
    a,b,c=np.array(a),np.array(b),np.array(c); ba,bc=a-b,c-b
    return np.degrees(np.arccos(np.clip(np.dot(ba,bc)/(np.linalg.norm(ba)*np.linalg.norm(bc)),-1,1)))

def make_bsu_state():
    return {"counter":0,"stage":None,"feedback_text":"Step onto the box — drive through heel",
            "last_feedback_time":time.time(),"buf":deque(maxlen=6),"last_hip_y":0.5}

def process_bsu_frame(frame, state):
    """Box Step-Up — detects hip rise indicating step-up.
    Tracks the normalised hip Y-position change.
    DOWN: hip at lower level (larger y)
    UP  : hip rises (smaller y) as person steps up
    Rep counted each time hip clearly rises then lowers.
    """
    fd=1.5; h,w=frame.shape[:2]; t=time.time()
    img=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB); img.flags.writeable=False
    with mp_pose.Pose(min_detection_confidence=0.5,min_tracking_confidence=0.5) as pose:
        res=pose.process(img)
    img.flags.writeable=True; frame=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    correct=False; lft=state.get("last_feedback_time",0)
    try:
        if not res.pose_landmarks: raise Exception()
        lm=res.pose_landmarks.landmark
        def g(p): return lm[mp_pose.PoseLandmark[p].value]
        lh,rh=g("LEFT_HIP"),g("RIGHT_HIP")
        lk,rk=g("LEFT_KNEE"),g("RIGHT_KNEE")
        lok=lh.visibility>confidence_threshold; rok=rh.visibility>confidence_threshold
        if not(lok or rok): raise Exception()
        hip_y=(lh.y+rh.y)/2 if lok and rok else (lh.y if lok else rh.y)
        lk_ang=calculate_angle([lh.x,lh.y],[lk.x,lk.y],[g("LEFT_ANKLE").x,g("LEFT_ANKLE").y]) if lok else 180
        state["buf"].append(hip_y); sm=sum(state["buf"])/len(state["buf"])
        last=state.get("last_hip_y",sm); state["last_hip_y"]=sm
        stg=state.get("stage")
        if sm<0.48:  # hip high = stepped up
            if stg!="up":
                state["stage"]="up"; correct=True
                if t-lft>fd: state["feedback_text"]="On top! Step back down"; state["last_feedback_time"]=t
        elif sm>0.58:  # hip low = on ground
            if stg=="up":
                state["counter"]+=1; state["stage"]="down"
                state["feedback_text"]=f"Rep {state['counter']}! Step back up!"; state["last_feedback_time"]=t
            elif stg!="down":
                state["stage"]="down"
                if t-lft>fd: state["feedback_text"]="Step up — drive through heel!"; state["last_feedback_time"]=t
        mp_drawing.draw_landmarks(frame,res.pose_landmarks,mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=drawing_spec,
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0),thickness=2))
        cv2.putText(frame,f"Hip Y: {sm:.2f}",(10,60),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)
    except:
        if t-state.get("last_feedback_time",0)>2:
            state["feedback_text"]="Stand sideways to camera — show full body"; state["last_feedback_time"]=t
    cv2.rectangle(frame,(0,h-70),(w,h),(0,0,0),-1)
    cv2.putText(frame,f"Step-Ups: {state.get('counter',0)}",(10,h-40),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
    cv2.putText(frame,state.get("feedback_text",""),(10,h-15),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
    return frame,state,state.get("feedback_text",""),state.get("counter",0),correct

def bsu_detection(user_id, log_exercise_fn):
    col_left,col_right=st.columns(2)
    with col_right:
        st.subheader("🎯 Box Step-Up — Correct Form")
        demo_tab,tips_tab=st.tabs(["📺 Demo","📝 Form Tips"])
        with demo_tab:
            dp=_os.path.join(_APP_DIR,"demo/BoxStepUp/box_step_ups_demo.mp4")
            if os.path.exists(dp):
                with open(dp,"rb") as vf: st.video(vf.read())
            else: st.info("Add demo to `demo/BoxStepUp/box_step_ups_demo.mp4`")
        with tips_tab:
            st.markdown("""**Do's ✅**\n- Drive through the heel of the stepping leg\n- Fully extend your hip at the top\n- Step down with control — one foot at a time\n- Keep chest up and core braced\n\n**Don'ts ❌**\n- Don't push off the back foot\n- Don't lean forward excessively\n- Don't let the knee cave inward\n- Don't skip full hip extension at the top""")
            st.info("📐 **Rep counted**: hip rises (step up) → hip lowers (step down) = 1 rep | Side view best")
    with col_left:
        st.subheader("📹 Your Input")
        itype=st.radio("Choose Input Method",["📷 Realtime Camera","🖼️ Upload Image","🎬 Upload Video"],horizontal=True,key="bsu_itype")
        if itype=="📷 Realtime Camera":
            run=st.checkbox("▶️ Start Camera",key="bsu_cam"); fw=st.empty(); dp2=st.empty()
            if run:
                cap=cv2.VideoCapture(0); state=make_bsu_state(); thr=FeedbackThrottler(3.0)
                st_t=time.time(); cf=tf=0
                while st.session_state.get("bsu_cam",False):
                    ret,frame=cap.read()
                    if not ret: break
                    frame=cv2.flip(frame,1)
                    proc,state,fb,cnt,cor=process_bsu_frame(frame,state)
                    el=time.time()-st_t; tf+=1
                    if cor: cf+=1
                    acc=(cf/tf*100) if tf>0 else 0
                    proc=draw_feedback_overlay(proc,fb,cor,cnt,str(state.get("stage","")),acc,"Box Step-Up")
                    fw.image(proc,channels="BGR",use_container_width=True)
                    with dp2.container(): render_dashboard(cnt,acc,str(state.get("stage","")),fb,cor,"Box Step-Up",el)
                    speak_js(fb,thr); time.sleep(0.05)
                dur=int(time.time()-st_t); cap.release()
                if dur>5:
                    log_exercise_fn(user_id,"Box Step-Up",dur,state["counter"]*6,acc)
                    st.success(f"✅ Saved — {state['counter']} reps, {acc:.0f}% accuracy")
        elif itype=="🖼️ Upload Image":
            up=st.file_uploader("Upload image",type=["jpg","jpeg","png"],key="bsu_img")
            if up:
                frame=cv2.cvtColor(np.array(Image.open(up)),cv2.COLOR_RGB2BGR)
                proc,s,fb,_,cor=process_bsu_frame(frame,make_bsu_state())
                st.image(proc,channels="BGR",use_container_width=True)
                st.success(f"✅ {fb}") if cor else st.warning(f"⚠️ {fb}")
        else:
            up=st.file_uploader("Upload video",type=["mp4","avi","mov"],key="bsu_vid")
            if up:
                tf2=tempfile.NamedTemporaryFile(delete=False,suffix=".mp4"); tf2.write(up.read()); tf2.close()
                cap=cv2.VideoCapture(tf2.name); state=make_bsu_state()
                fph=st.empty(); fbph=st.empty(); cf=tf=0
                while cap.isOpened():
                    ret,frame=cap.read()
                    if not ret: break
                    proc,state,fb,_,cor=process_bsu_frame(frame,state)
                    fph.image(proc,channels="BGR",use_container_width=True); tf+=1
                    if cor: cf+=1
                    fbph.success(f"✅ {fb}") if cor else fbph.warning(f"⚠️ {fb}")
                    time.sleep(0.04)
                acc=(cf/tf*100) if tf>0 else 0; cap.release(); os.unlink(tf2.name)
                st.info(f"📊 Done — {state['counter']} reps, {acc:.0f}% accuracy")
