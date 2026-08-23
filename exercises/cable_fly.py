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

def make_cf_state():
    return {"counter":0,"stage":None,"feedback_text":"Arms out wide — bring hands together",
            "last_feedback_time":time.time(),"left_buf":deque(maxlen=6),"right_buf":deque(maxlen=6)}

def process_cf_frame(frame, state):
    """Cable Fly / Chest Fly — tracks arm abduction angle (shoulder width).
    OPEN  : arms spread wide — wrist-to-wrist distance large
    CLOSED: hands meet in front — wrist-to-wrist distance small
    Rep: OPEN → CLOSED → OPEN
    """
    fd=2; h,w=frame.shape[:2]; t=time.time()
    img=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB); img.flags.writeable=False
    with mp_pose.Pose(min_detection_confidence=0.5,min_tracking_confidence=0.5) as pose:
        res=pose.process(img)
    img.flags.writeable=True; frame=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    correct=False; lft=state.get("last_feedback_time",0)
    try:
        if not res.pose_landmarks: raise Exception()
        lm=res.pose_landmarks.landmark
        def g(p): return lm[mp_pose.PoseLandmark[p].value]
        lw,rw=g("LEFT_WRIST"),g("RIGHT_WRIST")
        ls,rs=g("LEFT_SHOULDER"),g("RIGHT_SHOULDER")
        if lw.visibility<confidence_threshold or rw.visibility<confidence_threshold: raise Exception()
        wrist_dist=abs(lw.x-rw.x)
        sh_width=abs(ls.x-rs.x)+0.001
        ratio=wrist_dist/sh_width
        state["left_buf"].append(ratio)
        sm=sum(state["left_buf"])/len(state["left_buf"])
        stg=state.get("stage")
        if sm>2.0:  # arms wide open
            if stg!="open":
                state["stage"]="open"
                if t-lft>fd: state["feedback_text"]="Arms out wide — now bring together!"; state["last_feedback_time"]=t
        elif sm<0.6:  # hands together
            if stg=="open":
                state["counter"]+=1; state["stage"]="closed"
                state["feedback_text"]=f"Rep {state['counter']}! Open arms back out"; state["last_feedback_time"]=t
            elif stg!="closed": state["stage"]="closed"
            correct=True
        else:
            if t-lft>fd:
                state["feedback_text"]=("Bring hands fully together!" if sm>1.0 else "Open arms wider!")
                state["last_feedback_time"]=t
        def px(o): return (int(o.x*w),int(o.y*h))
        cv2.line(frame,px(lw),px(rw),(255,255,0),3)
        cv2.line(frame,px(ls),px(lw),(255,100,0),2)
        cv2.line(frame,px(rs),px(rw),(255,100,0),2)
        mp_drawing.draw_landmarks(frame,res.pose_landmarks,mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=drawing_spec,
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0),thickness=2))
        cv2.putText(frame,f"Arm spread: {sm:.2f}x",(10,60),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)
    except:
        if t-state.get("last_feedback_time",0)>2:
            state["feedback_text"]="Face camera — show both arms extended"; state["last_feedback_time"]=t
    cv2.rectangle(frame,(0,h-70),(w,h),(0,0,0),-1)
    cv2.putText(frame,f"Cable Fly: {state.get('counter',0)}",(10,h-40),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
    cv2.putText(frame,state.get("feedback_text",""),(10,h-15),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
    return frame,state,state.get("feedback_text",""),state.get("counter",0),correct

def cf_detection(user_id, log_exercise_fn):
    col_left,col_right=st.columns(2)
    with col_right:
        st.subheader("🎯 Cable Fly — Correct Form")
        demo_tab,tips_tab=st.tabs(["📺 Demo","📝 Form Tips"])
        with demo_tab:
            dp=_os.path.join(_APP_DIR,"demo/CableFly/cable_fly_demo.mp4")
            if os.path.exists(dp):
                with open(dp,"rb") as vf: st.video(vf.read())
            else: st.info("Add demo to `demo/CableFly/cable_fly_demo.mp4`")
        with tips_tab:
            st.markdown("""**Do's ✅**\n- Keep a slight bend in your elbows throughout\n- Bring hands together in a hugging arc\n- Squeeze chest at the peak contraction\n- Control the return — fight the resistance\n\n**Don'ts ❌**\n- Don't straighten your arms completely\n- Don't use your arms to press — it's a fly!\n- Don't rush the eccentric (opening) phase\n- Don't drop elbows below shoulder height""")
            st.info("📐 **Rep counted**: arms spread wide (>2x shoulder width) → hands meet (<0.6x) → open back")
    with col_left:
        st.subheader("📹 Your Input")
        itype=st.radio("Choose Input Method",["📷 Realtime Camera","🖼️ Upload Image","🎬 Upload Video"],horizontal=True,key="cf_itype")
        if itype=="📷 Realtime Camera":
            run=st.checkbox("▶️ Start Camera",key="cf_cam"); fw=st.empty(); dp2=st.empty()
            if run:
                cap=cv2.VideoCapture(0); state=make_cf_state(); thr=FeedbackThrottler(3.0)
                st_t=time.time(); cf=tf=0
                while st.session_state.get("cf_cam",False):
                    ret,frame=cap.read()
                    if not ret: break
                    frame=cv2.flip(frame,1)
                    proc,state,fb,cnt,cor=process_cf_frame(frame,state)
                    el=time.time()-st_t; tf+=1
                    if cor: cf+=1
                    acc=(cf/tf*100) if tf>0 else 0
                    proc=draw_feedback_overlay(proc,fb,cor,cnt,str(state.get("stage","")),acc,"Cable Fly")
                    fw.image(proc,channels="BGR",use_container_width=True)
                    with dp2.container(): render_dashboard(cnt,acc,str(state.get("stage","")),fb,cor,"Cable Fly",el)
                    speak_js(fb,thr); time.sleep(0.05)
                dur=int(time.time()-st_t); cap.release()
                if dur>5:
                    log_exercise_fn(user_id,"Cable Fly",dur,state["counter"]*4,acc)
                    st.success(f"✅ Saved — {state['counter']} reps, {acc:.0f}% accuracy")
        elif itype=="🖼️ Upload Image":
            up=st.file_uploader("Upload image",type=["jpg","jpeg","png"],key="cf_img")
            if up:
                frame=cv2.cvtColor(np.array(Image.open(up)),cv2.COLOR_RGB2BGR)
                proc,s,fb,_,cor=process_cf_frame(frame,make_cf_state())
                st.image(proc,channels="BGR",use_container_width=True)
                st.success(f"✅ {fb}") if cor else st.warning(f"⚠️ {fb}")
        else:
            up=st.file_uploader("Upload video",type=["mp4","avi","mov"],key="cf_vid")
            if up:
                tf2=tempfile.NamedTemporaryFile(delete=False,suffix=".mp4"); tf2.write(up.read()); tf2.close()
                cap=cv2.VideoCapture(tf2.name); state=make_cf_state()
                fph=st.empty(); fbph=st.empty(); cf=tf=0
                while cap.isOpened():
                    ret,frame=cap.read()
                    if not ret: break
                    proc,state,fb,_,cor=process_cf_frame(frame,state)
                    fph.image(proc,channels="BGR",use_container_width=True); tf+=1
                    if cor: cf+=1
                    fbph.success(f"✅ {fb}") if cor else fbph.warning(f"⚠️ {fb}")
                    time.sleep(0.04)
                acc=(cf/tf*100) if tf>0 else 0; cap.release(); os.unlink(tf2.name)
                st.info(f"📊 Done — {state['counter']} reps, {acc:.0f}% accuracy")
