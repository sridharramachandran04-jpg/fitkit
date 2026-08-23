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

def make_burpee_state():
    return {"counter":0,"stage":None,"feedback_text":"Stand tall — start your burpee!",
            "last_feedback_time":time.time(),"buf":deque(maxlen=6)}

def process_burpee_frame(frame, state):
    """
    Burpee detection via hip height transitions.
    STAND : hip y is high (small y value in normalised coords)
    PLANK : hip y drops (person prone/crouched)
    Rep: STAND → PLANK → STAND
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
        ls,rs=g("LEFT_SHOULDER"),g("RIGHT_SHOULDER")
        lok=lh.visibility>confidence_threshold; rok=rh.visibility>confidence_threshold
        if not(lok or rok): raise Exception()
        hip_y=(lh.y+rh.y)/2 if lok and rok else (lh.y if lok else rh.y)
        sh_y=(ls.y+rs.y)/2
        state["buf"].append(hip_y); sm=sum(state["buf"])/len(state["buf"])
        # standing: hip_y < 0.65 AND hip above shoulder (person upright)
        stg=state.get("stage")
        if sm<0.60 and sh_y < hip_y:  # standing
            if stg=="plank":
                state["counter"]+=1; state["stage"]="stand"
                state["feedback_text"]=f"Rep {state['counter']}! Jump up!"; state["last_feedback_time"]=t
            elif stg!="stand":
                state["stage"]="stand"
                if t-lft>fd: state["feedback_text"]="Drop to plank!"; state["last_feedback_time"]=t
            correct=True
        elif sm>0.70:  # prone/crouched
            if stg!="plank":
                state["stage"]="plank"
                if t-lft>fd: state["feedback_text"]="Push up — jump back to stand!"; state["last_feedback_time"]=t
        else:
            if t-lft>fd: state["feedback_text"]="Full movement — stand or drop fully!"; state["last_feedback_time"]=t
        def px(o): return (int(o.x*w),int(o.y*h))
        for ok,hi,sh in [(lok,lh,ls),(rok,rh,rs)]:
            if ok: cv2.line(frame,px(sh),px(hi),(255,255,0),3)
        mp_drawing.draw_landmarks(frame,res.pose_landmarks,mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=drawing_spec,
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0),thickness=2))
        cv2.putText(frame,f"Hip Y: {sm:.2f}  Stage:{stg}",(10,60),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)
    except:
        if t-state.get("last_feedback_time",0)>2:
            state["feedback_text"]="Stand in frame — show full body"; state["last_feedback_time"]=t
    cv2.rectangle(frame,(0,h-70),(w,h),(0,0,0),-1)
    cv2.putText(frame,f"Burpees: {state.get('counter',0)}",(10,h-40),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
    cv2.putText(frame,state.get("feedback_text",""),(10,h-15),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
    return frame,state,state.get("feedback_text",""),state.get("counter",0),correct

def burpee_detection(user_id, log_exercise_fn):
    col_left,col_right=st.columns(2)
    with col_right:
        st.subheader("🎯 Burpee — Correct Form")
        demo_tab,tips_tab=st.tabs(["📺 Demo","📝 Form Tips"])
        with demo_tab:
            dp=_os.path.join(_APP_DIR,"demo/Burpee/burpee_demo.mp4")
            if os.path.exists(dp):
                with open(dp,"rb") as vf: st.video(vf.read())
            else: st.info("Add demo to `demo/Burpee/burpee_demo.mp4`")
        with tips_tab:
            st.markdown("""
**Do's ✅**
- Stand tall at the top — full hip extension
- Keep core braced in the plank phase
- Chest to floor for full range of motion
- Explode upward — jump and clap overhead

**Don'ts ❌**
- Don't let hips sag in the plank
- Don't skip the jump — that's the cardio!
- Don't rush at the cost of form
- Don't look down — keep neck neutral
            """)
            st.info("📐 **Rep counted**: stand → drop to plank → stand back up")
    with col_left:
        st.subheader("📹 Your Input")
        itype=st.radio("Choose Input Method",["📷 Realtime Camera","🖼️ Upload Image","🎬 Upload Video"],horizontal=True,key="burp_itype")
        if itype=="📷 Realtime Camera":
            run=st.checkbox("▶️ Start Camera",key="burp_cam"); fw=st.empty(); dp2=st.empty()
            if run:
                cap=cv2.VideoCapture(0); state=make_burpee_state(); thr=FeedbackThrottler(2.0)
                st_t=time.time(); cf=tf=0
                while st.session_state.get("burp_cam",False):
                    ret,frame=cap.read()
                    if not ret: break
                    frame=cv2.flip(frame,1)
                    proc,state,fb,cnt,cor=process_burpee_frame(frame,state)
                    el=time.time()-st_t; tf+=1
                    if cor: cf+=1
                    acc=(cf/tf*100) if tf>0 else 0
                    proc=draw_feedback_overlay(proc,fb,cor,cnt,str(state.get("stage","")),acc,"Burpee")
                    fw.image(proc,channels="BGR",use_container_width=True)
                    with dp2.container(): render_dashboard(cnt,acc,str(state.get("stage","")),fb,cor,"Burpee",el)
                    speak_js(fb,thr); time.sleep(0.05)
                dur=int(time.time()-st_t); cap.release()
                if dur>5:
                    log_exercise_fn(user_id,"Burpee",dur,state["counter"]*10,acc)
                    st.success(f"✅ Saved — {state['counter']} reps, {acc:.0f}% accuracy")
        elif itype=="🖼️ Upload Image":
            up=st.file_uploader("Upload image",type=["jpg","jpeg","png"],key="burp_img")
            if up:
                frame=cv2.cvtColor(np.array(Image.open(up)),cv2.COLOR_RGB2BGR)
                proc,s,fb,_,cor=process_burpee_frame(frame,make_burpee_state())
                st.image(proc,channels="BGR",use_container_width=True)
                st.success(f"✅ {fb}") if cor else st.warning(f"⚠️ {fb}")
        else:
            up=st.file_uploader("Upload video",type=["mp4","avi","mov"],key="burp_vid")
            if up:
                tf2=tempfile.NamedTemporaryFile(delete=False,suffix=".mp4"); tf2.write(up.read()); tf2.close()
                cap=cv2.VideoCapture(tf2.name); state=make_burpee_state()
                fph=st.empty(); fbph=st.empty(); cf=tf=0
                while cap.isOpened():
                    ret,frame=cap.read()
                    if not ret: break
                    proc,state,fb,_,cor=process_burpee_frame(frame,state)
                    fph.image(proc,channels="BGR",use_container_width=True); tf+=1
                    if cor: cf+=1
                    fbph.success(f"✅ {fb}") if cor else fbph.warning(f"⚠️ {fb}")
                    time.sleep(0.04)
                acc=(cf/tf*100) if tf>0 else 0; cap.release(); os.unlink(tf2.name)
                st.info(f"📊 Done — {state['counter']} reps, {acc:.0f}% accuracy")
