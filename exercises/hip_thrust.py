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

def make_ht_state():
    return {"counter":0,"stage":None,"feedback_text":"Lie back — feet flat, drive hips up",
            "last_feedback_time":time.time(),"buf":deque(maxlen=6)}

def process_ht_frame(frame, state):
    """
    Hip Thrust — tracks shoulder→hip→knee angle.
    DOWN: hips low, angle ~90° (<100°)
    UP  : hips fully extended, body flat (>155°)
    Rep: DOWN → UP → DOWN
    Best detected from side view.
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
        ls,lh,lk=g("LEFT_SHOULDER"),g("LEFT_HIP"),g("LEFT_KNEE")
        rs,rh,rk=g("RIGHT_SHOULDER"),g("RIGHT_HIP"),g("RIGHT_KNEE")
        lok=all(x.visibility>confidence_threshold for x in [ls,lh,lk])
        rok=all(x.visibility>confidence_threshold for x in [rs,rh,rk])
        if not(lok or rok): raise Exception()
        angles=[]
        if lok: angles.append(calculate_angle([ls.x,ls.y],[lh.x,lh.y],[lk.x,lk.y]))
        if rok: angles.append(calculate_angle([rs.x,rs.y],[rh.x,rh.y],[rk.x,rk.y]))
        avg=sum(angles)/len(angles); state["buf"].append(avg)
        sm=sum(state["buf"])/len(state["buf"])
        stg=state.get("stage")
        if sm<100:
            if stg!="down":
                state["stage"]="down"
                if t-lft>fd: state["feedback_text"]="Drive hips up — squeeze glutes!"; state["last_feedback_time"]=t
        elif sm>155:
            if stg=="down":
                state["counter"]+=1; state["stage"]="up"
                state["feedback_text"]=f"Rep {state['counter']}! Lower with control"; state["last_feedback_time"]=t
            elif stg!="up": state["stage"]="up"
            correct=True
            if sm>170 and t-lft>fd:
                state["feedback_text"]="Don't hyperextend — hold at flat!"; state["last_feedback_time"]=t
        else:
            if t-lft>fd:
                state["feedback_text"]=("Push hips higher — full extension!" if sm>125 else "Lower all the way down first")
                state["last_feedback_time"]=t
        def px(o): return (int(o.x*w),int(o.y*h))
        if lok: cv2.line(frame,px(ls),px(lh),(255,255,0),3); cv2.line(frame,px(lh),px(lk),(255,255,0),3)
        if rok: cv2.line(frame,px(rs),px(rh),(255,255,0),3); cv2.line(frame,px(rh),px(rk),(255,255,0),3)
        mp_drawing.draw_landmarks(frame,res.pose_landmarks,mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=drawing_spec,
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0),thickness=2))
        cv2.putText(frame,f"Hip angle: {sm:.0f}°",(10,60),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)
    except:
        if t-state.get("last_feedback_time",0)>2:
            state["feedback_text"]="Lie sideways to camera — show full body"; state["last_feedback_time"]=t
    cv2.rectangle(frame,(0,h-70),(w,h),(0,0,0),-1)
    cv2.putText(frame,f"Hip Thrusts: {state.get('counter',0)}",(10,h-40),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
    cv2.putText(frame,state.get("feedback_text",""),(10,h-15),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
    return frame,state,state.get("feedback_text",""),state.get("counter",0),correct

def hip_thrust_detection(user_id, log_exercise_fn):
    col_left,col_right=st.columns(2)
    with col_right:
        st.subheader("🎯 Hip Thrust — Correct Form")
        demo_tab,tips_tab=st.tabs(["📺 Demo","📝 Form Tips"])
        with demo_tab:
            dp=_os.path.join(_APP_DIR,"demo/HipThrust/hip_thrust_demo.mp4")
            if os.path.exists(dp):
                with open(dp,"rb") as vf: st.video(vf.read())
            else: st.info("Add demo to `demo/HipThrust/hip_thrust_demo.mp4`")
        with tips_tab:
            st.markdown("""
**Do's ✅**
- Drive through your heels, not your toes
- Squeeze glutes hard at the top
- Keep chin tucked — don't strain neck
- Upper back rests on bench edge

**Don'ts ❌**
- Don't hyperextend your lower back at the top
- Don't let knees cave inward
- Don't rush — control the descent
- Don't roll onto your neck
            """)
            st.info("📐 **Rep counted**: hip angle <100° (down) → >155° (full extension)")
            st.warning("⚠️ Best detected from a **side view**")
    with col_left:
        st.subheader("📹 Your Input")
        itype=st.radio("Choose Input Method",["📷 Realtime Camera","🖼️ Upload Image","🎬 Upload Video"],horizontal=True,key="ht_itype")
        if itype=="📷 Realtime Camera":
            run=st.checkbox("▶️ Start Camera",key="ht_cam"); fw=st.empty(); dp2=st.empty()
            if run:
                cap=cv2.VideoCapture(0); state=make_ht_state(); thr=FeedbackThrottler(3.0)
                st_t=time.time(); cf=tf=0
                while st.session_state.get("ht_cam",False):
                    ret,frame=cap.read()
                    if not ret: break
                    frame=cv2.flip(frame,1)
                    proc,state,fb,cnt,cor=process_ht_frame(frame,state)
                    el=time.time()-st_t; tf+=1
                    if cor: cf+=1
                    acc=(cf/tf*100) if tf>0 else 0
                    proc=draw_feedback_overlay(proc,fb,cor,cnt,str(state.get("stage","")),acc,"Hip Thrust")
                    fw.image(proc,channels="BGR",use_container_width=True)
                    with dp2.container(): render_dashboard(cnt,acc,str(state.get("stage","")),fb,cor,"Hip Thrust",el)
                    speak_js(fb,thr); time.sleep(0.05)
                dur=int(time.time()-st_t); cap.release()
                if dur>5:
                    log_exercise_fn(user_id,"Hip Thrust",dur,state["counter"]*5,acc)
                    st.success(f"✅ Saved — {state['counter']} reps, {acc:.0f}% accuracy")
        elif itype=="🖼️ Upload Image":
            up=st.file_uploader("Upload image",type=["jpg","jpeg","png"],key="ht_img")
            if up:
                frame=cv2.cvtColor(np.array(Image.open(up)),cv2.COLOR_RGB2BGR)
                proc,s,fb,_,cor=process_ht_frame(frame,make_ht_state())
                st.image(proc,channels="BGR",use_container_width=True)
                st.success(f"✅ {fb}") if cor else st.warning(f"⚠️ {fb}")
        else:
            up=st.file_uploader("Upload video",type=["mp4","avi","mov"],key="ht_vid")
            if up:
                tf2=tempfile.NamedTemporaryFile(delete=False,suffix=".mp4"); tf2.write(up.read()); tf2.close()
                cap=cv2.VideoCapture(tf2.name); state=make_ht_state()
                fph=st.empty(); fbph=st.empty(); cf=tf=0
                while cap.isOpened():
                    ret,frame=cap.read()
                    if not ret: break
                    proc,state,fb,_,cor=process_ht_frame(frame,state)
                    fph.image(proc,channels="BGR",use_container_width=True); tf+=1
                    if cor: cf+=1
                    fbph.success(f"✅ {fb}") if cor else fbph.warning(f"⚠️ {fb}")
                    time.sleep(0.04)
                acc=(cf/tf*100) if tf>0 else 0; cap.release(); os.unlink(tf2.name)
                st.info(f"📊 Done — {state['counter']} reps, {acc:.0f}% accuracy")
