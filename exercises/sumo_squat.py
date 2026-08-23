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

def make_ss_state():
    return {"counter":0,"stage":None,"feedback_text":"Feet wide, toes out — squat down",
            "last_feedback_time":time.time(),"left_buf":deque(maxlen=6),"right_buf":deque(maxlen=6)}

def process_ss_frame(frame, state):
    """Sumo Squat — wide stance squat.
    Tracks knee angle (hip→knee→ankle).
    UP  : standing — knee angle >160°
    DOWN: sumo squat — knee angle <105°
    Also checks feet width (wider than hip-width).
    Rep: UP → DOWN → UP
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
        lh,lk,la=g("LEFT_HIP"),g("LEFT_KNEE"),g("LEFT_ANKLE")
        rh,rk,ra=g("RIGHT_HIP"),g("RIGHT_KNEE"),g("RIGHT_ANKLE")
        lok=all(x.visibility>confidence_threshold for x in [lh,lk,la])
        rok=all(x.visibility>confidence_threshold for x in [rh,rk,ra])
        if not(lok or rok): raise Exception()
        if lok: state["left_buf"].append(calculate_angle([lh.x,lh.y],[lk.x,lk.y],[la.x,la.y]))
        if rok: state["right_buf"].append(calculate_angle([rh.x,rh.y],[rk.x,rk.y],[ra.x,ra.y]))
        buf=list(state["left_buf"])+list(state["right_buf"]); sm=sum(buf)/len(buf)
        # Foot width vs hip width
        ankle_w=abs(la.x-ra.x) if lok and rok else 0
        hip_w=abs(lh.x-rh.x)+0.001
        wide_enough=ankle_w/hip_w>1.4
        stg=state.get("stage")
        if sm>160:
            if stg!="up":
                state["stage"]="up"
                if t-lft>fd: state["feedback_text"]="Wide stance — now sumo squat!"; state["last_feedback_time"]=t
        elif sm<105:
            if stg=="up":
                state["counter"]+=1; state["stage"]="down"
                state["feedback_text"]=f"Rep {state['counter']}! Drive up through heels"; state["last_feedback_time"]=t
            elif stg!="down": state["stage"]="down"
            correct=True
            if not wide_enough and t-lft>fd:
                state["feedback_text"]="Widen your stance more!"; state["last_feedback_time"]=t
        else:
            if t-lft>fd:
                state["feedback_text"]=("Squat lower — thighs parallel!" if sm>130 else "Stand tall — full extension!")
                state["last_feedback_time"]=t
        def px(o): return (int(o.x*w),int(o.y*h))
        if lok:
            for a2,b2 in [(px(lh),px(lk)),(px(lk),px(la))]: cv2.line(frame,a2,b2,(255,255,0),3)
        if rok:
            for a2,b2 in [(px(rh),px(rk)),(px(rk),px(ra))]: cv2.line(frame,a2,b2,(255,255,0),3)
        mp_drawing.draw_landmarks(frame,res.pose_landmarks,mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=drawing_spec,
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0),thickness=2))
        cv2.putText(frame,f"Knee: {sm:.0f}°  Wide:{'Y' if wide_enough else 'N'}",(10,60),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)
    except:
        if t-state.get("last_feedback_time",0)>2:
            state["feedback_text"]="Stand in frame — show full legs"; state["last_feedback_time"]=t
    cv2.rectangle(frame,(0,h-70),(w,h),(0,0,0),-1)
    cv2.putText(frame,f"Sumo Squats: {state.get('counter',0)}",(10,h-40),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
    cv2.putText(frame,state.get("feedback_text",""),(10,h-15),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
    return frame,state,state.get("feedback_text",""),state.get("counter",0),correct

def ss_detection(user_id, log_exercise_fn):
    col_left,col_right=st.columns(2)
    with col_right:
        st.subheader("🎯 Sumo Squat — Correct Form")
        demo_tab,tips_tab=st.tabs(["📺 Demo","📝 Form Tips"])
        with demo_tab:
            dp=_os.path.join(_APP_DIR,"demo/SumoSquat/sumo_squat_demo.mp4")
            if os.path.exists(dp):
                with open(dp,"rb") as vf: st.video(vf.read())
            else: st.info("Add demo to `demo/SumoSquat/sumo_squat_demo.mp4`")
        with tips_tab:
            st.markdown("""**Do's ✅**\n- Feet 1.5-2x shoulder width, toes pointed 45° out\n- Drive knees out over toes throughout\n- Keep chest up and spine neutral\n- Squeeze glutes at the top\n\n**Don'ts ❌**\n- Don't let knees cave inward\n- Don't lean torso too far forward\n- Don't go on your tiptoes\n- Don't bounce at the bottom""")
            st.info("📐 **Rep counted**: knee >160° (stand) → <105° (sumo squat) → back up")
    with col_left:
        st.subheader("📹 Your Input")
        itype=st.radio("Choose Input Method",["📷 Realtime Camera","🖼️ Upload Image","🎬 Upload Video"],horizontal=True,key="ss_itype")
        if itype=="📷 Realtime Camera":
            run=st.checkbox("▶️ Start Camera",key="ss_cam"); fw=st.empty(); dp2=st.empty()
            if run:
                cap=cv2.VideoCapture(0); state=make_ss_state(); thr=FeedbackThrottler(3.0)
                st_t=time.time(); cf=tf=0
                while st.session_state.get("ss_cam",False):
                    ret,frame=cap.read()
                    if not ret: break
                    frame=cv2.flip(frame,1)
                    proc,state,fb,cnt,cor=process_ss_frame(frame,state)
                    el=time.time()-st_t; tf+=1
                    if cor: cf+=1
                    acc=(cf/tf*100) if tf>0 else 0
                    proc=draw_feedback_overlay(proc,fb,cor,cnt,str(state.get("stage","")),acc,"Sumo Squat")
                    fw.image(proc,channels="BGR",use_container_width=True)
                    with dp2.container(): render_dashboard(cnt,acc,str(state.get("stage","")),fb,cor,"Sumo Squat",el)
                    speak_js(fb,thr); time.sleep(0.05)
                dur=int(time.time()-st_t); cap.release()
                if dur>5:
                    log_exercise_fn(user_id,"Sumo Squat",dur,state["counter"]*5,acc)
                    st.success(f"✅ Saved — {state['counter']} reps, {acc:.0f}% accuracy")
        elif itype=="🖼️ Upload Image":
            up=st.file_uploader("Upload image",type=["jpg","jpeg","png"],key="ss_img")
            if up:
                frame=cv2.cvtColor(np.array(Image.open(up)),cv2.COLOR_RGB2BGR)
                proc,s,fb,_,cor=process_ss_frame(frame,make_ss_state())
                st.image(proc,channels="BGR",use_container_width=True)
                st.success(f"✅ {fb}") if cor else st.warning(f"⚠️ {fb}")
        else:
            up=st.file_uploader("Upload video",type=["mp4","avi","mov"],key="ss_vid")
            if up:
                tf2=tempfile.NamedTemporaryFile(delete=False,suffix=".mp4"); tf2.write(up.read()); tf2.close()
                cap=cv2.VideoCapture(tf2.name); state=make_ss_state()
                fph=st.empty(); fbph=st.empty(); cf=tf=0
                while cap.isOpened():
                    ret,frame=cap.read()
                    if not ret: break
                    proc,state,fb,_,cor=process_ss_frame(frame,state)
                    fph.image(proc,channels="BGR",use_container_width=True); tf+=1
                    if cor: cf+=1
                    fbph.success(f"✅ {fb}") if cor else fbph.warning(f"⚠️ {fb}")
                    time.sleep(0.04)
                acc=(cf/tf*100) if tf>0 else 0; cap.release(); os.unlink(tf2.name)
                st.info(f"📊 Done — {state['counter']} reps, {acc:.0f}% accuracy")
