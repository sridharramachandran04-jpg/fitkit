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

def make_mc_state():
    return {"counter":0,"stage":None,"feedback_text":"Get into plank — drive knees to chest",
            "last_feedback_time":time.time(),"left_buf":deque(maxlen=5),"right_buf":deque(maxlen=5)}

def process_mc_frame(frame, state):
    """
    Mountain Climber — alternating knee drives from plank position.
    Tracks hip-knee-ankle angle for each leg.
    EXTENDED: knee out (angle >140°)
    TUCKED  : knee driven in (angle <80°)
    Rep counted each time a knee is tucked in.
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
        lh,lk,la=g("LEFT_HIP"),g("LEFT_KNEE"),g("LEFT_ANKLE")
        rh,rk,ra=g("RIGHT_HIP"),g("RIGHT_KNEE"),g("RIGHT_ANKLE")
        lok=all(x.visibility>confidence_threshold for x in [lh,lk,la])
        rok=all(x.visibility>confidence_threshold for x in [rh,rk,ra])
        if not(lok or rok): raise Exception()
        la_ang=calculate_angle([lh.x,lh.y],[lk.x,lk.y],[la.x,la.y]) if lok else 180
        ra_ang=calculate_angle([rh.x,rh.y],[rk.x,rk.y],[ra.x,ra.y]) if rok else 180
        min_ang=min(la_ang,ra_ang)
        state["left_buf"].append(la_ang); state["right_buf"].append(ra_ang)
        sl=sum(state["left_buf"])/len(state["left_buf"])
        sr=sum(state["right_buf"])/len(state["right_buf"])
        # Count each knee drive
        stg=state.get("stage")
        if min_ang<80:
            if stg!="tucked":
                state["stage"]="tucked"; state["counter"]+=1
                state["feedback_text"]=f"Rep {state['counter']}! Switch legs!"; state["last_feedback_time"]=t
            correct=True
        elif min_ang>140:
            if stg!="extended":
                state["stage"]="extended"
                if t-lft>fd: state["feedback_text"]="Drive that knee to chest!"; state["last_feedback_time"]=t
        else:
            if t-lft>fd: state["feedback_text"]="Drive knee further in!"; state["last_feedback_time"]=t
        # Back sag check
        ls,rs=g("LEFT_SHOULDER"),g("RIGHT_SHOULDER")
        avg_sy=(ls.y+rs.y)/2; avg_hy=(lh.y+rh.y)/2 if lok and rok else avg_sy
        if avg_hy>avg_sy+0.1 and t-lft>fd:
            state["feedback_text"]="Keep hips level — don't sag!"; state["last_feedback_time"]=t
        def px(o): return (int(o.x*w),int(o.y*h))
        if lok: cv2.line(frame,px(lh),px(lk),(255,255,0),3); cv2.line(frame,px(lk),px(la),(255,255,0),3)
        if rok: cv2.line(frame,px(rh),px(rk),(255,255,0),3); cv2.line(frame,px(rk),px(ra),(255,255,0),3)
        mp_drawing.draw_landmarks(frame,res.pose_landmarks,mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=drawing_spec,
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0),thickness=2))
        cv2.putText(frame,f"L:{sl:.0f}° R:{sr:.0f}°",(10,60),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)
    except:
        if t-state.get("last_feedback_time",0)>2:
            state["feedback_text"]="Get into plank position — show full body"; state["last_feedback_time"]=t
    cv2.rectangle(frame,(0,h-70),(w,h),(0,0,0),-1)
    cv2.putText(frame,f"Mountain Climbers: {state.get('counter',0)}",(10,h-40),cv2.FONT_HERSHEY_SIMPLEX,0.85,(255,255,255),2)
    cv2.putText(frame,state.get("feedback_text",""),(10,h-15),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
    return frame,state,state.get("feedback_text",""),state.get("counter",0),correct

def mountain_climber_detection(user_id, log_exercise_fn):
    col_left,col_right=st.columns(2)
    with col_right:
        st.subheader("🎯 Mountain Climbers — Correct Form")
        demo_tab,tips_tab=st.tabs(["📺 Demo","📝 Form Tips"])
        with demo_tab:
            dp=_os.path.join(_APP_DIR,"demo/MountainClimber/mc_demo.mp4")
            if os.path.exists(dp):
                with open(dp,"rb") as vf: st.video(vf.read())
            else: st.info("Add demo to `demo/MountainClimber/mc_demo.mp4`")
        with tips_tab:
            st.markdown("""
**Do's ✅**
- Start in a solid high plank position
- Drive each knee toward the chest quickly
- Keep hips level with your shoulders
- Alternate legs as fast as your form allows

**Don'ts ❌**
- Don't let hips pike up or sag down
- Don't bounce on your hands
- Don't look up — keep neck neutral
- Don't bring knee only halfway
            """)
            st.info("📐 **Rep counted** each time a knee is driven in (angle <80°)")
    with col_left:
        st.subheader("📹 Your Input")
        itype=st.radio("Choose Input Method",["📷 Realtime Camera","🖼️ Upload Image","🎬 Upload Video"],horizontal=True,key="mc_itype")
        if itype=="📷 Realtime Camera":
            run=st.checkbox("▶️ Start Camera",key="mc_cam"); fw=st.empty(); dp2=st.empty()
            if run:
                cap=cv2.VideoCapture(0); state=make_mc_state(); thr=FeedbackThrottler(2.0)
                st_t=time.time(); cf=tf=0
                while st.session_state.get("mc_cam",False):
                    ret,frame=cap.read()
                    if not ret: break
                    frame=cv2.flip(frame,1)
                    proc,state,fb,cnt,cor=process_mc_frame(frame,state)
                    el=time.time()-st_t; tf+=1
                    if cor: cf+=1
                    acc=(cf/tf*100) if tf>0 else 0
                    proc=draw_feedback_overlay(proc,fb,cor,cnt,str(state.get("stage","")),acc,"Mountain Climber")
                    fw.image(proc,channels="BGR",use_container_width=True)
                    with dp2.container(): render_dashboard(cnt,acc,str(state.get("stage","")),fb,cor,"Mountain Climber",el)
                    speak_js(fb,thr); time.sleep(0.04)
                dur=int(time.time()-st_t); cap.release()
                if dur>5:
                    log_exercise_fn(user_id,"Mountain Climber",dur,state["counter"]*3,acc)
                    st.success(f"✅ Saved — {state['counter']} reps, {acc:.0f}% accuracy")
        elif itype=="🖼️ Upload Image":
            up=st.file_uploader("Upload image",type=["jpg","jpeg","png"],key="mc_img")
            if up:
                frame=cv2.cvtColor(np.array(Image.open(up)),cv2.COLOR_RGB2BGR)
                proc,s,fb,_,cor=process_mc_frame(frame,make_mc_state())
                st.image(proc,channels="BGR",use_container_width=True)
                st.success(f"✅ {fb}") if cor else st.warning(f"⚠️ {fb}")
        else:
            up=st.file_uploader("Upload video",type=["mp4","avi","mov"],key="mc_vid")
            if up:
                tf2=tempfile.NamedTemporaryFile(delete=False,suffix=".mp4"); tf2.write(up.read()); tf2.close()
                cap=cv2.VideoCapture(tf2.name); state=make_mc_state()
                fph=st.empty(); fbph=st.empty(); cf=tf=0
                while cap.isOpened():
                    ret,frame=cap.read()
                    if not ret: break
                    proc,state,fb,_,cor=process_mc_frame(frame,state)
                    fph.image(proc,channels="BGR",use_container_width=True); tf+=1
                    if cor: cf+=1
                    fbph.success(f"✅ {fb}") if cor else fbph.warning(f"⚠️ {fb}")
                    time.sleep(0.04)
                acc=(cf/tf*100) if tf>0 else 0; cap.release(); os.unlink(tf2.name)
                st.info(f"📊 Done — {state['counter']} knee drives, {acc:.0f}% accuracy")
