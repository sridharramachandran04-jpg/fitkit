import cv2, mediapipe as mp, numpy as np, time, os, tempfile
from PIL import Image
import streamlit as st
from collections import deque
from realtime_feedback import draw_feedback_overlay, render_dashboard, speak_js, FeedbackThrottler
import os as _os
_BASE_DIR = _os.path.dirname(_os.path.abspath(__file__))
_APP_DIR  = _os.path.dirname(_BASE_DIR)

mp_drawing   = mp.solutions.drawing_utils
mp_pose      = mp.solutions.pose
drawing_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=1)
confidence_threshold = 0.5

def calculate_angle(a,b,c):
    a,b,c=np.array(a),np.array(b),np.array(c); ba,bc=a-b,c-b
    return np.degrees(np.arccos(np.clip(np.dot(ba,bc)/(np.linalg.norm(ba)*np.linalg.norm(bc)),-1,1)))

def make_pu_state():
    return {"counter":0,"stage":None,"feedback_text":"Hang from bar — arms fully extended",
            "last_feedback_time":time.time(),"left_buf":deque(maxlen=6),"right_buf":deque(maxlen=6)}

def process_pu_frame(frame, state):
    """
    Pull-up detection — tracks elbow angle (shoulder→elbow→wrist).
    DOWN (hang) : arms fully extended  >155°
    UP  (chin up): elbows bent <70° (chin over bar)
    Rep: DOWN → UP → DOWN
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
        ls,le,lw=g("LEFT_SHOULDER"),g("LEFT_ELBOW"),g("LEFT_WRIST")
        rs,re,rw=g("RIGHT_SHOULDER"),g("RIGHT_ELBOW"),g("RIGHT_WRIST")
        lok=all(x.visibility>confidence_threshold for x in [ls,le,lw])
        rok=all(x.visibility>confidence_threshold for x in [rs,re,rw])
        if not(lok or rok): raise Exception()
        if lok: state["left_buf"].append(calculate_angle([ls.x,ls.y],[le.x,le.y],[lw.x,lw.y]))
        if rok: state["right_buf"].append(calculate_angle([rs.x,rs.y],[re.x,re.y],[rw.x,rw.y]))
        buf=list(state["left_buf"])+list(state["right_buf"])
        sm=sum(buf)/len(buf)
        stg=state.get("stage")
        if sm>155:
            if stg!="down":
                state["stage"]="down"
                if t-lft>fd: state["feedback_text"]="Dead hang — now pull up!"; state["last_feedback_time"]=t
        elif sm<70:
            if stg=="down":
                state["counter"]+=1; state["stage"]="up"
                state["feedback_text"]=f"Rep {state['counter']}! Lower slowly"; state["last_feedback_time"]=t
            elif stg!="up": state["stage"]="up"
            correct=True
            # Check shoulder engagement — wrists should be above shoulders
            if lok and lw.y>ls.y and t-lft>fd:
                state["feedback_text"]="Pull higher — chin over the bar!"; state["last_feedback_time"]=t
        else:
            if t-lft>fd:
                state["feedback_text"]=("Pull higher — chin over bar!" if sm>110 else "Lower all the way — full extension!")
                state["last_feedback_time"]=t
        def px(o): return (int(o.x*w),int(o.y*h))
        for ok,s,e,wr in [(lok,ls,le,lw),(rok,rs,re,rw)]:
            if ok:
                cv2.line(frame,px(s),px(e),(255,255,0),3); cv2.line(frame,px(e),px(wr),(255,255,0),3)
        mp_drawing.draw_landmarks(frame,res.pose_landmarks,mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=drawing_spec,
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0),thickness=2))
        cv2.putText(frame,f"Elbow: {sm:.0f}°",(10,60),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)
    except:
        if t-state.get("last_feedback_time",0)>2:
            state["feedback_text"]="Show upper body in frame"; state["last_feedback_time"]=t
    cv2.rectangle(frame,(0,h-70),(w,h),(0,0,0),-1)
    cv2.putText(frame,f"Pull-ups: {state.get('counter',0)}",(10,h-40),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
    cv2.putText(frame,state.get("feedback_text",""),(10,h-15),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
    return frame,state,state.get("feedback_text",""),state.get("counter",0),correct

def pull_up_detection(user_id, log_exercise_fn):
    col_left,col_right=st.columns(2)
    with col_right:
        st.subheader("🎯 Pull-up — Correct Form")
        demo_tab,tips_tab=st.tabs(["📺 Demo","📝 Form Tips"])
        with demo_tab:
            dp=_os.path.join(_APP_DIR,"demo/PullUp/pull_up_demo.mp4")
            if os.path.exists(dp):
                with open(dp,"rb") as vf: st.video(vf.read())
            else:
                st.info("Add demo to `demo/PullUp/pull_up_demo.mp4`")
        with tips_tab:
            st.markdown("""
**Do's ✅**
- Start from a dead hang — arms fully extended
- Pull until chin is clearly above the bar
- Engage your lats — think "elbows to hips"
- Lower under control — don't just drop

**Don'ts ❌**
- Don't kip or swing your body
- Don't shrug your shoulders at the top
- Don't use only your arms — engage your back
- Don't let go of full extension at the bottom
            """)
            st.info("📐 **Rep counted**: elbow >155° (hang) → <70° (chin over bar)")
    with col_left:
        st.subheader("📹 Your Input")
        itype=st.radio("Choose Input Method",["📷 Realtime Camera","🖼️ Upload Image","🎬 Upload Video"],horizontal=True,key="pul_itype")
        if itype=="📷 Realtime Camera":
            run=st.checkbox("▶️ Start Camera",key="pul_cam")
            fw=st.empty(); dp2=st.empty()
            if run:
                cap=cv2.VideoCapture(0); state=make_pu_state(); thr=FeedbackThrottler(3.0)
                st_t=time.time(); cf=tf=0
                while st.session_state.get("pul_cam",False):
                    ret,frame=cap.read()
                    if not ret: break
                    frame=cv2.flip(frame,1)
                    proc,state,fb,cnt,cor=process_pu_frame(frame,state)
                    el=time.time()-st_t; tf+=1
                    if cor: cf+=1
                    acc=(cf/tf*100) if tf>0 else 0
                    proc=draw_feedback_overlay(proc,fb,cor,cnt,str(state.get("stage","")),acc,"Pull-up")
                    fw.image(proc,channels="BGR",use_container_width=True)
                    with dp2.container(): render_dashboard(cnt,acc,str(state.get("stage","")),fb,cor,"Pull-up",el)
                    speak_js(fb,thr); time.sleep(0.05)
                dur=int(time.time()-st_t); cap.release()
                if dur>5:
                    log_exercise_fn(user_id,"Pull-up",dur,state["counter"]*8,acc)
                    st.success(f"✅ Saved — {state['counter']} reps, {acc:.0f}% accuracy")
        elif itype=="🖼️ Upload Image":
            up=st.file_uploader("Upload image",type=["jpg","jpeg","png"],key="pul_img")
            if up:
                frame=cv2.cvtColor(np.array(Image.open(up)),cv2.COLOR_RGB2BGR)
                proc,state,fb,_,cor=process_pu_frame(frame,make_pu_state())
                st.image(proc,channels="BGR",use_container_width=True)
                st.success(f"✅ {fb}") if cor else st.warning(f"⚠️ {fb}")
        else:
            up=st.file_uploader("Upload video",type=["mp4","avi","mov"],key="pul_vid")
            if up:
                tf2=tempfile.NamedTemporaryFile(delete=False,suffix=".mp4"); tf2.write(up.read()); tf2.close()
                cap=cv2.VideoCapture(tf2.name); state=make_pu_state()
                fph=st.empty(); fbph=st.empty(); cf=tf=0
                while cap.isOpened():
                    ret,frame=cap.read()
                    if not ret: break
                    proc,state,fb,_,cor=process_pu_frame(frame,state)
                    fph.image(proc,channels="BGR",use_container_width=True); tf+=1
                    if cor: cf+=1
                    fbph.success(f"✅ {fb}") if cor else fbph.warning(f"⚠️ {fb}")
                    time.sleep(0.04)
                acc=(cf/tf*100) if tf>0 else 0; cap.release(); os.unlink(tf2.name)
                st.info(f"📊 Done — {state['counter']} reps, {acc:.0f}% accuracy")
