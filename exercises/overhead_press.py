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

def make_ohp_state():
    return {"counter":0,"stage":None,"feedback_text":"Bar at shoulder level — ready to press",
            "last_feedback_time":time.time(),"left_buf":deque(maxlen=6),"right_buf":deque(maxlen=6)}

def process_ohp_frame(frame, state):
    """
    Overhead Press (standing barbell/dumbbell).
    Tracks elbow angle + wrist height relative to head.
    DOWN: elbows bent ~90° (<110°), bar at shoulder
    UP  : arms fully extended overhead (>160°)
    Rep: DOWN → UP → DOWN
    """
    fd=2; h,w=frame.shape[:2]; t=time.time()
    img=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB); img.flags.writeable=False
    with mp_pose.Pose(min_detection_confidence=0.5,min_tracking_confidence=0.5,model_complexity=1) as pose:
        res=pose.process(img)
    img.flags.writeable=True; frame=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    correct=False; lft=state.get("last_feedback_time",0)
    try:
        if not res.pose_landmarks: raise Exception()
        lm=res.pose_landmarks.landmark
        def g(p): return lm[mp_pose.PoseLandmark[p].value]
        ls,le,lw=g("LEFT_SHOULDER"),g("LEFT_ELBOW"),g("LEFT_WRIST")
        rs,re,rw=g("RIGHT_SHOULDER"),g("RIGHT_ELBOW"),g("RIGHT_WRIST")
        nose=g("NOSE")
        lok=all(x.visibility>confidence_threshold for x in [ls,le,lw])
        rok=all(x.visibility>confidence_threshold for x in [rs,re,rw])
        if not(lok or rok): raise Exception()
        if lok: state["left_buf"].append(calculate_angle([ls.x,ls.y],[le.x,le.y],[lw.x,lw.y]))
        if rok: state["right_buf"].append(calculate_angle([rs.x,rs.y],[re.x,re.y],[rw.x,rw.y]))
        buf=list(state["left_buf"])+list(state["right_buf"]); sm=sum(buf)/len(buf)
        # Wrists above nose = fully overhead
        wrists_overhead = ((lw.y < nose.y-0.05) if lok else True) and ((rw.y < nose.y-0.05) if rok else True)
        stg=state.get("stage")
        if sm<110:
            if stg!="down":
                state["stage"]="down"
                if t-lft>fd: state["feedback_text"]="Good start — press overhead!"; state["last_feedback_time"]=t
        elif sm>160 and wrists_overhead:
            if stg=="down":
                state["counter"]+=1; state["stage"]="up"
                state["feedback_text"]=f"Rep {state['counter']}! Lower to shoulders"; state["last_feedback_time"]=t
            elif stg!="up": state["stage"]="up"
            correct=True
        elif sm>160 and not wrists_overhead:
            if t-lft>fd: state["feedback_text"]="Press the bar directly overhead — not forward!"; state["last_feedback_time"]=t
        else:
            if t-lft>fd:
                state["feedback_text"]=("Press higher — fully extend!" if sm>130 else "Lower to shoulder level")
                state["last_feedback_time"]=t
        def px(o): return (int(o.x*w),int(o.y*h))
        for ok,s,e,wr in [(lok,ls,le,lw),(rok,rs,re,rw)]:
            if ok: cv2.line(frame,px(s),px(e),(255,255,0),3); cv2.line(frame,px(e),px(wr),(255,255,0),3)
        mp_drawing.draw_landmarks(frame,res.pose_landmarks,mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=drawing_spec,
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0),thickness=2))
        cv2.putText(frame,f"Elbow: {sm:.0f}°",(10,60),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)
        cv2.putText(frame,f"Overhead: {'Yes' if wrists_overhead else 'No'}",(10,90),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)
    except:
        if t-state.get("last_feedback_time",0)>2:
            state["feedback_text"]="Stand straight — show full upper body"; state["last_feedback_time"]=t
    cv2.rectangle(frame,(0,h-70),(w,h),(0,0,0),-1)
    cv2.putText(frame,f"OHP: {state.get('counter',0)}",(10,h-40),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
    cv2.putText(frame,state.get("feedback_text",""),(10,h-15),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
    return frame,state,state.get("feedback_text",""),state.get("counter",0),correct

def overhead_press_detection(user_id, log_exercise_fn):
    col_left,col_right=st.columns(2)
    with col_right:
        st.subheader("🎯 Overhead Press — Correct Form")
        demo_tab,tips_tab=st.tabs(["📺 Demo","📝 Form Tips"])
        with demo_tab:
            dp=_os.path.join(_APP_DIR,"demo/OverheadPress/ohp_demo.mp4")
            if os.path.exists(dp):
                with open(dp,"rb") as vf: st.video(vf.read())
            else: st.info("Add demo to `demo/OverheadPress/ohp_demo.mp4`")
        with tips_tab:
            st.markdown("""
**Do's ✅**
- Start with bar at upper chest/chin height
- Press straight up — bar should travel vertically
- Lock out arms fully at the top
- Keep core braced — don't lean back

**Don'ts ❌**
- Don't press forward — bar must go straight up
- Don't hyperextend your lower back
- Don't shrug at the top of the movement
- Don't use leg drive (for strict press)
            """)
            st.info("📐 **Rep counted**: elbow <110° (down) → >160° with wrists overhead (up)")
    with col_left:
        st.subheader("📹 Your Input")
        itype=st.radio("Choose Input Method",["📷 Realtime Camera","🖼️ Upload Image","🎬 Upload Video"],horizontal=True,key="ohp_itype")
        if itype=="📷 Realtime Camera":
            run=st.checkbox("▶️ Start Camera",key="ohp_cam"); fw=st.empty(); dp2=st.empty()
            if run:
                cap=cv2.VideoCapture(0); state=make_ohp_state(); thr=FeedbackThrottler(3.0)
                st_t=time.time(); cf=tf=0
                while st.session_state.get("ohp_cam",False):
                    ret,frame=cap.read()
                    if not ret: break
                    frame=cv2.flip(frame,1)
                    proc,state,fb,cnt,cor=process_ohp_frame(frame,state)
                    el=time.time()-st_t; tf+=1
                    if cor: cf+=1
                    acc=(cf/tf*100) if tf>0 else 0
                    proc=draw_feedback_overlay(proc,fb,cor,cnt,str(state.get("stage","")),acc,"Overhead Press")
                    fw.image(proc,channels="BGR",use_container_width=True)
                    with dp2.container(): render_dashboard(cnt,acc,str(state.get("stage","")),fb,cor,"Overhead Press",el)
                    speak_js(fb,thr); time.sleep(0.05)
                dur=int(time.time()-st_t); cap.release()
                if dur>5:
                    log_exercise_fn(user_id,"Overhead Press",dur,state["counter"]*5,acc)
                    st.success(f"✅ Saved — {state['counter']} reps, {acc:.0f}% accuracy")
        elif itype=="🖼️ Upload Image":
            up=st.file_uploader("Upload image",type=["jpg","jpeg","png"],key="ohp_img")
            if up:
                frame=cv2.cvtColor(np.array(Image.open(up)),cv2.COLOR_RGB2BGR)
                proc,s,fb,_,cor=process_ohp_frame(frame,make_ohp_state())
                st.image(proc,channels="BGR",use_container_width=True)
                st.success(f"✅ {fb}") if cor else st.warning(f"⚠️ {fb}")
        else:
            up=st.file_uploader("Upload video",type=["mp4","avi","mov"],key="ohp_vid")
            if up:
                tf2=tempfile.NamedTemporaryFile(delete=False,suffix=".mp4"); tf2.write(up.read()); tf2.close()
                cap=cv2.VideoCapture(tf2.name); state=make_ohp_state()
                fph=st.empty(); fbph=st.empty(); cf=tf=0
                while cap.isOpened():
                    ret,frame=cap.read()
                    if not ret: break
                    proc,state,fb,_,cor=process_ohp_frame(frame,state)
                    fph.image(proc,channels="BGR",use_container_width=True); tf+=1
                    if cor: cf+=1
                    fbph.success(f"✅ {fb}") if cor else fbph.warning(f"⚠️ {fb}")
                    time.sleep(0.04)
                acc=(cf/tf*100) if tf>0 else 0; cap.release(); os.unlink(tf2.name)
                st.info(f"📊 Done — {state['counter']} reps, {acc:.0f}% accuracy")
