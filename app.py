import _check_mediapipe  # noqa: F401  — must run before any mediapipe import; exits with a clear fix if the version is wrong
import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import tempfile
import time
import base64
from PIL import Image
import io
try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except Exception:
    PYTTSX3_AVAILABLE = False
import threading
import os
import sqlite3
import hashlib
from datetime import datetime

# ── Resolve asset paths relative to this file — works from any working directory ──
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
def asset(rel): return os.path.join(BASE_DIR, rel)
from exercises.pushups        import pushup_detection
from exercises.biceps         import bicep_detection
from exercises.jumping_jacks  import jumping_jack_detection
from exercises.shoulder_press import shoulder_press_detection
from exercises.lateral_raises import lateral_raise_detection
from chatbot import chatbot_page
from realtime_feedback import draw_feedback_overlay, render_dashboard, speak_js, FeedbackThrottler
from exercises.lunges         import lunge_detection
from exercises.tricep_dips    import tricep_dip_detection
from exercises.deadlift        import deadlift_detection
# New batch
from exercises.bench_press     import bench_press_detection
from exercises.pull_ups        import pull_up_detection
from exercises.rows            import row_detection
from exercises.overhead_press  import overhead_press_detection
from exercises.mountain_climbers import mountain_climber_detection
from exercises.hip_thrust      import hip_thrust_detection
from exercises.burpees         import burpee_detection
from exercises.leg_press       import lp_detection
from exercises.cable_fly       import cf_detection
from exercises.sumo_squat      import ss_detection
from exercises.face_pull       import fp_detection
from exercises.box_step_ups    import bsu_detection
from exercises.hammer_curls import hammer_curls_detection
from exercises.arnold_press    import arnold_press_detection
from exercises.calf_raises     import calf_raise_detection
from exercises.high_knees      import high_knees_detection
from exercises.glute_bridge    import glute_bridge_detection

# Page config
st.set_page_config(page_title="FITKIT - AI Fitness Coach", layout="wide")

# Initialize MediaPipe
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# ==================== DATABASE SETUP ====================
def init_database():
    conn = sqlite3.connect(asset('fitkit.db'))
    c = conn.cursor()
    
    # Create users table (your existing code)
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (user_id INTEGER PRIMARY KEY AUTOINCREMENT,
                  email TEXT UNIQUE,
                  password TEXT,
                  name TEXT,
                  age INTEGER,
                  gender TEXT,
                  height REAL,
                  weight REAL,
                  goal TEXT,
                  activity_level TEXT,
                  dietary_prefs TEXT,
                  created_at TIMESTAMP)''')
    
    # Create exercise_log table (your existing code)
    c.execute('''CREATE TABLE IF NOT EXISTS exercise_log
                 (log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_id INTEGER,
                  exercise_name TEXT,
                  duration INTEGER,
                  calories REAL,
                  posture_accuracy REAL,
                  timestamp TIMESTAMP,
                  FOREIGN KEY (user_id) REFERENCES users (user_id))''')
    
    # Create workout_completion table (NEW)
    c.execute('''CREATE TABLE IF NOT EXISTS workout_completion
                 (completion_id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_id INTEGER,
                  completion_date DATE,
                  is_completed BOOLEAN,
                  notes TEXT,
                  timestamp TIMESTAMP,
                  FOREIGN KEY (user_id) REFERENCES users (user_id))''')
    
    # Create progress table (your existing code)
    c.execute('''CREATE TABLE IF NOT EXISTS progress
                 (progress_id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_id INTEGER,
                  date DATE,
                  weight REAL,
                  notes TEXT,
                  FOREIGN KEY (user_id) REFERENCES users (user_id))''')
    
    conn.commit()
    conn.close()

# Password hashing
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Call init
init_database()

# ==================== DATABASE FUNCTIONS ====================
def add_user(email, password, name, age, gender, height, weight, goal, activity, dietary):
    conn = sqlite3.connect(asset('fitkit.db'))
    c = conn.cursor()
    try:
        hashed_pw = hash_password(password)
        dietary_str = ','.join(dietary) if dietary else ''
        c.execute('''INSERT INTO users 
                     (email, password, name, age, gender, height, weight, goal, activity_level, dietary_prefs, created_at)
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                  (email, hashed_pw, name, age, gender, height, weight, goal, activity, dietary_str, datetime.now()))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def check_user(email, password):
    conn = sqlite3.connect(asset('fitkit.db'))
    c = conn.cursor()
    hashed_pw = hash_password(password)
    c.execute('SELECT * FROM users WHERE email = ? AND password = ?', (email, hashed_pw))
    user = c.fetchone()
    conn.close()
    return user

def get_user_by_email(email):
    conn = sqlite3.connect(asset('fitkit.db'))
    c = conn.cursor()
    c.execute('SELECT * FROM users WHERE email = ?', (email,))
    user = c.fetchone()
    conn.close()
    return user

def log_exercise(user_id, exercise_name, duration, calories, accuracy):
    conn = sqlite3.connect(asset('fitkit.db'))
    c = conn.cursor()
    c.execute('''INSERT INTO exercise_log
                 (user_id, exercise_name, duration, calories, posture_accuracy, timestamp)
                 VALUES (?, ?, ?, ?, ?, ?)''',
              (user_id, exercise_name, duration, calories, accuracy, datetime.now()))
    # Also mark workout_completion for today
    today = datetime.now().date()
    c.execute('''INSERT INTO workout_completion
                 (user_id, completion_date, is_completed, notes, timestamp)
                 VALUES (?, ?, ?, ?, ?)''',
              (user_id, today, True, f"{exercise_name} — {duration}s", datetime.now()))
    conn.commit()
    conn.close()

def get_user_progress(user_id):
    conn = sqlite3.connect(asset('fitkit.db'))
    c = conn.cursor()
    c.execute('''SELECT date, weight FROM progress 
                 WHERE user_id = ? ORDER BY date DESC LIMIT 7''', (user_id,))
    data = c.fetchall()
    conn.close()
    return data

# Initialize session state
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'user_data' not in st.session_state:
    st.session_state.user_data = {}
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Login"
if 'user_id' not in st.session_state:
    st.session_state.user_id = None
if 'chat_messages' not in st.session_state:
    st.session_state.chat_messages = []
if 'selected_exercise' not in st.session_state:
    st.session_state.selected_exercise = None

# Text to speech engine
def init_tts():
    if not PYTTSX3_AVAILABLE:
        return None
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)
        engine.setProperty('volume', 0.9)
        return engine
    except:
        return None

def speak_text(text):
    def _speak():
        try:
            engine = init_tts()
            if engine:
                engine.say(text)
                engine.runAndWait()
        except Exception:
            pass  # TTS is optional — never crash the app
    t = threading.Thread(target=_speak, daemon=True)
    t.start()

# Updated EXERCISES dictionary with WORKING URLs
EXERCISES = {
    "Plank": {
        "demo": asset("assets/planks.gif"),
        "thumbnail": "https://www.inspireusafoundation.org/wp-content/uploads/2022/06/plank-form.gif",
        "instructions": "Keep back straight, engage core, don't sag hips",
        "check": lambda landmarks: check_plank(landmarks),
        "form_points": [
            "✅ Keep your body in a straight line from head to heels",
            "✅ Engage your core and glutes",
            "✅ Keep your neck neutral (look at floor)",
            "❌ Don't let your hips sag toward the floor",
            "❌ Don't raise your hips too high (inverted V position)"
        ]
    },
    "Wall Sit": {
        "demo": asset("assets/wall_sit.webp"),
        "thumbnail": "https://www.inspireusafoundation.org/wp-content/uploads/2022/06/wall-sit.gif",
        "instructions": "Knees at 90°, back against wall, thighs parallel to ground",
        "check": lambda landmarks: check_wall_sit(landmarks),
        "form_points": [
            "✅ Keep your entire back flat against the wall",
            "✅ Bend knees to 90 degree angle",
            "✅ Keep thighs parallel to the ground",
            "❌ Don't let your knees go past your toes",
            "❌ Don't lean forward away from the wall"
        ]
    },
    "Squat Hold": {
        "demo": asset("assets/squat.gif"),
        "thumbnail": "https://www.inspireusafoundation.org/wp-content/uploads/2022/06/air-squat.gif",
        "instructions": "Thighs parallel to ground, chest up, knees over ankles",
        "check": lambda landmarks: check_squat(landmarks),
        "form_points": [
            "✅ Keep your chest up and back straight",
            "✅ Lower until thighs are parallel to ground",
            "✅ Keep knees aligned with toes",
            "❌ Don't let your knees cave inward",
            "❌ Don't round your lower back"
        ]
    }
}

# Posture checking functions
def check_plank(landmarks):
    feedback = []
    correct = True
    
    shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
    
    if abs(shoulder.y - hip.y) > 0.1:
        feedback.append("Keep your back straight!")
        correct = False
    
    if hip.y > shoulder.y + 0.1:
        feedback.append("Hips are too low - lift up!")
        correct = False
    elif hip.y < shoulder.y - 0.1:
        feedback.append("Hips are too high - lower down!")
        correct = False
    
    if correct:
        feedback.append("Perfect plank! You're doing great!")
        speak_text("Perfect form! Keep holding!")
    else:
        speak_text(feedback[0] + " You can do it!")
    
    return correct, feedback

def check_wall_sit(landmarks):
    feedback = []
    correct = True
    
    hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
    knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
    ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
    
    angle = calculate_angle(hip, knee, ankle)
    
    if angle < 70:
        feedback.append("Bend your knees more - go lower!")
        correct = False
    elif angle > 100:
        feedback.append("Too low - come up a bit!")
        correct = False
    
    if correct:
        feedback.append("Perfect wall sit! You're a pro!")
        speak_text("Excellent form! Keep going!")
    else:
        speak_text(feedback[0] + " You've got this!")
    
    return correct, feedback

def check_squat(landmarks):
    feedback = []
    correct = True
    
    shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
    knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
    ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
    
    angle = calculate_angle(hip, knee, ankle)
    
    if angle < 70:
        feedback.append("Too low - come up slightly!")
        correct = False
    elif angle > 100:
        feedback.append("Go lower - bend those knees!")
        correct = False
    
    if shoulder.y > hip.y:
        feedback.append("Chest up! Don't lean forward!")
        correct = False
    
    if correct:
        feedback.append("Perfect squat! You're crushing it!")
        speak_text("Amazing form! Keep holding!")
    else:
        speak_text(feedback[0] + " You can do it!")
    
    return correct, feedback

def calculate_angle(a, b, c):
    a = np.array([a.x, a.y])
    b = np.array([b.x, b.y])
    c = np.array([c.x, c.y])
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360-angle
        
    return angle

def process_frame(frame, exercise_name):
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = pose.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    feedback = ["No pose detected - stand in frame"]
    correct = False
    encouragement = "Get ready! You can do this!"
    
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        if exercise_name in EXERCISES:
            correct, feedback = EXERCISES[exercise_name]["check"](results.pose_landmarks.landmark)
        
        encouragements = [
            "You're doing great!",
            "Keep it up!",
            "Almost there!",
            "You've got this!",
            "Stay strong!"
        ]
        encouragement = np.random.choice(encouragements)
    
    return image, correct, feedback, encouragement

# Login/Signup page
def login_page():
    # App title and description at the top (shown in both tabs)
    st.title("💪 FITKIT - AI Fitness Coach")
    
    # App description/hero section (shown above tabs)
    st.markdown("""
    <style>
        .hero-section {
            background: linear-gradient(135deg, #2C3E50 0%, #1A1A1A 100%);
            padding: 30px;
            border-radius: 20px;
            color: white;
            text-align: center;
            margin-bottom: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
            border: 1px solid #D4AF37;
            position: relative;
            overflow: hidden;
        }
        .hero-section::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: radial-gradient(circle at 20% 50%, rgba(212, 175, 55, 0.1) 0%, transparent 50%);
            pointer-events: none;
        }
        .hero-title {
            font-size: 36px;
            font-weight: bold;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
            color: #D4AF37;
            letter-spacing: 1px;
        }
        .hero-subtitle {
            font-size: 18px;
            color: #E0E0E0;
            margin-bottom: 25px;
            font-weight: 300;
        }
        .feature-grid {
            display: flex;
            justify-content: center;
            gap: 20px;
            margin-top: 20px;
            flex-wrap: wrap;
        }
        .feature-item {
            text-align: center;
            background: rgba(44, 62, 80, 0.8);
            padding: 15px 20px;
            border-radius: 15px;
            flex: 1;
            min-width: 150px;
            border: 1px solid #D4AF37;
            transition: all 0.3s ease;
            backdrop-filter: blur(2px);
        }
        .feature-item:hover {
            transform: translateY(-5px);
            background: #C41E3A;
            border-color: #D4AF37;
            box-shadow: 0 5px 15px rgba(196, 30, 58, 0.3);
        }
        .feature-item:hover .feature-emoji {
            transform: scale(1.1);
        }
        .feature-emoji {
            font-size: 32px;
            margin-bottom: 8px;
            transition: transform 0.3s ease;
            display: inline-block;
        }
        .feature-item div:last-child {
            font-size: 14px;
            font-weight: 600;
            color: #E0E0E0;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .feature-item:hover div:last-child {
            color: white;
        }
        /* Gold accent line */
        .hero-section::after {
            content: '';
            position: absolute;
            bottom: 0;
            left: 10%;
            right: 10%;
            height: 2px;
            background: linear-gradient(90deg, transparent, #D4AF37, transparent);
        }
    </style>

    <div class="hero-section">
        <div class="hero-title">🏋️ Your Personal AI Fitness Coach</div>
        <div class="hero-subtitle">Transform your workouts with real-time posture correction and personalized guidance</div>
        <div class="feature-grid">
            <div class="feature-item">
                <div class="feature-emoji">🎯</div>
                <div>Real-time Posture Analysis</div>
            </div>
            <div class="feature-item">
                <div class="feature-emoji">🗣️</div>
                <div>Voice Feedback</div>
            </div>
            <div class="feature-item">
                <div class="feature-emoji">🎥</div>
                <div>Exercise Demo</div>
            </div>
            <div class="feature-item">
                <div class="feature-emoji">🥗</div>
                <div>Personalized Diet Plans</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["🔐 Login", "📝 Sign Up"])
    
    with tab1:
        # Login form
        col_left, col_right = st.columns([1, 1])
        
        with col_left:
            st.subheader("Welcome Back! 👋")
            st.markdown("Login to continue your fitness journey")
            
            email = st.text_input("📧 Email", key="login_email", placeholder="Enter your email")
            password = st.text_input("🔑 Password", type="password", key="login_password", placeholder="Enter your password")
            
            if st.button("🚀 Login", key="login_btn", use_container_width=True):
                user = check_user(email, password)
                if user:
                    st.session_state.logged_in = True
                    st.session_state.user_id = user[0]
                    st.session_state.user_data = {
                        "name": user[3],
                        "age": user[4],
                        "gender": user[5],
                        "height": user[6],
                        "weight": user[7],
                        "goal": user[8],
                        "activity": user[9],
                        "dietary": user[10].split(',') if user[10] else []
                    }
                    st.session_state.current_page = "Dashboard"
                    st.rerun()
                else:
                    st.error("❌ Invalid email or password")
            
            st.markdown("---")
            st.caption("🔒 Your data is secure and private")
        
        with col_right:
            # Show image only in login tab
            _base_dir    = os.path.dirname(os.path.abspath(__file__))
            _login_img   = os.path.join(_base_dir, "assets", "fitkit_login.png")
            try:
                st.image(_login_img, use_container_width=True)
                st.caption("🎯 AI-powered posture correction in real-time")
            except:
                # Fallback if image doesn't exist
                st.markdown("""
                <div style='background: linear-gradient(135deg, rgba(67, 203, 255, 0.9), rgba(151, 8, 204, 0.9)); 
                            padding: 40px; border-radius: 20px; color: white; 
                            text-align: center; height: 100%; display: flex; 
                            flex-direction: column; justify-content: center;
                            backdrop-filter: blur(5px);
                            border: 1px solid rgba(255,255,255,0.2);'>
                    <div style='font-size: 48px; margin-bottom: 20px;'>🎯</div>
                    <div style='font-size: 24px; font-weight: bold;'>AI Posture Detection</div>
                    <div style='margin-top: 15px; opacity: 0.9;'>Get real-time feedback on your form</div>
                </div>
                """, unsafe_allow_html=True)
    
    with tab2:
        # Sign Up form (NO IMAGE HERE)
        st.subheader("📝 Create Account - Tell Us About Yourself")
        st.markdown("Join thousands of users achieving their fitness goals! 🎯")
        
        with st.form("signup_form"):
            # Account details
            st.markdown("**📋 Account Information**")
            col1, col2 = st.columns(2)
            with col1:
                new_email = st.text_input("📧 Email*", key="signup_email", placeholder="your@email.com")
            with col2:
                name = st.text_input("👤 Full Name*", key="signup_name", placeholder="John Doe")
            
            col1, col2 = st.columns(2)
            with col1:
                new_password = st.text_input("🔑 Password*", type="password", key="signup_password", placeholder="Min 6 characters")
            with col2:
                confirm_password = st.text_input("✓ Confirm Password*", type="password", key="confirm_password", placeholder="Re-enter password")
            
            st.divider()
            
            # Personal details
            st.markdown("**🧑‍🤝‍🧑 Personal Information**")
            col1, col2, col3 = st.columns(3)
            with col1:
                age = st.number_input("🎂 Age*", min_value=15, max_value=100, value=25)
            with col2:
                gender = st.selectbox("⚥ Gender*", ["Male", "Female", "Other"])
            with col3:
                height = st.number_input("📏 Height (cm)*", min_value=100, max_value=250, value=170)
            
            col1, col2 = st.columns(2)
            with col1:
                weight = st.number_input("⚖️ Weight (kg)*", min_value=30, max_value=200, value=70)
            with col2:
                goal = st.selectbox("🎯 Fitness Goal*", 
                                   ["Weight Loss", "Muscle Gain", "Stay Fit", "Improve Posture"])
            
            # Additional details
            activity = st.select_slider("💪 Activity Level*",
                                       options=["Sedentary", "Light", "Moderate", "Active", "Very Active"],
                                       value="Moderate",
                                       help="How active are you in daily life?")
            
            dietary_pref = st.multiselect("🥗 Dietary Preferences (Optional)",
                                        ["Vegetarian", "Non-Veg", "Vegan", "Gluten-Free", "Dairy-Free", "None"],
                                        help="Select all that apply")
            
            st.markdown("---")
            submitted = st.form_submit_button("✨ Create Account", use_container_width=True)
            
            if submitted:
                # Validation
                if not new_email or not name or not new_password:
                    st.error("❌ Please fill all required fields")
                elif new_password != confirm_password:
                    st.error("❌ Passwords don't match")
                elif len(new_password) < 6:
                    st.error("❌ Password must be at least 6 characters")
                else:
                    # Add user with all details
                    success = add_user(new_email, new_password, name, age, gender, height, weight, 
                                     goal, activity, dietary_pref)
                    if success:
                        st.success("✅ Account created successfully! Please login.")
                    else:
                        st.error("❌ Email already exists")
            
            # Additional info at bottom
            st.markdown("---")
            st.caption("✨ By signing up, you agree to our Terms of Service and Privacy Policy")

# Dashboard
def dashboard_page():
    st.title(f"✨  Welcome, {st.session_state.user_data.get('name', 'User')}!")
    
    # Get exercise history
    conn = sqlite3.connect(asset('fitkit.db'))
    c = conn.cursor()
    c.execute('''SELECT COUNT(*), SUM(calories), AVG(posture_accuracy)
                 FROM exercise_log WHERE user_id=? AND date(timestamp)=date('now')''',
              (st.session_state.user_id,))
    stats = c.fetchone()
    conn.close()
    today_count    = stats[0] or 0
    today_calories = round(stats[1] or 0, 1)
    today_accuracy = round(stats[2] or 0, 1)

    # ── Today's Stats Banner ──────────────────────────────────────────────
    sc1, sc2, sc3 = st.columns(3)
    sc1.metric("🏋️ Sessions Today",   today_count)
    sc2.metric("🔥 Calories Burned",  f"{today_calories} kcal")
    sc3.metric("🎯 Avg Form Accuracy", f"{today_accuracy}%")
    
    # Calculate BMI
    weight = st.session_state.user_data.get('weight', 70)
    height_cm = st.session_state.user_data.get('height', 170)
    height_m = height_cm / 100
    bmi = weight / (height_m * height_m)
    
    # Determine BMI category
    if bmi < 18.5:
        bmi_category = "Underweight"
        bmi_color = "#FF6B6B"  # Red
        bmi_emoji = "⚠️"
    elif bmi < 25:
        bmi_category = "Normal"
        bmi_color = "#51CF66"  # Green
        bmi_emoji = "✅"
    else:
        bmi_category = "Overweight"
        bmi_color = "#FFA94D"  # Orange
        bmi_emoji = "⚠️"
    

    
    # ==================== BODY ASSESSMENT SECTION ====================
    st.markdown("---")
    st.markdown("## 🏆 **Body Assessment & Personalized Plan**")

    # Create two main columns
    col_assessment_left, col_assessment_right = st.columns(2, gap="large")

    with col_assessment_left:
            user_height = st.session_state.user_data.get('height', 170)
            st.markdown(f"### 📏 **Height-Weight Guide for {user_height} cm**")
            
            # Calculate weight ranges
            under_weight_end = 18.5 * (height_m ** 2)
            normal_weight_end = 25 * (height_m ** 2)
            
            # Create a styled HTML table instead of dataframe for better visibility
            bmi_color_map = {
                "Underweight": "#FF6B6B",
                "Normal": "#51CF66", 
                "Overweight": "#FFA94D"
            }
            
            # Determine which row to highlight
            under_highlight = "style='background-color: rgba(255,215,0,0.3); border: 2px solid gold; font-weight: bold; color: black;'" if bmi_category == "Underweight" else ""
            normal_highlight = "style='background-color: rgba(255,215,0,0.3); border: 2px solid gold; font-weight: bold; color: black;'" if bmi_category == "Normal" else ""
            over_highlight = "style='background-color: rgba(255,215,0,0.3); border: 2px solid gold; font-weight: bold; color: black;'" if bmi_category == "Overweight" else ""
            
            guide_html = f"""
            <style>
                .guide-table {{
                    width: 100%;
                    border-collapse: collapse;
                    font-size: 18px;
                    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                }}
                .guide-table th {{
                    background: linear-gradient(135deg, #667eea, #764ba2);
                    color: white;
                    padding: 15px;
                    font-size: 20px;
                    text-align: center;
                }}
                .guide-table td {{
                    padding: 15px;
                    text-align: center;
                    border-bottom: 1px solid #ddd;
                    background-color: #f8f9fa;
                    color: #000000;  /* BLACK text color for visibility */
                    font-weight: 500;
                }}
                .guide-table tr:hover td {{
                    background-color: #e9ecef;
                    color: #000000;  /* Keep black on hover */
                }}
                .bmi-card {{
                    background: linear-gradient(135deg, #667eea, #764ba2);
                    padding: 20px;
                    border-radius: 15px;
                    color: white;
                    text-align: center;
                    font-size: 24px;
                    font-weight: bold;
                    margin-top: 20px;
                    box-shadow: 0 10px 20px rgba(0,0,0,0.2);
                }}
            </style>
            
            <table class="guide-table">
                <tr>
                    <th>Category</th>
                    <th>Weight Range</th>
                    <th>BMI Range</th>
                </tr>
                <tr {under_highlight}>
                    <td>⚠️ Underweight</td>
                    <td>< {under_weight_end:.1f} kg</td>
                    <td>< 18.5</td>
                </tr>
                <tr {normal_highlight}>
                    <td>✅ Normal</td>
                    <td>{under_weight_end:.1f} - {normal_weight_end:.1f} kg</td>
                    <td>18.5 - 25</td>
                </tr>
                <tr {over_highlight}>
                    <td>⚠️ Overweight</td>
                    <td>> {normal_weight_end:.1f} kg</td>
                    <td>> 25</td>
                </tr>
            </table>
            
            <div class="bmi-card" style="background: {bmi_color_map[bmi_category]};">
                Your BMI: {bmi:.1f} - {bmi_category}
            </div>
            """
            
            st.markdown(guide_html, unsafe_allow_html=True)
    
    with col_assessment_right:
        st.markdown("### 🧑‍⚕️ **Your Health Status**")
        
        # Generate personalized health statement based on BMI
        if bmi_category == "Underweight":
            if bmi < 16:
                health_statement = "🍎 **Let's nourish!** You're in the underweight range. We'll focus on healthy calorie surplus and strength training."
            else:
                health_statement = "🌱 **Time to grow!** You're slightly underweight. Our plan will help you gain healthy weight with good nutrition."
        elif bmi_category == "Normal":
            health_statement = "✅ **Congratulations!** You are at a healthy weight. Let's maintain this and build a fit, toned body!"
            if st.session_state.user_data.get('goal') == "Weight Loss":
                health_statement = "💪 **Perfect weight for recomp!** You're healthy - let's tone up while keeping that good weight."
            elif st.session_state.user_data.get('goal') == "Muscle Gain":
                health_statement = "🏋️ **Ideal for muscle gain!** You're at the perfect weight to build strength and size."
            else:
                health_statement = "✨ **You're in the sweet spot!** Healthy weight achieved - let's maintain and have fun with fitness."
        else:  # Overweight
            if bmi > 30:
                health_statement = "🌟 **Your journey starts here!** You're in the overweight range. Don't worry - we'll take it step by step together."
            elif bmi > 27:
                health_statement = "💫 **Making progress!** You're moderately overweight. Consistency is key - you've got this!"
            else:
                health_statement = "🌿 **So close!** You're mildly overweight. Just a few steps away from your goal weight!"
        
        # User info card with enhanced styling
        with st.container():
            st.markdown(f"""
            <style>
                .health-card {{
                    background: linear-gradient(135deg, #43CBFF, #9708CC);
                    padding: 25px;
                    border-radius: 20px;
                    color: white;
                    box-shadow: 0 10px 20px rgba(0,0,0,0.2);
                    margin-bottom: 15px;
                }}
                .health-card h3 {{
                    margin-top: 0;
                    font-size: 28px;
                    border-bottom: 2px solid rgba(255,255,255,0.3);
                    padding-bottom: 10px;
                    margin-bottom: 15px;
                }}
                .health-card p {{
                    font-size: 18px;
                    margin: 12px 0;
                    display: flex;
                    justify-content: space-between;
                    border-bottom: 1px solid rgba(255,255,255,0.1);
                    padding-bottom: 8px;
                }}
                .health-card strong {{
                    opacity: 0.9;
                }}
                .health-statement {{
                    background: {bmi_color_map[bmi_category]};
                    padding: 20px;
                    border-radius: 15px;
                    color: white;
                    font-size: 18px;
                    font-weight: 500;
                    text-align: center;
                    box-shadow: 0 5px 15px rgba(0,0,0,0.2);
                    border: 2px solid rgba(255,255,255,0.3);
                    margin-top: 15px;
                    line-height: 1.6;
                }}
                .bmi-badge {{
                    background: rgba(255,255,255,0.2);
                    padding: 5px 15px;
                    border-radius: 50px;
                    font-size: 14px;
                    display: inline-block;
                    margin-left: 10px;
                }}
            </style>
            
            <div class="health-card">
                <h3>👤 {st.session_state.user_data.get('name', 'User')}</h3>
                <p><span>📏 Height</span> <strong>{height_cm} cm</strong></p>
                <p><span>⚖️ Weight</span> <strong>{weight} kg</strong></p>
                <p><span>🎂 Age</span> <strong>{st.session_state.user_data.get('age', 25)} years</strong></p>
                <p><span>🎯 Goal</span> <strong>{st.session_state.user_data.get('goal', 'Stay Fit')}</strong></p>
                <p><span>💪 Activity</span> <strong>{st.session_state.user_data.get('activity', 'Moderate')}</strong></p>
                <p><span>📊 BMI</span> <strong>{bmi:.1f} <span class="bmi-badge">{bmi_category}</span></strong></p>
            </div>
            
            <div class="health-statement" style="background: {bmi_color_map[bmi_category]};">
                {health_statement}
            </div>
            """, unsafe_allow_html=True)
        
        # Recommendation
        goal = st.session_state.user_data.get('goal', 'Stay Fit')
        if bmi_category == "Normal":
            if goal == "Weight Loss":
                rec = "⚠️ You're at a healthy weight but want to lose. Focus on body recomposition with strength training and slight calorie deficit."
            elif goal == "Muscle Gain":
                rec = "💪 Great! You're at a healthy weight for muscle gain. Focus on protein intake and progressive overload."
            else:
                rec = "✅ You're at an ideal weight! Maintain with balanced diet and regular exercise."
        elif bmi_category == "Underweight":
            if goal == "Weight Loss":
                rec = "⚠️ You're underweight - weight loss isn't recommended. Please consult a healthcare provider."
            else:
                rec = "🍽️ Focus on nutrient-dense foods and strength training to gain healthy weight."
        else:
            if goal == "Muscle Gain":
                rec = "💪 You can build muscle while losing fat. Focus on protein intake and consistent workouts."
            else:
                rec = "🔥 Great goal! Focus on cardio, portion control, and consistency."
        
        st.info(rec)
    
    # ==================== WORKOUT RECOMMENDATION SECTION ====================
    st.markdown("---")
    st.markdown("### 🎯 **Your Personalized Workout Plan**")
    
    col_workout_left, col_workout_right = st.columns(2)
    
    with col_workout_left:
        st.markdown("#### 🏋️ AI RECOMMENDED PLAN")
        
        # Calculate recommended workout based on BMI and goal
        if bmi_category == "Underweight":
            plans = {
                "Plank": "3 sets × 20 sec",
                "Wall Sit": "3 sets × 30 sec",
                "Squat Hold": "3 sets × 12 reps",
                "Total": "15 minutes",
                "Intensity": "Light to Moderate"
            }
        elif bmi_category == "Overweight":
            plans = {
                "Plank": "4 sets × 45 sec",
                "Wall Sit": "4 sets × 60 sec",
                "Squat Hold": "4 sets × 15 reps",
                "Total": "25 minutes",
                "Intensity": "Moderate to High"
            }
        else:  # Normal
            if goal == "Muscle Gain":
                plans = {
                    "Plank": "4 sets × 40 sec",
                    "Wall Sit": "4 sets × 50 sec",
                    "Squat Hold": "4 sets × 15 reps",
                    "Total": "22 minutes",
                    "Intensity": "High"
                }
            else:
                plans = {
                    "Plank": "3 sets × 30 sec",
                    "Wall Sit": "3 sets × 45 sec",
                    "Squat Hold": "3 sets × 12 reps",
                    "Total": "18 minutes",
                    "Intensity": "Moderate"
                }
        
        # Display workout plan in a styled box
        with st.container():
            st.markdown(f"""
            <div style='background: linear-gradient(145deg, #2B32B2, #1488CC); 
                        padding: 20px; border-radius: 15px; color: white;'>
                <div style='display: flex; justify-content: space-between; 
                           background: rgba(255,255,255,0.1); padding: 10px; 
                           border-radius: 10px; margin-bottom: 10px;'>
                    <span>🔹 Plank</span>
                    <span style='background: gold; color: black; padding: 5px 15px; 
                                 border-radius: 20px; font-weight: bold;'>{plans['Plank']}</span>
                </div>
                <div style='display: flex; justify-content: space-between; 
                           background: rgba(255,255,255,0.1); padding: 10px; 
                           border-radius: 10px; margin-bottom: 10px;'>
                    <span>🔹 Wall Sit</span>
                    <span style='background: gold; color: black; padding: 5px 15px; 
                                 border-radius: 20px; font-weight: bold;'>{plans['Wall Sit']}</span>
                </div>
                <div style='display: flex; justify-content: space-between; 
                           background: rgba(255,255,255,0.1); padding: 10px; 
                           border-radius: 10px; margin-bottom: 10px;'>
                    <span>🔹 Squat Hold</span>
                    <span style='background: gold; color: black; padding: 5px 15px; 
                                 border-radius: 20px; font-weight: bold;'>{plans['Squat Hold']}</span>
                </div>
                <div style='background: rgba(0,0,0,0.3); padding: 15px; 
                           border-radius: 10px; text-align: center; margin-top: 15px;
                           border: 2px solid gold;'>
                    <div style='font-size: 20px;'>⏱️ Total Duration</div>
                    <div style='font-size: 28px; font-weight: bold; color: gold;'>{plans['Total']}</div>
                    <div style='color: #ddd;'>Intensity: {plans['Intensity']}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    with col_workout_right:
        st.markdown("#### ⚙️ YOUR CUSTOM PLAN")
        
        # Custom selectbox
        workout_choice = st.selectbox(
            "Choose your workout intensity:",
            ["🔋 Recommended (AI Optimized)", 
             "⚡ Beginner (Lighter)", 
             "🔥 Intermediate (Balanced)", 
             "💪 Advanced (Intensive)"],
            key="workout_choice"
        )
        
        # Generate custom plan based on selection
        if workout_choice == "🔋 Recommended (AI Optimized)":
            custom_plans = {
                "Plank": plans['Plank'],
                "Wall Sit": plans['Wall Sit'],
                "Squat": plans['Squat Hold'],
                "Note": "✓ Optimized for your BMI & Goal"
            }
        elif workout_choice == "⚡ Beginner (Lighter)":
            custom_plans = {
                "Plank": "2 sets × 20 sec",
                "Wall Sit": "2 sets × 30 sec",
                "Squat": "2 sets × 10 reps",
                "Note": "🌱 Perfect for starting out"
            }
        elif workout_choice == "🔥 Intermediate (Balanced)":
            custom_plans = {
                "Plank": "3 sets × 40 sec",
                "Wall Sit": "3 sets × 50 sec",
                "Squat": "3 sets × 15 reps",
                "Note": "⭐ Great for steady progress"
            }
        else:
            custom_plans = {
                "Plank": "5 sets × 60 sec",
                "Wall Sit": "5 sets × 75 sec",
                "Squat": "5 sets × 20 reps",
                "Note": "⚡ For experienced athletes"
            }
        
        # Display custom plan
        with st.container():
            st.markdown(f"""
            <div style='background: linear-gradient(145deg, #E44D2E, #F09819); 
                        padding: 20px; border-radius: 15px; color: white;'>
                <div style='background: rgba(255,255,255,0.2); padding: 20px; 
                        border-radius: 10px; margin-bottom: 15px;'>
                    <h4 style='color: white; text-align: center; margin-top:0; font-size: 22px;'>📋 Your Selected Plan</h4>
                    <p style='font-size: 18px; margin: 12px 0;'><strong>Plank:</strong> {custom_plans['Plank']}</p>
                    <p style='font-size: 18px; margin: 12px 0;'><strong>Wall Sit:</strong> {custom_plans['Wall Sit']}</p>
                    <p style='font-size: 18px; margin: 12px 0;'><strong>Squat Hold:</strong> {custom_plans['Squat']}</p>
                    <div style='background: rgba(255,215,0,0.3); border-radius: 10px; 
                                padding: 15px; text-align: center; margin-top: 10px;'>
                        <span style='color: #8B4513; font-weight: bold; font-size: 18px;'>✨ {custom_plans['Note']}</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Store the selected plan
    st.session_state.selected_plan = {
        "plank": custom_plans['Plank'],
        "wallsit": custom_plans['Wall Sit'],
        "squat": custom_plans['Squat'],
        "choice": workout_choice
    }
    
    # Original bottom section
    st.markdown("---")
    col_original_left, col_original_right = st.columns([2, 1])
    
    with col_original_left:
        st.subheader("🎯 Today's Tasks")
        
        goal = st.session_state.user_data.get('goal', 'Stay Fit')
        if goal == "Weight Loss":
            st.info("🔥 **Weight Loss Focus:**")
            st.write("• 30 mins cardio")
            st.write(f"• Plank {custom_plans['Plank']}")
            st.write(f"• Wall Sit {custom_plans['Wall Sit']}")
            st.write(f"• Squat Hold {custom_plans['Squat']}")
            st.write("• 500 calorie deficit target")
        elif goal == "Muscle Gain":
            st.info("💪 **Muscle Gain Focus:**")
            st.write(f"• Squat Hold {custom_plans['Squat']}")
            st.write("• Push-ups 3x10 reps")
            st.write(f"• Plank {custom_plans['Plank']}")
            st.write("• Protein: 1.6g/kg body weight")
        else:
            st.info("🧘 **Maintenance Focus:**")
            st.write(f"• Plank {custom_plans['Plank']}")
            st.write(f"• Wall Sit {custom_plans['Wall Sit']}")
            st.write(f"• Squat Hold {custom_plans['Squat']}")
            st.write("• Stay active throughout day")
    
    with col_original_right:
        st.subheader("💡 Quick Tip")
        tips = [
            "Drink water before workout!",
            "Don't forget to warm up",
            "Rest 30 sec between sets",
            "Track your meals",
            "Sleep 7-8 hours",
            "Stretch after workout"
        ]
        st.info(f"✨ {np.random.choice(tips)}")
        
        st.subheader("🎯 Your Goal")
        st.success(f"**{st.session_state.user_data.get('goal', 'Stay Fit')}**")
        
        if st.button("🏋️ Start Workout →", use_container_width=True):
            st.session_state.current_page = "Exercise"
            st.rerun()

# Exercise page with working demos
def exercise_page():
    st.title("🏋️ Exercise Zone")

    # ── Category definitions with metadata ───────────────────────────────────
    EXERCISE_CATALOG = {
        "Upper Body 💪": [
            {"name":"Bicep Curl",      "emoji":"💪","desc":"Classic curl — elbow flex"},
            {"name":"Hammer Curl",     "emoji":"🔨","desc":"Neutral-grip curl — brachialis"},
            {"name":"Push-up",         "emoji":"⬆️","desc":"Chest & tricep bodyweight"},
            {"name":"Bench Press",     "emoji":"🏋️","desc":"Horizontal chest press"},
            {"name":"Overhead Press",  "emoji":"🙌","desc":"Vertical press — deltoids"},
            {"name":"Arnold Press",    "emoji":"🌀","desc":"Rotating shoulder press"},
            {"name":"Shoulder Press",  "emoji":"🏅","desc":"Dumbbell shoulder press"},
            {"name":"Lateral Raise",   "emoji":"🦅","desc":"Side deltoid isolation"},
            {"name":"Face Pull",       "emoji":"🎯","desc":"Rear delt & rotator cuff"},
            {"name":"Bent-over Row",   "emoji":"🚣","desc":"Back thickness builder"},
            {"name":"Pull-up",         "emoji":"🔝","desc":"Vertical pull — lats & biceps"},
            {"name":"Cable Fly",       "emoji":"🪁","desc":"Chest fly — pec isolation"},
            {"name":"Tricep Dip",      "emoji":"⬇️","desc":"Tricep compound bodyweight"},
        ],
        "Lower Body 🦵": [
            {"name":"Squat Hold",      "emoji":"🪑","desc":"Isometric squat hold"},
            {"name":"Sumo Squat",      "emoji":"🤼","desc":"Wide-stance quad & inner"},
            {"name":"Lunge",           "emoji":"🚶","desc":"Unilateral leg builder"},
            {"name":"Deadlift",        "emoji":"⚓","desc":"Hip hinge — posterior chain"},
            {"name":"Leg Press",       "emoji":"🦿","desc":"Machine quad & glute press"},
            {"name":"Hip Thrust",      "emoji":"🍑","desc":"Glute isolator — weighted"},
            {"name":"Glute Bridge",    "emoji":"🌉","desc":"Floor-based glute activator"},
            {"name":"Calf Raise",      "emoji":"🦶","desc":"Gastrocnemius & soleus"},
            {"name":"Box Step-Up",     "emoji":"📦","desc":"Unilateral functional leg"},
        ],
        "Core & Full Body 🔥": [
            {"name":"Plank",           "emoji":"🏄","desc":"Isometric core hold"},
            {"name":"Wall Sit",        "emoji":"🧱","desc":"Isometric quad hold"},
            {"name":"Mountain Climber","emoji":"⛰️","desc":"Core + cardio combo"},
            {"name":"Burpee",          "emoji":"💥","desc":"Full-body HIIT movement"},
            {"name":"Jumping Jack",    "emoji":"⭐","desc":"Cardio warm-up classic"},
            {"name":"High Knees",      "emoji":"🏃","desc":"Running cardio drill"},
        ],
        "Back & Shoulders 🎯": [
            {"name":"Bent-over Row",   "emoji":"🚣","desc":"Back thickness builder"},
            {"name":"Face Pull",       "emoji":"🎯","desc":"Rear delt & rotator cuff"},
            {"name":"Lateral Raise",   "emoji":"🦅","desc":"Side deltoid isolation"},
            {"name":"Shoulder Press",  "emoji":"🏅","desc":"Dumbbell shoulder press"},
        ],
    }

    # ── Check if an exercise is already selected ──────────────────────────────
    if "selected_exercise" not in st.session_state:
        st.session_state.selected_exercise = None

    # ── Back button ───────────────────────────────────────────────────────────
    if st.session_state.selected_exercise:
        col_back, col_title = st.columns([1, 5])
        with col_back:
            if st.button("← Back to List", key="ex_back"):
                st.session_state.selected_exercise = None
                st.rerun()
        with col_title:
            st.subheader(f"🏋️ {st.session_state.selected_exercise}")
        st.markdown("---")
        _dispatch_exercise(st.session_state.selected_exercise)
        return

    # ── Exercise selection grid ───────────────────────────────────────────────
    st.markdown("### 🏋️ Choose Your Exercise")
    st.caption("Select a category and click any exercise card to start.")

    # Deduplicate for display (some appear in multiple categories)
    displayed = set()
    for category, exercises_list in EXERCISE_CATALOG.items():
        # Filter unique within this category for display
        unique = [e for e in exercises_list if e["name"] not in displayed]
        if not unique:
            continue
        for e in unique:
            displayed.add(e["name"])

        st.markdown(f"#### {category}")
        cols = st.columns(4)
        for i, ex in enumerate(unique):
            with cols[i % 4]:
                # Card button using container + markdown trick
                card_html = f"""
<div style='background:linear-gradient(135deg,#1e1e2e,#2a2a3e);
     border:1px solid #D4AF37;border-radius:12px;padding:14px 10px;
     text-align:center;cursor:pointer;margin:4px 0;min-height:100px;'>
  <div style='font-size:26px;'>{ex['emoji']}</div>
  <div style='color:#D4AF37;font-weight:700;font-size:13px;margin-top:4px;'>{ex['name']}</div>
  <div style='color:#aaa;font-size:11px;margin-top:3px;'>{ex['desc']}</div>
</div>"""
                st.markdown(card_html, unsafe_allow_html=True)
                if st.button("Start", key=f"ex_btn_{ex['name']}", use_container_width=True):
                    st.session_state.selected_exercise = ex["name"]
                    st.rerun()
        st.markdown("")


def _dispatch_exercise(exercise: str):
    """Route the selected exercise to its detection function."""
    uid = st.session_state.user_id

    dispatch = {
        "Bicep Curl":       lambda: bicep_detection(uid, log_exercise),
        "Push-up":          lambda: pushup_detection(uid, log_exercise),
        "Jumping Jack":     lambda: jumping_jack_detection(uid, log_exercise),
        "Shoulder Press":   lambda: shoulder_press_detection(uid, log_exercise),
        "Lateral Raise":    lambda: lateral_raise_detection(uid, log_exercise),
        "Lunge":            lambda: lunge_detection(uid, log_exercise),
        "Tricep Dip":       lambda: tricep_dip_detection(uid, log_exercise),
        "Deadlift":         lambda: deadlift_detection(uid, log_exercise),
        "Bench Press":      lambda: bench_press_detection(uid, log_exercise),
        "Pull-up":          lambda: pull_up_detection(uid, log_exercise),
        "Bent-over Row":    lambda: row_detection(uid, log_exercise),
        "Overhead Press":   lambda: overhead_press_detection(uid, log_exercise),
        "Mountain Climber": lambda: mountain_climber_detection(uid, log_exercise),
        "Hip Thrust":       lambda: hip_thrust_detection(uid, log_exercise),
        "Burpee":           lambda: burpee_detection(uid, log_exercise),
        "Leg Press":        lambda: lp_detection(uid, log_exercise),
        "Cable Fly":        lambda: cf_detection(uid, log_exercise),
        "Sumo Squat":       lambda: ss_detection(uid, log_exercise),
        "Face Pull":        lambda: fp_detection(uid, log_exercise),
        "Box Step-Up":      lambda: bsu_detection(uid, log_exercise),
        "Hammer Curl":      lambda: hammer_curls_detection(uid, log_exercise),
        "Arnold Press":     lambda: arnold_press_detection(uid, log_exercise),
        "Calf Raise":       lambda: calf_raise_detection(uid, log_exercise),
        "High Knees":       lambda: high_knees_detection(uid, log_exercise),
        "Glute Bridge":     lambda: glute_bridge_detection(uid, log_exercise),
    }

    # Posture-hold exercises handled inline
    posture_exercises = {"Plank", "Wall Sit", "Squat Hold"}

    if exercise in dispatch:
        dispatch[exercise]()
    elif exercise in posture_exercises:
        _posture_exercise(exercise)
    else:
        st.warning(f"Exercise '{exercise}' not found.")


def _posture_exercise(exercise: str):
    """Handle Plank / Wall Sit / Squat Hold inline."""
    exercise_info = EXERCISES[exercise]
    input_type = st.radio("Choose Input Method",
                          ["Realtime Camera", "Upload Image", "Upload Video"],
                          horizontal=True)
    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader("📹 Your Input")
        if input_type == "Realtime Camera":
            run            = st.checkbox("Start Camera")
            FRAME_WINDOW   = st.empty()
            dashboard_panel = st.empty()
            if run:
                cap = cv2.VideoCapture(0)
                throttler  = FeedbackThrottler(interval=3.0)
                start_time = time.time()
                correct_frames = 0
                total_frames   = 0
                while run:
                    ret, frame = cap.read()
                    if ret:
                        frame = cv2.flip(frame, 1)
                        processed_frame, correct, feedback_list, encouragement = process_frame(frame, exercise)
                        total_frames += 1
                        if correct: correct_frames += 1
                        elapsed  = time.time() - start_time
                        accuracy = (correct_frames / total_frames) * 100 if total_frames > 0 else 0
                        feedback_str = feedback_list[0] if feedback_list else encouragement
                        processed_frame = draw_feedback_overlay(
                            processed_frame, feedback_str, correct,
                            0, "HOLD" if correct else "FIX FORM", accuracy, exercise)
                        FRAME_WINDOW.image(processed_frame, channels="BGR", use_container_width=True)
                        with dashboard_panel.container():
                            render_dashboard(0, accuracy, "HOLD" if correct else "",
                                             feedback_str, correct, exercise, elapsed)
                        speak_js(feedback_str, throttler)
                        time.sleep(0.1)
                duration = int(time.time() - start_time)
                accuracy = (correct_frames / total_frames) * 100 if total_frames > 0 else 0
                if duration > 5:
                    log_exercise(st.session_state.user_id, exercise, duration, duration * 5, accuracy)
                cap.release()
        elif input_type == "Upload Image":
            uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
            if uploaded_file:
                image = Image.open(uploaded_file)
                frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                processed_frame, correct, feedback_list, encouragement = process_frame(frame, exercise)
                st.image(processed_frame, channels="BGR")
                for fb in feedback_list:
                    if "Perfect" in fb or "great" in fb: st.success(f"✅ {fb}")
                    else: st.warning(f"⚠️ {fb}")
                st.success(f"💪 {encouragement}") if correct else st.info(f"🎯 {encouragement}")
        else:
            uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov"])
            if uploaded_file:
                tfile = tempfile.NamedTemporaryFile(delete=False)
                tfile.write(uploaded_file.read())
                cap = cv2.VideoCapture(tfile.name)
                correct_frames = total_frames = 0
                fp2 = st.empty(); fb2 = st.empty()
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret: break
                    processed_frame, correct, feedback_list, encouragement = process_frame(frame, exercise)
                    fp2.image(processed_frame, channels="BGR")
                    total_frames += 1
                    if correct: correct_frames += 1
                    with fb2.container():
                        for fb in feedback_list:
                            if "Perfect" in fb or "great" in fb: st.success(f"✅ {fb}")
                            else: st.warning(f"⚠️ {fb}")
                    time.sleep(0.1)
                accuracy = (correct_frames / total_frames) * 100 if total_frames > 0 else 0
                st.info(f"📊 Accuracy: {accuracy:.0f}%")
                cap.release(); os.unlink(tfile.name)
    with col_right:
        st.subheader("🎯 Correct Form")
        demo_tab, tips_tab = st.tabs(["📺 Demo", "📝 Tips"])
        with demo_tab:
            st.image(exercise_info["demo"], caption=f"✅ {exercise} Form", use_container_width=True)
            st.info(f"**Instructions:** {exercise_info['instructions']}")
        with tips_tab:
            for point in exercise_info["form_points"]:
                if "✅" in point: st.success(point)
                elif "❌" in point: st.error(point)
                else: st.write(point)


# Advisory page
def advisory_page():
    st.title("🥗 Advisory & Recommendations")
    
    user = st.session_state.user_data
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🍽️ Recommended Foods")
        
        goal = user.get('goal', 'Stay Fit')
        dietary = user.get('dietary', [])
        
        # Enhanced dietary preference badges
        if dietary:
            badges = []
            for diet in dietary:
                if diet == "Vegetarian":
                    badges.append("🌱 Veg")
                elif diet == "Non-Veg":
                    badges.append("🍗 Non-Veg")
                elif diet == "Vegan":
                    badges.append("🌿 Vegan")
                elif diet == "Gluten-Free":
                    badges.append("🌾 GF")
                elif diet == "Dairy-Free":
                    badges.append("🥛 DF")
                elif diet == "None":
                    badges.append("⚪ No Restrictions")
            
            st.caption(f"**Your Preferences:** {' | '.join(badges)}")
        
        if goal == "Weight Loss":
            # Protein section based on dietary preference
            if "Vegan" in dietary:
                st.success("**🥩 Proteins:** Tofu, tempeh, lentils, chickpeas, black beans")
            elif "Vegetarian" in dietary:
                st.success("**🥩 Proteins:** Eggs, Greek yogurt, paneer, tofu, lentils")
            elif "Non-Veg" in dietary:
                st.success("**🥩 Proteins:** Chicken breast, fish, eggs, turkey")
            else:
                st.success("**🥩 Proteins:** Chicken, fish, eggs, tofu, legumes")
            
            st.success("**🥦 Vegetables:** Broccoli, spinach, bell peppers, zucchini, kale")
            st.success("**🍎 Fruits:** Berries, apples, grapefruit, oranges, pears")
            st.success("**🌾 Carbs:** Quinoa, oats, brown rice, sweet potatoes")
            st.warning("⚠️ **Avoid:** Sugary drinks, processed foods, excess carbs, fried items")
            st.info(f"📊 **Your Calorie Target:** {(user.get('weight', 70) * 24):.0f} kcal/day")
        
        elif goal == "Muscle Gain":
            # Protein section based on dietary preference
            if "Vegan" in dietary:
                st.success("**🌱 Plant Proteins:** Tofu, tempeh, seitan, lentils, chickpeas")
                st.success("**🥜 Nuts & Seeds:** Peanut butter, almonds, chia seeds, hemp seeds")
                st.success("**🌾 Grains:** Quinoa (complete protein), oats")
            elif "Vegetarian" in dietary:
                st.success("**🥚 Eggs & Dairy:** Eggs, Greek yogurt, cottage cheese, paneer")
                st.success("**🌱 Plant Proteins:** Tofu, lentils, chickpeas, beans")
                st.success("**🥜 Nuts:** Peanut butter, almonds, walnuts")
            elif "Non-Veg" in dietary:
                st.success("**🥩 Lean Meats:** Chicken breast, lean beef, turkey")
                st.success("**🐟 Fish:** Salmon, tuna, tilapia (great for omega-3)")
                st.success("**🥚 Eggs:** Whole eggs for healthy fats")
            else:
                st.success("**🥩 Proteins:** Chicken, fish, eggs, lean beef, tofu")
            
            st.success("**🌾 Complex Carbs:** Brown rice, quinoa, oats, sweet potatoes, whole wheat pasta")
            st.success("**🥜 Healthy Fats:** Avocado, olive oil, nuts, seeds")
            
            protein_target = user.get('weight', 70) * 1.6
            st.warning(f"💪 **Protein Target:** {protein_target:.0f}g per day")
            
            if "Non-Veg" in dietary:
                st.info("🐟 **Omega-3 Tip:** Include fatty fish (salmon, mackerel) twice a week")
            elif "Vegetarian" in dietary:
                st.info("🌱 **Complete Proteins:** Combine rice + lentils or wheat + beans for complete amino acids")
        
        else:  # Stay Fit / Maintenance
            if "Vegan" in dietary:
                st.success("**🌱 Plant-Based:** Variety of fruits, vegetables, legumes, whole grains")
                st.success("**🥜 Healthy Fats:** Avocado, nuts, seeds, olive oil")
            elif "Vegetarian" in dietary:
                st.success("**🥚 Balanced:** Eggs, dairy, legumes, vegetables, fruits")
                st.success("**🌾 Whole grains for sustained energy**")
            elif "Non-Veg" in dietary:
                st.success("**🍗 Lean proteins in moderation**")
                st.success("**🥗 Plenty of vegetables and fruits**")
            else:
                st.success("**🍳 Balanced meals with all food groups**")
                st.success("**🥗 Fresh fruits & vegetables**")
            
            st.success("**💧 Stay hydrated with 8-10 glasses of water**")
            st.success("**🏃 Regular physical activity**")
    
    with col2:
        st.subheader("💪 Daily Tips")
        
        # Personalized tips based on dietary preference
        base_tips = [
            "💧 Drink 8 glasses of water daily",
            "😴 Sleep 7-8 hours for muscle recovery",
            "🏃 Never skip warm-up (5-10 mins)",
            "📊 Track your protein intake",
            "🔄 Rest 48 hours between same muscle groups",
            "🧘 Stretch after every workout"
        ]
        
        # Add dietary-specific tips
        dietary_tips = []
        if "Vegan" in dietary:
            dietary_tips = [
                "🌿 Consider B12 supplementation",
                "🌱 Eat a variety of protein sources for complete amino acids",
                "🥛 Try fortified plant milks for calcium"
            ]
        elif "Vegetarian" in dietary:
            dietary_tips = [
                "🥚 Eggs are a great complete protein source",
                "🌱 Combine grains and legumes for complete proteins",
                "🥛 Don't forget calcium-rich dairy or alternatives"
            ]
        elif "Non-Veg" in dietary:
            dietary_tips = [
                "🍗 Choose lean cuts and remove skin from poultry",
                "🐟 Include fatty fish for omega-3 fatty acids",
                "🥩 Limit red meat to 1-2 times per week"
            ]
        
        if "Gluten-Free" in dietary:
            dietary_tips.append("🌾 Choose gluten-free grains: rice, quinoa, buckwheat, oats (certified)")
        
        if "Dairy-Free" in dietary:
            dietary_tips.append("🥛 Use fortified plant-based milks and check labels for hidden dairy")
        
        # Display tips (mix of base and dietary-specific)
        all_tips = base_tips[:4] + dietary_tips[:2]  # Show 4 base + 2 specific tips
        
        for tip in all_tips:
            st.info(tip)
        
        st.subheader("📊 Your Stats")
        bmi = user.get('weight', 70) / ((user.get('height', 170)/100) ** 2)
        
        # Create a nice BMI display
        col_bmi1, col_bmi2 = st.columns(2)
        with col_bmi1:
            st.metric("BMI", f"{bmi:.1f}")
        with col_bmi2:
            if bmi < 18.5:
                st.markdown("⚠️ **Underweight**")
            elif bmi < 25:
                st.markdown("✅ **Normal**")
            else:
                st.markdown("⚠️ **Overweight**")
        
        # BMI Progress bar
        if bmi < 18.5:
            st.progress(bmi/40, f"Underweight: {bmi:.1f}")
        elif bmi < 25:
            st.progress(bmi/30, f"Normal range: {bmi:.1f}")
        else:
            st.progress(min(bmi/40, 1.0), f"Overweight: {bmi:.1f}")
        
        # Personalized advice based on BMI and goal
        if bmi < 18.5:
            if goal == "Weight Loss":
                st.warning("⚠️ **Note:** You're underweight - weight loss isn't recommended. Focus on nutrient-dense foods and strength training.")
            else:
                st.info("🍽️ **Focus on:** Calorie surplus with protein-rich foods and strength training")
        elif bmi < 25:
            if goal == "Weight Loss":
                st.info("🔥 **Approach:** Slight calorie deficit with increased protein to preserve muscle")
            elif goal == "Muscle Gain":
                st.info("💪 **Perfect!:** Moderate calorie surplus with progressive overload training")
            else:
                st.info("✨ **Maintain:** Balanced diet and regular exercise to stay in this zone")
        else:
            if goal == "Muscle Gain":
                st.info("💪 **Body Recomposition:** Build muscle while losing fat with high protein and consistent training")
            else:
                st.info("🔥 **Weight Loss Focus:** Calorie deficit, increased cardio, and portion control")
        
        # Recent workouts
        st.subheader("📋 Recent Activity")
        conn = sqlite3.connect(asset('fitkit.db'))
        c = conn.cursor()
        c.execute('''SELECT exercise_name, duration, posture_accuracy, timestamp 
                     FROM exercise_log 
                     WHERE user_id=? 
                     ORDER BY timestamp DESC LIMIT 3''', (st.session_state.user_id,))
        recent = c.fetchall()
        conn.close()
        
        if recent:
            for ex in recent:
                st.caption(f"• {ex[0]}: {ex[1]}s ({ex[2]:.0f}% accuracy) on {ex[3][:10]}")
        else:
            st.caption("No workouts yet. Start exercising!")

# Main app with enhanced sidebar
def main():
    if st.session_state.logged_in:
        with st.sidebar:
            # Center the logo — safe path resolution
            _base_dir  = os.path.dirname(os.path.abspath(__file__))
            _logo_path = os.path.join(_base_dir, "assets", "fitkit_logo.jpg")
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                try:
                    st.image(_logo_path, use_container_width=True)
                except Exception:
                    st.markdown(
                        "<h2 style='text-align:center; color:#D4AF37;'>💪 FITKIT</h2>",
                        unsafe_allow_html=True
                    )
            
            st.markdown("---")
            
            # Enhanced user greeting
            user_name = st.session_state.user_data.get('name', 'User')
            user_goal = st.session_state.user_data.get('goal', 'Not set')
            
            # Get today's workout status
            conn = sqlite3.connect(asset('fitkit.db'))
            c = conn.cursor()
            c.execute('''SELECT COUNT(*) FROM exercise_log 
                         WHERE user_id=? AND date(timestamp)=date('now')''', 
                      (st.session_state.user_id,))
            today_workouts = c.fetchone()[0] or 0
            conn.close()
            
            workout_status = "✅ Completed" if today_workouts > 0 else "⏳ Pending"
            
            # Stylish user card
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, #667eea, #764ba2); 
                        padding: 15px; border-radius: 15px; color: white; 
                        text-align: center; margin-bottom: 20px;'>
                <h3 style='margin:0; font-size: 22px;'>👋 {user_name}</h3>
                <p style='margin:5px 0; opacity:0.9;'>Goal: {user_goal}</p>

            </div>
            """, unsafe_allow_html=True)
            
            # Navigation with icons - Card style vertical
            st.markdown("### 🧭 Navigation")

            nav_style = """
            <style>
                .nav-button {
                    margin: 8px 0;
                }
                div.stButton > button {
                    text-align: left;
                    padding: 15px 20px;
                    font-size: 18px;
                    font-weight: 500;
                    background: white;
                    border: 1px solid #e0e0e0;
                    border-left: 4px solid #D4AF37;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
                }
                div.stButton > button:hover {
                    background: #f8f8f8;
                    border-left: 4px solid #C41E3A;
                    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                }
            </style>
            """

            st.markdown(nav_style, unsafe_allow_html=True)

            # Simple vertical buttons
            if st.button("📊  Dashboard", use_container_width=True, help="Go to Dashboard"):
                st.session_state.current_page = "Dashboard"
                st.rerun()

            if st.button("🏋️  Exercise", use_container_width=True, help="Go to Exercise Zone"):
                st.session_state.current_page = "Exercise"
                st.rerun()

            if st.button("🥗  Advisory", use_container_width=True, help="Get Diet & Health Advice"):
                st.session_state.current_page = "Advisory"
                st.rerun()

            if st.button("🤖  AI Coach", use_container_width=True, help="Chat with FitKit AI"):
                st.session_state.current_page = "Chatbot"
                st.rerun()
            
            st.markdown("---")
            
 
            
            # Get additional stats
            conn = sqlite3.connect(asset('fitkit.db'))
            c = conn.cursor()
            
            # Get total workouts
            c.execute('''SELECT COUNT(*) FROM exercise_log WHERE user_id=?''', 
                      (st.session_state.user_id,))
            total_workouts = c.fetchone()[0] or 0
            
            # Get average accuracy
            c.execute('''SELECT AVG(posture_accuracy) FROM exercise_log WHERE user_id=?''', 
                      (st.session_state.user_id,))
            avg_accuracy = c.fetchone()[0] or 0
            
            conn.close()
            
            
            st.markdown("""
            <style>
                /* Make logout button text black */
                div.stButton > button:last-of-type {
                    color: black !important;
                    font-weight: 500;
                }
            </style>
            """, unsafe_allow_html=True)

            if st.button("🚪 LOGOUT", use_container_width=True):
                st.session_state.logged_in = False
                st.session_state.user_data = {}
                st.session_state.user_id = None
                st.session_state.current_page = "Login"
                st.rerun()
                
    if not st.session_state.logged_in:
        login_page()
    else:
        if st.session_state.current_page == "Dashboard":
            dashboard_page()
        elif st.session_state.current_page == "Exercise":
            exercise_page()
        elif st.session_state.current_page == "Advisory":
            advisory_page()
        elif st.session_state.current_page == "Chatbot":
            chatbot_page()
        else:
            dashboard_page()

if __name__ == "__main__":
    main()