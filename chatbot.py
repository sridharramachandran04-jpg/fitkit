import streamlit as st
import sqlite3
import os
import time
import google.generativeai as genai   # old SDK — used consistently throughout this file

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "fitkit.db")
MODEL_CANDIDATES = [
    "gemini-flash-latest",
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-2.5-flash-lite",
    "gemini-pro-latest",
]
MODEL = MODEL_CANDIDATES[0]  # kept for reference/back-compat; call_gemini() actually loops through MODEL_CANDIDATES

SYSTEM_PROMPT = """You are FitKit AI — an expert gym fitness coach and certified nutritionist built into the FITKIT app.

You specialize in:
- Gym exercises: form, technique, sets, reps, rest, progressive overload.
- Diet & nutrition: calories, macros, meal plans, pre/post workout nutrition.
- Workout plans: PPL, upper/lower, full body, bro split, periodization.
- Muscle groups, recovery, injury prevention, and progress tracking.

Rules:
- Give specific, actionable advice.
- Use numbers when relevant.
- Keep answers practical and under 300 words unless a detailed plan is requested.
- Do not give medical diagnoses, unsafe advice, or illegal/PED advice.
"""

QUICK_PROMPTS = {
    "Workout Plans": [
        "Create a 5-day gym workout plan for muscle gain",
        "What is PPL split and give me a full week plan",
        "Best beginner gym routine for weight loss",
        "How should I structure my workouts for body recomposition?",
    ],
    "Diet & Nutrition": [
        "Calculate my daily calories and macros for weight loss",
        "What should I eat before and after a workout?",
        "High protein vegetarian meal plan for muscle gain",
        "Best foods to eat for fat loss while preserving muscle",
    ],
    "Exercise Form": [
        "Teach me proper deadlift form step by step",
        "How do I fix knee cave in squats?",
        "What muscles does the lateral raise work?",
        "Difference between Romanian deadlift and conventional deadlift",
    ],
    "Progress & Recovery": [
        "Why am I not seeing progress after 3 months?",
        "How many rest days do I need per week?",
        "How to break through a strength plateau?",
        "Best recovery practices after intense leg day",
    ],
}


def list_available_models():
    """Debug helper: run this manually (e.g. in a Python shell) to see which
    models your API key actually has access to, since Google periodically
    retires model names without much warning.

        import chatbot
        chatbot.load_api_key_and_configure()
        chatbot.list_available_models()
    """
    for m in genai.list_models():
        if "generateContent" in m.supported_generation_methods:
            print(m.name)


def load_api_key_and_configure():
    key = load_api_key()
    genai.configure(api_key=key)
    return key


def load_api_key():
    key = ""
    try:
        key = st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        key = ""
    key = str(key).strip()
    if not key:
        key = os.environ.get("GEMINI_API_KEY", "")
    return str(key).strip()


def get_user_context(user_data):
    if not user_data:
        return ""
    try:
        h = float(user_data.get("height", 0) or 0)
        w = float(user_data.get("weight", 0) or 0)
        bmi = round(w / ((h / 100) ** 2), 1) if h > 0 and w > 0 else "?"
        dietary = user_data.get("dietary", [])
        if isinstance(dietary, str):
            dietary = [dietary]
        return (
            f"User profile: {user_data.get('name','?')}, Age {user_data.get('age','?')}, "
            f"{user_data.get('gender','?')}, Height {h}cm, Weight {w}kg, BMI {bmi}, "
            f"Goal: {user_data.get('goal','?')}, Activity: {user_data.get('activity','?')}, "
            f"Dietary prefs: {', '.join(dietary) if dietary else 'None specified'}."
        )
    except Exception:
        return ""


def get_recent_exercises(user_id):
    if not user_id:
        return ""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute(
            """SELECT exercise_name, duration, posture_accuracy, timestamp
               FROM exercise_log WHERE user_id=?
               ORDER BY timestamp DESC LIMIT 5""",
            (user_id,),
        )
        rows = c.fetchall()
        conn.close()
        if not rows:
            return "No recent workouts logged."
        lines = [f"- {r[0]}: {r[1]}s, {r[2]:.0f}% accuracy ({str(r[3])[:10]})" for r in rows]
        return "Recent workouts:\n" + "\n".join(lines)
    except Exception:
        return ""


def build_system_content(user_data, user_id):
    content = SYSTEM_PROMPT
    ctx = get_user_context(user_data)
    recent = get_recent_exercises(user_id)
    if ctx:
        content += f"\n\n{ctx}"
    if recent:
        content += f"\n\n{recent}"
    return content


def call_gemini(messages, user_data, user_id):
    api_key = load_api_key()
    if not api_key or len(api_key) < 20 or "YOUR_GEMINI_API_KEY_HERE" in api_key:
        return (
            "⚠️ API key not loaded.\n\n"
            "Create `.streamlit/secrets.toml` with:\n"
            "GEMINI_API_KEY = \"your-real-key\"\n\n"
            "Then restart Streamlit."
        )

    genai.configure(api_key=api_key)
    system_content = build_system_content(user_data, user_id)

    chat_text = ""
    for role_name, content in messages:
        prefix = "User" if role_name == "user" else "Assistant"
        chat_text += f"{prefix}: {content}\n"

    prompt = f"{system_content}\n\nConversation:\n{chat_text}\nAssistant:"

    last_error = None
    for candidate in MODEL_CANDIDATES:
        try:
            model = genai.GenerativeModel(candidate)
            response = model.generate_content(prompt)
            return response.text or "⚠️ Gemini returned no text."
        except Exception as e:
            msg = str(e)
            last_error = msg
            if "429" in msg or "Too Many Requests" in msg:
                return "⚠️ Quota exceeded. Wait 60 seconds and try again."
            if "403" in msg:
                return "⚠️ Invalid API key or API access issue."
            if "404" in msg or "not found" in msg.lower():
                # this model name isn't available for this key — try the next candidate
                continue
            return f"⚠️ Error: {msg}"

    return (
        f"⚠️ None of the tried models ({', '.join(MODEL_CANDIDATES)}) are available "
        f"for your API key. Last error: {last_error}\n\n"
        "Run list_available_models() to see exactly which models your key supports, "
        "then add the exact name to MODEL_CANDIDATES."
    )


@st.cache_data(show_spinner=False)
def cached_gemini_response(messages, user_data_tuple, user_id):
    user_data = dict(user_data_tuple)
    return call_gemini(list(messages), user_data, user_id)


def chatbot_page():
    st.title("🤖 FitKit AI Coach")
    st.markdown("Ask me anything about **gym exercises**, **diet plans**, **workout programming**, or **nutrition**.")

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    if "last_api_call" not in st.session_state:
        st.session_state.last_api_call = 0.0

    with st.sidebar:
        st.markdown("### 💡 Quick Questions")
        for category, prompts in QUICK_PROMPTS.items():
            with st.expander(category):
                for idx, prompt in enumerate(prompts):
                    if st.button(prompt, key=f"{category}_{idx}", use_container_width=True):
                        st.session_state.chat_messages.append({"role": "user", "content": prompt})
                        st.rerun()

        st.markdown("---")
        st.caption("🔑 Powered by Google Gen AI SDK")
        st.markdown("---")

        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_messages = []
            st.session_state.last_api_call = 0.0
            st.cache_data.clear()
            st.rerun()

        user_data = st.session_state.get("user_data", {})
        if user_data:
            st.markdown("---")
            st.markdown("### 👤 Your Profile")
            st.caption(f"Goal: **{user_data.get('goal','?')}**")
            st.caption(f"Activity: **{user_data.get('activity','?')}**")
            h = user_data.get("height", 0)
            w = user_data.get("weight", 0)
            if h and w:
                try:
                    bmi = round(float(w) / ((float(h) / 100) ** 2), 1)
                    st.caption(f"BMI: **{bmi}**")
                except Exception:
                    pass

    if not st.session_state.chat_messages:
        st.markdown("""
<div style='background: linear-gradient(135deg, #2C3E50, #1A1A1A);
    padding: 25px; border-radius: 15px; color: white;
    border: 1px solid #D4AF37; margin-bottom: 20px;'>
    <h3 style='color: #D4AF37; margin-top:0;'>👋 Hey! I'm your FitKit AI Coach</h3>
    <p style='margin:0; color: #E0E0E0;'>
        I can help you with personalized workout plans, diet advice, exercise technique,
        and anything gym-related. Try a quick question or type below.
    </p>
</div>
        """, unsafe_allow_html=True)
    else:
        for msg in st.session_state.chat_messages:
            with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "🤖"):
                st.markdown(msg["content"])

    # ✅ AUTO-RESPOND when quick prompt adds a user message
    if (
        st.session_state.chat_messages
        and st.session_state.chat_messages[-1]["role"] == "user"
    ):
        if len(st.session_state.chat_messages) == 1 or \
           st.session_state.chat_messages[-2]["role"] == "assistant":

            with st.chat_message("assistant", avatar="🤖"):
                with st.spinner("FitKit AI is thinking..."):
                    api_msgs = [
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state.chat_messages[-20:]
                    ]
                    user_data = st.session_state.get("user_data", {})
                    user_data_tuple = tuple(sorted((k, str(v)) for k, v in user_data.items()))
                    reply = cached_gemini_response(
                        tuple((m["role"], m["content"]) for m in api_msgs),
                        user_data_tuple,
                        st.session_state.get("user_id"),
                    )
                    st.markdown(reply)

            st.session_state.chat_messages.append({"role": "assistant", "content": reply})
            st.rerun()

    user_input = st.chat_input("Ask about workouts, diet, exercises, form tips...")
    if user_input:
        st.session_state.chat_messages.append({"role": "user", "content": user_input})
        with st.chat_message("user", avatar="🧑"):
            st.markdown(user_input)

        now = time.time()
        if now - st.session_state.last_api_call < 8:
            reply = "⚠️ Please wait a few seconds before sending another message."
        else:
            st.session_state.last_api_call = now
            with st.chat_message("assistant", avatar="🤖"):
                with st.spinner("FitKit AI is thinking..."):
                    api_msgs = [
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state.chat_messages[-20:]
                    ]
                    user_data = st.session_state.get("user_data", {})
                    user_data_tuple = tuple(sorted((k, str(v)) for k, v in user_data.items()))
                    reply = cached_gemini_response(
                        tuple((m["role"], m["content"]) for m in api_msgs),
                        user_data_tuple,
                        st.session_state.get("user_id"),
                    )
                    st.markdown(reply)

        st.session_state.chat_messages.append({"role": "assistant", "content": reply})
        st.rerun()