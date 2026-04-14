import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import shap
import time
from io import BytesIO

# Import custom utilities
import sys
sys.path.append(os.getcwd())
from utils.feature_engineering import apply_feature_engineering
from utils.preprocessing import preprocess_data
from utils.auth import login, sign_up, check_auth, logout

# Page Configuration
st.set_page_config(
    page_title="Student Wellness App",
    page_icon="🌿",
    layout="wide"
)

# Initialize Session State
if 'current_page' not in st.session_state:
    st.session_state['current_page'] = 'login'
if 'user_inputs' not in st.session_state:
    st.session_state['user_inputs'] = {}
if 'reveal_complete' not in st.session_state:
    st.session_state['reveal_complete'] = False

# Custom Styling (Modern Sliders & Pages)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .main {
        background: linear-gradient(135deg, #f0f4f8 0%, #d9e2ec 100%);
    }
    
    /* Modern Slider Styling */
    div[data-baseweb="slider"] {
        padding: 20px 0;
    }
    div[data-baseweb="slider"] > div:first-child {
        background: #3b82f6 !important;
        height: 8px !important;
        border-radius: 12px;
    }
    div[data-testid="stThumbValue"] {
        display: none;
    }
    div[role="slider"] {
        background-color: white !important;
        border: 2px solid #3b82f6 !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1) !important;
        height: 24px !important;
        width: 24px !important;
        transition: transform 0.2s ease;
    }
    div[role="slider"]:hover {
        transform: scale(1.15);
    }

    /* Card Styling with Magic Transitions */
    .dashboard-card {
        background-color: rgba(255, 255, 255, 0.9);
        padding: 30px;
        border-radius: 24px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.05), 0 8px 10px -6px rgba(0, 0, 0, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.6);
        backdrop-filter: blur(20px);
        margin-bottom: 24px;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    }
    .dashboard-card-blue {
        padding-top: 0 !important;
        overflow: hidden;
    }
    .dashboard-card-blue .card-title {
        background: #3b82f6;
        color: white !important;
        padding: 15px 30px;
        margin: 0 -30px 20px -30px;
        font-size: 1.25rem;
    }
    .dashboard-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
    }
    .card-title { color: #1e40af; font-size: 1.2rem; font-weight: 700; margin-bottom: 12px; display: flex; align-items: center; gap: 10px; }
    .card-stat { font-size: 2.8rem; font-weight: 800; color: #3b82f6; line-height: 1; margin: 10px 0; }
    .card-desc { font-size: 0.95rem; color: #475569; }

    /* Buttons */
    .stButton>button {
        border-radius: 16px;
        background: #3b82f6;
        color: white;
        border: none;
        padding: 12px 32px;
        font-weight: 700;
        letter-spacing: 0.5px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .stButton>button:hover {
        transform: translateY(-3px) scale(1.02);
        box-shadow: 0 10px 15px -3px rgba(59, 130, 246, 0.4);
    }
    
    /* Support Message */
    .support-msg {
        text-align: center;
        padding: 40px;
        color: #1e3a8a;
        font-size: 1.2rem;
        font-style: italic;
        opacity: 0.9;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to load models
@st.cache_resource
def load_assets():
    model = joblib.load('models/saved_model.joblib')
    scaler = joblib.load('models/scaler.joblib')
    explainer = joblib.load('models/shap_explainer.joblib')
    return model, scaler, explainer

def dashboard_stat_card(title, value, desc, color="#6366f1"):
    st.markdown(f"""
    <div class="dashboard-card">
        <div class="card-title">{title}</div>
        <div class="card-stat" style="color: {color}">{value}</div>
        <div class="card-desc">{desc}</div>
    </div>
    """, unsafe_allow_html=True)

# --- PAGE 1: LOGIN ---
def show_login_page():
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: center; color: #1e3a8a; font-size: 3rem;'>🌿 Student Wellness</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #64748b;'>Enter your portal to begin your journey ✨</p>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 1.5, 1])
    with col2:
        auth_mode = st.tabs(["Login", "Sign Up"])
        with auth_mode[0]:
            login_user = st.text_input("Username", key="login_user")
            login_pass = st.text_input("Password", type="password", key="login_pass")
            if st.button("Login"):
                success, message = login(login_user, login_pass)
                if success:
                    st.session_state['authenticated'] = True
                    st.session_state['username'] = login_user
                    st.session_state['current_page'] = 'input'
                    st.rerun()
                else:
                    st.error(message)
        with auth_mode[1]:
            new_user = st.text_input("Username", key="new_user")
            new_pass = st.text_input("Password", type="password", key="new_pass")
            confirm_pass = st.text_input("Confirm Password", type="password", key="confirm_pass")
            if st.button("Sign Up"):
                if new_pass != confirm_pass: st.error("Passwords do not match.")
                elif not new_user or not new_pass: st.error("Fields cannot be empty.")
                else:
                    success, message = sign_up(new_user, new_pass)
                    if success: st.success(message)
                    else: st.error(message)

# --- PAGE 2: INPUT ---
def show_input_page():
    st.markdown(f"<h1 style='text-align: center; color: #1e3a8a;'>👋 Hello, {st.session_state.username}</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #64748b;'>How has your week been? Let's check in on your wellness ✨</p>", unsafe_allow_html=True)
    
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            sleep = st.slider("Sleep Hours (Daily)", 3.0, 10.0, 7.0, 0.5)
            study = st.slider("Study Hours (Daily)", 1.0, 12.0, 5.0, 0.5)
            assigns = st.slider("Assignment Load", 1, 15, 5)
            social = st.slider("Social Activity (Hrs/Week)", 0.0, 20.0, 5.0, 1.0)
        with col2:
            screen = st.slider("Screen Time (Hrs/Day)", 1.0, 12.0, 4.0, 0.5)
            physical = st.slider("Physical Activity (Hrs/Week)", 0.0, 10.0, 2.0, 0.5)
            diet = st.selectbox("Diet Quality", ["Poor", "Average", "Good"], index=1)
            gpa = st.slider("Current GPA", 0.0, 10.0, 8.5, 0.1)

        st.markdown("<br>", unsafe_allow_html=True)
        col_btn, _ = st.columns([1, 2])
        with col_btn:
            if st.button("🚀 Analyze Now"):
                with st.spinner("Analyzing your wellness... 🌿"):
                    time.sleep(1.5)
                    st.session_state['user_inputs'] = {
                        'Sleep_Hours': sleep, 'Study_Hours': study, 'Assignment_Load': assigns,
                        'Social_Activity': social, 'Screen_Time': screen, 'Physical_Activity': physical,
                        'Diet_Quality': diet, 'GPA': gpa
                    }
                    st.session_state['current_page'] = 'results'
                    st.session_state['reveal_complete'] = False
                    st.rerun()

    if st.sidebar.button("🚪 Logout"):
        logout()
        st.session_state['current_page'] = 'login'
        st.rerun()

# --- PAGE 3: RESULTS ---
def show_results_page():
    inputs = st.session_state['user_inputs']
    model, scaler, explainer = load_assets()

    # Calculation Logic
    input_df = pd.DataFrame({
        'Sleep_Hours': [inputs['Sleep_Hours']], 'Study_Hours': [inputs['Study_Hours']],
        'Assignment_Load': [inputs['Assignment_Load']], 'Social_Activity': [inputs['Social_Activity']],
        'Screen_Time': [inputs['Screen_Time']], 'Physical_Activity': [inputs['Physical_Activity']],
        'Diet_Quality': [inputs['Diet_Quality']], 'GPA': [inputs['GPA'] / 2.5]
    })
    input_df = apply_feature_engineering(input_df)
    X_processed, _ = preprocess_data(input_df, is_training=False)
    
    prob = model.predict_proba(X_processed)[0]
    pred = np.argmax(prob)
    status_map = {0: 'Good', 1: 'Moderate', 2: 'Poor'}
    health_status = status_map[pred]
    status_color = {"Good": "#10b981", "Moderate": "#f59e0b", "Poor": "#ef4444"}[health_status]
    wellness_score = (prob[0] * 1.0 + prob[1] * 0.5 + prob[2] * 0.0) * 100
    focus_level = max(0, min(100, 100 - (inputs['Screen_Time'] * 5 + (8 - inputs['Sleep_Hours']) * 5)))
    
    base_gpa = inputs['GPA']
    if health_status == 'Good': gpa_range = f"{min(base_gpa + 0.5, 10.0):.2f}"
    elif health_status == 'Moderate': gpa_range = f"{max(base_gpa - 0.25, 0.0):.2f}"
    else: gpa_range = f"{max(base_gpa - 1.0, 0.0):.2f}"

    st.title("🧩 Your Wellness Journey ✨")
    
    # MAGIC REVEAL SEQUENCE
    if not st.session_state.get('reveal_complete', False):
        reveal_delay = 0.6
    else:
        reveal_delay = 0

    # 1. Gauge Section
    fig = go.Figure(go.Indicator(
        mode = "gauge+number", value = wellness_score,
        number = {'suffix': "%", 'font': {'size': 70, 'color': "white"}},
        title = {'text': "Mental Wellness Score", 'font': {'size': 26, 'color': "rgba(255,255,255,0.8)"}},
        gauge = {'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "white"},
                 'bar': {'color': "rgba(0,0,0,0)"}, 'bgcolor': "rgba(0,0,0,0)", 'borderwidth': 0,
                 'steps': [{'range': [0, 33], 'color': '#ef4444'}, {'range': [33, 66], 'color': '#f59e0b'}, {'range': [66, 100], 'color': '#3b82f6'}],
                 'threshold': {'line': {'color': "white", 'width': 4}, 'thickness': 0.8, 'value': wellness_score}}))
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font={'color': "white", 'family': "Inter"}, height=350, margin=dict(l=30, r=30, t=50, b=30))
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    
    if reveal_delay > 0: time.sleep(reveal_delay)

    # 2. Top Row Stats
    m_col = st.empty()
    with m_col.container():
        m1, m2, m3 = st.columns(3)
        with m1: dashboard_stat_card("🧠 Wellness", health_status, "Overall Equilibrium", status_color)
        with m2: dashboard_stat_card("🎯 GPA Est.", f"{gpa_range}", "Predicted Goal", "#6366f1")
        with m3: dashboard_stat_card("📊 Focus", f"{focus_level:.0f}%", "Concentration", "#ec4899")
    
    if reveal_delay > 0: time.sleep(reveal_delay)

    # 3. Model Insights Card
    st.markdown('<div class="dashboard-card dashboard-card-blue">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">🔍 Why this result? (Model Insights)</div>', unsafe_allow_html=True)
    if health_status == 'Good':
        st.write("- **😊 High Recovery**: Your sleep and social balance support strong mental health.")
        st.write("- **😊 Academic Flow**: Your study hours match your workload perfectly.")
    elif health_status == 'Moderate':
        st.write("- **😐 Pressure Check**: High screen time or assignments are starting to create some stress.")
        st.write("- **😐 Sleep Focus**: Irregular rest may be affecting your focus and energy.")
    else:
        st.write("- **😔 Burnout Level**: Your sleep is quite low and screen time is high. Your body may need more rest.")
        st.write("- **😐 Workload Balance**: Your assignments are a bit high. Try managing tasks step by step.")
    st.markdown('</div>', unsafe_allow_html=True)

    if reveal_delay > 0: time.sleep(reveal_delay)

    # 4. Improvement Plan
    st.markdown('<div class="dashboard-card dashboard-card-blue">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">🚀 Growth Roadmap</div>', unsafe_allow_html=True)
    imp_df = pd.DataFrame([
        ["🛌 Sleep", f"{inputs['Sleep_Hours']}h", "7-8h", "• Sleep 30m earlier • No phone in bed"],
        ["📱 Screens", f"{inputs['Screen_Time']}h", "4-6h", "• App time limits • Hourly eyes-off breaks"],
        ["📚 Study", f"{inputs['Study_Hours']}h", "5-7h", "• Pomodoro method • Clean workspace"],
        ["😰 Stress", health_status, "😊 Calm", "• 5m deep breathing • Nature walks"],
        ["🏃 Exercise", f"{inputs['Physical_Activity']}h", "😊 4h+", "• 20m Daily walk • Stretch blocks"]
    ], columns=["Area", "Current", "Target", "How to Bloom 🌿"])
    st.table(imp_df)
    st.markdown('</div>', unsafe_allow_html=True)

    if reveal_delay > 0: time.sleep(reveal_delay)

    # 5. Detail Row (Roadmap & Counsel)
    c1, c2 = st.columns([1.2, 1])
    with c1:
        st.markdown('<div class="dashboard-card"><div class="card-title">📅 Daily Harmony</div>', unsafe_allow_html=True)
        st.table(pd.DataFrame({"Time": ["06:00", "07:30", "13:00", "22:00"], "Moment": ["Wake up 🌿", "Deep Study 📚", "Mindful Lunch 🍎", "Deep Rest 😴"]}))
        st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="dashboard-card"><div class="card-title">🌼 Wellness Corner</div>', unsafe_allow_html=True)
        if health_status == 'Poor': st.write("😔 It's okay to feel overwhelmed. You are doing enough. Try to breathe and take things one step at a time.")
        else: st.write("😊 You've found a wonderful rhythm! We are so proud of your balance.")
        st.write("✨ Remember, you are growing every day.")
        st.markdown('</div>', unsafe_allow_html=True)

    if reveal_delay > 0: time.sleep(reveal_delay)

    # 6. Appointment Section
    st.markdown('<div class="dashboard-card dashboard-card-blue">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">🏥 Book Appointment</div>', unsafe_allow_html=True)
    st.write("If you feel you need help, you can consult a professional.")
    
    b1, b2 = st.columns(2)
    with b1:
        st.markdown('<div class="dashboard-card dashboard-card-blue" style="margin-bottom:0;">', unsafe_allow_html=True)
        st.markdown('''
        <div class="card-title" style="background: white; color: #1e40af; flex-direction: column; text-align: center; justify-content: center; height: auto; border-bottom: 2px solid #3b82f6; padding: 20px;">
            <div style="font-weight: 800; font-size: 1.25rem;">🧠 Psychologist</div>
            <div style="font-weight: 400; font-size: 0.9rem; margin-top: 8px; color: #475569;">Professional online counseling for mental health balance.</div>
        </div>
        ''', unsafe_allow_html=True)
        st.markdown('<div style="text-align: center; padding-top: 20px;">', unsafe_allow_html=True)
        st.link_button("🌸 Book Now", "https://www.betterhelp.com")
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with b2:
        st.markdown('<div class="dashboard-card dashboard-card-blue" style="margin-bottom:0;">', unsafe_allow_html=True)
        st.markdown('''
        <div class="card-title" style="background: white; color: #1e40af; flex-direction: column; text-align: center; justify-content: center; height: auto; border-bottom: 2px solid #3b82f6; padding: 20px;">
            <div style="font-weight: 800; font-size: 1.25rem;">🏥 Nearby Hospital</div>
            <div style="font-weight: 400; font-size: 0.9rem; margin-top: 8px; color: #475569;">Find medical support at a health center near you.</div>
        </div>
        ''', unsafe_allow_html=True)
        st.markdown('<div style="text-align: center; padding-top: 20px;">', unsafe_allow_html=True)
        st.link_button("🚀 Book Now", "https://www.google.com/maps/search/hospital+near+me")
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # SEARCH SECTION (Learn More)
    st.markdown("---")
    with st.container():
        st.markdown('<div class="dashboard-card dashboard-card-blue">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">🔍 Search Health Topics</div>', unsafe_allow_html=True)
        
        # Suggested Topics Logic
        if 'search_query' not in st.session_state:
            st.session_state['search_query'] = ""
            
        st.write("**Suggested Topics:**")
        c1, c2 = st.columns(2)
        with c1:
            st.write("🧠 *Mental Health:*")
            m_cols = st.columns(2)
            if m_cols[0].button("Stress", key="btn_stress"): st.session_state['search_query'] = "Stress"; st.rerun()
            if m_cols[1].button("Anxiety", key="btn_anxiety"): st.session_state['search_query'] = "Anxiety"; st.rerun()
            if m_cols[0].button("Focus", key="btn_focus"): st.session_state['search_query'] = "Focus"; st.rerun()
            if m_cols[1].button("Burnout", key="btn_burnout"): st.session_state['search_query'] = "Burnout"; st.rerun()
        with c2:
            st.write("💪 *Physical Health:*")
            p_cols = st.columns(2)
            if p_cols[0].button("Sleep", key="btn_sleep"): st.session_state['search_query'] = "Sleep"; st.rerun()
            if p_cols[1].button("Diet", key="btn_diet"): st.session_state['search_query'] = "Diet & Nutrition"; st.rerun()
            if p_cols[0].button("Exercise", key="btn_exercise"): st.session_state['search_query'] = "Exercise"; st.rerun()
            if p_cols[1].button("Fitness", key="btn_fitness"): st.session_state['search_query'] = "Exercise"; st.rerun() # Mapping fitness to exercise for data
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Modern Search Input
        q = st.text_input("", value=st.session_state['search_query'], placeholder="Search stress, sleep, focus, diet, exercise...", key="health_search_final")
        
        # Knowledge Base (Expanded for Topics and Questions)
        kb = {
            "Stress": {
                "ans": "Stress is your body's response to pressure or too many tasks. It can affect your focus, mood, and energy levels.",
                "do": ["Take short breaks between study sessions", "Practice deep breathing for a few minutes", "Reduce unnecessary screen time"],
                "diet": ["Eat Omega-3 rich foods like walnuts", "Drink chamomile tea to relax"],
                "habit": ["Sleep at least 7 hours nightly", "Spend some time relaxing or walking"],
                "help": "If stress continues for a long time, consider consulting a professional."
            },
            "Anxiety": {
                "ans": "Anxiety is a feeling of worry or unease about the future. It's common for students but can be managed with simple steps.",
                "do": ["Focus on things you can control right now", "Try grounding yourself (5-4-3-2-1 technique)", "Talk to a friend or mentor"],
                "diet": ["Avoid processed sugars and high caffeine", "Eat magnesium-rich leafy greens"],
                "habit": ["Limit caffeine and energy drinks", "Practice daily mindfulness"],
                "help": "If anxiety prevents you from doing basic tasks, consider consulting a professional."
            },
            "Focus": {
                "ans": "Focus is your ability to concentrate on one task. It's normal for it to dip when you're tired or distracted.",
                "do": ["Use the Pomodoro technique (25min work, 5min break)", "Put your phone in another room while studying", "Keep your workspace clean and organized"],
                "diet": ["Eat dark chocolate (in moderation) for alertness", "Snack on blueberries for brain health"],
                "habit": ["Plan your day the night before", "Stay hydrated with plenty of water"],
                "help": "If you struggle to focus despite these steps, consider consulting a professional."
            },
            "Burnout": {
                "ans": "Burnout happens when you've been pushed too hard for too long. It feels like complete exhaustion or lack of interest.",
                "do": ["Take a full day off from studying", "Lower your expectations for a few days", "Ask for help with your assignments"],
                "diet": ["Eat protein-rich snacks for stable energy", "Prioritize vitamin C (citrus fruits) for immunity"],
                "habit": ["Set a strict 'stop work' time every evening", "Spend time on a non-academic hobby"],
                "help": "If you feel hopeless or completely detached, please consult a professional immediately."
            },
            "Sleep": {
                "ans": "Sleep is when your brain recharges and organises your memories. Good sleep is the secret to getting better grades.",
                "do": ["Go to bed at the same time every night", "Avoid your phone 30 minutes before sleep", "Make your room as dark and quiet as possible"],
                "diet": ["Eat bananas (rich in potassium/magnesium)", "Have a warm glass of milk before bed"],
                "habit": ["Get 10 minutes of sunlight in the morning", "Avoid large meals late at night"],
                "help": "If you have trouble sleeping for more than two weeks, consider consulting a professional."
            },
            "Diet": {
                "ans": "The food you eat is fuel for your brain. Eating well keeps your energy stable so you don't crash while studying.",
                "do": ["Eat more fruits and leafy greens", "Carry a water bottle and sip throughout the day", "Choose nuts or seeds over sugary snacks"],
                "diet": ["Focus on whole grains for sustained energy", "Limit processed fast foods"],
                "habit": ["Don't skip breakfast", "Limit sugary drinks and fast food"],
                "help": "If you have extreme fatigue or sudden weight changes, consider consulting a professional."
            },
            "Exercise": {
                "ans": "Moving your body releases 'happy chemicals' and clears your mind. It's the fastest way to improve your mood.",
                "do": ["Go for a 20-minute walk outside", "Try light stretching at your desk hourly", "Join a sports club or gym"],
                "diet": ["Hydrate with electrolytes after workouts", "Eat light protein like eggs or yogurt"],
                "habit": ["Take the stairs instead of the elevator", "Walk while listening to your lectures"],
                "help": "If you have persistent pain or dizziness, consider consulting a professional."
            },
            "Migraine": {
                "ans": "Migraine is a type of headache that causes strong pain, often on one side of the head. It may also cause nausea and light sensitivity.",
                "do": ["Rest in a quiet and dark room", "Apply a cold compress to your forehead or neck", "Stay hydrated with water or electrolytes"],
                "diet": ["Eat ginger to help with nausea", "Avoid aged cheeses or chocolate (common triggers)"],
                "habit": ["Maintain regular sleep patterns", "Avoid skipping meals"],
                "help": "If migraines happen often or change intensity, consult a doctor."
            },
            "Headache": {
                "ans": "Headaches can be caused by stress, dehydration, or eye strain. They are common but usually easy to treat.",
                "do": ["Drink a large glass of water", "Massage your temples gently", "Take a short screen break"],
                "diet": ["Snack on almonds (magnesium helps relaxation)", "Drink ginger and honey water"],
                "habit": ["Check your desk posture", "Stay hydrated throughout the day"],
                "help": "If you have a very sudden or severe headache, consult a professional immediately."
            },
            "Social Anxiety": {
                "ans": "Social anxiety is a fear of being judged by others in social situations. It's common in university settings.",
                "do": ["Practice small interactions first", "Prepare a few conversation topics in advance", "Focus on the person you are talking to, not yourself"],
                "diet": ["Avoid alcohol (it increases anxiety later)", "Snack on pumpkin seeds to lower cortisol"],
                "habit": ["Challenge one small social fear weekly", "Practice slow breathing before social events"],
                "help": "If social fear stops you from attending classes or meetings, consult a counselor."
            },
            "Hydration": {
                "ans": "Water is essential for every function in your body, especially brain focus and physical energy.",
                "do": ["Drink water as soon as you wake up", "Carry a reusable water bottle everywhere", "Set reminders to drink water every hour"],
                "diet": ["Eat hydrating foods like cucumber/watermelon", "Avoid too much caffeine which dehydrates"],
                "habit": ["Drink a glass of water before every meal", "Choose water over soda"],
                "help": "If you feel thirsty and dizzy even after drinking water, consult a professional."
            },
            "Back Pain": {
                "ans": "Back pain for students is often caused by long hours of sitting with poor posture.",
                "do": ["Adjust your chair so your feet are flat", "Do 5 minutes of back stretches daily", "Get up and move every 30 minutes"],
                "diet": ["Eat anti-inflammatory turmeric/ginger", "Ensure enough calcium/vitamin D intake"],
                "habit": ["Check your posture every time you sit", "Use a laptop stand to keep your screen at eye level"],
                "help": "If back pain is sharp or travels down your legs, consult a professional."
            }
        }
        
        # Search Trigger
        if st.button("🔍 Search"):
            if q and q != "":
                found = False
                query = q.lower().strip()
                # Check for keyword matches in the query
                for key, content in kb.items():
                    if key.lower() in query:
                        st.markdown(f"### Answer for: {key}")
                        st.write(content['ans'])
                        st.write("**What you can do:**")
                        for item in content['do']:
                            st.write(f"- {item}")
                        
                        st.write("**🥗 Diet Plan:**")
                        for item in content['diet']:
                            st.write(f"- {item}")

                        st.write("**Small daily habits:**")
                        for item in content['habit']:
                            st.write(f"- {item}")
                        if 'help' in content:
                            st.write(content['help'])
                        found = True
                        break
                
                if not found:
                    st.write("😐 I couldn't find a specific answer for that. Try searching for topics like 'Stress', 'Sleep', or 'Migraine'.")
            else:
                st.write("Please type something to search!")
        st.markdown('</div>', unsafe_allow_html=True)

    # FINAL TOUCH
    st.markdown("<div class='support-msg'>✨ You are on a journey of growth. Small steps today will create a better tomorrow.</div>", unsafe_allow_html=True)
    
    st.session_state['reveal_complete'] = True

    if st.button("⬅️ Edit Inputs"):
        st.session_state['current_page'] = 'input'
        st.rerun()

# --- MAIN ROUTER ---
if not check_auth():
    show_login_page()
else:
    if st.session_state['current_page'] == 'input': show_input_page()
    elif st.session_state['current_page'] == 'results': show_results_page()
    else: show_input_page()
