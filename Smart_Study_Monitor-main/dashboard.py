import streamlit as st
import cv2
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector
from ultralytics import YOLO
import ctypes
import os
import time
import pandas as pd
import matplotlib.pyplot as plt
import database

# Ensure DB is initialized
database.init_db()

# Set page config
st.set_page_config(
    page_title="Smart Study Monitor", 
    page_icon="🎓", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling Injection (Premium Dark Theme UI/UX)
st.markdown(
    """
    <style>
    /* Global Font Style */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"], .stApp {
        font-family: 'Poppins', sans-serif !important;
        background-color: #0b0f19 !important;
        color: #f1f5f9 !important;
    }
    
    /* Clean Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0f172a !important;
        border-right: 1px solid #1e293b;
    }
    
    /* Rounded & Modern WebCam Image Container */
    .stImage img {
        border-radius: 16px;
        border: 4px solid #1e293b;
        box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
        transition: border-color 0.3s ease;
    }
    
    /* Custom Styled Buttons */
    .stButton>button {
        border-radius: 12px !important;
        font-weight: 600 !important;
        padding: 0.6rem 1.2rem !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        transition: all 0.2s ease-in-out !important;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
    }
    
    /* Tabs Navigation bar design */
    div[data-testid="stTabBar"] {
        background: #0f172a;
        border-radius: 16px;
        padding: 6px;
        border: 1px solid #1e293b;
        margin-bottom: 25px;
    }
    button[data-baseweb="tab"] {
        border-radius: 12px !important;
        padding: 10px 24px !important;
        color: #94a3b8 !important;
        font-weight: 500 !important;
        border: none !important;
    }
    button[aria-selected="true"] {
        background-color: #2563eb !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3);
    }
    
    /* Metric Cards Grid layout styling */
    .metric-card-wrapper {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 16px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card-wrapper:hover {
        transform: translateY(-4px);
        border-color: #3b82f6;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Database Query Helper
DB_NAME = "study_monitor.db"
db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), DB_NAME)

def get_data(query):
    if not os.path.exists(db_path):
        return pd.DataFrame()
    try:
        conn = database.get_db_connection()
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    except Exception as e:
        return pd.DataFrame()

# Windows Short Path Helper to bypass folder space issues in Windows
def get_short_path(filepath):
    try:
        buf = ctypes.create_unicode_buffer(512)
        ctypes.windll.kernel32.GetShortPathNameW(filepath, buf, 512)
        return buf.value
    except Exception:
        return filepath

# Audio MCI Helper Functions (Uses Windows native audio system)
def play_mci_sound(alias, filepath):
    # Ensure any previous instance of this alias is closed first
    ctypes.windll.winmm.mciSendStringW(f'close {alias}', None, 0, 0)
    cmd = f'open "{filepath}" type mpegvideo alias {alias}'
    res = ctypes.windll.winmm.mciSendStringW(cmd, None, 0, 0)
    if res == 0:
        ctypes.windll.winmm.mciSendStringW(f'play {alias}', None, 0, 0)
        return True
    return False

def stop_mci_sound(alias):
    ctypes.windll.winmm.mciSendStringW(f'stop {alias}', None, 0, 0)
    ctypes.windll.winmm.mciSendStringW(f'close {alias}', None, 0, 0)

def stop_all_sounds():
    stop_mci_sound("alarm_sleep")
    stop_mci_sound("alarm_facehide")
    stop_mci_sound("alarm_phone")

# Initialize session state variables robustly
if 'session_active' not in st.session_state:
    st.session_state.session_active = False
if 'session_paused' not in st.session_state:
    st.session_state.session_paused = False
if 'session_id' not in st.session_state:
    st.session_state.session_id = None
if 'cap' not in st.session_state:
    st.session_state.cap = None
if 'face_detector' not in st.session_state:
    st.session_state.face_detector = None
if 'phone_detector' not in st.session_state:
    st.session_state.phone_detector = None

if 'total_sec' not in st.session_state:
    st.session_state.total_sec = 0
if 'focus_sec' not in st.session_state:
    st.session_state.focus_sec = 0
if 'sleep_count' not in st.session_state:
    st.session_state.sleep_count = 0
if 'phone_count' not in st.session_state:
    st.session_state.phone_count = 0
if 'cover_count' not in st.session_state:
    st.session_state.cover_count = 0
if 'was_sleepy' not in st.session_state:
    st.session_state.was_sleepy = False
if 'was_phone' not in st.session_state:
    st.session_state.was_phone = False
if 'was_covered' not in st.session_state:
    st.session_state.was_covered = False
if 'current_playing' not in st.session_state:
    st.session_state.current_playing = None
if 'last_time' not in st.session_state:
    st.session_state.last_time = time.time()
if 'covered_frames' not in st.session_state:
    st.session_state.covered_frames = 0
if 'closed_frames' not in st.session_state:
    st.session_state.closed_frames = 0
if 'last_played_time' not in st.session_state:
    st.session_state.last_played_time = 0

# Ensure webcam releases if session is inactive
if not st.session_state.session_active:
    if st.session_state.cap is not None:
        st.session_state.cap.release()
        st.session_state.cap = None

# Audio file paths (converted to 8.3 short paths)
base_dir = os.path.dirname(os.path.abspath(__file__))
path_sleep = get_short_path(os.path.join(base_dir, "alarm.mp3"))
path_facehide = get_short_path(os.path.join(base_dir, "faudio.mp3"))
path_phone = get_short_path(os.path.join(base_dir, "paudio.mp3"))

# ----------------- SIDEBAR SETTINGS -----------------
st.sidebar.markdown(
    """
    <div style="text-align: center; margin-bottom: 25px;">
        <h2 style="color: #3b82f6; font-weight: 700; margin-bottom: 5px;">STUDY CO-PILOT</h2>
        <span style="color: #64748b; font-size: 0.85rem;">Smart Distraction Monitoring</span>
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.subheader("Alarm Swaps 🚨")
mute_audio = st.sidebar.checkbox("Mute Audio Alarms", value=False)
enable_sleep = st.sidebar.checkbox("Enable Sleep Alarm", value=True)
enable_phone = st.sidebar.checkbox("Enable Phone Alarm", value=True)
enable_facehide = st.sidebar.checkbox("Enable Cover Alarm", value=True)

st.sidebar.markdown("---")
st.sidebar.subheader("Sensitivity Controls ⚙️")

eye_ratio_threshold = st.sidebar.slider("Eye Aspect Threshold", min_value=5.0, max_value=20.0, value=11.0, step=0.5)
sleep_threshold_frames = st.sidebar.slider("Sleep Trigger (Frames)", min_value=5, max_value=50, value=15)
phone_conf_threshold = st.sidebar.slider("Phone Detection Thresh (%)", min_value=30, max_value=90, value=50) / 100.0

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    <div style="text-align: center; font-size: 0.8rem; color: #475569;">
        Powered by YOLOv8 & MediaPipe Mesh
    </div>
    """,
    unsafe_allow_html=True
)

# ----------------- MAIN INTERFACE -----------------
st.markdown(
    """
    <div style="display: flex; align-items: center; gap: 15px; margin-bottom: 10px;">
        <div style="background-color: #2563eb; padding: 12px; border-radius: 12px; font-size: 1.8rem; line-height: 1;">🎓</div>
        <div>
            <h1 style="margin: 0; font-size: 2.2rem; font-weight: 700; background: linear-gradient(to right, #3b82f6, #60a5fa); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">Smart Study Monitor</h1>
            <p style="margin: 0; color: #94a3b8; font-size: 1rem;">Track study sessions, boost concentration metrics, and minimize phone distractions.</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

tabs = st.tabs(["🎥 Live Monitor", "📊 Focus Analytics", "📋 Session History"])

# ----- TAB 1: LIVE MONITOR -----
with tabs[0]:
    # Session control action bar
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
    
    with col_btn1:
        if not st.session_state.session_active:
            if st.button("▶️ Start New Session", type="primary", use_container_width=True):
                st.session_state.session_id = database.start_session()
                st.session_state.session_active = True
                st.session_state.session_paused = False
                st.session_state.total_sec = 0
                st.session_state.focus_sec = 0
                st.session_state.sleep_count = 0
                st.session_state.phone_count = 0
                st.session_state.cover_count = 0
                st.session_state.was_sleepy = False
                st.session_state.was_phone = False
                st.session_state.was_covered = False
                st.session_state.current_playing = None
                st.session_state.last_time = time.time()
                st.session_state.covered_frames = 0
                st.session_state.closed_frames = 0
                st.session_state.last_played_time = 0
                
                # Pre-initialize detectors once to avoid slow start on rerun
                if st.session_state.face_detector is None:
                    st.session_state.face_detector = FaceMeshDetector(maxFaces=1)
                if st.session_state.phone_detector is None:
                    st.session_state.phone_detector = YOLO("yolov8n.pt")
                
                st.rerun()
        else:
            if st.button("⏹️ Stop & Save Session", type="primary", use_container_width=True):
                # Release camera
                if st.session_state.cap is not None:
                    st.session_state.cap.release()
                    st.session_state.cap = None
                stop_all_sounds()
                database.end_session(st.session_state.session_id, st.session_state.total_sec, st.session_state.focus_sec)
                st.session_state.session_active = False
                st.session_state.session_id = None
                st.toast("Session completed and saved! Check the Analytics tab.", icon="💾")
                st.rerun()
                
    with col_btn2:
        if st.session_state.session_active:
            if not st.session_state.session_paused:
                if st.button("⏸️ Pause Session", use_container_width=True):
                    st.session_state.session_paused = True
                    stop_all_sounds()
                    st.rerun()
            else:
                if st.button("▶️ Resume Session", use_container_width=True):
                    st.session_state.session_paused = False
                    st.session_state.last_time = time.time()
                    st.rerun()
                    
    with col_btn3:
        # Status indicators
        if st.session_state.session_active:
            if st.session_state.session_paused:
                st.markdown(
                    """
                    <div style="background-color: rgba(234, 179, 8, 0.15); border: 1px solid #eab308; color: #facc15; border-radius: 12px; padding: 10px 15px; font-weight: 600; text-align: center; font-size: 0.95rem;">
                        Status: Paused ⏸️
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    """
                    <div style="background-color: rgba(34, 197, 94, 0.15); border: 1px solid #22c55e; color: #4ade80; border-radius: 12px; padding: 10px 15px; font-weight: 600; text-align: center; font-size: 0.95rem;">
                        Status: Monitoring Live 🟢
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        else:
            st.markdown(
                """
                <div style="background-color: rgba(148, 163, 184, 0.15); border: 1px solid #64748b; color: #94a3b8; border-radius: 12px; padding: 10px 15px; font-weight: 600; text-align: center; font-size: 0.95rem;">
                    Status: Session Inactive ⚪
                </div>
                """,
                unsafe_allow_html=True
            )
            
    st.markdown("---")

    if st.session_state.session_active and not st.session_state.session_paused:
        # Pre-initialize detectors dynamically if they are None (due to session resets)
        if st.session_state.face_detector is None:
            st.session_state.face_detector = FaceMeshDetector(maxFaces=1)
        if st.session_state.phone_detector is None:
            st.session_state.phone_detector = YOLO("yolov8n.pt")
            
        # Initialize camera
        if st.session_state.cap is None:
            st.session_state.cap = cv2.VideoCapture(0)
            st.session_state.cap.read()
            
        # Metric placeholder and video feed container
        metric_ph = st.empty()
        video_feed_ph = st.empty()
        
        # Run a 30-frame smooth, flicker-free rendering loop within the single run
        for frame_idx in range(30):
            # Check if user paused or stopped the session in between
            if not st.session_state.session_active or st.session_state.session_paused:
                break
                
            # Read frame
            success, img = st.session_state.cap.read()
            if not success:
                time.sleep(0.03)
                continue
                
            img = cv2.flip(img, 1)
            
            # 1. Face Mesh Sleep & Cover detection
            img, faces = st.session_state.face_detector.findFaceMesh(img, draw=False)
            is_sleepy = False
            is_face_covered = False
            
            LEFT_EYE_TOP = 159
            LEFT_EYE_BOTTOM = 145
            FACE_LEFT = 130
            FACE_RIGHT = 243
            
            if faces:
                st.session_state.covered_frames = 0
                face = faces[0]
                try:
                    # Draw a nice clean face mesh outline on screen (makes it look "wow")
                    eye_dist, _ = st.session_state.face_detector.findDistance(face[LEFT_EYE_TOP], face[LEFT_EYE_BOTTOM])
                    face_dist, _ = st.session_state.face_detector.findDistance(face[FACE_LEFT], face[FACE_RIGHT])
                    ratio = (eye_dist / face_dist) * 100
                except Exception:
                    ratio = 15.0
                    
                if ratio < eye_ratio_threshold:
                    st.session_state.closed_frames += 1
                else:
                    st.session_state.closed_frames = 0
                    
                if st.session_state.closed_frames >= sleep_threshold_frames:
                    is_sleepy = True
                    
                # Visual feed decoration: Draw simple mesh dots around eyes
                cv2.circle(img, face[LEFT_EYE_TOP], 3, (34, 197, 94), -1)      # Green dot
                cv2.circle(img, face[LEFT_EYE_BOTTOM], 3, (34, 197, 94), -1)   # Green dot
                cvzone.putTextRect(img, f"Eye Aspect Ratio: {int(ratio)}", (30, 40), scale=1, thickness=1, colorR=(15, 23, 42))
            else:
                st.session_state.closed_frames = 0
                st.session_state.covered_frames += 1
                if st.session_state.covered_frames >= 20:
                    is_face_covered = True
                    
            # 2. YOLO Phone Detection
            results = st.session_state.phone_detector.predict(img, stream=True, verbose=False)
            phone_detected = False
            classNames = st.session_state.phone_detector.names
            
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    
                    if classNames[cls_id] == "cell phone" and conf > phone_conf_threshold:
                        phone_detected = True
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cv2.rectangle(img, (x1, y1), (x2, y2), (236, 72, 153), 2) # Magenta glowing box
                        cvzone.putTextRect(img, f"Phone: {int(conf*100)}%", (x1, max(y1 - 10, 30)), scale=1, thickness=1, colorR=(236, 72, 153))
                        
            # 3. Log transitions to Database
            if is_face_covered and not st.session_state.was_covered:
                database.log_distraction(st.session_state.session_id, "face_covered")
                st.session_state.cover_count += 1
            if is_sleepy and not st.session_state.was_sleepy:
                database.log_distraction(st.session_state.session_id, "sleep")
                st.session_state.sleep_count += 1
            if phone_detected and not st.session_state.was_phone:
                database.log_distraction(st.session_state.session_id, "phone")
                st.session_state.phone_count += 1
                
            st.session_state.was_covered = is_face_covered
            st.session_state.was_sleepy = is_sleepy
            st.session_state.was_phone = phone_detected
            
            # 4. Alarm Playing & Overlay Warning Text
            status_text = "FOCUSING..."
            status_text_color = "#34d399" # Emerald green
            now_time = time.time()
            distracted = False
            
            if is_face_covered:
                status_text = "FACE HIDDEN!"
                status_text_color = "#ef4444" # Vivid Red
                cvzone.putTextRect(img, "DONT COVER YOUR FACE!", (50, 100), scale=2, thickness=3, colorR=(239, 68, 68), colorT=(255, 255, 255))
                distracted = True
                
                if enable_facehide and not mute_audio:
                    # Play if not already playing or cooldown expired (4.0s)
                    if st.session_state.current_playing != "alarm_facehide" or (now_time - st.session_state.last_played_time > 4.0):
                        play_mci_sound("alarm_facehide", path_facehide)
                        st.session_state.current_playing = "alarm_facehide"
                        st.session_state.last_played_time = now_time
            elif is_sleepy:
                status_text = "SLEEPING!"
                status_text_color = "#ef4444" # Vivid Red
                cvzone.putTextRect(img, "WAKE UP & STUDY!", (50, 100), scale=2, thickness=3, colorR=(239, 68, 68), colorT=(255, 255, 255))
                distracted = True
                
                if enable_sleep and not mute_audio:
                    # Play if not already playing or cooldown expired (4.0s)
                    if st.session_state.current_playing != "alarm_sleep" or (now_time - st.session_state.last_played_time > 4.0):
                        play_mci_sound("alarm_sleep", path_sleep)
                        st.session_state.current_playing = "alarm_sleep"
                        st.session_state.last_played_time = now_time
            elif phone_detected:
                status_text = "PHONE DETECTED!"
                status_text_color = "#f97316" # Orange
                cvzone.putTextRect(img, "PUT THE PHONE AWAY!", (50, 100), scale=2, thickness=3, colorR=(249, 115, 22), colorT=(255, 255, 255))
                distracted = True
                
                if enable_phone and not mute_audio:
                    # Play if not already playing or cooldown expired (4.0s)
                    if st.session_state.current_playing != "alarm_phone" or (now_time - st.session_state.last_played_time > 4.0):
                        play_mci_sound("alarm_phone", path_phone)
                        st.session_state.current_playing = "alarm_phone"
                        st.session_state.last_played_time = now_time
                        
            # If no distraction is active, silence any running alarms immediately
            if not distracted:
                if st.session_state.current_playing is not None:
                    stop_all_sounds()
                    st.session_state.current_playing = None
                    
            # 5. Timer Increment
            now = time.time()
            if now - st.session_state.last_time >= 1.0:
                st.session_state.total_sec += 1
                if not (is_sleepy or phone_detected or is_face_covered):
                    st.session_state.focus_sec += 1
                st.session_state.last_time = now
                
            tot_h = st.session_state.total_sec // 3600
            tot_m = (st.session_state.total_sec % 3600) // 60
            tot_s = st.session_state.total_sec % 60
            time_str = f"{tot_h:02d}:{tot_m:02d}:{tot_s:02d}"
            
            focus_pct = int((st.session_state.focus_sec / st.session_state.total_sec) * 100) if st.session_state.total_sec > 0 else 100
            distractions_tally = st.session_state.sleep_count + st.session_state.phone_count + st.session_state.cover_count
            
            # Write to custom HTML metrics bar (Premium look)
            metrics_html = f"""
            <div style="display: flex; gap: 15px; justify-content: space-between; margin-bottom: 25px; flex-wrap: wrap;">
                <div style="flex: 1; min-width: 150px; background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); border-radius: 16px; padding: 15px; text-align: center; border: 1px solid #334155; box-shadow: 0 10px 15px -3px rgba(0,0,0,0.3);">
                    <div style="font-size: 0.8rem; color: #94a3b8; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;">Study Timer ⏱️</div>
                    <div style="font-size: 1.8rem; font-weight: 700; color: #38bdf8; font-family: monospace; margin-top: 5px;">{time_str}</div>
                </div>
                <div style="flex: 1; min-width: 150px; background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); border-radius: 16px; padding: 15px; text-align: center; border: 1px solid #334155; box-shadow: 0 10px 15px -3px rgba(0,0,0,0.3);">
                    <div style="font-size: 0.8rem; color: #94a3b8; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;">Focus Score 🎯</div>
                    <div style="font-size: 1.8rem; font-weight: 700; color: #34d399; margin-top: 5px;">{focus_pct}%</div>
                </div>
                <div style="flex: 1; min-width: 150px; background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); border-radius: 16px; padding: 15px; text-align: center; border: 1px solid #334155; box-shadow: 0 10px 15px -3px rgba(0,0,0,0.3);">
                    <div style="font-size: 0.8rem; color: #94a3b8; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;">Alert Status 🚨</div>
                    <div style="font-size: 1.6rem; font-weight: 700; color: {status_text_color}; margin-top: 5px;">{status_text}</div>
                </div>
                <div style="flex: 1; min-width: 150px; background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); border-radius: 16px; padding: 15px; text-align: center; border: 1px solid #334155; box-shadow: 0 10px 15px -3px rgba(0,0,0,0.3);">
                    <div style="font-size: 0.8rem; color: #94a3b8; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;">Distractions ⚠️</div>
                    <div style="font-size: 1.8rem; font-weight: 700; color: #f87171; margin-top: 5px;">{distractions_tally}</div>
                </div>
            </div>
            """
            metric_ph.markdown(metrics_html, unsafe_allow_html=True)
            
            # Display image (updates in-place via WebSocket, completely flicker-free!)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            video_feed_ph.image(img_rgb, channels="RGB", use_container_width=True)
            
            # Control frame rate to approx. 30 FPS
            time.sleep(0.03)
            
        # Trigger next batch execution frame sequence if session remains active
        if st.session_state.session_active and not st.session_state.session_paused:
            st.rerun()
            
    elif st.session_state.session_active and st.session_state.session_paused:
        st.info("Your study session is currently paused. Press 'Resume Session' above to continue monitoring.")
        
    else:
        st.markdown(
            """
            <div style="background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); border: 1px solid #334155; border-radius: 20px; padding: 30px; box-shadow: 0 20px 25px -5px rgba(0,0,0,0.3); margin-top: 15px;">
                <h3 style="margin-top: 0; color: #3b82f6; font-size: 1.4rem;">Get Started with your Study Co-pilot 🚀</h3>
                <p style="color: #94a3b8; font-size: 0.95rem;">This utility leverages advanced computer vision to ensure you stay in your deep-work zone.</p>
                <div style="margin-top: 20px;">
                    <h5 style="color: #cbd5e1; font-weight: 600; margin-bottom: 8px;">Instructions:</h5>
                    <ol style="color: #94a3b8; padding-left: 20px; font-size: 0.9rem; line-height: 1.8;">
                        <li>Tune your distraction thresholds in the left Configuration Panel.</li>
                        <li>Ensure your environment is well-lit and press <b style="color:#60a5fa;">Start New Session</b> above.</li>
                        <li>The system will track your focus score, eye closures, face coverings, and phone usage.</li>
                        <li>Real-time notifications and audio tones will play if distraction is flagged.</li>
                        <li>Click <b style="color:#f87171;">Stop & Save Session</b> to commit stats to the local database, unlocking insights under the Focus Analytics tab.</li>
                    </ol>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

# ----- TAB 2: FOCUS ANALYTICS -----
with tabs[1]:
    st.subheader("Concentration & Productivity Reports")
    
    # Query database for session entries
    df_sessions = get_data("SELECT * FROM sessions")
    
    if df_sessions.empty:
        st.info("No study sessions logged yet. Complete at least one study session in the Live Monitor tab to see analytics!")
    else:
        total_sessions = len(df_sessions)
        df_sessions['start_time'] = pd.to_datetime(df_sessions['start_time'])
        df_sessions['end_time'] = pd.to_datetime(df_sessions['end_time'])
        
        total_duration_hours = df_sessions['total_duration_sec'].sum() / 3600.0
        total_focus_hours = df_sessions['focus_duration_sec'].sum() / 3600.0
        
        total_sec = df_sessions['total_duration_sec'].sum()
        avg_focus_score = int((df_sessions['focus_duration_sec'].sum() / total_sec) * 100) if total_sec > 0 else 100
        
        df_distractions = get_data("SELECT * FROM distractions")
        total_distractions = len(df_distractions)
        
        # Grid of stats (Custom Premium HTML metric cards)
        st.markdown(
            f"""
            <div style="display: flex; gap: 20px; justify-content: space-between; margin-bottom: 30px; flex-wrap: wrap;">
                <div class="metric-card-wrapper" style="flex: 1; min-width: 200px;">
                    <div style="font-size: 0.85rem; color: #94a3b8; font-weight: 600; text-transform: uppercase;">Total Sessions 📅</div>
                    <div style="font-size: 2.2rem; font-weight: 700; color: #3b82f6; margin-top: 5px;">{total_sessions}</div>
                </div>
                <div class="metric-card-wrapper" style="flex: 1; min-width: 200px;">
                    <div style="font-size: 0.85rem; color: #94a3b8; font-weight: 600; text-transform: uppercase;">Total Study Time ⌛</div>
                    <div style="font-size: 2.2rem; font-weight: 700; color: #a855f7; margin-top: 5px;">{total_duration_hours:.2f} <span style="font-size: 1rem; font-weight: 500; color:#a855f7;">hrs</span></div>
                </div>
                <div class="metric-card-wrapper" style="flex: 1; min-width: 200px;">
                    <div style="font-size: 0.85rem; color: #94a3b8; font-weight: 600; text-transform: uppercase;">Focused Study Time 🎯</div>
                    <div style="font-size: 2.2rem; font-weight: 700; color: #34d399; margin-top: 5px;">{total_focus_hours:.2f} <span style="font-size: 1rem; font-weight: 500; color:#34d399;">hrs</span></div>
                </div>
                <div class="metric-card-wrapper" style="flex: 1; min-width: 200px;">
                    <div style="font-size: 0.85rem; color: #94a3b8; font-weight: 600; text-transform: uppercase;">Average Focus score ⭐</div>
                    <div style="font-size: 2.2rem; font-weight: 700; color: #f59e0b; margin-top: 5px;">{avg_focus_score}%</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        st.markdown("---")
        
        col_left, col_right = st.columns(2)
        with col_left:
            st.subheader("Study vs. Focus Duration Trends")
            # Customize Matplotlib for Dark Theme
            plt.style.use('dark_background')
            fig, ax = plt.subplots(figsize=(8, 5))
            fig.patch.set_facecolor('#0b0f19')
            ax.set_facecolor('#0f172a')
            
            x = range(1, len(df_sessions) + 1)
            ax.plot(x, df_sessions['total_duration_sec'] / 60.0, label='Total Session Time', marker='o', color='#3b82f6', linewidth=2.5)
            ax.plot(x, df_sessions['focus_duration_sec'] / 60.0, label='Focused Study Time', marker='s', color='#10b981', linewidth=2.5)
            ax.fill_between(x, df_sessions['focus_duration_sec'] / 60.0, alpha=0.15, color='#10b981')
            
            ax.set_xlabel("Session Number", color='#94a3b8', fontsize=10, fontweight='semibold')
            ax.set_ylabel("Duration (Minutes)", color='#94a3b8', fontsize=10, fontweight='semibold')
            ax.tick_params(colors='#64748b')
            ax.spines['bottom'].color = '#334155'
            ax.spines['top'].color = 'none'
            ax.spines['left'].color = '#334155'
            ax.spines['right'].color = 'none'
            ax.legend(facecolor='#0f172a', edgecolor='#334155')
            ax.grid(True, linestyle='--', alpha=0.15, color='#cbd5e1')
            st.pyplot(fig)
            
        with col_right:
            st.subheader("Distraction Analysis")
            if not df_distractions.empty:
                dist_counts = df_distractions['type'].value_counts()
                mapping = {'sleep': 'Sleeping', 'phone': 'Using Phone', 'face_covered': 'Face Covered'}
                dist_counts.index = dist_counts.index.map(lambda x: mapping.get(x, x))
                
                # Dark themed pie chart
                fig, ax = plt.subplots(figsize=(8, 5))
                fig.patch.set_facecolor('#0b0f19')
                ax.set_facecolor('#0f172a')
                
                colors = ['#f43f5e','#3b82f6','#10b981'] # Red, Blue, Green
                wedges, texts, autotexts = ax.pie(
                    dist_counts, 
                    labels=dist_counts.index, 
                    autopct='%1.1f%%', 
                    startangle=90, 
                    colors=colors[:len(dist_counts)], 
                    wedgeprops=dict(width=0.4, edgecolor='#0b0f19', linewidth=2),
                    textprops=dict(color='#cbd5e1', weight='semibold')
                )
                for autotext in autotexts:
                    autotext.set_color('#ffffff')
                    
                ax.set_title("Share of Distraction Violations", color='#94a3b8', fontsize=12, fontweight='bold')
                st.pyplot(fig)
            else:
                st.success("No distraction flags detected! Outstanding performance! 🌟")
                
        st.markdown("---")
        
        if not df_distractions.empty:
            st.subheader("Distraction Distribution Throughout the Day")
            df_distractions['timestamp'] = pd.to_datetime(df_distractions['timestamp'])
            df_distractions['hour'] = df_distractions['timestamp'].dt.hour
            hourly_counts = df_distractions.groupby('hour').size().reindex(range(0, 24), fill_value=0).reset_index(name='count')
            
            fig, ax = plt.subplots(figsize=(14, 4))
            fig.patch.set_facecolor('#0b0f19')
            ax.set_facecolor('#0f172a')
            
            ax.bar(hourly_counts['hour'], hourly_counts['count'], color='#f97316', alpha=0.8, edgecolor='#7c2d12', linewidth=1)
            ax.set_xlabel("Hour of the Day (24-Hour scale)", color='#94a3b8', fontsize=10, fontweight='semibold')
            ax.set_ylabel("Distractions Count", color='#94a3b8', fontsize=10, fontweight='semibold')
            ax.set_xticks(range(0, 24))
            ax.tick_params(colors='#64748b')
            ax.spines['bottom'].color = '#334155'
            ax.spines['top'].color = 'none'
            ax.spines['left'].color = '#334155'
            ax.spines['right'].color = 'none'
            ax.grid(axis='y', linestyle='--', alpha=0.15, color='#cbd5e1')
            st.pyplot(fig)

# ----- TAB 3: SESSION HISTORY LOG -----
with tabs[2]:
    st.subheader("Study Sessions Log")
    df_sessions = get_data("SELECT * FROM sessions")
    if df_sessions.empty:
        st.info("No study sessions logged yet.")
    else:
        df_sessions_display = df_sessions.copy()
        df_sessions_display['focus_score'] = (df_sessions_display['focus_duration_sec'] / df_sessions_display['total_duration_sec'] * 100).fillna(100).astype(int)
        df_sessions_display['focus_score'] = df_sessions_display['focus_score'].apply(lambda x: f"{x}%")
        df_sessions_display['total_duration'] = df_sessions_display['total_duration_sec'].apply(lambda x: f"{x//60}m {x%60}s")
        df_sessions_display['focus_duration'] = df_sessions_display['focus_duration_sec'].apply(lambda x: f"{x//60}m {x%60}s")
        df_sessions_display['start_time'] = pd.to_datetime(df_sessions_display['start_time']).dt.strftime('%Y-%m-%d %H:%M:%S')
        df_sessions_display['end_time'] = pd.to_datetime(df_sessions_display['end_time']).dt.strftime('%Y-%m-%d %H:%M:%S')
        
        df_display = df_sessions_display[['id', 'start_time', 'end_time', 'total_duration', 'focus_duration', 'focus_score']].rename(columns={
            'id': 'Session ID',
            'start_time': 'Start Time',
            'end_time': 'End Time',
            'total_duration': 'Total Session Time',
            'focus_duration': 'Focused Study Time',
            'focus_score': 'Focus Score'
        })
        
        st.dataframe(df_display, use_container_width=True)
