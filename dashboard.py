import streamlit as st
import cv2
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector
from ultralytics import YOLO
import os
import time
import base64
import pandas as pd
import matplotlib.pyplot as plt
import database
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode

# Ensure DB is initialized
database.init_db()

# Set page config
st.set_page_config(
    page_title="Smart Study Monitor", 
    page_icon="🎓", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium Styling Injection (Glassmorphism & Advanced Animations)
st.markdown(
    """
    <style>
    /* Global Font & Smooth Layout */
    @import url('https://fonts.googleapis.com/css2?family=Cabinet+Grotesk:wght@500;700;800&family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"], .stApp {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        background-color: #0b0f19 !important;
        color: #f8fafc !important;
    }
    
    /* Premium Headers */
    h1, h2, h3, h4 {
        font-family: 'Cabinet Grotesk', sans-serif !important;
        font-weight: 700 !important;
    }
    
    /* Clean Glassmorphic Sidebar */
    section[data-testid="stSidebar"] {
        background-color: rgba(15, 23, 42, 0.8) !important;
        backdrop-filter: blur(16px);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    /* Rounded & Modern WebCam Containers */
    .stImage img, div[data-testid="stWebRtcStreamer"] iframe {
        border-radius: 20px;
        border: 2px solid rgba(255, 255, 255, 0.05);
        box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.7);
    }
    
    /* Premium Hover Buttons */
    .stButton>button {
        border-radius: 14px !important;
        font-weight: 600 !important;
        padding: 0.7rem 1.5rem !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        background: linear-gradient(135deg, rgba(37, 99, 235, 0.2) 0%, rgba(37, 99, 235, 0.05) 100%) !important;
        color: #f8fafc !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        background: linear-gradient(135deg, rgba(37, 99, 235, 0.4) 0%, rgba(37, 99, 235, 0.1) 100%) !important;
        box-shadow: 0 8px 20px rgba(37, 99, 235, 0.25);
        border-color: rgba(37, 99, 235, 0.4) !important;
    }
    
    /* Glassmorphic Tabs Navigation */
    div[data-testid="stTabBar"] {
        background: rgba(15, 23, 42, 0.5);
        backdrop-filter: blur(12px);
        border-radius: 18px;
        padding: 6px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        margin-bottom: 25px;
    }
    button[data-baseweb="tab"] {
        border-radius: 14px !important;
        padding: 10px 24px !important;
        color: #94a3b8 !important;
        font-weight: 600 !important;
        border: none !important;
        transition: all 0.25s ease !important;
    }
    button[aria-selected="true"] {
        background: #2563eb !important;
        color: #ffffff !important;
        box-shadow: 0 4px 15px rgba(37, 99, 235, 0.4);
    }
    
    /* Interactive Glassmorphic Metric Cards */
    .metric-card-wrapper {
        background: rgba(30, 41, 59, 0.4) !important;
        backdrop-filter: blur(12px) !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        border-radius: 20px !important;
        padding: 20px !important;
        text-align: center;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    .metric-card-wrapper:hover {
        transform: translateY(-4px) scale(1.01) !important;
        border-color: #3b82f6 !important;
        box-shadow: 0 20px 30px -5px rgba(59, 130, 246, 0.15) !important;
    }
    
    /* Animated Gradient Banner */
    .dashboard-banner {
        background: linear-gradient(135deg, rgba(37, 99, 235, 0.1) 0%, rgba(168, 85, 247, 0.1) 100%);
        border: 1px solid rgba(59, 130, 246, 0.15);
        border-radius: 24px;
        padding: 35px;
        margin-bottom: 30px;
        position: relative;
        overflow: hidden;
        box-shadow: 0 20px 40px -15px rgba(0, 0, 0, 0.5);
    }
    .dashboard-banner::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(59, 130, 246, 0.07) 0%, transparent 60%);
        animation: rotate 20s linear infinite;
        z-index: 0;
    }
    @keyframes rotate {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Clean Sidebar Sliders */
    div[data-testid="stSlider"] {
        padding-top: 10px;
    }
    
    /* Alert Glowing Badge */
    .status-badge {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 30px;
        font-weight: 700;
        font-size: 0.85rem;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        animation: pulse-glow 2s infinite;
    }
    @keyframes pulse-glow {
        0% { transform: scale(1); }
        50% { transform: scale(1.03); opacity: 0.9; }
        100% { transform: scale(1); }
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
    except Exception:
        return pd.DataFrame()

# Base64 Audio helper to embed and play audio directly in client browser (cloud-compatible)
def get_audio_base64(filename):
    filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    if os.path.exists(filepath):
        try:
            with open(filepath, "rb") as f:
                data = f.read()
            return base64.b64encode(data).decode('utf-8')
        except Exception:
            return ""
    return ""

audio_sleep_b64 = get_audio_base64("alarm.mp3")
audio_facehide_b64 = get_audio_base64("faudio.mp3")
audio_phone_b64 = get_audio_base64("paudio.mp3")

# Cache model resources at module level to prevent reloading on every Streamlit rerun
@st.cache_resource
def load_yolo_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "yolov8n.pt")
    return YOLO(model_path)

@st.cache_resource
def load_face_detector():
    return FaceMeshDetector(maxFaces=1)

yolo_model = load_yolo_model()
face_mesh = load_face_detector()

# WebRTC Video Processor (runs on background thread, updates thread-safe variables)
class StudyVideoProcessor(VideoProcessorBase):
    def __init__(self, yolo_model, face_mesh):
        self.phone_detector = yolo_model
        self.face_detector = face_mesh
        
        # User thresholds (synced dynamically from sidebar)
        self.eye_ratio_threshold = 11.0
        self.sleep_threshold = 15
        self.phone_conf_threshold = 0.5
        
        self.enable_sleep = True
        self.enable_phone = True
        self.enable_facehide = True
        
        # Thread status outputs
        self.is_sleepy = False
        self.phone_detected = False
        self.is_face_covered = False
        
        # Detection frame counters
        self.closed_frames = 0
        self.covered_frames = 0
        self.frame_counter = 0
        self.last_phone_detected = False
        self.last_phone_boxes = []
        self.last_faces = None
        self.last_is_sleepy = False
        self.last_is_face_covered = False
        self.last_ratio = 15.0
        
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        self.frame_counter += 1
        
        # 1. Face Mesh Sleep & Cover detection (Only run once every 2 frames)
        if self.frame_counter % 2 == 0 or self.last_faces is None:
            _, faces = self.face_detector.findFaceMesh(img, draw=False)
            is_sleepy = False
            is_face_covered = False
            ratio = 15.0
            
            LEFT_EYE_TOP = 159
            LEFT_EYE_BOTTOM = 145
            FACE_LEFT = 130
            FACE_RIGHT = 243
            
            if faces:
                self.covered_frames = 0
                face = faces[0]
                try:
                    eye_dist, _ = self.face_detector.findDistance(face[LEFT_EYE_TOP], face[LEFT_EYE_BOTTOM])
                    face_dist, _ = self.face_detector.findDistance(face[FACE_LEFT], face[FACE_RIGHT])
                    ratio = (eye_dist / face_dist) * 100
                except Exception:
                    ratio = 15.0
                    
                if ratio < self.eye_ratio_threshold:
                    self.closed_frames += 1
                else:
                    self.closed_frames = 0
                    
                if self.closed_frames >= self.sleep_threshold:
                    is_sleepy = True
            else:
                self.closed_frames = 0
                self.covered_frames += 1
                if self.covered_frames >= 20:
                    is_face_covered = True
                    
            self.last_faces = faces
            self.last_is_sleepy = is_sleepy
            self.last_is_face_covered = is_face_covered
            self.last_ratio = ratio
            
        faces = self.last_faces
        is_sleepy = self.last_is_sleepy
        is_face_covered = self.last_is_face_covered
        ratio = self.last_ratio
        
        # Draw face features on all frames
        if faces:
            face = faces[0]
            LEFT_EYE_TOP = 159
            LEFT_EYE_BOTTOM = 145
            cv2.circle(img, face[LEFT_EYE_TOP], 3, (34, 197, 94), -1)
            cv2.circle(img, face[LEFT_EYE_BOTTOM], 3, (34, 197, 94), -1)
            cvzone.putTextRect(img, f"Eye Ratio: {int(ratio)}", (30, 40), scale=1, thickness=1, colorR=(15, 23, 42))
            
        # 2. YOLO Phone Detection (Only run once every 6 frames for performance)
        if self.frame_counter % 6 == 0 or self.frame_counter < 10:
            results = self.phone_detector.predict(img, stream=True, verbose=False)
            phone_detected = False
            classNames = self.phone_detector.names
            phone_boxes = []
            
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    
                    if classNames[cls_id] == "cell phone" and conf > self.phone_conf_threshold:
                        phone_detected = True
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        phone_boxes.append((x1, y1, x2, y2, int(conf*100)))
            
            self.last_phone_detected = phone_detected
            self.last_phone_boxes = phone_boxes
            
        phone_detected = self.last_phone_detected
        for (x1, y1, x2, y2, conf_pct) in self.last_phone_boxes:
            cv2.rectangle(img, (x1, y1), (x2, y2), (236, 72, 153), 2)
            cvzone.putTextRect(img, f"Phone: {conf_pct}%", (x1, max(y1 - 10, 30)), scale=1, thickness=1, colorR=(236, 72, 153))
                    
        # Update thread-safe status fields
        self.is_sleepy = is_sleepy
        self.phone_detected = phone_detected
        self.is_face_covered = is_face_covered
        
        # Warning Banners overlay
        if is_face_covered:
            cvzone.putTextRect(img, "DONT COVER YOUR FACE!", (50, 100), scale=2, thickness=3, colorR=(239, 68, 68), colorT=(255, 255, 255))
        elif is_sleepy:
            cvzone.putTextRect(img, "WAKE UP & STUDY!", (50, 100), scale=2, thickness=3, colorR=(239, 68, 68), colorT=(255, 255, 255))
        elif phone_detected:
            cvzone.putTextRect(img, "PUT THE PHONE AWAY!", (50, 100), scale=2, thickness=3, colorR=(249, 115, 22), colorT=(255, 255, 255))
            
        import av
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Initialize session state variables robustly
if 'session_active' not in st.session_state:
    st.session_state.session_active = False
if 'session_paused' not in st.session_state:
    st.session_state.session_paused = False
if 'session_id' not in st.session_state:
    st.session_state.session_id = None
if 'cap' not in st.session_state:
    st.session_state.cap = None

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
if 'last_played_time' not in st.session_state:
    st.session_state.last_played_time = 0
if 'covered_frames' not in st.session_state:
    st.session_state.covered_frames = 0
if 'closed_frames' not in st.session_state:
    st.session_state.closed_frames = 0

# Ensure webcam releases if session is inactive
if not st.session_state.session_active:
    if st.session_state.cap is not None:
        st.session_state.cap.release()
        st.session_state.cap = None

# ----------------- SIDEBAR CONFIGURATION -----------------
st.sidebar.markdown(
    """
    <div style="margin-bottom: 25px;">
        <h2 style="color: #3b82f6; font-size: 1.5rem; letter-spacing: -0.02em; margin-bottom: 0px;">FOCUS SPACE</h2>
        <span style="color: #64748b; font-size: 0.8rem; font-weight: 500;">Active Session Controls</span>
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.subheader("Camera Input Mode 🎥")
camera_mode = st.sidebar.selectbox(
    "Select Source",
    ["Local Webcam (OpenCV)", "Cloud WebRTC (Browser)"],
    help="Select Local Webcam for native zero-lag execution. Choose Cloud WebRTC when testing browser-only from a deployed link."
)

st.sidebar.markdown("---")
st.sidebar.subheader("Alert Toggles 🔔")
mute_audio = st.sidebar.checkbox("Mute sound alarms", value=False)
enable_sleep = st.sidebar.checkbox("Enable drowsiness alerts", value=True)
enable_phone = st.sidebar.checkbox("Enable phone detection alerts", value=True)
enable_facehide = st.sidebar.checkbox("Enable camera-covered alerts", value=True)

st.sidebar.markdown("---")
st.sidebar.subheader("Fine-Tuning ⚙️")

eye_ratio_threshold = st.sidebar.slider("Drowsiness Eye Threshold", min_value=5.0, max_value=20.0, value=11.0, step=0.5)
sleep_threshold_frames = st.sidebar.slider("Drowsiness Duration (Frames)", min_value=5, max_value=50, value=15)
phone_conf_threshold = st.sidebar.slider("YOLO Phone Confidence (%)", min_value=30, max_value=90, value=50) / 100.0

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    <div style="font-size: 0.8rem; color: #475569; font-weight: 500; text-align: center;">
        Built for distraction-free deep work.
    </div>
    """,
    unsafe_allow_html=True
)

# ----------------- MAIN LAYOUT -----------------
st.markdown(
    """
    <div style="display: flex; align-items: center; gap: 15px; margin-bottom: 25px;">
        <div style="background: linear-gradient(135deg, #2563eb 0%, #7c3aed 100%); padding: 12px; border-radius: 16px; font-size: 1.8rem; line-height: 1; box-shadow: 0 10px 20px rgba(37, 99, 235, 0.2);">🎓</div>
        <div>
            <h1 style="margin: 0; font-size: 2.2rem; font-weight: 800; background: linear-gradient(to right, #3b82f6, #a855f7); -webkit-background-clip: text; -webkit-text-fill-color: transparent; letter-spacing: -0.03em;">Smart Study Monitor</h1>
            <p style="margin: 0; color: #94a3b8; font-size: 0.95rem; font-weight: 400;">An interactive, real-time companion designed to maximize study habits and keep distractions away.</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

tabs = st.tabs(["🎥 Focus Space", "📊 Analytics", "📋 History"])

# ----- TAB 1: FOCUS SPACE -----
with tabs[0]:
    
    # ------------------ MODE 1: LOCAL OPENCV STREAM ------------------
    if camera_mode == "Local Webcam (OpenCV)":
        st.subheader("Webcam Session Stream")
        
        col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
        with col_btn1:
            if not st.session_state.session_active:
                if st.button("▶️ Start Session", type="primary", use_container_width=True):
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
                    st.rerun()
            else:
                if st.button("⏹️ Stop & Save", type="primary", use_container_width=True):
                    if st.session_state.cap is not None:
                        st.session_state.cap.release()
                        st.session_state.cap = None
                    database.end_session(st.session_state.session_id, st.session_state.total_sec, st.session_state.focus_sec)
                    st.session_state.session_active = False
                    st.session_state.session_id = None
                    st.toast("Session successfully logged! Head to Analytics to review your stats.", icon="💾")
                    st.rerun()
                    
        with col_btn2:
            if st.session_state.session_active:
                if not st.session_state.session_paused:
                    if st.button("⏸️ Pause Session", use_container_width=True):
                        st.session_state.session_paused = True
                        st.rerun()
                else:
                    if st.button("▶️ Resume Session", use_container_width=True):
                        st.session_state.session_paused = False
                        st.session_state.last_time = time.time()
                        st.rerun()
                        
        with col_btn3:
            if st.session_state.session_active:
                if st.session_state.session_paused:
                    st.markdown('<div style="text-align:center;"><span class="status-badge" style="background:rgba(234,179,8,0.15); border:1px solid #eab308; color:#facc15;">Session Paused</span></div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div style="text-align:center;"><span class="status-badge" style="background:rgba(34,197,94,0.15); border:1px solid #22c55e; color:#4ade80;">Monitoring Active</span></div>', unsafe_allow_html=True)
            else:
                st.markdown('<div style="text-align:center;"><span class="status-badge" style="background:rgba(148,163,184,0.15); border:1px solid #64748b; color:#94a3b8;">Session Inactive</span></div>', unsafe_allow_html=True)
                
        st.markdown("---")

        if st.session_state.session_active and not st.session_state.session_paused:
            if st.session_state.cap is None:
                st.session_state.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
                st.session_state.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                st.session_state.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                st.session_state.cap.read()
                
            metric_ph = st.empty()
            video_feed_ph = st.empty()
            audio_ph = st.empty()
            
            failed_reads = 0
            # 100-frame in-place loop to reduce Streamlit rerun overhead stutters
            for frame_idx in range(100):
                if not st.session_state.session_active or st.session_state.session_paused:
                    break
                    
                success, img = st.session_state.cap.read()
                if not success:
                    failed_reads += 1
                    time.sleep(0.1)
                    if failed_reads > 40:
                        st.session_state.session_active = False
                        if st.session_state.cap is not None:
                            st.session_state.cap.release()
                            st.session_state.cap = None
                        st.error("Could not connect to webcam. Please ensure your camera is connected and not in use by another app.")
                        st.rerun()
                        break
                    continue
                    
                img = cv2.flip(img, 1)
                
                # 1. Face Mesh Sleep & Cover detection (Only run once every 2 frames)
                if 'last_faces' not in st.session_state:
                    st.session_state.last_faces = None
                if 'last_is_sleepy' not in st.session_state:
                    st.session_state.last_is_sleepy = False
                if 'last_is_face_covered' not in st.session_state:
                    st.session_state.last_is_face_covered = False
                if 'last_ratio' not in st.session_state:
                    st.session_state.last_ratio = 15.0

                if frame_idx % 2 == 0 or st.session_state.last_faces is None:
                    _, faces = face_mesh.findFaceMesh(img, draw=False)
                    is_sleepy = False
                    is_face_covered = False
                    ratio = 15.0
                    
                    LEFT_EYE_TOP = 159
                    LEFT_EYE_BOTTOM = 145
                    FACE_LEFT = 130
                    FACE_RIGHT = 243
                    
                    if faces:
                        st.session_state.covered_frames = 0
                        face = faces[0]
                        try:
                            eye_dist, _ = face_mesh.findDistance(face[LEFT_EYE_TOP], face[LEFT_EYE_BOTTOM])
                            face_dist, _ = face_mesh.findDistance(face[FACE_LEFT], face[FACE_RIGHT])
                            ratio = (eye_dist / face_dist) * 100
                        except Exception:
                            ratio = 15.0
                            
                        if ratio < eye_ratio_threshold:
                            st.session_state.closed_frames += 1
                        else:
                            st.session_state.closed_frames = 0
                            
                        if st.session_state.closed_frames >= sleep_threshold_frames:
                            is_sleepy = True
                    else:
                        st.session_state.closed_frames = 0
                        st.session_state.covered_frames += 1
                        if st.session_state.covered_frames >= 20:
                            is_face_covered = True
                            
                    st.session_state.last_faces = faces
                    st.session_state.last_is_sleepy = is_sleepy
                    st.session_state.last_is_face_covered = is_face_covered
                    st.session_state.last_ratio = ratio
                    
                faces = st.session_state.last_faces
                is_sleepy = st.session_state.last_is_sleepy
                is_face_covered = st.session_state.last_is_face_covered
                ratio = st.session_state.last_ratio
                
                # Draw facial dots on all frames
                if faces:
                    face = faces[0]
                    LEFT_EYE_TOP = 159
                    LEFT_EYE_BOTTOM = 145
                    cv2.circle(img, face[LEFT_EYE_TOP], 3, (34, 197, 94), -1)
                    cv2.circle(img, face[LEFT_EYE_BOTTOM], 3, (34, 197, 94), -1)
                    cvzone.putTextRect(img, f"Eye Aspect Ratio: {int(ratio)}", (30, 40), scale=1, thickness=1, colorR=(15, 23, 42))
                        
                # 2. YOLO Phone Detection (Only run once every 6 frames)
                if 'last_phone_detected' not in st.session_state:
                    st.session_state.last_phone_detected = False
                if 'last_phone_boxes' not in st.session_state:
                    st.session_state.last_phone_boxes = []
                    
                if frame_idx % 6 == 0:
                    results = yolo_model.predict(img, stream=True, verbose=False)
                    phone_detected = False
                    classNames = yolo_model.names
                    phone_boxes = []
                    
                    for r in results:
                        boxes = r.boxes
                        for box in boxes:
                            cls_id = int(box.cls[0])
                            conf = float(box.conf[0])
                            
                            if classNames[cls_id] == "cell phone" and conf > phone_conf_threshold:
                                phone_detected = True
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                phone_boxes.append((x1, y1, x2, y2, int(conf*100)))
                                
                    st.session_state.last_phone_detected = phone_detected
                    st.session_state.last_phone_boxes = phone_boxes
                    
                phone_detected = st.session_state.last_phone_detected
                for (x1, y1, x2, y2, conf_pct) in st.session_state.last_phone_boxes:
                    cv2.rectangle(img, (x1, y1), (x2, y2), (236, 72, 153), 2)
                    cvzone.putTextRect(img, f"Phone: {conf_pct}%", (x1, max(y1 - 10, 30)), scale=1, thickness=1, colorR=(236, 72, 153))
                            
                # 3. DB Logging
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
                
                # 4. Audio Alert Playing (Browser Client-side Autoplay base64 HTML5)
                status_text = "FOCUSING..."
                status_text_color = "#34d399"
                distracted = False
                now = time.time()
                
                if is_face_covered:
                    status_text = "FACE HIDDEN!"
                    status_text_color = "#ef4444"
                    cvzone.putTextRect(img, "DONT COVER YOUR FACE!", (50, 100), scale=2, thickness=3, colorR=(239, 68, 68), colorT=(255, 255, 255))
                    distracted = True
                    if enable_facehide and not mute_audio and audio_facehide_b64:
                        if st.session_state.current_playing != "facehide" or (now - st.session_state.last_played_time > 4.0):
                            st.session_state.current_playing = "facehide"
                            st.session_state.last_played_time = now
                            audio_ph.markdown(f'<audio autoplay src="data:audio/mp3;base64,{audio_facehide_b64}"></audio>', unsafe_allow_html=True)
                elif is_sleepy:
                    status_text = "SLEEPING!"
                    status_text_color = "#ef4444"
                    cvzone.putTextRect(img, "WAKE UP & STUDY!", (50, 100), scale=2, thickness=3, colorR=(239, 68, 68), colorT=(255, 255, 255))
                    distracted = True
                    if enable_sleep and not mute_audio and audio_sleep_b64:
                        if st.session_state.current_playing != "sleep" or (now - st.session_state.last_played_time > 4.0):
                            st.session_state.current_playing = "sleep"
                            st.session_state.last_played_time = now
                            audio_ph.markdown(f'<audio autoplay src="data:audio/mp3;base64,{audio_sleep_b64}"></audio>', unsafe_allow_html=True)
                elif phone_detected:
                    status_text = "PHONE DETECTED!"
                    status_text_color = "#f97316"
                    cvzone.putTextRect(img, "PUT THE PHONE AWAY!", (50, 100), scale=2, thickness=3, colorR=(249, 115, 22), colorT=(255, 255, 255))
                    distracted = True
                    if enable_phone and not mute_audio and audio_phone_b64:
                        if st.session_state.current_playing != "phone" or (now - st.session_state.last_played_time > 4.0):
                            st.session_state.current_playing = "phone"
                            st.session_state.last_played_time = now
                            audio_ph.markdown(f'<audio autoplay src="data:audio/mp3;base64,{audio_phone_b64}"></audio>', unsafe_allow_html=True)
                            
                if not distracted:
                    st.session_state.current_playing = None
                    audio_ph.empty()
                    
                # 5. Timer Increment
                now_t = time.time()
                if now_t - st.session_state.last_time >= 1.0:
                    st.session_state.total_sec += 1
                    if not (is_sleepy or phone_detected or is_face_covered):
                        st.session_state.focus_sec += 1
                    st.session_state.last_time = now_t
                    
                tot_h = st.session_state.total_sec // 3600
                tot_m = (st.session_state.total_sec % 3600) // 60
                tot_s = st.session_state.total_sec % 60
                time_str = f"{tot_h:02d}:{tot_m:02d}:{tot_s:02d}"
                
                focus_pct = int((st.session_state.focus_sec / st.session_state.total_sec) * 100) if st.session_state.total_sec > 0 else 100
                distractions_tally = st.session_state.sleep_count + st.session_state.phone_count + st.session_state.cover_count
                
                # Update Metrics HTML
                metrics_html = f"""
                <div style="display: flex; gap: 15px; justify-content: space-between; margin-bottom: 25px; flex-wrap: wrap;">
                    <div class="metric-card-wrapper" style="flex: 1; min-width: 150px;">
                        <div style="font-size: 0.8rem; color: #94a3b8; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;">Timer ⏱️</div>
                        <div style="font-size: 1.8rem; font-weight: 700; color: #38bdf8; font-family: monospace; margin-top: 5px;">{time_str}</div>
                    </div>
                    <div class="metric-card-wrapper" style="flex: 1; min-width: 150px;">
                        <div style="font-size: 0.8rem; color: #94a3b8; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;">Focus Score 🎯</div>
                        <div style="font-size: 1.8rem; font-weight: 700; color: #34d399; margin-top: 5px;">{focus_pct}%</div>
                    </div>
                    <div class="metric-card-wrapper" style="flex: 1; min-width: 150px;">
                        <div style="font-size: 0.8rem; color: #94a3b8; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;">Status 🚨</div>
                        <div style="font-size: 1.6rem; font-weight: 700; color: {status_text_color}; margin-top: 5px;">{status_text}</div>
                    </div>
                    <div class="metric-card-wrapper" style="flex: 1; min-width: 150px;">
                        <div style="font-size: 0.8rem; color: #94a3b8; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;">Distractions ⚠️</div>
                        <div style="font-size: 1.8rem; font-weight: 700; color: #f87171; margin-top: 5px;">{distractions_tally}</div>
                    </div>
                </div>
                """
                metric_ph.markdown(metrics_html, unsafe_allow_html=True)
                
                # Render Image
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                video_feed_ph.image(img_rgb, channels="RGB", use_container_width=True)
                time.sleep(0.01)
                
            if st.session_state.session_active and not st.session_state.session_paused:
                st.rerun()
                
        elif st.session_state.session_active and st.session_state.session_paused:
            st.info("Your study session is paused. Click 'Resume Session' above to continue.")
        else:
            st.markdown(
                """
                <div class="dashboard-banner">
                    <div style="position: relative; z-index: 1;">
                        <h3 style="margin-top: 0; color: #3b82f6; font-size: 1.6rem; font-weight: 700; letter-spacing: -0.02em;">Find your flow state. 🧠</h3>
                        <p style="color: #cbd5e1; font-size: 1rem; line-height: 1.6; max-width: 800px; margin-bottom: 25px;">
                            Welcome to your quiet study space. This dashboard acts as a smart focus companion, keeping you alert and phone-free while you study.
                        </p>
                        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 20px;">
                            <div style="background: rgba(15, 23, 42, 0.6); padding: 22px; border-radius: 18px; border: 1px solid rgba(255,255,255,0.05);">
                                <div style="font-size: 1.6rem; margin-bottom: 10px;">👁️</div>
                                <h5 style="color: #f1f5f9; margin: 0 0 6px 0; font-weight: 600;">Drowsiness Alarm</h5>
                                <p style="color: #94a3b8; font-size: 0.85rem; margin: 0; line-height: 1.45;">Tracks eye closure duration and plays a gentle audio tone to keep you awake.</p>
                            </div>
                            <div style="background: rgba(15, 23, 42, 0.6); padding: 22px; border-radius: 18px; border: 1px solid rgba(255,255,255,0.05);">
                                <div style="font-size: 1.6rem; margin-bottom: 10px;">📱</div>
                                <h5 style="color: #f1f5f9; margin: 0 0 6px 0; font-weight: 600;">Phone Shield</h5>
                                <p style="color: #94a3b8; font-size: 0.85rem; margin: 0; line-height: 1.45;">Detects when your mobile phone wanders into the frame and overlays warning cues.</p>
                            </div>
                            <div style="background: rgba(15, 23, 42, 0.6); padding: 22px; border-radius: 18px; border: 1px solid rgba(255,255,255,0.05);">
                                <div style="font-size: 1.6rem; margin-bottom: 10px;">📈</div>
                                <h5 style="color: #f1f5f9; margin: 0 0 6px 0; font-weight: 600;">Habit Analytics</h5>
                                <p style="color: #94a3b8; font-size: 0.85rem; margin: 0; line-height: 1.45;">Saves session metrics to a secure local database, mapping your long-term focus habits.</p>
                            </div>
                        </div>
                        <div style="margin-top: 30px; font-size: 0.88rem; color: #94a3b8;">
                            💡 Configure your triggers on the sidebar and click <b>Start Session</b> above to begin.
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

    # ------------------ MODE 2: CLOUD WEBRTC STREAM ------------------
    elif camera_mode == "Cloud WebRTC (Browser)":
        st.subheader("Webcam Session Stream (WebRTC)")
        
        ctx = webrtc_streamer(
            key="study-monitor-webrtc",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=lambda: StudyVideoProcessor(yolo_model, face_mesh),
            media_stream_constraints={"video": True, "audio": False},
        )
        
        if ctx.state.playing:
            audio_ph = st.empty()
            # Sync sidebar settings dynamically to WebRTC thread
            if ctx.video_processor:
                ctx.video_processor.eye_ratio_threshold = eye_ratio_threshold
                ctx.video_processor.sleep_threshold = sleep_threshold_frames
                ctx.video_processor.phone_conf_threshold = phone_conf_threshold
                ctx.video_processor.enable_sleep = enable_sleep
                ctx.video_processor.enable_phone = enable_phone
                ctx.video_processor.enable_facehide = enable_facehide
                
                is_sleepy = ctx.video_processor.is_sleepy
                phone_detected = ctx.video_processor.phone_detected
                is_face_covered = ctx.video_processor.is_face_covered
            else:
                is_sleepy = False
                phone_detected = False
                is_face_covered = False
                
            # Initialize DB Session
            if st.session_state.session_id is None:
                st.session_state.session_id = database.start_session()
                st.session_state.total_sec = 0
                st.session_state.focus_sec = 0
                st.session_state.sleep_count = 0
                st.session_state.phone_count = 0
                st.session_state.cover_count = 0
                st.session_state.was_sleepy = False
                st.session_state.was_phone = False
                st.session_state.was_covered = False
                st.session_state.last_time = time.time()
                st.session_state.last_played_time = 0
                st.session_state.current_playing = None
                
            # Timer increment
            now = time.time()
            if now - st.session_state.last_time >= 1.0:
                st.session_state.total_sec += 1
                if not (is_sleepy or phone_detected or is_face_covered):
                    st.session_state.focus_sec += 1
                st.session_state.last_time = now
                
            # Tally transitions and log to DB
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
            
            # Client Audio Playback
            status_text = "FOCUSING..."
            status_text_color = "#34d399"
            distracted = False
            
            if is_face_covered:
                status_text = "FACE HIDDEN!"
                status_text_color = "#ef4444"
                distracted = True
                if enable_facehide and not mute_audio and audio_facehide_b64:
                    if st.session_state.current_playing != "facehide" or (now - st.session_state.last_played_time > 4.0):
                        st.session_state.current_playing = "facehide"
                        st.session_state.last_played_time = now
                        audio_ph.markdown(f'<audio autoplay src="data:audio/mp3;base64,{audio_facehide_b64}"></audio>', unsafe_allow_html=True)
            elif is_sleepy:
                status_text = "SLEEPING!"
                status_text_color = "#ef4444"
                distracted = True
                if enable_sleep and not mute_audio and audio_sleep_b64:
                    if st.session_state.current_playing != "sleep" or (now - st.session_state.last_played_time > 4.0):
                        st.session_state.current_playing = "sleep"
                        st.session_state.last_played_time = now
                        audio_ph.markdown(f'<audio autoplay src="data:audio/mp3;base64,{audio_sleep_b64}"></audio>', unsafe_allow_html=True)
            elif phone_detected:
                status_text = "PHONE DETECTED!"
                status_text_color = "#f97316"
                distracted = True
                if enable_phone and not mute_audio and audio_phone_b64:
                    if st.session_state.current_playing != "phone" or (now - st.session_state.last_played_time > 4.0):
                        st.session_state.current_playing = "phone"
                        st.session_state.last_played_time = now
                        audio_ph.markdown(f'<audio autoplay src="data:audio/mp3;base64,{audio_phone_b64}"></audio>', unsafe_allow_html=True)
                        
            if not distracted:
                st.session_state.current_playing = None
                audio_ph.empty()
                
            # Stats Panel Display
            tot_h = st.session_state.total_sec // 3600
            tot_m = (st.session_state.total_sec % 3600) // 60
            tot_s = st.session_state.total_sec % 60
            time_str = f"{tot_h:02d}:{tot_m:02d}:{tot_s:02d}"
            
            focus_pct = int((st.session_state.focus_sec / st.session_state.total_sec) * 100) if st.session_state.total_sec > 0 else 100
            distractions_tally = st.session_state.sleep_count + st.session_state.phone_count + st.session_state.cover_count
            
            metrics_html = f"""
            <div style="display: flex; gap: 15px; justify-content: space-between; margin-top: 20px; margin-bottom: 25px; flex-wrap: wrap;">
                <div class="metric-card-wrapper" style="flex: 1; min-width: 150px;">
                    <div style="font-size: 0.8rem; color: #94a3b8; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;">Timer ⏱️</div>
                    <div style="font-size: 1.8rem; font-weight: 700; color: #38bdf8; font-family: monospace; margin-top: 5px;">{time_str}</div>
                </div>
                <div class="metric-card-wrapper" style="flex: 1; min-width: 150px;">
                    <div style="font-size: 0.8rem; color: #94a3b8; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;">Focus Score 🎯</div>
                    <div style="font-size: 1.8rem; font-weight: 700; color: #34d399; margin-top: 5px;">{focus_pct}%</div>
                </div>
                <div class="metric-card-wrapper" style="flex: 1; min-width: 150px;">
                    <div style="font-size: 0.8rem; color: #94a3b8; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;">Status 🚨</div>
                    <div style="font-size: 1.6rem; font-weight: 700; color: {status_text_color}; margin-top: 5px;">{status_text}</div>
                </div>
                <div class="metric-card-wrapper" style="flex: 1; min-width: 150px;">
                    <div style="font-size: 0.8rem; color: #94a3b8; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;">Distractions ⚠️</div>
                    <div style="font-size: 1.8rem; font-weight: 700; color: #f87171; margin-top: 5px;">{distractions_tally}</div>
                </div>
            </div>
            """
            st.markdown(metrics_html, unsafe_allow_html=True)
            
            # Keep Streamlit running to update stats
            time.sleep(1.0)
            st.rerun()
        else:
            if st.session_state.session_id is not None:
                database.end_session(st.session_state.session_id, st.session_state.total_sec, st.session_state.focus_sec)
                st.session_state.session_id = None
                st.toast("Session successfully logged! Head to Analytics to review your stats.", icon="💾")
                st.rerun()
                
            st.markdown(
                """
                <div class="dashboard-banner">
                    <div style="position: relative; z-index: 1;">
                        <h3 style="margin-top: 0; color: #3b82f6; font-size: 1.6rem; font-weight: 700; letter-spacing: -0.02em;">Find your flow state. 🧠</h3>
                        <p style="color: #cbd5e1; font-size: 1rem; line-height: 1.6; max-width: 800px; margin-bottom: 25px;">
                            Welcome to your quiet study space. This dashboard acts as a smart focus companion, keeping you alert and phone-free while you study.
                        </p>
                        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 20px;">
                            <div style="background: rgba(15, 23, 42, 0.6); padding: 22px; border-radius: 18px; border: 1px solid rgba(255,255,255,0.05);">
                                <div style="font-size: 1.6rem; margin-bottom: 10px;">👁️</div>
                                <h5 style="color: #f1f5f9; margin: 0 0 6px 0; font-weight: 600;">Drowsiness Alarm</h5>
                                <p style="color: #94a3b8; font-size: 0.85rem; margin: 0; line-height: 1.45;">Tracks eye closure duration and plays a gentle audio tone to keep you awake.</p>
                            </div>
                            <div style="background: rgba(15, 23, 42, 0.6); padding: 22px; border-radius: 18px; border: 1px solid rgba(255,255,255,0.05);">
                                <div style="font-size: 1.6rem; margin-bottom: 10px;">📱</div>
                                <h5 style="color: #f1f5f9; margin: 0 0 6px 0; font-weight: 600;">Phone Shield</h5>
                                <p style="color: #94a3b8; font-size: 0.85rem; margin: 0; line-height: 1.45;">Detects when your mobile phone wanders into the frame and overlays warning cues.</p>
                            </div>
                            <div style="background: rgba(15, 23, 42, 0.6); padding: 22px; border-radius: 18px; border: 1px solid rgba(255,255,255,0.05);">
                                <div style="font-size: 1.6rem; margin-bottom: 10px;">📈</div>
                                <h5 style="color: #f1f5f9; margin: 0 0 6px 0; font-weight: 600;">Habit Analytics</h5>
                                <p style="color: #94a3b8; font-size: 0.85rem; margin: 0; line-height: 1.45;">Saves session metrics to a secure local database, mapping your long-term focus habits.</p>
                            </div>
                        </div>
                        <div style="margin-top: 30px; font-size: 0.88rem; color: #94a3b8;">
                            💡 Configure your triggers on the sidebar and click <b>Start</b> on the camera component above to begin.
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

# ----- TAB 2: ANALYTICS -----
with tabs[1]:
    st.subheader("Concentration & Productivity Reports")
    
    # Query database for session entries
    df_sessions = get_data("SELECT * FROM sessions")
    
    if df_sessions.empty:
        st.info("Your focus metrics will appear here once you complete a study session. Head over to the Focus Space tab to start your first session! 🚀")
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
                    <div style="font-size: 0.85rem; color: #94a3b8; font-weight: 600; text-transform: uppercase;">Total Sessions logged 📅</div>
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

# ----- TAB 3: HISTORY -----
with tabs[2]:
    st.subheader("Session History Log 📋")
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
