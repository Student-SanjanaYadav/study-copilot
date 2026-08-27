import cv2
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector
from ultralytics import YOLO
import ctypes
import threading
import queue
import time
import os
import database

def play_mci_sound(alias, filepath):
    # Ensure any previous instances are closed
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

def is_mci_sound_playing(alias):
    buf = ctypes.create_unicode_buffer(128)
    ctypes.windll.winmm.mciSendStringW(f'status {alias} mode', buf, 128, 0)
    return buf.value.strip().lower() == 'playing'

class StudyMonitorThread(threading.Thread):
    def __init__(self, frame_queue, session_id=None):
        super().__init__()
        self.frame_queue = frame_queue
        self.session_id = session_id
        self.running = False
        self.paused = False
        
        # Thresholds (can be updated dynamically from GUI)
        self.eye_ratio_threshold = 11.0
        self.sleep_threshold_frames = 15
        self.cover_threshold_frames = 20
        self.phone_conf_threshold = 0.5
        
        # Audio & Alarm switches
        self.mute_audio = False
        self.enable_sleep_alarm = True
        self.enable_phone_alarm = True
        self.enable_facehide_alarm = True
        
        # Counters
        self.closed_frames = 0
        self.covered_frames = 0
        
        # State tracking (to prevent spamming the database)
        self.was_sleepy = False
        self.was_phone_detected = False
        self.was_face_covered = False
        
        # Audio state
        self.current_playing = None
        
        # Sound file paths
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.path_sleep = os.path.join(base_dir, "alarm.mp3")
        self.path_facehide = os.path.join(base_dir, "faudio.mp3")
        self.path_phone = os.path.join(base_dir, "paudio.mp3")
        
    def stop_all_sounds(self):
        stop_mci_sound("alarm_sleep")
        stop_mci_sound("alarm_facehide")
        stop_mci_sound("alarm_phone")
        self.current_playing = None
        
    def stop(self):
        self.running = False
        self.stop_all_sounds()
        
    def set_paused(self, paused):
        self.paused = paused
        if paused:
            self.stop_all_sounds()

    def run(self):
        self.running = True
        
        # OpenCV Camera
        cap = cv2.VideoCapture(0)
        
        # Detectors
        face_detector = FaceMeshDetector(maxFaces=1)
        base_dir = os.path.dirname(os.path.abspath(__file__))
        phone_detector = YOLO(os.path.join(base_dir, "yolov8n.pt"))
        classNames = phone_detector.names
        
        # Landmark Indices for Sleep Detection
        LEFT_EYE_TOP = 159
        LEFT_EYE_BOTTOM = 145
        FACE_LEFT = 130
        FACE_RIGHT = 243
        
        while self.running:
            if self.paused:
                time.sleep(0.1)
                continue
                
            success, img = cap.read()
            if not success:
                # If reading failed, sleep to avoid busy loop
                time.sleep(0.03)
                continue
                
            # Flip image horizontally for natural mirror view
            img = cv2.flip(img, 1)
            
            # Check if audio is busy
            is_audio_busy = False
            if self.current_playing:
                is_audio_busy = is_mci_sound_playing(self.current_playing)
                if not is_audio_busy:
                    self.current_playing = None
                
            # ----------------- 1. Face & Sleep Detection -----------------
            img, faces = face_detector.findFaceMesh(img, draw=False)
            
            is_sleepy = False
            is_face_covered = False
            
            if faces:
                self.covered_frames = 0
                face = faces[0]
                
                try:
                    eye_dist, _ = face_detector.findDistance(face[LEFT_EYE_TOP], face[LEFT_EYE_BOTTOM])
                    face_dist, _ = face_detector.findDistance(face[FACE_LEFT], face[FACE_RIGHT])
                    
                    # Eye Aspect Ratio calculation
                    ratio = (eye_dist / face_dist) * 100
                except Exception:
                    ratio = 15.0 # default normal eye aspect ratio if calculation fails
                    
                if ratio < self.eye_ratio_threshold:
                    self.closed_frames += 1
                else:
                    self.closed_frames = 0
                    
                if self.closed_frames >= self.sleep_threshold_frames:
                    is_sleepy = True
                    
                cvzone.putTextRect(img, f"Eye Ratio: {int(ratio)}", (30, 40), scale=1, thickness=1)
            else:
                self.closed_frames = 0
                self.covered_frames += 1
                if self.covered_frames >= self.cover_threshold_frames:
                    is_face_covered = True
                    
            # ----------------- 2. Phone Detection -----------------
            results = phone_detector.predict(img, stream=True, verbose=False)
            phone_detected = False
            
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    
                    if classNames[cls_id] == "cell phone" and conf > self.phone_conf_threshold:
                        phone_detected = True
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 2)
                        cvzone.putTextRect(img, f"Phone: {int(conf*100)}%", (x1, max(y1 - 10, 30)), scale=1, thickness=1, colorR=(255, 0, 255))
                        
            # ----------------- 3. Database Logging (Distraction Transitions) -----------------
            # We want to log when a state flips from False -> True
            if is_face_covered and not self.was_face_covered:
                database.log_distraction(self.session_id, "face_covered")
            if is_sleepy and not self.was_sleepy:
                database.log_distraction(self.session_id, "sleep")
            if phone_detected and not self.was_phone_detected:
                database.log_distraction(self.session_id, "phone")
                
            self.was_face_covered = is_face_covered
            self.was_sleepy = is_sleepy
            self.was_phone_detected = phone_detected
            
            # ----------------- 4. Alarm Sound Logic -----------------
            # Priority 1: Face covered warning
            if is_face_covered:
                cvzone.putTextRect(img, "DONT COVER YOUR FACE!", (50, 100), scale=2, thickness=3, colorR=(0, 0, 255))
                if self.enable_facehide_alarm and not self.mute_audio and not is_audio_busy:
                    play_mci_sound("alarm_facehide", self.path_facehide)
                    self.current_playing = "alarm_facehide"
                    
            # Priority 2: Sleep warning
            elif is_sleepy:
                cvzone.putTextRect(img, "WAKE UP & STUDY!", (50, 100), scale=2, thickness=3, colorR=(0, 0, 255))
                if self.enable_sleep_alarm and not self.mute_audio and not is_audio_busy:
                    play_mci_sound("alarm_sleep", self.path_sleep)
                    self.current_playing = "alarm_sleep"
                    
            # Priority 3: Phone warning
            elif phone_detected:
                cvzone.putTextRect(img, "PUT THE PHONE AWAY!", (50, 100), scale=2, thickness=3, colorR=(0, 165, 255))
                if self.enable_phone_alarm and not self.mute_audio and not is_audio_busy:
                    play_mci_sound("alarm_phone", self.path_phone)
                    self.current_playing = "alarm_phone"
                    
            # If audio is still finishing up, show current warning overlay
            elif is_audio_busy:
                if self.current_playing == 'alarm_facehide':
                    cvzone.putTextRect(img, "DONT COVER YOUR FACE!", (50, 100), scale=2, thickness=3, colorR=(0, 0, 255))
                elif self.current_playing == 'alarm_sleep':
                    cvzone.putTextRect(img, "WAKE UP & STUDY!", (50, 100), scale=2, thickness=3, colorR=(0, 0, 255))
                elif self.current_playing == 'alarm_phone':
                    cvzone.putTextRect(img, "PUT THE PHONE AWAY!", (50, 100), scale=2, thickness=3, colorR=(0, 165, 255))
                    
            # Put the processed frame & status into queue
            try:
                # Keep queue small (e.g. maxsize=2) to avoid lag.
                # If queue is full, remove old item and insert new.
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                self.frame_queue.put_nowait({
                    "frame": cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                    "is_sleepy": is_sleepy,
                    "phone_detected": phone_detected,
                    "is_face_covered": is_face_covered
                })
            except queue.Full:
                pass
                
            time.sleep(0.01) # slight yield
            
        cap.release()
        self.stop_all_sounds()
