import tkinter as tk
import customtkinter as ctk
import queue
import time
from PIL import Image, ImageTk
import database
import monitor
import os
import sys

# Set up styling
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class SmartStudyApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Smart Study Monitor")
        self.root.geometry("1100x680")
        self.root.resizable(False, False)
        
        # Ensure database is initialized
        database.init_db()
        
        # State variables
        self.session_active = False
        self.session_paused = False
        self.session_id = None
        self.total_duration_sec = 0
        self.focus_duration_sec = 0
        self.last_timer_update = 0
        
        self.sleep_count = 0
        self.phone_count = 0
        self.cover_count = 0
        
        self.was_sleepy = False
        self.was_phone_detected = False
        self.was_face_covered = False
        
        # Thread & Queue
        self.frame_queue = queue.Queue(maxsize=5)
        self.monitor_thread = None
        
        # UI Elements Construction
        self.create_layout()
        
        # Handle close window
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Start GUI polling loop
        self.update_gui()
        
    def create_layout(self):
        # Configure grid layout
        self.root.grid_columnconfigure(0, weight=0) # Sidebar
        self.root.grid_columnconfigure(1, weight=1) # Video
        self.root.grid_columnconfigure(2, weight=0) # Settings
        self.root.grid_rowconfigure(0, weight=1)
        
        # ----------------- LEFT SIDEBAR -----------------
        self.sidebar_frame = ctk.CTkFrame(self.root, width=260, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.sidebar_frame.grid_propagate(False)
        
        # App Title
        self.title_label = ctk.CTkLabel(
            self.sidebar_frame, 
            text="STUDY MONITOR", 
            font=ctk.CTkFont(size=20, weight="bold")
        )
        self.title_label.pack(padx=20, pady=(20, 10))
        
        self.subtitle_label = ctk.CTkLabel(
            self.sidebar_frame, 
            text="Stay Focused & Productive", 
            font=ctk.CTkFont(size=12, slant="italic"),
            text_color="gray"
        )
        self.subtitle_label.pack(padx=20, pady=(0, 20))
        
        # Session Control Section
        self.session_label = ctk.CTkLabel(
            self.sidebar_frame, 
            text="STUDY SESSION", 
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="cyan"
        )
        self.session_label.pack(anchor="w", padx=20, pady=(10, 5))
        
        self.timer_label = ctk.CTkLabel(
            self.sidebar_frame, 
            text="00:00:00", 
            font=ctk.CTkFont(size=32, weight="bold", family="Courier")
        )
        self.timer_label.pack(padx=20, pady=10)
        
        self.start_btn = ctk.CTkButton(
            self.sidebar_frame, 
            text="Start Session", 
            command=self.start_session,
            fg_color="green",
            hover_color="darkgreen"
        )
        self.start_btn.pack(padx=20, pady=5, fill="x")
        
        self.pause_btn = ctk.CTkButton(
            self.sidebar_frame, 
            text="Pause Session", 
            command=self.pause_session,
            state="disabled"
        )
        self.pause_btn.pack(padx=20, pady=5, fill="x")
        
        self.stop_btn = ctk.CTkButton(
            self.sidebar_frame, 
            text="Stop Session", 
            command=self.stop_session,
            fg_color="red",
            hover_color="darkred",
            state="disabled"
        )
        self.stop_btn.pack(padx=20, pady=5, fill="x")
        
        # Separator
        self.sep = ctk.CTkFrame(self.sidebar_frame, height=2, fg_color="gray30")
        self.sep.pack(padx=20, pady=15, fill="x")
        
        # Stats Section
        self.stats_title = ctk.CTkLabel(
            self.sidebar_frame, 
            text="SESSION STATS", 
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="cyan"
        )
        self.stats_title.pack(anchor="w", padx=20, pady=(0, 5))
        
        self.focus_pct_label = ctk.CTkLabel(self.sidebar_frame, text="Focus Score: 100%")
        self.focus_pct_label.pack(anchor="w", padx=20, pady=2)
        
        self.focus_progressbar = ctk.CTkProgressBar(self.sidebar_frame)
        self.focus_progressbar.pack(padx=20, pady=5, fill="x")
        self.focus_progressbar.set(1.0)
        
        self.sleep_count_label = ctk.CTkLabel(self.sidebar_frame, text="Sleep Alerts: 0", text_color="pink")
        self.sleep_count_label.pack(anchor="w", padx=20, pady=2)
        
        self.phone_count_label = ctk.CTkLabel(self.sidebar_frame, text="Phone Alerts: 0", text_color="pink")
        self.phone_count_label.pack(anchor="w", padx=20, pady=2)
        
        self.cover_count_label = ctk.CTkLabel(self.sidebar_frame, text="Face Hidden Alerts: 0", text_color="pink")
        self.cover_count_label.pack(anchor="w", padx=20, pady=2)
        
        # ----------------- CENTER VIDEO CONTAINER -----------------
        self.center_frame = ctk.CTkFrame(self.root)
        self.center_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        
        self.status_indicator = ctk.CTkLabel(
            self.center_frame, 
            text="STATUS: READY TO STUDY", 
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color="lightgreen"
        )
        self.status_indicator.pack(pady=10)
        
        # Video label container
        self.video_container = ctk.CTkFrame(self.center_frame, fg_color="black")
        self.video_container.pack(padx=10, pady=10, expand=True, fill="both")
        
        self.video_label = tk.Label(self.video_container, bg="black")
        self.video_label.pack(expand=True, fill="both")
        
        # Initial black screen placeholder
        placeholder = Image.new("RGB", (640, 480), (15, 15, 15))
        imgtk = ImageTk.PhotoImage(image=placeholder)
        self.video_label.configure(image=imgtk)
        self.video_label.imgtk = imgtk
        
        # ----------------- RIGHT SETTINGS -----------------
        self.settings_frame = ctk.CTkFrame(self.root, width=260, corner_radius=0)
        self.settings_frame.grid(row=0, column=2, sticky="nsew", padx=10, pady=10)
        self.settings_frame.grid_propagate(False)
        
        self.settings_title = ctk.CTkLabel(
            self.settings_frame, 
            text="MONITOR SETTINGS", 
            font=ctk.CTkFont(size=16, weight="bold")
        )
        self.settings_title.pack(padx=20, pady=(20, 10))
        
        # Toggle Switches
        self.mute_switch = ctk.CTkSwitch(
            self.settings_frame, 
            text="Mute Alarms", 
            command=self.update_monitor_settings
        )
        self.mute_switch.pack(anchor="w", padx=20, pady=10)
        
        self.sleep_alarm_switch = ctk.CTkSwitch(
            self.settings_frame, 
            text="Enable Sleep Alarm", 
            command=self.update_monitor_settings
        )
        self.sleep_alarm_switch.pack(anchor="w", padx=20, pady=5)
        self.sleep_alarm_switch.select()
        
        self.phone_alarm_switch = ctk.CTkSwitch(
            self.settings_frame, 
            text="Enable Phone Alarm", 
            command=self.update_monitor_settings
        )
        self.phone_alarm_switch.pack(anchor="w", padx=20, pady=5)
        self.phone_alarm_switch.select()
        
        self.facehide_alarm_switch = ctk.CTkSwitch(
            self.settings_frame, 
            text="Enable Cover Alarm", 
            command=self.update_monitor_settings
        )
        self.facehide_alarm_switch.pack(anchor="w", padx=20, pady=5)
        self.facehide_alarm_switch.select()
        
        # Sliders
        self.eye_label = ctk.CTkLabel(self.settings_frame, text="Eye Ratio Threshold: 11.0")
        self.eye_label.pack(anchor="w", padx=20, pady=(15, 0))
        self.eye_slider = ctk.CTkSlider(
            self.settings_frame, 
            from_=5.0, 
            to=20.0, 
            number_of_steps=30,
            command=self.on_eye_slider
        )
        self.eye_slider.pack(padx=20, pady=5, fill="x")
        self.eye_slider.set(11.0)
        
        self.sleep_frames_label = ctk.CTkLabel(self.settings_frame, text="Sleep Sensitivity: 15 fr")
        self.sleep_frames_label.pack(anchor="w", padx=20, pady=(10, 0))
        self.sleep_frames_slider = ctk.CTkSlider(
            self.settings_frame, 
            from_=5, 
            to=50, 
            number_of_steps=45,
            command=self.on_sleep_frames_slider
        )
        self.sleep_frames_slider.pack(padx=20, pady=5, fill="x")
        self.sleep_frames_slider.set(15)
        
        self.phone_conf_label = ctk.CTkLabel(self.settings_frame, text="Phone Conf. Thresh: 50%")
        self.phone_conf_label.pack(anchor="w", padx=20, pady=(10, 0))
        self.phone_conf_slider = ctk.CTkSlider(
            self.settings_frame, 
            from_=30, 
            to=90, 
            number_of_steps=60,
            command=self.on_phone_conf_slider
        )
        self.phone_conf_slider.pack(padx=20, pady=5, fill="x")
        self.phone_conf_slider.set(50)
        
    def start_session(self):
        if not self.session_active:
            # Create session in database
            self.session_id = database.start_session()
            self.session_active = True
            self.session_paused = False
            self.total_duration_sec = 0
            self.focus_duration_sec = 0
            self.sleep_count = 0
            self.phone_count = 0
            self.cover_count = 0
            self.last_timer_update = time.time()
            self.session_start_time = time.time()
            
            # Start background thread
            self.monitor_thread = monitor.StudyMonitorThread(self.frame_queue, self.session_id)
            self.update_monitor_settings() # Sync settings
            self.monitor_thread.start()
            
            # UI Updates
            self.start_btn.configure(state="disabled", text="Running...")
            self.pause_btn.configure(state="normal", text="Pause Session")
            self.stop_btn.configure(state="normal")
            
            self.sleep_count_label.configure(text="Sleep Alerts: 0")
            self.phone_count_label.configure(text="Phone Alerts: 0")
            self.cover_count_label.configure(text="Face Hidden Alerts: 0")
            self.status_indicator.configure(text="STATUS: MONITORING STARTED", text_color="lightgreen")
            
        elif self.session_paused:
            self.session_paused = False
            self.monitor_thread.set_paused(False)
            self.last_timer_update = time.time()
            
            # UI Updates
            self.start_btn.configure(state="disabled", text="Running...")
            self.pause_btn.configure(text="Pause Session")
            self.status_indicator.configure(text="STATUS: MONITORING RESUMED", text_color="lightgreen")
            
    def pause_session(self):
        if self.session_active and not self.session_paused:
            self.session_paused = True
            self.monitor_thread.set_paused(True)
            
            # UI Updates
            self.start_btn.configure(state="normal", text="Resume Session")
            self.pause_btn.configure(text="Paused")
            self.status_indicator.configure(text="STATUS: PAUSED", text_color="yellow")
            
    def stop_session(self):
        if self.session_active:
            # Stop thread
            if self.monitor_thread:
                self.monitor_thread.stop()
                self.monitor_thread.join(timeout=1.0)
                self.monitor_thread = None
                
            # Log end session to DB
            database.end_session(self.session_id, self.total_duration_sec, self.focus_duration_sec)
            
            # State clean up
            self.session_active = False
            self.session_paused = False
            self.session_id = None
            
            # UI Updates
            self.start_btn.configure(state="normal", text="Start Session")
            self.pause_btn.configure(state="disabled", text="Pause Session")
            self.stop_btn.configure(state="disabled")
            self.status_indicator.configure(text="STATUS: SESSION COMPLETE", text_color="cyan")
            
            # Reset video feed placeholder
            placeholder = Image.new("RGB", (640, 480), (15, 15, 15))
            imgtk = ImageTk.PhotoImage(image=placeholder)
            self.video_label.configure(image=imgtk)
            self.video_label.imgtk = imgtk
            
    def on_eye_slider(self, val):
        self.eye_label.configure(text=f"Eye Ratio Threshold: {val:.1f}")
        self.update_monitor_settings()
        
    def on_sleep_frames_slider(self, val):
        self.sleep_frames_label.configure(text=f"Sleep Sensitivity: {int(val)} fr")
        self.update_monitor_settings()
        
    def on_phone_conf_slider(self, val):
        self.phone_conf_label.configure(text=f"Phone Conf. Thresh: {int(val)}%")
        self.update_monitor_settings()
        
    def update_monitor_settings(self):
        if self.monitor_thread:
            self.monitor_thread.mute_audio = self.mute_switch.get() == 1
            self.monitor_thread.enable_sleep_alarm = self.sleep_alarm_switch.get() == 1
            self.monitor_thread.enable_phone_alarm = self.phone_alarm_switch.get() == 1
            self.monitor_thread.enable_facehide_alarm = self.facehide_alarm_switch.get() == 1
            self.monitor_thread.eye_ratio_threshold = self.eye_slider.get()
            self.monitor_thread.sleep_threshold_frames = int(self.sleep_frames_slider.get())
            self.monitor_thread.phone_conf_threshold = self.phone_conf_slider.get() / 100.0

    def update_gui(self):
        try:
            # Retrieve all frames from queue
            while True:
                data = self.frame_queue.get_nowait()
                frame = data["frame"]
                is_sleepy = data["is_sleepy"]
                phone_detected = data["phone_detected"]
                is_face_covered = data["is_face_covered"]
                
                # Update transitions for counter
                if is_sleepy and not self.was_sleepy:
                    self.sleep_count += 1
                    self.sleep_count_label.configure(text=f"Sleep Alerts: {self.sleep_count}")
                if phone_detected and not self.was_phone_detected:
                    self.phone_count += 1
                    self.phone_count_label.configure(text=f"Phone Alerts: {self.phone_count}")
                if is_face_covered and not self.was_face_covered:
                    self.cover_count += 1
                    self.cover_count_label.configure(text=f"Face Hidden Alerts: {self.cover_count}")
                    
                self.was_sleepy = is_sleepy
                self.was_phone_detected = phone_detected
                self.was_face_covered = is_face_covered
                
                # Draw video frame
                img = Image.fromarray(frame)
                # Fit 640x480 container
                img = img.resize((640, 480), Image.Resampling.LANCZOS)
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)
                
                # Update warning status indicators
                if is_face_covered:
                    self.status_indicator.configure(text="STATUS: FACE HIDDEN", text_color="red")
                elif is_sleepy:
                    self.status_indicator.configure(text="STATUS: SLEEP DETECTED", text_color="red")
                elif phone_detected:
                    self.status_indicator.configure(text="STATUS: PHONE DETECTED", text_color="orange")
                else:
                    self.status_indicator.configure(text="STATUS: FOCUSING...", text_color="green")
        except queue.Empty:
            pass
            
        # Update session clock and focus percentage
        if self.session_active and not self.session_paused:
            now = time.time()
            if now - self.last_timer_update >= 1.0:
                self.total_duration_sec += 1
                if not (self.was_sleepy or self.was_phone_detected or self.was_face_covered):
                    self.focus_duration_sec += 1
                    
                self.timer_label.configure(text=self.format_time(self.total_duration_sec))
                
                focus_pct = int((self.focus_duration_sec / self.total_duration_sec) * 100) if self.total_duration_sec > 0 else 100
                self.focus_pct_label.configure(text=f"Focus Score: {focus_pct}%")
                self.focus_progressbar.set(focus_pct / 100.0)
                
                self.last_timer_update = now
                
        self.root.after(15, self.update_gui)
        
    def format_time(self, seconds):
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    def on_closing(self):
        if self.session_active:
            self.stop_session()
        self.root.destroy()
        sys.exit(0)

if __name__ == "__main__":
    root = ctk.CTk()
    app = SmartStudyApp(root)
    root.mainloop()
