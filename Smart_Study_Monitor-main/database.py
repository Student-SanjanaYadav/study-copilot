import sqlite3
import datetime
import os

DB_NAME = "study_monitor.db"

def get_db_connection():
    # Make sure DB is created in the same directory as database.py
    db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), DB_NAME)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Create sessions table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            start_time TEXT NOT NULL,
            end_time TEXT,
            total_duration_sec INTEGER DEFAULT 0,
            focus_duration_sec INTEGER DEFAULT 0
        )
    """)
    
    # Create distractions table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS distractions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER,
            timestamp TEXT NOT NULL,
            type TEXT NOT NULL,
            FOREIGN KEY (session_id) REFERENCES sessions (id) ON DELETE CASCADE
        )
    """)
    
    conn.commit()
    conn.close()

def start_session():
    init_db()  # Ensure tables exist
    conn = get_db_connection()
    cursor = conn.cursor()
    start_time = datetime.datetime.now().isoformat()
    cursor.execute(
        "INSERT INTO sessions (start_time) VALUES (?)",
        (start_time,)
    )
    session_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return session_id

def end_session(session_id, total_duration_sec, focus_duration_sec):
    conn = get_db_connection()
    cursor = conn.cursor()
    end_time = datetime.datetime.now().isoformat()
    cursor.execute(
        """
        UPDATE sessions 
        SET end_time = ?, total_duration_sec = ?, focus_duration_sec = ?
        WHERE id = ?
        """,
        (end_time, total_duration_sec, focus_duration_sec, session_id)
    )
    conn.commit()
    conn.close()

def log_distraction(session_id, distraction_type):
    if session_id is None:
        return
    conn = get_db_connection()
    cursor = conn.cursor()
    timestamp = datetime.datetime.now().isoformat()
    cursor.execute(
        "INSERT INTO distractions (session_id, timestamp, type) VALUES (?, ?, ?)",
        (session_id, timestamp, distraction_type)
    )
    conn.commit()
    conn.close()

def get_all_sessions():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM sessions ORDER BY id DESC")
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]

def get_session_details(session_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM sessions WHERE id = ?", (session_id,))
    session = cursor.fetchone()
    if session:
        session = dict(session)
        cursor.execute("SELECT * FROM distractions WHERE session_id = ? ORDER BY timestamp ASC", (session_id,))
        distractions = [dict(r) for r in cursor.fetchall()]
        session['distractions'] = distractions
    conn.close()
    return session

def get_all_distractions():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM distractions ORDER BY id DESC")
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]

