import cv2
from deepface import DeepFace
import numpy as np
from flask import Flask, render_template, redirect, Response, request, flash, url_for, jsonify
import requests
import json
import time
import os
import threading

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "secret123")

# ================= CONFIG =================
ACCOUNT_ID = os.getenv("ACCOUNT_ID")
DB_ID = os.getenv("DB_ID")
API_TOKEN = os.getenv("API_TOKEN")

session = requests.Session()

# ================= DB =================
def run_query(sql, params=None):
    url = f"https://api.cloudflare.com/client/v4/accounts/{ACCOUNT_ID}/d1/database/{DB_ID}/query"
    headers = {
        "Authorization": f"Bearer {API_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {"sql": sql}
    if params:
        payload["params"] = params
    try:
        res = session.post(url, headers=headers, json=payload, timeout=10)
        return res.json()
    except:
        return {"success": False}

# ================= INIT DB (runs in background) =================
def init_db():
    run_query("""CREATE TABLE IF NOT EXISTS admins (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        email TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL)""")
    run_query("""CREATE TABLE IF NOT EXISTS students (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL, class TEXT, image TEXT)""")
    run_query("""CREATE TABLE IF NOT EXISTS embeddings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        student_id INTEGER, student_name TEXT, embedding TEXT)""")
    run_query("""CREATE TABLE IF NOT EXISTS attendance (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        student_id INTEGER, session_id TEXT, timestamp TEXT, status TEXT)""")

    admin_email = os.getenv("ADMIN_EMAIL", "admin@school.com")
    admin_password = os.getenv("ADMIN_PASSWORD", "admin123")
    existing = run_query("SELECT * FROM admins WHERE email = ?", [admin_email])
    if existing.get("success"):
        results = existing.get("result", [{}])[0].get("results", [])
        if not results:
            run_query("INSERT INTO admins (email, password) VALUES (?, ?)",
                      [admin_email, admin_password])
            print(f"✅ Admin created: {admin_email}")

# ================= LOAD EMBEDDINGS =================
DB_EMB, DB_IDS, DB_NAMES = [], [], []

def load_embeddings():
    global DB_EMB, DB_IDS, DB_NAMES
    res = run_query("SELECT * FROM embeddings")
    if not res.get("success"):
        return
    rows = res["result"][0].get("results", [])
    embs, ids, names = [], [], []
    for r in rows:
        emb = np.array(json.loads(r["embedding"]))
        emb = emb / np.linalg.norm(emb)
        embs.append(emb)
        ids.append(r["student_id"])
        names.append(r["student_name"])
    DB_EMB, DB_IDS, DB_NAMES = embs, ids, names

# ================= VIDEO =================
latest_frame = None
processed_frame = None
cap = None

def init_video():
    global cap
    video_source = os.getenv("VIDEO_PATH", "0")
    try:
        video_source = int(video_source)
    except ValueError:
        pass
    cap = cv2.VideoCapture(video_source)

# ================= DETECTOR =================
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# ================= ATTENDANCE =================
attendance_buffer = []
marked_today = set()

def flush_attendance():
    while True:
        if attendance_buffer:
            batch = attendance_buffer.copy()
            attendance_buffer.clear()
            for student_id in batch:
                run_query(
                    "INSERT INTO attendance (student_id, session_id, timestamp, status) VALUES (?, ?, datetime('now'), ?)",
                    [student_id, "SESSION1", "Present"]
                )
        time.sleep(5)

# ================= CAPTURE =================
def capture_frames():
    global latest_frame
    while True:
        if cap and cap.isOpened():
            success, frame = cap.read()
            if success:
                latest_frame = frame
        time.sleep(0.01)

# ================= RECOGNITION =================
def recognize_face(face_img):
    try:
        emb = DeepFace.represent(img_path=face_img, model_name='Facenet', enforce_detection=False)
        embedding = np.array(emb[0]['embedding'])
        embedding = embedding / np.linalg.norm(embedding)
        min_dist = float('inf')
        best_name = "Unknown"
        best_id = None
        for i, db_emb in enumerate(DB_EMB):
            dist = np.linalg.norm(embedding - db_emb)
            if dist < min_dist:
                min_dist = dist
                best_name = DB_NAMES[i]
                best_id = DB_IDS[i]
        if min_dist < 0.6:
            return best_name, best_id
        elif min_dist < 0.75:
            return "Uncertain", None
        else:
            return "Unknown", None
    except:
        return "Error", None

# ================= PROCESS =================
def process_faces():
    global latest_frame, processed_frame
    frame_count = 0
    confidence_counter = {}
    while True:
        if latest_frame is None:
            time.sleep(0.01)
            continue
        frame = latest_frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        frame_count += 1
        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]
            name = "Detecting..."
            student_id = None
            if frame_count % 10 == 0:
                face_img_resized = cv2.resize(face_img, (160, 160))
                name, student_id = recognize_face(face_img_resized)
                if student_id:
                    confidence_counter[student_id] = confidence_counter.get(student_id, 0) + 1
                    if confidence_counter[student_id] >= 3 and student_id not in marked_today:
                        marked_today.add(student_id)
                        attendance_buffer.append(student_id)
            color = (0,0,255) if name == "Unknown" else (0,165,255) if name == "Uncertain" else (0,255,0)
            cv2.rectangle(frame, (x,y), (x+w,y+h), color, 2)
            cv2.putText(frame, name, (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        processed_frame = frame
        time.sleep(0.02)

# ================= STREAM =================
def generate_frames():
    while True:
        frame = processed_frame if processed_frame is not None else latest_frame
        if frame is None:
            time.sleep(0.01)
            continue
        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# ================= BACKGROUND STARTUP =================
def background_startup():
    """Run all heavy tasks AFTER Flask has started and bound the port"""
    time.sleep(3)  # wait for Flask to bind port first
    print("🔄 Running background startup...")
    init_db()
    load_embeddings()
    init_video()
    threading.Thread(target=capture_frames, daemon=True).start()
    threading.Thread(target=process_faces, daemon=True).start()
    threading.Thread(target=flush_attendance, daemon=True).start()
    print("✅ Background startup complete")

threading.Thread(target=background_startup, daemon=True).start()

# ================= ROUTES =================
@app.route('/')
def home():
    return render_template("login.html")

@app.route('/login', methods=['POST'])
def login():
    email = request.form.get('email')
    password = request.form.get('password')
    res = run_query("SELECT * FROM admins WHERE email = ? AND password = ?", [email, password])
    if res.get("success") and res["result"][0]["results"]:
        return redirect(url_for('dashboard'))
    else:
        flash("Invalid credentials")
        return redirect(url_for('home'))

@app.route('/dashboard')
def dashboard():
    students = run_query("SELECT * FROM students")["result"][0]["results"]
    attendance = run_query("SELECT * FROM attendance ORDER BY timestamp DESC")["result"][0]["results"]
    student_map = {s["id"]: s["name"] for s in students}
    for a in attendance:
        a["name"] = student_map.get(a["student_id"], "Unknown")
    total = len(students)
    present_ids = set([x["student_id"] for x in attendance if x["status"] == "Present"])
    return render_template("dashboard.html",
        total_students=total,
        present=len(present_ids),
        absent=total - len(present_ids),
        recent=attendance[:5])

@app.route('/add-student', methods=['GET','POST'])
def add_student():
    if request.method == 'POST':
        name = request.form.get('name')
        student_class = request.form.get('class')
        file = request.files['image']
        upload_folder = "static/uploads"
        os.makedirs(upload_folder, exist_ok=True)
        filename = str(int(time.time())) + "_" + file.filename
        upload_path = os.path.join(upload_folder, filename)
        file.save(upload_path)
        image_path = f"/static/uploads/{filename}"
        emb = DeepFace.represent(img_path=upload_path, model_name='Facenet', enforce_detection=False)
        vector = emb[0]['embedding']
        embedding_json = json.dumps(vector)
        res = run_query("INSERT INTO students (name, class, image) VALUES (?, ?, ?)",
                        [name, student_class, image_path])
        student_id = res["result"][0]["meta"]["last_row_id"]
        run_query("INSERT INTO embeddings (student_id, student_name, embedding) VALUES (?, ?, ?)",
                  [student_id, name, embedding_json])
        load_embeddings()
        return redirect(url_for('students'))
    return render_template("add_student.html")

@app.route('/students')
def students():
    students = run_query("SELECT * FROM students")["result"][0]["results"]
    return render_template("students.html", students=students)

@app.route('/attendance')
def attendance():
    records = run_query("SELECT * FROM attendance ORDER BY timestamp DESC")["result"][0]["results"]
    students_list = run_query("SELECT * FROM students")["result"][0]["results"]
    student_map = {s["id"]: s["name"] for s in students_list}
    for r in records:
        r["name"] = student_map.get(r["student_id"], "Unknown")
    total = len(students_list)
    present_ids = set([x["student_id"] for x in records if x["status"] == "Present"])
    return render_template("attendance.html",
        records=records, total=total,
        present=len(present_ids),
        absent=total - len(present_ids))

@app.route('/system-status')
def system_status():
    return jsonify({
        "camera": cap.isOpened() if cap else False,
        "model": True,
        "detection": latest_frame is not None
    })

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# ================= RUN =================
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
