import cv2
from deepface import DeepFace
import numpy as np
from flask import Flask, render_template, redirect, Response, request, flash, url_for
import requests
import json
import time
import dotenv
import os

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY")

ACCOUNT_ID = os.getenv("ACCOUNT_ID")
DB_ID = os.getenv("DB_ID")
API_TOKEN = os.getenv("API_TOKEN")

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

    res = requests.post(url, headers=headers, json=payload)
    return res.json()

# ================= LOAD EMBEDDINGS =================

def load_embeddings():
    res = run_query("SELECT * FROM embeddings")

    if not res.get("success"):
        return [], [], []

    rows = res["result"][0].get("results", [])

    db_embeddings, db_ids, db_names = [], [], []

    for r in rows:
        emb = np.array(json.loads(r["embedding"]))
        emb = emb / np.linalg.norm(emb)

        db_embeddings.append(emb)
        db_ids.append(r["student_id"])
        db_names.append(r["student_name"])

    return db_embeddings, db_ids, db_names

DB_EMB, DB_IDS, DB_NAMES = load_embeddings()

# ================= ROUTES =================

@app.route('/')
def home():
    return render_template("login.html")

@app.route('/login', methods=['POST'])
def login():
    email = request.form.get('email')
    password = request.form.get('password')

    res = run_query(
        "SELECT * FROM admins WHERE email = ? AND password = ?",
        [email, password]
    )

    if not res.get("success"):
        flash("Database error")
        return redirect(url_for('home'))

    rows = res["result"][0]["results"]

    if rows:
        return redirect(url_for('dashboard'))
    else:
        flash("Invalid credentials")
        return redirect(url_for('home'))

@app.route('/dashboard', methods=['GET'])
def dashboard():
    students = run_query("SELECT * FROM students")["result"][0]["results"]
    attendance = run_query("SELECT * FROM attendance")["result"][0]["results"]

    total = len(students)
    present_ids = set([x["student_id"] for x in attendance if x["status"] == "Present"])

    return render_template(
        "dashboard.html",
        total_students=total,
        present=len(present_ids),
        absent=total - len(present_ids)
    )

@app.route('/add-student', methods=['GET', 'POST'])
def add_student():
    if request.method == 'POST':
        name = request.form.get('name')
        student_class = request.form.get('class')
        file = request.files['image']

        path = os.path.join("static/uploads", file.filename)
        file.save(path)

        emb = DeepFace.represent(img_path=path, model_name='Facenet', enforce_detection=False)
        vector = emb[0]['embedding']

        embedding_json = json.dumps(vector)

        res = run_query("INSERT INTO students (name,class) VALUES (?,?)", [name, student_class])
        student_id = res["result"][0]["meta"]["last_row_id"]

        run_query(
            "INSERT INTO embeddings (student_id, student_name, embedding) VALUES (?, ?, ?)",
            [student_id, name, embedding_json]
        )

        global DB_EMB, DB_IDS, DB_NAMES
        DB_EMB, DB_IDS, DB_NAMES = load_embeddings()

        return redirect(url_for('dashboard'))

    return render_template("add_student.html")

@app.route('/students')
def students():
    res = run_query("SELECT * FROM students")
    students = res["result"][0]["results"]
    return render_template("students.html", students=students)

# ================= VIDEO =================

video_source = r"C:\Users\Jeet Thakkar\Downloads\demo.mp4"
marked_today = set()

def generate_frames():
    cap = cv2.VideoCapture(video_source)

    while True:
        success, frame = cap.read()
        if not success:
            break

        faces = DeepFace.extract_faces(
            img_path=frame,
            detector_backend="retinaface",
            enforce_detection=False
        )

        for face in faces:
            area = face['facial_area']
            x, y, w, h = area['x'], area['y'], area['w'], area['h']

            face_img = face['face']
            face_img = (face_img * 255).astype(np.uint8)
            face_img = cv2.resize(face_img, (160,160))

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
                label = best_name
                color = (0,255,0)

                if best_id not in marked_today:
                    marked_today.add(best_id)

                    run_query(
                        "INSERT INTO attendance (student_id, session_id, timestamp, status) VALUES (?, ?, datetime('now'), ?)",
                        [best_id, "SESSION1", "Present"]
                    )

            else:
                label = "Unknown"
                color = (0,0,255)

                send_to_n8n({
                    "type": "unknown"
                })

            cv2.rectangle(frame,(x,y),(x+w,y+h),color,2)
            cv2.putText(frame,label,(x,y-10),
                        cv2.FONT_HERSHEY_SIMPLEX,0.8,color,2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        time.sleep(0.03)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# ================= RUN =================

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=1232, debug=True)
