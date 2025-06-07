from flask import Flask, render_template, request, redirect, url_for, session, jsonify, flash
import os
import sqlite3
import secrets
import smtplib
from email.mime.text import MIMEText
from PIL import Image
from werkzeug.security import generate_password_hash, check_password_hash
from ultralytics import YOLO
import numpy as np
import cv2
import uuid

# === CONFIG ===
app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY')  # Set this securely in production

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

DATABASE = 'database/users.db'

EMAIL_SENDER = os.getenv('EMAIL_SENDER')
EMAIL_PASSWORD = os.getenv('EMAIL_PASSWORD')

# === YOLOv9 MODEL SETUP ===
MODEL_PATH = "/Users/hardikm-visiobyte/Desktop/Oral_Cancer_Ampire/yolo_project/static/model_file/best.pt"
model = YOLO(MODEL_PATH)

# === DB Setup ===
def init_db():
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            is_verified INTEGER DEFAULT 0,
            otp TEXT
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# === Helper Function ===
def send_otp_email(receiver_email, otp):
    message = MIMEText(f"Your OTP code is: {otp}")
    message['Subject'] = 'Your OTP for Oral Cancer Detection Portal'
    message['From'] = EMAIL_SENDER
    message['To'] = receiver_email

    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
        server.login(EMAIL_SENDER, EMAIL_PASSWORD)
        server.send_message(message)

# === Routes ===

@app.route('/')
def home():
    if not session.get('user_id'):
        return redirect(url_for('login'))
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form['email']
        password = generate_password_hash(request.form['password'])
        otp = secrets.token_hex(3)  # 6-digit OTP

        conn = sqlite3.connect(DATABASE)
        c = conn.cursor()
        try:
            c.execute('INSERT INTO users (email, password, otp) VALUES (?, ?, ?)', (email, password, otp))
            conn.commit()
            send_otp_email(email, otp)
            session['email_temp'] = email
            flash('OTP sent to your email!', 'info')
            return redirect(url_for('verify_otp'))
        except sqlite3.IntegrityError:
            flash('Email already exists.', 'danger')
        finally:
            conn.close()
    return render_template('register.html')

@app.route('/verify', methods=['GET', 'POST'])
def verify_otp():
    if request.method == 'POST':
        user_otp = request.form['otp']
        email = session.get('email_temp')

        conn = sqlite3.connect(DATABASE)
        c = conn.cursor()
        c.execute('SELECT otp FROM users WHERE email=?', (email,))
        real_otp = c.fetchone()
        if real_otp and real_otp[0] == user_otp:
            c.execute('UPDATE users SET is_verified=1, otp=NULL WHERE email=?', (email,))
            conn.commit()
            flash('Email verified! You can now log in.', 'success')
            return redirect(url_for('login'))
        else:
            flash('Invalid OTP!', 'danger')
        conn.close()
    return render_template('verify_otp.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        conn = sqlite3.connect(DATABASE)
        c = conn.cursor()
        c.execute('SELECT id, password, is_verified FROM users WHERE email=?', (email,))
        user = c.fetchone()
        conn.close()

        if user and check_password_hash(user[1], password):
            if user[2] == 1:
                session['user_id'] = user[0]
                return redirect(url_for('home'))
            else:
                flash('Please verify your email first.', 'warning')
        else:
            flash('Incorrect credentials.', 'danger')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

model = YOLO("/Users/hardikm-visiobyte/Desktop/Oral_Cancer_Ampire/yolo_project/static/model_file/best.pt")  # replace with your actual model path
# model.names = {0: 'not cancer', 1: 'cancer'}  # explicitly set class names

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    print("In /predict route")

    file = request.files['image']
    img_bytes = file.read()
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    results = model(img)
    detections = results[0].boxes.data.cpu().numpy()

    output = []
    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        output.append({
            'class_id': int(cls),
            'confidence': float(conf),
            'bbox': [int(x1), int(y1), int(x2), int(y2)],
            'label': model.names[int(cls)]
        })

    # Handle no detections
    if not output:
        return jsonify({
            "label": "No cancer detected",
            "confidence": 0,
            "image_url": ""
        })

    # Get the highest confidence result
    top = max(output, key=lambda x: x["confidence"])

    # Draw all detections on the image
    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        label = f"{model.names[int(cls)]} {conf:.2f}"
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
        cv2.putText(img, label, (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Save result image
    filename = f"pred_{uuid.uuid4().hex[:8]}.jpg"
    filepath = os.path.join("static", filename)
    os.makedirs("static", exist_ok=True)
    cv2.imwrite(filepath, img)

    return jsonify({
        "label": top["label"],  # will be 'cancer' or 'not cancer'
        "confidence": int(top["confidence"] * 100),
        "image_url": f"/static/{filename}"
    })


if __name__ == '__main__':
    app.run(debug=True)
