from flask import Flask, render_template, request, redirect, url_for, session,flash, jsonify, Response
from flask_mysqldb import MySQL
import MySQLdb.cursors
import re
import requests
import dateutil.parser
from flask_socketio import SocketIO, emit
import tensorflow as tf
import facenet

import pickle
import align.detect_face
import numpy as np
import cv2
from datetime import datetime

app = Flask(__name__)

# Change this to your secret key (can be anything, it's for extra protection)
app.secret_key = '1a2b3c4d5e6d7g8h9i10'

# Enter your database connection details below
app.config['MYSQL_HOST'] = 'bug0usy6grg9homkxqes-mysql.services.clever-cloud.com'
app.config['MYSQL_USER'] = 'uu42yrnrehb2xqsx'
app.config['MYSQL_PASSWORD'] = "99v5Iw2SYzHpHUiJG3Tr"
app.config['MYSQL_DB'] = 'bug0usy6grg9homkxqes'


# Intialize MySQL
mysql1 = MySQL(app)
socketio = SocketIO(app)
current_subject = None
last_id = None
time = None
ma_sinhvien_list = None
list_attendented = []
# http://localhost:5000/pythonlogin/ - this will be the login page, we need to use both GET and POST requests
@app.route('/pythonlogin/', methods=['GET', 'POST'])
def login():
# Output message if something goes wrong...
    # Check if "username" and "password" POST requests exist (user submitted form)
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form:
        # Create variables for easy access
        username = request.form['username']
        password = request.form['password']
        # Check if account exists using MySQL
        cursor = mysql1.connection.cursor(MySQLdb.cursors.DictCursor)
        print(username, password)
        cursor.execute('SELECT * FROM giang_vien WHERE email = %s AND mat_khau = %s', (username, password))
        # Fetch one record and return result
        account = cursor.fetchone()
                # If account exists in accounts table in out database
        if account:
            # Create session data, we can access this data in other routes
            session['loggedin'] = True
            session['id'] = account['id']
            session['email'] = account['email']
            # Redirect to home page
            return redirect(url_for('home'))
        else:
            # Account doesnt exist or username/password incorrect
            flash("Incorrect username/password!", "danger")
    return render_template('auth/login.html',title="Login")


# http://localhost:5000/pythonlogin/register 
# This will be the registration page, we need to use both GET and POST requests
@app.route('/pythonlogin/register', methods=['GET', 'POST'])
def register():
    # Check if "username", "password" and "email" POST requests exist (user submitted form)
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form and 'email' in request.form:
        # Create variables for easy access
        username = request.form['username']
        password = request.form['password']
        email = request.form['email']
                # Check if account exists using MySQL
        cursor = mysql1.connection.cursor(MySQLdb.cursors.DictCursor)
        # cursor.execute('SELECT * FROM accounts WHERE username = %s', (username))
        cursor.execute( "SELECT * FROM accounts WHERE username LIKE %s", [username] )
        account = cursor.fetchone()
        # If account exists show error and validation checks
        if account:
            flash("Account already exists!", "danger")
        elif not re.match(r'[^@]+@[^@]+\.[^@]+', email):
            flash("Invalid email address!", "danger")
        elif not re.match(r'[A-Za-z0-9]+', username):
            flash("Username must contain only characters and numbers!", "danger")
        elif not username or not password or not email:
            flash("Incorrect username/password!", "danger")
        else:
        # Account doesnt exists and the form data is valid, now insert new account into accounts table
            cursor.execute('INSERT INTO accounts VALUES (NULL, %s, %s, %s)', (username,email, password))
            mysql1.connection.commit()
            flash("You have successfully registered!", "success")
            return redirect(url_for('login'))

    elif request.method == 'POST':
        # Form is empty... (no POST data)
        flash("Please fill out the form!", "danger")
    # Show registration form with message (if any)
    return render_template('auth/register.html',title="Register")

# http://localhost:5000/pythinlogin/home 
# This will be the home page, only accessible for loggedin users

@app.route('/')
def home():
    # Check if user is loggedin
    if 'loggedin' in session:
        # User is loggedin show them the home page
        return render_template('home/home.html', email=session['email'],title="Home")
    # User is not loggedin redirect to login page
    return redirect(url_for('login'))


def initialize_attendance():
    global list_attendented  # Khai báo biến toàn cục
    list_attendented = []  # Set lại list_attended về danh sách rỗng



# API lấy danh sách học phần
@app.route('/api/subjects', methods=['GET'])
def get_subjects():
    try:
        cursor = mysql1.connection.cursor(MySQLdb.cursors.DictCursor)
        query = "SELECT * FROM subjects where  lecturer_id = %s"
        cursor.execute(query, (session['id'],))
        subjects = cursor.fetchall()
        return jsonify(subjects)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/profile')
def profile():
    # Check if user is loggedin
    if 'loggedin' in session:
        # User is loggedin show them the home page
        return render_template('auth/profile.html', email=session['email'],title="Profile")
    # User is not loggedin redirect to login page
    return redirect(url_for('login'))


@app.route('/diem_danh')
def diem_danh():
    # Check if user is loggedin
    if 'loggedin' in session:
        # User is loggedin show them the home page
        return render_template('diemdanh/index.html', email=session['email'],title="Điểm Danh")
    # User is not loggedin redirect to login page
    return redirect(url_for('login'))



@app.route('/thong_ke')
def thong_ke():
    # Check if user is loggedin
    if 'loggedin' in session:
        # User is loggedin show them the home page
        return render_template('thongke/index.html', email=session['email'],title="Thống Kê")
    # User is not loggedin redirect to login page
    return redirect(url_for('login'))


camera_url = "http://192.168.137.214/stream"

@app.route('/video_feed')
def video_feed():
    global list_student
    global list_attendented
    camera = cv2.VideoCapture(0)
    # print(current_subject)
    list_attendented = []
    def generate_frames():
        while True:
            # print(last_id)

            success, frame = camera.read()
            if not success:
                break
            else:
                list_attendented = []
                frame = process_frame(frame)
                # list_student = get_list_student(current_subject)
                # Encode frame thành JPEG
                _, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()

                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')




@app.route('/tick/<student_id>', methods=['GET'])
def tick_checkbox(student_id):
    # Khi hàm được gọi, gửi thông báo đến front-end
    socketio.emit('tick_checkbox', {'student_id': student_id})
    return f"Checkbox for student {student_id} will be ticked."


def send_tick_event(student_id, tick_state=True):
    if not student_id:
        raise ValueError("student_id is required")
    # Gửi sự kiện qua WebSocket
    socketio.emit('update_checkbox', {'student_id': student_id, 'tick_state': tick_state})
    return {'success': True, 'message': f'Checkbox {student_id} set to {tick_state}'}



@app.route('/api/attendance/<int:subject_id>', methods=['GET'])
def get_attendance(subject_id):
    cursor = mysql1.connection.cursor(MySQLdb.cursors.DictCursor)
    try:
        # Lấy danh sách buổi học của môn học
        cursor.execute("""
            SELECT idbuoi_hoc, Thoi_gian
            FROM tiet_hoc
            WHERE Ma_mon_hoc = %s
        """, (subject_id,))
        sessions = cursor.fetchall()

        if not sessions:
            return jsonify({"students": [], "sessions": []})

        # Lấy danh sách sinh viên của môn học
        query = """
                SELECT sv.MSSV AS ma_sinhvien, sv.ten_sinh_vien AS ten_sinhvien, sv.lop
                FROM sinhvien_hocphan shp
                JOIN sinh_vien sv ON shp.ma_sinh_vien = sv.MSSV
                WHERE shp.ma_hoc_phan = %s
                """
        cursor.execute(query, (subject_id,))  # Truyền subject_id vào truy vấn
        students = cursor.fetchall()

        # Lấy trạng thái điểm danh
        cursor.execute("""
            SELECT dd.ma_buoi_hoc, dd.ma_sinh_vien
            FROM diem_danh dd
            JOIN tiet_hoc th ON dd.ma_buoi_hoc = th.idbuoi_hoc
            WHERE th.Ma_mon_hoc = %s
        """, (subject_id,))
        diem_danh = cursor.fetchall()

        print(diem_danh)
        # Tạo cấu trúc điểm danh cho sinh viên
        attendance = {}
        for record in diem_danh:
            if record['ma_sinh_vien'] not in attendance:
                attendance[record['ma_sinh_vien']] = {}
            attendance[record['ma_sinh_vien']][record['ma_buoi_hoc']] = True

        # Kết hợp dữ liệu sinh viên và buổi học với trạng thái điểm danh
        for student in students:
            student['attendance'] = [
                attendance.get(student['ma_sinhvien'], {}).get(session['idbuoi_hoc'], False)
                for session in sessions
            ]

        print(students)
        print(sessions)
        # Trả về dữ liệu dưới dạng JSON
        return jsonify({
            "students": students,
            "sessions": [{"idbuoi_hoc": s['idbuoi_hoc'], "Thoi_gian": s['Thoi_gian'].isoformat()} for s in sessions]
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        cursor.close()






# Tải model Facenet (dạng TensorFlow)
def load_facenet_model(model_path):
    with tf.Graph().as_default():
        graph = tf.Graph()
        with graph.as_default():
            with tf.io.gfile.GFile(model_path, 'rb') as f:
                graph_def = tf.compat.v1.GraphDef()
                graph_def.ParseFromString(f.read())
                tf.import_graph_def(graph_def, name='')
        return graph


FACENET_MODEL_PATH = 'Models/20180402-114759.pb'
CLASSIFIER_PATH = 'Models/facemodel.pkl'
facenet_graph = load_facenet_model(FACENET_MODEL_PATH)
facenet_session = tf.compat.v1.Session(graph=facenet_graph)

# Tải model phân loại khuôn mặt
with open(CLASSIFIER_PATH, 'rb') as file:
    model, class_names = pickle.load(file)

def get_face_embedding(image, graph, session):
    with graph.as_default():
        images_placeholder = graph.get_tensor_by_name("input:0")
        embeddings = graph.get_tensor_by_name("embeddings:0")
        phase_train_placeholder = graph.get_tensor_by_name("phase_train:0")

        # Chuẩn hóa hình ảnh
        prewhitened = (image - np.mean(image)) / np.std(image)
        prewhitened = np.expand_dims(prewhitened, axis=0)

        # Tính embedding
        feed_dict = {images_placeholder: prewhitened, phase_train_placeholder: False}
        embedding = session.run(embeddings, feed_dict=feed_dict)
    return embedding
# Phát hiện và nhận diện khuôn mặt
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def process_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        resized_face = cv2.resize(face, (160, 160))

        # Tính vector nhúng
        embedding = get_face_embedding(resized_face, facenet_graph, facenet_session)

        # Dự đoán danh tính
        prediction = model.predict_proba(embedding)[0]
        name = class_names[np.argmax(prediction)]
        unknown = "unknown"
        confidence = np.max(prediction)
        print(list_attendented)
        print(ma_sinhvien_list)
        if  confidence > 0.85 :
        # Vẽ khung và hiển thị danh tính
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"{name} ({confidence:.2f})", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            if (name in ma_sinhvien_list ):
                if (name in list_attendented):
                    pass
                else:
                    list_attendented.append(name)
                    print(last_id, name)
                    send_tick_event(name, True)
                    attendent_sv(last_id,name)
                    response = requests.get("https://blynk.cloud/external/api/update?token=UiJZFmSeUTSkmS0vtK99MgagMJ2vg51A&V2=1")
                    response1 = requests.get("https://blynk.cloud/external/api/update?token=UiJZFmSeUTSkmS0vtK99MgagMJ2vg51A&V2=0")

                    # response = requests.get("https://blynk.cloud/external/api/update?token=UiJZFmSeUTSkmS0vtK99MgagMJ2vg51A&V2=1")
        else:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(frame, f"{unknown}  ({confidence:.2f})", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            # response = requests.get("https://blynk.cloud/external/api/update?token=UiJZFmSeUTSkmS0vtK99MgagMJ2vg51A&V2=0")
    return frame





@app.route('/api/students/<int:subject_id>', methods=['GET'])
def get_students(subject_id):
    try:
        # Tạo con trỏ cơ sở dữ liệu
        if mysql1 is None or mysql1.connection is None: raise Exception("Kết nối cơ sở dữ liệu không hợp lệ.")
        cursor = mysql1.connection.cursor(MySQLdb.cursors.DictCursor)


        # Truy vấn danh sách sinh viên từ bảng trung gian
        query = """
        SELECT sv.MSSV AS ma_sinhvien, sv.ten_sinh_vien AS ten_sinhvien, sv.lop
        FROM sinhvien_hocphan shp
        JOIN sinh_vien sv ON shp.ma_sinh_vien = sv.MSSV
        WHERE shp.ma_hoc_phan = %s
        """


        cursor.execute(query, (subject_id,))  # Truyền subject_id vào truy vấn
        # Lấy kết quả
        students = cursor.fetchall()
        global ma_sinhvien_list
        # Nếu không có sinh viên nào được tìm thấy
        if not students:
            return jsonify({'message': 'No students found for the given subject'}), 404
        ma_sinhvien_list = [student['ma_sinhvien'] for student in students]

        # Trả về danh sách sinh viên dưới dạng JSON
        return jsonify(students)
    except Exception as e:
        # Xử lý lỗi
        app.logger.error(f"Error fetching students for subject_id {subject_id}: {e}")
        return jsonify({'error': 'An error occurred while fetching students.'}), 500







@app.route('/start_attendance', methods=['POST'])
def start_attendance():
    global current_subject, current_camera_url, timestamp
    data = request.get_json()
    subject = data.get('subject')
    cameraURL = data.get('cameraURL')
    timestamp = data.get('timestamp')

    query = """
    SELECT sv.MSSV AS ma_sinhvien, sv.ten_sinh_vien AS ten_sinhvien, sv.lop
    FROM sinhvien_hocphan shp
    JOIN sinh_vien sv ON shp.ma_sinh_vien = sv.MSSV
    WHERE shp.ma_hoc_phan = %s
    """

    if not subject:
        return jsonify({'error': 'Subject is required'}), 400
    if not cameraURL:
        return jsonify({'error': 'Camera URL is required'}), 400

    # Lưu trữ giá trị subject và cameraURL vào các biến toàn cục
    current_subject = subject
    current_camera_url = cameraURL

    initialize_attendance()

    print(current_subject)
    # Lưu thông tin điểm danh vào cơ sở dữ liệu
    # insert_attendance_to_db(subject, timestamp, 'start')
    attendent(current_subject, timestamp)
    # Thực hiện các thao tác liên quan đến bắt đầu điểm danh

    return jsonify({'message': f'Starting attendance for subject: {subject}'}), 200

def attendent(current_subject, timestamp):
    with app.app_context():
        try:
            # Tạo con trỏ cơ sở dữ liệu
            cursor = mysql1.connection.cursor(MySQLdb.cursors.DictCursor)

            # Truy vấn chèn dữ liệu vào bảng tiet_hoc
            query = """
            INSERT INTO tiet_hoc (Thoi_gian, Ma_mon_hoc)
            VALUES (%s, %s);
            """
            global  last_id


            # Thực thi truy vấn với các giá trị
            thoi_gian = datetime.now()
            cursor.execute(query, (thoi_gian, current_subject))
            mysql1.connection.commit()  # Lưu thay đổi vào cơ sở dữ liệu
            last_id = cursor.lastrowid
            print("Dữ liệu đã được chèn thành công. attendent " )
        except Exception as e:
            print(f"Lỗi khi chèn dữ liệu: {e}")
        finally:
            cursor.close()


def attendent_sv(ma_buoi_hoc, ma_sinh_vien):
    with app.app_context():
        cursor1 = None  # Initialize cursor1 here to ensure it exists even in case of an error
        try:
            print
            if mysql1 is None or mysql1.connection is None:
                mysql1.connect()

            # Tạo con trỏ cơ sở dữ liệu
            cursor1 = mysql1.connection.cursor(MySQLdb.cursors.DictCursor)

            # Truy vấn chèn dữ liệu vào bảng diem_danh

            query = """
                       INSERT INTO diem_danh (ma_buoi_hoc, ma_sinh_vien, thoi_gian_diem_danh)
                       VALUES (%s, %s, %s);
                       """


            # Thực thi truy vấn với các giá trị
            thoi_gian = datetime.now()
            cursor1.execute(query, (ma_buoi_hoc, ma_sinh_vien, thoi_gian))
            mysql1.connection.commit()  # Lưu thay đổi vào cơ sở dữ liệu
            print("Dữ liệu đã được chèn thành công . attendent_sv")

        except Exception as e:
            print(f"Lỗi khi chèn dữ liệu: {e}")
        finally:
            if cursor1:
                cursor1.close()  # Đảm bảo rằng con trỏ được đóng sau khi sử dụng


@app.route('/stop_attendance', methods=['POST'])
def stop_attendance():
    global current_subject, current_camera_url
    data = request.get_json()
    subject = data.get('subject')
    cameraURL = data.get('cameraURL')
    timestamp = data.get('timestamp')

    if not subject:
        return jsonify({'error': 'Subject is required'}), 400
    if not cameraURL:
        return jsonify({'error': 'Camera URL is required'}), 400

    # Lưu thông tin điểm danh vào cơ sở dữ liệu
    # insert_attendance_to_db(subject, timestamp, 'stop')

    # Thực hiện các thao tác liên quan đến dừng điểm danh
    # print(f"Stopping attendance for subject: {subject} with camera URL: {cameraURL} at {timestamp}")

    # Xóa giá trị subject và cameraURL
    current_subject = None
    current_camera_url = None

    return jsonify({'message': f'Stopping attendance for subject: {subject}'}), 200










if __name__ =='__main__':
	app.run(debug=True)
