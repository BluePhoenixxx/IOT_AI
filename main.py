from flask import Flask, render_template, request, redirect, url_for, session,flash, jsonify, Response
from flask_mysqldb import MySQL
import MySQLdb.cursors
import re


import tensorflow as tf
import facenet

import pickle
import align.detect_face
import numpy as np
import cv2


app = Flask(__name__)

# Change this to your secret key (can be anything, it's for extra protection)
app.secret_key = '1a2b3c4d5e6d7g8h9i10'

# Enter your database connection details below
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = "Bian1907@" #Replace ******* with  your database password.
app.config['MYSQL_DB'] = 'quanlydiemdanh'


# Intialize MySQL
mysql = MySQL(app)


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
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
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
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
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
            mysql.connection.commit()
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


# API lấy danh sách học phần
@app.route('/api/subjects', methods=['GET'])
def get_subjects():
    try:
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute("SELECT id, name FROM subjects")
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
        return render_template('diemdanh/index.html', email=session['email'],title="Profile")
    # User is not loggedin redirect to login page
    return redirect(url_for('login'))



@app.route('/thong_ke')
def thong_ke():
    # Check if user is loggedin
    if 'loggedin' in session:
        # User is loggedin show them the home page
        return render_template('auth/profile.html', email=session['email'],title="Profile")
    # User is not loggedin redirect to login page
    return redirect(url_for('login'))


camera_url = "http://192.168.137.111/stream"

@app.route('/video_feed')
def video_feed():
    camera = cv2.VideoCapture(0)

    def generate_frames():
        while True:
            success, frame = camera.read()
            if not success:
                break
            else:
                frame = process_frame(frame)

                # Encode frame thành JPEG
                _, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()

                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


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
        confidence = np.max(prediction)

        # Vẽ khung và hiển thị danh tính
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f"{name} ({confidence:.2f})", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    return frame




# def run_face_recognition(detected_faces):
#     """
#     Continuously process video frames, perform face detection and recognition,
#     and yield frames as JPEGs for streaming via Flask.
#
#     Parameters:
#     detected_faces (list): A shared list to store names of detected faces.
#     """
#     # Load the pre-trained FaceNet model and the classifier
#     print("Loading FaceNet model...")
#     facenet.load_model('Models/20180402-114759.pb')
#
#     # Load the classifier and class names
#     with open('Models/facemodel.pkl', 'rb') as file:
#         model, class_names = pickle.load(file)
#     print("Loaded classifier and class names")
#
#     # Initialize camera feed
#     video_capture = cv2.VideoCapture(0)  # Use webcam; replace with your camera index or URL if needed
#
#     # MTCNN for face detection
#     pnet, rnet, onet = align.detect_face.create_mtcnn(tf.compat.v1.Session(), "src/align")
#
#     # Parameters for MTCNN
#     minsize = 20  # Minimum size of the face
#     threshold = [0.6, 0.7, 0.7]  # Three-step threshold
#     factor = 0.709  # Scale factor
#
#     while True:
#         ret, frame = video_capture.read()
#         if not ret:
#             print("Failed to capture frame")
#             break
#
#         # Resize frame for faster processing
#         frame = imutils.resize(frame, width=600)
#         rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB for face detection
#
#         # Detect faces in the frame
#         bounding_boxes, _ = align.detect_face.detect_face(
#             rgb_frame, minsize, pnet, rnet, onet, threshold, factor
#         )
#
#         faces_found = bounding_boxes.shape[0]
#         if faces_found > 0:
#             for i in range(faces_found):
#                 # Extract bounding box
#                 x1, y1, x2, y2 = map(int, bounding_boxes[i][:4])
#
#                 # Crop and preprocess the face
#                 cropped_face = frame[y1:y2, x1:x2]
#                 if cropped_face.size == 0:
#                     continue
#                 resized_face = cv2.resize(cropped_face, (160, 160))  # Resize to FaceNet input size
#                 prewhitened_face = facenet.prewhiten(resized_face)  # Normalize the image
#                 face_array = prewhitened_face.reshape(1, 160, 160, 3)  # Reshape for the model
#
#                 # Perform face embedding
#                 feed_dict = {
#                     tf.compat.v1.get_default_graph().get_tensor_by_name("input:0"): face_array,
#                     tf.compat.v1.get_default_graph().get_tensor_by_name("phase_train:0"): False,
#                 }
#                 embeddings = tf.compat.v1.get_default_graph().get_tensor_by_name("embeddings:0")
#                 face_embedding = sess.run(embeddings, feed_dict=feed_dict)
#
#                 # Perform classification
#                 predictions = model.predict_proba(face_embedding)
#                 best_class_index = np.argmax(predictions, axis=1)[0]
#                 best_class_probability = predictions[0][best_class_index]
#
#                 # Check if confidence is above the threshold
#                 if best_class_probability > 0.7:
#                     name = class_names[best_class_index]
#                     detected_faces.append(name)  # Add name to the shared list
#                     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw green box
#                     cv2.putText(frame, f"{name} ({best_class_probability:.2f})", (x1, y2 + 20),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
#                 else:
#                     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Draw red box
#                     cv2.putText(frame, "Unknown", (x1, y2 + 20),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
#
#         # Encode the frame as JPEG
#         ret, jpeg = cv2.imencode('.jpg', frame)
#         if not ret:
#             print("Failed to encode frame")
#             continue
#
#         # Yield the frame for Flask video streaming
#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
#
#     # Release the video capture when done
#     video_capture.release()
#     print("Stopped video capture")

@app.route('/api/students/<int:subject_id>', methods=['GET'])
def get_students(subject_id):
    try:
        # Tạo con trỏ cơ sở dữ liệu
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)

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

        # Nếu không có sinh viên nào được tìm thấy
        if not students:
            return jsonify({'message': 'No students found for the given subject'}), 404

        # Trả về danh sách sinh viên dưới dạng JSON
        return jsonify(students)
    except Exception as e:
        # Xử lý lỗi
        app.logger.error(f"Error fetching students for subject_id {subject_id}: {e}")
        return jsonify({'error': 'An error occurred while fetching students.'}), 500







if __name__ =='__main__':
	app.run(debug=True)
