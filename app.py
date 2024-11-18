from flask import Flask, render_template, request, jsonify, Response
import cv2
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

age_model = load_model('models/age_model.h5')
gender_model = load_model('models/gender_model.h5')

gender_labels = ['Male', 'Female']

def preprocess_for_age_model(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized_img = cv2.resize(gray_img, (200, 200))
    normalized_img = resized_img / 255.0
    return np.expand_dims(normalized_img, axis=(0, -1))


# Preprocessing for gender model (expects RGB)
def preprocess_for_gender_model(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized_img = cv2.resize(gray_img, (200, 200))
    normalized_img = resized_img / 255.0
    return np.expand_dims(normalized_img, axis=(0, -1))

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    # Read the uploaded image
    file_stream = file.stream
    np_img = np.frombuffer(file_stream.read(), np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    
    # Preprocess for each model
    age_input = preprocess_for_age_model(img)
    gender_input = preprocess_for_gender_model(img)
    
    # Predict age and gender
    age = age_model.predict(age_input).flatten()[0]
    gender = gender_labels[np.argmax(gender_model.predict(gender_input))]
    
    return jsonify({'age': int(age), 'gender': gender})


# Process a single frame for age and gender predictions
def process_frame(frame):
    age_input = preprocess_for_age_model(frame)
    gender_input = preprocess_for_gender_model(frame)
    
    # Predict age and gender
    age = age_model.predict(age_input).flatten()[0]
    gender = gender_labels[np.argmax(gender_model.predict(gender_input))]
    return age, gender


# Route for the home page
@app.route('/')
def index():
    return render_template('index.html')


# Route for camera view page
@app.route('/camera')
def camera():
    return render_template('camera.html')


# Route for video feed
@app.route('/video_feed')
def video_feed():
    def generate():
        cap = cv2.VideoCapture(0)  # Open the webcam
        if not cap.isOpened():
            raise RuntimeError("Could not start video capture")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            age, gender = process_frame(frame)
            
            cv2.putText(frame, f'Age: {int(age)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f'Gender: {gender}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            _, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        cap.release()
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)

# from flask import Flask, render_template, request, jsonify, Response
# import cv2
# import numpy as np
# from tensorflow.keras.models import load_model

# app = Flask(__name__)

# # Load the models
# age_model = load_model('models/age_model.h5')
# gender_model = load_model('models/gender_model.h5')

# gender_labels = ['Male', 'Female']

# # Preprocess image for age prediction
# def preprocess_for_age_model(img):
#     resized_img = cv2.resize(img, (200, 200))  # Match model input size
#     normalized_img = resized_img / 255.0  # Normalize to [0, 1]
#     return np.expand_dims(normalized_img, axis=0)  # Add batch dimension

# # Preprocess image for gender prediction
# def preprocess_for_gender_model(img):
#     resized_img = cv2.resize(img, (200, 200))  # Match model input size
#     normalized_img = resized_img / 255.0  # Normalize to [0, 1]
#     return np.expand_dims(normalized_img, axis=0)  # Add batch dimension

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/upload', methods=['POST'])
# def upload():
#     if 'file' not in request.files:
#         return jsonify({'error': 'No file uploaded'})
    
#     file = request.files['file']
#     if file.filename == '':
#         return jsonify({'error': 'No selected file'})
    
#     # Read the uploaded image
#     file_stream = file.stream
#     np_img = np.frombuffer(file_stream.read(), np.uint8)
#     img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

#     # Preprocess image for models
#     age_input = preprocess_for_age_model(img)
#     gender_input = preprocess_for_gender_model(img)

#     # Predict age and gender
#     age = age_model.predict(age_input).flatten()[0] + 10  # Added bias of 10 for testing
#     gender = gender_labels[np.argmax(gender_model.predict(gender_input))]

#     # Render the result page with predictions
#     return render_template('result.html', age=int(age), gender=gender)

# # Route for camera predictions (optional, kept for completeness)
# @app.route('/camera')
# def camera():
#     return render_template('camera.html')

# @app.route('/video_feed')
# def video_feed():
#     def generate():
#         cap = cv2.VideoCapture(0)
#         if not cap.isOpened():
#             raise RuntimeError("Could not start video capture")
        
#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 break
            
#             age_input = preprocess_for_age_model(frame)
#             gender_input = preprocess_for_gender_model(frame)

#             age = age_model.predict(age_input).flatten()[0]
#             gender = gender_labels[np.argmax(gender_model.predict(gender_input))]

#             cv2.putText(frame, f'Age: {int(age)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#             cv2.putText(frame, f'Gender: {gender}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

#             _, buffer = cv2.imencode('.jpg', frame)
#             yield (b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
#         cap.release()
    
#     return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

# if __name__ == '__main__':
#     app.run(debug=True)
