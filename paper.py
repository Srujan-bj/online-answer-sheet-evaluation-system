from flask import Flask, request, jsonify
from flask_cors import CORS
import easyocr
import os
from werkzeug.utils import secure_filename
from deepface import DeepFace
import cv2
import numpy as np

app = Flask(__name__)
CORS(app)

reader = easyocr.Reader(['en'])

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join("uploads", filename)
        os.makedirs("uploads", exist_ok=True)
        file.save(filepath)

        result = reader.readtext(filepath, detail=0)  # return only text
        os.remove(filepath)  # optional cleanup

        return jsonify({"text": result})
    except Exception as e:
        print("OCR Error:", e)
        return jsonify({"error": str(e)}), 500

@app.route('/detect_emotion', methods=['POST'])
def detect_emotion():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    npimg = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    try:
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        emotion = result[0]['dominant_emotion']
        print(f"Detected emotion: {emotion}")   
        return jsonify({'emotion': emotion})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
