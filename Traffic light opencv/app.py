from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np

app = Flask(__name__)


# Function to detect traffic lights and return color information
def detect_traffic_lights_and_color(frame):
    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define color ranges for red, yellow, and green in HSV
    lower_red = np.array([160, 100, 100])
    upper_red = np.array([180, 255, 255])

    lower_yellow = np.array([25, 50, 100])
    upper_yellow = np.array([35, 255, 255])

    lower_green = np.array([40, 50, 50])
    upper_green = np.array([80, 255, 255])

    # Create masks for each color
    red_mask = cv2.inRange(hsv, lower_red, upper_red)
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    green_mask = cv2.inRange(hsv, lower_green, upper_green)

    # Find the color using the mask
    color_detected = None
    if cv2.countNonZero(red_mask) > 0:
        color_detected = "Red...Stop"
    elif cv2.countNonZero(yellow_mask) > 0:
        color_detected = "Yellow...Prepare to go"
    elif cv2.countNonZero(green_mask) > 0:
        color_detected = "Green...Go"
    else:
        color_detected = "No color detected"

    return color_detected


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    if request.method == 'POST':
        # Check if a file was uploaded
        if 'file' not in request.files:
            return jsonify({'result': 'No file uploaded'})

        file = request.files['file']

        # Check if the file has an allowed extension
        if file.filename == '':
            return jsonify({'result': 'No selected file'})

        if file:
            # Save the uploaded file
            filename = 'uploaded_image.jpg'
            file.save(filename)

            # Read the uploaded image
            frame = cv2.imread(filename)

            # Detect traffic lights and determine color
            result = detect_traffic_lights_and_color(frame)

            return jsonify({'result': result})


if __name__ == "__main__":
    app.run(debug=True)
