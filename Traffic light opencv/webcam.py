import cv2
import numpy as np

def detect_traffic_light(image):
    # Convert image to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define range of colors in HSV
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])
    lower_green = np.array([36, 0, 0])
    upper_green = np.array([86, 255, 255])

    # Threshold the HSV image to get masks for each color
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    # Combine masks to detect red, yellow, and green
    mask_combined = cv2.bitwise_or(mask_red, mask_yellow)
    mask_combined = cv2.bitwise_or(mask_combined, mask_green)

    # Find contours in the combined mask
    contours, _ = cv2.findContours(mask_combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Iterate through contours to draw rectangles around detected colors
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 100:  # Adjust threshold as needed
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Check if any color is detected
    red_detected = cv2.countNonZero(mask_red) > 0
    yellow_detected = cv2.countNonZero(mask_yellow) > 0
    green_detected = cv2.countNonZero(mask_green) > 0

    if red_detected:
        return 'Red light detected! Stop!'
    elif yellow_detected:
        return 'Yellow light detected! Prepare to stop!'
    elif green_detected:
        return 'Green light detected! Go!'
    else:
        return 'No traffic light detected.'

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Apply traffic light detection function
    result = detect_traffic_light(frame)

    # Display the resulting frame
    cv2.putText(frame, result, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Traffic Light Detection', frame)

    # Check for key press to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture
cap.release()
cv2.destroyAllWindows()
