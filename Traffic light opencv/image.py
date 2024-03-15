import cv2


def detect_traffic_lights_image(image_path):
    # Read image
    frame = cv2.imread(image_path)

    # Convert frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define color ranges for red, yellow, and green traffic light signals
    red_lower = (170, 100, 100)
    red_upper = (180, 255, 255)
    yellow_lower = (25, 50, 100)
    yellow_upper = (35, 255, 255)
    green_lower = (40, 50, 100)
    green_upper = (80, 255, 255)

    # Create masks for each color
    red_mask = cv2.inRange(hsv, red_lower, red_upper)
    yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
    green_mask = cv2.inRange(hsv, green_lower, green_upper)

    # Combine masks using bitwise OR
    mask = cv2.bitwise_or(red_mask, yellow_mask)
    mask = cv2.bitwise_or(mask, green_mask)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours based on area and aspect ratio
    traffic_lights = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = float(w) / h
        if area > 100 and 0.5 < aspect_ratio < 2:
            traffic_lights.append(cnt)

    # Draw bounding boxes around detected traffic lights
    for cnt in traffic_lights:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the image with detected traffic lights
    cv2.imshow("Traffic Light Detection", frame)
    cv2.waitKey(0)  # Wait for a key press to close the window


if __name__ == "__main__":
    image_path = "images/yellow lights.jpg"  # Replace with your image path
    detect_traffic_lights_image(image_path)

    cv2.destroyAllWindows()
