import qi
import time
import numpy as np
import cv2
from ultralytics import YOLO  # Import the YOLOv8 model

# Load the YOLOv8 Nano model
model = YOLO('best40_gray.pt')  # Replace with the path to your trained YOLOv8 model


def calculate_error_signal(img, results):
    """
    Calculate the error signal for aligning the robot's head with the center of the detected bounding box.
    """
    # Get frame dimensions
    frame_height, frame_width, _ = img.shape
    frame_center_x = frame_width / 2
    frame_center_y = frame_height / 2

    # Extract detections
    boxes = results[0].boxes.xyxy.numpy()  # Bounding box coordinates (x1, y1, x2, y2)
    confidences = results[0].boxes.conf.numpy()  # Confidence scores

    try:
        # Find the detection with the highest confidence
        max_confidence_idx = np.argmax(confidences)
        max_box = boxes[max_confidence_idx]  # Bounding box of the highest confidence detection

        # Compute box center
        x1, y1, x2, y2 = max_box
        box_center_x = (x1 + x2) / 2
        box_center_y = (y1 + y2) / 2

        # Compute error signal
        error_x = box_center_x - frame_center_x
        error_y = box_center_y - frame_center_y

        print(f"Error Signal -> X: {error_x}, Y: {error_y}")
        return error_x, error_y
    except ValueError:
        # Handle cases where there are no detections
        print("No detections found.")
        return 0, 0



def set_led_color(session, color):
    """
    Set the LED color of the NAO robot based on the input color.
    color should be an integer in the format 0xRRGGBB (hexadecimal color format).
    """
    leds = session.service("ALLeds")
    
    # Convert the color from 0xRRGGBB to RGB components (0-1 range)
    red = ((color >> 16) & 0xFF) / 255.0
    green = ((color >> 8) & 0xFF) / 255.0
    blue = (color & 0xFF) / 255.0

    # Set the LED color of the eyes (you can change 'RightEyes'/'LeftEyes' or 'FaceLeds' as needed)
    leds.fadeRGB("FaceLeds", red, green, blue, 0.0)  # Set color without fading


def video(session):
    video = session.service("ALVideoDevice")
    video.setActiveCamera(0)  # 0: top camera, 1: bottom camera
    resolution = 1  # VGA
    colorSpace = 0  # RGB

    id = video.subscribe("lala26", resolution, colorSpace, 5)

    while True:
        img = video.getImageRemote(id)
        if img is None:
            continue
        img2 = np.frombuffer(img[6], dtype=np.uint8).reshape(img[1], img[0], -1)
        video.releaseImages(id)
        
        # Convert the image to RGB
        img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

        # Run inference on the captured frame using YOLOv8
        results = model(img2.copy())  # Predict objects in the image

        # Calculate the error signal for the detected object
        error_x, error_y = calculate_error_signal(img2, results)

        # Get the predictions (bounding boxes, labels, and confidences)
        boxes = results[0].boxes.xyxy.numpy()  # Bounding box coordinates (x1, y1, x2, y2)
        confidences = results[0].boxes.conf.numpy()  # Confidence scores
        class_ids = results[0].boxes.cls.numpy()  # Class ids

        try:
            # Find the result with the highest confidence
            max_confidence_idx = np.argmax(confidences)
            max_confidence = confidences[max_confidence_idx]
            max_class_id = class_ids[max_confidence_idx]

            if max_confidence > 0.7:  # Only act if the confidence is above 70%
                # Check the class and set LED color accordingly
                if max_class_id == 2:  # 'standing' class
                    print(f"Confidence: {max_confidence:.2f} - Standing")
                    set_led_color(session, 0x00FF00)  # Green
                elif max_class_id == 1:  # 'falling' class
                    print(f"Confidence: {max_confidence:.2f} - Falling")
                    set_led_color(session, 0x0000FF)  # Blue
                elif max_class_id == 0:  # 'fallen' class
                    print(f"Confidence: {max_confidence:.2f} - Fallen")
                    set_led_color(session, 0xFF0000)  # Red
        except ValueError:
            # If confidences is empty or an error occurs, skip processing
            print("No detections found or invalid data.")
            set_led_color(session, 0x000000)  
        
        # Show the result in OpenCV (optional)
        # cv2.imshow('Robot Video Stream', img2)
        # key = cv2.waitKey(1)
        # if key == 27:  # 27 corresponds to the 'Esc' key
        #     break

    cv2.destroyAllWindows()
    video.unsubscribe(id)

# Initialize and start the app
app = qi.Application(url="tcp://10.104.64.18:9559")
app.start()
session = app.session

# Call the video function to start the processing
video(session)
