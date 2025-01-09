import qi
import time
import numpy as np
import cv2
from ultralytics import YOLO
import signal
import sys

# Tuning constants
k_x, k_y = 1.0, 1.0
speed = 0.2

# Video configuration
RESOLUTION = 1  # VGA
COLOR_SPACE = 0  # RGB

# Load the YOLOv8 Nano model
model = YOLO('best.pt')

def calculate_error_signal(img, results): 
    frame_height, frame_width, _ = img.shape
    frame_center_x, frame_center_y = frame_width / 2, frame_height / 2

    # Get the highest confidence detection
    boxes = results[0].boxes.xyxy.cpu().numpy()  
    confidences = results[0].boxes.conf.cpu().numpy() 

    if len(confidences) == 0:
        print("No detections found.")
        return 0, 0

    max_confidence_idx = np.argmax(confidences)
    x1, y1, x2, y2 = boxes[max_confidence_idx]
    box_center_x = (x1 + x2) / 2
    box_center_y = (y1 + y2) / 2

    error_x = 2 * (box_center_x - frame_center_x) / frame_width
    error_y = 2 * (box_center_y - frame_center_y) / frame_height

    print(f"Error Signal -> X: {error_x}, Y: {error_y}")
    return error_x, error_y

def head_follower(error_x, error_y, motion_proxy, threshold=0.05):
    """
    Adjusts the robot's head to follow the object based on the error signal,
    but ignores insignificant errors below the specified threshold.
    """
    # Ignore adjustments if errors are too small
    if abs(error_x) < threshold and abs(error_y) < threshold:
        print(f"Errors below threshold: X={error_x}, Y={error_y}. No movement.")
        return

    # Calculate adjustments
    yaw_adjustment = -k_x * error_x
    pitch_adjustment = -k_y * error_y

    # Get current head angles
    current_yaw = motion_proxy.getAngles("HeadYaw", True)[0]
    current_pitch = motion_proxy.getAngles("HeadPitch", True)[0]

    # Update angles with limits
    new_yaw = np.clip(current_yaw + yaw_adjustment, -2, 2)
    new_pitch = np.clip(current_pitch + pitch_adjustment, -0.7, 0.4)

    # Command the robot to move
    motion_proxy.setAngles(["HeadYaw", "HeadPitch"], [new_yaw, new_pitch], speed)

def set_led_color(session, color):
    leds = session.service("ALLeds")
    red = ((color >> 16) & 0xFF) / 255.0
    green = ((color >> 8) & 0xFF) / 255.0
    blue = (color & 0xFF) / 255.0
    leds.fadeRGB("FaceLeds", red, green, blue, 0.0)

def video(session, motion_proxy):
    video = session.service("ALVideoDevice")
    video.setActiveCamera(0) 
    id = video.subscribe("lala26", RESOLUTION, COLOR_SPACE, 5)

    def signal_handler(sig, frame):
        print("Exiting...")
        video.unsubscribe(id)
        cv2.destroyAllWindows()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    while True:
        img = video.getImageRemote(id)
        if img is None:
            continue
        img2 = np.frombuffer(img[6], dtype=np.uint8).reshape(img[1], img[0], -1)

        results = model(img2.copy())

        confidences = results[0].boxes.conf.numpy()  
        if len(confidences) == 0:
            set_led_color(session, 0x000000)
            continue

        max_confidence_idx = np.argmax(confidences)
        max_confidence = confidences[max_confidence_idx]
        max_class_id = results[0].boxes.cls.numpy()[max_confidence_idx]

        if max_confidence > 0.7:
            if max_class_id == 2:  # 'standing'
                print(f"Confidence: {max_confidence:.2f} - Standing")
                set_led_color(session, 0x00FF00)  # Green
                error_x, error_y = calculate_error_signal(img2, results)
                head_follower(error_x, error_y, motion_proxy, threshold=0.05)
            elif max_class_id == 1:  # 'falling'
                print(f"Confidence: {max_confidence:.2f} - Falling")
                set_led_color(session, 0x0000FF)  # Blue
            elif max_class_id == 0:  # 'fallen'
                print(f"Confidence: {max_confidence:.2f} - Fallen")
                set_led_color(session, 0xFF0000)  # Red

def set_stiffness_to_standing(motion):
    joints = [
        "HeadYaw", "HeadPitch", "LShoulderPitch", "LShoulderRoll", "LElbowYaw", 
        "LElbowRoll", "LWristYaw", "LHand", "RShoulderPitch", "RShoulderRoll", 
        "RElbowYaw", "RElbowRoll", "RWristYaw", "RHand", "LHipYawPitch", 
        "LHipRoll", "LHipPitch", "LKneePitch", "LAnklePitch", "LAnkleRoll", 
        "RHipYawPitch", "RHipRoll", "RHipPitch", "RKneePitch", "RAnklePitch", 
        "RAnkleRoll"
    ]
    stiffnesses = [0.8] * len(joints)
    motion.setStiffnesses(joints, stiffnesses)

def main():
    app = qi.Application(url="tcp://10.104.64.18:9559")
    app.start()
    session = app.session
    motion = session.service("ALMotion")
    set_stiffness_to_standing(motion)
    video(session, motion)

if __name__ == "__main__":
    main()