import qi
import time
import numpy as np
import cv2
from ultralytics import YOLO
import signal
import sys


k_x, k_y = 0.6, 0.3
speed = 0.12
# Video configuration
RESOLUTION = 1  # VGA
COLOR_SPACE = 0  # RGB

# Load the YOLOv8 Nano model
model = YOLO('best.pt')

def get_max_confidence_box(results):
    """
    Extract the box with the maximum confidence from the YOLO results.
    Returns the bounding box, center coordinates, confidence, and class ID, or None if no detections exist.
    """

    if len(results[0].boxes) == 0:  # Check if there are any detections
        print("No detections found.")
        return None, None, None, None, None
    
    boxes = results[0].boxes.xyxy.cpu().numpy()
    confidences = results[0].boxes.conf.cpu().numpy()
    class_ids = results[0].boxes.cls.cpu().numpy()

    

    max_conf_idx = np.argmax(confidences)
    x1, y1, x2, y2 = boxes[max_conf_idx]
    box_center_x = (x1 + x2) / 2
    box_center_y = (y1 + y2) / 2

    return (x1, y1, x2, y2), box_center_x, box_center_y, confidences[max_conf_idx], class_ids[max_conf_idx]

def calculate_error_signal(img, box_center_x, box_center_y, hysteresis=0.1): 
    """
    Calculate error signal for head movement with hysteresis to avoid unnecessary adjustments.
    Hysteresis prevents movements for very small deviations.
    """
    frame_height, frame_width, _ = img.shape
    frame_center_x, frame_center_y = frame_width / 2, frame_height / 2

    error_x = 2 * (box_center_x - frame_center_x) / frame_width
    error_y = -2 * (box_center_y - frame_center_y) / frame_height

    # Apply hysteresis: ignore small errors
    if abs(error_x) < hysteresis:
        error_x = 0
    if abs(error_y) < hysteresis:
        error_y = 0
        
    print(f"Error in x:{error_x} and errory:{error_y}")
    return error_x, error_y


def head_follower(error_x, error_y, motion_proxy, threshold=0.015):
    """
    Adjusts the robot's head to follow the object based on the error signal,
    but ignores insignificant errors below the specified threshold.
    """
    if abs(error_x) < threshold and abs(error_y) < threshold:
        print(f"Errors below threshold: X={error_x}, Y={error_y}. No movement.")
        return

    yaw_adjustment = -k_x * error_x
    pitch_adjustment = -k_y * error_y

    current_yaw = motion_proxy.getAngles("HeadYaw", True)[0]
    current_pitch = motion_proxy.getAngles("HeadPitch", True)[0]

    new_yaw = float(np.clip(current_yaw + yaw_adjustment, -2, 2))
    new_pitch = float(np.clip(current_pitch + pitch_adjustment, -0.7, 0.4))

    motion_proxy.setAngles(["HeadYaw", "HeadPitch"], [new_yaw, new_pitch], speed)

def set_led_color(session, color):
    try:
        leds = session.service("ALLeds")
        red = ((color >> 16) & 0xFF) / 255.0
        green = ((color >> 8) & 0xFF) / 255.0
        blue = (color & 0xFF) / 255.0
        leds.fadeRGB("FaceLeds", red, green, blue, 0.0)
    except Exception as e:
        print(f"Error setting LED color: {e}")

def video(session, motion_proxy):
    video = session.service("ALVideoDevice")
    video.setActiveCamera(0) 
    id = video.subscribe("yahir5", RESOLUTION, COLOR_SPACE, 10)

    def signal_handler(sig, frame):
        print("Exiting program...")
        set_led_color(session, 0xFFFFFF)
        motion_proxy.rest()
        video.unsubscribe(id)
        cv2.destroyAllWindows()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    while True:
        img = video.getImageRemote(id)
        if img is None:
            continue

        img2 = np.frombuffer(img[6], dtype=np.uint8).reshape(img[1], img[0], -1)


        video.releaseImages(id)
        
        # Convert the image to RGB
        img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

        # Prevent unnecessary conversion: assuming RGB is already provided by the camera
        # No need to use cv2.COLOR_GRAY2BGR as COLOR_SPACE = 0 implies RGB

        results = model(img2.copy())

        box, box_center_x, box_center_y, max_confidence, max_class_id = get_max_confidence_box(results)
        if box:  # Ensure box is not None
            x1, y1, x2, y2 = box  # Unpack the tuple
            height = y2-y1
            print(f"Altura:{height}")

        if max_confidence is None or max_confidence <= 0.7:
            set_led_color(session, 0x000000)
            continue

        # Check for None values before calculating the error signal
        if box_center_x is None or box_center_y is None:
            print("Skipping error calculation due to no detections.")
            error_x, error_y = 0.0, 0.0  # Or set default values
        else:
            error_x, error_y = calculate_error_signal(img2, box_center_x, box_center_y)

        head_follower(error_x, error_y, motion_proxy, threshold=0.05)

        if max_class_id == 2:  # 'standing'
            set_led_color(session, 0x00FF00)  # Green
        elif max_class_id == 1:  # 'falling'
            set_led_color(session, 0x0000FF)  # Blue
        elif max_class_id == 0:  # 'fallen'
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

def move_to_standing(motion_proxy, posture_proxy):
    motion_proxy.wakeUp()
    posture_proxy.goToPosture("StandInit", 0.5)

def main():
    app = qi.Application(url="tcp://10.104.64.18:9559")
    app.start()
    session = app.session

    motion = session.service("ALMotion")
    posture = session.service("ALRobotPosture")

    move_to_standing(motion, posture)
    set_stiffness_to_standing(motion)
    video(session, motion)

if __name__ == "__main__":
    main()