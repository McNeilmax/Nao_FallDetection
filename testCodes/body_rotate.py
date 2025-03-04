import qi
import time
import numpy as np
import cv2
from ultralytics import YOLO
import signal
import sys

# Initialize the timer
# fallen_start_time = None
# alignment_duration = 10  # seconds

k_x, k_y = 0.6, 0.4
speed = 0.1

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
    error_y = -2 * (box_center_y - frame_center_y) / frame_height

    return error_x, error_y

def head_follower(error_x, error_y, motion_proxy, threshold=0.01):
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
    new_yaw = float(np.clip(current_yaw + yaw_adjustment, -2, 2))
    new_pitch = float(np.clip(current_pitch + pitch_adjustment, -0.7, 0.4))

    # Command the robot to move
    motion_proxy.setAngles(["HeadYaw", "HeadPitch"], [new_yaw, new_pitch], speed)

def align_body_with_head(motion_proxy, threshold=0.2):
    """
    Aligns the robot's body with the current head orientation if the head yaw
    angle is sufficiently large (greater than the threshold).
    """
    # Get the current head yaw angle
    head_yaw_angle = motion_proxy.getAngles("HeadYaw", True)[0]

    # Check if the yaw angle exceeds the threshold
    if abs(head_yaw_angle) < threshold:
        print(f"Yaw angle too small ({head_yaw_angle:.2f}). No body alignment needed.")
        return  # Do not perform body alignment if angle is too small

    print(f"Aligning body with head yaw angle: {head_yaw_angle:.2f}")

    # Command the robot to rotate its torso to align with the head yaw angle
    motion_proxy.moveTo(0, 0, head_yaw_angle)

    # Reset the head yaw to center
    motion_proxy.setAngles("HeadYaw", 0.0, speed)


def set_led_color(session, color):
    leds = session.service("ALLeds")
    red = ((color >> 16) & 0xFF) / 255.0
    green = ((color >> 8) & 0xFF) / 255.0
    blue = (color & 0xFF) / 255.0
    leds.fadeRGB("FaceLeds", red, green, blue, 0.0)

def video(session, motion_proxy):
    video = session.service("ALVideoDevice")
    video.setActiveCamera(0) 
    id = video.subscribe("team2", RESOLUTION, COLOR_SPACE, 10)

    def signal_handler(sig, frame):
        print("Exiting program...")

        # Turn the eyes white
        set_led_color(session, 0xFFFFFF)  # White color

        # Move the robot to the resting position
        motion_proxy.rest()

        # Unsubscribe from the video feed and clean up
        video.unsubscribe(id)
        cv2.destroyAllWindows()
        sys.exit(0)  # Exit the program

    # Set up the signal handler for Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)

    while True:
        img = video.getImageRemote(id)
        if img is None:
            continue
        img2 = np.frombuffer(img[6], dtype=np.uint8).reshape(img[1], img[0], -1)

        video.releaseImages(id)
        
        # Convert the image to RGB
        img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

        results = model(img2.copy())

        confidences = results[0].boxes.conf.numpy()  
        if len(confidences) == 0:
            set_led_color(session, 0x000000)
            continue

        max_confidence_idx = np.argmax(confidences)
        max_confidence = confidences[max_confidence_idx]
        max_class_id = results[0].boxes.cls.numpy()[max_confidence_idx]

        if max_confidence > 0.7:
            error_x, error_y = calculate_error_signal(img2, results)
            head_follower(error_x, error_y, motion_proxy, threshold=0.05)
            if max_class_id == 2:  # 'standing'
                set_led_color(session, 0x00FF00)  # Green
            elif max_class_id == 1:  # 'falling'
                set_led_color(session, 0x0000FF)  # Blue
                align_body_with_head(motion_proxy)
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
    """
    Moves the NAO robot to a standing position.
    """
    # Wake up the robot
    motion_proxy.wakeUp()

    # Use the ALRobotPosture service to move the robot to the "StandInit" posture
    posture_proxy.goToPosture("StandInit", 0.5)

def main():
    app = qi.Application(url="tcp://10.104.64.18:9559")
    app.start()
    session = app.session

    motion = session.service("ALMotion")
    posture = session.service("ALRobotPosture")

    # Move the robot to a standing posture
    move_to_standing(motion, posture)

    # Ensure joints are stiff and ready for perception
    set_stiffness_to_standing(motion)

    # Start the video perception loop
    video(session, motion)

if __name__ == "__main__":
    main()