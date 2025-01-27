import qi
import numpy as np
import cv2
from ultralytics import YOLO
import signal
import sys
import time

#HEAD CONTROLLER GAINS
head_kx = 0.5           # Proportional gain for head alignment in x axis
head_ky = 0.3           # Proportional gain for head alignment in y axis
speed = 0.11            # Angular velocity for head alignment
#BODY ALIGN GAINS
body_kp = 0.3           # Proportional gain for body alignment
body_kd = 0             # Derivative gain for body alignment
prev_yaw_angle = 0.0    # Initialize previous yaw angle
#WALKING GAINS
walking_kp = 0.002      # Proportional gain for approaching person

###GLOBAL CONTROLL FLAGS###
head_aligned = False
body_aligned = False
person_reached = False
first_fall = False
# current_led_color = None

posiciones = np.arange(-2, 2.1, 0.1).tolist()
indice = 0
direccion = 1

buscando = False
counter = 0
###########################

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

def calculate_error_signal(img, box_center_x, box_center_y): 
    """
    Calculate error signal for head movement 
    """
    frame_height, frame_width, _ = img.shape
    frame_center_x, frame_center_y = frame_width / 2, frame_height / 2

    #Normalized error [-1, 1]   
    error_x = 2 * (box_center_x - frame_center_x) / frame_width
    error_y = -2 * (box_center_y - frame_center_y) / frame_height

    return error_x, error_y

def head_follower(error_x, error_y, motion_proxy, threshold=0.12):
    """
    Adjusts the robot's head to follow the object based on the error signal,
    but ignores insignificant errors below the specified threshold.
    """
    global head_aligned

    if abs(error_x) < threshold and abs(error_y) < threshold:
        head_aligned = True
        #print(f"Errors below threshold: X={error_x}, Y={error_y}. No movement.")
        return
    
    head_aligned = False

    yaw_adjustment = -head_kx * error_x
    pitch_adjustment = -head_ky * error_y

    current_yaw = motion_proxy.getAngles("HeadYaw", True)[0]
    current_pitch = motion_proxy.getAngles("HeadPitch", True)[0]

    new_yaw = float(np.clip(current_yaw + yaw_adjustment, -2, 2))
    new_pitch = float(np.clip(current_pitch + pitch_adjustment, -0.7, 0.4))

    motion_proxy.setAngles(["HeadYaw", "HeadPitch"], [new_yaw, new_pitch], speed)

def align_body_with_head(motion_proxy, threshold=0.2):
    """
    Aligns the robot's body with the current head orientation using a proportional-derivative controller.
    The robot turns smoothly while maintaining head tracking without interrupting other processes.
    """
    global prev_yaw_angle, body_aligned, head_aligned

    if not head_aligned:
        motion_proxy.move(0, 0, 0)
        return

    # Get the current head yaw angle
    head_yaw_angle = motion_proxy.getAngles("HeadYaw", True)[0]

    if abs(head_yaw_angle) < threshold:  # Threshold 
        if not body_aligned:  # Only stop once when alignment is achieved
            motion_proxy.move(0, 0, 0)
            body_aligned = True
        return
    
    body_aligned = False
    
    # Calculate derivative term
    d_theta = head_yaw_angle - prev_yaw_angle
    prev_yaw_angle = head_yaw_angle

    # Calculate proportional-derivative control signal for body alignment
    theta_velocity = float(np.clip(body_kp * head_yaw_angle + body_kd * d_theta, -0.8, 0.8))

    # Apply the control signal to move the body
    motion_proxy.move(0, 0, theta_velocity)

def make_robot_speak(session, message):
    """
    Make the NAO robot speak a given message.
    """
    tts = session.service("ALTextToSpeech")
    tts.say(message)

def run_behavior(session, behavior_name):
    behavior_manager = session.service("ALBehaviorManager")
    if behavior_manager.isBehaviorInstalled(behavior_name):
        behavior_manager.runBehavior(behavior_name)
    else:
        print(f"Behavior '{behavior_name}' not found!")

def walk_to_person(session, motion_proxy, box, goal_distance = 110, threshold_distance = 10):
    """
    Walk towards the fallen person based on bounding box size.
    """
    global person_reached, body_aligned, first_fall

    if not body_aligned:
        return
    
    if not first_fall:
        first_fall = True
        time.sleep(1)
        return
    
    first_fall = False
    x1, y1, x2, y2 = box

    # Estimate distance based on bounding box height
    box_height = y2 - y1
    distance_error = goal_distance - box_height

    if abs(distance_error) < threshold_distance:
        if not person_reached: #Person is within minimum distance. Stop walking
            person_reached = True
            motion_proxy.move(0, 0, 0)
            make_robot_speak(session, "Are you ok? Please turn up you arms")
            time.sleep(1)
            run_behavior(session, "animations/Stand/Dance")    
        return

    person_reached = False # the controll task is not yet archived

    #calculate walking speed
    walking_speed = float(np.clip(walking_kp * distance_error, -0.08, 0.08))

    motion_proxy.move(walking_speed, 0, 0)  # Move forward in small steps

def set_led_color(session, color):
    try:
        leds = session.service("ALLeds")
        red = ((color >> 16) & 0xFF) / 255.0
        green = ((color >> 8) & 0xFF) / 255.0
        blue = (color & 0xFF) / 255.0
        leds.fadeRGB("FaceLeds", red, green, blue, 0.0)
    except Exception as e:
        print(f"Error setting LED color: {e}")

def turn_off_eyes(session):
    """
    Turn off the eyes' LEDs of the NAO robot.
    """
    try:
        leds = session.service("ALLeds")
        leds.fadeRGB("FaceLeds", 0x000000, 0.0)  # Black (off) color
        print("Eyes' LEDs have been turned off.")
    except Exception as e:
        print(f"Error turning off LEDs: {e}")

def find_closest_index(x):
    global posiciones
    
    closest_index = min(range(len(posiciones)), key=lambda i: abs(posiciones[i] - x))
    return closest_index

def buscar_persona(motion_proxy):

    print("BUSCANDO PERSONA")
    
    global indice, posiciones, direccion

    #move head according to serch a person
    motion_proxy.setAngles(["HeadYaw", "HeadPitch"], [posiciones[indice], 0.0], 0.1)
    
    if(indice == len(posiciones)-1 or  indice == 0):
        direccion = direccion * (-1)
    indice += direccion

def video(session, motion_proxy):
    #Setup
    video = session.service("ALVideoDevice")
    video.setActiveCamera(0) 
    id = video.subscribe("team13", RESOLUTION, COLOR_SPACE, 10)

    global  counter, buscando, indice

    # If closed (Ctrl + C)do the following
    def signal_handler(sig, frame):
        print("Exiting program...")
        #set_led_color(session, 0xFFFFFF)
        motion_proxy.rest()
        video.unsubscribe(id)
        cv2.destroyAllWindows()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    #Capture image
    while True:
        #Get frame
        img = video.getImageRemote(id)
        if img is None:
            continue
        img2 = np.frombuffer(img[6], dtype=np.uint8).reshape(img[1], img[0], -1)

        video.releaseImages(id)
        
        # Convert the image to RGB
        img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
        
        # Prevent unnecessary conversion: assuming RGB is already provided by the camera
        # No need to use cv2.COLOR_GRAY2BGR as COLOR_SPACE = 0 implies RGB

        results = model(img2.copy())#Do Yolov8 model class classification inference

        #Obtain only max confidence result parameters
        box, box_center_x, box_center_y, max_confidence, max_class_id = get_max_confidence_box(results)
        #Threshold for min confidence of detection

        if box is None or max_confidence < 0.7:
            counter += 1

            if counter >= 5:
                if not buscando:
                    #stop robot
                    motion_proxy.move(0, 0, 0)
                    set_led_color(session, 0x000000)#White

                    current_yaw = motion_proxy.getAngles("HeadYaw", True)[0]
                    indice = find_closest_index(current_yaw)
                    buscando = True
                    
                buscar_persona(motion_proxy)
            continue

        counter = 0
        buscando = False

        #Get error with reaspect to head alignment
        error_x, error_y = calculate_error_signal(img2, box_center_x, box_center_y)

        #Follow detction dounding box with NAO's head
        head_follower(error_x, error_y, motion_proxy)

        if max_class_id == 2:  # 'standing'
            #set_led_color(session, 0x00FF00)
            #current_led_color = 0x00FF00

            align_body_with_head(motion_proxy)

        elif max_class_id == 1:  # 'falling'
            #set_led_color(session, 0x0000FF)
            #current_led_color = 0x0000FF

            align_body_with_head(motion_proxy)

        elif max_class_id == 0:  # 'fallen'
            #set_led_color(session, 0xFF0000)
            # current_led_color = 0xFF0000    
            align_body_with_head(motion_proxy)
            walk_to_person(session, motion_proxy, box)               

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
    #Define NAO Robot connection 
    app = qi.Application(url="tcp://10.104.64.18:9559")
    app.start()
    session = app.session
    #Movement services
    motion = session.service("ALMotion")
    posture = session.service("ALRobotPosture")

    turn_off_eyes(session)
    move_to_standing(motion, posture)# stand up from resting position
    set_stiffness_to_standing(motion)# Give actuators some stiffness
    video(session, motion)#Capture video 10 HZ

if __name__ == "__main__":
    main()
