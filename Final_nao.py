import qi
import numpy as np
import cv2
from ultralytics import YOLO
import signal
import sys
import time
from bisect import bisect_left
from config import HEAD_GAINS, BODY_ALIGNMENT, WALKING, VIDEO, DETECTION, SCAN_POSITIONS

# HEAD CONTROLLER GAINS
head_kx = HEAD_GAINS["kx"]
head_ky = HEAD_GAINS["ky"]
speed_head = HEAD_GAINS["speed"]
head_threshold = HEAD_GAINS["threshold"]
# BODY ALIGNMENT GAINS
body_kp = BODY_ALIGNMENT["kp"]
body_threshold = BODY_ALIGNMENT["threshold"]
# WALKING GAINS
walking_kp = WALKING["kp"]
height_goal = WALKING["goal_height"]
walking_threshold = WALKING["walking_threshold"]

###GLOBAL CONTROLL FLAGS###
#if any of this are true that means the control task is done (threshold) or event ocurred
head_aligned = False
body_aligned = False
person_reached = False
first_fall = False

###Scan Room with head global variables
# SCANNING VARIABLES
posiciones = SCAN_POSITIONS["angles"]
index = 0
direction = 1
buscando = False
counter = 0
###########################

# Video configuration
RESOLUTION = VIDEO["resolution"]  # 320, 240
COLOR_SPACE = VIDEO["color_space"] # RGB kYuvColorSpace
FPS = VIDEO["fps"] # Desired Video Frame Rate capture

# Load the YOLOv8 Nano model
model = YOLO(DETECTION["model_path"])
min_confidence = DETECTION["confidence_threshold"] # Min confidence to consider a detection valid


def get_max_confidence_box(results):
    """
    Extract the box with the maximum confidence from the YOLO results.
    Returns the bounding box, center coordinates, confidence, and class ID, 
    or None in case that no detections exist.
    """

    if len(results[0].boxes) == 0: #Boxes non existant 
        return None, None, None, None, None # Return None = no detections
    
    #Get all the results
    boxes = results[0].boxes.xyxy.cpu().numpy()
    confidences = results[0].boxes.conf.cpu().numpy()
    class_ids = results[0].boxes.cls.cpu().numpy()

    #Get oly the max confidence results
    max_conf_idx = np.argmax(confidences)
    x1, y1, x2, y2 = boxes[max_conf_idx]
    box_center_x = (x1 + x2) / 2
    box_center_y = (y1 + y2) / 2

    return (x1, y1, x2, y2), box_center_x, box_center_y, confidences[max_conf_idx], class_ids[max_conf_idx]

def calculate_error_signal(img, box_center_x, box_center_y): 
    """
    Calculate error signal for head movement, uses bounding box max from max detection
    returns diference betwen the box and the center of teh screen (yaw, pitch) error
    """
    frame_height, frame_width, _ = img.shape #320, 240 pixels -> specified in global variable RESOLUTION
    frame_center_x, frame_center_y = frame_width / 2, frame_height / 2 # Get reference middle point

    #Normalized error [-1, 1]   
    error_x = 2 * (box_center_x - frame_center_x) / frame_width
    error_y = -2 * (box_center_y - frame_center_y) / frame_height

    return error_x, error_y

def head_follower(error_x, error_y, motion_proxy, threshold, head_angles):
    """
    Adjusts the robot's head to follow the object based on Proportional control,
    but ignores insignificant errors below the specified threshold.
    """
    global head_aligned
    #Completion of task if head is  within threashold
    if abs(error_x) < threshold and abs(error_y) < threshold:
        head_aligned = True
        #print(f"Errors below threshold: X={error_x}, Y={error_y}. No movement.")
        return
    
    head_aligned = False
    #Compute proportional controler
    yaw_adjustment = -head_kx * error_x
    pitch_adjustment = -head_ky * error_y
    

    #current_yaw = motion_proxy.getAngles("HeadYaw", True)[0]
    #current_pitch = motion_proxy.getAngles("HeadPitch", True)[0]

    # Cache current angles 
    current_yaw, current_pitch = head_angles[0], head_angles[1]

    #Clip and conversion to double
    new_yaw = float(np.clip(current_yaw + yaw_adjustment, -2, 2))
    new_pitch = float(np.clip(current_pitch + pitch_adjustment, -0.7, 0.4))
    #Apply control signal to joints
    motion_proxy.setAngles(["HeadYaw", "HeadPitch"], [new_yaw, new_pitch], speed_head)

def align_body_with_head(motion_proxy, threshold, yaw_head):
    """
    Aligns the robot's body with the current head orientation using a proportional controler.
    The robot turns smoothly while maintaining head tracking without interrupting other processes.
    """
    global body_aligned, head_aligned
    #head needs to be aligned first
    if not head_aligned:
        motion_proxy.move(0, 0, 0)
        return

    # Get the current head yaw angle
    #head_yaw_angle = motion_proxy.getAngles("HeadYaw", True)[0]

    #Completion of task if head is  within threashold
    if abs(yaw_head) < threshold:  
        if not body_aligned:  # Only stop once when alignment is achieved
            motion_proxy.move(0, 0, 0)
            body_aligned = True
        return
    
    body_aligned = False
    
    # Calculate and clip proportional control signal 
    theta_velocity = float(np.clip(body_kp * yaw_head, -0.8, 0.8))

    # Apply the control signal to change only angular velocity rads/s
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

def walk_to_person(session, motion_proxy, box, height_goal, walking_threshold):
    """
    Walk towards the fallen person based on bounding box size, using proportional controller.}
    Makes shure that the robot is aligned before walikng
    """
    global person_reached, body_aligned, first_fall

    if not body_aligned:
        return
    
    if not first_fall:
        first_fall = True
        time.sleep(0.5)
        return
    
    first_fall = False
    x1, y1, x2, y2 = box # unpack box tuple coordinates

    # Estimate distance based on bounding box height
    box_height = y2 - y1
    distance_error = height_goal - box_height

    #control task finished if error falls within threshold boundaries
    if abs(distance_error) < walking_threshold:
        if not person_reached: #Person is within minimum distance. Stop walking
            person_reached = True
            motion_proxy.move(0, 0, 0)
            time.sleep(1)
            make_robot_speak(session, "Are you ok? Let me ask for help")#Speak
            
            run_behavior(session, "animations/Stand/Emotions/Neutral/AskForAttention_2")    #Do gesture
        return

    person_reached = False 

    #calculate proportional walking distance
    walking_distance = float(np.clip(walking_kp * distance_error, -0.08, 0.08))

    motion_proxy.move(walking_distance, 0, 0)  # Move forward in small steps

def set_led_color(session, color):
    if not isinstance(color, int) or color < 0 or color > 0xFFFFFF:
        print(f"Invalid color value: {color}")
        return

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
        #print("Eyes' LEDs have been turned off.")
    except Exception as e:
        print(f"Error turning off LEDs: {e}")

# def find_closest_index(x):
#     global posiciones
    
#     closest_index = min(range(len(posiciones)), key=lambda i: abs(posiciones[i] - x))
#     return closest_index
def find_closest_index(x):
    global posiciones
    
    # Find the index where x would be inserted
    idx = bisect_left(posiciones, x)
    
    # Handle edge cases
    if idx == 0:
        return 0
    if idx == len(posiciones):
        return len(posiciones) - 1
    
    # Compare neighbors and return the closest
    left = idx - 1
    right = idx
    return left if abs(posiciones[left] - x) <= abs(posiciones[right] - x) else right

def buscar_persona(motion_proxy):
    
    global index, posiciones, direction

    #move head according to serch a person
    motion_proxy.setAngles(["HeadYaw", "HeadPitch"], [posiciones[index], 0.0], 0.1)
    
    if index in {0, len(posiciones) - 1}:
        direction *= -1
    index += direction

def get_head_angles(motion_proxy):
    """
    Retrieves head angles (yaw, pitch) and returns them.
    """
    try:
        return motion_proxy.getAngles(["HeadYaw", "HeadPitch"], True)
    except Exception as e:
        print(f"Error retrieving head angles: {e}")
        return [0.0, 0.0]  # Fallback to default angles

def video(session, motion_proxy):
    #Setup
    video = session.service("ALVideoDevice")
    video.setActiveCamera(0) 
    id = video.subscribe("naoFINAL1", RESOLUTION, COLOR_SPACE, FPS) # Resolution, Colorspace and Frame rate

    global  counter, buscando, index #Global variables for looking around the room

    def signal_handler(sig, frame):
        print("Exiting program...")
        try:
            motion_proxy.rest()
            video.unsubscribe(id)
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"Error during cleanup: {e}")
        sys.exit(0)


    signal.signal(signal.SIGINT, signal_handler)
    #Capture image
    while True:
        #Get frame
        try:
            img = video.getImageRemote(id)
            if img is None:
                continue
        except Exception as e:
            print(f"Error capturing video frame: {e}")
            continue

        
        img2 = np.frombuffer(img[6], dtype=np.uint8).reshape(img[1], img[0], -1)

        video.releaseImages(id)
        
        # Convert the image to RGB
        img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

        #results = model(img2.copy())#Do Yolov8 model class classification inference
        results = model(img2)  # Pass image directly


        #Obtain only max confidence result parameters by custom method
        box, box_center_x, box_center_y, max_confidence, max_class_id = get_max_confidence_box(results)
        #Threshold for min confidence of detection

        if box is None or max_confidence < min_confidence:
            counter += 1 # asumes no detection or no valid detection is found

            if counter >= 5: #Start searching acroos the room
                if not buscando:
                    #stop robot
                    motion_proxy.move(0, 0, 0)
                    set_led_color(session, 0x000000)#White

                    current_yaw = motion_proxy.getAngles("HeadYaw", True)[0] # Get currect head yaw
                    index = find_closest_index(current_yaw) # Use it to find closes position of next looking point
                    buscando = True
                    
                buscar_persona(motion_proxy) # Moves to the next yaw position contained in "posciciones" list
            continue

        counter = 0
        buscando = False

        #Get error with reaspect to head alignment
        error_x, error_y = calculate_error_signal(img2, box_center_x, box_center_y)

        #Follow detction dounding box with NAO's head
        head_angles = get_head_angles(motion_proxy)
        head_follower(error_x, error_y, motion_proxy, head_threshold, head_angles)
        align_body_with_head(motion_proxy, body_threshold, head_angles[0])
        if max_class_id == 2:  # 'standing'
            set_led_color(session, 0x00FF00)
            #current_led_color = 0x00FF00
        elif max_class_id == 1:  # 'falling'
            set_led_color(session, 0x0000FF)
            #current_led_color = 0x0000FF

        elif max_class_id == 0:  # 'fallen'
            set_led_color(session, 0xFF0000)
            # current_led_color = 0xFF0000    
            walk_to_person(session, motion_proxy, box, height_goal, walking_threshold)               


def set_stiffness_to_standing(motion):
    ALL_JOINTS = motion.getBodyNames("Body")
    motion.setStiffnesses(ALL_JOINTS, 0.8)

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