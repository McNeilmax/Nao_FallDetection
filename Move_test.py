import qi
import time

def set_stiffness_to_standing(session):
    """
    Set robot stiffness to allow it to stand in an initial walking position.
    This targets the main joints involved in standing and walking.
    """
    motion = session.service("ALMotion")
    
    # List of main joints for walking (e.g., legs and torso)
    joints = [
        "HeadYaw", "HeadPitch", "LShoulderPitch", "LShoulderRoll", "LElbowYaw", 
        "LElbowRoll", "LWristYaw", "LHand", "RShoulderPitch", "RShoulderRoll", 
        "RElbowYaw", "RElbowRoll", "RWristYaw", "RHand", "LHipYawPitch", 
        "LHipRoll", "LHipPitch", "LKneePitch", "LAnklePitch", "LAnkleRoll", 
        "RHipYawPitch", "RHipRoll", "RHipPitch", "RKneePitch", "RAnklePitch", 
        "RAnkleRoll"
    ]
    
    # Set all joints to average stiffness (0.8)
    stiffnesses = [0.8] * len(joints)
    
    # Apply stiffness to all joints
    motion.setStiffnesses(joints, stiffnesses)

def move_robot(session, x_velocity, y_velocity, theta_velocity):
    """
    Move the robot using the move() function with specified velocities.
    """
    motion = session.service("ALMotion")
    
    # Move the robot with specified velocities (x = forward, theta = turn)
    motion.move(x_velocity, y_velocity, theta_velocity)  # Using (x, y, theta) where y is 0

def set_eye_color_blue(session):
    """
    Set the NAO robot's eyes color to blue.
    """
    leds = session.service("ALLeds")
    leds.fadeRGB("FaceLeds", 0.0, 0.0, 1.0, 0.0)  # RGB: Blue

def make_robot_speak(session, message):
    """
    Make the NAO robot speak a given message.
    """
    tts = session.service("ALTextToSpeech")
    tts.say(message)

def main():
    """
    Main function to control the robot.
    """
    # Connect to the robot
    app = qi.Application(url="tcp://10.104.64.18:9559")  # Replace with your robot's IP
    app.start()
    session = app.session

    # Set the robot's stiffness to the initial walking position
    set_stiffness_to_standing(session)

    # Move the robot forward with a velocity of 0.2 m/s and a small rotation of 0.02 rad/s
    move_robot(session, 1, 0, -.3)

    # Allow the robot to move for 5 seconds
    time.sleep(5)  # The robot will move for 5 seconds

    # Set eye color to blue to show completion
    set_eye_color_blue(session)

    move_robot(session, 0, 0, 0)

    # Make the robot speak
    make_robot_speak(session, "Robot finished moving")

if __name__ == "__main__":
    main()
