import qi
import time

def set_stiffness_to_standing(session):
    """
    Set robot stiffness to allow it to stand in an initial position.
    """
    motion = session.service("ALMotion")
    
    # List of main joints for walking (e.g., legs and torso)
    joints = [
        "HeadYaw", "HeadPitch"
    ]
    
    # Set all joints to average stiffness (0.8)
    stiffnesses = [0.8] * len(joints)
    
    # Apply stiffness to all joints
    motion.setStiffnesses(joints, stiffnesses)

def move_head_in_square(session):
    """
    Move the robot's head in a square pattern: left, up, right, down.
    """
    motion = session.service("ALMotion")
    
    # Define the angles for the head movements
    head_positions = [
        [0.5, 0.0],  # Move head to the left (HeadYaw, HeadPitch)
        [0.0, 0.5],  # Move head up
        [-0.5, 0.0], # Move head to the right
        [0.0, -0.5]  # Move head down
    ]
    
    # Set average speed for the movement
    fraction_max_speed = 0.2  # 20% of the maximum speed

    # Perform the square pattern head movement
    for position in head_positions:
        # Move head to the desired position (HeadYaw, HeadPitch)
        motion.setAngles(["HeadYaw", "HeadPitch"], position, fraction_max_speed)
        time.sleep(1.0)  # Wait for 1 second at each position

def set_eye_color_blue(session):
    """
    Set the NAO robot's eyes color to blue after the head movement is completed.
    """
    leds = session.service("ALLeds")
    leds.fadeRGB("FaceLeds", 0.0, 0.0, 1.0, 0.0)  # RGB: Blue

def main():
    """
    Main function to control the robot.
    """
    # Connect to the robot
    app = qi.Application(url="tcp://10.104.64.18:9559")  # Adjust the IP address as needed
    app.start()
    session = app.session

    # Set the robot's stiffness to the standing position
    set_stiffness_to_standing(session)

    # Move the robot's head in a square pattern (left, up, right, down)
    move_head_in_square(session)

    # Set the eye color to blue to indicate completion
    set_eye_color_blue(session)

    # Print completion message
    print("Robot has finished the head movement and eye color change.")

if __name__ == "__main__":
    main()
