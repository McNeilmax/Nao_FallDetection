import qi
import time
import numpy as np
import cv2
import os

def video(session):
    # Access the robot's camera service
    video = session.service("ALVideoDevice")
    video.setActiveCamera(0)  # 0: top camera, 1: bottom camera
    resolution = 1  # VGA resolution
    colorSpace = 11  # RGB color space

    id = video.subscribe("RobotStream", resolution, colorSpace, 6)

    # Create a directory to save frames if it doesn't exist
    if not os.path.exists("saved_frames"):
        os.makedirs("saved_frames")

    frame_count = 0  # Frame counter for naming saved frames

    while True:
        img = video.getImageRemote(id)
        if img is None:
            continue
        img2 = np.frombuffer(img[6], dtype=np.uint8).reshape(img[1], img[0], -1)
        video.releaseImages(id)
        
        # Convert the image from BGR to RGB (OpenCV uses BGR by default)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

        # Show the live video feed
        cv2.imshow("Robot Nao Video Stream", img2)

        # Wait for key press
        key = cv2.waitKey(1)

        # If the 'Esc' key is pressed, exit the loop
        if key == 27:
            break

        # If any other key is pressed, save the current frame
        if key != -1:
            frame_count += 1
            # Save the frame with a unique name
            frame_filename = f"saved_frames/frame_{frame_count}.png"
            cv2.imwrite(frame_filename, cv2.cvtColor(img2, cv2.COLOR_RGB2BGR))
            print(f"Saved frame: {frame_filename}")

    # Clean up and close the video stream
    cv2.destroyAllWindows()
    video.unsubscribe(id)

# Initialize and start the app
app = qi.Application(url="tcp://10.104.64.18:9559")
app.start()
session = app.session
video(session)
