import qi
import time
import numpy as np
import cv2
from ultralytics import YOLO  # Import the YOLOv8 model

# Load the YOLOv8 Nano model
model = YOLO('best.pt')  # Replace with the path to your trained YOLOv8 model

def stuff(session):
    #  tts = session.service("ALTextToSpeech")
    #sayOp = tts.say("Hello Word")
    motion = session.service("ALMotion")
    #oveOp = motion.wakeUp()
    #sayOp.wait()
    
    names  = ["RShoulderRoll", "RShoulderPitch"]
    print(motion.getAngles(names, True))
    motion.setStiffnesses(names, [1.0, 1.0])
    angles  = [-1.2, 0.0]
    fractionMaxSpeed  = 0.2
    #for i in range(100):
    motion.setAngles(names, angles, fractionMaxSpeed)
    time.sleep(3.0)
    print(motion.getAngles(names, True))
    #
    #motion.setStiffnesses(names, [0.0, 0.0])

def eye_color(session):
    leds = session.service("ALLeds")
    timeForRotation = 0.1
    totalDuration = 10.0
    color = 0x00FF0000
    leds.rotateEyes(color, timeForRotation, totalDuration)

def video(session):
    video = session.service("ALVideoDevice")
    video.setActiveCamera(0)  # 0: top camera, 1: bottom camera
    resolution = 1  # VGA
    colorSpace = 0  # RGB

    id = video.subscribe("lala12", resolution, colorSpace, 10)

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

        # Get the predictions (bounding boxes, labels, and confidences)
        boxes = results[0].boxes.xyxy.numpy()  # Bounding box coordinates (x1, y1, x2, y2)
        confidences = results[0].boxes.conf.numpy()  # Confidence scores
        class_ids = results[0].boxes.cls.numpy()  # Class ids

        # Print the confidence scores for each detection
        for conf in confidences:
            print(f"Confidence: {conf:.2f}")

        # Convert the result back to BGR for displaying in OpenCV
        #img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)

        # Show the result in the OpenCV window
        #cv2.imshow('Robot Video Stream', img2)
        
        #key = cv2.waitKey(1)
        #if key == 27:  # 27 corresponds to the 'Esc' key
        #    break

    cv2.destroyAllWindows()
    video.unsubscribe(id)

# Initialize and start the app
app = qi.Application(url="tcp://10.104.64.18:9559")
app.start()
session = app.session
video(session)
