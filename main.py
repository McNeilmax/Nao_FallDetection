import qi
import time
import numpy as np
import cv2
from matplotlib import pyplot as plt
import matplotlib
from PIL import Image


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
    video.setActiveCamera(0) # 0: top camera, 1: bottom camera
    resolution = 1    # VGA
    colorSpace = 11   # RGB

    id = video.subscribe("lala2", resolution, colorSpace, 6)

    while True:
        img = video.getImageRemote(id)
        if img is None:
            continue
        img2 = np.frombuffer(img[6], dtype=np.uint8).reshape(img[1], img[0], -1)
        video.releaseImages(id)
    # im = Image.fromarray(img2)
        img2  = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        cv2.imshow('Robot Video Stream', img2)
        key = cv2.waitKey(1)
        if key == 27:  # 27 corresponds to the 'Esc' key
            break

    #im.save("your_file.jpeg")
    #cam = video.openCamera(0)
    #img = 
    #video.closeCamera(cam)
    cv2.destroyAllWindows()
    video.unsubscribe(id)


app = qi.Application(url="tcp://10.104.64.18:9559")
app.start()
session = app.session
video(session)