import cv2
import time
import numpy as np
import HandTracking as ht

wCam, hCam = 640, 480
pTime = 0


cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

detector = ht.HandDetector(detectionCon=0.7)


while True:
    success, img = cap.read()
    img = detector.findHands(img)
    
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(
        img, f"FPS: {int(fps)}", (40, 70), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0), 3
    )

    cv2.imshow("Image", img)
    cv2.waitKey(1)
