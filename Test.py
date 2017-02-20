import cv2
import numpy as np

capture = cv2.VideoCapture(0)
capture.open(0)

while True:
    ret, frame = capture.read()
   # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
