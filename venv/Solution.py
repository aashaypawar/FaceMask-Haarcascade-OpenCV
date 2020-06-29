import numpy as np
import math
import cv2

cap = cv2.VideoCapture(0)
f_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
s_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
    faces = f_cascade.detectMultiScale(gray, 1.3, 5)
    green = 0
    red = 0
    msg = "Message"
    mask = True
    for x, y, w, h in faces:
        smiles = s_cascade.detectMultiScale(gray[y:y + h, x:x + w], 1.05, 350)
        for ex, ey, ew, eh in smiles:
            cv2.rectangle(gray[y:y + h, x:x + w], (ex, ey), (ex + ew, ey + eh), (0, 0, 255), 2)
            cv2.putText(gray, 'No Mask Detected', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            mask = False

    if mask:
        green = 255
        red = 0
        msg = "GO"
    else:
        green = 0
        red = 255
        msg = "STOP"

    cv2.putText(gray, msg, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, green, red), 4)
    cv2.imshow('frame', gray)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()