import numpy as np
import cv2

recognizer = cv2.face.createLBPHFaceRecognizer()
recognizer.load('trainner/trainner.yml')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX
while(True):
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        id = recognizer.predict(gray[y:y+h,x:x+w])
        if(id==1):
            id="Tun"
        else:
            id="Unknown"

        cv2.putText(img,str(id),(x,y+h),cv2.FONT_HERSHEY_SIMPLEX,2,255);

    cv2.imshow('Face',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
