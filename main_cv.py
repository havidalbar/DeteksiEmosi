import cv2
import os
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.models import model_from_json
import numpy as np
from tensorflow.python.keras.backend import set_session
import tensorflow as tf
import imutils

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
model_json_file = "model.json"
model_weight = "model_weights.h5"
load_model = model_from_json(open(model_json_file, "r").read())
load_model.load_weights(model_weight)
font = cv2.FONT_HERSHEY_SIMPLEX
video_capture = cv2.VideoCapture(0)


while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    frame = imutils.resize(frame, width=1000)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.3, 5)    
    EMOTIONS_LIST = ["Marah", "Jijik", "Takut", "Senang", "Netral", "Sedih", "Kaget"]
    for (x, y, w, h) in faces:
        face_frame = gray[y:y+h,x:x+w]
        face_frame = cv2.resize(face_frame, (48, 48))
        preds = load_model.predict(face_frame[np.newaxis, :, :, np.newaxis])
        label = EMOTIONS_LIST[np.argmax(preds)]
        color = (0, 255, 0)
        cv2.putText(frame, label, (x, y- 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        cv2.rectangle(frame, (x, y), (x + w, y + h),color, 2)
        # Display the resulting frame    
    cv2.imshow('Video', frame) 
    # out.write(frame)
    
    if cv2.waitKey(33) & 0xFF == ord('q'):
        break
        
video_capture.release()
# out.release()
cv2.destroyAllWindows()