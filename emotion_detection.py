import cv2
import numpy as np
from fer.fer import FER

class EmotionDetector:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.emotion_detector = FER()
        
        
    def detect_faces(self,frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        return faces
    
    def detect_emotions(self, frame):
        
        faces = self.detect_faces(frame)
        emotions = []
        for(x, y, w, h) in faces:
            
            face_img = frame[y:y+h, x:x+w]
            emotion_data = self.emotion_detector.detect_emotions(face_img)
            
            if emotion_data:
                emotion_scores = emotion_data[0]['emotions']
                emotion = max(emotion_scores, key = emotion_scores.get)
            
            else:
                emotion = 'neutral'
            emotions.append({'position':(x, y, w, h), 'emotion': emotion})
        return emotions
    

if __name__ == "__main__":
    detector = EmotionDetector()
    #usage of web cam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("cannot open camera")
        exit()
        
    while True:
        ret, frame = cap.read()
        if  not ret:
            print("can't receive frame(streaming end)? Exiting...")
            break
        
        emotions = detector.detect_emotions(frame)
        #displaying
        
        for em in emotions:
            x, y, w, h = em['position']
            cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2 )
            cv2.putText(frame, em["emotion"], (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255, 12), 2)
        
        cv2.imshow("Emotion Detection",frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
