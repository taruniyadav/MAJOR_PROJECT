import cv2
import numpy as np

class EmotionDetector:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        self.emotions = ['Angry','Digest','Fear','Happy','Sad','Surprise','Neutral']
        
    def detect_face(self,frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor = 1.1, minNeighbors = 5, minSize= (30,30))
        
        return faces
    
    def detect_emotion(self, face_roi):
        
        return np.random.choice(self.emotions)
    
    
    def process_frame(self, frame):
        faces = self.detect_face(frame)
        detected_emotions = []
        for(x, y, w, h) in faces:
            face_roi = frame [y: y+h, x: x+w]
            emotion = self.detect_emotion(face_roi)
            detected_emotions.append({"bbox":(x, y, w, h), "emotion": emotion})
        
        return detected_emotions
    
if __name__ == "__main__":
    detector = EmotionDetector()
    #usage of web cam
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if  not ret:
            break
        emotions = detector.process_frame(frame)
        #displaying
        
        for emo in emotions:
            x, y, w, h = emo["bbox"]
            cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2 )
            cv2.putText(frame, emo["emotion"], (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255, 12), 2)
        
        cv2.imshow("Emotion Detection",frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        cap.release()
        cv2.destroyAllWindows()
