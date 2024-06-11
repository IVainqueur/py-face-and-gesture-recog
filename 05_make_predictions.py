import sqlite3
import cv2
import numpy as np
import mediapipe as mp
from contextlib import closing

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

class DatabaseManager:
    def __init__(self):
        self.conn = sqlite3.connect('customer_faces_data.db', check_same_thread=False)
        self.cursor = self.conn.cursor()

    def query(self, sql, params=None):
        self.cursor.execute(sql, params)
        return self.cursor

    def get_customer_name(self, predicted_id):
        cursor = self.query("SELECT customer_name FROM customers WHERE customer_uid =?", (predicted_id,))
        result = cursor.fetchone()
        if result:
            return result[0]
        else:
            return "Unknown"

    def add_ok_sign_column(self):
        self.query("ALTER TABLE customers ADD COLUMN ok_sign_detected INTEGER DEFAULT 0")

    def update_ok_sign_detected(self, predicted_id, ok_sign_detected):
        self.query("UPDATE customers SET ok_sign_detected =? WHERE customer_uid =?", (ok_sign_detected, predicted_id))


def detect_ok_sign(image, hand_landmarks):
    if hand_landmarks:
        for hand_landmark in hand_landmarks:
            thumb_tip = hand_landmark.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmark.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            index_mcp = hand_landmark.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
            
            if (abs(thumb_tip.x - index_tip.x) < 0.05 and
                abs(thumb_tip.y - index_tip.y) < 0.05 and
                index_tip.y < index_mcp.y):
                return True
    return False

def main():
    faceRecognizer = cv2.face.LBPHFaceRecognizer_create()
    faceRecognizer.read("models/trained_lbph_face_recognizer_model.yml")

    # Load Haarcascade for face detection
    faceCascade = cv2.CascadeClassifier("models/haarcascade_frontalface_default.xml")
    
    cam = cv2.VideoCapture(0)
    hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
    
    db_manager = DatabaseManager()
    
    while True:
        ret, frame = cam.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100))
        
        conf = -1  # Initialize conf to a default value
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + w]
            id_, conf = faceRecognizer.predict(roi_gray)
            
            if conf >= 45:
                customer_name = db_manager.get_customer_name(id_)
                label = f"{customer_name} - {round(conf, 2)}%"
            else:
                label = "Unknown"
            
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        ok_sign_detected = detect_ok_sign(rgb_frame, results.multi_hand_landmarks)
        
        if ok_sign_detected:
            cv2.putText(frame, "OK Sign Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
            if conf >= 45:
                db_manager.update_ok_sign_detected(id_, 0)
                # break  # Uncomment to stop the script after updating the database
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        cv2.imshow('Face and Hand Gesture Recognition', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cam.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # add when first running it
    db_manager = DatabaseManager()
    # db_manager.add_ok_sign_column()
    main()
