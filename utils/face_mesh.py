import cv2 # lädt OPENCV
import mediapipe as mp # lädt Mediapipe 
import numpy as np # lädt NUMpy für matematische Operationen
from mediapipe.tasks import python
from mediapipe.tasks.python import vision # schnelle & moderne API(Programmierschnittstelle)

# 1. lädt das trainierte Face_Landmark_Modell
base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
# Wie Mediapipe arbeiten soll
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    running_mode=vision.RunningMode.IMAGE # IMAGE Mode: Jedes Frame wird einzeln verarbeitet
) # Mediapipe arbeitet intern mit Wahrscheinlichkeit

# baut den Gesichtserkennungs Detektor
# Bilder reingeben
# Landmarken bekommen
detector = vision.FaceLandmarker.create_from_options(options)
# Liste von Landmarken_Indizen (stammen von der offiziellen Python-Website)
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323,
             361, 288, 397, 365, 379, 378, 400, 377, 152,
             148, 176, 149, 150, 136, 172, 58, 132, 93,
             234, 127, 162, 21, 54, 103, 67, 109]

def get_forehead_roi(frame, draw=True):
    h, w, _ = frame.shape 
    
    # Farbkonvertierung von (BGR) zu (RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    
    # Gesicht erkennen
    results = detector.detect(mp_image)

    # Kein Gesicht also Kein ROI
    if not results.face_landmarks:
        return None, frame 

    # Erstes Gesicht auswählen
    face_landmarks = results.face_landmarks[0]
    
    # Landmarken --> Pixelkoordinaten
    face_pts = np.array([
        (int(face_landmarks[i].x * w), int(face_landmarks[i].y * h)) 
        for i in FACE_OVAL
    ]) 
    
    # Bounding Box des Gesichtes
    fx, fy, fw, fh = cv2.boundingRect(face_pts) 

    # Dynamische Stirn-ROI
    roi_width = int(fw * 0.55)   
    roi_height = int(fh * 0.22)  
    x = fx + (fw - roi_width) // 2 
    y = fy + int(fh * 0.0)       

    # Bidgrenzen absichern
    x = max(0, min(x, w - 1))
    y = max(0, min(y, h - 1))
    roi_width = min(roi_width, w - x)
    roi_height = min(roi_height, h - y)

    # ROI ausschneiden
    roi = frame[y:y+roi_height, x:x+roi_width] 

    # Zeichnen ein Rechteck
    if draw:
        cv2.rectangle(frame, (x, y), (x+roi_width, y+roi_height), (0, 255, 0), 2) 
        cv2.putText(frame, "Forehead ROI", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2) 

    return roi, frame