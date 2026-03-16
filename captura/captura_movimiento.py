import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import time

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# guardaremos los últimos 30 frames
secuencia = deque(maxlen=30)

with mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
) as hands:

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        time.sleep(0.1)  # para no saturar el buffer
        
        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        resultado = hands.process(rgb)

        if resultado.multi_hand_landmarks:

            for hand_landmarks in resultado.multi_hand_landmarks:

                puntos = []

                for lm in hand_landmarks.landmark:
                    puntos.append(lm.x)
                    puntos.append(lm.y)

                secuencia.append(puntos)

                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

        cv2.putText(frame, f"Frames guardados: {len(secuencia)}",
                    (10,40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0,255,0),
                    2)

        cv2.imshow("Movimiento", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()