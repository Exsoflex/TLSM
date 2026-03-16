import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def dedo_extendido(tip, pip, lm):
    return lm[tip].y < lm[pip].y

cap = cv2.VideoCapture(0)

texto = ""

with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7) as hands:

    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        resultado = hands.process(rgb)
        
        cv2.rectangle(frame,(0,0),(640,80),(0,0,0),-1)

        cv2.putText(frame, texto, (20,60),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,(0,255,0),3)

        if resultado.multi_hand_landmarks:

            for hand_landmarks in resultado.multi_hand_landmarks:

                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                lm = hand_landmarks.landmark

                dedos = []

                dedos.append(dedo_extendido(8,6,lm))   # índice
                dedos.append(dedo_extendido(12,10,lm)) # medio
                dedos.append(dedo_extendido(16,14,lm)) # anular
                dedos.append(dedo_extendido(20,18,lm)) # meñique

                total = dedos.count(True)

                # Traducción simple
                if total == 1:
                    texto = "UNO"

                elif total == 2:
                    texto = "DOS"

                elif total == 4:
                    texto = "HOLA"

                else:
                    texto = ""

        cv2.imshow("Traductor LSM", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

        
cap.release()
cv2.destroyAllWindows()