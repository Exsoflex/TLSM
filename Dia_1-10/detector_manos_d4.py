import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def dedo_extendido(tip, pip, lm):
    return lm[tip].y < lm[pip].y

cap = cv2.VideoCapture(0)

with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5) as hands:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:

            for hand in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
                lm = hand.landmark

                # DEDOS (excepto pulgar)
                dedo1 = dedo_extendido(8, 6, lm)   # Índice
                dedo2 = dedo_extendido(12, 10, lm) # Medio
                dedo3 = dedo_extendido(16, 14, lm) # Anular
                dedo4 = dedo_extendido(20, 18, lm) # Meñique

                # PULGAR: comparación en X
                pulgar = lm[4].x > lm[3].x

                dedos = [pulgar, dedo1, dedo2, dedo3, dedo4]

                # Clasificación de números
                if dedos == [0,0,0,0,0]: numero = "0"
                elif dedos == [0,1,0,0,0]: numero = "1"
                elif dedos == [0,1,1,0,0]: numero = "2"
                elif dedos == [0,1,1,1,0]: numero = "3"
                elif dedos == [0,1,1,1,1]: numero = "4"
                elif dedos == [1,1,1,1,1]: numero = "5"
                else:
                    numero = "Gesto no reconocido"

                cv2.putText(frame, f"Numero: {numero}", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

        cv2.imshow("Dia 4 - Numeros con la mano", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
