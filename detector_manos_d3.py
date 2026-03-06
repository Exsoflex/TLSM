import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

def dedo_extend(dedo_tip, dedo_mcp, landmarks):
    return landmarks[dedo_tip].y < landmarks[dedo_mcp].y

cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as hands:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                lm = hand_landmarks.landmark

                dedos = {
                    "indice": dedo_extend(8, 5, lm),
                    "medio": dedo_extend(12, 9, lm),
                    "anular": dedo_extend(16, 13, lm),
                    "meñique": dedo_extend(20, 17, lm)
                }

                # El pulgar se analiza diferente
                pulgar = lm[4].x > lm[3].x

                dedos_extendidos = sum(dedos.values()) + pulgar

                if dedos_extendidos >= 4:
                    texto = "Mano abierta"
                elif dedos_extendidos <= 1:
                    texto = "Punio"
                else:
                    texto = "Intermedio"


                cv2.putText(frame, texto, (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

        cv2.imshow("Día 3 - Detector Gestos Básicos", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
