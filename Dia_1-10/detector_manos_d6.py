import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Función para saber si un dedo está extendido
def dedo_extendido(tip, pip, lm):
    return lm[tip].y < lm[pip].y

cap = cv2.VideoCapture(0)

with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7) as hands:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        letra = ""

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                lm = hand_landmarks.landmark

                # Dedos
                indice = dedo_extendido(8, 6, lm)
                medio = dedo_extendido(12, 10, lm)
                anular = dedo_extendido(16, 14, lm)
                meñique = dedo_extendido(20, 18, lm)

                # Regla simple: solo índice extendido
                if indice and not medio and not anular and not meñique:
                    letra = "D"

                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )

        cv2.putText(frame, f"Letra: {letra}", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Dia 6 - Reconocimiento de letra", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
