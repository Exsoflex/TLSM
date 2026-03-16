import cv2
import mediapipe as mp
import time

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def dedo_extendido(tip, pip, lm):
    return lm[tip].y < lm[pip].y


cap = cv2.VideoCapture(0)

texto = ""

# tiempo de espera entre letras
cooldown = 1.5
ultimo_tiempo = time.time()

with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7) as hands:

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        resultado = hands.process(rgb)

        letra_detectada = ""

        if resultado.multi_hand_landmarks:

            for hand_landmarks in resultado.multi_hand_landmarks:

                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

                lm = hand_landmarks.landmark

                indice = dedo_extendido(8, 6, lm)
                medio = dedo_extendido(12, 10, lm)
                anular = dedo_extendido(16, 14, lm)
                menique = dedo_extendido(20, 18, lm)

                dedos = [indice, medio, anular, menique]

                # Ejemplo de letras
                if dedos == [False, False, False, False]:
                    letra_detectada = "A"

                elif dedos == [True, True, True, True]:
                    letra_detectada = "B"

                elif dedos == [True, False, False, False]:
                    letra_detectada = "D"

        # CONTROL DE TIEMPO
        tiempo_actual = time.time()

        if letra_detectada != "":
            if tiempo_actual - ultimo_tiempo > cooldown:
                texto += letra_detectada
                ultimo_tiempo = tiempo_actual

        # mostrar letra actual
        cv2.putText(frame, "Letra: " + letra_detectada,
                    (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2)

        # mostrar palabra
        cv2.putText(frame, "Texto: " + texto,
                    (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 0, 0),
                    2)

        cv2.imshow("Traductor LSM", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()