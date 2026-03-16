import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def dedo_extendido(tip, pip, lm):
    return lm[tip].y < lm[pip].y


def pulgar_extendido(lm, mano):
    # thumb_tip = 4
    # index_mcp = 5

    thumb_tip = lm[4].x
    index_mcp = lm[5].x

    if mano == "Right":
        return thumb_tip < index_mcp
    else:  # Left
        return thumb_tip > index_mcp



cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as hands:

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            hand_label = "Unknown"

            # Detectar si es mano izquierda o derecha
            if result.multi_handedness:
                hand_label = result.multi_handedness[0].classification[0].label

            for hand in result.multi_hand_landmarks:

                mp_drawing.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
                lm = hand.landmark

                dedos = []

                # Pulgar según la mano
                dedos.append(1 if pulgar_extendido(lm, hand_label) else 0)

                # Índice, medio, anular, meñique
                tips = [8, 12, 16, 20]
                pips = [6, 10, 14, 18]

                for t, p in zip(tips, pips):
                    dedos.append(1 if dedo_extendido(t, p, lm) else 0)

                # Mostrar el vector
                cv2.putText(frame, f"Dedos: {dedos} | Mano: {hand_label}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

                # Clasificacion simple
                if dedos == [0,0,0,0,0]:
                    texto = "A / Punio"
                elif dedos == [1,1,1,1,1]:
                    texto = "5"
                elif dedos == [0,1,0,0,0]:
                    texto = "1"
                elif dedos == [0,1,1,0,0]:
                    texto = "2"
                elif dedos == [0,1,1,1,0]:
                    texto = "3"
                elif dedos == [0,1,1,1,1]:
                    texto = "4"
                else:
                    texto = "Desconocido"

                cv2.putText(frame, f"Senia: {texto}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        cv2.imshow("Dia 5 - Ambas manos", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
