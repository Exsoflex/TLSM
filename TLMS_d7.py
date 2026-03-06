import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)

with mp_hands.Hands(max_num_hands=1) as hands:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            lm = result.multi_hand_landmarks[0].landmark

            # Punto base: muñeca (landmark 0)
            base_x = lm[0].x
            base_y = lm[0].y

            landmarks_normalizados = []

            for punto in lm:
                x_rel = punto.x - base_x
                y_rel = punto.y - base_y
                landmarks_normalizados.append((x_rel, y_rel))

            # Ejemplo: mostrar coordenada normalizada del índice
            x_i, y_i = landmarks_normalizados[8]
            cv2.putText(frame,
                        f"Indice norm: {x_i:.2f}, {y_i:.2f}",
                        (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2)

        cv2.imshow("Dia 7 - Normalizacion", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
