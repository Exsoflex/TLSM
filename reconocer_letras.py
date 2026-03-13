import cv2
import mediapipe as mp
import joblib
import numpy as np
from collections import deque, Counter

# cargar modelo entrenado
modelo = joblib.load("modelo_señas.pkl")

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# buffer para estabilidad
buffer_predicciones = deque(maxlen=15)

with mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
) as hands:

    while True:

        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        resultado = hands.process(rgb)

        letra_mostrada = ""

        if resultado.multi_hand_landmarks:

            for hand_landmarks in resultado.multi_hand_landmarks:

                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

                # coordenadas de la muñeca
                muñeca_x = hand_landmarks.landmark[0].x
                muñeca_y = hand_landmarks.landmark[0].y

                datos = []

                for lm in hand_landmarks.landmark:
                    datos.append(lm.x - muñeca_x)
                    datos.append(lm.y - muñeca_y)

                datos = np.array(datos).reshape(1, -1)

                prediccion = modelo.predict(datos)

                buffer_predicciones.append(prediccion[0])

                if len(buffer_predicciones) == buffer_predicciones.maxlen:
                    letra_mas_comun = Counter(buffer_predicciones).most_common(1)[0][0]
                    letra_mostrada = letra_mas_comun

        if letra_mostrada != "":
            cv2.putText(
                frame,
                f"Letra: {letra_mostrada}",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (0, 255, 0),
                3
            )

        cv2.imshow("Reconocimiento de señas", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()