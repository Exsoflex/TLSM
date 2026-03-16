import cv2
import mediapipe as mp
import numpy as np
import joblib
from collections import defaultdict
import os
import time

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(1)

archivo_dataset = "dataset_movimiento.pkl"

frames_objetivo = 20
capturando = False
secuencia = []

cooldown_frames = 0.06  # tiempo entre frames (60 ms)
ultimo_frame = 0

# cargar dataset si ya existe
if os.path.exists(archivo_dataset):
    dataset = joblib.load(archivo_dataset)
else:
    dataset = []

conteo = defaultdict(int)

for secuencia, etiqueta in dataset:
    conteo[etiqueta] += 1

etiquetas = {
    ord('1'): "J",
    ord('2'): "K",
    ord('3'): "N_tilde",
    ord('4'): "Hola"
}

etiqueta_actual = "J"

frames_objetivo = 20
capturando = False
secuencia = []


def coordenadas_relativas(hand_landmarks):

    puntos = []

    muñeca_x = hand_landmarks.landmark[0].x
    muñeca_y = hand_landmarks.landmark[0].y

    for lm in hand_landmarks.landmark:

        x_rel = lm.x - muñeca_x
        y_rel = lm.y - muñeca_y

        puntos.append(x_rel)
        puntos.append(y_rel)

    return puntos


with mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7) as hands:

    while True:

        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        resultado = hands.process(rgb)

        if resultado.multi_hand_landmarks:

            for hand_landmarks in resultado.multi_hand_landmarks:

                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

                if capturando:

                    tiempo_actual = time.time()

                    if tiempo_actual - ultimo_frame > cooldown_frames:

                        coords = coordenadas_relativas(hand_landmarks)

                        secuencia.append(coords)

                        ultimo_frame = tiempo_actual

                    cv2.putText(frame,
                                f"Capturando {len(secuencia)}/{frames_objetivo}",
                                (10,40),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1,
                                (0,255,0),
                                2)

                    if len(secuencia) == frames_objetivo:

                        dataset.append([secuencia.copy(), etiqueta_actual])
                        conteo[etiqueta_actual] += 1

                        joblib.dump(dataset, archivo_dataset)

                        print("Secuencia guardada:", etiqueta_actual)

                        secuencia = []
                        capturando = False

        else:

            cv2.putText(frame,
                        "Mano no detectada",
                        (10,40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0,0,255),
                        2)

        # mostrar etiqueta actual
        cv2.putText(frame,
                    f"Etiqueta: {etiqueta_actual}",
                    (10,80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (255,255,0),
                    2)

        # mostrar conteo
        y = 120
        for et, num in conteo.items():

            cv2.putText(frame,
                        f"{et}: {num}",
                        (10,y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255,255,255),
                        2)

            y += 30

        cv2.putText(frame,
                    "S = capturar | 1-4 cambiar seña | ESC salir",
                    (10,420),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (200,200,200),
                    2)

        cv2.imshow("Captura movimiento", frame)

        key = cv2.waitKey(1)

        # cambiar etiqueta
        if key in etiquetas:
            etiqueta_actual = etiquetas[key]
            print("Etiqueta cambiada a:", etiqueta_actual)

        # iniciar captura
        if key == ord('s') and not capturando:
            capturando = True
            secuencia = []
            print("Capturando movimiento...")

        # salir
        if key == 27:
            break


cap.release()
cv2.destroyAllWindows()

joblib.dump(dataset, archivo_dataset)

print("Dataset guardado correctamente")