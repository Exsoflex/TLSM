import cv2
import mediapipe as mp
import csv
import os

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

archivo = "dataset_señas.csv"

# crear archivo si no existe
if not os.path.exists(archivo):
    with open(archivo, mode="w", newline="") as f:
        writer = csv.writer(f)
        header = []

        for i in range(21):
            header.append(f"x{i}")
            header.append(f"y{i}")

        header.append("letra")
        writer.writerow(header)

cap = cv2.VideoCapture(0)

letra = input("Ingresa la letra que quieres capturar: ").upper()

contador = 0
max_muestras = 300

with mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7) as hands:

    while True:

        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        resultado = hands.process(rgb)

        if resultado.multi_hand_landmarks:

            for hand_landmarks in resultado.multi_hand_landmarks:

                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                fila = []

                # coordenadas de la muñeca (landmark 0)
                muñeca_x = hand_landmarks.landmark[0].x
                muñeca_y = hand_landmarks.landmark[0].y

                for lm in hand_landmarks.landmark:
                    fila.append(lm.x - muñeca_x)
                    fila.append(lm.y - muñeca_y)

                fila.append(letra)

                with open(archivo, mode="a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(fila)

                contador += 1

                cv2.putText(frame, f"Muestras: {contador}",
                            (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0,255,0),
                            2)

                if contador >= max_muestras:
                    break

        cv2.imshow("Recolectando datos", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

        if contador >= max_muestras:
            break

cap.release()
cv2.destroyAllWindows()

print("Datos guardados correctamente.")