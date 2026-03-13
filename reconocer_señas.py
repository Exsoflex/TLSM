import cv2
import mediapipe as mp
import joblib
import numpy as np
from collections import deque, Counter
import pyttsx3
import threading

def hablar(texto):

    def _hablar():
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)
        engine.say(texto)
        engine.runAndWait()
        engine.stop()

    hilo = threading.Thread(target=_hablar)
    hilo.daemon = True
    hilo.start()

# cargar modelo entrenado
modelo = joblib.load("modelo_señas.pkl")

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# buffer para estabilidad
buffer_predicciones = deque(maxlen=15)

# texto formado
texto = ""

# control de letra actual
letra_actual = ""
letra_confirmada = ""

# estabilidad mínima requerida
MIN_VOTOS = 10

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

                    conteo = Counter(buffer_predicciones)
                    letra_mas_comun, votos = conteo.most_common(1)[0]

                    # aplicar filtro de estabilidad
                    if votos >= MIN_VOTOS:

                        letra_mostrada = letra_mas_comun

                        # si cambia la seña, guardar la anterior
                        if letra_mas_comun != letra_actual:

                            if letra_actual != "":
                                texto += letra_actual

                            letra_actual = letra_mas_comun

        else:
            # si la mano desaparece, guardar última letra
            if letra_actual != "":
                texto += letra_actual
                letra_actual = ""

        # mostrar letra detectada
        cv2.putText(
            frame,
            f"Letra actual: {letra_mostrada}",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 0),
            3
        )

        # mostrar texto formado
        cv2.putText(
            frame,
            f"Texto: {texto}",
            (10, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (255, 255, 255),
            2
        )

        cv2.imshow("Reconocimiento de señas", frame)

        tecla = cv2.waitKey(1) & 0xFF

        # espacio
        if tecla == 32:
            texto += " "

        # borrar
        elif tecla == 8:
            texto = texto[:-1]

        # hablar (reiniciando motor)
        elif tecla == 13:
            hablar(texto)

        # limpiar texto
        elif tecla == ord('c'):
            texto = ""

        # salir
        elif tecla == 27:
            break

cap.release()
cv2.destroyAllWindows()