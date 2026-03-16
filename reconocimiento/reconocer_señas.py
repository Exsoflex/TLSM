import cv2
import mediapipe as mp
import joblib
import numpy as np
from collections import deque, Counter
import pyttsx3
import threading
import time

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
buffer_predicciones = deque(maxlen=10)

# texto formado
texto = ""

# control de letra actual
letra_actual = ""

# parámetros de estabilidad
MIN_VOTOS = 7
MIN_CONFIANZA = 0.60

COOLDOWN = 0.6
ultimo_registro = time.time()


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
        confianza_mostrada = 0


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

                # predicción del modelo
                probabilidades = modelo.predict_proba(datos)[0]

                # verificar si el modelo está dudando
                top2 = np.sort(probabilidades)[-2:]

                if (top2[1] - top2[0]) < 0.15:
                    buffer_predicciones.clear()
                    continue

                indice = np.argmax(probabilidades)

                letra_predicha = modelo.classes_[indice]
                confianza = probabilidades[indice]

                confianza_mostrada = confianza

                # filtro de confianza
                if confianza > MIN_CONFIANZA:

                    buffer_predicciones.append(letra_predicha)

                else:

                    buffer_predicciones.clear()
                    continue


                # detectar transición entre señas
                if len(buffer_predicciones) >= 7:

                    ultimas = list(buffer_predicciones)[-5:]

                    if len(set(ultimas)) > 1:
                        buffer_predicciones.clear()


                # verificar estabilidad
                if len(buffer_predicciones) == buffer_predicciones.maxlen:

                    conteo = Counter(buffer_predicciones)
                    letra_mas_comun, votos = conteo.most_common(1)[0]

                    tiempo_actual = time.time()

                    if votos >= MIN_VOTOS and (tiempo_actual - ultimo_registro) > COOLDOWN:

                        letra_mostrada = letra_mas_comun

                        if letra_mas_comun != letra_actual:

                            if letra_actual != "":
                                texto += letra_actual

                            letra_actual = letra_mas_comun
                            ultimo_registro = tiempo_actual

                            buffer_predicciones.clear()


        else:
            # si la mano desaparece, guardar última letra
            if letra_actual != "":
                texto += letra_actual
                letra_actual = ""

            buffer_predicciones.clear()


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

        # mostrar confianza
        cv2.putText(
            frame,
            f"Confianza: {confianza_mostrada:.2f}",
            (10, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0,255,255),
            2
        )

        # mostrar texto formado
        cv2.putText(
            frame,
            f"Texto: {texto}",
            (10, 120),
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

        # hablar
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