import cv2
import mediapipe as mp
import numpy as np
import joblib
import time
from collections import deque, Counter

# ===============================
# CONFIGURACIÓN
# ===============================

VENTANA_PREDICCIONES = 1
BUFFER_FRAMES = 20
COOLDOWN_SEGUNDOS = 1.5
UMBRAL_CONFIANZA = 0.40

UMBRAL_MOVIMIENTO_INICIO = 0.01
UMBRAL_MOVIMIENTO_TOTAL = 0.05

FRAMES_ESTABILIDAD = 2

# ===============================
# CARGAR MODELO
# ===============================

modelo = joblib.load("modelos/modelo_movimiento.pkl")

# ===============================
# MEDIAPIPE
# ===============================

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# ===============================
# BUFFERS
# ===============================

buffer_frames = deque(maxlen=BUFFER_FRAMES)
buffer_posiciones = deque(maxlen=BUFFER_FRAMES)

predicciones = deque(maxlen=VENTANA_PREDICCIONES)

seña_confirmada = ""
ultima_deteccion = 0

estado = "ESPERANDO_MOVIMIENTO"

frames_estables = 0

# ===============================
# FUNCIONES
# ===============================

def calcular_movimiento(p1, p2):

    if p1 is None or p2 is None:
        return 0

    return np.linalg.norm(np.array(p1) - np.array(p2))


def movimiento_total(buffer):

    if len(buffer) < 2:
        return 0

    total = 0

    for i in range(1, len(buffer)):
        total += calcular_movimiento(buffer[i], buffer[i-1])

    return total


# ===============================
# CÁMARA
# ===============================

cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
) as hands:

    posicion_anterior = None

    while True:

        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        resultado = hands.process(rgb)

        pred_actual = ""
        confianza = 0

        if resultado.multi_hand_landmarks:

            for hand_landmarks in resultado.multi_hand_landmarks:

                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

                lm = hand_landmarks.landmark

                muñeca_x = lm[0].x
                muñeca_y = lm[0].y

                datos = []

                for punto in lm:

                    x = punto.x - muñeca_x
                    y = punto.y - muñeca_y

                    datos.extend([x, y])

                # posición de referencia para movimiento
                posicion_actual = (lm[0].x, lm[0].y)

                movimiento = calcular_movimiento(posicion_actual, posicion_anterior)

                posicion_anterior = posicion_actual

                buffer_posiciones.append(posicion_actual)

                # ===============================
                # DETECTAR INICIO DE MOVIMIENTO
                # ===============================

                if estado == "ESPERANDO_MOVIMIENTO":

                    if movimiento > UMBRAL_MOVIMIENTO_INICIO:

                        estado = "CAPTURANDO"

                        buffer_frames.clear()
                        #buffer_posiciones.clear()

                # ===============================
                # CAPTURAR FRAMES DEL GESTO
                # ===============================

                if estado == "CAPTURANDO":

                    buffer_frames.append(datos)

                    if movimiento < UMBRAL_MOVIMIENTO_INICIO:

                        frames_estables += 1
                    else:
                        frames_estables = 0

                    # gesto terminó
                    if frames_estables >= FRAMES_ESTABILIDAD:

                        estado = "PROCESAR"

                # ===============================
                # PROCESAR GESTO
                # ===============================
                if estado == "PROCESAR":

                    mov_total = movimiento_total(buffer_posiciones)
                    
                    print("Frames:", len(buffer_frames), "Movimiento:", mov_total)


                    if len(buffer_frames) >= 10 and mov_total > UMBRAL_MOVIMIENTO_TOTAL:
                        
                        frames = list(buffer_frames)

                        while len(frames) < BUFFER_FRAMES:
                            frames.append(frames[-1])

                        entrada = np.array(frames).flatten().reshape(1, -1)

                        pred = modelo.predict(entrada)[0]

                        if hasattr(modelo, "predict_proba"):

                            probs = modelo.predict_proba(entrada)[0]

                            confianza = max(probs)

                        pred_actual = pred

                        if confianza > UMBRAL_CONFIANZA:

                            predicciones.append(pred)
                            
                        print("Pred:", pred, "Confianza:", confianza)

                    # votación
                    if len(predicciones) == VENTANA_PREDICCIONES:

                        conteo = Counter(predicciones)

                        pred_final = conteo.most_common(1)[0][0]

                        tiempo_actual = time.time()

                        if tiempo_actual - ultima_deteccion > COOLDOWN_SEGUNDOS:

                            seña_confirmada = pred_final
                            ultima_deteccion = tiempo_actual

                    estado = "ESPERANDO_MOVIMIENTO"
                    
                    frames_estables = 0

        else:

            buffer_frames.clear()
            buffer_posiciones.clear()
            predicciones.clear()

        # ===============================
        # MOSTRAR TEXTO
        # ===============================

        cv2.putText(frame,
                    f"Estado: {estado}",
                    (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255,255,0),
                    2)

        cv2.putText(frame,
                    f"Prediccion actual: {pred_actual}",
                    (10,60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0,255,255),
                    2)

        cv2.putText(frame,
                    f"Confianza: {confianza:.2f}",
                    (10,90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255,200,0),
                    2)

        cv2.putText(frame,
                    f"Sena confirmada: {seña_confirmada}",
                    (10,130),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0,255,0),
                    3)

        cv2.imshow("Reconocimiento Movimiento", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()