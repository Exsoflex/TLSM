import cv2
import mediapipe as mp
import numpy as np
import joblib
import time
import pyttsx3
import threading
from collections import deque, Counter

# ==============================
# MODELOS
# ==============================

modelo_estatico = joblib.load("modelos/modelo_señas.pkl")
modelo_movimiento = joblib.load("modelos/modelo_movimiento.pkl")

# ==============================
# VOZ
# ==============================

def hablar(texto):

    def _hablar():
        engine = pyttsx3.init()
        engine.setProperty('rate',150)
        engine.say(texto)
        engine.runAndWait()

    hilo = threading.Thread(target=_hablar)
    hilo.daemon = True
    hilo.start()

# ==============================
# MEDIAPIPE
# ==============================

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# ==============================
# CONFIGURACION
# ==============================

BUFFER_MOVIMIENTO = 20
UMBRAL_MOVIMIENTO = 0.015
MOVIMIENTO_MINIMO_GESTO = 0.04

COOLDOWN_LETRA = 0.7
COOLDOWN_GESTO = 1.2

ultimo_registro = time.time()
ultimo_gesto = 0

MAX_CHARS = 22

# buffers
buffer_pred = deque(maxlen=10)
buffer_frames = deque(maxlen=BUFFER_MOVIMIENTO)
buffer_pos = deque(maxlen=BUFFER_MOVIMIENTO)
buffer_mov_pred = deque(maxlen=3)

# suavizado
buffer_suavizado = deque(maxlen=5)

# texto
texto = ""
letra_actual = ""

# control mano
mano_presente = False
ultimo_tiempo_mano = time.time()

# ==============================
# FUNCIONES
# ==============================

def movimiento(p1,p2):

    if p1 is None or p2 is None:
        return 0

    return np.linalg.norm(np.array(p1)-np.array(p2))


def movimiento_total(buffer):

    total = 0

    for i in range(1,len(buffer)):
        total += movimiento(buffer[i],buffer[i-1])

    return total


# dirección del movimiento (mejora día 21)
def direccion_movimiento(buffer):

    if len(buffer) < 2:
        return (0,0)

    inicio = np.array(buffer[0])
    fin = np.array(buffer[-1])

    vector = fin - inicio

    return vector


def obtener_posicion_representativa(lm):

    puntos = [0,8,12]

    xs = [lm[i].x for i in puntos]
    ys = [lm[i].y for i in puntos]

    pos = (np.mean(xs), np.mean(ys))

    buffer_suavizado.append(pos)

    xs = [p[0] for p in buffer_suavizado]
    ys = [p[1] for p in buffer_suavizado]

    return (np.mean(xs),np.mean(ys))


# ==============================
# AJUSTE DE TEXTO AL PANEL
# ==============================

def dividir_texto(texto, ancho_max, font, escala, grosor):

    palabras = texto.split(" ")
    lineas = []
    linea_actual = ""

    for palabra in palabras:

        prueba = linea_actual + palabra + " "

        (w, h), _ = cv2.getTextSize(prueba, font, escala, grosor)

        if w <= ancho_max:
            linea_actual = prueba
        else:
            lineas.append(linea_actual)
            linea_actual = palabra + " "

    lineas.append(linea_actual)

    return lineas


# ==============================
# CAMARA
# ==============================

cap = cv2.VideoCapture(1)

with mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
) as hands:

    pos_anterior = None

    while True:

        inicio_frame = time.time()

        ret,frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame,1)

        rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

        resultado = hands.process(rgb)

        letra_detectada = ""

        if resultado.multi_hand_landmarks:

            mano_presente = True
            ultimo_tiempo_mano = time.time()

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

                for p in lm:

                    x = p.x - muñeca_x
                    y = p.y - muñeca_y

                    datos.extend([x,y])

                datos = np.array(datos)

                # ======================
                # POSICION REPRESENTATIVA
                # ======================

                pos_actual = obtener_posicion_representativa(lm)

                mov = movimiento(pos_actual,pos_anterior)

                pos_anterior = pos_actual

                buffer_pos.append(pos_actual)

                # ======================
                # DETECTAR MOVIMIENTO
                # ======================

                if mov > UMBRAL_MOVIMIENTO:

                    buffer_frames.append(datos)

                # ======================
                # MODELO MOVIMIENTO
                # ======================

                if len(buffer_frames) == BUFFER_MOVIMIENTO:

                    mov_total = movimiento_total(buffer_pos)

                    vector = direccion_movimiento(buffer_pos)

                    if mov_total > MOVIMIENTO_MINIMO_GESTO and time.time()-ultimo_gesto > COOLDOWN_GESTO:

                        frames = list(buffer_frames)

                        entrada = np.array(frames).flatten().reshape(1,-1)

                        pred = modelo_movimiento.predict(entrada)[0]

                        buffer_mov_pred.append(pred)

                        if len(buffer_mov_pred) == buffer_mov_pred.maxlen:

                            conteo = Counter(buffer_mov_pred)

                            pred_final,votos = conteo.most_common(1)[0]

                            if votos >= 2:

                                letra_detectada = pred_final
                                ultimo_gesto = time.time()

                            buffer_mov_pred.clear()

                    buffer_frames.clear()

                else:

                    # ======================
                    # MODELO ESTATICO
                    # ======================

                    entrada = datos.reshape(1,-1)

                    probs = modelo_estatico.predict_proba(entrada)[0]

                    idx = np.argmax(probs)

                    confianza = probs[idx]

                    if confianza > 0.65:

                        letra_pred = modelo_estatico.classes_[idx]

                        buffer_pred.append(letra_pred)

                # ======================
                # ESTABILIDAD
                # ======================

                if len(buffer_pred) == buffer_pred.maxlen:

                    conteo = Counter(buffer_pred)

                    letra,votos = conteo.most_common(1)[0]

                    if votos >= 8:

                        letra_detectada = letra
                        buffer_pred.clear()

        else:

            if mano_presente:
                mano_presente = False
                ultimo_tiempo_mano = time.time()

            if not mano_presente:

                if time.time() - ultimo_tiempo_mano > 2:

                    if not texto.endswith(" "):
                        texto += " "

                    ultimo_tiempo_mano = time.time()

            buffer_pred.clear()
            buffer_frames.clear()
            buffer_pos.clear()

        # ==============================
        # AGREGAR LETRA AL TEXTO
        # ==============================

        tiempo = time.time()

        if letra_detectada != "" and (tiempo-ultimo_registro)>COOLDOWN_LETRA:

            if len(texto)==0 or texto[-1] != letra_detectada:

                texto += letra_detectada

                letra_actual = letra_detectada

                ultimo_registro = tiempo

        # ==============================
        # PANEL TEXTO MEJORADO
        # ==============================

        panel = np.zeros((480,400,3),dtype=np.uint8)

        cursor = "_"
        texto_mostrar = texto + cursor

        cv2.putText(panel,"TEXTO:",(20,50),
                    cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)

        # dividir texto en varias lineas
        lineas = dividir_texto(
            texto_mostrar,
            360,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            2
        )

        y = 120

        for linea in lineas[-6:]:  # máximo 6 líneas visibles

            cv2.putText(panel,linea,(20,y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0,255,0),
                        2)

            y += 40

        cv2.putText(panel,f"Detectado: {letra_detectada}",
                    (20,200),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0,200,255),
                    2)

        cv2.putText(panel,"ESPACIO = espacio",(20,300),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,(200,200,200),1)

        cv2.putText(panel,"ENTER = hablar",(20,330),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,(200,200,200),1)

        cv2.putText(panel,"C = limpiar",(20,360),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,(200,200,200),1)

        cv2.putText(panel,"ESC = salir",(20,390),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,(200,200,200),1)

        pantalla = np.hstack((frame,panel))

        # FPS
        fps = int(1/(time.time()-inicio_frame))
        cv2.putText(frame,f"FPS: {fps}",(10,30),
                    cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)

        cv2.imshow("Traductor LSM",pantalla)

        tecla = cv2.waitKey(1) & 0xFF

        if tecla == 32:
            texto += " "

        elif tecla == 8:
            texto = texto[:-1]

        elif tecla == 13:
            hablar(texto)

        elif tecla == ord('c'):
            texto = ""

        elif tecla == 27:
            break

cap.release()
cv2.destroyAllWindows()