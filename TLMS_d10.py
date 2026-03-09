import cv2
import mediapipe as mp
import time

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
# Función para obtener coordenadas relativas a la muñeca
def coordenadas_relativas(lm):

    muñeca_x = lm[0].x
    muñeca_y = lm[0].y

    coords = []

    for punto in lm:
        x_rel = punto.x - muñeca_x
        y_rel = punto.y - muñeca_y
        coords.append((x_rel, y_rel))

    return coords

# Función para saber si un dedo está extendido
def dedo_extendido(tip, pip, coords):

    vertical = coords[tip][1] < coords[pip][1] - 0.02
    horizontal = abs(coords[tip][0] - coords[pip][0]) > 0.04

    return vertical or horizontal

# Función para saber si el pulgar esta extendido
def pulgar_extendido(coords, label):

    horizontal = abs(coords[4][0] - coords[3][0]) > 0.03
    vertical = coords[4][1] < coords[3][1] - 0.02

    return horizontal or vertical


cap = cv2.VideoCapture(0)

texto = ""
cooldown = 1
ultimo_tiempo = 0

max_letras_por_linea = 20

with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7) as hands:

    while True:

        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        resultado = hands.process(rgb)

        letra = ""

        if resultado.multi_hand_landmarks:

            for hand_landmarks, handedness in zip(resultado.multi_hand_landmarks, resultado.multi_handedness):

                label = handedness.classification[0].label

                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

                lm = hand_landmarks.landmark
                
                coords = coordenadas_relativas(lm)
                

                pulgar = pulgar_extendido(coords, label)
                indice = dedo_extendido(8,6,coords)
                medio = dedo_extendido(12,10,coords)
                anular = dedo_extendido(16,14,coords)
                menique = dedo_extendido(20,18,coords)

                dedos = [pulgar, indice, medio, anular, menique]

                cv2.putText(frame, label,
                            (10,80),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0,255,255),
                            2)
                
                cv2.putText(frame, str(dedos),
                            (10,400),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (255,0,0),
                            2)
                
                if dedos[1:] == [0,0,0,0]:
                    letra = "A"

                elif indice and not medio and not anular and not menique:
                    letra = "D"

                elif indice and medio and anular and menique:
                    letra = "B"

                elif pulgar and not indice and not medio and not anular and menique:
                    letra = "Y"

                tiempo_actual = time.time()

                if letra != "" and tiempo_actual - ultimo_tiempo > cooldown:

                    texto += letra
                    ultimo_tiempo = tiempo_actual

        lineas = [texto[i:i+max_letras_por_linea] for i in range(0, len(texto), max_letras_por_linea)]

        y = 50
        for linea in lineas[-5:]:  # muestra solo las ultimas 5 lineas
            cv2.putText(frame, linea,
                        (10,y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0,255,0),
                        2)
            y += 40

        cv2.imshow("Traductor LSM", frame)
            
        tecla = cv2.waitKey(1) & 0xFF

        if tecla == 27:
            break

        if tecla == ord('c'):
            texto = ""

cap.release()
cv2.destroyAllWindows()