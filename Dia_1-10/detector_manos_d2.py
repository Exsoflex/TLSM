import cv2
import mediapipe as mp

# Inicializar MediaPipe
mpManos = mp.solutions.hands
manos = mpManos.Hands(static_image_mode=False,
                      max_num_hands=1,
                      min_detection_confidence=0.5)
mpDibujo = mp.solutions.drawing_utils

# Activar cámara
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Procesar mano
    resultados = manos.process(frameRGB)

    if resultados.multi_hand_landmarks:
        for mano in resultados.multi_hand_landmarks:
            
            # Dibujar puntos
            mpDibujo.draw_landmarks(frame, mano, mpManos.HAND_CONNECTIONS)

            # Extraer coordenadas
            for i, lm in enumerate(mano.landmark):
                h, w, c = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                print(f"Landmark {i}: x={cx} y={cy}")

    cv2.imshow("Mano", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
