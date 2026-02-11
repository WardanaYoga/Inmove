import cv2
import mediapipe as mp
import serial
import time
import numpy as np
import math

# ================= SERIAL =================
arduino_right = serial.Serial('COM6', 115200)  # tangan kanan
arduino_left  = serial.Serial('COM4', 115200)  # tangan kiri
time.sleep(2)

# ================= MEDIAPIPE =================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# ================= HELPER =================
def distance(a, b):
    return math.sqrt(
        (a.x - b.x) ** 2 +
        (a.y - b.y) ** 2 +
        (a.z - b.z) ** 2
    )

def map_value(x, in_min, in_max, out_min, out_max):
    return int(np.interp(x, [in_min, in_max], [out_min, out_max]))

# ================= MAIN LOOP =================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(result.multi_hand_landmarks):
            hand_label = result.multi_handedness[idx].classification[0].label
            lm = hand_landmarks.landmark

            # Hitung jarak tiap jari
            thumb  = distance(lm[2],  lm[4])
            index  = distance(lm[5],  lm[8])
            middle = distance(lm[9],  lm[12])
            ring   = distance(lm[13], lm[16])
            pinky  = distance(lm[17], lm[20])

            # Mapping jarak → servo angle
            servo = [
                map_value(thumb,  0.02, 0.10, 180, 0),
                map_value(index,  0.02, 0.12, 180, 0),
                map_value(middle, 0.02, 0.12, 180, 0),
                map_value(ring,   0.02, 0.12, 180, 0),
                map_value(pinky,  0.02, 0.10, 180, 0),
            ]

            servo = [max(0, min(180, s)) for s in servo]
            data = ','.join(map(str, servo)) + '\n'

            # Kirim ke Arduino sesuai tangan
            if hand_label == 'Right':
                arduino_right.write(data.encode())
            else:
                arduino_left.write(data.encode())

            mp_draw.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
            )

    cv2.imshow("Hand Tracking Servo Control", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
