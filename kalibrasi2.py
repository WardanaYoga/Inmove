import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import mediapipe as mp
import serial
import time
import numpy as np
import math
import logging

logging.getLogger('mediapipe').setLevel(logging.ERROR)

# ================= SERIAL =================
arduino_right = serial.Serial('COM4', 115200, timeout=0)
arduino_left  = serial.Serial('COM6', 115200, timeout=0)

time.sleep(2)

arduino_right.reset_input_buffer()
arduino_right.reset_output_buffer()
arduino_left.reset_input_buffer()
arduino_left.reset_output_buffer()

# ================= MEDIAPIPE =================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,
    model_complexity=0,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# ================= HELPER =================
def distance(a, b):
    return math.sqrt(
        (a.x - b.x) ** 2 +
        (a.y - b.y) ** 2 +
        (a.z - b.z) ** 2
    )

def map_value(x, in_min, in_max, out_min, out_max):
    return int(np.interp(x, [in_min, in_max], [out_min, out_max]))

# ================= KALIBRASI =================

# ---- KANAN ----
MIN_R = [0.084345, 0.039253, 0.052669, 0.038520, 0.026840]
MAX_R = [0.143778, 0.220916, 0.238488, 0.220815, 0.173168]

# ---- KIRI ----
MIN_L = [0.075956, 0.043887, 0.055661, 0.047455, 0.026855]
MAX_L = [0.117637, 0.186761, 0.212108, 0.198718, 0.154956]

# ================= FILTER & CONTROL =================
alpha = 0.3
servo_smooth_r = [90]*5
servo_smooth_l = [90]*5

last_sent_r = [0]*5
last_sent_l = [0]*5

SEND_INTERVAL = 0.05  # 20Hz
CHANGE_THRESHOLD = 3
last_send_time = 0

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

            distances = [
                distance(lm[2],  lm[4]),
                distance(lm[5],  lm[8]),
                distance(lm[9],  lm[12]),
                distance(lm[13], lm[16]),
                distance(lm[17], lm[20])
            ]

            servo_raw = []

            # ===== Mapping sesuai tangan =====
            if hand_label == "Right":
                for i in range(5):
                    angle = map_value(distances[i], MIN_R[i], MAX_R[i], 170, 10)
                    angle = max(10, min(170, angle))
                    servo_raw.append(angle)
            else:
                for i in range(5):
                    angle = map_value(distances[i], MIN_L[i], MAX_L[i], 170, 10)
                    angle = max(10, min(170, angle))
                    servo_raw.append(angle)

            # ===== Smoothing =====
            if hand_label == "Right":
                for i in range(5):
                    servo_smooth_r[i] = int(
                        alpha * servo_raw[i] +
                        (1 - alpha) * servo_smooth_r[i]
                    )
                servo_send = servo_smooth_r
            else:
                for i in range(5):
                    servo_smooth_l[i] = int(
                        alpha * servo_raw[i] +
                        (1 - alpha) * servo_smooth_l[i]
                    )
                servo_send = servo_smooth_l

            # ===== Anti Flood Serial =====
            current_time = time.time()

            if current_time - last_send_time > SEND_INTERVAL:

                if hand_label == "Right":
                    diff = [abs(servo_send[i] - last_sent_r[i]) for i in range(5)]
                    if any(d > CHANGE_THRESHOLD for d in diff):
                        data = ','.join(map(str, servo_send)) + '\n'
                        arduino_right.write(data.encode())
                        last_sent_r = servo_send.copy()

                else:
                    diff = [abs(servo_send[i] - last_sent_l[i]) for i in range(5)]
                    if any(d > CHANGE_THRESHOLD for d in diff):
                        data = ','.join(map(str, servo_send)) + '\n'
                        arduino_left.write(data.encode())
                        last_sent_l = servo_send.copy()

                last_send_time = current_time

            mp_draw.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
            )

    cv2.imshow("Dual Hand Tracking Stable", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
arduino_right.close()
arduino_left.close()
