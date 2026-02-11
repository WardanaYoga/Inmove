import cv2
import mediapipe as mp
import serial
import time
import numpy as np
import math

# ================= SERIAL =================
#arduino_right = serial.Serial('COM6', 115200)
arduino_left  = serial.Serial('COM4', 115200)
time.sleep(2)

# ================= MEDIAPIPE =================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,
    model_complexity=0,  # lebih ringan
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

# ================= KALIBRASI =================
mode_kalibrasi = True
print("Tekan O (open), C (close), S (stop kalibrasi)")

calib_min = {"thumb": 999, "index": 999, "middle": 999, "ring": 999, "pinky": 999}
calib_max = {"thumb": 0, "index": 0, "middle": 0, "ring": 0, "pinky": 0}

# ================= MAIN LOOP =================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    key = cv2.waitKey(1) & 0xFF  # HANYA SEKALI

    if result.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(result.multi_hand_landmarks):
            hand_label = result.multi_handedness[idx].classification[0].label
            lm = hand_landmarks.landmark

            thumb  = distance(lm[2],  lm[4])
            index  = distance(lm[5],  lm[8])
            middle = distance(lm[9],  lm[12])
            ring   = distance(lm[13], lm[16])
            pinky  = distance(lm[17], lm[20])

            # ===== KALIBRASI =====
            if mode_kalibrasi:
                if key == ord('o'):
                    calib_max["thumb"]  = max(calib_max["thumb"], thumb)
                    calib_max["index"]  = max(calib_max["index"], index)
                    calib_max["middle"] = max(calib_max["middle"], middle)
                    calib_max["ring"]   = max(calib_max["ring"], ring)
                    calib_max["pinky"]  = max(calib_max["pinky"], pinky)
                    print("OPEN captured")

                if key == ord('c'):
                    calib_min["thumb"]  = min(calib_min["thumb"], thumb)
                    calib_min["index"]  = min(calib_min["index"], index)
                    calib_min["middle"] = min(calib_min["middle"], middle)
                    calib_min["ring"]   = min(calib_min["ring"], ring)
                    calib_min["pinky"]  = min(calib_min["pinky"], pinky)
                    print("CLOSE captured")

                if key == ord('s'):
                    mode_kalibrasi = False
                    print("=== HASIL KALIBRASI ===")
                    print("MIN:", calib_min)
                    print("MAX:", calib_max)

            # ===== MAPPING =====
            if not mode_kalibrasi:
                servo = [
                    map_value(thumb,  calib_min["thumb"],  calib_max["thumb"], 180, 0),
                    map_value(index,  calib_min["index"],  calib_max["index"], 180, 0),
                    map_value(middle, calib_min["middle"], calib_max["middle"],180, 0),
                    map_value(ring,   calib_min["ring"],   calib_max["ring"],  180, 0),
                    map_value(pinky,  calib_min["pinky"],  calib_max["pinky"], 180, 0),
                ]

                servo = [max(0, min(180, s)) for s in servo]
                data = ','.join(map(str, servo)) + '\n'

                if hand_label == 'Right':
                    arduino_right.write(data.encode())
                else:
                    arduino_left.write(data.encode())

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Hand Tracking Servo Control", frame)

    if key == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
