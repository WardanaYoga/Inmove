#include <Servo.h>

Servo servo[5];
int pins[5] = {3, 5, 6, 9, 10};

void setup() {
  Serial.begin(115200);
  for (int i = 0; i < 5; i++) {
    servo[i].attach(pins[i]);
    servo[i].write(90);
  }
}

void loop() {
  if (Serial.available()) {
    String data = Serial.readStringUntil('\n');
    int angle[5];

    sscanf(data.c_str(), "%d,%d,%d,%d,%d",
           &angle[0], &angle[1], &angle[2], &angle[3], &angle[4]);

    for (int i = 0; i < 5; i++) {
      servo[i].write(constrain(angle[i], 0, 180));
    }
  }
}
