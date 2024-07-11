#include "Arduino.h"

#define STEPPER 9
#define STEPPER_STEP 12

#define STEPPER_MICROSENCOND (unsigned long)80000 * 8

bool l1 = false;
bool l2 = false;

void setup() {

  Serial.begin(9600);
  pinMode(STEPPER, OUTPUT);
  pinMode(STEPPER_STEP, OUTPUT);
  digitalWrite(STEPPER, LOW);
  pinMode(2, OUTPUT);
  pinMode(3, OUTPUT);

  Serial.println("START STEPPER ON LOOP");
}

void loop() {
  // read the command from the serial port
  // // "l1" -> laser(2)
  // // "l2" -> laser(3)
  // // "step:n" -> stepper_step
  // // "df" -> stepper(0)
  // // "db" -> stepper(1)

  while (Serial.available() == 0) {
  }
  String data = Serial.readString();

  data.trim();
  if (data == "l1") {
    l1 = !l1;
    digitalWrite(2, l1);
  } 
  else if (data == "l2") {
    l2 = !l2;
    digitalWrite(3, l2);
  } 
  else if (data.startsWith("step:")) {
    Serial.println(data.substring(5));
    int n = data.substring(5).toInt();
    for (int i = 0; i < n ; i++) {
      stepper_step();
      Serial.println("step:1");
      delay(100);
    }
  } 
  else if (data == "df") {
    digitalWrite(STEPPER, LOW);
  } 
  else if (data == "db") {
    digitalWrite(STEPPER, HIGH);
  }
}




void stepper_step() {
  for (int j = 0; j < 10; j++) {
    digitalWrite(STEPPER_STEP, HIGH);
    delayMicroseconds(STEPPER_MICROSENCOND);
    digitalWrite(STEPPER_STEP, LOW);
    delayMicroseconds(STEPPER_MICROSENCOND);
  }
}