// AI-Based Chilli Plant Health & Smart Irrigation System
// University final-semester project - IoT irrigation module

const int sensorPin = A0;
const int pumpPin = 13;

// Calibrate these values for the actual sensor and soil.
int dryValue = 1023;
int wetValue = 200;

const int pumpOnThreshold = 30;
const int pumpOffThreshold = 70;

void setup() {
  Serial.begin(9600);
  pinMode(sensorPin, INPUT);
  pinMode(pumpPin, OUTPUT);

  // Pump starts OFF.
  digitalWrite(pumpPin, LOW);
  Serial.println("Chilli Plant Smart Irrigation System Started...");
}

void loop() {
  const int sensorValue = analogRead(sensorPin);

  int moisturePercent = map(sensorValue, dryValue, wetValue, 0, 100);
  moisturePercent = constrain(moisturePercent, 0, 100);

  Serial.print("Sensor Value: ");
  Serial.print(sensorValue);
  Serial.print(" | Moisture: ");
  Serial.print(moisturePercent);
  Serial.println("%");

  // Hysteresis prevents rapid ON/OFF switching around one threshold.
  if (moisturePercent < pumpOnThreshold) {
    digitalWrite(pumpPin, HIGH);
    Serial.println("Status: Pump ON - soil is dry.");
  } else if (moisturePercent > pumpOffThreshold) {
    digitalWrite(pumpPin, LOW);
    Serial.println("Status: Pump OFF - soil is sufficiently moist.");
  }

  delay(5000);
}
