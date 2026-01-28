void setup() {
  Serial.begin(9600);
  Serial.println("Arduino ready (zone = pin)");
}

void loop() {
  if (Serial.available()) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();

    if (cmd.startsWith("Z")) {
      int colonIndex = cmd.indexOf(':');
      if (colonIndex > 1) {
        int pin = cmd.substring(1, colonIndex).toInt() + 2;
        int state = cmd.substring(colonIndex + 1).toInt();

        // Basic safety check
        if (pin >= 2 && pin <= 11) {   // adjust max pin if needed
          pinMode(pin, OUTPUT);
          digitalWrite(pin, state ? HIGH : LOW);

          Serial.print("Pin ");
          Serial.print(pin);
          Serial.print(" -> ");
          Serial.println(state ? "ON" : "OFF");
        } else {
          Serial.println("Invalid pin number");
        }
      }
    }
  }
}
