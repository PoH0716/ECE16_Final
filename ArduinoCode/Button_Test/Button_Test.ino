const int button1Pin = 11;
const int button2Pin = 12;
int button1State = 0;
int button2State = 0;

void setup() {
  pinMode(button1Pin, INPUT);
  pinMode(button2Pin, INPUT);
  Serial.begin(9600);
}

void loop() {
  button1State = digitalRead(button1Pin);
  button2State = digitalRead(button2Pin);

  if (button1State == HIGH) {
    Serial.println("button 1 pressed");
  }
  if (button2State == HIGH) {
    Serial.println("button 2 pressed");
  }
}
