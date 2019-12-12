// IMU library includes
#include "I2Cdev.h"
#include "MPU6050.h"
#include "Wire.h"

// BT Library
#include <AltSoftSerial.h>
AltSoftSerial hm10; // create the AltSoftSerial connection for the HM10

// Global Variables
char in_text[64];
bool bleConnected = false;
unsigned long sendTimer = 0;

// Define button pin variables
const int buttonPin = 3;
bool asleep = false;
volatile bool buttonPressed = false;

// IMU data variables
int16_t ax, ay, az, gx, gy, gz;
long Aax = 0;
long Aay = 0;
long Aaz = 0;
char Data[20];

// IMU setup
const int MPU_addr=0x68;    // I2C address of the MPU-6050
MPU6050 IMU(MPU_addr);      // Instantiate IMU object

// --------------------------------------------------------------------------------
// This function handles the BLE handshake
// It detects if central is sending "AT+...", indicating the handshake is not complete
// If a "T" is received right after an "A", we send back the handshake confirmation
// The function returns true if a connection is established and false otherwise
// --------------------------------------------------------------------------------
bool bleHandshake(char input) {
  static char lastChar;

  if(lastChar == 'A' && input == 'T'){
    hm10.print('#');
    delay(50);
    hm10.flushInput();
    lastChar = "";
    bleConnected = true;
    return true;
    Serial.println(bleConnected);
  }
  lastChar = input;
  return false;
}

// --------------------------------------------------------------------------------
// This function reads characters from the HM-10
// It calls the bleHandshake() function to see if we are connecting
// Otherwise, it fills a buffer "in_text" until we see a ";" (our newline stand-in)
// --------------------------------------------------------------------------------
bool readBLE() {
  static int i = 0;
  char c = hm10.read();

  if (bleHandshake(c)) {
    i = 0;
  }
  else {
    // If the buffer overflows, go back to its beginning
    if (i >= sizeof(in_text)-1)
      i = 0;
  
    // All of our messages will terminate with ';' instead of a newline
    if (c == ';') {
      in_text[i] = '\0'; // terminate the string
      i = 0;
      return true;
    }
    else
      in_text[i++] = c;
  }

  return false;
}

// --------------------------------------------------------------------------------
// Forward data from Serial to the BLE module
// This is useful to set the modes of the BLE module
// --------------------------------------------------------------------------------
void writeBLE() {
  static boolean newline = true;
  
  while (Serial.available()) {
    char c = Serial.read();
  
    // We cannot send newline to the HM-10 so we have to catch it
    if (c!='\n' & c!='\r')
      hm10.print(c);

    // Also print to Serial Monitor so we can see what we typed
    // If there is a new line character, print the ">" character
    if (newline) {
      Serial.print("\n>");
      newline = false;
    }

    Serial.print(c);
    if (c=='\n')
      newline = true;
  }
}

void buttonISR(){
  static unsigned long lastInterrupt = 0;
  unsigned long interruptTime = micros();
  if(interruptTime - lastInterrupt > 200)
    buttonPressed = true;
  lastInterrupt = interruptTime;
}

// --------------------------------------------------------------------------------
// This function makes the BLE sleep or wake up when the button is pressed
// --------------------------------------------------------------------------------
void toggleSleep() {
  //complete the if statement conditional for when the system should be going to sleep
  if(buttonPressed) {
    asleep = !asleep;
    if(asleep) {
      Serial.println("Going to sleep!");
      hm10.write("AT");
      delay(300);
      hm10.write("AT+ADTY3");
      delay(300);
      hm10.write("AT+SLEEP");
      bleConnected = false;
      asleep = true;
    }
    else {
      Serial.println("Waking up!");
      hm10.write("AT+hailramsinhailramsinhailramsinhailramsinhailramsinhailramsinhailramsinhailramsin");
      delay(300);
      hm10.write("AT+ADTY0");
      delay(300);
      hm10.write("AT+RESET");
      asleep = false;
    }
    
    buttonPressed = false;
  }
}

void initIMU() {
  IMU.initialize();
  Wire.begin();
  Wire.beginTransmission(MPU_addr);
  Wire.write(MPU_addr);               // PWR_MGMT_1 register
  Wire.write(0);                      // Set to zero (wakes up the MPU-6050)
  Wire.endTransmission(true);
}

bool getData() {
  bool newData = false;
  IMU.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);
  newData = true;
  return newData;
}

void readAxis() {
  Aax = abs(ax);
  Aay = abs(ay);
  Aaz = abs(az);
}

void sendData() {
  readAxis();
  sprintf(Data, "%5ld,%5ld,%5ld;", Aax, Aay, Aaz);
  Serial.print(Data);
  hm10.print(Data);
}

// --------------------------------------------------------------------------------
// Setup: executed once at startup or reset
// --------------------------------------------------------------------------------
void setup() {
  pinMode(buttonPin, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(buttonPin), buttonISR, FALLING);

  initIMU();
  Serial.begin(9600);
  hm10.begin(9600);
  
  Serial.println("==============================");
  Serial.println("BLE Handshake Code Started");
  Serial.println("==============================");
} 

// --------------------------------------------------------------------------------
// Loop: main code; executed in an infinite loop
// --------------------------------------------------------------------------------
void loop() {
  toggleSleep();
  
  if (hm10.available()) {
    if (readBLE()) {
      Serial.println(in_text);
    }
  }
  
  if (Serial.available()) {
    writeBLE();
  }

  if (bleConnected && !asleep){
    if(getData()) {
      if (micros() - sendTimer > 40000) {
        sendTimer = micros();
        sendData();  
      }
    }
  }
}
