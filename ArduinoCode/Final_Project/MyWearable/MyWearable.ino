/********************************************************************************
** Bidirectional BLE Communication using a 2-step handshake protocol
*********************************************************************************/

// BT Library
#include <AltSoftSerial.h>
AltSoftSerial hm10; // create the AltSoftSerial connection for the HM10

// IMU library includes
#include "I2Cdev.h"
#include "MPU6050_6Axis_MotionApps20.h"
#include "Wire.h"

// OLED library includes
#include "U8x8lib.h"

// OLED setup
#define OLED_RESET 4 // this value resets the OLED
U8X8_SSD1306_128X32_UNIVISION_HW_I2C u8x8(OLED_RESET);

// Global Variables
char in_text[64];                           // Character buffer
bool bleConnected = false;                  // false == not connected, true == connected
unsigned long sendTimer = 0;                // timer for sending data once connected

const int buttonPin = 3;
bool asleep = false;
volatile bool buttonPressed = false;

// Define IMU pin variables
const int IMUInterrupt = 2;
volatile bool imuDataReady = false;


// IMU data variables
int16_t ax, ay, az, tp, gx, gy, gz;

// IMU setup
const int MPU_addr=0x68;    // I2C address of the MPU-6050
MPU6050 IMU(MPU_addr);      // Instantiate IMU object

// IR PPG setup
const int ppgPin = A2; // Analog input pin for the PPG
int ppgValue = 0;      // Value read from PPG (0-1023)

void IMUInterruptISR() {
  imuDataReady = true;
}

// --------------------------------------------------------------------------------
// Initialize the IMU (only on startup)
// --------------------------------------------------------------------------------
void initIMU() {

  // Initialize the IMU and the DMP (Digital Motion Processor) on the IMU
  IMU.initialize();
  IMU.dmpInitialize();
  IMU.setDMPEnabled(true);

  // Initialize I2C communications
  Wire.begin();
  Wire.beginTransmission(MPU_addr);
  Wire.write(MPU_addr);               // PWR_MGMT_1 register
  Wire.write(0);                      // Set to zero (wakes up the MPU-6050)
  Wire.endTransmission(true);

  // Create an interrupt for pin2, which is connected to the INT pin of the MPU6050
  pinMode(IMUInterrupt, INPUT);
  attachInterrupt(digitalPinToInterrupt(IMUInterrupt), IMUInterruptISR, RISING);
}

// --------------------------------------------------------------------------------
// Function to read a single sample of IMU data
// Currently, this reads 3 acceleration axis, temperature, and 3 gyro axis.
// You should edit this to read only the sensors you end up using.
// For this, you need to edit the number of registers/addresses requested
// --------------------------------------------------------------------------------
void readIMU() {
  Wire.beginTransmission(MPU_addr);
  Wire.write(0x3B);                   // starting with register 0x3B (ACCEL_XOUT_H)
  Wire.endTransmission(false);
  
  Wire.requestFrom(MPU_addr,14,true); // request a total of 14 registers
  
  //Accelerometer (3 Axis)
  ax = Wire.read()<<8|Wire.read();      // 0x3B (ACCEL_XOUT_H) & 0x3C (ACCEL_XOUT_L)    
  ay = Wire.read()<<8|Wire.read();      // 0x3D (ACCEL_YOUT_H) & 0x3E (ACCEL_YOUT_L)
  az = Wire.read()<<8|Wire.read();      // 0x3F (ACCEL_ZOUT_H) & 0x40 (ACCEL_ZOUT_L)
  
  //Temperature
  tp = Wire.read()<<8|Wire.read();      // 0x41 (TEMP_OUT_H) & 0x42 (TEMP_OUT_L)
  
  //Gyroscope (3 Axis)
  gx = Wire.read()<<8|Wire.read();      // 0x43 (GYRO_XOUT_H) & 0x44 (GYRO_XOUT_L)
  gy = Wire.read()<<8|Wire.read();      // 0x45 (GYRO_YOUT_H) & 0x46 (GYRO_YOUT_L)
  gz = Wire.read()<<8|Wire.read();      // 0x47 (GYRO_ZOUT_H) & 0x48 (GYRO_ZOUT_L)
}

// --------------------------------------------------------------------------------
// Function to grab new samples
// --------------------------------------------------------------------------------
bool getData()
{
  bool newData = false;
  if (imuDataReady)
  {
    readIMU();
    newData = true;
  }
  return newData;
}

void sendData(){
  if(getData()){
    char string_buffer[25];
    //sprintf(string_buffer, "%d, %d, %d;\0", ax, ay, az);
    long norm = abs(ax) + abs(ay) + abs(az);
    long timestamp = millis();
    int ppgValue = analogRead(ppgPin);
//    sprintf(string_buffer, "%8ld, %5ld, %5d;\0", timestamp, norm, ppgValue);
    sprintf(string_buffer, "%8ld, %5d, %5d;\0", timestamp, ax, ppgValue);
    hm10.print(string_buffer);
    Serial.println(string_buffer);
//    showMessage(string_buffer, 2, false);
  }
}

// --------------------------------------------------------------------------------
// This function handles the BLE handshake
// It detects if central is sending "AT+...", indicating the handshake is not complete
// If a "T" is received right after an "A", we send back the handshake confirmation
// The function returns true if a connection is established and false otherwise
// --------------------------------------------------------------------------------
bool bleHandshake(char input) {
  static char lastChar;
  
  //  1. Check if lastChar == 'A' and the current input == 'T' (AT was received == we are connected)
  //    1.1. If so, print '#' to the hm10
  //    1.2. Delay for 50 milliseconds to allow the handshake to complete
  //    1.3. Flush the hm10 input buffer (hint: flushInput())
  //    1.4. Reset lastChar
  //    1.5. Set 'bleConnected' to true
  //    1.6. Return true
  //  2. Set lastChar to input
  //  3. Return false
  if(lastChar=='A' && input =='T'){
    hm10.print('#');
    delay(50);
    hm10.flushInput();
    lastChar = 0;
    bleConnected = true;
    return true;
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

  return false; // nothing to print
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

// --------------------------------------------------------------------------------
// will sleep/wake the BLE whenever the button is pressed
// --------------------------------------------------------------------------------
void toggleSleep() {
  //complete the if statement conditional for when the system should be going to sleep
  if(buttonPressed) {
    asleep = !asleep;
    if(asleep) {
      Serial.print("Going to sleep");
      hm10.write("AT");
      delay(300);
      hm10.write("AT+ADTY3");
      delay(300);
      hm10.write("AT+SLEEP");
      asleep = true;
    }
    // Complete the if statement for when the system should be waking up
    else {
      Serial.println("Waking up!");
      hm10.write("AT+hello");
      delay(300);
      hm10.write("AT+ADTY0");
      delay(300);
      hm10.write("AT+RESET");
      asleep = false;
      Serial.print("Awake");
    }
    
    buttonPressed = false;
  }
}

void buttonInterruptISR(){
  static unsigned long lastInterrupt = 0;
  unsigned long interruptTime = millis();
  if(interruptTime - lastInterrupt > 200)
    buttonPressed = true;
  lastInterrupt = interruptTime;
}

// --------------------------------------------------------------------------------
// Initialize the OLED with base font for fast refresh
// --------------------------------------------------------------------------------
void initDisplay() {
  u8x8.begin();
  u8x8.setPowerSave(0);
  u8x8.setFont(u8x8_font_amstrad_cpc_extended_r);
  u8x8.setCursor(0, 0);
}

// --------------------------------------------------------------------------------
// A function to write a message on the display
// "row" specifies which row to print on... 1, 2, 3, etc.
// "clearDisplay" specifies if everything should be wiped or not
// --------------------------------------------------------------------------------
void showMessage(const char * message, int row, bool cleardisplay) {
  if(cleardisplay){
    u8x8.clearDisplay();
  }
  u8x8.setCursor(0, row);
  u8x8.print(message);
}

// --------------------------------------------------------------------------------
// Setup: executed once at startup or reset
// --------------------------------------------------------------------------------
void setup() {
  Serial.begin(9600);
  hm10.begin(9600);
  pinMode(3, INPUT_PULLUP);
  pinMode(ppgPin, INPUT);
  attachInterrupt(digitalPinToInterrupt(buttonPin), buttonInterruptISR, FALLING);
  
  initIMU();
  
  Serial.println("==============================");
  Serial.println("BLE Handshake Code Started");
  Serial.println("==============================");

  initDisplay();
  u8x8.clearDisplay();
  showMessage("pedometer", 0, true);
} 

// --------------------------------------------------------------------------------
// Loop: main code; executed in an infinite loop
// --------------------------------------------------------------------------------
void loop() {

  // Check if Python is sending a message to the Arduino. Also checks for handshake.
  if (hm10.available()) {
    Serial.println("recieved char");
    // Read a message from the BLE module and send to the Serial Monitor
    if (readBLE()){
      Serial.println(in_text);
      showMessage(in_text, 1, false);
    }
  }

  // Read from the Serial Monitor and send to the BLE module
  if (Serial.available())
    writeBLE();

  // If we know we are connected, start sending data back
//  if (bleConnected) {
//    if (millis() - sendTimer > 1000) {
//      sendTimer = millis();
//      hm10.print("*;");
//    }
//  }

  static double lastTime = 0.0;
  static double currTime = 0.0;
  if(!asleep){
    currTime = millis();
    if(currTime - lastTime > 40){ // we want 25Hz
      lastTime = currTime;
      sendData();  // it would seem the IMU samples at 33hz
      
    }
  }
  
  toggleSleep();
}
