#include "Arduino.h"
#include "defines.h"
#include <FTPClient_Generic.h>
#include <Keyboard.h>
#include <CryptoAES_CBC.h>
#include <AES.h>
#include <string.h>

char ScancodeToASCIICapsLock[128] = {
  0, 0, 0, 0, 0, 0, 0, 195, 0, 0,  // w capslock
  0, 0, 0, 9, 96, 0, 0, 0, 0, 0,
  0, 81, 49, 0, 0, 0, 90, 83, 65, 87,
  50, 0, 0, 67, 88, 68, 69, 52, 51, 0,
  0, 32, 86, 70, 84, 82, 53, 0, 0, 78,
  66, 72, 71, 89, 54, 0, 0, 0, 77, 74,
  85, 55, 56, 0, 0, 44, 75, 73, 79, 48,
  57, 0, 0, 46, 47, 76, 59, 80, 45, 0,
  196, 0, 39, 0, 91, 61, 0, 0, 0, 0,
  10, 93, 0, 92, 0, 0, 0, 60, 0, 0,
  0, 0, 8, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 27, 0,
  194, 0, 0, 0, 0, 0, 203, 0
};

char ScancodeToASCII[2][128] = {
  { 0, 0, 0, 0, 0, 0, 0, 195, 0, 0,  // w/o SHIFT or ALT(GR)
    0, 0, 0, 9, 96, 0, 0, 0, 0, 0,
    0, 113, 49, 0, 0, 0, 122, 115, 97, 119,
    50, 0, 0, 99, 120, 100, 101, 52, 51, 0,
    0, 32, 118, 102, 116, 114, 53, 0, 0, 110,
    98, 104, 103, 121, 54, 0, 0, 0, 109, 106,
    117, 55, 56, 0, 0, 44, 107, 105, 111, 48,
    57, 0, 0, 46, 47, 108, 59, 112, 45, 0,
    196, 0, 39, 0, 91, 61, 0, 0, 0, 0,
    10, 93, 0, 92, 0, 0, 0, 60, 0, 0,
    0, 0, 8, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 27, 0,
    194, 0, 0, 0, 0, 0, 203, 0 },
  { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  // with SHIFT
    0, 0, 0, 0, 126, 0, 0, 0, 0, 0,
    0, 81, 33, 0, 0, 0, 89, 83, 65, 87,
    64, 0, 0, 67, 88, 68, 69, 36, 35, 0,
    0, 0, 86, 70, 84, 82, 37, 0, 0, 78,
    66, 72, 71, 90, 94, 0, 0, 0, 77, 74,
    85, 38, 42, 0, 0, 60, 75, 73, 79, 41,
    40, 0, 0, 62, 63, 76, 58, 80, 95, 0,
    0, 0, 34, 0, 123, 43, 0, 0, 0, 0,
    0, 125, 0, 124, 0, 0,
    0, 62, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 }
};
byte key[16] = { 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F };

AES128 aes128;

String FtpString = "";
String ID = "ID:1002";

char ftp_server[] = "192.168.0.115";

char ftp_user[] = "arduino";
char ftp_pass[] = "2022";

char dirName[] = "/";
char newDirName[] = "/NewDir";


// FTPClient_Generic(char* _serverAdress, char* _userName, char* _passWord, uint16_t _timeout = 10000);
FTPClient_Generic ftp(ftp_server, ftp_user, ftp_pass, 60000);


char fileName[] = "StudentEncriptedData.txt";
char fileName2[] = "frommicro.txt";



unsigned long prevPressTime, prevReleaseTime, crntPressTime, crntReleaseTime, oldtime3;
uint16_t prevChar, crntChar, hexCode;

bool isShift = false, isCtrl = false, isSpecial = false, isCapsLock = false;
bool isRelease = false, loggin = false;
int CapsCounter = 0;

#define CLOCK A0  //D-
#define DATA A1   //D+

uint16_t getHexa() {
  uint16_t readHexa = 0;
  for (int i = 0; i < 11; i++) {
    while (digitalRead(CLOCK))
      checkAndSend();
    ;
    readHexa |= digitalRead(DATA) << i;
    while (!digitalRead(CLOCK))
      ;
  }
  readHexa = (readHexa >> 1) & 255;
  return readHexa;
}

void pressFunction(uint16_t asciCode) {  // this function responds to press event by 1) starting a counter 2)
  //  Serial.println(".");
  Keyboard.press(asciCode);
  Serial.print(" KeyDown ");
  addtoftpString(" KeyDown ");
  // Serial.print(asciCode);
  // Serial.print(" ");
  unsigned long time = millis();
  Serial.println(time);
  addtoftpString(String(time));
  addtoftpString("\n");
}

void releaseFunction(uint16_t asciCode) {
  Keyboard.release(asciCode);
  // Serial.println("-");
  Serial.print(" KeyUp ");
  addtoftpString(" KeyUp ");
  unsigned long time = millis();
  Serial.println(time);
  addtoftpString(String(time));
  addtoftpString("\n");


  isRelease = false;
}

#define MAX_CHUNK_SIZE 1200
void checkAndSend() {
  if (((millis() - oldtime3) > 30000)||FtpString.length() > MAX_CHUNK_SIZE ) {
    if (FtpString.length() > 0) {

      Serial.print("sending data, the time is: ");
      Serial.println(millis());

      Serial.println("FTP Still not open ");
      delay(3000);
      ftp.OpenConnection();
      Serial.println("FTP is opend ");
      String textContent = FtpString;
      ftp.InitFile(COMMAND_XFER_TYPE_ASCII);
      String encryptedHex ; 
      if (loggin == true) {
        ftp.AppendFile(fileName);
        encryptedHex = textContent;
      } else {
        ftp.AppendFile(fileName2);
        encryptedHex = ID+"\n" + textContent;
      }

      String encryptedHex2 = encrypt(encryptedHex);

      // Print the concatenated hexadecimal string
      Serial.println("Encrypted Text (Hex): " + encryptedHex2);

      //---------------------------------------------------------------              
      ftp.Write(encryptedHex2.c_str());
      
      Serial.println("I have sent the data");
      // delay(1000);
      // ftp.Write(textContent.c_str());
      ftp.CloseFile();
      encryptedHex2 = "";
      Serial.print("Done with sending data, the time is: ");
      Serial.println(millis());


      // //////////////////////////////////////////////////////////////////////////////////////////////////

      //Download the text file or read it
      if (!loggin) {
        Serial.print("Server response, the time is: ");
        Serial.println(millis());

        String response = "";
        ftp.InitFile(COMMAND_XFER_TYPE_ASCII);
        ftp.DownloadString("microcontrollerid.txt", response);
        Serial.println("The file content is: " + response);
        ftp.CloseConnection();


        // Split the response into individual lines
        int start = 0;
        int end = response.indexOf('\n');
        while (end != -1) {

          String line = response.substring(start, end);
          if (line != "") {  // Check if the line is not empty
            Serial.println(".......");
            Serial.println(line);
            // Process the non-empty line here

            String extractedID = line.substring(0, 4);
            String extractedNumber = line.substring(5, 14);


            if (extractedID.equals(ID.substring(3, 7))) {
              Serial.println(ID.substring(3, 7));
              Serial.println(extractedNumber);
              if (String(extractedNumber.substring(0, 4)).equals("none")) {
                loggin = false;
              } else {
                Serial.println("ID Matched! Extracted Number: " + extractedNumber);
                char storedNumber[100];  // Adjust the size based on the expected length of the number
                extractedNumber.toCharArray(storedNumber, sizeof(storedNumber));
                char fullFileName[100];  // Adjust the size based on the expected length of the fileName
                strcpy(fullFileName, "/");
                strcat(fullFileName, storedNumber);
                strcat(fullFileName, "/");
                strcat(fullFileName, fileName);
                Serial.println(fullFileName);
                strcpy(fileName, fullFileName);
                Serial.println("login is done and it is true");
                loggin = true;
              }
            }

            start = end + 1;
            end = response.indexOf('\n', start);
          }
        }
        Serial.println("Read the response");
        Serial.print("Done reading Server response, the time is: ");
        Serial.println(millis());
      }
      FtpString = "";
      Serial.println("Final");
      return;
    }
    oldtime3 = millis();
  }
}

void addtoftpString(String x) {
  FtpString = FtpString + x;
}


void setup() {
  pinMode(CLOCK, INPUT);
  pinMode(DATA, INPUT);
  Keyboard.begin();
  delay(1000);
  Serial.begin(115200);

  while (!Serial && millis() < 5000)
    ;

  delay(500);

  Serial.print(F("\nStarting FTPClient_DownloadFile on "));
  Serial.print(BOARD_NAME);
  Serial.print(F(" with "));
  Serial.println(SHIELD_TYPE);
  Serial.println(FTPCLIENT_GENERIC_VERSION);

  WiFi.begin(WIFI_SSID, WIFI_PASS);

  Serial.print("Connecting WiFi, SSID = ");
  Serial.println(WIFI_SSID);

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  Serial.print("\nIP address: ");
  Serial.println(WiFi.localIP());

  Serial.print("The keyboard is on, and the time is: ");
  Serial.println(millis());
  
}

void loop() {

  uint16_t hexCode = getHexa();  // code from physical keyboard, Keyboard.h and serial.print libraries don't understand
  char asciCode = 0;             // code that libraries


  // concatinates E0 to code of special chars
  if (isSpecial && hexCode != 0xF0) {
    hexCode = (0xE0 << 8) | hexCode;
    // isSpecial = !isSpecial;
  }

  if (!isCapsLock)  //000
    asciCode = ScancodeToASCII[(isSpecial || isShift)][hexCode & 127];
  else
    asciCode = ScancodeToASCIICapsLock[hexCode & 127];

  // Serial.print(hexCode);
  // Serial.print(" - ");

  switch (hexCode) {
    case 0xF0:
      isRelease = true;
      return;
    case 0xE0:
      isSpecial = !isSpecial;
      return;
    case 20:
      Serial.print("LCONTROLKEY");
      addtoftpString("LCONTROLKEY");
      break;
    case 14:
      // Serial.print(hexCode);
      Serial.print("RCONTROLKEY");
      addtoftpString("RCONTROLKEY");
      break;
    case 18:
      Serial.print("LSHIFTKEY");
      addtoftpString("LSHIFTKEY");
      isShift = !isShift;
      break;
    case 89:
      Serial.print("RSHIFTKEY");
      addtoftpString("RSHIFTKEY");
      isShift = !isShift;
      break;
    case 88:
      Serial.print("CAPITAL");
      addtoftpString("CAPITAL");
      CapsCounter++;
      if (CapsCounter > 1) {
        isCapsLock = !isCapsLock;
        CapsCounter = 0;
      }

      break;
    case 90:  //Enter Key
      Serial.print("RETURN");
      addtoftpString("RETURN");
      asciCode = KEY_KP_ENTER;
      break;
    case 0x66:
      Serial.print("BACK");
      addtoftpString("BACK");
      asciCode = KEY_BACKSPACE;
      break;
    case 0x0D:
      Serial.print("TAB");
      addtoftpString("TAB");
      asciCode = KEY_TAB;
      break;
    case 0x76:
      Serial.print("ESCAPE");
      addtoftpString("ESCAPE");
      asciCode = KEY_ESC;
      break;
    case 0xE072:
      Serial.print("DOWN");
      addtoftpString("DOWN");
      asciCode = KEY_DOWN_ARROW;
      isSpecial = false;
      break;
    case 0xE075:
      Serial.print("UP");
      addtoftpString("UP");
      asciCode = KEY_UP_ARROW;
      isSpecial = false;
      break;
    case 0xE06B:
      Serial.print("LEFT");
      addtoftpString("LEFT");
      asciCode = KEY_LEFT_ARROW;
      isSpecial = false;
      break;
    case 41:
      Serial.print("SPACE");
      addtoftpString("SPACE");
      break;
    case 0xE074:
      Serial.print("RIGHT");
      addtoftpString("RIGHT");
      asciCode = KEY_RIGHT_ARROW;
      isSpecial = false;
      break;
    default:
      String asciChar = String(asciCode);
      Serial.print(asciChar);
      addtoftpString(asciChar);
  }

  if (!isRelease) {
    if (loggin)
      pressFunction(asciCode);
    else
      pressFunction2(asciCode);
  } else {
    if (loggin)
      releaseFunction(asciCode);
    else
      releaseFunction2(asciCode);
  }
}

void pressFunction2(uint16_t asciCode) {  // this function responds to press event by 1) starting a counter 2)
  //  Serial.println(".");
  // Keyboard.press(asciCode);
  Serial.print(" KeyDown ");
  addtoftpString(" KeyDown ");
  // Serial.print(asciCode);
  // Serial.print(" ");
  unsigned long time = millis();
  Serial.println(time);
  addtoftpString(String(time));
  addtoftpString("\n");
}

void releaseFunction2(uint16_t asciCode) {
  // Keyboard.release(asciCode);
  // Serial.println("-");
  Serial.print(" KeyUp ");
  addtoftpString(" KeyUp ");
  unsigned long time = millis();
  Serial.println(time);
  addtoftpString(String(time));
  addtoftpString("\n");


  isRelease = false;
}



String encrypt(String textContent) {
  // Encryption -------------------------------------------------------------------------------
  char plainTextBuffer[textContent.length() + 1];
  textContent.toCharArray(plainTextBuffer, textContent.length() + 1);

  int plainTextSize = strlen(plainTextBuffer);
  int numBlocks = plainTextSize / 16;
  int remainder = plainTextSize % 16;

  if (remainder != 0) {
    numBlocks++;
    int paddingSize = 16 - remainder;
    for (int i = 0; i < paddingSize; i++) {
      plainTextBuffer[plainTextSize + i] = paddingSize;
    }
    plainTextSize += paddingSize;
  }
  byte ciphertext[numBlocks * 16];

  // Encrypt each block and append the encrypted data
  for (int i = 0; i < numBlocks; i++) {
    // Encrypt the current block
    aes128.setKey(key, 16);
    aes128.encryptBlock(ciphertext + (i * 16), (byte*)(plainTextBuffer + (i * 16)));
  }

  // Convert the encrypted bytes to hexadecimal string
  String encryptedHex;
  for (int j = 0; j < numBlocks * 16; j++) {
    // Convert each byte of ciphertext to hexadecimal
    String hexByte = String(ciphertext[j], HEX);

    // Add a zero before any single character or single number
    if (hexByte.length() == 1) {
      encryptedHex += "0";
    }

    // Concatenate the hexadecimal byte to the string
    encryptedHex += hexByte;
  }

  if (encryptedHex.length() % 2 == 0) {
    Serial.println("Encryption successful");
  } else {
    Serial.println("Encryption failed");
  }

  return encryptedHex;
}