# Anti-cheating Keylogger: -

Our project is a  micro-controller based keylogger that runs on arduino R4 WiFi to prevent cheating in computer-based exams. It records keystrokes and sends them to an FTP server that analyzes them for cheating keywords and for analyzing the typing pattern using ML, it also has a GUI to communicate the results to the instructor
The following are the topics covered in this readme.txt file:

[1] Keylogger setup

[2] Server-side processing

[3] Usage Instructions

[4] Graphical User Interface (GUI)

[5] File Structure

[6] Requirements

[7] Notes



# [1] Keylogger Setup

To run the keylogger on the student's computer, follow these steps:

1. Connect your Arduino Uno R4 WiFi to your computer.
2. Upload the provided code to your Arduino board using the Arduino IDE.
3. Ensure the required libraries are installed in your Arduino development environment.
4. Set up your WiFi credentials and other configuration parameters in the code.
5. Connect the necessary hardware components (keyboard, clock, data lines, etc.).
6. Power up your Arduino and monitor the serial output for debugging and status messages.

# [2] Server-side Processing

The server-side processing involves several Python scripts to handle student input, decryption, cheating detection, and machine learning-based authentication. The available scripts are:

1. `sign-in.py`: Facilitates student login to the system.
2. `decryption.py`: Decrypts the encrypted student input.
3. `assemble.py`: Assembles scattered keystrokes into plaintext.
4. `CheatingDetectionAndConnection.py`: Detects cheating attempts and monitors student connection and typing accuracy.
5. `train.py`: Manages the training and testing of machine learning models for typing pattern authentication.
6. `3model.py`: Implements machine learning for typing pattern authentication.

#[3] Usage Instructions 

1. Run the provided Python scripts on the server, focusing on `sign-in.py`, `decryption.py`, `CheatingDetectionAndConnection.py`, and `train.py`.
2. Ensure the correct file structure is maintained on the server for proper functioning.
3. Customize the scripts and configurations as needed for your specific setup and requirements.
4. Monitor the server output for debugging information and system status updates.

# [4] Graphical User Interface (GUI)

The GUI component, implemented using the Python Tkinter library, provides an intuitive interface for instructors to monitor student activities during exams. Simply install the Tkinter library and upload the provided code to the instructor interface. Input the FTP server details to begin monitoring student activities effectively.

# [5] File Structure
- The instructor_GUI.py should be run from instructor's device.
- The Server_files should be put on a running FTP server.
- The Arduino_IDE_Project should be uploaded into the uController using Arduino IDE.

![File structure](https://github.com/Abdullah-Mahran1/anti-cheating-keylogger-sdp/assets/62305013/45607428-9fb2-4bc0-bdf7-d5f98fcedf27)

The above image illustrates the required file structure on the server for the system to operate seamlessly.
The server code is expecting to have a folder called "permenant" that contains training_data.h5, and 3model.py. It also assumes that if a student signed-in with student_id (=2032145 for example) as a pre-registered stuednt, then it's expected that we find a folder with the name of '2032145' containing model.keras, studentKeyStrocks.txt, and other files.

# [6] Requirements

Ensure you have the necessary hardware components and software libraries installed and configured correctly for the system to function as intended.

# [7] Notes

- Customize the system configuration and scripts according to your specific requirements and environment.
- Monitor system output and logs for any errors or warnings during operation.
- For further assistance, refer to the documentation or contact the project team for support.

Enjoy using the exam monitoring system for efficient and secure exam administration!
