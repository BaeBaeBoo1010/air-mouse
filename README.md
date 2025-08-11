# AI Mouse Control with Hand Gestures

This project allows you to control your mouse cursor and perform clicks and scrolling using hand gestures detected by your webcam. It leverages **MediaPipe Hands** for hand tracking, OpenCV for video capture, and other Python libraries for mouse control and smoothing.

---

## Features

- Move the mouse cursor with right hand index finger.
- Left click by pinching right hand thumb and index finger.
- Right click by pinching right hand thumb and middle finger.
- Scroll by pinching right hand thumb and ring finger.
- Smooth cursor movement using One-Euro Filter.
- Supports selecting from multiple connected cameras.

---

## Installation (Step-by-step)

1. **Install Python 3.7+**

   - Download from [python.org](https://www.python.org/downloads/windows/).
   - Run installer and check "Add Python to PATH" before installing.
   - Verify installation in Command Prompt:
     ```
     python --version
     ```

2. **Download the project**

   - Clone with Git:
     ```
     git clone https://github.com/BaeBaeBoo1010/air-mouse.git
     cd air-mouse
     ```
   - Or download ZIP from GitHub and extract.

3. **(Optional) Create and activate virtual environment**
  Creating a virtual environment is recommended to isolate dependencies:
  
  - Create the virtual environment (named `venv`):
     ```
     python -m venv venv
     ```
Activate it:
  - On Command Prompt (cmd.exe):
    ```
    venv\Scripts\activate
    ```
  - On PowerShell:
    ```
    .\venv\Scripts\Activate.ps1
    ```

Your terminal prompt will change to indicate the environment is active, e.g., `(venv)`.
4. **Install required Python packages**
  - With the virtual environment activated (or not), install the dependencies using `requirements.txt`:
    ```
    pip install --upgrade pip
    pip install -r requirements.txt
    ```
    This will install all required libraries: `opencv-python`, `mediapipe`, `pyautogui`, `pygrabber`, `numpy`, `pynput`, etc.
5. **Run the program**
    ```
    python main.py
    ```
  - When running, select the camera from the list by entering its number or press Enter for default.
  - Press `q` in the video window to quit the program.

## Author

BaeBaeBoo1010
