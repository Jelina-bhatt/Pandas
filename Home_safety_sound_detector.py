import sounddevice as sd
import numpy as np
import datetime
import os

DURATION = 3  # seconds to record
FS = 44100    # sampling rate
LOG_FILE = "alerts_log.txt"

def beep():
    try:
        # Windows beep
        import winsound
        winsound.Beep(1000, 500)  # frequency 1000Hz, duration 500ms
    except ImportError:
        # Mac/Linux fallback
        print('\a')

print("Home Safety Sound Detector Running. Press Ctrl+C to stop.")

while True:
    try:
        print("Recording audio...")
        audio = sd.rec(int(DURATION * FS), samplerate=FS, channels=1)
        sd.wait()
        audio = audio.flatten()

        # Simple alert: if max volume > threshold
        if np.max(np.abs(audio)) > 0.3:  # adjust threshold
            print("⚠️ Suspicious sound detected!", datetime.datetime.now())
            beep()
            with open(LOG_FILE, "a") as f:
                f.write(f"{datetime.datetime.now()}: Suspicious sound detected\n")
        else:
            print("Sound normal")

    except KeyboardInterrupt:
        print("Stopping detector...")
        break
