import os
import queue
import threading
import time
import sounddevice as sd
import vosk
import json
import cv2
import numpy as np

# === CONFIGURATION ===
model_path = r"D:\Transign\models\vosk-model-small-en-us-0.15"  # ✅ Adjust if needed
samplerate = 16000

# === Load sign mapping ===
with open("signs/signs.json", "r") as f:
    video_map = json.load(f)

# === Setup Vosk ===
if not os.path.exists(model_path):
    raise FileNotFoundError("❌ Vosk model not found.")
model = vosk.Model(model_path)
q = queue.Queue()

# === Audio callback ===
def callback(indata, frames, time_info, status):
    if status:
        print(status)
    q.put(bytes(indata))

# === Global state ===
current_word = ""
sign_video_path = None
lock = threading.Lock()

# === Sign display thread (Webcam + Sign Video) ===
def sign_display_loop():
    global current_word, sign_video_path
    cap_cam = cv2.VideoCapture(0)

    cap_sign = None
    last_sign_path = None

    while True:
        ret_cam, frame_cam = cap_cam.read()
        if not ret_cam:
            break

        frame_cam = cv2.resize(frame_cam, (480, 360))

        # Load new sign video if path changed
        with lock:
            if sign_video_path != last_sign_path:
                if cap_sign:
                    cap_sign.release()
                cap_sign = None
                last_sign_path = sign_video_path
                if sign_video_path and os.path.exists(sign_video_path):
                    cap_sign = cv2.VideoCapture(sign_video_path)

        # Read sign video frame (or show blank if not playing)
        frame_sign = 255 * np.ones((360, 480, 3), dtype=np.uint8)
        if cap_sign:
            ret_sign, temp_frame = cap_sign.read()
            if ret_sign:
                frame_sign = cv2.resize(temp_frame, (480, 360))
            else:
                cap_sign.release()
                cap_sign = None
                with lock:
                    sign_video_path = None

        # Subtitle below webcam
        cv2.rectangle(frame_cam, (0, 300), (480, 360), (0, 0, 0), -1)
        with lock:
            cv2.putText(frame_cam, current_word, (10, 340),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        combined = cv2.hconcat([frame_cam, frame_sign])
        cv2.imshow("🧑 You (Live) | 🤟 Sign", combined)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap_cam.release()
    if cap_sign:
        cap_sign.release()
    cv2.destroyAllWindows()
    os._exit(0)

# === Start Webcam Thread ===
def start_video_thread():
    t = threading.Thread(target=sign_display_loop)
    t.daemon = True
    t.start()

# === Main Recognition Loop ===
def main():
    global current_word, sign_video_path
    print("🎤 Speak continuously. Say 'q' to quit.\n")

    start_video_thread()

    with sd.RawInputStream(samplerate=samplerate, blocksize=8000, dtype='int16',
                           channels=1, callback=callback):
        rec = vosk.KaldiRecognizer(model, samplerate)
        word_history = set()
        buffer_text = ""

        while True:
            data = q.get()

            if rec.AcceptWaveform(data):
                result = json.loads(rec.Result())
                text = result.get("text", "")
                buffer_text += " " + text

                words = buffer_text.strip().split()
                buffer_text = ""  # Clear after using

                for word in words:
                    if word == 'q':
                        print("👋 Quitting.")
                        return
                    if word in word_history:
                        continue
                    word_history.add(word)

                    path = video_map.get(word)
                    with lock:
                        current_word = word
                        sign_video_path = path if path and os.path.exists(path) else None

                    if sign_video_path:
                        print(f"🤟 Showing: {word} → {sign_video_path}")
                    else:
                        print(f"❌ No sign found for: {word}")
            else:
                # Optional: could print partial result here
                pass

if __name__ == "__main__":
    main()
