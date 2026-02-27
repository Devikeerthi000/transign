import os
import cv2
import torch
import tempfile
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
from transformers import pipeline
import json
import string

# Load signs
with open("signs/signs.json", "r") as f:
    video_map = json.load(f)

# Whisper setup
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = pipeline("automatic-speech-recognition", model="openai/whisper-base", device=0 if device == "cuda" else -1)
print(f"\n📦 Whisper model loaded on {device}")

# Record voice
def record_audio(duration=5, samplerate=16000):
    print(f"\n🎙️ Listening for {duration} seconds...")
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='int16')
    sd.wait()
    return samplerate, audio

# Transcribe voice to text
def transcribe_audio(duration=5):
    samplerate, audio = record_audio(duration)
    tmpfile = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmpfile.close()
    wav.write(tmpfile.name, samplerate, audio)

    try:
        result = pipe(tmpfile.name)
        return result["text"]
    except Exception as e:
        print("❌ Transcription error:", str(e))
        return None
    finally:
        os.remove(tmpfile.name)

# Play webcam and sign video side by side
def play_dual_feed(text):
    clean_text = text.lower().translate(str.maketrans('', '', string.punctuation))
    words = clean_text.split()

    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("❌ Webcam failed.")
        return

    for word in words:
        video_path = video_map.get(word)
        if not video_path or not os.path.exists(video_path):
            print(f"❌ No sign video for: {word}")
            continue

        print(f"🎬 Showing: {word} → {video_path}")
        sign = cv2.VideoCapture(video_path)

        if not sign.isOpened():
            print(f"⚠️ Could not open: {video_path}")
            continue

        while True:
            ret_cam, cam_frame = cam.read()
            ret_sign, sign_frame = sign.read()

            if not ret_sign:
                break
            if not ret_cam:
                print("⚠️ Webcam lost.")
                break

            cam_frame = cv2.resize(cam_frame, (480, 360))
            sign_frame = cv2.resize(sign_frame, (480, 360))

            # Show word as subtitle
            cv2.rectangle(cam_frame, (0, 300), (480, 360), (0, 0, 0), -1)
            cv2.putText(cam_frame, word, (10, 340), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            combined = cv2.hconcat([cam_frame, sign_frame])
            cv2.imshow("🧑 You (Live) | 🤟 Sign Video", combined)

            if cv2.waitKey(30) & 0xFF == ord('q'):
                cam.release()
                sign.release()
                cv2.destroyAllWindows()
                return

        sign.release()

    cam.release()
    cv2.destroyAllWindows()

# Main driver
if __name__ == "__main__":
    while True:
        text = transcribe_audio(duration=5)
        if text:
            print(f"\n📝 You said: {text}")
            play_dual_feed(text)

        again = input("\n🔁 Speak again? (y/n): ").strip().lower()
        if again != 'y':
            break
