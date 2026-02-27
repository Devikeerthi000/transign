import json
import os

# Load the WLASL dataset
with open("WLASL_v0.3.json", "r") as f:
    wlasl_data = json.load(f)

video_map = {}

# Base folder where videos are stored
video_folder = "signs/videos"

for entry in wlasl_data:
    gloss = entry["gloss"].lower()
    if not entry["instances"]:
        continue
    instance = entry["instances"][0]
    video_id = instance["video_id"] + ".mp4"
    video_path = os.path.join(video_folder, video_id)
    
    if os.path.exists(video_path):  # Ensure the video file is actually downloaded
        video_map[gloss] = video_path

# Save the mapping to a JSON file
with open("signs/signs.json", "w") as out:
    json.dump(video_map, out, indent=4)

print(f"✅ Generated mapping for {len(video_map)} words.")
