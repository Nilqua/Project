import os
import subprocess
from concurrent.futures import ThreadPoolExecutor

base_dir = "/home/naslia/Study/Project1/Code/Dataset/15. Thai H2H"
audio_dir = os.path.join(base_dir, "audiofiles")
mixed_dir = os.path.join(base_dir, "mixed_audio")

os.makedirs(mixed_dir, exist_ok=True)

# Find all left files
left_files = [f for f in os.listdir(audio_dir) if f.endswith('_left.wav')]

def mix_pair(left_file):
    idx = left_file.split('_')[0]
    right_file = f"{idx}_right.wav"
    out_file = f"{idx}_mixed.wav"
    
    left_path = os.path.join(audio_dir, left_file)
    right_path = os.path.join(audio_dir, right_file)
    out_path = os.path.join(mixed_dir, out_file)
    
    if not os.path.exists(right_path):
        return
        
    if os.path.exists(out_path):
        return
        
    cmd = [
        "ffmpeg", "-y", "-i", left_path, "-i", right_path,
        "-filter_complex", "amix=inputs=2:duration=longest",
        "-ac", "1", # Ensure mono output
        "-ar", "16000", # Set 16kHz for speech models
        out_path
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

print(f"Mixing {len(left_files)} pairs of audio...")
with ThreadPoolExecutor(max_workers=8) as executor:
    executor.map(mix_pair, left_files)
    
print(f"Finished mixing audio to {mixed_dir}")
