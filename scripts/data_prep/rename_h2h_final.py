import os
import glob
import csv

base_dir = "/home/naslia/Study/Project1/Code/Dataset/15. Thai H2H"
audio_dir = os.path.join(base_dir, "audiofiles")
script_dir = os.path.join(base_dir, "scriptfiles")

files = os.listdir(audio_dir)
left_files = [f for f in files if f.endswith('_left.wav')]
right_files = [f for f in files if f.endswith('_right.wav')]

calls = {}
for f in left_files:
    call_id = f[:24]
    calls.setdefault(call_id, {'left': [], 'right': []})
    calls[call_id]['left'].append(f)

for f in right_files:
    call_id = f[:24]
    calls.setdefault(call_id, {'left': [], 'right': []})
    calls[call_id]['right'].append(f)

pairs = []
for call_id, data in calls.items():
    lefts = sorted(data['left'], key=lambda x: os.path.getsize(os.path.join(audio_dir, x)))
    rights = sorted(data['right'], key=lambda x: os.path.getsize(os.path.join(audio_dir, x)))
    
    for l in lefts:
        l_size = os.path.getsize(os.path.join(audio_dir, l))
        best_r = None
        min_diff = float('inf')
        for r in rights:
            r_size = os.path.getsize(os.path.join(audio_dir, r))
            if abs(l_size - r_size) < min_diff:
                min_diff = abs(l_size - r_size)
                best_r = r
        if best_r:
            rights.remove(best_r)
            pairs.append((l, best_r))

with open(os.path.join(base_dir, 'mapping_rename.csv'), 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['New_ID', 'Old_Left_Audio', 'Old_Right_Audio', 'Old_Left_Script', 'Old_Right_Script'])
    
    for idx, (l_wav, r_wav) in enumerate(pairs, 1):
        l_txt = l_wav.replace('.wav', '.txt')
        r_txt = r_wav.replace('.wav', '.txt')
        
        # Rename audio
        os.rename(os.path.join(audio_dir, l_wav), os.path.join(audio_dir, f"{idx}_left.wav"))
        os.rename(os.path.join(audio_dir, r_wav), os.path.join(audio_dir, f"{idx}_right.wav"))
        
        # Rename script if exists
        l_txt_path = os.path.join(script_dir, l_txt)
        r_txt_path = os.path.join(script_dir, r_txt)
        
        has_l_txt = os.path.exists(l_txt_path)
        has_r_txt = os.path.exists(r_txt_path)
        
        if has_l_txt:
            os.rename(l_txt_path, os.path.join(script_dir, f"{idx}_left.txt"))
        if has_r_txt:
            os.rename(r_txt_path, os.path.join(script_dir, f"{idx}_right.txt"))
            
        writer.writerow([idx, l_wav, r_wav, l_txt if has_l_txt else '', r_txt if has_r_txt else ''])

print(f"Successfully renamed {len(pairs)} pairs of audio and script files!")
