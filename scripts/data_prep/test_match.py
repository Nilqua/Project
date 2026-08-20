import os
import glob

base_dir = "/home/naslia/Study/Project1/Code/Dataset/15. Thai H2H"
audio_dir = os.path.join(base_dir, "audiofiles")

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
    
    # Match by closest size
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
            pairs.append((l, best_r, l_size, os.path.getsize(os.path.join(audio_dir, best_r))))

print(f"Total matched pairs: {len(pairs)}")
for i, (l, r, ls, rs) in enumerate(pairs[:5]):
    print(f"Pair {i+1}:")
    print(f"  Left : {l} (Size: {ls})")
    print(f"  Right: {r} (Size: {rs})")
    print(f"  Diff : {abs(ls-rs)} bytes")
