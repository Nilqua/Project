import json
import os
from pathlib import Path
from collections import Counter


def organize_dataset():
    # Project paths
    base_dir = Path(__file__).resolve().parent
    dataset_dir = base_dir / "Dataset"
    thaiser_dir = dataset_dir / "ThaiSER"
    cleaned_dir = dataset_dir / "ThaiSER_cleaned"
    label_path = thaiser_dir / "emotion_label.json"

    if not label_path.exists():
        print(f"Error: {label_path} not found.")
        return

    with open(label_path, "r", encoding="utf-8") as f:
        labels = json.load(f)

    # Core emotion categories
    emotions = ["Neutral", "Angry", "Happy", "Sad", "Frustrated"]
    sessions = ["script", "imp"]

    # Create directory structure
    for session in sessions:
        for emo in emotions:
            (cleaned_dir / session / emo).mkdir(parents=True, exist_ok=True)

    flac_files = list(thaiser_dir.glob("**/*.flac"))
    print(f"Processing {len(flac_files)} flac files...")

    counts = Counter()

    for src_path in flac_files:
        fname = src_path.name

        # Determine session
        if "_script" in fname:
            session = "script"
        elif "_impro" in fname:
            session = "imp"
        else:
            continue

        # Mic normalization for label lookup
        lookup_name = fname.replace("_clip_", "_con_").replace("_middle_", "_con_")
        meta_list = labels.get(lookup_name, [])

        if not meta_list:
            continue

        meta = meta_list[0]
        # Use assigned_emo for clean 5-class categorization
        emo = meta.get("assigned_emo", "Neutral")

        if emo not in emotions:
            emo = "Neutral"

        dest_dir = cleaned_dir / session / emo
        dest_link = dest_dir / fname

        # Calculate relative symlink target
        rel_target = os.path.relpath(src_path, start=dest_dir)

        if dest_link.is_symlink() or dest_link.exists():
            dest_link.unlink()

        dest_link.symlink_to(rel_target)
        counts[(session, emo)] += 1

    print("\nThaiSER Dataset cleaned successfully!")
    print(f"Cleaned dataset root: {cleaned_dir}")
    print("\nSummary per folder:")
    for session in sessions:
        print(f"\n--- [{session.upper()}] ---")
        for emo in emotions:
            c = counts[(session, emo)]
            print(f"  {cleaned_dir / session / emo}: {c:,} files")


if __name__ == "__main__":
    organize_dataset()
