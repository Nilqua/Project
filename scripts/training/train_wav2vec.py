import os
import glob
import re
import random
import datetime
import json
import argparse
import numpy as np
import librosa
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# --- CONFIGURATION ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_DATA_DIR = os.path.join(PROJECT_ROOT, "Dataset", "ThaiSER_cleaned")
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "experiments")
MODELS_PATH = os.path.join(PROJECT_ROOT, "Models")
DEFAULT_MODEL_NAME = "airesearch/wav2vec2-large-xlsr-53-th"  # ค่าเริ่มต้นโมเดลภาษาไทย

BATCH_SIZE = 4  # ลด Batch ลงเพื่อไม่ให้ RAM การ์ดจอเต็ม เพราะใช้โมเดลใหญ่และเสียงยาวขึ้น
EPOCHS = 5
LEARNING_RATE = 1e-5
SAMPLE_RATE = 16000
MAX_LEN = 16000 * 8  # เพิ่มเป็น 8 วินาที เพื่อเก็บเสียงสนทนาให้ครบถ้วน

CLASSES = ["Angry", "Frustrated", "Happy", "Neutral", "Sad"]
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASSES)}
IDX_TO_CLASS = {idx: name for idx, name in enumerate(CLASSES)}


def get_actor_id(filename):
    # ดึงเลข actor ออกจากชื่อไฟล์ เช่น actor001_xxx.flac -> "001"
    match = re.search(r"actor(\d+)", filename)
    if match:
        return match.group(1)
    return "unknown"


def load_raw_audio(filepath):
    # โหลดคลื่นเสียงดิบ (Raw Waveform) ด้วย librosa ที่ 16kHz
    wav, sr = librosa.load(filepath, sr=SAMPLE_RATE)
    # ไม่ต้อง pad ด้วย numpy ตรงนี้ ปล่อยให้ FeatureExtractor จัดการพร้อมสร้าง attention_mask
    return wav


def normalize_subset(subset):
    if subset in ["sentence", "script"]:
        return "sentence"
    elif subset in ["improvisation", "imp"]:
        return "improvisation"
    elif subset == "both":
        return "both"
    return subset


def prepare_dataset(subset="both", data_dir=None):
    if data_dir is None:
        data_dir = DEFAULT_DATA_DIR

    if subset in ["sentence", "script"]:
        sessions = ["script"]
    elif subset in ["improvisation", "imp"]:
        sessions = ["imp"]
    elif subset == "both":
        sessions = ["script", "imp"]
    else:
        raise ValueError(f"Unknown subset: '{subset}'. Must be one of ['sentence', 'improvisation', 'both', 'script', 'imp'].")

    # เก็บไฟล์เสียงแยกตาม actor ก่อน เพื่อไม่ให้เสียงคนเดียวกันหลุดไปทั้ง train และ val
    actor_data = {}

    for session in sessions:
        for cname in CLASSES:
            folder = os.path.join(data_dir, session, cname)
            if not os.path.exists(folder):
                continue

            label = CLASS_TO_IDX[cname]
            files = glob.glob(os.path.join(folder, "*.flac"))

            for fpath in files:
                fname = os.path.basename(fpath)
                actor = get_actor_id(fname)
                actor_data.setdefault(actor, []).append((fpath, label))

    actors = list(actor_data.keys())
    if not actors:
        raise ValueError(f"No audio files found in {data_dir} for sessions: {sessions}")

    random.seed(42)
    random.shuffle(actors)

    val_size = max(1, int(len(actors) * 0.2))
    val_actors = set(actors[:val_size])
    train_actors = set(actors[val_size:])

    train_files = []
    for a in train_actors:
        train_files += actor_data[a]

    val_files = []
    for a in val_actors:
        val_files += actor_data[a]

    return train_files, val_files


class ThaiSERWav2VecDataset(Dataset):
    # PyDataset สำหรับป้อนคลื่นเสียงดิบให้ FeatureExtractor ของ Wav2Vec2
    def __init__(self, file_list, feature_extractor):
        self.file_list = file_list
        self.feature_extractor = feature_extractor

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        filepath, label = self.file_list[idx]
        wav = load_raw_audio(filepath)

        # สกัดฟีเจอร์สำหรับ Wav2Vec2 และสร้าง attention_mask
        inputs = self.feature_extractor(
            wav,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            padding="max_length",
            max_length=MAX_LEN,
            truncation=True,
            return_attention_mask=True,
            do_normalize=True
        )

        input_values = inputs.input_values.squeeze(0)
        attention_mask = inputs.attention_mask.squeeze(0)
        return input_values, attention_mask, torch.tensor(label, dtype=torch.long)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Wav2Vec2 on ThaiSER dataset")
    parser.add_argument(
        "--subset",
        choices=["sentence", "improvisation", "both", "script", "imp"],
        default="both",
        help="Dataset subset to train on (default: both)"
    )
    parser.add_argument(
        "--data_dir",
        default=DEFAULT_DATA_DIR,
        help=f"Dataset directory path (default: {DEFAULT_DATA_DIR})"
    )
    parser.add_argument(
        "--model_name",
        default=DEFAULT_MODEL_NAME,
        help=f"HuggingFace model name or path (default: {DEFAULT_MODEL_NAME})"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs (default: 5)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size (default: 4)"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Learning rate (default: 1e-5)"
    )
    parser.add_argument(
        "--output_dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to save experiment results (default: {DEFAULT_OUTPUT_DIR})"
    )
    parser.add_argument(
        "--save_model",
        action="store_true",
        default=True,
        help="Save best model weights (default: True)"
    )
    parser.add_argument(
        "--no_save_model",
        dest="save_model",
        action="store_false",
        help="Do not save model weights"
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Run only 1 train batch and 1 val batch to verify pipeline"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    norm_subset = normalize_subset(args.subset)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 65)
    print("Wav2Vec2 Speech Emotion Recognition")
    print(f"Model Name   : {args.model_name}")
    print(f"Subset       : {norm_subset} (raw: {args.subset})")
    print(f"Data Dir     : {args.data_dir}")
    print(f"Output Dir   : {args.output_dir}")
    print(f"Device       : {device} (PyTorch {torch.__version__})")
    print(f"Epochs       : {args.epochs} | Batch Size: {args.batch_size} | LR: {args.learning_rate}")
    print(f"Save Model   : {args.save_model} | Dry Run: {args.dry_run}")
    print("=" * 65)

    # 1. เตรียมข้อมูล
    train_files, val_files = prepare_dataset(subset=args.subset, data_dir=args.data_dir)
    print(f"Train samples: {len(train_files)}")
    print(f"Val samples  : {len(val_files)}")

    feature_extractor = AutoFeatureExtractor.from_pretrained(args.model_name)

    train_dataset = ThaiSERWav2VecDataset(train_files, feature_extractor)
    val_dataset = ThaiSERWav2VecDataset(val_files, feature_extractor)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # 2. โหลด Pre-trained Wav2Vec2 Model
    model = AutoModelForAudioClassification.from_pretrained(
        args.model_name,
        num_labels=len(CLASSES),
        label2id=CLASS_TO_IDX,
        id2label=IDX_TO_CLASS,
    )
    # Freeze the feature encoder to prevent loss=nan and gradient explosion
    try:
        model.freeze_feature_encoder()
    except AttributeError:
        model.freeze_feature_extractor()
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    os.makedirs(MODELS_PATH, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    best_model_path = os.path.join(MODELS_PATH, f"best_wav2vec_{norm_subset}_{timestamp}.pt")

    best_val_loss = float("inf")
    best_metrics = None

    num_epochs = 1 if args.dry_run else args.epochs

    # 3. เทรนโมเดล
    print("\nStarting Wav2Vec2 training...")
    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs} [Train]")
        for input_values, attention_mask, labels in pbar:
            input_values = input_values.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(input_values=input_values, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * input_values.size(0)
            preds = torch.argmax(logits, dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            if args.dry_run:
                print("  [Dry Run] Finished 1 train batch.")
                break

        train_loss = train_loss / max(1, total)
        train_acc = (correct / max(1, total)) * 100.0

        # Evaluation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_preds = []
        val_labels = []

        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch}/{num_epochs} [Val]  ")
            for input_values, attention_mask, labels in val_pbar:
                input_values = input_values.to(device)
                attention_mask = attention_mask.to(device)
                labels = labels.to(device)

                outputs = model(input_values=input_values, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                logits = outputs.logits

                val_loss += loss.item() * input_values.size(0)
                preds = torch.argmax(logits, dim=-1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

                val_preds.extend(preds.cpu().numpy().tolist())
                val_labels.extend(labels.cpu().numpy().tolist())

                if args.dry_run:
                    print("  [Dry Run] Finished 1 val batch.")
                    break

        val_loss = val_loss / max(1, val_total)
        val_acc = (val_correct / max(1, val_total)) * 100.0

        # Calculate metrics using sklearn
        acc = float(accuracy_score(val_labels, val_preds)) if val_labels else 0.0
        macro_prec = float(precision_score(val_labels, val_preds, average="macro", zero_division=0)) if val_labels else 0.0
        macro_rec = float(recall_score(val_labels, val_preds, average="macro", zero_division=0)) if val_labels else 0.0
        macro_f1 = float(f1_score(val_labels, val_preds, average="macro", zero_division=0)) if val_labels else 0.0

        prec_per_class = precision_score(val_labels, val_preds, average=None, labels=list(range(len(CLASSES))), zero_division=0) if val_labels else [0.0]*len(CLASSES)
        rec_per_class = recall_score(val_labels, val_preds, average=None, labels=list(range(len(CLASSES))), zero_division=0) if val_labels else [0.0]*len(CLASSES)
        f1_per_class = f1_score(val_labels, val_preds, average=None, labels=list(range(len(CLASSES))), zero_division=0) if val_labels else [0.0]*len(CLASSES)

        per_class_metrics = {}
        for idx, cname in enumerate(CLASSES):
            per_class_metrics[cname] = {
                "precision": round(float(prec_per_class[idx]), 4),
                "recall": round(float(rec_per_class[idx]), 4),
                "f1_score": round(float(f1_per_class[idx]), 4)
            }

        print(f"\nEpoch {epoch}/{num_epochs} Summary:")
        print(f"  Train Loss : {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss   : {val_loss:.4f} | Val Acc  : {val_acc:.2f}%")
        print(f"  Accuracy   : {acc:.4f} ({acc * 100:.2f}%)")
        print(f"  Macro Avg  : Precision={macro_prec:.4f}, Recall={macro_rec:.4f}, F1-Score={macro_f1:.4f}")
        print("  Per-Class Metrics:")
        for cname, m in per_class_metrics.items():
            print(f"    {cname:12s}: Precision={m['precision']:.4f}, Recall={m['recall']:.4f}, F1={m['f1_score']:.4f}")

        # Update best metrics
        if val_loss < best_val_loss or best_metrics is None:
            if val_loss < best_val_loss:
                print(f"  Val loss improved from {best_val_loss:.4f} to {val_loss:.4f}")
            best_val_loss = val_loss
            best_metrics = {
                "model": "wav2vec2",
                "subset": norm_subset,
                "model_name": args.model_name,
                "best_epoch": epoch,
                "train_loss": round(train_loss, 4),
                "train_acc": round(train_acc / 100.0, 4),
                "val_loss": round(val_loss, 4),
                "accuracy": round(acc, 4),
                "precision": round(macro_prec, 4),
                "recall": round(macro_rec, 4),
                "f1_score": round(macro_f1, 4),
                "per_class": per_class_metrics
            }
            if args.save_model and not args.dry_run:
                print(f"  Saving model to {best_model_path}")
                torch.save(model.state_dict(), best_model_path)
                best_metrics["model_path"] = best_model_path

        print()

    # Save summary JSON
    os.makedirs(args.output_dir, exist_ok=True)
    json_path = os.path.join(args.output_dir, f"wav2vec_{norm_subset}_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(best_metrics, f, indent=2, ensure_ascii=False)
    print(f"Results successfully saved to {json_path}")
    return best_metrics


if __name__ == "__main__":
    main()
