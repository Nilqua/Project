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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# --- CONFIGURATION ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_DATA_DIR = os.path.join(PROJECT_ROOT, "Dataset", "ThaiSER_cleaned")
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "experiments")
MODELS_PATH = os.path.join(PROJECT_ROOT, "Models")
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 0.001
SAMPLE_RATE = 16000
MAX_LEN = 16000 * 3  # 3 วินาที

# Audio features config
N_FFT = 1024
HOP_LENGTH = 512
N_MELS = 64

# Model hyperparameters
L2_REG = 1e-4
DROPOUT_CONV = 0.25
DROPOUT_DENSE = 0.5

CLASSES = ["Angry", "Frustrated", "Happy", "Neutral", "Sad"]
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASSES)}


def get_actor_id(filename):
    # ดึงเลข actor ออกจากชื่อไฟล์ เช่น actor001_xxx.flac -> "001"
    match = re.search(r"actor(\d+)", filename)
    if match:
        return match.group(1)
    return "unknown"


def augment_audio(wav):
    # สุ่มใส่ white noise
    if random.random() < 0.5:
        noise = np.random.normal(0, 0.002, len(wav))
        wav = wav + noise

    # สุ่ม pitch shift
    if random.random() < 0.5:
        step = random.uniform(-2.0, 2.0)
        wav = librosa.effects.pitch_shift(y=wav, sr=SAMPLE_RATE, n_steps=step)

    return wav


def load_spectrogram(filepath, augment=False):
    # โหลดไฟล์เสียง librosa จะ resample และแปลง mono ให้เองอัตโนมัติ
    wav, sr = librosa.load(filepath, sr=SAMPLE_RATE)

    # ทำให้ทุกไฟล์ยาวเท่ากัน (3 วิ) ถ้าสั้นไปก็เติม 0 ถ้ายาวไปก็ตัดทิ้ง
    if len(wav) < MAX_LEN:
        wav = np.pad(wav, (0, MAX_LEN - len(wav)))
    else:
        wav = wav[:MAX_LEN]

    # ทำ data augmentation หากเป็นชุดฝึก
    if augment:
        wav = augment_audio(wav)

    # แปลงคลื่นเสียงเป็น mel spectrogram แล้วแปลงเป็น dB
    mel = librosa.feature.melspectrogram(
        y=wav, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # ปรับมิติเป็น (1, 64, 94) ให้สอดคล้องกับ PyTorch (Channel First: 1, Height, Width)
    mel_db = np.expand_dims(mel_db, axis=0)
    return mel_db


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

class ThaiSERDataset(Dataset):
    # PyTorch Dataset สำหรับป้อนข้อมูล Mel-Spectrogram
    def __init__(self, file_list, augment=False):
        self.file_list = file_list
        self.augment = augment

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        filepath, label = self.file_list[idx]
        spec = load_spectrogram(filepath, augment=self.augment)
        return torch.tensor(spec, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


class SimplePyTorchCNN(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(DROPOUT_CONV),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(DROPOUT_CONV),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 16 * 23, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(DROPOUT_DENSE),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def parse_args():
    parser = argparse.ArgumentParser(description="Train CNN on ThaiSER dataset")
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
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs (default: 10)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size (default: 64)"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
        help="Learning rate (default: 0.001)"
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
    print("PyTorch CNN Speech Emotion Recognition")
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

    train_dataset = ThaiSERDataset(train_files, augment=True)
    val_dataset = ThaiSERDataset(val_files, augment=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=(len(train_dataset) > args.batch_size)
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # 2. สร้างโมเดล
    model = SimplePyTorchCNN(num_classes=len(CLASSES)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=L2_REG)

    os.makedirs(MODELS_PATH, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    best_model_path = os.path.join(MODELS_PATH, f"best_cnn_{norm_subset}_{timestamp}.pt")

    best_val_loss = float("inf")
    best_metrics = None

    num_epochs = 1 if args.dry_run else args.epochs

    # 3. เทรนโมเดล
    print("\nStarting CNN training...")
    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs} [Train]")
        for inputs, labels in pbar:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            if args.dry_run:
                print("  [Dry Run] Finished 1 train batch.")
                break

        train_loss = train_loss / max(1, total)
        train_acc = (correct / max(1, total)) * 100.0

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_preds = []
        val_labels = []

        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch}/{num_epochs} [Val]  ")
            for inputs, labels in val_pbar:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
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
                "model": "cnn",
                "subset": norm_subset,
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
    json_path = os.path.join(args.output_dir, f"cnn_{norm_subset}_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(best_metrics, f, indent=2, ensure_ascii=False)
    print(f"Results successfully saved to {json_path}")
    return best_metrics


if __name__ == "__main__":
    main()