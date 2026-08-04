import os
import glob
import re
import random
import datetime
import numpy as np
import librosa
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# --- CONFIGURATION ---
DATASET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Dataset", "ThaiSER_cleaned", "script")
MODELS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Models")
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


def prepare_dataset():
    # เก็บไฟล์เสียงแยกตาม actor ก่อน เพื่อไม่ให้เสียงคนเดียวกันหลุดไปทั้ง train และ val
    actor_data = {}

    for cname in CLASSES:
        folder = os.path.join(DATASET_PATH, cname)
        if not os.path.exists(folder):
            continue

        label = CLASS_TO_IDX[cname]
        files = glob.glob(os.path.join(folder, "*.flac"))

        for fpath in files:
            fname = os.path.basename(fpath)
            actor = get_actor_id(fname)
            actor_data.setdefault(actor, []).append((fpath, label))

    actors = list(actor_data.keys())
    random.seed(42)
    random.shuffle(actors)

    val_size = int(len(actors) * 0.2)
    val_actors = actors[:val_size]
    train_actors = actors[val_size:]

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


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("PyTorch version:", torch.__version__)
    print("Using device:", device)

    # 1. เตรียมข้อมูล
    train_files, val_files = prepare_dataset()
    print("Train samples:", len(train_files))
    print("Val samples  :", len(val_files))

    train_dataset = ThaiSERDataset(train_files, augment=True)
    val_dataset = ThaiSERDataset(val_files, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 2. สร้างโมเดล
    model = SimplePyTorchCNN(num_classes=len(CLASSES)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=L2_REG)

    # สร้างโฟลเดอร์สำหรับเซฟโมเดล
    os.makedirs(MODELS_PATH, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    best_model_path = os.path.join(MODELS_PATH, f"best_cnn_model_{timestamp}.pt")

    best_val_loss = float("inf")

    # 3. เทรนโมเดล
    print("\nStarting CNN training...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [Train]")
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

        train_loss = train_loss / total
        train_acc = (correct / total) * 100.0

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch}/{EPOCHS} [Val]  "):
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_loss = val_loss / val_total
        val_acc = (val_correct / val_total) * 100.0

        print(f"Epoch {epoch}/{EPOCHS} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss  : {val_loss:.4f} | Val Acc  : {val_acc:.2f}%")

        if val_loss < best_val_loss:
            print(f"  Val loss improved from {best_val_loss:.4f} to {val_loss:.4f}, saving model to {best_model_path}")
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
        print()


if __name__ == "__main__":
    main()