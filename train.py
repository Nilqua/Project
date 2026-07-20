import os
import glob
import re
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torchaudio.transforms as T
from tqdm import tqdm

# --- CONFIGURATION ---
DATASET_PATH = "/home/naslia/Study/Project1/Dataset/ThaiSER_cleaned/script"
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001
SAMPLE_RATE = 16000
MAX_LEN = int(SAMPLE_RATE * 3.0)  # 3 seconds audio

CLASSES = ["Angry", "Frustrated", "Happy", "Neutral", "Sad"]
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASSES)}


# --- DATASET CLASS ---
class ThaiSERDataset(Dataset):
    def __init__(self, file_list):
        self.file_list = file_list
        self.mel_transform = T.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=1024,
            hop_length=512,
            n_mels=64,
        )
        self.amplitude_to_db = T.AmplitudeToDB()

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        filepath, label = self.file_list[idx]

        # Load audio file
        waveform, sr = torchaudio.load(filepath)

        # Resample to 16kHz if needed
        if sr != SAMPLE_RATE:
            resampler = T.Resample(sr, SAMPLE_RATE)
            waveform = resampler(waveform)

        # Convert stereo to mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Pad or crop audio length
        if waveform.shape[1] < MAX_LEN:
            pad = MAX_LEN - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad))
        else:
            waveform = waveform[:, :MAX_LEN]

        # Compute Spectrogram
        mel_spec = self.mel_transform(waveform)
        mel_spec_db = self.amplitude_to_db(mel_spec)

        return mel_spec_db, label


# --- SIMPLE CNN MODEL ---
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=5):
        super(SimpleCNN, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x


# --- HELPER: SPLIT BY ACTOR (PREVENT DATA LEAKAGE) ---
def get_actor_id(filename):
    match = re.search(r"actor(\d+)", filename)
    return match.group(1) if match else "unknown"


def prepare_dataset():
    actor_data = {}

    for cname in CLASSES:
        folder = os.path.join(DATASET_PATH, cname)
        if not os.path.exists(folder):
            continue

        label_idx = CLASS_TO_IDX[cname]
        files = glob.glob(os.path.join(folder, "*.flac"))

        for fpath in files:
            actor = get_actor_id(os.path.basename(fpath))
            if actor not in actor_data:
                actor_data[actor] = []
            actor_data[actor].append((fpath, label_idx))

    # Split actors into Train (80%) and Validation (20%)
    actors = list(actor_data.keys())
    random.seed(42)
    random.shuffle(actors)

    val_size = int(len(actors) * 0.2)
    val_actors = set(actors[:val_size])
    train_actors = set(actors[val_size:])

    train_files = []
    for act in train_actors:
        train_files.extend(actor_data[act])

    val_files = []
    for act in val_actors:
        val_files.extend(actor_data[act])

    return train_files, val_files


# --- MAIN TRAINING LOOP ---
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 1. Prepare data
    train_files, val_files = prepare_dataset()
    print(f"Train samples: {len(train_files)} | Val samples: {len(val_files)}")

    train_dataset = ThaiSERDataset(train_files)
    val_dataset = ThaiSERDataset(val_files)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 2. Setup model, loss, optimizer
    model = SimpleCNN(num_classes=5).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 3. Train loop
    print("\nStarting training...")
    for epoch in range(1, EPOCHS + 1):
        # --- TRAIN ---
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [Train]"):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            train_correct += predicted.eq(labels).sum().item()
            train_total += labels.size(0)

        epoch_train_loss = train_loss / train_total
        epoch_train_acc = train_correct / train_total * 100

        # --- VALIDATION ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch}/{EPOCHS} [Val]  "):
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                val_correct += predicted.eq(labels).sum().item()
                val_total += labels.size(0)

        epoch_val_loss = val_loss / val_total
        epoch_val_acc = val_correct / val_total * 100

        print(f"Epoch {epoch}/{EPOCHS} Summary:")
        print(f"  Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.2f}%")
        print(f"  Val Loss  : {epoch_val_loss:.4f} | Val Acc  : {epoch_val_acc:.2f}%\n")


if __name__ == "__main__":
    main()
