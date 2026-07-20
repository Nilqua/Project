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

DATASET_PATH = "/home/naslia/Study/Project1/Dataset/ThaiSER_cleaned/script"
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001
SAMPLE_RATE = 16000
MAX_LEN = 16000 * 3

CLASSES = ["Angry", "Frustrated", "Happy", "Neutral", "Sad"]

CLASS_TO_IDX = {}
for idx, name in enumerate(CLASSES):
    CLASS_TO_IDX[name] = idx


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
        item = self.file_list[idx]
        filepath = item[0]
        label = item[1]

        waveform, sr = torchaudio.load(filepath)

        if sr != SAMPLE_RATE:
            resampler = T.Resample(sr, SAMPLE_RATE)
            waveform = resampler(waveform)

        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        current_len = waveform.shape[1]
        if current_len < MAX_LEN:
            pad = MAX_LEN - current_len
            waveform = torch.nn.functional.pad(waveform, (0, pad))
        else:
            waveform = waveform[:, :MAX_LEN]

        mel_spec = self.mel_transform(waveform)
        mel_spec_db = self.amplitude_to_db(mel_spec)

        return mel_spec_db, label


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=5):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 16 * 23, 64)
        self.fc2 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def get_actor_id(filename):
    match = re.search(r"actor(\d+)", filename)
    if match:
        return match.group(1)
    else:
        return "unknown"


def prepare_dataset():
    actor_data = {}

    for cname in CLASSES:
        folder = os.path.join(DATASET_PATH, cname)
        if os.path.exists(folder):
            label_idx = CLASS_TO_IDX[cname]
            files = glob.glob(os.path.join(folder, "*.flac"))

            for fpath in files:
                fname = os.path.basename(fpath)
                actor = get_actor_id(fname)

                if actor not in actor_data:
                    actor_data[actor] = []

                actor_data[actor].append((fpath, label_idx))

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


def main():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print("Using device:", device)

    train_files, val_files = prepare_dataset()
    print("Train samples:", len(train_files))
    print("Val samples  :", len(val_files))

    train_dataset = ThaiSERDataset(train_files)
    val_dataset = ThaiSERDataset(val_files)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = SimpleCNN(num_classes=5).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("\nStarting training...")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [Train]"):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)

            for i in range(len(labels)):
                if predicted[i] == labels[i]:
                    correct += 1
                total += 1

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
                _, predicted = torch.max(outputs, 1)

                for i in range(len(labels)):
                    if predicted[i] == labels[i]:
                        val_correct += 1
                    val_total += 1

        val_loss = val_loss / val_total
        val_acc = (val_correct / val_total) * 100.0

        print(f"Epoch {epoch}/{EPOCHS} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss  : {val_loss:.4f} | Val Acc  : {val_acc:.2f}%\n")


if __name__ == "__main__":
    main()
