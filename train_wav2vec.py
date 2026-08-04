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
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

# --- CONFIGURATION ---
DATASET_PATH = "/home/naslia/Study/Project1/Code/Dataset/ThaiSER_cleaned/script"
MODELS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Models")
MODEL_NAME = "facebook/wav2vec2-base"  # เปลี่ยนเป็น pure pre-trained model เพื่อแก้ nan loss

BATCH_SIZE = 8
EPOCHS = 5
LEARNING_RATE = 1e-5
SAMPLE_RATE = 16000
MAX_LEN = 16000 * 3  # 3 วินาที

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


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 1. เตรียมข้อมูล
    train_files, val_files = prepare_dataset()
    print("Train samples:", len(train_files))
    print("Val samples  :", len(val_files))

    feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)

    train_dataset = ThaiSERWav2VecDataset(train_files, feature_extractor)
    val_dataset = ThaiSERWav2VecDataset(val_files, feature_extractor)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 2. โหลด Pre-trained Wav2Vec2 Model
    model = AutoModelForAudioClassification.from_pretrained(
        MODEL_NAME,
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

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # สร้างโฟลเดอร์สำหรับเซฟโมเดล
    os.makedirs(MODELS_PATH, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    best_model_path = os.path.join(MODELS_PATH, f"best_wav2vec2_model_{timestamp}.pt")

    best_val_loss = float("inf")

    # 3. เทรนโมเดล
    print("\nStarting Wav2Vec2 training...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [Train]")
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

        train_loss = train_loss / total
        train_acc = (correct / total) * 100.0

        # Evaluation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for input_values, attention_mask, labels in tqdm(val_loader, desc=f"Epoch {epoch}/{EPOCHS} [Val]  "):
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
