import os
import glob
import re
import random
import datetime
import numpy as np
import librosa
import tensorflow as tf
keras = tf.keras

# --- CONFIGURATION ---
DATASET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Dataset", "ThaiSER_cleaned", "script")
MODELS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Models")
BATCH_SIZE = 16
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

    # เพิ่มมิติ channel ท้ายสุด ให้เป็น (64, 94, 1) ตามที่ CNN ต้องการ
    mel_db = np.expand_dims(mel_db, axis=-1)
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


class DataGenerator(keras.utils.Sequence):
    # ตัวนี้ทำหน้าที่ป้อนข้อมูลทีละ batch ให้โมเดลตอน train/val
    def __init__(self, file_list, batch_size=32, shuffle=True, augment=False):
        super().__init__()
        self.file_list = file_list
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.indices = np.arange(len(file_list))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        # จำนวน batch ทั้งหมดใน 1 epoch
        return int(np.ceil(len(self.file_list) / self.batch_size))

    def __getitem__(self, index):
        # ดึงไฟล์ของ batch นี้มา preprocess ทีละไฟล์
        start = index * self.batch_size
        end = start + self.batch_size
        batch_idx = self.indices[start:end]

        X = []
        y = []
        for i in batch_idx:
            filepath, label = self.file_list[i]
            X.append(load_spectrogram(filepath, augment=self.augment))
            y.append(label)

        return np.array(X), np.array(y)

    def on_epoch_end(self):
        # สลับลำดับใหม่ทุกจบ epoch (เฉพาะตอน train)
        if self.shuffle:
            np.random.shuffle(self.indices)


def build_model():
    reg = keras.regularizers.l2(L2_REG)
    model = keras.Sequential([
        keras.layers.Input(shape=(N_MELS, 94, 1)),

        keras.layers.Conv2D(16, 3, padding="same", activation="relu", kernel_regularizer=reg),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D(2),
        keras.layers.Dropout(DROPOUT_CONV),

        keras.layers.Conv2D(32, 3, padding="same", activation="relu", kernel_regularizer=reg),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D(2),
        keras.layers.Dropout(DROPOUT_CONV),

        keras.layers.Flatten(),
        keras.layers.Dense(64, activation="relu", kernel_regularizer=reg),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(DROPOUT_DENSE),
        keras.layers.Dense(len(CLASSES), activation="softmax"),
    ])
    return model


def main():
    print("TensorFlow version:", tf.__version__)

    # 1. เตรียมข้อมูล
    train_files, val_files = prepare_dataset()
    print("Train samples:", len(train_files))
    print("Val samples  :", len(val_files))

    train_gen = DataGenerator(train_files, batch_size=BATCH_SIZE, shuffle=True, augment=True)
    val_gen = DataGenerator(val_files, batch_size=BATCH_SIZE, shuffle=False, augment=False)

    # 2. สร้างโมเดล
    model = build_model()
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    # สร้างโฟลเดอร์สำหรับเซฟโมเดลเคียงข้างโฟลเดอร์ Dataset
    os.makedirs(MODELS_PATH, exist_ok=True)

    # ตั้งชื่อไฟล์โมเดลพร้อมประทับเวลา (Timestamp)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(MODELS_PATH, f"best_model_{timestamp}.keras")

    # เซฟเฉพาะโมเดลที่ดีที่สุด (val_loss ต่ำที่สุด)
    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath=model_path,
        monitor="val_loss",
        save_best_only=True,
        mode="min",
        verbose=1
    )

    # 3. เทรนโมเดล
    print("\nStarting training...")
    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=[checkpoint]
    )


if __name__ == "__main__":
    main()