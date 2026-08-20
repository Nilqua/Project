# Call Center Emotion Analysis

ระบบวิเคราะห์อารมณ์จากเสียงสนทนาใน Call Center โดยใช้ Deep Learning
แยกเสียงพนักงานกับลูกค้าออกจากกัน แล้ววิเคราะห์อารมณ์ตลอดสาย เพื่อประเมินคุณภาพการบริการ

## Project Structure

```
Code/
├── scripts/
│   ├── data_prep/              # สคริปต์เตรียมข้อมูล
│   │   ├── organize_thaiser.py # จัดระเบียบ ThaiSER dataset
│   │   ├── rename_h2h_final.py # เปลี่ยนชื่อไฟล์ Thai H2H ให้อ่านง่าย
│   │   ├── mix_audio.py        # ผสมเสียง left/right เป็น mono
│   │   ├── diarize_audio.py    # แยกเสียงคนพูด (Speaker Diarization)
│   │   └── test_match.py       # ทดสอบจับคู่ไฟล์เสียง
│   ├── training/               # สคริปต์เทรนโมเดล
│   │   ├── train_cnn.py        # Baseline CNN + Mel-Spectrogram
│   │   └── train_wav2vec.py    # Fine-tune Wav2Vec 2.0 (XLSR-53-TH)
│   └── evaluation/             # สคริปต์ทดสอบและวาดกราฟ
│       └── use_graph.py        # Sliding window + กราฟอารมณ์ตามเวลา
├── Dataset/                    # ชุดข้อมูลเสียง (ไม่เข้า git)
├── Models/                     # โมเดลที่เทรนแล้ว (ไม่เข้า git)
├── outputs/                    # รูปกราฟผลลัพธ์ (ไม่เข้า git)
└── Project_1_Obsidian/         # เอกสารแผนงาน
    ├── plan.md
    └── Todo.md
```

## Phases

| Phase | รายละเอียด | สถานะ |
|-------|-----------|-------|
| 1. Emotion Model | เทรนโมเดลจับอารมณ์จากเสียงภาษาไทย (ThaiSER) | ✅ ทำงานได้ |
| 2. Call Analysis Pipeline | แยกเสียงคนพูด (Diarization) → วิเคราะห์อารมณ์ตามไทม์ไลน์ | 🔄 กำลังทำ |
| 3. Agent Performance | สรุปคะแนนประเมินพนักงานจากอารมณ์ลูกค้า | ⬜ ยังไม่เริ่ม |
| 4. Web Dashboard | แดชบอร์ดแสดงผลวิเคราะห์ | ⬜ ยังไม่เริ่ม |

## Datasets

- **ThaiSER** — ชุดข้อมูลอารมณ์เสียงภาษาไทย (5 อารมณ์: Angry, Happy, Neutral, Sad, Frustrated)
  - ที่มา: [vistec-AI/dataset-releases](https://github.com/vistec-AI/dataset-releases/releases/tag/v1)
- **Thai H2H** — เสียงสนทนา Call Center จริง แยกซ้าย-ขวา (พนักงาน-ลูกค้า) พร้อม transcript

## Getting Started

### 1. สร้าง Virtual Environment

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. เตรียมข้อมูล ThaiSER (Phase 1)

ดาวน์โหลด ThaiSER dataset แล้ววางไว้ที่ `Dataset/ThaiSER/`

```bash
python scripts/data_prep/organize_thaiser.py
```

### 3. เทรนโมเดลจับอารมณ์

```bash
# Baseline CNN
python scripts/training/train_cnn.py

# Wav2Vec 2.0 (ต้องมี GPU)
python scripts/training/train_wav2vec.py
```

### 4. เตรียมข้อมูล Thai H2H (Phase 2)

ดาวน์โหลด Thai H2H dataset แล้ววางไว้ที่ `Dataset/ThaiH2H/`

```bash
# จับคู่และเปลี่ยนชื่อไฟล์
python scripts/data_prep/rename_h2h_final.py

# ผสมเสียง left+right เป็นไฟล์เดียว (จำลอง mono)
python scripts/data_prep/mix_audio.py
```

### 5. รัน Speaker Diarization

ต้องมี [Hugging Face Token](https://huggingface.co/settings/tokens) (ฟรี) และกด Accept ที่:
- [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
- [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
- [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1)

ใส่ Token ในไฟล์ `scripts/data_prep/diarize_audio.py` แล้วรัน:

```bash
python scripts/data_prep/diarize_audio.py
```

### 6. วาดกราฟอารมณ์

```bash
python scripts/evaluation/use_graph.py
```

## Tech Stack

- **PyTorch** — Deep Learning framework
- **Wav2Vec 2.0 (XLSR-53-TH)** — Pre-trained speech model สำหรับภาษาไทย
- **pyannote.audio** — Speaker Diarization (แยกเสียงคนพูด)
- **librosa** — Audio feature extraction
