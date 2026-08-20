import torch
from pyannote.audio import Pipeline

import os
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), ".env"))
HF_TOKEN = os.environ["HF_TOKEN"]

def run_diarization(audio_path):
    print(f"Loading pyannote pipeline... (This might take a while for the first time)")
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        token=HF_TOKEN
    )

    # ส่งเข้า GPU ถ้ามี
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline.to(device)

    print(f"Analyzing audio: {audio_path}")
    diarization = pipeline(audio_path)

    # พิมพ์ผลลัพธ์ว่าใครพูดช่วงไหนบ้าง
    print("Diarization Results:")
    # เวอร์ชันใหม่ return DiarizeOutput ต้องดึง .speaker_diarization ออกมาก่อน
    annotation = diarization.speaker_diarization
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        print(f"[{turn.start:05.1f}s - {turn.end:05.1f}s] Speaker {speaker}")
    return annotation

if __name__ == "__main__":
    test_audio = "/home/naslia/Study/Project1/Code/Dataset/ThaiH2H/mixed_audio/1_mixed.wav"
    run_diarization(test_audio)
