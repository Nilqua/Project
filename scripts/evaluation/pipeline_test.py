import os
import torch
import librosa
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
import torch.nn.functional as F
from pyannote.audio import Pipeline
from dotenv import load_dotenv

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

MODEL_PATH = os.path.join(PROJECT_ROOT, "Models", "best_wav2vec2_model_20260804_233106.pt") 
MODEL_NAME = "facebook/wav2vec2-base"
SAMPLE_RATE = 16000

CLASSES = ["Neutral", "Happy", "Sad", "Angry", "Frustrated"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IDX_TO_CLASS = {i: c for i, c in enumerate(CLASSES)}

# แปลงอารมณ์เป็นคะแนนเพื่อวาดกราฟดูเทรนด์ขึ้นลง
SCORE_MAP = {
    "Happy": 1.0,
    "Neutral": 0.0,
    "Sad": -0.5,
    "Frustrated": -0.75,
    "Angry": -1.0
}

def predict_emotion(chunk, model, feature_extractor, device):
    inputs = feature_extractor(
        chunk, 
        sampling_rate=SAMPLE_RATE, 
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=SAMPLE_RATE * 10, # จำกัดความยาวสูงสุด 10 วินาทีเพื่อไม่ให้ OOM
        return_attention_mask=True,
    )
    input_values = inputs.input_values.to(device)
    attention_mask = inputs.attention_mask.to(device)
    
    with torch.no_grad():
        outputs = model(input_values=input_values, attention_mask=attention_mask)
        probs = F.softmax(outputs.logits, dim=-1).squeeze(0).cpu().numpy()
        
    dominant_idx = np.argmax(probs)
    return IDX_TO_CLASS[dominant_idx], probs

def run_pipeline(audio_file, diarize_model, emotion_model, feature_extractor, device):
    print(f"\nProcessing: {os.path.basename(audio_file)}")
    
    # 1. Diarization (แยกเสียง)
    print("  -> Running Diarization...")
    diarization = diarize_model(audio_file)
    annotation = diarization.speaker_diarization
    
    # 2. Load Audio
    wav, sr = librosa.load(audio_file, sr=SAMPLE_RATE)
    
    # 3. Emotion Prediction per Segment
    print("  -> Predicting Emotions...")
    results = {"SPEAKER_00": {"time": [], "score": [], "emotion": []}, 
               "SPEAKER_01": {"time": [], "score": [], "emotion": []}}
    
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        start_sample = int(turn.start * sr)
        end_sample = int(turn.end * sr)
        chunk = wav[start_sample:end_sample]
        
        if len(chunk) < sr * 0.5: # ข้ามท่อนที่สั้นกว่า 0.5 วินาที
            continue
            
        emotion, probs = predict_emotion(chunk, emotion_model, feature_extractor, device)
        mid_time = (turn.start + turn.end) / 2.0
        
        if speaker not in results:
            results[speaker] = {"time": [], "score": [], "emotion": []}
            
        results[speaker]["time"].append(mid_time)
        results[speaker]["score"].append(SCORE_MAP[emotion])
        results[speaker]["emotion"].append(emotion)
        print(f"     [{turn.start:05.1f}s - {turn.end:05.1f}s] {speaker}: {emotion}")
        
    return results

def plot_results(results, save_path):
    plt.figure(figsize=(15, 5))
    colors = {"SPEAKER_00": "blue", "SPEAKER_01": "orange"}
    labels = {"SPEAKER_00": "Speaker 0 (Agent)", "SPEAKER_01": "Speaker 1 (Customer)"}
    
    for spk, data in results.items():
        if not data["time"]:
            continue
        # วาดเส้นกราฟ
        plt.plot(data["time"], data["score"], label=labels.get(spk, spk), 
                 color=colors.get(spk, "gray"), marker='o', linestyle='-', linewidth=2, alpha=0.8)
        
    plt.axhline(0, color='black', linewidth=1, linestyle='--')
    plt.yticks([-1.0, -0.75, -0.5, 0.0, 1.0], ["Angry", "Frustrated", "Sad", "Neutral", "Happy"])
    plt.xlabel("Time (seconds)")
    plt.ylabel("Emotion Valence")
    plt.title(f"Conversation Emotion Trend ({os.path.basename(save_path)})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"  -> Saved plot to {save_path}")
    plt.close()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("Loading Diarization Model...")
    diarize_model = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", token=os.environ["HF_TOKEN"])
    diarize_model.to(device)
    
    print("Loading Emotion Model...")
    feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
    emotion_model = AutoModelForAudioClassification.from_pretrained(
        MODEL_NAME, num_labels=len(CLASSES), label2id=CLASS_TO_IDX, id2label=IDX_TO_CLASS
    )
    if os.path.exists(MODEL_PATH):
        emotion_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print("Loaded trained emotion weights.")
    
    emotion_model.to(device)
    emotion_model.eval()
    
    # ทดสอบแค่ 2 ไฟล์ตามคำขอ
    test_files = [
        os.path.join(PROJECT_ROOT, "Dataset", "ThaiH2H", "mixed_audio", "1_mixed.wav"),
        os.path.join(PROJECT_ROOT, "Dataset", "ThaiH2H", "mixed_audio", "2_mixed.wav")
    ]
    
    out_dir = os.path.join(PROJECT_ROOT, "outputs")
    os.makedirs(out_dir, exist_ok=True)
    
    for idx, f in enumerate(test_files, 1):
        if not os.path.exists(f):
            print(f"File not found: {f}")
            continue
            
        res = run_pipeline(f, diarize_model, emotion_model, feature_extractor, device)
        plot_path = os.path.join(out_dir, f"pipeline_test_plot_{idx}.png")
        plot_results(res, plot_path)

if __name__ == "__main__":
    main()
