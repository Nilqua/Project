import os
import torch
import librosa
import soundfile as sf
import numpy as np
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
import torch.nn.functional as F
from pyannote.audio import Pipeline
from dotenv import load_dotenv

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

MAX_CLIPS_PER_EMOTION = 10
SAMPLE_RATE = 16000

def predict_emotion(chunk, model, feature_extractor, device):
    inputs = feature_extractor(chunk, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True, truncation=True, max_length=SAMPLE_RATE * 10, return_attention_mask=True)
    with torch.no_grad():
        outputs = model(input_values=inputs.input_values.to(device), attention_mask=inputs.attention_mask.to(device))
        probs = F.softmax(outputs.logits, dim=-1).squeeze(0).cpu().numpy()
    classes = ["Neutral", "Happy", "Sad", "Angry", "Frustrated"]
    return classes[np.argmax(probs)]

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Extracting up to {MAX_CLIPS_PER_EMOTION} clips for EACH emotion...")

    diarize_model = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", token=os.environ["HF_TOKEN"]).to(device)
    model_name = "facebook/wav2vec2-base"
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    
    classes = ["Neutral", "Happy", "Sad", "Angry", "Frustrated"]
    class2idx = {c: i for i, c in enumerate(classes)}
    idx2class = {i: c for i, c in enumerate(classes)}
    
    emotion_model = AutoModelForAudioClassification.from_pretrained(model_name, num_labels=5, label2id=class2idx, id2label=idx2class)
    model_path = os.path.join(PROJECT_ROOT, "Models", "best_wav2vec2_model_20260804_233106.pt")
    if os.path.exists(model_path):
        emotion_model.load_state_dict(torch.load(model_path, map_location=device))
    emotion_model.to(device)
    emotion_model.eval()

    # ตั้งค่าจำนวนที่ต้องการสกัดของแต่ละอารมณ์
    counts = {e: 0 for e in classes}
    
    # หากเคยสกัดมาแล้ว (เช่น Angry) ก็นับไว้ก่อนจะได้ไม่สกัดซ้ำซ้อนจนเกิน
    for e in classes:
        out_dir = os.path.join(PROJECT_ROOT, "outputs", "samples", e)
        if os.path.exists(out_dir):
            counts[e] = len([f for f in os.listdir(out_dir) if f.endswith('.wav')])
        print(f"Current {e} count: {counts[e]}/{MAX_CLIPS_PER_EMOTION}")

    audio_dir = os.path.join(PROJECT_ROOT, "Dataset", "ThaiH2H", "mixed_audio")
    
    for file in sorted(os.listdir(audio_dir)):
        if all(c >= MAX_CLIPS_PER_EMOTION for c in counts.values()):
            print("Finished collecting all emotions!")
            break
        if not file.endswith(".wav"): continue
        
        filepath = os.path.join(audio_dir, file)
        print(f"Scanning: {file}")
        
        diarization = diarize_model(filepath)
        wav, sr = librosa.load(filepath, sr=SAMPLE_RATE)
        
        for turn, _, speaker in diarization.speaker_diarization.itertracks(yield_label=True):
            if all(c >= MAX_CLIPS_PER_EMOTION for c in counts.values()):
                break
            
            start_s, end_s = turn.start, turn.end
            if end_s - start_s < 1.0: continue
            
            start_sample, end_sample = int(start_s * sr), int(end_s * sr)
            chunk = wav[start_sample:end_sample]
            
            emo = predict_emotion(chunk, emotion_model, feature_extractor, device)
            
            if counts[emo] < MAX_CLIPS_PER_EMOTION:
                counts[emo] += 1
                out_dir = os.path.join(PROJECT_ROOT, "outputs", "samples", emo)
                os.makedirs(out_dir, exist_ok=True)
                out_name = f"{file.split('.')[0]}_{speaker}_{start_s:.1f}s-{end_s:.1f}s.wav"
                out_path = os.path.join(out_dir, out_name)
                sf.write(out_path, chunk, sr)
                print(f"[{counts[emo]}/{MAX_CLIPS_PER_EMOTION}] Saved {emo}: {out_name}")

if __name__ == "__main__":
    main()
