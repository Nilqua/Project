import os
import torch
import librosa
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
import torch.nn.functional as F

# --- การตั้งค่าเบื้องต้น ---
# เปลี่ยนชื่อไฟล์ให้ตรงกับโมเดลจริงที่คุณเทรนเสร็จ
MODEL_PATH = "Models/best_wav2vec2_model_20260804_233106.pt" 
MODEL_NAME = "facebook/wav2vec2-base"
SAMPLE_RATE = 16000
WINDOW_SIZE_SEC = 3.0  # ขนาดหน้าต่างที่ใช้ฟังเสียง (3 วินาที)
HOP_SIZE_SEC = 1.0     # ความละเอียดในการขยับไปข้างหน้า (ทีละ 1 วินาที)

CLASSES = ["Neutral", "Happy", "Sad", "Angry", "Frustrated"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IDX_TO_CLASS = {i: c for i, c in enumerate(CLASSES)}

def analyze_emotion_change(audio_path, model, feature_extractor, device):
    """
    ฟังก์ชันสแกนไฟล์เสียงแบบ Sliding Window 
    เพื่อดูค่าความมั่นใจ (Probability) ของอารมณ์ต่างๆ ในทุกๆ 1 วินาที
    """
    print(f"Loading audio: {audio_path}")
    wav, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
    total_duration = len(wav) / sr
    
    window_samples = int(WINDOW_SIZE_SEC * sr)
    hop_samples = int(HOP_SIZE_SEC * sr)
    
    timestamps = []
    emotion_probs = {c: [] for c in CLASSES}
    dominant_emotions = []
    
    model.eval()
    with torch.no_grad():
        if len(wav) < window_samples:
            # ถ้าไฟล์สั้นกว่าหน้าต่าง 3 วิ ให้วิเคราะห์ทั้งไฟล์รวดเดียว
            starts = [0]
            ends = [len(wav)]
        else:
            starts = list(range(0, len(wav) - window_samples + 1, hop_samples))
            ends = [s + window_samples for s in starts]
            
        for start_idx, end_idx in zip(starts, ends):
            chunk = wav[start_idx:end_idx]
            
            # บันทึกเวลาจุดกึ่งกลางของหน้าต่างเสียง
            mid_time = (start_idx + end_idx) / 2.0 / sr
            timestamps.append(mid_time)
            
            # เตรียมข้อมูลเข้าโมเดล
            inputs = feature_extractor(
                chunk, 
                sampling_rate=SAMPLE_RATE, 
                return_tensors="pt",
                padding="max_length",
                max_length=window_samples,
                truncation=True,
                return_attention_mask=True,
                do_normalize=True
            )
            
            input_values = inputs.input_values.to(device)
            attention_mask = inputs.attention_mask.to(device)
            
            # ให้โมเดลทำนายอารมณ์ และแปลงค่าออกมาเป็น % Probability
            outputs = model(input_values=input_values, attention_mask=attention_mask)
            probs = F.softmax(outputs.logits, dim=-1).squeeze(0).cpu().numpy()
            
            for i, c in enumerate(CLASSES):
                emotion_probs[c].append(probs[i])
                
            dominant_idx = np.argmax(probs)
            dominant_emotions.append(IDX_TO_CLASS[dominant_idx])

    return timestamps, emotion_probs, dominant_emotions, total_duration

def plot_emotion_graph(timestamps, emotion_probs, dominant_emotions, save_path="emotion_change.png"):
    """
    สร้างกราฟแสดงความเปลี่ยนแปลงของอารมณ์ (พร้อม Smoothing แก้ปัญหากราฟยุ่บยั่บ)
    """
    plt.figure(figsize=(24, 6)) # ขยายความยาวกราฟ
    
    # 1. ทำ Moving Average Smoothing ให้กราฟเนียนขึ้น ไม่กระโดดไปมาทุกวินาที
    window_len = 5
    smoothed_probs = {}
    for c in CLASSES:
        probs = np.array(emotion_probs[c])
        # เกลี่ยค่าด้วยวินโดว์ 5 วินาที
        if len(probs) >= window_len:
            smoothed = np.convolve(probs, np.ones(window_len)/window_len, mode='same')
        else:
            smoothed = probs
        smoothed_probs[c] = smoothed
        plt.plot(timestamps, smoothed, label=c, alpha=0.8, linewidth=2, marker='o')
        
    # 2. คำนวณอารมณ์หลัก (Dominant) ใหม่จากค่าที่เกลี่ยแล้ว
    new_dominant = []
    for i in range(len(timestamps)):
        vals = [smoothed_probs[c][i] for c in CLASSES]
        new_dominant.append(IDX_TO_CLASS[np.argmax(vals)])
        
    plt.title("Emotion Probability over Time")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Probability (0.0 - 1.0)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    
    # 3. หาจุดเปลี่ยนอารมณ์จากเส้นที่ Smooth แล้ว
    change_points = []
    for i in range(1, len(new_dominant)):
        if new_dominant[i] != new_dominant[i-1]:
            change_time = (timestamps[i] + timestamps[i-1]) / 2.0
            change_points.append((change_time, new_dominant[i-1], new_dominant[i]))
            
            # plt.axvline(x=change_time, color='red', linestyle='--', alpha=0.8, linewidth=1)
            # plt.text(change_time, 1.02, f"{new_dominant[i]}", color='red', rotation=45, fontsize=8, ha='left')
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Graph saved to {save_path}")
    return change_points

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("Loading model for segmentation analysis...")
    feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
    model = AutoModelForAudioClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(CLASSES),
        label2id=CLASS_TO_IDX,
        id2label=IDX_TO_CLASS,
    )
    
    # โหลดค่าน้ำหนักที่เทรนไว้ (ถ้าหาไม่เจอจะรันแบบดัมมี่เพื่อโชว์กราฟเฉยๆ)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print("Loaded trained weights.")
    else:
        print(f"Warning: Trained weights '{MODEL_PATH}' not found. Using pre-trained baseline.")
        
    model.to(device)
    
    # ใช้ไฟล์ sample.mp3 ทดสอบ
    test_audio_path = "/home/naslia/Study/Project1/Code/Dataset/ThaiSER_cleaned/script/Angry/s001_clip_actor001_script2_2_2b.flac"
    
    # เริ่มสแกนเสียง
    timestamps, emotion_probs, dominant_emotions, duration = analyze_emotion_change(
        test_audio_path, model, feature_extractor, device
    )
    
    # วาดกราฟ
    change_points = plot_emotion_graph(timestamps, emotion_probs, dominant_emotions)
    
    print("\n--- Audio Segmentation Logic Report ---")
    print(f"Total Audio Duration: {duration:.2f} sec")
    if not change_points:
         dominant = dominant_emotions[0] if dominant_emotions else "Unknown"
         print(f"No emotion changes detected. The entire clip is classified as: '{dominant}'")
    else:
        for t, old_e, new_e in change_points:
            print(f"-> CUT POINT DETECTED: Emotion shifted from '{old_e}' to '{new_e}' at {t:.2f} seconds.")

if __name__ == "__main__":
    main()
