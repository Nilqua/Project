import os
import torch
import torchaudio
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from pyannote.audio import Model
from pyannote.audio import Inference
from dotenv import load_dotenv

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading Speaker Embedding model...")
    
    # ใช้โมเดล Embedding มาตรฐานของ Pyannote 3.1
    model = Model.from_pretrained("pyannote/wespeaker-voxceleb-resnet34-LM", token=os.environ["HF_TOKEN"])
    model.to(device)
    inference = Inference(model, window="whole")

    audio_dir = os.path.join(PROJECT_ROOT, "Dataset", "ThaiH2H", "audiofiles")
    
    embeddings = []
    labels = []
    
    # ทดสอบแค่ 30 ไฟล์แรกเพื่อความรวดเร็ว
    max_files = 30
    count = 0
    
    print("Extracting voice fingerprints (embeddings)...")
    for i in range(1, 100):
        if count >= max_files: break
        for side in ["left", "right"]:
            file_name = f"{i}_{side}.wav"
            file_path = os.path.join(audio_dir, file_name)
            
            if not os.path.exists(file_path): continue
            
            # โหลดไฟล์และตัดมาแค่ 10 วินาทีตรงกลางเพื่อเลี่ยงความเงียบตอนต้น
            wav, sr = torchaudio.load(file_path)
            
            if wav.shape[1] > sr * 10:
                mid_point = wav.shape[1] // 2
                wav = wav[:, mid_point - (sr*5) : mid_point + (sr*5)]
            
            # สกัดลายนิ้วมือเสียง (Embedding)
            emb = inference({"waveform": wav, "sample_rate": sr})
            
            # แปลงเป็น 1D array กรณีที่เป็น 2D
            if isinstance(emb, np.ndarray):
                emb = emb.flatten()
                
            embeddings.append(emb)
            labels.append(file_name)
            count += 1
            if count >= max_files: break

    print("Clustering speakers...")
    X = np.array(embeddings)
    
    # ใช้ Agglomerative Clustering จัดกลุ่มเสียงที่คล้ายกัน
    # Threshold 0.3 คือระยะห่าง (Cosine distance) ยิ่งน้อยยิ่งต้องเหมือนกันมากถึงจะจับกลุ่มกัน
    clustering = AgglomerativeClustering(
        n_clusters=None, 
        distance_threshold=0.3, 
        metric="cosine", 
        linkage="average"
    )
    cluster_labels = clustering.fit_predict(X)
    
    # จัดเรียงผลลัพธ์
    clusters = {}
    for label, cluster_id in zip(labels, cluster_labels):
        if cluster_id not in clusters:
            clusters[cluster_id] = []
        clusters[cluster_id].append(label)
        
    print("\n=== Speaker Clustering Results ===")
    for cid, files in sorted(clusters.items()):
        if len(files) > 1:
            print(f"✅ Cluster {cid} (Agent / โผล่หลายสาย): {files}")
        else:
            print(f"👤 Cluster {cid} (Customer / ลูกค้าทั่วไป): {files}")

if __name__ == "__main__":
    main()
