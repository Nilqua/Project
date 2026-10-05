# Project 1: Call Center Speech Emotion Analysis & Service Rating

## Phase 1: Speech Emotion Recognition (SER) [Completed]
- [x] ดาวน์โหลดและจัดโครงสร้าง ThaiSER Dataset (5 อารมณ์: Angry, Frustrated, Happy, Neutral, Sad)
- [x] พัฒนา Baseline CNN Model จาก Mel-Spectrogram (`train_cnn.py`)
- [x] Fine-tune Wav2Vec 2.0 (`airesearch/wav2vec2-large-xlsr-53-th`)
- [x] รันการทดลองเปรียบเทียบครบ 6 Configurations (2 โมเดล × 3 subsets) พร้อมบันทึกผล 4 metrics (Acc, Precision, Recall, F1)
- [x] บันทึกตารางสรุปผลลงใน `outputs/experiments/experiment_results_summary.md`

## Phase 2: Speaker Diarization & Audio Pipeline [Completed / In Progress]
- [x] เตรียมและจัดระเบียบชุดข้อมูลบทสนทนาจริง Thai-H2H Call Center
- [x] เชื่อมต่อ `pyannote.audio 3.1` สำหรับ Speaker Diarization
- [x] พัฒนาเทคนิค Sliding Window (Window 4.0s, Hop 1.0s) พร้อม Batch Inference บน GPU
- [x] สกัดค่า Continuous Valence จาก Softmax Probabilities
- [x] ทดสอบ Pipeline บนไฟล์สนทนาจริง (`call_1`, `call_2`) ใน `evaluate_call_center_real.py`
- [ ] พัฒนาระบบ Role Assignment อัตโนมัติ (ตรวจจับท่อนทักทายเปิดสายของ Agent เพื่อระบุตัวตนคู่สนทนาบน Mono Audio)

## Phase 3: Service Rating Engine & Analytics [Completed / Refinement]
- [x] ออกแบบสูตรคำนวณคะแนนการบริการ (Agent Baseline Stability + Customer De-escalation Bonus)
- [x] พัฒนาระบบแบ่งช่วงบทสนทนา 3 ระยะ (Initial, Mid, Final) และวัด Emotion Trajectory
- [x] พัฒนากราฟิกสรุปผลระดับ Presentation:
  - Graph 1: Dual-Track Emotion Timeline (Customer vs Agent)
  - Graph 2: Phase Shift & Trajectory Comparison
  - Graph 3: QA Service Score Card (0–5 ดาว) & Customer Emotion Distribution Pie Chart
- [ ] ปรับเกณฑ์ Threshold การตัดคะแนนความหงุดหงิด/โกรธ และโบนัสคลี่คลายอารมณ์ให้เสถียรยิ่งขึ้น

## Phase 4: Web Application Prototype & Dashboard [Next Focus]
- [ ] **Backend (FastAPI/Flask) - ธนากร:**
  - [ ] API Endpoint สำหรับอัปโหลดไฟล์เสียงบทสนทนา (.wav, .mp3)
  - [ ] Pipeline Worker: Preprocessing (16kHz Mono) -> Diarization -> Wav2Vec2 Sliding Window -> Rating Engine
  - [ ] API Endpoint ส่งผลลัพธ์ JSON (Timestamps, Valence curve, KPI Scores, Critical Segments)
- [ ] **Frontend & Dashboard (React/HTML5) - ฉัตรมงคล:**
  - [ ] หน้า Upload Call Audio
  - [ ] หน้า Agent Performance Overview (Service Score, Status, Summary Stats)
  - [ ] แดชบอร์ดแสดงผลกราฟเส้นคู่ขนาน (Interactive Dual-Track Emotion Timeline)
  - [ ] เครื่องเล่นเสียงพร้อม Critical Incident Highlighting (คลิกข้ามไปฟังจุดที่อารมณ์หลุดได้ทันที)

## Phase 5: Documentation & Presentation
- [x] จัดทำเอกสารข้อเสนอโครงงาน CS02D-V1.4 และ CS02S
- [x] ปรึกษาและรับคำแนะนำจากอาจารย์ที่ปรึกษา (ผศ.ดร.วีณาวดี ม่วงอ้น)
- [ ] เตรียมสไลด์นำเสนอความคืบหน้าโครงงาน (ใช้วิเคราะห์กราฟจริงจาก Call 1 และ Call 2)
- [ ] จัดทำเล่มรายงานปริญญานิพนธ์ฉบับสมบูรณ์ (ตามกำหนดการ ต.ค. 2569)