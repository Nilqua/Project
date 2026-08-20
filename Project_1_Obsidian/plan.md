# แผนงานโครงการ: ระบบ AI วิเคราะห์คุณภาพการทำงานของพนักงาน Call Center

> **Thai Call Center Speech Analytics System**
> ระบบใช้ Speech Emotion Recognition + Speaker Diarization เพื่อวิเคราะห์ Interaction ระหว่าง Customer กับ Agent และประเมินคุณภาพการรับสายของ Agent

## ภาพรวมสถาปัตยกรรม (Architecture Overview)

```text
                 Call Center Audio
                        │
                        ▼
                Speaker Diarization
                        │
                 ┌──────┴──────┐
                 ▼             ▼
              Customer       Agent
                 │             │
                 ▼             ▼
             Emotion Model  Emotion Model
                 │             │
                 └──────┬──────┘
                        ▼
                Emotion Timeline
                        │
                        ▼
                Conversation Analysis
                        │
                        ▼
                Agent Performance
                        │
                        ▼
                  Web Dashboard
```

---

## ลำดับการทำงาน (Phases)

### Phase 1 — สร้าง “หู” ให้ AI (Speech Emotion Recognition)
**สถานะ:** เสร็จสิ้นเกือบทั้งหมด (Completed)
**เป้าหมาย:** สร้างโมเดลฟังภาษาไทยแล้วจับอารมณ์ได้จริง โดยยังไม่สนใจว่าเป็นใครพูด
- [x] หา Dataset และจัดกลุ่ม (ใช้ชุดข้อมูล ThaiSER)
- [x] เลือก Emotion labels (Neutral, Happy, Sad, Angry, Frustrated)
- [x] ทำ Baseline (สร้างทั้ง CNN และ Fine-tune pretrained `facebook/wav2vec2-base` ได้ความแม่นยำสูง)
- [x] สร้างสคริปต์ตรวจจับอารมณ์และการวาดกราฟตัดแบ่งเวลา (`use_graph.py`)

### Phase 2 — เอา “หู” ไปฟังสาย Call Center (Call Analysis Pipeline)
**สถานะ:** รอดำเนินการ (Next Step)
**เป้าหมาย:** นำเสียงสนทนาจริงมาวิเคราะห์แยกว่า "ใครพูดตอนไหน" และจับคู่ความสัมพันธ์ของอารมณ์
- [ ] ติดตั้งระบบ Speaker Diarization เพื่อแยก Speaker 1 (Agent) และ Speaker 2 (Customer)
- [ ] นำช่วงเสียงของแต่ละคน (Timestamps) เข้า Emotion Model เพื่อวิเคราะห์อารมณ์แบบเจาะจงบุคคล
- [ ] พล็อต Emotion Timeline ของทั้งคู่ เพื่อดู Interaction (เช่น ลูกค้าโกรธ -> พนักงานรับมือด้วยความสงบ -> ลูกค้ากลับมาปกติ)

### Phase 3 — ประเมินคุณภาพการทำงาน (Performance Analytics)
**สถานะ:** รอดำเนินการ
**เป้าหมาย:** แปลงความเปลี่ยนแปลงทางอารมณ์ให้เป็น Performance Score ของพนักงาน
- [ ] กำหนด Business Logic สำหรับวัดคะแนน Agent (เช่น การระงับอารมณ์, ความสงบ, De-escalation)
- [ ] กำหนด Metric ฝั่ง Customer (เช่น อารมณ์แรกเริ่ม, จุดโกรธสูงสุด, การฟื้นฟูอารมณ์)
- [ ] ทดสอบสร้างสรุปรายงาน Performance Score ควบคู่กับ Ground Truth

### Phase 4 — พัฒนาระบบแสดงผล (Web App)
**สถานะ:** รอดำเนินการ
**เป้าหมาย:** สร้าง UI สำหรับผู้ใช้งานจริง (Dashboard)
- [ ] พัฒนาหน้า Dashboard ให้อัปโหลดไฟล์เสียง (Upload Call)
- [ ] แสดงผลลัพธ์จากการวิเคราะห์ทั้งหมดในหน้าเดียว (Emotion Timeline คู่ขนาน + Performance Score + Summary)
