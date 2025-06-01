from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import shutil
import os
from faster_whisper import WhisperModel
from datetime import datetime
import sqlite3

app = FastAPI()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# DB 초기화
def init_db():
    conn = sqlite3.connect('transcriptions.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS transcriptions
        (id INTEGER PRIMARY KEY AUTOINCREMENT,
         filename TEXT,
         duration REAL,
         text TEXT,
         summary TEXT,
         created_at TIMESTAMP)
    ''')
    conn.commit()
    conn.close()

init_db()

# Whisper 모델 미리 로딩 (서버 시작 시)
model = WhisperModel("medium", device="auto")

@app.post("/upload-audio/")
async def upload_audio(file: UploadFile = File(...)):
    allowed_extensions = [".wav", ".aac", ".mp3", ".ogg"]
    filename = file.filename
    ext = os.path.splitext(filename)[1].lower()
    
    if ext not in allowed_extensions:
        return JSONResponse(
            status_code=400,
            content={"error": f"Unsupported file type: {ext}"}
        )
    
    file_path = os.path.join(UPLOAD_DIR, filename)

    # 파일 저장
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 🔥 저장된 파일로 Whisper 음성 인식 진행
    segments, info = model.transcribe(file_path)

    result = []
    full_text = ""
    for segment in segments:
        result.append({
            "start": segment.start,
            "end": segment.end,
            "text": segment.text
        })
        full_text += segment.text + " "

    print(full_text.strip())

    # 전사된 텍스트 요약 진행
    from summ import summarize
    summary = summarize(full_text.strip())
    print("\n[요약 결과]")
    print(summary)

    # DB에 저장
    conn = sqlite3.connect('transcriptions.db')
    c = conn.cursor()
    c.execute('''
        INSERT INTO transcriptions (filename, duration, text, summary, created_at)
        VALUES (?, ?, ?, ?, ?)
    ''', (file_path, info.duration, full_text.strip(), summary, datetime.now()))
    conn.commit()
    conn.close()

    return {
        "filename": filename,
        "duration": info.duration,
        "segments": result,
        "full_text": full_text.strip(),
        "summary": summary,
        "message": "Upload, transcription and DB storage successful"
    }
# 🔥 main 실행 시 uvicorn 서버 자동 실행
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)