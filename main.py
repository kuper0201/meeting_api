from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import shutil
import os
from faster_whisper import WhisperModel
from datetime import datetime
import sqlite3
import json

app = FastAPI()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# DB 초기화
def init_db():
    conn = sqlite3.connect('transcriptions.db', check_same_thread=False)
    conn.execute("PRAGMA encoding = 'UTF-8'")
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

@app.delete("/delete/{transcription_id}")
async def delete_transcription(transcription_id: int):
    conn = sqlite3.connect('transcriptions.db', check_same_thread=False)
    conn.execute("PRAGMA encoding = 'UTF-8'")
    c = conn.cursor()
    
    # 해당 ID의 데이터가 존재하는지 확인
    c.execute('SELECT filename FROM transcriptions WHERE id = ?', (transcription_id,))
    result = c.fetchone()
    
    if result is None:
        conn.close()
        error_content = json.dumps({"error": "Transcription not found"}, ensure_ascii=False)
        return JSONResponse(
            status_code=404,
            content=json.loads(error_content),
            media_type="application/json; charset=utf-8"
        )
    
    # 연관된 오디오 파일 삭제
    filename = result[0]
    file_path = os.path.join(UPLOAD_DIR, filename)
    if os.path.exists(file_path):
        os.remove(file_path)
    
    # DB에서 데이터 삭제
    c.execute('DELETE FROM transcriptions WHERE id = ?', (transcription_id,))
    conn.commit()
    conn.close()
    
    success_content = json.dumps({"message": "Transcription deleted successfully"}, ensure_ascii=False)
    return JSONResponse(
        content=json.loads(success_content),
        media_type="application/json; charset=utf-8"
    )


@app.get("/transcriptions/")
async def get_transcriptions():
    conn = sqlite3.connect('transcriptions.db', check_same_thread=False)
    conn.execute("PRAGMA encoding = 'UTF-8'")
    conn.row_factory = sqlite3.Row  # 딕셔너리 형태로 결과 반환
    c = conn.cursor()
    c.execute('''
        SELECT id, filename, duration, text, summary, created_at 
        FROM transcriptions 
        ORDER BY created_at DESC
    ''')
    rows = c.fetchall()
    conn.close()
    
    transcriptions = []
    for row in rows:
        transcriptions.append({
            "id": row["id"],
            "filename": row["filename"],
            "duration": row["duration"],
            "text": row["text"],
            "summary": row["summary"],
            "created_at": row["created_at"]
        })
    
    # JSON 직렬화 시 ensure_ascii=False 설정
    json_content = json.dumps({"transcriptions": transcriptions}, ensure_ascii=False, indent=2)
    
    return JSONResponse(
        content=json.loads(json_content),
        media_type="application/json; charset=utf-8",
        headers={"Content-Type": "application/json; charset=utf-8"}
    )

# Whisper 모델 미리 로딩 (서버 시작 시)
model = WhisperModel("medium", device="auto")

@app.post("/upload-audio/")
async def upload_audio(file: UploadFile = File(...)):
    allowed_extensions = [".wav", ".aac", ".mp3", ".ogg", ".m4a"]
    filename = file.filename
    ext = os.path.splitext(filename)[1].lower()
    
    if ext not in allowed_extensions:
        error_content = json.dumps({"error": f"Unsupported file type: {ext}"}, ensure_ascii=False)
        return JSONResponse(
            status_code=400,
            content=json.loads(error_content),
            media_type="application/json; charset=utf-8",
            headers={"Content-Type": "application/json; charset=utf-8"}
        )
    
    file_path = os.path.join(UPLOAD_DIR, filename)

    # 파일 저장
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 파일 업로드 성공 응답
    response = {
        "filename": filename,
        "file_path": file_path,
        "message": "File upload successful"
    }

    # 백그라운드 작업으로 STT 및 요약 처리
    async def process_audio_task(filename: str):
        file_path = os.path.join(UPLOAD_DIR, filename)
        
        # 🔥 저장된 파일로 Whisper 음성 인식 진행
        segments, info = model.transcribe(file_path, language="ko")

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
        conn = sqlite3.connect('transcriptions.db', check_same_thread=False)
        conn.execute("PRAGMA encoding = 'UTF-8'")
        c = conn.cursor()
        c.execute('''
            INSERT INTO transcriptions (filename, duration, text, summary, created_at)
            VALUES (?, ?, ?, ?, ?)
        ''', (filename, info.duration, full_text.strip(), summary, datetime.now()))
        conn.commit()
        conn.close()

    # 백그라운드에서 처리 시작
    import asyncio
    asyncio.create_task(process_audio_task(filename))
    
    json_content = json.dumps(response, ensure_ascii=False)
    return JSONResponse(
        content=json.loads(json_content),
        media_type="application/json; charset=utf-8",
        headers={"Content-Type": "application/json; charset=utf-8"}
    )

@app.post("/process-audio/{filename}")
async def process_audio(filename: str):
    file_path = os.path.join(UPLOAD_DIR, filename)
    
    # 🔥 저장된 파일로 Whisper 음성 인식 진행
    segments, info = model.transcribe(file_path, language="ko")

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
    conn = sqlite3.connect('transcriptions.db', check_same_thread=False)
    conn.execute("PRAGMA encoding = 'UTF-8'")
    c = conn.cursor()
    c.execute('''
        INSERT INTO transcriptions (filename, duration, text, summary, created_at)
        VALUES (?, ?, ?, ?, ?)
    ''', (filename, info.duration, full_text.strip(), summary, datetime.now()))
    conn.commit()
    conn.close()

    response_data = {
        "filename": filename,
        "duration": info.duration,
        "segments": result,
        "full_text": full_text.strip(),
        "summary": summary,
        "message": "Processing completed"
    }
    
    json_content = json.dumps(response_data, ensure_ascii=False)
    return JSONResponse(
        content=json.loads(json_content),
        media_type="application/json; charset=utf-8",
        headers={"Content-Type": "application/json; charset=utf-8"}
    )

# 🔥 main 실행 시 uvicorn 서버 자동 실행
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)