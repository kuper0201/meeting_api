from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import shutil
import os

app = FastAPI()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

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

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {"filename": filename, "message": "Upload successful"}

# 🔥 main 실행 시 uvicorn 서버 자동 실행
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
