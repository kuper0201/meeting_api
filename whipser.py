"""
from pyannote.audio import Pipeline
import whisper
import wave
import contextlib

# 1. 화자 분리
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token="your_huggingface_token")
diarization = pipeline("S000017/000000.wav")

# 2. Whisper로 전체 음성 텍스트 추출
model = whisper.load_model("medium")
result = model.transcribe("S000017/000000.wav")

# 3. 화자 구간을 기준으로 출력 포맷 구성
for turn, _, speaker in diarization.itertracks(yield_label=True):
    print(f"[{speaker}] {turn.start:.1f}s - {turn.end:.1f}s")


import whisper
import torch

print(torch.backends.mps.is_available())
model = whisper.load_model("medium", device="mps")  # tiny, base, small, medium, large 중 택 1
result = model.transcribe("DGBAH21000201.wav")
print(result["text"])
"""

from faster_whisper import WhisperModel

model = WhisperModel("medium", device="auto")  # 자동으로 GPU(MPS) 사용

#segments, info = model.transcribe("DGBAH21000201.wav")
segments, info = model.transcribe("mac_record.wav")

for segment in segments:
    print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
