"""
import soundfile as sf
import matplotlib.pyplot as plt
from simple_diarizer.diarizer import Diarizer
from simple_diarizer.utils import combined_waveplot

# 다이어라이저 초기화
diar = Diarizer(
    embed_model='xvec',      # 'xvec' 또는 'ecapa' 중 선택
    cluster_method='sc'      # 'ahc' 또는 'sc' 중 선택
)

# 음성 파일 경로와 화자 수 설정
WAV_FILE = '000017.wav'
NUM_SPEAKERS = 2  # 예: 2명의 화자

# 화자 분리 수행
segments = diar.diarize(WAV_FILE, num_speakers=NUM_SPEAKERS)

# 결과 시각화
signal, fs = sf.read(WAV_FILE)
combined_waveplot(signal, fs, segments)
plt.show()
"""

import soundfile as sf
import matplotlib.pyplot as plt
from simple_diarizer.diarizer import Diarizer
from simple_diarizer.utils import combined_waveplot

# 다이어라이저 초기화
diar = Diarizer(embed_model='xvec', cluster_method='sc')

# 오디오 파일과 화자 수 설정
WAV_FILE = "000017.wav"
NUM_SPEAKERS = 2

# 화자 분리 수행
segments = diar.diarize(WAV_FILE, num_speakers=NUM_SPEAKERS)

# 오디오 로드
signal, fs = sf.read(WAV_FILE)

# 시각화
combined_waveplot(signal, fs, segments)

# ⏱️ 시간축을 1초 단위로 설정
duration = len(signal) / fs
plt.xticks(
    ticks=[i for i in range(int(duration) + 1)],
    labels=[str(i) for i in range(int(duration) + 1)]
)
plt.xlabel("Time (seconds)")
plt.tight_layout()
plt.show()
