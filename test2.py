import librosa
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Carregar áudio
file_path = Path(r"D:/Projetos/Unirio/venv/dataset_donateacry_corpus/belly_pain/69BDA5D6-0276-4462-9BF7-951799563728-1436936185-1.1-m-26-bp.wav")  # coloque o caminho do seu arquivo de áudio aqui
audio, sr = librosa.load(file_path, sr=None)  # sr=None para manter a taxa original

print(f"Duração do áudio: {len(audio)/sr:.2f} segundos")
print(f"Taxa de amostragem: {sr} Hz")
print(f"Array de áudio shape: {audio.shape}")

# Plotar waveform
plt.figure(figsize=(10, 4))
plt.plot(np.linspace(0, len(audio)/sr, num=len(audio)), audio)
plt.xlabel('Tempo (s)')
plt.ylabel('Amplitude')
plt.title('Forma de onda do áudio')
plt.show()