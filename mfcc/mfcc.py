import librosa
import matplotlib.pyplot as plt
import numpy as np
from helpers import show_mfcc_spectogram, save_file_mfcc_spectogram

audio_file = "dataset_donateacry_corpus/belly_pain/69BDA5D6-0276-4462-9BF7-951799563728-1436936185-1.1-m-26-bp.wav"

## CARREGAR O ARQUIVO DE AUDIO =================================================
# Retorna o "signal" que é um array NumPy 1D contendo os valores da forma de onda do áudio (amplitudes normalizadas entre -1.0 e 1.0).
# Retorna o "sample_rate" que é a taxa de amostragem do áudio (número de amostras por segundo).
signal, sample_rate = librosa.load(audio_file)

## EXTRAIR O MFCC (coeficientes cepstrais de frequência de Mel) de um sinal de áudio. ========================================================================
# n_mfcc: número de coeficientes MFCC a serem extraídos (13 é padrão em muitos estudos de fala/áudio).
# retorna MFCCs que é um array NumPy 2D (shape: (n_mfcc, n_frames)) onde cada coluna representa os coeficientes MFCC para um determinado quadro de tempo do sinal de áudio.
mfccs = librosa.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=13)
print(f"Mfccs shape: {mfccs.shape}") # Imprime o shape do array MFCCs (n_mfcc, n_frames)

## CALCULAR OS DELTAS ==========================================================
# O delta e delta-delta são recursos adicionais que capturam como o som muda ao longo do tempo, o que é extremamente útil para modelos de classificação de áudio
# MFCCs capturam características estáticas do som (em uma pequena janela de tempo).
# Delta (ou 1ª derivada) mostra a velocidade com que os MFCCs estão mudando.
# Delta-delta (ou 2ª derivada) mostra a aceleração ou variação da variação dos MFCCs.
delta_mfccs = librosa.feature.delta(mfccs) # Calcula a primeira derivada dos MFCCs (delta)
delta2_mfccs = librosa.feature.delta(mfccs, order=2) # Calcula a segunda derivada dos MFCCs (delta-delta)

print(f"Delta_mfccs shape: {delta_mfccs.shape}")
print(f"Delta2_mfccs shape: {delta2_mfccs.shape}")

# Junta os três arrays (MFCCs, deltas e delta-deltas) em um único array 2D empilhando verticalmente
mfccs_concateneted = np.concatenate((mfccs, delta_mfccs, delta2_mfccs), axis=0)
print(f"mfccs_concateneted shape: {mfccs_concateneted.shape}")

## VIZUALIZAÇÃO ===========================================================
# # Mostra o espectrograma de cada
# show_mfcc_spectogram(mfccs, delta_mfccs, delta2_mfccs, sample_rate, mfccs_concateneted)

# # Salva o espectrograma de cada
# save_file_mfcc_spectogram(mfccs, delta_mfccs, delta2_mfccs, sample_rate, "mfcc_spectogram_belly_pain_example.png", mfccs_concateneted)