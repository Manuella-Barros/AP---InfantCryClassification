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
# retorna MFCCs que é um array NumPy 2D (shape: (n_mfcc, n_frames). n_mfcc → número de coeficientes (ex: 13). n_frames → número de "janelas de tempo" que o áudio foi dividido (depende da duração e do hop_length)
mfccs = librosa.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=13)
mfccs_transposta = mfccs.T  # Transpõe o array para de (n_mfcc, n_frames) para (n_frames, n_mfcc)
print(f"Mfccs shape: {mfccs.shape}")

## CALCULAR OS DELTAS ==========================================================
# O delta e delta-delta são recursos adicionais que capturam como o som muda ao longo do tempo, o que é extremamente útil para modelos de classificação de áudio
# MFCCs capturam características estáticas do som (em uma pequena janela de tempo).
# Delta (ou 1ª derivada) mostra a velocidade com que os MFCCs estão mudando.
# Delta-delta (ou 2ª derivada) mostra a aceleração ou variação da variação dos MFCCs.
delta_mfccs = librosa.feature.delta(mfccs) # Calcula a primeira derivada dos MFCCs (delta)
delta_mfccs_transposta = delta_mfccs.T  # Transpõe o array para de (n_mfcc, n_frames) para (n_frames, n_mfcc)

delta2_mfccs = librosa.feature.delta(mfccs, order=2) # Calcula a segunda derivada dos MFCCs (delta-delta)
delta2_mfccs_transposta = delta2_mfccs.T  # Transpõe o array para de (n_mfcc, n_frames) para (n_frames, n_mfcc)

print(f"Delta_mfccs shape: {delta_mfccs.shape}")
print(f"Delta2_mfccs shape: {delta2_mfccs.shape}")

# JUNTANDO OS 3 ARRAYS (MFCCs, deltas e delta-deltas) em um único array 2D empilhando verticalmente ======================================================
mfccs_concateneted = np.concatenate((mfccs_transposta, delta_mfccs_transposta, delta2_mfccs_transposta), axis=0)
print(f"mfccs_concateneted shape: {mfccs_concateneted.shape}")

## POR QUE TRANSPOR? ========================================================
# A transposição é feita para que cada linha represente um "frame" de tempo e cada coluna represente um coeficiente MFCC, delta ou delta-delta.
# na pratica o mfcc e os deltas retornam (n_mfcc, n_frames) e o modelo espera (n_frames, n_mfcc) para cada amostra.
# Então vc transpõe para adptar o resultado ao formato esperado pelo modelo.

## VIZUALIZAÇÃO ===========================================================
# # Mostra o espectrograma de cada
# show_mfcc_spectogram(mfccs, delta_mfccs, delta2_mfccs, sample_rate, mfccs_concateneted)

# # Salva o espectrograma de cada
# save_file_mfcc_spectogram(mfccs, delta_mfccs, delta2_mfccs, sample_rate, "mfcc_spectogram_belly_pain_example.png", mfccs_concateneted)