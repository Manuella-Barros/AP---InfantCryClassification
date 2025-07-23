import librosa
import matplotlib.pyplot as plt
import numpy as np

def get_mfcc_from_file(audio_file):
    ## CARREGAR O ARQUIVO DE AUDIO =============================================
    signal, sample_rate = librosa.load(audio_file)

    ## EXTRAIR O MFCC do áudio. ================================================
    mfccs = librosa.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=13)
    mfccs_transposta = mfccs.T 

    ## CALCULAR OS DELTAS =====================================================
    delta_mfccs = librosa.feature.delta(mfccs)
    delta_mfccs_transposta = delta_mfccs.T

    delta2_mfccs = librosa.feature.delta(mfccs, order=2)
    delta2_mfccs_transposta = delta2_mfccs.T

    # JUNTANDO OS 3 ARRAYS (MFCCs, deltas e delta-deltas) em um único array 2D empilhando verticalmente ==================================================
    mfccs_concateneted = np.concatenate((mfccs_transposta, delta_mfccs_transposta, delta2_mfccs_transposta), axis=0)
    return mfccs_concateneted

def get_mfcc_from_file_list(audio_file_list):
    mfccs_list = []
    for audio_file in audio_file_list:
        mfccs = get_mfcc_from_file(audio_file)
        mfccs_list.append(mfccs)
    return mfccs_list