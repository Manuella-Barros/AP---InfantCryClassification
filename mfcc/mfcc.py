import librosa
import numpy as np
from sklearn.preprocessing import StandardScaler

def get_mfcc_from_file(audio_file, n_mfcc=13, sr=16000, max_frames=300):
    """Extrai MFCCs com deltas, normalização e padding"""
    
    # 1. Carregamento com pré-ênfase
    signal, sr = librosa.load(audio_file, sr=sr)
    signal = librosa.effects.preemphasis(signal)
    
    # 2. Extração de MFCCs com parâmetros otimizados
    mfccs = librosa.feature.mfcc(
        y=signal,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=2048,
        hop_length=512,
        win_length=1024,
        fmin=50,
        fmax=8000
    )
    
    # 3. Cálculo de deltas
    delta1 = librosa.feature.delta(mfccs)
    delta2 = librosa.feature.delta(mfccs, order=2)
    
    # 4. Normalização por arquivo
    features = np.vstack([mfccs, delta1, delta2])  # (3*n_mfcc, time_steps)
    features = StandardScaler().fit_transform(features.T)  # Normaliza por coluna
    
    # 5. Padding/Truncamento
    if features.shape[0] < max_frames:
        pad_width = ((0, max_frames - features.shape[0]), (0, 0))
        features = np.pad(features, pad_width, mode='constant')
    else:
        features = features[:max_frames]
    
    return features  # (max_frames, 3*n_mfcc)

def get_mfcc_from_file_list(audio_file_list, max_frames=300):
    """Processa lista de arquivos garantindo consistência"""
    mfccs_list = []
    for audio_file in audio_file_list:
        try:
            mfccs = get_mfcc_from_file(audio_file, max_frames=max_frames)
            mfccs_list.append(mfccs)
        except Exception as e:
            print(f"Erro ao processar {audio_file}: {str(e)}")
            continue
            
    return np.array(mfccs_list)