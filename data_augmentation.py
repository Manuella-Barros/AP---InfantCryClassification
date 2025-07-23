import librosa
import numpy as np
import soundfile as sf
# manipulação de caminhos e diretórios
import os

import data_augmentation_testes as aug

AUGMENTED_DIR = "dataset_augmented"
os.makedirs(AUGMENTED_DIR, exist_ok=True)

import os
import librosa
import soundfile as sf
import data_augmentation_testes as aug

AUGMENTED_DIR = "dataset_augmented"
os.makedirs(AUGMENTED_DIR, exist_ok=True)

def augmented_train_dataset(x_train, y_train):
    x_train_augmented = []
    y_train_augmented = []

    for i, path in enumerate(x_train):
        label = y_train[i]
        signal, sr = librosa.load(path, sr=None)

        # Time stretch (áudio mais lento)
        stretched = aug.time_stretch(signal, stretch_rate=0.5)
        # Pitch shift (muda o tom)
        pitched = aug.pitch_scale(signal, sr=sr, num_steps=2)

        # Nome base do arquivo sem extensão
        base_name = os.path.splitext(os.path.basename(path))[0]

        # Salva os arquivos aumentados
        sf.write(os.path.join(AUGMENTED_DIR, f"{base_name}_stretched.wav"), stretched, sr)
        sf.write(os.path.join(AUGMENTED_DIR, f"{base_name}_pitched.wav"), pitched, sr)

        # Adiciona os sinais aumentados à lista
        x_train_augmented.append(os.path.join(AUGMENTED_DIR, f"{base_name}_stretched.wav"))
        y_train_augmented.append(label)

        x_train_augmented.append(os.path.join(AUGMENTED_DIR, f"{base_name}_pitched.wav"))
        y_train_augmented.append(label)

    # Concatena os originais com os aumentados
    x_train_final = x_train + x_train_augmented
    y_train_final = y_train + y_train_augmented

    return x_train_final, y_train_final