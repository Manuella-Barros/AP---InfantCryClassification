import librosa
import soundfile as sf
import numpy as np
import random
import os
from collections import defaultdict

def apply_augmentation(signal, sr, augmentation_type):
    """Aplica uma transformação de áudio específica"""
    if augmentation_type == 'time_stretch':
        rate = random.uniform(0.8, 1.2)  # ±20% de velocidade
        return librosa.effects.time_stretch(signal, rate=rate)
    elif augmentation_type == 'pitch_shift':
        steps = random.randint(-3, 3)  # ±3 semitons
        return librosa.effects.pitch_shift(signal, sr=sr, n_steps=steps)
    elif augmentation_type == 'noise':
        noise = np.random.normal(0, 0.005, len(signal))  # 0.5% de ruído
        return signal + noise
    elif augmentation_type == 'time_shift':
        shift = random.randint(0, int(sr * 0.3))  # Até 300ms de deslocamento
        return np.roll(signal, shift)
    else:
        return signal

def augmented_train_dataset(x_train, y_train, target_size_per_class=300, output_dir="augmented_audio"):
    """
    Balanceia o dataset aplicando data augmentation nas classes minoritárias
    
    :param x_train: Lista de caminhos para arquivos de áudio
    :param y_train: Lista de rótulos correspondentes
    :param target_size_per_class: Número desejado de amostras por classe
    :param output_dir: Diretório para salvar arquivos aumentados
    :return: augmented_x, augmented_y (listas balanceadas)
    """
    os.makedirs(output_dir, exist_ok=True)
    augmentation_techniques = ['time_stretch', 'pitch_shift', 'noise', 'time_shift']
    
    class_counts = defaultdict(list)
    for x, y in zip(x_train, y_train):
        class_counts[y].append(x)

    augmented_x = []
    augmented_y = []

    for label, files in class_counts.items():
        current_count = len(files)
        needed = max(0, target_size_per_class - current_count)
        
        # Primeiro adiciona todos os originais
        augmented_x.extend(files)
        augmented_y.extend([label] * current_count)

        # Gera amostras aumentadas se necessário
        if needed > 0:
            print(f"Gerando {needed} amostras aumentadas para classe: {label}")
            
            for i in range(needed):
                # Escolhe um arquivo aleatório da classe
                original_file = random.choice(files)
                signal, sr = librosa.load(original_file, sr=None)
                
                # Aplica 1-3 transformações aleatórias
                num_transforms = random.randint(1, 3)
                augmented_signal = signal.copy()
                for _ in range(num_transforms):
                    tech = random.choice(augmentation_techniques)
                    augmented_signal = apply_augmentation(augmented_signal, sr, tech)
                
                # Salva o novo arquivo
                base_name = os.path.splitext(os.path.basename(original_file))[0]
                new_filename = f"{base_name}_aug_{i}.wav"
                output_path = os.path.join(output_dir, new_filename)
                sf.write(output_path, augmented_signal, sr)
                
                # Adiciona ao dataset
                augmented_x.append(output_path)
                augmented_y.append(label)

    return augmented_x, augmented_y