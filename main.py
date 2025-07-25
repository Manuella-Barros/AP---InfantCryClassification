from pathlib import Path
from sklearn.model_selection import train_test_split
from data_augmentation import augmented_train_dataset
from mfcc.mfcc import get_mfcc_from_file_list, get_mfcc_from_file
from models.mlp import mlp

## PEGA E AJUSTA OS DADOS INICIAIS =============================================
# Caminho para a pasta principal contendo subpastas (cada subpasta é uma classe)
data_dir = Path("dataset_donateacry_corpus")
file_paths = [] # vetor com o caminho dos arquivos de áudio
labels = [] # vetor com as labels dos arquivos de áudio

# Coleta todos os arquivos de áudio e suas classes (nomes das pastas)
for class_dir in data_dir.iterdir():
    if class_dir.is_dir():
        for audio_file in class_dir.glob("*.wav"):
            file_paths.append(audio_file)
            labels.append(class_dir.name)

## DIVIDE ENTRE TREINO E TESTE =================================================
# O x é o audio e y é a classe
x_train, x_test, y_train, y_test = train_test_split(file_paths, labels, test_size=0.2, stratify=labels, random_state=42)

# print(f"Número de arquivos de treino: {len(x_train)}")
# print(f"Número de arquivos de teste: {len(x_test)}")
# print(f"Classes únicas: {set(labels)}")

## APLICA O DATA AUGMENTATION NOS AUDIOS DE TREINAMENTO ========================
# # Gera os dados aumentados e atualiza os conjuntos de treino
x_train, y_train = augmented_train_dataset(x_train, y_train)

# print(f"file_paths: {file_paths}")
# print(f"labels: {labels}")
# print(f"Dataset Augmentado: ")
# print(f"Número de arquivos de treino: {len(x_train)}")
# print(f"Número de arquivos de teste: {len(x_test)}")
# print(f"Classes únicas: {set(labels)}")

## EXTRAI OS MFCCS DOS ARQUIVOS DE ÁUDIO =======================================
x_train_mfcc = get_mfcc_from_file_list(x_train)
x_test_mfcc = get_mfcc_from_file_list(x_test)

print(f"MFCCs de treino: {len(x_train_mfcc)}")
print(f"MFCCs de teste: {len(x_test_mfcc)}")

mlp(x_train_mfcc, y_train, x_test_mfcc, y_test)