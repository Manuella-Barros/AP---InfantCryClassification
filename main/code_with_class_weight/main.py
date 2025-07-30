from pathlib import Path
import sys
# diz pro python procurar os modulos a partir da raiz
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from sklearn.model_selection import train_test_split
from data_augmentation import augmented_train_dataset
from mfcc.mfcc import get_mfcc_from_file_list
from models.mlp import mlp
from models.cnn import cnn
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder
from helpers.generate_classes_distributions import generate_classes_distributions
from helpers.generate_confusion_matrix import generate__confusion_matrix

## PEGA E AJUSTA OS DADOS INICIAIS =============================================
# Caminho para a pasta principal contendo subpastas (cada subpasta é uma classe)
data_dir = Path("Baby Cry Pattern Archive")
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
print(f"Dataset Augmentado: ")
print(f"Número de arquivos de treino: {len(x_train)}")
print(f"Número de arquivos de teste: {len(x_test)}")
print(f"Classes únicas: {set(labels)}")

## EXTRAI OS MFCCS DOS ARQUIVOS DE ÁUDIO =======================================
x_train_mfcc = get_mfcc_from_file_list(x_train)
x_test_mfcc = get_mfcc_from_file_list(x_test)

print(f"MFCCs de treino: {len(x_train_mfcc)}")
print(f"MFCCs de teste: {len(x_test_mfcc)}")

# retorna as classes reiais e as classes previstas
y_true_classes, y_pred_classes = mlp(x_train_mfcc, y_train, x_test_mfcc, y_test)
# y_true_classes, y_pred_classes = cnn(x_train_mfcc, y_train, x_test_mfcc, y_test)

assert len(set(x_train).intersection(set(x_test))) == 0

# Matrix de confusão
generate__confusion_matrix(y_true_classes, y_pred_classes)

# Distribuição de classes
generate_classes_distributions(y_train, y_test)

plt.show()