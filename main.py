from pathlib import Path
from sklearn.model_selection import train_test_split
from data_augmentation import augmented_train_dataset

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

# Divide entre treino e teste
# O x é o audio e y é a classe
x_train, x_test, y_train, y_test = train_test_split(file_paths, labels, test_size=0.2, stratify=labels, random_state=42)

print(f"Número de arquivos de treino: {len(x_train)}")
print(f"Número de arquivos de teste: {len(x_test)}")
print(f"Classes únicas: {set(labels)}")

# Gera os dados aumentados e atualiza os conjuntos de treino
x_train, y_train = augmented_train_dataset(x_train, y_train)

# print(f"file_paths: {file_paths}")
# print(f"labels: {labels}")
print(f"Dataset Augmentado: ")
print(f"Número de arquivos de treino: {len(x_train)}")
print(f"Número de arquivos de teste: {len(x_test)}")
print(f"Classes únicas: {set(labels)}")