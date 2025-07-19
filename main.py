import tensorflow as tf
from pathlib import Path

# Caminho para a pasta principal contendo subpastas (cada subpasta é uma classe)
data_dir = Path("dataset_donateacry_corpus")
default_seed = 123  # Define uma semente para garantir a reprodutibilidade

# Cria o dataset de treinamento (80%)
train_ds = tf.keras.utils.audio_dataset_from_directory(
    directory=data_dir,
    labels="inferred",  # infere as classes a partir dos nomes das subpastas
    validation_split=0.2, # divide 20% para validação/teste e sobra 80% para treinamento
    subset="training",
    seed=default_seed,  # define uma semente para garantir divisão reproduzível
    output_sequence_length=16000,  # a rede neural precisa receber as entradas de mesma duração, então serve para ajustar a duração dos áudios (em samples)
    batch_size=32 # quantos audios por batch (lote) serão processados de cada vez
)

# Cria o dataset de validação/teste (20%)
val_ds = tf.keras.utils.audio_dataset_from_directory(
    directory=data_dir,
    labels="inferred",
    validation_split=0.2,
    subset="validation",
    seed=default_seed,
    output_sequence_length=16000,
    batch_size=32
)

print(f"Número de batches de treinamento: {train_ds}")
print(f"Número de batches de validação: {val_ds}")