import numpy as np
import soundfile as sf
import librosa

from helper import _plot_signal_and_augmented_signal

# adicionando ruídos ao áudio
def add_noise(signal, noise_factor):
    # Gera ruído com média 0, desvio padrão igual ao do sinal original e define a quantidade de amostras do ruído.
    noise = np.random.normal(0, signal.std(), signal.size)
    augmented_signal = signal + noise * noise_factor
    return augmented_signal

def time_stretch(signal, stretch_rate):
    # fator de estiramento (0.5 = metade da velocidade)
    return librosa.effects.time_stretch(y=signal, rate=stretch_rate)

def pitch_scale(signal, sr, num_steps):
    return librosa.effects.pitch_shift(y=signal, sr=sr, n_steps=num_steps)

def invert_polarity(signal):
    return signal * -1

def randomg_gain(signal, min_gain_factor, max_gain_factor):
    gain_factor = np.random.uniform(min_gain_factor, max_gain_factor)
    return signal * gain_factor

if __name__ == "__main__":
    signal, sr = librosa.load("dataset_donateacry_corpus/burping/5afc6a14-a9d8-45f8-b31d-c79dd87cc8c6-1430757039803-1.7-m-48-bu.wav")
    # augmented_signal = add_noise(signal, 0.5)
    # augmented_signal = time_stretch(signal, 0.5)
    # augmented_signal = pitch_scale(signal, sr, 24)
    # augmented_signal = randomg_gain(signal, 2, 4)

    sf.write("augmented.wav", augmented_signal, sr)
    # Mostra graficamente o sinal original e com ruído para comparação.
    _plot_signal_and_augmented_signal(signal, augmented_signal, sr)


