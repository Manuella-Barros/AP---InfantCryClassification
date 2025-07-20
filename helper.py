import librosa.display
import matplotlib.pyplot as plt

def _plot_signal_and_augmented_signal(signal, augmented_signal, sr):
    # cria os dois gráficos
    fig, ax = plt.subplots(nrows=2)
    librosa.display.waveshow(signal, sr=sr, ax=ax[0])
    ax[0].set(title="Original signal")
    librosa.display.waveshow(augmented_signal, sr=sr, ax=ax[1])
    ax[1].set(title="Augmented signal")
    plt.show()