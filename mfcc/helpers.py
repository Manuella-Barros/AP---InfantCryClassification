import matplotlib.pyplot as plt
import librosa

def create_mfcc_spectogram(mfccs, delta_mfccs, delta2_mfccs, sample_rate, mfccs_concateneted):
    plt.figure(figsize=(20, 10))

    plt.subplot(4, 2, 1)
    librosa.display.specshow(mfccs, sr=sample_rate, x_axis='time')
    plt.colorbar(format='%+2f') 
    plt.title('MFCCs')

    plt.subplot(4, 2, 2)
    librosa.display.specshow(delta_mfccs, sr=sample_rate, x_axis='time')
    plt.colorbar(format='%+2f') 
    plt.title('Delta MFCCs')

    plt.subplot(4, 2, 3)
    librosa.display.specshow(delta2_mfccs, sr=sample_rate, x_axis='time')
    plt.colorbar(format='%+2f') 
    plt.title('Delta-Delta MFCCs')

    plt.subplot(4, 2, 4)
    librosa.display.specshow(mfccs_concateneted, sr=sample_rate, x_axis='time')
    plt.colorbar(format='%+2f') 
    plt.title('All concateneted')

    plt.tight_layout()

def show_mfcc_spectogram(mfccs, delta_mfccs, delta2_mfccs, sample_rate, mfccs_concateneted):
    create_mfcc_spectogram(mfccs, delta_mfccs, delta2_mfccs, sample_rate, mfccs_concateneted)
    plt.show()

def save_file_mfcc_spectogram(mfccs, delta_mfccs, delta2_mfccs, sample_rate, file_name, mfccs_concateneted):
    create_mfcc_spectogram(mfccs, delta_mfccs, delta2_mfccs, sample_rate, mfccs_concateneted)
    plt.savefig("mfcc/" + file_name)
    plt.close()