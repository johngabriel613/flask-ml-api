import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import io
import base64

# Switch backend to avoid GUI warning in server environment
plt.switch_backend('Agg')

# Apply dark mode settings
plt.style.use('dark_background')

def plot_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight', transparent=True)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    plt.close(fig)
    return img_base64

def get_audio_visualizations(audio_path):
    # Load the audio file
    audio_sample, sr = librosa.load(audio_path, sr=22050)

    # Define n_fft and hop_length once for reuse
    n_fft = 2048
    hop_length = 512

    # Create figures and convert to base64
    images = {}

    # Waveform
    fig_waveform = plt.figure(figsize=(20, 5))
    librosa.display.waveshow(audio_sample, sr=sr, color='cyan', alpha=0.7)
    plt.xlabel("Time", color='white')
    plt.ylabel("Amplitude", color='white')
    plt.grid(color='gray', linestyle='--', linewidth=0.5)
    images['waveform'] = plot_to_base64(fig_waveform)

    # Audio Spectrum
    fft_normal = np.fft.fft(audio_sample)
    magnitude_normal = np.abs(fft_normal)
    freq_normal = np.linspace(0, sr, len(magnitude_normal))
    half_freq = freq_normal[:int(len(freq_normal) / 2)]
    half_magnitude = magnitude_normal[:int(len(freq_normal) / 2)]
    fig_spectrum = plt.figure(figsize=(12, 8))
    plt.plot(half_freq, half_magnitude, color='lime', alpha=0.7)
    plt.title("Spectrum", color='white')
    plt.xlabel("Frequency", color='white')
    plt.ylabel("Magnitude", color='white')
    plt.grid(color='gray', linestyle='--', linewidth=0.5)
    images['spectrum'] = plot_to_base64(fig_spectrum)

    # Spectrogram
    stft_normal = librosa.stft(audio_sample, n_fft=n_fft, hop_length=hop_length)
    spectrogram = np.abs(stft_normal)
    log_spectrogram = librosa.amplitude_to_db(spectrogram)
    fig_spectrogram = plt.figure(figsize=(15, 10))
    librosa.display.specshow(log_spectrogram, sr=sr, hop_length=hop_length, x_axis='time', y_axis='log', cmap='magma')
    plt.colorbar(format='%+2.0f dB')
    plt.xlabel("Time", color='white')
    plt.ylabel("Frequency", color='white')
    plt.title("Spectrogram", color='white')
    images['spectrogram'] = plot_to_base64(fig_spectrogram)

    # MFCC
    MFCCs = librosa.feature.mfcc(y=audio_sample, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mfcc=52)
    fig_mfcc = plt.figure(figsize=(20, 5))
    librosa.display.specshow(MFCCs, sr=sr, hop_length=hop_length, x_axis='time', cmap='viridis')
    plt.colorbar(format='%+2.0f dB')
    plt.xlabel("Time", color='white')
    plt.ylabel("MFCC Coefficients", color='white')
    plt.title("MFCC", color='white')
    images['mfcc'] = plot_to_base64(fig_mfcc)

    return images
