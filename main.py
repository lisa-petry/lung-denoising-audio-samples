#!/usr/bin/env -S uv run
# /// script
# dependencies = [
#   "numpy",
#   "matplotlib",
#   "scipy",
# ]
# ///

import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal

def create_audio_visualization(wav_file, output_image='audio_viz.png'):
    """
    Generate waveform and spectrogram visualization from a WAV file.

    Args:
        wav_file: Path to the input WAV file
        output_image: Path to save the output visualization image
    """
    # Read the WAV file
    sample_rate, audio_data = wavfile.read(wav_file)
    # If stereo, convert to mono by averaging channels
    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=1)

    # Normalize audio data
    audio_data = audio_data / np.max(np.abs(audio_data))

    # Create time array
    duration = len(audio_data) / sample_rate
    time = np.linspace(0, duration, len(audio_data))

    # Create figure with two subplots - no spacing
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Remove all spacing
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0, hspace=0, wspace=0)

    # Plot waveform - minimal style
    ax1.plot(time, audio_data, linewidth=0.5, color='#2196F3')
    ax1.set_xlim([0, duration])
    ax1.set_ylim([-1, 1])
    ax1.axis('off')  # Remove all axes

    # Compute and plot spectrogram
    # frequencies, times, spectrogram = signal.spectrogram(
    #     audio_data,
    #     sample_rate,
    #     nperseg=1024,
    #     noverlap=512
    # )

    # Plot spectrogram - minimal style
    # pcm = ax2.pcolormesh(times, frequencies, 10 * np.log10(spectrogram + 1e-10),
    #                     shading='gouraud', cmap='viridis')
    #
    ax2.specgram(audio_data, Fs=sample_rate, scale='dB', vmin=-60, vmax=-10)
    ax2.set_xlim([0, duration])
    ax2.set_ylim([0, 1000])
    ax2.axis('off')  # Remove all axes

    # Save with no padding
    plt.savefig(output_image, dpi=500, bbox_inches='tight', pad_inches=0)
    print(f"Visualization saved to: {output_image}")

    return duration

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python audio_visualizer.py <wav_file> [output_image]")
        sys.exit(1)

    wav_file = sys.argv[1]
    output_image = sys.argv[2] if len(sys.argv) > 2 else 'audio_viz.png'

    duration = create_audio_visualization(wav_file, output_image)
    print(f"Audio duration: {duration:.2f} seconds")
