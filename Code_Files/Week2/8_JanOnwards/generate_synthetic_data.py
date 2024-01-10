import numpy as np
import matplotlib.pyplot as plt
import json
import os

def generate_sine_wave(frequency, amplitude, time_array):
    """
    Generates a sine wave for the given frequency and amplitude.
    :param frequency: The frequency of the sine wave.
    :param amplitude: The amplitude of the sine wave.
    :param time_array: Array of time points.
    :return: Sine wave array.
    """
    return amplitude * np.sin(2 * np.pi * frequency * time_array)

def plot_and_save_wave(signal, title, filename, sample_points=None):
    """
    Plots and saves the signal to a file.
    :param signal: Signal array to be plotted.
    :param title: Title of the plot.
    :param filename: Filename for saving the plot.
    :param sample_points: Time points for sampling, used for sampled signal plots.
    """
    plt.figure(figsize=(10, 4))
    if sample_points is not None:
        plt.plot(sample_points, signal, 'o-')
    else:
        plt.plot(signal)
    plt.title(title)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.savefig(filename)
    plt.close()

def generate_and_plot_channel_data(channel, num_frequencies, time_array, output_path):
    """
    Generates and plots the data for a single EEG channel.
    :param channel: Channel number.
    :param num_frequencies: Number of frequencies to generate.
    :param time_array: Array of time points.
    :param output_path: Directory path to save the plots.
    """
    freqs = np.random.uniform(0.5, 60, size=num_frequencies)
    amps = np.random.uniform(0.1, 5.0, size=num_frequencies)

    combined_signal = np.zeros_like(time_array)
    for f, amp in zip(freqs, amps):
        sine_wave = generate_sine_wave(f, amp, time_array)
        combined_signal += sine_wave

    plot_and_save_wave(combined_signal, f'Combined Signal - Channel {channel}', 
                       f'{output_path}CombinedSignal_Channel{channel}.png')

    # Sample the signal at 200 Hz
    sampling_interval = len(time_array) // (200 * 600)  # Number of time points per sample
    sampled_signal = combined_signal[::sampling_interval]
    sampled_time = time_array[::sampling_interval]
    plot_and_save_wave(sampled_signal, f'Sampled Signal - Channel {channel}', 
                       f'{output_path}SampledSignal_Channel{channel}.png', sampled_time)

    return freqs, amps, combined_signal

def main():
    sampling_rate = 120000  # Original signal's sampling rate (200 samples per second for 600 seconds)
    duration = 600          # seconds
    num_channels = 21       # Total number of EEG channels

    print("Initializing EEG data generation...")
    time_array = np.linspace(0, duration, sampling_rate, endpoint=False)
    output_path = './code_output/gen_syn_datapy/'
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    data = {}
    for channel in range(1, num_channels + 1):
        print(f"Processing Channel {channel}...")
        num_freqs = np.random.randint(12, 1000)
        freqs, amps, signal = generate_and_plot_channel_data(channel, num_freqs, time_array, output_path)
        data[f'Channel_{channel}'] = {'Signal': signal.tolist(), 'Frequencies': freqs.tolist(), 'Amplitudes': amps.tolist()}

    file_path = os.path.join(output_path, 'synthetic_eeg_data.json')
    print(f"Saving EEG data to {file_path}...")
    with open(file_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)
    print("EEG data successfully saved.")

if __name__ == "__main__":
    main()
