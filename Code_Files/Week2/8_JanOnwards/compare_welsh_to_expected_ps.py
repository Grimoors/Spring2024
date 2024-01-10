import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
from os import path, makedirs

# Global parameters
FS = 200  # Sampling frequency (Hz)
NPERSEG = 1024  # Length of each segment for Welch's method
INPUT_DIR = "./code_input/compare_welsh_to_expected_pspy/"
OUTPUT_DIR = "./code_output/compare_welsh_to_expected_pspy/"

def load_data(filename):
    """
    Load synthetic EEG data from a JSON file.
    :param filename: Path to the JSON file.
    :return: Data loaded from the JSON file.
    """
    print(f"Loading synthetic EEG data from {filename}...")
    with open(filename, 'r') as json_file:
        data = json.load(json_file)
    print("Data loaded successfully.")
    return data

def compute_expected_spectrum(frequencies, amplitudes):
    """
    Computes the expected power spectrum based on frequencies and amplitudes.
    :param frequencies: Array of frequencies.
    :param amplitudes: Array of amplitudes corresponding to each frequency.
    :return: Expected power spectrum array.
    """
    print("Computing expected power spectrum...")
    power_spectrum = np.zeros_like(frequencies)
    for i in range(len(frequencies)):
        power_spectrum[i] = amplitudes[i] ** 2 / 2  # Power is proportional to amplitude squared
    print("Expected power spectrum computation complete.")
    return power_spectrum

def analyze_channel(channel, channel_data):
    """
    Analyze a single EEG channel.
    :param channel: Channel identifier.
    :param channel_data: Signal, frequencies, and amplitudes of the channel.
    """
    print(f"Processing {channel}...")

    signal = np.array(channel_data['Signal'])
    freqs = np.array(channel_data['Frequencies'])
    amps = np.array(channel_data['Amplitudes'])

    '''
        
    Welch's Method:
    ---------------
    Welch's method is used for estimating the power spectral density (PSD) of a signal.
    It involves dividing the signal into overlapping segments, computing a modified periodogram 
    for each segment, and then averaging these periodograms.

    Key Advantages:
    - Reduces noise in the estimated power spectra.
    - Suitable for analyzing non-stationary and/or short-lived signals like EEG.

    Parameters:
    - fs: Sampling frequency of the input signal.
    - nperseg: Length of each segment. Affects the frequency resolution of the PSD estimate.

    The method returns two arrays:
    - Frequencies: Array of sample frequencies.
    - Pxx_den: Power spectral density of the signal for each frequency.
    
    '''
    f, Pxx_den = welch(signal, fs=FS, nperseg=NPERSEG)
    expected_power_spectrum = compute_expected_spectrum(freqs, amps)

    plot_power_spectrum(channel, f, Pxx_den, freqs, expected_power_spectrum)
    plot_and_save_residuals(channel, f, Pxx_den, freqs, expected_power_spectrum)

# def plot_power_spectrum(channel, frequencies, power, expected_freqs, expected_power):
#     """
#     Plot and save the power spectrum for a channel.
#     :param channel: Channel identifier.
#     :param frequencies: Frequencies array from Welch's method.
#     :param power: Power array from Welch's method.
#     :param expected_freqs: Frequencies array of the expected power spectrum.
#     :param expected_power: Power array of the expected power spectrum.
#     """
#     print(f"Plotting power spectrum for {channel}...")
#     plt.figure(figsize=(10, 6))
#     plt.semilogy(frequencies, power, label='Generated Power Spectrum')
#     plt.semilogy(expected_freqs, expected_power,  label='Expected Power Spectrum', color='red')
#     plt.xlabel('Frequency [Hz]')
#     plt.ylabel('Log Power')
#     plt.title(f'Power Spectrum Analysis - {channel}')
#     plt.legend()
#     plt.grid(True)

#     plt.savefig(f'{OUTPUT_DIR}Overlay_Plot_{channel}.png')
#     print(f"Overlay plot saved as {OUTPUT_DIR}Overlay_Plot_{channel}.png.")
#     plt.close()

def plot_power_spectrum(channel, frequencies, power, expected_freqs, expected_power):
    """
    Plot and save the overlaid power spectra for a channel.
    :param channel: Channel identifier.
    :param frequencies: Frequencies array from Welch's method.
    :param power: Power array from Welch's method.
    :param expected_freqs: Expected frequencies array.
    :param expected_power: Expected power array.
    """
    print(f"Plotting overlay power spectrum for {channel}...")

    # Define the full frequency range (0.5 to 60 Hz) and initialize power arrays to zero
    full_freq_range = np.arange(0.5, 60 + FS/NPERSEG, FS/NPERSEG)
    actual_power_full_range = np.zeros_like(full_freq_range)
    expected_power_full_range = np.zeros_like(full_freq_range)

    # Assign actual power values to the corresponding frequencies
    for i, freq in enumerate(frequencies):
        closest_idx = np.argmin(np.abs(full_freq_range - freq))
        actual_power_full_range[closest_idx] = power[i]

    # Assign expected power values to the corresponding frequencies
    for i, freq in enumerate(expected_freqs):
        closest_idx = np.argmin(np.abs(full_freq_range - freq))
        expected_power_full_range[closest_idx] = expected_power[i]

    plt.figure(figsize=(10, 6))
    plt.semilogy(full_freq_range, actual_power_full_range, label='Generated Power Spectrum')
    # plt.semilogy(full_freq_range, expected_power_full_range, 'o', label='Expected Power Spectrum')
    plt.semilogy(full_freq_range, expected_power_full_range, label='Expected Power Spectrum', color='red' )
    
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Log Power')
    plt.title(f'Overlay Power Spectrum Analysis - {channel}')
    plt.legend()
    plt.grid(True)

    plt.savefig(f'{OUTPUT_DIR}Overlay_Plot_{channel}.png')
    print(f"Overlay plot saved as {OUTPUT_DIR}Overlay_Plot_{channel}.png.")
    plt.close()


def plot_and_save_residuals(channel, frequencies, power, expected_freqs, expected_power):
    """
    Plot and save the residuals of the power spectra.
    :param channel: Channel identifier.
    :param frequencies: Frequencies array from Welch's method.
    :param power: Power array from Welch's method.
    :param expected_freqs: Expected frequencies array.
    :param expected_power: Expected power array.
    """
    print(f"Plotting residuals for {channel}...")

    # Define the full frequency range (0.5 to 60 Hz) and initialize power arrays to zero
    full_freq_range = np.arange(0.5, 60 + FS/NPERSEG, FS/NPERSEG)
    actual_power_full_range = np.zeros_like(full_freq_range)
    expected_power_full_range = np.zeros_like(full_freq_range)

    # Assign actual power values to the corresponding frequencies
    for i, freq in enumerate(frequencies):
        closest_idx = np.argmin(np.abs(full_freq_range - freq))
        actual_power_full_range[closest_idx] = power[i]

    # Assign expected power values to the corresponding frequencies
    for i, freq in enumerate(expected_freqs):
        closest_idx = np.argmin(np.abs(full_freq_range - freq))
        expected_power_full_range[closest_idx] = expected_power[i]

    # Calculate residuals
    residuals = expected_power_full_range - actual_power_full_range

    plt.figure(figsize=(10, 6))
    plt.plot(full_freq_range, residuals)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Residual Power')
    plt.title(f'Residuals Plot - {channel}')
    plt.grid(True)

    plt.savefig(f'{OUTPUT_DIR}Residuals_Plot_{channel}.png')
    print(f"Residuals plot saved as {OUTPUT_DIR}Residuals_Plot_{channel}.png.")
    plt.close()

    residuals_filename = f'{OUTPUT_DIR}residuals_{channel}.json'
    with open(residuals_filename, 'w') as file:
        json.dump({'frequencies': full_freq_range.tolist(), 'residuals': residuals.tolist()}, file, indent=4)
    print(f"Residuals data saved as {residuals_filename}.")

def main():
    if not path.exists(OUTPUT_DIR):
        makedirs(OUTPUT_DIR)

    data = load_data(path.join(INPUT_DIR, 'synthetic_eeg_data.json'))

    for channel, channel_data in data.items():
        analyze_channel(channel, channel_data)

    print("All channel analyses completed.")

if __name__ == "__main__":
    main()
