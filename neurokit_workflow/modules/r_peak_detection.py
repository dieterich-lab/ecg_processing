import neurokit2 as nk
import numpy as np


def _is_ecg_signal(signal, sampling_rate):
    try:
        # Process the signal using NeuroKit's ECG pipeline
        _, info = nk.ecg_peaks(signal, sampling_rate)
        return len(info['ECG_R_Peaks']) > 0  # True if R-peaks detected
    except Exception as e:
        print(f"Processing error: {e}")
        return False


def clean_ecg(ecg_data, sampling_rate):
    N, num_leads, num_samples = ecg_data.shape

    # Initialize an array for the cleaned ECG signals
    cleaned_ecg_data = np.empty_like(ecg_data)

    # Iterate over each sample and lead
    for j in range(N):
        for i in range(num_leads):
            try:
                # Clean the ECG signal for the current sample and lead
                cleaned_ecg_data[j, i, :] = nk.ecg_clean(ecg_data[j, i, :], sampling_rate=sampling_rate)
            except Exception as e:
                print(f"Error cleaning ECG signal for sample {j}, lead {i}: {e}")
                cleaned_ecg_data[j, i, :] = np.nan  # Replace with NaNs in case of error

    return cleaned_ecg_data


def detect_r_peaks(cleaned_ecg, sampling_rate, channel_seq=None, dataset=None):
    r_peaks = np.empty((cleaned_ecg.shape[1], cleaned_ecg.shape[0]), dtype=object)  # Shape: (12, N)

    for i in range(cleaned_ecg.shape[1]):  # Iterate over each lead (12 leads)
        for j in range(cleaned_ecg.shape[0]):  # Iterate over each sample
            signal = cleaned_ecg[j, i, :]
            try:
                if _is_ecg_signal(signal, sampling_rate):
                    _, rpeaks = nk.ecg_peaks(signal, sampling_rate=sampling_rate)
                    r_peaks[i, j] = rpeaks['ECG_R_Peaks']
                else:
                    raise ValueError(f"Invalid ECG signal in lead {i} for sample {j}")
            except Exception as e:
                print(f"Error processing ECG lead {i} for sample {j}: {e}")

    return r_peaks
