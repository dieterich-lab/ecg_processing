import neurokit2 as nk
import numpy as np


def delineate_ecg(cleaned_ecg, r_peaks, sampling_rate, dataset=None):
    """
    Performs ECG delineation for each lead in the cleaned ECG data using NeuroKit.
    cleaned_ecg: A 3D numpy array (N, 12, samples)
    r_peaks: Detected R-peaks, array of shape (12, N) where each entry is a list of R-peak indices
    """
    delineation_results = np.empty_like(cleaned_ecg, dtype=object)  # Shape: (N, 12)
    for i in range(cleaned_ecg.shape[1]):  # Iterate over each lead
        for j in range(cleaned_ecg.shape[0]):  # Iterate over each sample
            ecg_sample = cleaned_ecg[j, i, :]
            rpeaks_sample = r_peaks[i, j]

            if (rpeaks_sample is not None and len(rpeaks_sample) > 3) and ecg_sample is not None:
                signal, waves = nk.ecg_delineate(ecg_sample, r_peaks[i, j], sampling_rate=sampling_rate, method="dwt")
                delineation_results[j, i] = waves

    return delineation_results
