import os
import pydicom
import numpy as np
from scipy.signal import find_peaks

CHANNELS_SEQ = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']


def read_dicom(path):
    ds = pydicom.dcmread(path)
    dicom_array = ds.waveform_array(0)
    waveform_seq = ds.WaveformSequence[0]
    return dicom_array, waveform_seq


def remove_pacemaker_spikes(dicom_array):
    ecg_12_lead = []
    for lead_arr in dicom_array:

        # Find positive spikes (peaks)
        positive_spikes, properties_p = find_peaks(lead_arr, height=100, threshold=50, distance=5, width=1.5)

        # Find negative spikes (valleys)
        negative_spikes, properties_n = find_peaks(-lead_arr, height=100, threshold=50, distance=5, width=1.5)

        # Combine the spike indices
        all_spike_indices = np.sort(np.concatenate((positive_spikes, negative_spikes)))

        extension = 1
        # Replace spikes with interpolated values
        for index in all_spike_indices:
            start_idx = max(0, index - extension)  # Ensure index does not go below 0
            end_idx = min(len(lead_arr), index + extension + 1)  # Ensure index does not exceed signal length
            lead_arr[start_idx:end_idx] = 0

        ecg_12_lead.append(lead_arr)

    return np.array(ecg_12_lead)


def load_mediconnect_data(dir_path):
    """
    Loads Mediconnect ECGs and converts it to a 3D numpy array of shape (N, 12, samples).
    N: Number of ECG samples (e.g., patients or records)
    12: Number of ECG leads
    samples: Number of samples in each ECG signal
    """
    dicom_files = sorted(os.listdir(dir_path))
    ecg_data = []

    samp_freq = None
    for i, filename in enumerate(dicom_files):
        if filename.endswith('.dcm'):
            filename = filename.rstrip('\n')
            file_path = os.path.join(dir_path, filename)
            dicom_array, waveform_seq = read_dicom(file_path)
            remove_pacemaker_spikes(dicom_array)
            samp_freq = waveform_seq.SamplingFrequency
            ecg_data.append(dicom_array.T)

    ecg_array = np.array(ecg_data)

    return ecg_array, samp_freq, CHANNELS_SEQ
