import os
import pydicom
import numpy as np

CHANNELS_SEQ = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']


def read_dicom(path):
    ds = pydicom.dcmread(path)
    dicom_array = ds.waveform_array(0)
    waveform_seq = ds.WaveformSequence[0]
    return dicom_array, waveform_seq


def load_mediconnect_data(dir_path):
    """
    Loads Mediconnect data and converts it to a 3D numpy array of shape (N, 12, samples).
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
            samp_freq = waveform_seq.SamplingFrequency
            ecg_data.append(dicom_array.T)

    ecg_array = np.array(ecg_data)

    return ecg_array, samp_freq, CHANNELS_SEQ
