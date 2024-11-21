import os
import wfdb
import pandas as pd
import numpy as np

CHANNELS_SEQ = ['I', 'II', 'III', 'aVR', 'aVF', 'aVL', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']


def _setup(dir_path):
    df_records = pd.read_csv(os.path.join(dir_path, 'record_list.csv'))
    df_records = df_records[:100]
    df_records['path'] = df_records['path'].apply(lambda x: os.path.join(dir_path, x))
    df_records['record'] = df_records['path'].apply(_read_record)

    for idx, row in df_records.iterrows():
        ecg_signals = _extract_signal(row['record'])
        samp_freq = row['record'].fs
        if ecg_signals is not None:
            num_leads, num_samples = ecg_signals.shape[1], ecg_signals.shape[0]
            break

    return df_records, samp_freq, num_leads, num_samples


def _read_record(file_path):
    try:
        return wfdb.rdrecord(file_path)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None


def _extract_signal(r):
    if r is not None:
        return r.p_signal
    else:
        return None


def load_mimic_data(dir_path):
    df_records, samp_freq, num_leads, num_samples = _setup(dir_path)

    # Create an empty array to store ECG signals
    signals_array = np.empty((len(df_records), num_leads, num_samples), dtype=np.float64)

    # Fill the array with ECG signals
    for idx, row in df_records.iterrows():
        ecg_signals = _extract_signal(row['record'])
        if ecg_signals is not None and ecg_signals.shape == (num_samples, num_leads):
            signals_array[idx] = np.transpose(ecg_signals)
        else:
            # Handle cases where the signal shape does not match the expected dimensions
            print(f"Record at index {idx} does not have the expected shape or is None.")

    return signals_array, samp_freq, CHANNELS_SEQ



