import os
import pandas as pd
import numpy as np
import wfdb

from utils import extract_rpeaks, plot_12_lead_ecg

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', None)

CHANNELS_SEQ = ['I', 'II', 'III', 'aVR', 'aVF', 'aVL', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']


class MIMICAnalyzer:
    def __init__(self, directory_path: str):
        self.mimic_path = directory_path
        self.frequency = None
        self.num_leads = None
        self.num_samples = None

    def setup(self):
        df_records = pd.read_csv(os.path.join(self.mimic_path, 'record_list.csv'))
        df_records['path'] = df_records['path'].apply(lambda x: os.path.join(self.mimic_path, x))

        df_records = df_records[:100]

        def read_record(file_path):
            try:
                return wfdb.rdrecord(file_path)
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                return None

        df_records['record'] = df_records['path'].apply(read_record)

        for idx, row in df_records.iterrows():
            ecg_signals = self._extract_signal(row['record'])
            self.frequency = row['record'].fs
            if ecg_signals is not None:
                self.num_leads, self.num_samples = ecg_signals.shape[1], ecg_signals.shape[0]
                break

        if self.num_leads is None or self.num_samples is None:
            raise ValueError("Unable to determine the dimensions of the ECG signals.")

        return df_records

    def extract_ecgs(self):
        df_records = self.setup()

        # Create an empty array to store ECG signals
        signals_array = np.empty((len(df_records), self.num_leads, self.num_samples), dtype=np.float64)

        # Fill the array with ECG signals
        for idx, row in df_records.iterrows():
            ecg_signals = self._extract_signal(row['record'])
            if ecg_signals is not None and ecg_signals.shape == (self.num_samples, self.num_leads):
                signals_array[idx] = np.transpose(ecg_signals)
            else:
                # Handle cases where the signal shape does not match the expected dimensions
                print(f"Record at index {idx} does not have the expected shape or is None.")

        print(signals_array.shape)
        np.save('processed_files/mimic_iv/mimic_ecgs.npy', signals_array)

    @staticmethod
    def _extract_signal(r):
        if r is not None:
            return r.p_signal
        else:
            return None

    def preprocess(self):
        self.setup()
        ecg_array = np.load('processed_files/mimic_iv/mimic_ecgs.npy')

        plot_path = 'processed_files/mimic_iv/'

        def plot_all(ecg_all_leads, rpeak_all_leads, idx):
            plot_12_lead_ecg(ecg_all_leads, rpeak_all_leads, self.num_samples, self.frequency, CHANNELS_SEQ, idx, plot_path)
            # plot_average_beat(ecg_all_leads, rpeak_all_leads, self.frequency, CHANNELS_SEQ, idx)
            # plot_delineated_ecg(ecg_all_leads, rpeak_all_leads, self.frequency, CHANNELS_SEQ, idx)

        for i, ecg in enumerate(ecg_array):
            df = pd.DataFrame(data=ecg.T, columns=CHANNELS_SEQ)
            ecg_all_leads, rpeak_all_leads = extract_rpeaks(df, self.frequency, dataset='MIMIC')

            # Call the helper function to plot all required plots

            plot_all(ecg_all_leads, rpeak_all_leads, i)

            if i == 4:
                break
