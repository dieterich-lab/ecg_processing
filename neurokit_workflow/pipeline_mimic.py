import math
import os
import pandas as pd
import numpy as np
import wfdb
from matplotlib import pyplot as plt
import neurokit2 as nk

from utils import extract_rpeaks, plot_12_lead_ecg, plot_average_beat, plot_delineated_ecg, filter_ecgs

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
            ecg_signals, _ = self._extract_signal(row['record'])
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
            ecg_signals, _ = self._extract_signal(row['record'])
            if ecg_signals is not None and ecg_signals.shape == (self.num_samples, self.num_leads):
                signals_array[idx] = np.transpose(ecg_signals)
            else:
                # Handle cases where the signal shape does not match the expected dimensions
                print(f"Record at index {idx} does not have the expected shape or is None.")

        print(signals_array.shape)
        np.save('plots/mimic_iv/mimic_ecgs.npy', signals_array)

    def preprocess(self):
        df_records = self.setup()

        ecg_array = np.load('plots/mimic_iv/mimic_ecgs.npy')
        print(ecg_array.shape)

        plot_path = 'plots/mimic_iv/'

        def plot_all(ecgs, rpeaks, idx):
            # plot_12_lead_ecg(ecgs, rpeaks, self.num_samples, self.frequency, CHANNELS_SEQ, idx, plot_path)
            # mean_heartbeat = plot_average_beat(ecgs, rpeaks, self.frequency, CHANNELS_SEQ, idx)
            waves = plot_delineated_ecg(ecgs, rpeaks, self.frequency, CHANNELS_SEQ, idx, 'mimic_iv')

        for i, ecg in enumerate(ecg_array):
            df = pd.DataFrame(data=ecg.T, columns=CHANNELS_SEQ)
            ecg_all_leads, rpeak_all_leads = extract_rpeaks(df, self.frequency, dataset='MIMIC')

            # plot 12-lead ecgs with detected r-peaks
            if len(ecg_all_leads) == 12:
                plot_all(ecg_all_leads, rpeak_all_leads, i)

            if i == 10:
                break

    def extract_features(self):
        df_records = self.setup()
        df_measurements = self._load_measurements()
        df_merged = self._merge_data(df_records, df_measurements)

        print(df_merged.head())

        # Apply process_row to the first 5 rows only and expand into new columns
        df_merged.drop(['rr_interval', 'p_onset', 'qrs_onset', 'p_axis', 'qrs_axis', 't_axis'], axis=1, inplace=True)
        new_features = df_merged[:10].apply(self._process_row, axis=1, result_type='expand')

        # Merge the new columns into df_merged
        df_merged = pd.concat([df_merged, new_features], axis=1)

        print(df_merged.head(7))

        df_annotations = self.generate_annotations(df_merged)
        # df_annotations.to_csv('processed_files/mimic_iv/mimic_iv_ecg_annotations.csv', index=False)

        print(df_annotations)

    def _process_row(self, row):
        ecg_baseline_features = {}
        signal, units = self._extract_signal(row['record'])
        signal_t = np.transpose(signal)
        ecg_lead2 = signal_t[1]  # Directly accessing Lead II (assuming it's the second row)

        ecg_clean, rpeaks, waves = self._analyze_ecg(ecg_lead2, row)

        self._extract_on_offsets(ecg_baseline_features, ecg_clean, waves)

        ecg_rate = nk.ecg_rate(rpeaks, sampling_rate=self.frequency)

        rpeak_time = self._get_time_msec(rpeaks['ECG_R_Peaks'])
        rr_interval_avg = self._calculate_rr_interval(rpeak_time)

        pr_interval_avg, qrs_complex_avg, qt_interval_avg, st_segment_avg = self._extract_intervals(
            ecg_baseline_features)

        ecg_baseline_features['heart_rate'] = np.mean(ecg_rate)
        ecg_baseline_features['pr_interval'] = pr_interval_avg
        ecg_baseline_features['qrs_complex'] = qrs_complex_avg
        ecg_baseline_features['qt_interval'] = qt_interval_avg
        ecg_baseline_features['rr_interval'] = rr_interval_avg
        ecg_baseline_features['st_segment'] = st_segment_avg

        return ecg_baseline_features

    def _extract_on_offsets(self, ecg_baseline_features, ecg_clean, waves):
        for key, val in waves.items():
            if key.endswith('_Peaks'):
                val = [v for v in val if str(v) != 'nan']
                ecg_baseline_features[key[4:].lower()] = np.mean(ecg_clean[val])
            elif key.endswith('_Onsets') or key.endswith('_Offsets'):
                time_msec = self._get_time_msec(waves[key])
                ecg_baseline_features[key[4:].lower()] = time_msec

    def _extract_intervals(self, ecg_baseline_features):
        pr_interval_avg = self._calculate_interval(
            ecg_baseline_features['p_onsets'], ecg_baseline_features['p_offsets']
        )
        qrs_complex_avg = self._calculate_interval(
            ecg_baseline_features['r_onsets'], ecg_baseline_features['r_offsets']
        )
        qt_interval_avg = self._calculate_interval(
            ecg_baseline_features['r_onsets'], ecg_baseline_features['t_offsets']
        )
        st_segment_avg = self._calculate_interval(
            ecg_baseline_features['r_offsets'], ecg_baseline_features['t_onsets']
        )
        return pr_interval_avg, qrs_complex_avg, qt_interval_avg, st_segment_avg

    def _load_measurements(self):
        df_measurements = pd.read_csv(os.path.join(self.mimic_path, 'machine_measurements.csv'))
        return df_measurements[
            ['subject_id', 'study_id', 'rr_interval', 'p_onset', 'qrs_onset', 'p_axis', 'qrs_axis', 't_axis']]

    def _analyze_ecg(self, ecg_lead2, row):
        ecg_clean = nk.ecg_clean(ecg_lead2, sampling_rate=self.frequency)
        _, rpeaks = nk.ecg_peaks(ecg_clean, sampling_rate=self.frequency)
        signals, waves = nk.ecg_delineate(ecg_clean, rpeaks, sampling_rate=self.frequency, show=True, show_type='all')

        # Optionally save the delineation plot
        plt.savefig(f'processed_files/mimic_iv/delineated_ecg_dwt_{row.name}_modified.png')

        return ecg_clean, rpeaks, waves

    def generate_annotations(self, df):
        # Initialize list to store annotations
        annotations = []

        df = df.convert_dtypes()
        print(df.info())

        # Loop through each column in the DataFrame
        for col in df.columns:
            # Get data type
            data_type = str(df[col].dtype)

            # Append dictionary for each column
            annotations.append({
                'Column Name': col,
                'Description': self._get_description(col),
                'Data Type': data_type,
                'Units': self._get_units(col)
            })

        # Convert to DataFrame
        annotations_df = pd.DataFrame(annotations)
        return annotations_df

    @staticmethod
    def _get_description(col):
        if col.endswith('_onsets') or col.endswith('_offsets'):
            desc = f'Time at the {col[2:-1]} of the {col[0].upper()}-wave'
        elif col.endswith('_peaks'):
            desc = f'Amplitude of the {col[0].upper()}-wave'
        elif col == 'pr_interval':
            desc = 'Time between onset of P-wave to onset of R-wave'
        elif col == 'qrs_complex':
            desc = 'Time between onset of Q-wave to offset of S-wave'
        elif col == 'qt_interval':
            desc = 'Time between onset of Q-wave to offset of T-wave'
        elif col == 'st_segment':
            desc = 'Time between offset of S-wave to onset of T-wave'
        elif col == 'rr_interval':
            desc = 'Time between successive R-waves'
        elif col == 'heart_rate':
            desc = 'Number of contractions of the heart per minute'
        elif col == 'record':
            desc = 'WFDB Record object'
        else:
            desc = f'{col[:-3].title()} {col[-2:].upper()}'

        return desc

    @staticmethod
    def _get_units(col):
        if col.endswith('_onsets') or col.endswith('_offsets'):
            unit = 'msec'
        elif col.endswith('_peaks'):
            unit = 'mV'
        elif col.endswith('_segment') or col.endswith('_complex') or col.endswith('_interval'):
            unit = 'msec'
        elif col == 'heart_rate':
            unit = 'bpm'
        else:
            unit = ''

        return unit

    @staticmethod
    def _merge_data(df_records, df_measurements):
        df_merged = pd.merge(df_records, df_measurements, on=['subject_id', 'study_id'], how='left')
        df_merged.drop(['file_name', 'ecg_time', 'path'], inplace=True, axis=1)
        return df_merged

    @staticmethod
    def _get_time_msec(p_onset_idx):
        p_onset_time = []
        for idx in p_onset_idx:
            p_onset_time.append(idx * 2)

        return p_onset_time

    @staticmethod
    def _extract_signal(r):
        if r is not None:
            return r.p_signal, r.units
        else:
            return None

    @staticmethod
    def _calculate_interval(onset, offset):
        if isinstance(onset, list) and isinstance(offset, list):
            interval = []
            for a, b in zip(onset, offset):
                if math.isnan(a) or math.isnan(b):  # Check for NaN in either value
                    continue
                else:
                    interval.append(b - a)  # Calculate the interval
        else:
            # Handle single values, assuming they are not lists
            if math.isnan(onset) or math.isnan(offset):
                interval = float('nan')
            else:
                interval = offset - onset

        return np.mean(interval)

    @staticmethod
    def _calculate_rr_interval(rpeak_time):
        # Calculate RR intervals (difference between consecutive R peak times)
        rr_intervals = np.diff(rpeak_time)

        # Calculate the average RR interval
        average_rr = np.mean(rr_intervals)

        return average_rr
