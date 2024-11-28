import pandas as pd
import numpy as np
import math
import neurokit2 as nk


class ECGFeatureExtractor:
    def __init__(self, cleaned_ecg, r_peaks, waves, sampling_rate, ecg_channels, channel='II'):
        self.frequency = sampling_rate
        self.channel_seq = ecg_channels
        self.channel = channel
        self.cleaned_ecg = cleaned_ecg
        self.r_peaks = r_peaks.T
        self.waves = waves

    def extract_features(self):
        features_list = []
        channel_1_idx = self.channel_seq.index(self.channel)

        for idx, ecg_sample in enumerate(self.cleaned_ecg):

            try:
                ecg_channel_1 = ecg_sample[channel_1_idx]
                r_peaks_channel_1 = self.r_peaks[idx][channel_1_idx]
                delineated_items = self.waves[idx][channel_1_idx][0]

                ecg_baseline_features = {}

                if delineated_items is None:
                    print(f"Warning: Delineated items are missing for sample {idx}. Skipping.")
                    continue
                    
                ecg_baseline_features = self._extract_on_offsets(ecg_baseline_features, ecg_channel_1,
                                                                 delineated_items)

                ecg_rate = nk.ecg_rate(r_peaks_channel_1, sampling_rate=self.frequency)
                ecg_rate = np.array(ecg_rate)
                rpeak_time = self._get_time_msec(r_peaks_channel_1)
                rr_interval_avg = self._calculate_rr_interval(rpeak_time)
                pr_interval_avg, qrs_complex_avg, qt_interval_avg, st_segment_avg = self._extract_intervals(
                    ecg_baseline_features)

                # Aggregate all extracted features
                ecg_baseline_features.update({
                    'sample_idx': idx,
                    'lead': self.channel,
                    'heart_rate': np.mean(ecg_rate) if len(ecg_rate) > 0 else None,
                    'r_peaks': r_peaks_channel_1,
                    'pr_interval': pr_interval_avg,
                    'qrs_complex': qrs_complex_avg,
                    'qt_interval': qt_interval_avg,
                    'rr_interval': rr_interval_avg,
                    'st_segment': st_segment_avg,
                })

                features_list.append(ecg_baseline_features)

            except Exception as e:
                print(f"Error processing ECG sample {idx} for lead {self.channel}: {e} ")
                continue

        features_df = pd.DataFrame(features_list)
        return features_df

    def generate_annotations(self, df):
        # Initialize list to store annotations
        annotations = []

        df = df.convert_dtypes()

        # Loop through each column in the DataFrame
        for col in df.columns:
            # Get data type
            data_type = str(df[col].dtype)

            # Append dictionary for each column
            annotations.append({
                'Features': col,
                'Description': self._get_description(col),
                'Data Type': data_type,
                'Units': self._get_units(col)
            })

        # Convert to DataFrame
        annotations_df = pd.DataFrame(annotations)
        return annotations_df

    def _extract_on_offsets(self, ecg_baseline_features, ecg_clean, waves):
        for key, val in waves.items():
            if key.endswith('_Peaks'):
                val = [v for v in val if str(v) != 'nan']
                ecg_baseline_features[key[4:].lower()] = np.mean(ecg_clean[val])
            elif key.endswith('_Onsets') or key.endswith('_Offsets'):
                time_msec = self._get_time_msec(waves[key])
                ecg_baseline_features[key[4:].lower()] = time_msec

        return ecg_baseline_features

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

    @staticmethod
    def _get_time_msec(indices):
        time_ms = []
        for idx in indices:
            time_ms.append(idx * 2)

        return time_ms

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
        else:
            words = col.split('_')  # Split by underscore
            desc = ' '.join(word.capitalize() for word in words)

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
