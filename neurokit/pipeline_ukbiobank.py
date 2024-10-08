import os

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET

from utils import extract_rpeaks, plot_12_lead_ecg, plot_average_beat, plot_delineated_ecg

CHANNELS_SEQ = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']


class UKBiobankAnalyzer:
    def __init__(self, directory_path: str):
        self.ukbiobank_path = directory_path
        self.frequency = None
        self.num_leads = None
        self.num_samples = None

    def setup(self):
        """Explicitly extract metadata before running other methods."""
        ecg_xml_files = [f for f in os.listdir(self.ukbiobank_path) if '20205' in f]

        if len(ecg_xml_files) == 0:
            raise ValueError("No ECG files found in the directory.")

        ecg_path = os.path.join(self.ukbiobank_path, ecg_xml_files[0])
        tree = ET.parse(ecg_path)
        root = tree.getroot()

        ecg_12_lead_root = root.find('.//StripData')
        self.frequency = int(ecg_12_lead_root.find('SampleRate').text)
        self.num_samples = int(ecg_12_lead_root.find('ChannelSampleCountTotal').text)
        self.num_leads = int(ecg_12_lead_root.find('NumberOfLeads').text)

        return ecg_xml_files

    def extract_ecgs(self):
        ecg_xml_files = self.setup()

        signal_array = np.empty((len(ecg_xml_files), self.num_leads, self.num_samples), dtype=np.float64)

        for idx, xml in enumerate(ecg_xml_files):
            ecg_path = os.path.join(self.ukbiobank_path, xml)

            # Parse the XML file
            tree = ET.parse(ecg_path)
            root = tree.getroot()

            ecg_data = {}
            for waveform in root.findall('.//WaveformData')[12:]:
                lead_name = waveform.get('lead')
                lead_values = waveform.text.strip().split(',')
                ecg_data[lead_name] = list(map(int, lead_values))

            # Convert to Pandas DataFrame
            df = pd.DataFrame(ecg_data)
            ecg_array = df.to_numpy().transpose()

            if ecg_array is not None:
                signal_array[idx] = ecg_array

        np.save('processed_files/uk_biobank/ukbiobank_ecgs.npy', signal_array)

    def preprocess(self):
        self.setup()
        ecg_array = np.load('processed_files/uk_biobank/ukbiobank_ecgs.npy')

        # Helper function to handle all plotting
        def plot_all(ecg_all_leads, rpeak_all_leads, idx):
            plot_12_lead_ecg(ecg_all_leads, rpeak_all_leads, self.num_samples, self.frequency, CHANNELS_SEQ, idx)
            plot_average_beat(ecg_all_leads, rpeak_all_leads, self.frequency, CHANNELS_SEQ, idx)
            plot_delineated_ecg(ecg_all_leads, rpeak_all_leads, self.frequency, CHANNELS_SEQ, idx)

        for i, ecg in enumerate(ecg_array):
            df = pd.DataFrame(data=ecg.T, columns=CHANNELS_SEQ)
            ecg_all_leads, rpeak_all_leads = extract_rpeaks(df, self.frequency, dataset='UKBIOBANK')

            # Call the helper function to plot all required plots
            plot_all(ecg_all_leads, rpeak_all_leads, i)

            if i == 4:
                break
