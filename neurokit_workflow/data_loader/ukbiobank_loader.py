import os
import xml.etree.ElementTree as ET
import numpy as np

CHANNELS_SEQ = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']


def _setup(dir_path):
    """Explicitly extract metadata before running other methods."""
    ecg_xml_files = [f for f in os.listdir(dir_path) if '20205' in f]

    if len(ecg_xml_files) == 0:
        raise ValueError("No ECG files found in the directory.")

    ecg_path = os.path.join(dir_path, ecg_xml_files[0])
    tree = ET.parse(ecg_path)
    root = tree.getroot()

    ecg_12_lead_root = root.find('.//StripData')
    samp_freq = int(ecg_12_lead_root.find('SampleRate').text)
    num_samples = int(ecg_12_lead_root.find('ChannelSampleCountTotal').text)
    num_leads = int(ecg_12_lead_root.find('NumberOfLeads').text)

    return num_samples, num_leads, ecg_xml_files, samp_freq


def load_ukbiobank_data(dir_path):
    num_samples, num_leads, ecg_xml_files, samp_freq = _setup(dir_path)

    # Create an empty array to store ECG signals
    signal_array = np.empty((len(ecg_xml_files), num_leads, num_samples), dtype=np.float64)

    # Fill the array with ECG signals
    for idx, xml in enumerate(ecg_xml_files):
        ecg_path = os.path.join(dir_path, xml)

        tree = ET.parse(ecg_path)
        root = tree.getroot()

        ecg_data = {}
        for waveform in root.findall('.//WaveformData')[12:]:
            lead_name = waveform.get('lead')
            lead_values = waveform.text.strip().split(',')
            ecg_data[lead_name] = list(map(int, lead_values))

        ecg_array = np.array(list(ecg_data.values()))
        if ecg_array is not None:
            signal_array[idx] = ecg_array

    return signal_array, samp_freq, CHANNELS_SEQ
