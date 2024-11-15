import json
import os
import pandas as pd
from data_loader.mediconnect_loader import load_mediconnect_data
from data_loader.ukbiobank_loader import load_ukbiobank_data
from data_loader.mimic_loader import load_mimic_data
from modules.r_peak_detection import clean_ecg, detect_r_peaks
from modules.delineation import delineate_ecg
from modules.feature_extraction import ECGFeatureExtractor

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', None)

# Load configuration
with open('config.json', 'r') as config_file:
    config = json.load(config_file)

DATASET = 'UKBIOBANK'


def process_ecg_data():
    # Step 1: Load ECG data
    if DATASET == 'MEDICONNECT':
        ecg_array, samp_freq, channel_seq = load_mediconnect_data(config['DATASETS']['MEDICONNECT']['DICOM_DIR'])
    elif DATASET == 'UKBIOBANK':
        ecg_array, samp_freq, channel_seq = load_ukbiobank_data(config['DATASETS']['UKBIOBANK']['DIR_PATH'])
    elif DATASET == 'MIMIC':
        ecg_array, samp_freq, channel_seq = load_mimic_data(config['DATASETS']['MIMIC']['DIR_PATH'])

    # clean the ecgs to remove any noise and baseline wandering
    cleaned_ecg = clean_ecg(ecg_array, sampling_rate=samp_freq)

    # detect the R peaks for every lead of the ecgs separately
    r_peaks = detect_r_peaks(cleaned_ecg, sampling_rate=samp_freq)

    # delineate the cleaned ecg into its components
    delineation_results = delineate_ecg(cleaned_ecg, r_peaks, sampling_rate=samp_freq)

    feature_extractor = ECGFeatureExtractor(cleaned_ecg, r_peaks, delineation_results, samp_freq, channel_seq)
    features_df = feature_extractor.extract_features()

    annotations_df = feature_extractor.generate_annotations(features_df)
    annotations_df = annotations_df.convert_dtypes()

    annotations_df.to_csv(os.path.join(config['OUTPUT_DIRECTORY'], 'annotations.csv'))


if __name__ == '__main__':
    process_ecg_data()
