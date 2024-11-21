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

DATASET = 'MIMIC'


def process_ecg_data():
    # Load ECG data from their custom data loader
    if DATASET == 'MEDICONNECT':
        ecg_array, samp_freq, channel_seq = load_mediconnect_data(config['DATASETS']['MEDICONNECT']['DICOM_DIR'])
    elif DATASET == 'UKBIOBANK':
        ecg_array, samp_freq, channel_seq = load_ukbiobank_data(config['DATASETS']['UKBIOBANK']['DIR_PATH'])
    elif DATASET == 'MIMIC':
        ecg_array, samp_freq, channel_seq = load_mimic_data(config['DATASETS']['MIMIC']['DIR_PATH'])
    else:
        raise ValueError('Dataset not supported and no data loader is created')

    # clean the ecgs to remove any noise and baseline wandering
    cleaned_ecg = clean_ecg(ecg_array, sampling_rate=samp_freq)

    # detect the R peaks for every lead of the ecgs separately
    r_peaks = detect_r_peaks(cleaned_ecg, sampling_rate=samp_freq, dataset=DATASET)

    # delineate the cleaned ecg into its components
    delineation_results = delineate_ecg(cleaned_ecg, r_peaks, sampling_rate=samp_freq)

    # extract ecg baseline features with related intervals
    feature_extractor = ECGFeatureExtractor(cleaned_ecg, r_peaks, delineation_results, samp_freq, channel_seq)
    features_df = feature_extractor.extract_features()

    # get the columns with the main baseline features
    features_baseline = features_df[['sample_idx', 'lead', 'heart_rate', 'r_peaks', 'pr_interval', 'qrs_complex',
                                     'qt_interval', 'rr_interval', 'st_segment']]

    # extract the ecg annotations with metadata of features
    annotations_df = feature_extractor.generate_annotations(features_df)
    annotations_df = annotations_df.convert_dtypes()

    # create output directory if it does not already exist
    output_directory = config['OUTPUT_DIRECTORY']
    if not os.path.exists(output_directory):
        os.makedirs(output_directory, exist_ok=True)

    # save the features and their metadata to a CSV file
    features_baseline.to_csv(os.path.join(output_directory, f'{DATASET}_ecg_baseline_features.csv'),
                             float_format='%.3f', index=False)
    annotations_df.to_csv(os.path.join(output_directory, 'annotations.csv'), index=False)


if __name__ == '__main__':
    process_ecg_data()
