from utils import qrs_detection_mediconnect
from pipeline_mimic import MIMICAnalyzer
from pipeline_ukbiobank import UKBiobankAnalyzer

TEST_FILES = './neurokit/test_files/test_ecg_filenames.txt'
DICOM_DIRECTORY_PATH = '/prj/acribis/DICOM-Dateien_zu_EKG-Daten/'
DICOM_DIR = '../WP4_Selection/DICOM/'
DATASET = 'MIMIC'

if __name__ == '__main__':
    if DATASET == 'MEDICONNECT':
        qrs_detection_mediconnect(DICOM_DIR)
    elif DATASET == 'MIMIC':
        DIR_PATH = '/biodb/mimic-iv/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0'
        mimic_pipeline = MIMICAnalyzer(DIR_PATH)
        mimic_pipeline.preprocess()
    elif DATASET == 'UKBIOBANK':
        DIR_PATH = '/biodb/ukbiobank/bulkdata'
        ukbb_pipeline = UKBiobankAnalyzer(DIR_PATH)
        ukbb_pipeline.preprocess()
