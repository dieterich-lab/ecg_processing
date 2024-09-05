from ecg_analysis import qrs_detection

TEST_FILES = './neurokit/test_files/test_ecg_filenames.txt'
DICOM_DIRECTORY_PATH = '/prj/acribis/DICOM-Dateien_zu_EKG-Daten/'
DICOM_DIR = '../WP4_Selection/DICOM/'

if __name__ == '__main__':
    qrs_detection(DICOM_DIR)
