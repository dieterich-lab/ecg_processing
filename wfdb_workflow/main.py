from dicom_plotter import DICOMPlotter


DICOM_DIRECTORY_PATH = 'T:/ext01/MED3-ACRIBIS/DICOM-Dateien_zu_EKG-Daten/'
FILE_NAME = '1.3.6.1.4.1.20029.40.20191202081016.17.1.1.dcm'

if __name__ == '__main__':
    DICOMPlotter(DICOM_DIRECTORY_PATH, FILE_NAME)
