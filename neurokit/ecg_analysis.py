import os
import numpy as np
import pandas as pd
import pydicom
import neurokit2 as nk
import matplotlib.pyplot as plt

LEAD_SEQUENCE = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

def get_test_ecgs():
    dicom_root = '/prj/acribis/DICOM-Dateien_zu_EKG-Daten/'
    dicom_list = os.listdir(dicom_root)
    test_files = []

    for filename in dicom_list:
        dicom_path = os.path.join(dicom_root, filename)
        ds = pydicom.dcmread(dicom_path)
        patient_name = str(ds.PatientName)
        if 'test' in patient_name.lower():
            print(patient_name)
            test_files.append(filename)

    with open('neurokit/test_files/test_ecg_filenames.txt', 'w') as f:
        for file in test_files:
            f.write(f"{file}\n")

def neurokit(ecg, sampling_rate):
    signal, info = nk.ecg_peaks(ecg, sampling_rate=sampling_rate, method="neurokit")
    return info["ECG_R_Peaks"]


def kalidas2017(ecg, sampling_rate):
    signal, info = nk.ecg_peaks(ecg, sampling_rate=sampling_rate, method="kalidas2017")
    return info["ECG_R_Peaks"]


def calculate_average_beat(ecg, rpeaks, ax, samp_freq):
    nk.ecg_segment(ecg, rpeaks=rpeaks, sampling_rate=samp_freq, show=True, ax=ax)
    return ax

def ecg_delineate(ecg, rpeaks, samp_freq, ax):
    signals, waves = nk.ecg_delineate(ecg, rpeaks, samp_freq, show=True, show_type='all')
    return ax


def plot_12_lead_ecg(ecg_all_leads, rpeaks_all_leads, filename, num_samples, samp_freq):
    fig, axs = plt.subplots(6, 2, figsize=(20, 20))
    plt.subplots_adjust(hspace=0.5)
    ticks = np.linspace(0, num_samples/samp_freq, num_samples)
    for lead, ax in zip(LEAD_SEQUENCE, axs.ravel()):
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude (\u03BCV)')
        idx = LEAD_SEQUENCE.index(lead)
        rpeaks = rpeaks_all_leads[idx]
        ax.plot(ticks, ecg_all_leads[idx], 'b', label='Cleaned ECG Signal')
        ax.plot(ticks[rpeaks], ecg_all_leads[idx][rpeaks], 'o', markersize=6, color='red', label='R-peaks')
        ax.set_title(f'ECG Signal with R-peaks for Lead - {lead.upper()}')
    plt.show()
    # plt.savefig(f'neurokit/processed_files/ecg_rpeaks/{filename}_rpeaks.png')


def plot_average_beat(ecg_all_leads, rpeaks_all_leads, filename, samp_freq):
    fig, axs = plt.subplots(6, 2, figsize=(15, 15))
    plt.subplots_adjust(hspace=0.5)
    for lead, ax in zip(LEAD_SEQUENCE, axs.ravel()):
        idx = LEAD_SEQUENCE.index(lead)
        try:
            calculate_average_beat(ecg_all_leads[idx], rpeaks_all_leads[idx], ax, samp_freq)
        except:
            print(f'Average beat calculation failed for the ECG lead {lead}')
            continue
        ax.set_ylabel(f'Lead {lead} (\u03BCV)')
    # plt.savefig(f'neurokit/processed_files/ecg_average_beats/{filename}_average_beats.png')


def plot_delineated_ecg(ecg_all_leads, rpeaks_all_leads, filename, samp_freq):
    fig, axs = plt.subplots(6, 2, figsize=(15, 15))
    plt.subplots_adjust(hspace=0.5)
    for lead, ax in zip(LEAD_SEQUENCE, axs.ravel()):
        idx = LEAD_SEQUENCE.index(lead)
        ax = ecg_delineate(ecg_all_leads[idx], rpeaks_all_leads[idx], samp_freq, ax)
        ax.set_ylabel(f'Lead {lead} (\u03BCV)')
    # plt.savefig(f'neurokit/processed_files/ecg_delineated/{filename}_delineated.png')


def read_dicom(path):
    ds = pydicom.dcmread(path)
    dicom_array = ds.waveform_array(0)
    df_ecg = pd.DataFrame(columns=LEAD_SEQUENCE, data=dicom_array)
    waveform_seq = ds.WaveformSequence[0]
    return df_ecg, waveform_seq


def extract_rpeaks(df_ecg, samp_freq):
    ecg_all_leads, rpeaks_all_leads, qrs_epochs = [], [], []
    for lead in df_ecg.columns:
        # using default neurokit method for peak detection
        clean_ecg_original = nk.ecg_clean(df_ecg[lead], sampling_rate=samp_freq)
        ecg_fixed, is_inverted = nk.ecg_invert(df_ecg[lead], samp_freq)
        clean_ecg_fixed = nk.ecg_clean(ecg_fixed, sampling_rate=samp_freq)

        try:
            rpeaks = neurokit(clean_ecg_fixed, samp_freq)
        except:
            print(f'Something wrong with the ECG lead {lead}')
            rpeaks = []

        ecg_all_leads.append(clean_ecg_original)
        rpeaks_all_leads.append(rpeaks)

    return ecg_all_leads, rpeaks_all_leads


def qrs_detection(directory_path):
    # test_files = './neurokit/test_files/test_ecg_filenames.txt'
    # with open(test_files, 'r', newline='') as ecg_filenames:
    dicom_files = os.listdir(directory_path)
    print('Total number of ECG Dicom files', len(dicom_files))
    for i, filename in enumerate(dicom_files):
        if filename.endswith('.dcm'):
            print(filename)
            filename = filename.rstrip('\n')
            file_path = os.path.join(directory_path, filename)
            ds = pydicom.dcmread(file_path)
            patient_name = str(ds.PatientName)
            if 'test' not in patient_name.lower():
                df_ecg, waveform_seq = read_dicom(file_path)
                num_samples = waveform_seq.NumberOfWaveformSamples
                samp_freq = waveform_seq.SamplingFrequency
                ecg_all_leads, rpeak_all_leads = extract_rpeaks(df_ecg, samp_freq)
            else:
                print('Skipping test ECG files')

        plot_12_lead_ecg(ecg_all_leads, rpeak_all_leads, filename, num_samples, samp_freq)
        if i == 5:
            exit()
        # plot_average_beat(ecg_all_leads, rpeak_all_leads, filename, samp_freq)
        # plot_delineated_ecg(ecg_all_leads, rpeak_all_leads, filename, samp_freq)





