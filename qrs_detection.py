import os
import numpy as np
import pandas as pd
import pydicom
import neurokit2 as nk
import matplotlib.pyplot as plt

def read_dicom():
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

    with open('test_files/test_ecg_filenames.txt', 'w') as f:
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


def plot_12_lead_ecg(ecg_all_leads, rpeaks_all_leads, lead_sequence, filename, num_samples, samp_freq):
    fig, axs = plt.subplots(6, 2, figsize=(20, 20))
    plt.subplots_adjust(hspace=0.5)
    ticks = np.linspace(0, num_samples/samp_freq, num_samples)
    for lead, ax in zip(lead_sequence, axs.ravel()):
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude (\u03BCV)')
        idx = lead_sequence.index(lead)
        rpeaks = rpeaks_all_leads[idx]
        ax.plot(ticks, ecg_all_leads[idx], 'b', label='Cleaned ECG Signal')
        ax.plot(ticks[rpeaks], ecg_all_leads[idx][rpeaks], 'o', markersize=6, color='red', label='R-peaks')
        ax.set_title(f'ECG Signal with R-peaks for Lead - {lead.upper()}')
    plt.savefig(f'test_files/processed_files/{filename}_rpeaks.png')


def plot_average_beat(ecg_all_leads, rpeaks_all_leads, lead_sequence, filename, samp_freq):
    fig, axs = plt.subplots(6, 2, figsize=(15, 15))
    plt.subplots_adjust(hspace=0.5)
    for lead, ax in zip(lead_sequence, axs.ravel()):
        idx = lead_sequence.index(lead)
        calculate_average_beat(ecg_all_leads[idx], rpeaks_all_leads[idx], ax, samp_freq)
        ax.set_ylabel(f'Lead {lead} (\u03BCV)')
    plt.savefig(f'test_files/processed_files/{filename}_average_beats.png')


def qrs_detection(file):
    dicom_root = '/prj/acribis/DICOM-Dateien_zu_EKG-Daten/'
    lead_sequence = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    with open(file, 'r', newline='') as ecg_filenames:
        for name in ecg_filenames:
            name = name.rstrip('\n')
            test_path = os.path.join(dicom_root, name)
            print(test_path)
            ds = pydicom.dcmread(test_path)
            dicom_array = ds.waveform_array(0)
            df_ecg = pd.DataFrame(columns=lead_sequence, data=dicom_array)
            wave_seq = ds.WaveformSequence[0]
            num_samples = wave_seq.NumberOfWaveformSamples
            samp_freq = wave_seq.SamplingFrequency
            ecg_all_leads, rpeaks_all_leads, qrs_epochs = [], [], []
            for lead in df_ecg.columns:
                # using default neurokit method for peak detection
                clean_ecg_original = nk.ecg_clean(df_ecg[lead], sampling_rate=samp_freq)
                ecg_fixed, is_inverted = nk.ecg_invert(df_ecg[lead], samp_freq)
                clean_ecg_fixed = nk.ecg_clean(ecg_fixed, sampling_rate=samp_freq)
                rpeaks = neurokit(clean_ecg_fixed, samp_freq)
                ecg_all_leads.append(clean_ecg_original)
                rpeaks_all_leads.append(rpeaks)

            plot_12_lead_ecg(ecg_all_leads, rpeaks_all_leads, lead_sequence, name, num_samples, samp_freq)
            plot_average_beat(ecg_all_leads, rpeaks_all_leads, lead_sequence, name, samp_freq)



qrs_detection('test_files/test_ecg_filenames.txt')


