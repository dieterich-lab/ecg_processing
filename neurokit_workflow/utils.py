import os
import numpy as np
import pandas as pd
import pydicom
import neurokit2 as nk
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, butter, sosfiltfilt


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
    qrs_epochs = nk.ecg_segment(ecg, rpeaks=rpeaks, sampling_rate=samp_freq, show=True, ax=ax)
    return ax, qrs_epochs


def ecg_delineate(ecg, rpeaks, samp_freq, ax):
    ecg_rate = nk.ecg_rate(rpeaks, sampling_rate=samp_freq)
    ecg_rate_mean = np.mean(ecg_rate)
    print(ecg_rate_mean)
    signals, waves = nk.ecg_delineate(ecg, rpeaks, samp_freq, show=True, show_type='all')
    return ax, waves


def plot_12_lead_ecg(ecg_all_leads, rpeaks_all_leads, num_samples, samp_freq, lead_seq, sample_idx, plot_path):
    fig, axs = plt.subplots(6, 2, figsize=(20, 20))
    plt.subplots_adjust(hspace=0.5)
    ticks = np.linspace(0, num_samples / samp_freq, num_samples)
    for lead, ax in zip(lead_seq, axs.ravel()):
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude (\u03BCV)')
        idx = lead_seq.index(lead)
        rpeaks = rpeaks_all_leads[idx]
        ax.plot(ticks, ecg_all_leads[idx], 'b', label='Cleaned ECG Signal')
        ax.plot(ticks[rpeaks], ecg_all_leads[idx][rpeaks], 'o', markersize=6, color='red', label='R-peaks')
        ax.set_title(f'ECG Signal with R-peaks for Lead - {lead.upper()}')
    # plt.show()
    plot_path = os.path.join(plot_path, f'{sample_idx}_rpeaks_after_invert.png')
    plt.savefig(plot_path)


def plot_average_beat(ecg_all_leads, rpeaks_all_leads, samp_freq, lead_seq, sample_idx):
    fig, axs = plt.subplots(6, 2, figsize=(15, 15))
    plt.subplots_adjust(hspace=0.5)
    for lead, ax in zip(lead_seq, axs.ravel()):
        idx = lead_seq.index(lead)
        try:
            _, qrs_epochs = calculate_average_beat(ecg_all_leads[idx], rpeaks_all_leads[idx], ax, samp_freq)
        except Exception as e:
            print(f'Average beat calculation failed for the ECG lead {lead} with error - {e}')
            continue
        ax.set_ylabel(f'Lead {lead} (\u03BCV)')

        if lead == 'II':
            # convert epochs to dataframe
            df = pd.concat(qrs_epochs)
            df["Time"] = df.index.get_level_values(1).values
            df = df.reset_index(drop=True)
            mean_heartbeat = df.groupby("Time")[['Signal']].mean()

            plt.figure(figsize=(10, 5))
            plt.plot(mean_heartbeat)
            plt.savefig(f'processed_files/mimic_iv/{sample_idx}_{lead}_manual_average_beats.png')
    plt.savefig(f'processed_files/mimic_iv/{sample_idx}_average_beats.png')

    return mean_heartbeat


def plot_delineated_ecg(ecg_all_leads, rpeaks_all_leads, samp_freq, lead_seq, sample_idx, dataset):
    fig, axs = plt.subplots(6, 2, figsize=(15, 15))
    plt.subplots_adjust(hspace=0.5)
    for lead, ax in zip(lead_seq, axs.ravel()):
        if lead == 'I':
            idx = lead_seq.index(lead)
            ax, waves = ecg_delineate(ecg_all_leads[idx], rpeaks_all_leads[idx], samp_freq, ax)
            ax.set_ylabel(f'Lead {lead} (\u03BCV)')
            plt.savefig(f'processed_files/{dataset}/{sample_idx}_delineated_modified.png')
    return waves


def plot_ecg(ecg, rpeaks, filename, samp_freq, num_samples, raw_ecg, all_spike_indices):
    fig, axs = plt.subplots(1, 2, figsize=(30, 10), gridspec_kw={'width_ratios': [1.5, 5]})
    ticks = np.linspace(0, num_samples / samp_freq, num_samples)

    try:
        calculate_average_beat(ecg, rpeaks, axs[0], samp_freq)
    except:
        print(f'Average beat calculation failed for the ECG lead I')

    axs[1].plot(ticks, raw_ecg, label='Raw ECG Lead I')
    axs[1].plot(ticks[all_spike_indices], raw_ecg[all_spike_indices], 'o', markersize=7, color='red',
                label='Pacemaker Spikes')
    for index in all_spike_indices:
        axs[1].annotate('Pacemaker',
                        (ticks[index], raw_ecg[index]),
                        textcoords="offset points",
                        xytext=(0, 10),
                        ha='center',
                        fontsize=12,
                        color='black')
    axs[1].plot(ticks[rpeaks], raw_ecg[rpeaks], 'o', markersize=7, color='red', label='R-peaks')
    for index in rpeaks:
        axs[1].annotate('R-peak',
                        (ticks[index], raw_ecg[index]),
                        textcoords="offset points",
                        xytext=(0, 10),
                        ha='center',
                        fontsize=12,
                        color='black')
    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylabel('Amplitude (\u03BCV)')
    axs[1].set_title(f'ECG Signal with R-peaks for Lead - I')
    plt.tight_layout()
    plt.savefig(f'comparison_ecgs/PNGs/{filename}.png')


def read_dicom(path):
    ds = pydicom.dcmread(path)
    dicom_array = ds.waveform_array(0)
    df_ecg = pd.DataFrame(columns=LEAD_SEQUENCE, data=dicom_array)
    waveform_seq = ds.WaveformSequence[0]
    return df_ecg, waveform_seq


def extract_rpeaks(df_ecg, samp_freq, dataset):
    def is_ecg_signal(signal, sampling_rate):
        try:
            # Process the signal using NeuroKit's ECG pipeline
            _, info = nk.ecg_peaks(signal, sampling_rate)
            return len(info['ECG_R_Peaks']) > 0  # True if R-peaks detected
        except Exception as e:
            print(f"Processing error: {e}")
            return False

    ecg_all_leads, rpeaks_all_leads = [], []
    for lead in df_ecg.columns:
        # using default neurokit method for peak detection
        clean_ecg_original = nk.ecg_clean(df_ecg[lead], sampling_rate=samp_freq)

        if dataset == 'UKBIOBANK_temp' and lead.upper() == 'AVR':
            ecg_fixed, is_inverted = nk.ecg_invert(clean_ecg_original, samp_freq)
        elif dataset == 'MIMIC':
            ecg_fixed, is_inverted = nk.ecg_invert(clean_ecg_original, samp_freq)
        else:
            ecg_fixed = clean_ecg_original

        try:
            rpeaks = neurokit(ecg_fixed, samp_freq)
        except Exception as e:
            print(f"Error processing ECG lead {lead}: {e}")
            rpeaks = []
        try:
            # Check if this lead is a valid ECG signal
            if is_ecg_signal(ecg_fixed, samp_freq):
                ecg_all_leads.append(ecg_fixed)
                rpeaks_all_leads.append(rpeaks)
            else:
                raise ValueError(f"Invalid ECG signal in lead {lead}")

        except Exception as e:
            print(f"Skipping ECG recording due to error: {e}")
            continue

    return ecg_all_leads, rpeaks_all_leads


def extract_rpeaks_1lead(df_ecg, samp_freq, lead, num_samples, filename):
    raw_ecg = df_ecg[lead]

    # Find positive spikes (peaks)
    positive_spikes, properties_p = find_peaks(raw_ecg, height=100, threshold=50, distance=5, width=1.5)

    # Find negative spikes (valleys)
    negative_spikes, properties_n = find_peaks(-raw_ecg, height=100, threshold=50, distance=5, width=1.5)

    # Combine the spike indices
    all_spike_indices = np.sort(np.concatenate((positive_spikes, negative_spikes)))

    raw_ecg_copy = raw_ecg.copy()

    extension = 1
    # Replace spikes with interpolated values
    for index in all_spike_indices:
        start_idx = max(0, index - extension)  # Ensure index does not go below 0
        end_idx = min(len(raw_ecg_copy), index + extension + 1)  # Ensure index does not exceed signal length
        raw_ecg_copy[start_idx:end_idx] = 0
        # if 1 < index < len(raw_ecg_copy) - 2:
        #     raw_ecg_copy[index] = 0  #np.mean([df_ecg[lead][index - 1], df_ecg[lead][index + 1]])

    clean_ecg = nk.ecg_clean(raw_ecg_copy, sampling_rate=samp_freq)
    ecg_fixed, is_inverted = nk.ecg_invert(clean_ecg, samp_freq)
    clean_ecg_fixed = nk.ecg_clean(ecg_fixed, sampling_rate=samp_freq)
    # filtered_ecg = medfilt(clean_ecg_fixed, kernel_size=11)

    try:
        rpeaks = neurokit(clean_ecg_fixed, samp_freq)
    except:
        print(f'Something wrong with the ECG lead {lead}')
        rpeaks = []

    plot_ecg(raw_ecg, rpeaks, filename, samp_freq, num_samples, raw_ecg, all_spike_indices)

    return rpeaks, raw_ecg, all_spike_indices


def qrs_detection_mediconnect(directory_path):
    dicom_files = sorted(os.listdir(directory_path))
    print('Total number of ECG Dicom files', len(dicom_files))
    with open('comparison_ecgs/Locations.txt', 'w') as loc_file:
        for i, filename in enumerate(dicom_files):
            if filename.endswith('.dcm'):
                print(filename)
                loc_file.write(f'{filename}\n')
                filename = filename.rstrip('\n')
                file_path = os.path.join(directory_path, filename)
                ds = pydicom.dcmread(file_path)
                patient_name = str(ds.PatientName)
                if 'test' not in patient_name.lower():
                    df_ecg, waveform_seq = read_dicom(file_path)
                    num_samples = waveform_seq.NumberOfWaveformSamples
                    samp_freq = waveform_seq.SamplingFrequency
                    # ecg_all_leads, rpeak_all_leads = extract_rpeaks(df_ecg, samp_freq)
                    rpeaks, ecg, pacemaker_spikes = extract_rpeaks_1lead(df_ecg, samp_freq, 'I', num_samples, filename)
                    rpeaks_list = ', '.join(map(str, rpeaks))
                    p_spikes_list = ', '.join(map(str, pacemaker_spikes))
                    loc_file.write(f'Pacemaker spikes: {p_spikes_list}\n')
                    loc_file.write(f"QRS: {rpeaks_list}\n")
                else:
                    print('Skipping test ECG files')

            # plot_12_lead_ecg(ecg_all_leads, rpeak_all_leads, filename, num_samples, samp_freq)
            # plot_average_beat(ecg_all_leads, rpeak_all_leads, filename, samp_freq)
            # plot_delineated_ecg(ecg_all_leads, rpeak_all_leads, filename, samp_freq)

            loc_file.write('--------------------------------------------------------------------------------------'
                           '--------------\n')


def filter_ecgs(input_data):
    import antropy as ant

    non_ecgs = []
    template_1 = input_data[8, 1, :]
    template_2 = input_data[47, 1, :]
    fig, axes = plt.subplots(2, 1, figsize=(14, 6))
    axes[0].plot(template_1)
    axes[1].plot(template_2)
    plt.show()
    for i, data in enumerate(input_data):
        signal = data[1]
        samp_entropy = ant.sample_entropy(signal, order=2, metric='chebyshev')
        corr1 = np.correlate(signal - np.mean(signal), template_1 - np.mean(template_1), mode='full')
        corr2 = np.correlate(signal - np.mean(signal), template_2 - np.mean(template_2), mode='full')
        norm_factor1 = len(signal) * np.std(signal) * np.std(template_1)
        norm_factor2 = len(signal) * np.std(signal) * np.std(template_2)
        corr1 = corr1 / norm_factor1
        corr2 = corr2 / norm_factor2

        plt.figure(figsize=(14, 6))
        plt.plot(signal)
        plt.title(f'Signal Index {i}')
        plt.savefig(f'plots/ecg_{i}.png')

        # Define a threshold for correlation and sample entropy
        if (max(corr1) > 0.12 or max(corr2) > 0.12) and (samp_entropy < 0.5):
            continue
        else:
            non_ecgs.append(i)

    return non_ecgs
