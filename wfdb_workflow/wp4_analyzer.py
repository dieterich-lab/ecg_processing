import numpy as np
import numpy.typing as npt
from scipy.signal import butter, sosfiltfilt
from wfdb import processing


class WP4Analyzer:
    def __init__(self):
        self.channel_i = None
        self.current_channel = None
        self.filtered = None
        self.cleaned = None
        self.frequency = None
        self.pacemaker_spikes = []
        self.r_peaks_corrected = None
        self.threshold_param = None
        self.hp_filtered_threshold = None
        self.beats = None
        self.average_beat = None

    def __hp_filter_signal(self):
        sos = butter(10, 150, 'high', False, 'sos', self.frequency)
        filtered: npt.NDArray = sosfiltfilt(sos, self.channel_i)
        self.filtered = filtered - np.mean(filtered)

    def __detect_pacemaker_spikes(self):
        self.hp_filtered_threshold = self.filtered.std() * (5 / self.threshold_param)
        pm_candidates1 = [idx for idx, value in enumerate(self.filtered.tolist()) if
                          abs(value) > self.hp_filtered_threshold]
        pm_candidates2 = set()

        window_radius = 10
        for j in range(4):
            for i in pm_candidates1:
                start = max(0, i - window_radius - 5)
                signal_window = self.channel_i[start:min(i + window_radius + 6, len(self.channel_i))]
                signal_window = [abs(v) for v in signal_window]
                pm_candidates2.add(start + np.argmax(signal_window))
            pm_candidates1 = list(pm_candidates2)
            if j < 3:
                pm_candidates2.clear()

        for i in pm_candidates2:
            start = max(0, i - window_radius)
            signal_window = self.channel_i[start:min(i + window_radius + 1, len(self.channel_i))]
            if self.channel_i[i] > 0:
                threshold = self.channel_i.mean() + 2 * self.channel_i.std()
                crossed = [value > threshold for value in signal_window]
            else:
                threshold = self.channel_i.mean() - 2 * self.channel_i.std()
                crossed = [value < threshold for value in signal_window]
            if 0 < sum(crossed) <= 10 + self.threshold_param:
                change = 0
                current = False
                for c in crossed:
                    if c != current:
                        current = c
                        change += 1
                if change < 3:
                    self.pacemaker_spikes.append(i)

    def __clean_signal(self):
        self.cleaned = self.channel_i.copy()
        for pm in self.pacemaker_spikes:
            start = max(0, pm - 20)
            end = min(pm + 20, len(self.channel_i))
            self.cleaned[start:end] = [0] * (end - start)

    def __wfdb_detect_qrs(self):
        xqrs = processing.XQRS(self.cleaned, self.frequency)
        xqrs.detect(verbose=False)
        self.r_peaks_corrected = processing.correct_peaks(self.cleaned, xqrs.qrs_inds, int(self.frequency) // 40,
                                                          int(self.frequency) // 10,
                                                          'compare')

    def __remove_outliers(self):
        self.beats = []
        rr = [(self.r_peaks_corrected[i + 1] - r) for i, r in enumerate(self.r_peaks_corrected)
              if i < len(self.r_peaks_corrected) - 2]
        window_radius = int(np.median(rr) / 2)
        for r_peak in self.r_peaks_corrected:
            start = max(0, r_peak - window_radius)
            end = min(r_peak + window_radius, len(self.current_channel))
            signal_window = self.current_channel[start:end]
            signal_window = signal_window - signal_window.mean()
            if r_peak - window_radius < 0:
                signal_window = np.insert(signal_window, 0, [np.nan] * abs(r_peak - window_radius))
            if r_peak + window_radius > len(self.current_channel):
                signal_window = np.append(signal_window,
                                          [np.nan] * (r_peak + window_radius - len(self.current_channel)))
            self.beats.append(signal_window)
        stacked = np.stack(self.beats)
        median = np.nanmedian(stacked, 0)
        sums = [np.nansum(np.square(w - median)) for w in self.beats]
        remove_threshold = np.quantile(sums, 0.7)
        self.beats = [v for i, v in enumerate(self.beats) if sums[i] <= remove_threshold]

    def __calc_average_beat(self):
        stacked = np.stack(self.beats)
        dividend = np.nansum(stacked, 0)
        divisor = np.full(np.size(stacked, 1), np.size(stacked, 0), np.float64) - np.sum(np.isnan(stacked), 0)
        for i, value in enumerate(divisor):
            if value == 0:
                divisor[i] = 1
        self.average_beat = np.divide(dividend, divisor)

    def update(self):
        self.pacemaker_spikes.clear()
        self.__hp_filter_signal()
        self.__detect_pacemaker_spikes()
        self.__clean_signal()
        self.__wfdb_detect_qrs()
        self.__remove_outliers()
        self.__calc_average_beat()
