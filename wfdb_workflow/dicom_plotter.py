import os

import pydicom
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backend_bases import Event, MouseEvent, KeyEvent
import numpy.typing as npt
from typing import Any, cast
from datetime import datetime
from wp4_analyzer import WP4Analyzer
from plot_controller_widget import PlotControllerWidget

CSV_LOOKUP_PATH = 'Resources/Parameters.csv'
CHANNEL_NAMES = {
    "I": 0,
    "II": 1,
    "III": 2,
    "aVR": 3,
    "aVL": 4,
    "aVF": 5,
    "V1": 6,
    "V2": 7,
    "V3": 8,
    "V4": 9,
    "V5": 10,
    "V6": 11
}


class DICOMPlotter:
    def __init__(self, directory_path: str, filename: str = None, channel: str = 'I'):
        self.controller = PlotControllerWidget(self.update)
        self.ds = None
        self.analyzer = WP4Analyzer()
        self.dicom_channel = channel
        self.directory_path = directory_path
        directory = os.fsencode(directory_path)
        self.dicom_files = [os.fsdecode(file) for file in os.listdir(directory) if os.fsdecode(file).endswith('.dcm')]
        if filename is not None:
            self.filename = filename
            self.dir_index = self.dicom_files.index(self.filename)
        else:
            self.dir_index = 0
            self.filename = self.dicom_files[self.dir_index]
        self.plot_rep_beat = False
        self.parameters = self.read_parameters()
        self.fig, (self.ax_rep_beat, self.ax_rhythm) = plt.subplots(1, 2, width_ratios=[1, 3], figsize=(15, 5))
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)
        self.fig.canvas.mpl_connect('key_press_event', self.on_press)
        self.fig.canvas.mpl_connect('close_event', self.on_close)
        print(f"Filename: {self.filename} {self.dir_index}"
              f"\nNumber of DICOM files in directory: {len(self.dicom_files)}")
        self.update()
        plt.figure(self.controller.fig)
        plt.show()

    def update(self, new_file: bool = True):
        self.ax_rep_beat.clear()
        self.ax_rhythm.clear()
        if new_file:
            self.ds = pydicom.filereader.dcmread(self.directory_path + self.filename)
        self.fig.suptitle(f"DICOM Data\n Channel {self.dicom_channel}")
        self.update_analyzer()
        self.plot_representative_beat()
        self.plot_rhythm()
        self.fig.canvas.draw_idle()

    def update_analyzer(self):
        dicom_data: npt.NDArray[(Any,), np.float64] = self.ds.waveform_array(0)
        self.analyzer.channel_i = dicom_data[:, 0]
        self.analyzer.current_channel = dicom_data[:, CHANNEL_NAMES[self.dicom_channel]]
        self.analyzer.frequency = self.ds.WaveformSequence[0].SamplingFrequency
        self.analyzer.threshold_param = self.controller.slider_value
        self.analyzer.update()

    @staticmethod
    def read_parameters() -> list:
        parameters = []
        with open(CSV_LOOKUP_PATH, newline='') as file:
            reader = csv.DictReader(file, delimiter=',')

            for line in reader:
                parameters.append(line)
        return parameters

    @staticmethod
    def get_points(annotation_sequenz: pydicom.Sequence, parameter: dict[str, str]) -> dict[int, int]:
        points = {}
        for element in annotation_sequenz:
            if "ConceptNameCodeSequence" in element:
                for subelement in element.ConceptNameCodeSequence:
                    if (subelement.CodingSchemeDesignator == parameter["CodingSchemeDesignator"] and
                            subelement.CodingSchemeVersion == parameter["CodingSchemeVersion"] and
                            subelement.CodeValue == parameter["CodeValue"]):
                        points[element.AnnotationGroupNumber] = element.ReferencedSamplePositions
        return points

    def plot_representative_beat(self):
        if 'WaveformData' not in self.ds.WaveformSequence[1]:
            return
        dicom_array = self.ds.waveform_array(1)
        dicom_channel_data: npt.NDArray[(Any,), np.float64] = dicom_array[:, CHANNEL_NAMES[self.dicom_channel]]

        all_point_values = []
        all_point_labels = []
        markers = {}
        if self.controller.buttons['Show Annotations']:
            for par in self.parameters:
                points = self.get_points(self.ds.WaveformAnnotationSequence, par)
                # print(f"{par['CodeMeaning']}: {points}")
                if 2 in points and points[2] != 0:
                    all_point_values.append(points[2])
                    all_point_labels.append(par['CodeMeaning'])
                    markers[par['CodeMeaning']] = points[2] - 1

        self.plot_dicom_channel(self.ax_rep_beat, 'Representative Beat', 'ms (Millisecond)', '\u03BCV (Mikrovolt)',
                                dicom_channel_data, all_point_values, all_point_labels, label='Mortara Algorithm')
        self.plot_dicom_channel(self.ax_rep_beat, 'Representative Beat', 'ms (Millisecond)', '\u03BCV (Mikrovolt)',
                                self.analyzer.average_beat, line_color='k', label='Own Algorithm')
        if not self.controller.buttons['Hide Legend']:
            self.ax_rep_beat.legend(loc='upper left')
        # self.calc_p_parameters(markers)

    def plot_rhythm(self):
        dicom_array = self.ds.waveform_array(0)
        dicom_channel_data: npt.NDArray[(Any,), np.float64] = dicom_array[:, CHANNEL_NAMES[self.dicom_channel]]
        additional_points = []
        if self.controller.buttons['Show R Peaks (WFDB)']:
            additional_points.append({idx: 'QRS (WFDB)' for idx in self.analyzer.r_peaks_corrected})
        if self.controller.buttons['Show Pacemaker Spikes']:
            additional_points.append({idx: 'Pacemaker' for idx in self.analyzer.pacemaker_spikes})
        if self.controller.buttons['Show HP Filtered Signal']:
            self.ax_rhythm.plot(self.analyzer.filtered, 'r')
        if self.controller.buttons['Show HP Filtered Threshold']:
            self.ax_rhythm.axhline(self.analyzer.hp_filtered_threshold)
            self.ax_rhythm.axhline(-1 * self.analyzer.hp_filtered_threshold)

        all_point_values = []
        all_point_labels = []
        if self.controller.buttons['Show Annotations']:
            for par in self.parameters:
                points = self.get_points(self.ds.WaveformAnnotationSequence, par)
                for key, value in points.items():
                    if key != 2 and value != 0:
                        all_point_values.append(value)
                        all_point_labels.append(par['CodeMeaning'])

        self.plot_dicom_channel(self.ax_rhythm, 'Rhythm', 'ms (Millisecond)', '\u03BCV (Mikrovolt)',
                                dicom_channel_data, all_point_values, all_point_labels, additional_points)

    @staticmethod
    def plot_dicom_channel(axes: plt.Axes,
                           title: str,
                           x_label: str,
                           y_label: str,
                           dicom_channel_data: npt.NDArray[np.float64],
                           points: list[int] = None,
                           point_labels: list[str] = None,
                           additional_points: list[dict[int, str]] = None,
                           line_color: str = 'b',
                           label: str | None = None):
        axes.set_title(title)
        axes.set(xlabel=x_label)
        axes.set(ylabel=y_label)
        axes.plot(dicom_channel_data, line_color, linewidth=.5, label=label)
        if points is not None and len(points) > 0:
            points = [p - 1 for p in points]
            axes.plot(points, [dicom_channel_data[i] for i in points], 'r+')
        if point_labels is not None and points is not None and len(point_labels) > 0:
            for i, lable in enumerate(point_labels):
                axes.annotate(lable, (points[i], float(dicom_channel_data[points[i]])), ha='left', va='top',
                              rotation=-60)
        if additional_points is not None and len(additional_points) > 0:
            for group in additional_points:
                axes.plot([k for k in group.keys()], [dicom_channel_data[k] for k in group.keys()],
                          'r*')
                for k in group.keys():
                    axes.annotate(group[k], (k, float(dicom_channel_data[k])), ha='left', va='top',
                                  rotation=-60)

    def on_click(self, event: Event):
        if event.canvas.toolbar.mode != '':
            return
        event = cast(MouseEvent, event)
        if event.button == 1:
            if self.dir_index >= len(self.dicom_files) - 1:
                return
            else:
                self.dir_index += 1
        elif event.button == 3:
            if self.dir_index <= 0:
                return
            else:
                self.dir_index -= 1
        else:
            return
        self.ax_rep_beat.clear()
        self.ax_rhythm.clear()
        self.filename = self.dicom_files[self.dir_index]
        print(f"\n\nFilename: {self.filename}")
        self.update()

    def on_scroll(self, event: Event):
        event = cast(MouseEvent, event)
        increment = -1 if event.button == 'up' else 1
        self.change_channel(increment)

    def on_press(self, event: Event):
        event = cast(KeyEvent, event)
        if event.key == 'up':
            self.change_channel(1)
        elif event.key == 'down':
            self.change_channel(-1)

    def on_close(self, _):
        self.controller.close()

    def change_channel(self, increment: int):
        n = CHANNEL_NAMES[self.dicom_channel]
        n = n + increment
        if n in CHANNEL_NAMES.values():
            self.dicom_channel = [k for k, v in CHANNEL_NAMES.items() if v == n][0]
            self.ax_rep_beat.clear()
            self.ax_rhythm.clear()
            print(f"\n\nFilename: {self.filename}")
            print(f"Channel: {self.dicom_channel}")
            self.update(False)

    def calc_p_parameters(self, markers: dict[str, int]):
        sampling_frequency = self.ds.WaveformSequence[1].SamplingFrequency
        full_name: str = self.ds.PatientName.encode('UTF-8').decode('UTF-8')
        dob = datetime.strptime(self.ds.PatientBirthDate, '%Y%m%d').strftime('%Y-%m-%d') \
            if self.ds.PatientBirthDate != '' else None
        channel_ii = self.ds.waveform_array(1)[:, CHANNEL_NAMES['II']]
        channel_iii = self.ds.waveform_array(1)[:, CHANNEL_NAMES['III']]
        channel_avf = self.ds.waveform_array(1)[:, CHANNEL_NAMES['aVF']]
        channel_v1 = self.ds.waveform_array(1)[:, CHANNEL_NAMES['V1']]
        p_wave_data_ii = channel_ii[markers['P Onset']:markers['P Offset'] + 1]
        p_wave_data_iii = channel_iii[markers['P Onset']:markers['P Offset'] + 1]
        p_wave_data_avf = channel_avf[markers['P Onset']:markers['P Offset'] + 1]
        p_wave_data_v1 = channel_v1[markers['P Onset']:markers['P Offset'] + 1]

        temp = np.abs(p_wave_data_ii - np.median(p_wave_data_ii))
        median_distance = np.median(temp)
        scaled = temp / median_distance if median_distance else np.zeros(len(temp))
        p_wave_no_spike = p_wave_data_ii[scaled < 1.5]

        p_duration = (markers['P Offset'] - markers['P Onset']) * (1000 / sampling_frequency)
        p_area = 0.5 * p_duration * (p_wave_no_spike.max() / 1000)

        is_biphasic_ii = self.is_biphasic(p_wave_data_ii, 'II')
        is_biphasic_iii = self.is_biphasic(p_wave_data_iii, 'III')
        is_biphasic_avf = self.is_biphasic(p_wave_data_avf, 'aVF')

        if any([is_biphasic_ii, is_biphasic_iii, is_biphasic_avf]) and p_duration >= 120:
            aib = 'Yes'
        else:
            aib = 'No'

        ptav1 = None
        ptdv1 = None
        ptfv1 = None
        p_pulmonale = 'No'
        p_mitrale = 'No'

        if self.is_biphasic(p_wave_data_v1, 'V1'):
            ptav1, ptdv1, ptfv1 = self.calc_ptfv1(p_wave_data_v1, sampling_frequency, markers)

        if not is_biphasic_ii:
            if p_duration < 120 and p_wave_data_ii.max() > 250:
                p_pulmonale = 'Yes'
            elif p_duration > 120 and p_wave_data_ii.max() <= 250:
                p_mitrale = 'Yes'

        p_parameters: dict[str, Any] = dict()
        p_parameters['Name'] = full_name.split('^')[1] + ' ' + full_name.split('^')[0][0] + '.'
        p_parameters['Gender'] = self.ds.PatientSex
        p_parameters['D.O.B'] = dob
        p_parameters['Age'] = self.ds.PatientAge
        p_parameters['Date'] = (datetime.strptime(self.ds.AcquisitionDateTime, '%Y%m%d%H%M%S')
                                .strftime('%Y-%m-%dT%H:%M:%S'))
        p_parameters['HR'] = self.get_hr()
        p_parameters['PWD'] = p_duration
        p_parameters['PWA'] = p_area
        p_parameters['P wave axis'] = self.get_p_axis()
        p_parameters['AIB'] = aib
        p_parameters['PTAV1'] = ptav1
        p_parameters['PTDV1'] = ptdv1
        p_parameters['PTFV1'] = ptfv1
        p_parameters['P Pulmonale'] = p_pulmonale
        p_parameters['P Mitrale'] = p_mitrale
        p_parameters['PR Interval'] = self.get_pr_interval()
        p_parameters['Lagetyp'] = self.get_lagetyp(self.get_qrs_axis())
        p_parameters['QRS duration'] = self.get_qrs_duration()
        p_parameters['Sokolow Lyon index'] = self.get_sokolow_lyon(markers)
        p_parameters['QTc Bazett'] = self.get_qtc_bazett()
        p_parameters['Text'] = '\n'.join(
            [element.UnformattedTextValue for element in self.ds.WaveformAnnotationSequence
             if 'UnformattedTextValue' in element])
        for key, value in p_parameters.items():
            print(f"'{key}': '{value}'")
        self.plot_p_parameters(markers)

    def plot_p_parameters(self, markers: dict[str, int]):
        channel_ii = self.ds.waveform_array(1)[:, CHANNEL_NAMES['II']]
        p_wave_data_ii = channel_ii[markers['P Onset']:markers['P Offset'] + 1]

        self.ax_rep_beat.vlines(x=[markers['P Onset'], markers['P Offset']], ymin=0,
                                ymax=self.ax_rep_beat.get_ylim()[1], ls='--',
                                label='P Duration')
        if self.dicom_channel == 'II':
            self.ax_rep_beat.vlines(x=np.argmax(p_wave_data_ii) + markers['P Onset'],
                                    ymin=0,
                                    ymax=p_wave_data_ii.max(),
                                    colors=['r'],
                                    ls='--',
                                    label='P wave amplitude')

    @staticmethod
    def is_biphasic(p_wave_data: npt.NDArray[np.float64], channel: str) -> bool:
        baseline = (p_wave_data[0] + p_wave_data[-1]) / 2
        p_wave_data_adjusted = p_wave_data - baseline
        first_half = p_wave_data_adjusted[0:int(p_wave_data_adjusted.size / 2)]
        second_half = p_wave_data_adjusted[int(p_wave_data_adjusted.size / 2):]
        if np.sum(first_half >= 0) > first_half.size / 2 and np.sum(second_half < 0) > second_half.size / 2:
            print(f"P Wave is biphasic in {channel}")
            return True
        return False

    def calc_ptfv1(self, p_wave_data: npt.NDArray[np.float64], sampling_frequency: int, markers: dict[str, int]) -> (
            tuple)[float, float, float]:
        baseline = (p_wave_data[0] + p_wave_data[-1]) / 2
        p_wave_data_adjusted = p_wave_data - baseline
        i = 1
        split = -1
        best = 0
        while i < p_wave_data_adjusted.size - 1:
            metric = np.sum(p_wave_data_adjusted[:i] > 0) + np.sum(p_wave_data_adjusted[i:] < 0)
            if metric > best:
                best = metric
                split = i
            i += 1
        if split < 0:
            raise RuntimeError('Could not split biphasic wave!')
        second_half = p_wave_data[split:]
        ptdv1 = second_half.size * (1000 / sampling_frequency)
        ptav1 = abs(second_half.min()) / 1000
        ptfv1 = ptav1 * ptdv1

        if self.dicom_channel == 'V1':
            self.ax_rep_beat.vlines(x=markers['P Onset'] + split - 1 + np.argmin(second_half),
                                    ymin=0,
                                    ymax=second_half.min(),
                                    colors=['g'],
                                    ls='--',
                                    label='PTAV1')
            self.ax_rep_beat.hlines(y=baseline,
                                    xmin=markers['P Onset'] + split,
                                    xmax=markers['P Offset'],
                                    colors=['g'],
                                    label='PTDV1')
        return ptav1, ptdv1, ptfv1

    @staticmethod
    def get_lagetyp(qrs_axis: float) -> str:
        if qrs_axis is None:
            return ''
        if qrs_axis < -30:
            return '\u00DCberdrehter Linkstyp'
        if -30 <= qrs_axis < 0:
            return 'Linkstyp'
        if 0 <= qrs_axis < 30:
            return 'Linkstyp/Horizontaltyp'
        if 30 <= qrs_axis < 60:
            return 'Indifferenztyp'
        if 60 <= qrs_axis < 90:
            return 'Steiltyp'
        if 90 <= qrs_axis < 120:
            return 'Rechtstyp'
        if qrs_axis >= 120:
            return '\u00DCberdrehter Rechtstyp'

    def get_sokolow_lyon(self, markers: dict[str, int]) -> float | None:
        if not all(key in markers for key in ['QRS Onset', 'QRS Offset', 'Fiducial Point']):
            return None

        v1 = self.ds.waveform_array(1)[:, CHANNEL_NAMES['V1']]
        v5 = self.ds.waveform_array(1)[:, CHANNEL_NAMES['V5']]
        v6 = self.ds.waveform_array(1)[:, CHANNEL_NAMES['V6']]
        v1_s = self.get_s_peak(markers, v1)
        v5_r = self.get_r_peak(markers, v5)
        v6_r = self.get_r_peak(markers, v6)
        if v6_r[1] > v5_r[1]:
            print('Sokolow-Lyon index: Using V6')
            sokolow_lyon = v6_r[1] + v1_s[1]
            r = v6_r
            channel = 'V6'
        else:
            print('Sokolow-Lyon index: Using V5')
            sokolow_lyon = v5_r[1] + v1_s[1]
            r = v5_r
            channel = 'V5'
        sokolow_lyon /= 1000

        if self.dicom_channel == 'V1':
            self.ax_rep_beat.annotate('S', (v1_s[0], v1_s[1]),
                                      ha='left', va='top',
                                      rotation=-60)
            self.ax_rep_beat.plot(v1_s[0], v1_s[1], 'r+')

        if self.dicom_channel == channel:
            self.ax_rep_beat.annotate('R', (r[0], r[1]),
                                      ha='left', va='top',
                                      rotation=-60)
            self.ax_rep_beat.plot(r[0], r[1], 'r+')
        return sokolow_lyon

    @staticmethod
    def get_r_peak(markers: dict[str, int], data: npt.NDArray[np.float64]) -> tuple[int, float]:
        if data[markers['Fiducial Point']] > 0:
            index = np.argmax(data[markers['QRS Onset']:markers['QRS Offset'] + 1])
        else:
            index = np.argmin(data[markers['QRS Onset']:markers['QRS Offset'] + 1])
        index += markers['QRS Onset']
        return int(index), float(data[index])

    @staticmethod
    def get_s_peak(markers: dict[str, int], data: npt.NDArray[np.float64]) -> tuple[int, float]:
        if data[markers['Fiducial Point']] > 0:
            index = np.argmin(data[markers['Fiducial Point']:markers['QRS Offset'] + 1])
        else:
            index = np.argmax(data[markers['Fiducial Point']:markers['QRS Offset'] + 1])
        index += markers['Fiducial Point']
        return int(index), float(data[index])

    def get_hr(self) -> float | None:
        return self.get_annotation('5.10.2.5-1', 'SCPECG')

    def get_p_axis(self) -> float | None:
        return self.get_annotation('5.10.3-11', 'SCPECG')

    def get_qrs_axis(self) -> float | None:
        return self.get_annotation('5.10.3-13', 'SCPECG')

    def get_pr_interval(self) -> float | None:
        return self.get_annotation('5.13.5-7', 'SCPECG')

    def get_qrs_duration(self) -> float | None:
        return self.get_annotation('5.13.5-9', 'SCPECG')

    def get_qtc_bazett(self) -> float | None:
        return self.get_annotation('2:15880', 'SCPECG')

    def get_annotation(self, code_value: str, coding_scheme_designator: str) -> float | None:
        for element in self.ds.WaveformAnnotationSequence:
            if 'ConceptNameCodeSequence' in element:
                for subelement in element.ConceptNameCodeSequence:
                    if (subelement.CodeValue == code_value and
                            subelement.CodingSchemeDesignator == coding_scheme_designator and
                            element.NumericValue is not None):
                        return float(element.NumericValue)
        return None
