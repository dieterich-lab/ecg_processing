from typing import Callable
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, CheckButtons


class PlotControllerWidget:
    def __init__(self, update: Callable):
        self.buttons = {'Hide Legend': False,
                        'Show R Peaks (WFDB)': False,
                        'Show Pacemaker Spikes': False,
                        'Show Annotations': False,
                        'Show HP Filtered Signal': False,
                        'Show HP Filtered Threshold': False}
        self.update = update

        self.fig, (self.check_buttons_ax, self.slider_ax) = plt.subplots(2, 1, height_ratios=[3, 1], figsize=(8, 2))

        self.check_buttons = CheckButtons(self.check_buttons_ax, list(self.buttons.keys()))
        self.slider = Slider(ax=self.slider_ax, label='Pacemaker\nThreshold', valmin=1, valmax=10, valinit=5, valstep=1)
        self.slider_value = self.slider.val

        self.slider.on_changed(self.on_slider_changed)
        self.check_buttons.on_clicked(self.on_check_button_clicked)
        self.fig.canvas.draw_idle()

    def on_slider_changed(self, val):
        self.slider_value = val
        self.update()

    def on_check_button_clicked(self, label: str | None):
        if label is None:
            self.buttons = dict.fromkeys(self.buttons, False)
        else:
            self.buttons[label] = not self.buttons[label]
        self.update()

    def close(self):
        plt.close(self.fig)
