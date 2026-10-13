"""
Building PDV from 1st principles
"""

import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import re
from matplotlib.widgets import Button, RangeSlider, RadioButtons, Slider
from scipy.signal import hilbert

class SigFile:
    def __init__(self, fname):
        self.has_sig = False
        self.has_phase = False
        self.has_fft = False
        self.fname = fname
        self.simple_name = re.split(r'[/.]', fname)[1]
#        self.simple_name = fname.split("/")[1]
        self.open()
        
    def open(self):
        test_data = pd.read_csv(self.fname)
        num_cols = len([i for i in test_data.keys()])
        del test_data
        if num_cols == 5:
            self.data = pd.read_csv(self.fname, header = None, names = ["_", "-", "*", "time", "V"])
            self.data = self.data.iloc[:, 3:]
        elif num_cols == 2:
            self.data = pd.read_csv(self.fname, header = None, names = ["time", "V"])
        else:
            raise Exception("# cols is not 2 or 5")
        self.has_sig = True

    def show_sig(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(10,4))
            ax.plot(self.data["time"], self.data["V"]*1e3, label = "Voltage", c = "r", alpha = 0.9, linestyle = "-", linewidth = 1)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Voltage (mV)')
            ax.set_title(self.simple_name)
            fig.savefig(str(self.simple_name) + '.tiff')
            

    def fourier(self, ax=None):
        time = self.data["time"].values
        voltage = self.data["V"].values
        dt = time[1] - time[0]

        self.spectrum = np.fft.rfft(voltage)
        self.freqs = np.fft.rfftfreq(len(voltage), d=dt)
        self.fft_mag = np.abs(self.spectrum)
        self.has_fft = True
    
        if ax is not None:
            ax.plot(self.freqs, self.fft_mag)
            ax.set_xlabel("Frequency (Hz)")
            ax.set_ylabel("Magnitude")
            ax.set_title(f"Fourier Transform: {self.simple_name}")

        return self.freqs, self.fft_mag

    def clean_sig(self, low_hz=None, high_hz=None):
        if not self.has_fft:
            self.fourier()

        freqs = self.freqs
        n = len(self.data["V"].values)

        filtered_spectrum = self.spectrum.copy()
        if low_hz is not None:
            filtered_spectrum[freqs < low_hz] = 0
        if high_hz is not None:
            filtered_spectrum[freqs > high_hz] = 0

        cleaned = np.fft.irfft(filtered_spectrum, n=n)
        self.data["V_clean"] = cleaned

    def get_freq_velo(self, breakout_time=None, wavelength=1550e-9):
        if "V_clean" not in self.data.columns:
            raise ValueError("Filter the signal first.")

        analytic_signal = hilbert(self.data["V_clean"].values)
        self.phase = np.unwrap(np.angle(analytic_signal))

        time = self.data["time"].values
        dt = time[1] - time[0]
        self.time_mid = time[:-1] #get time for plotting freq/velo
        self.beat_freq = np.diff(self.phase) / (2 * np.pi * dt) #convert phase to frequency
        self.velocity = (wavelength / 2.0) * self.beat_freq #From Dolan: freq ==> velo

        if breakout_time is not None:
            mask = self.time_mid < breakout_time
            self.beat_freq[mask] = 0
            self.velocity[mask] = 0

        self.has_phase = True
        return self.time_mid, self.beat_freq, self.velocity

class Plotter:
    def __init__(self, file:SigFile):
        self.file = file
        self.fig = plt.figure(figsize = (18, 5))
        self.sig_ax = self.fig.add_subplot(1, 2, 1)
        self.velo_ax = self.fig.add_subplot(1, 2, 2)
        self.phase_ax = self.velo_ax.twinx()
        self.fig.suptitle(file.simple_name)
        self.sig_ax.set_xlabel("Time (s)")
        self.sig_ax.set_ylabel("Voltage (V)")
        self.velo_ax.set_xlabel("Time (s)")
        self.velo_ax.set_ylabel("Velo (m/s)")
        self.sig_ax.set_title("Signal")
        self.velo_ax.set_title("Phase/Velocity")
        self.phase_ax.set_ylabel("Beat Freq. (Hz)")

        #buttons
        #FOURIER
        self.fig.subplots_adjust(bottom=0.35)
        button_ax = self.fig.add_axes([0.15, 0.18, 0.1, 0.075])
        self.fourier_button = Button(button_ax, "Fourier")
        self.fourier_button.on_clicked(self.on_fourier_clicked)

        #Take Phase
        phase_button_ax = self.fig.add_axes([0.4, 0.18, 0.1, 0.075])
        self.phase_button = Button(phase_button_ax, "Take Phase")
        self.phase_button.on_clicked(self.on_phase_clicked)

        #adjust breakout time
        #Take Phase
        breakout_ax = self.fig.add_axes([0.4, 0.08, 0.1, 0.075])
        self.zero_button = Button(breakout_ax, "Zero Sig")
        self.zero_button.on_clicked(self.on_zero_clicked)

        #breakout time slider
        breakout_slider_ax = self.fig.add_axes([0.15, 0.08, 0.2, 0.05])
        time_min = self.file.data["time"].values[0]
        time_max = self.file.data["time"].values[-1]
        self.breakout_time = time_min
        self.breakout_slider = Slider(breakout_slider_ax, "Breakout t", time_min, time_max, valinit=time_min)
        self.breakout_slider.on_changed(self.on_breakout_changed)

        self.breakout_line = self.sig_ax.axvline(time_min, color='purple', linewidth=2)

        self.fourier_fig = None
        self.freq_slider = None
        self.filter_button = None
        self.filtered_line = None
        self.low_line = None
        self.high_line = None
        self.phase_line = None
        self.velo_line = None

    def show(self):
        if self.file.has_sig == True:
            self.file.show_sig(self.sig_ax)
            self.sig_ax.legend()
        
    def on_fourier_clicked(self, event):
        self.show_fourier()

    def show_fourier(self):
        self.fourier_fig, ax = plt.subplots()
        self.file.fourier(ax=ax)
        self.fourier_fig.subplots_adjust(bottom=0.3)

        slider_ax = self.fourier_fig.add_axes([0.2, 0.15, 0.6, 0.05])
        self.freq_slider = RangeSlider(
            slider_ax, "Freq (Hz)",
            valmin=self.file.freqs[0], valmax=self.file.freqs[-1],
            valinit=(self.file.freqs[0], self.file.freqs[-1])
        )

        self.low_line = ax.axvline(self.file.freqs[0], color='red', linestyle='--')
        self.high_line = ax.axvline(self.file.freqs[-1], color='red', linestyle='--')

        self.freq_slider.on_changed(self.on_slider_changed)

        filter_button_ax = self.fourier_fig.add_axes([0.4, 0.03, 0.2, 0.06])
        self.filter_button = Button(filter_button_ax, "Filter")
        self.filter_button.on_clicked(self.on_filter_clicked)
        self.fourier_fig.show()

    def on_filter_clicked(self, event):
        low_hz, high_hz = self.freq_slider.val
        self.file.clean_sig(low_hz=low_hz, high_hz=high_hz)

        if self.filtered_line is not None:
            self.filtered_line.remove()

        self.filtered_line, = self.sig_ax.plot(
            self.file.data["time"], self.file.data["V_clean"], label="Filtered", c = "red", linewidth = 2, linestyle = "-"
        )
        self.sig_ax.legend()
        self.fig.canvas.draw_idle()

    def on_phase_clicked(self, event):
        print("getting phase ...")
        if "V_clean" not in self.file.data.columns:
            print("Filter the signal first.")
            return

        time_mid, inst_freq_hz, velocity = self.file.get_freq_velo(self.breakout_time)
        self.phase = inst_freq_hz

        if self.phase_line is None:
            self.phase_line, = self.phase_ax.plot(time_mid, inst_freq_hz, c="tab:blue")
        else:
            self.phase_line.set_data(time_mid, inst_freq_hz)
            self.phase_ax.relim()
            self.phase_ax.autoscale_view()

        if self.velo_line is None:
            self.velo_line, = self.velo_ax.plot(time_mid, velocity, c="tab:orange")
        else:
            self.velo_line.set_data(time_mid, velocity)
            self.velo_ax.relim()
            self.velo_ax.autoscale_view()

        self.fig.canvas.draw_idle()

    def on_slider_changed(self, val):
        low_hz, high_hz = val
        self.low_line.set_xdata([low_hz, low_hz])
        self.high_line.set_xdata([high_hz, high_hz])
        self.fourier_fig.canvas.draw_idle()

    def on_breakout_changed(self, val):
        self.breakout_time = val
        self.breakout_line.set_xdata([val, val])
        self.fig.canvas.draw_idle()

    def on_zero_clicked(self, event):
        if "V_clean" not in self.file.data.columns:
            return

        time = self.file.data["time"].values
        mask = time < self.breakout_time
        self.file.data.loc[mask, "V_clean"] = 0

        if self.filtered_line is not None:
            self.filtered_line.remove()

        self.filtered_line, = self.sig_ax.plot(
            self.file.data["time"], self.file.data["V_clean"], label="Filtered", c="red", linewidth=2, linestyle="-"
        )
        self.sig_ax.legend()
        self.fig.canvas.draw_idle()
    