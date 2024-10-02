import numpy as np
import matplotlib.pyplot as plt
from mne.time_frequency import psd_array_welch

class PSDAnalyzer:
    def __init__(self, frequencies, sfreq=2048, tmin=0, tmax=5, fmin=25, fmax=120, ):
        self.frequencies = frequencies 
        self.sfreq = sfreq
        self.tmin = tmin
        self.tmax = tmax
        self.fmin = fmin
        self.fmax = fmax
        self.psds = None 
        self.freqs = None 
        self.snrs = None 

    def calculate_psd_snr(self, epochs, frequencies):
        spectrum = epochs.compute_psd(
            method="welch",
            n_fft=int(self.sfreq * (self.tmax - self.tmin)),
            n_overlap=int(self.sfreq * (self.tmax - self.tmin) / 2),
            n_per_seg=None,
            tmin=self.tmin,
            tmax=self.tmax,
            fmin=self.fmin,
            fmax=self.fmax,
            verbose=False,
        )
        self.psds, self.freqs = spectrum.get_data(return_freqs=True)
        self.snrs = self.snr_spectrum(self.psds, noise_n_neighbor_freqs=2, noise_skip_neighbor_freqs=1)

        return self.psds, self.freqs, self.snrs

    def snr_spectrum(self, psd, noise_n_neighbor_freqs, noise_skip_neighbor_freqs):
        """Compute SNR spectrum from PSD spectrum using convolution.

        Parameters
        ----------
        psd : ndarray, shape ([n_trials, n_channels,] n_frequency_bins)
            Data object containing PSD values. Works with arrays as produced by
            MNE's PSD functions or channel/trial subsets.
        noise_n_neighbor_freqs : int
            Number of neighboring frequencies used to compute noise level.
            increment by one to add one frequency bin ON BOTH SIDES
        noise_skip_neighbor_freqs : int
            set this >=1 if you want to exclude the immediately neighboring
            frequency bins in noise level calculation

        Returns
        -------
        snr : ndarray, shape ([n_trials, n_channels,] n_frequency_bins)
            Array containing SNR for all epochs, channels, frequency bins.
            NaN for frequencies on the edges, that do not have enough neighbors on
            one side to calculate SNR.
        """
        # Construct a kernel that calculates the mean of the neighboring
        # frequencies
        averaging_kernel = np.concatenate(
            (
                np.ones(noise_n_neighbor_freqs),
                np.zeros(2 * noise_skip_neighbor_freqs + 1),
                np.ones(noise_n_neighbor_freqs),
            )
        )
        averaging_kernel /= averaging_kernel.sum()

        # Calculate the mean of the neighboring frequencies by convolving with the
        # averaging kernel.
        mean_noise = np.apply_along_axis(
            lambda psd_: np.convolve(psd_, averaging_kernel, mode="valid"), axis=-1, arr=psd
        )

        # The mean is not defined on the edges so we will pad it with nas. The
        # padding needs to be done for the last dimension only so we set it to
        # (0, 0) for the other ones.
        edge_width = noise_n_neighbor_freqs + noise_skip_neighbor_freqs
        pad_width = [(0, 0)] * (mean_noise.ndim - 1) + [(edge_width, edge_width)]
        mean_noise = np.pad(mean_noise, pad_width=pad_width, constant_values=np.nan)

        return psd / mean_noise

    def plot_psd_snr(self, psds, freqs, snrs, stim, file_name):
        fig, axes = plt.subplots(2, 1, sharex="all", sharey="none", figsize=(8, 5))
        freq_range = range(
            np.where(np.floor(freqs) == self.fmin)[0][0], 
            np.where(np.ceil(freqs) == self.fmax - 1)[0][0]
        )

        psds_plot = psds
        psds_mean = psds_plot.mean(axis=(0, 1))[freq_range]
        psds_std = psds_plot.std(axis=(0, 1))[freq_range]
        axes[0].plot(freqs[freq_range], psds_mean, color="b")
        axes[0].set(title="PSD spectrum", ylabel="Power (µV²/Hz)")
        axes[0].axvline(x=stim, label="{}Hz".format(stim), color='r', linestyle='--')

        # SNR spectrum
        snr_mean = snrs.mean(axis=(0, 1))[freq_range]
        snr_std = snrs.std(axis=(0, 1))[freq_range]

        axes[1].plot(freqs[freq_range], snr_mean, color="r")
        axes[1].axvline(x=stim, label="{}Hz".format(stim), color='b', linestyle='--')
        axes[1].set(
            title="SNR spectrum",
            xlabel="Frequency [Hz]",
            ylabel="SNR",
            ylim=[0, 6],
            xlim=[self.fmin, self.fmax],
        )

        plt.legend(bbox_to_anchor=(1.0, 1), loc='upper left')
        
        fig.show()
        namef1, namef2 = str(stim).split(".")
        #fig.savefig(file_name + f"_PSD_SNR_{namef1}_{namef2}Hz", bbox_inches='tight')

# Example usage:
# analyzer = PSDSNRAnalyzer()
# psds, freqs, snrs = analyzer.calculate_psd_snr(epochs, frequencies)
# analyzer.plot_psd_snr(psds, freqs, snrs, stim, file_name)