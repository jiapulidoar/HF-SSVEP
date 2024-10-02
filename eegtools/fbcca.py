import scipy
import numpy as np
from scipy.stats import pearsonr, mode
from sklearn.cross_decomposition import CCA

class FBCCA:
    
    def __init__(self, num_harms=3, num_fbs=5, a=1.25, b=0.25):
        self.num_harms = num_harms
        self.num_fbs = num_fbs
        self.a = a
        self.b = b
        self.cca = CCA(n_components=1)

    def cca_reference(self, list_freqs, fs, num_smpls):
        num_freqs = len(list_freqs)
        tidx = np.arange(1, num_smpls + 1) / fs  # time index

        y_ref = np.zeros((num_freqs, 2 * self.num_harms, num_smpls))
        for freq_i in range(num_freqs):
            tmp = []
            for harm_i in range(1, self.num_harms + 1):
                stim_freq = list_freqs[freq_i]  # in HZ
                # Sin and Cos
                tmp.extend([np.sin(2 * np.pi * tidx * harm_i * stim_freq),
                            np.cos(2 * np.pi * tidx * harm_i * stim_freq)])
            y_ref[freq_i] = tmp  # 2*num_harms because include both sin and cos

        return y_ref

    def fbcca(self, eeg, list_freqs, fs):
        fb_coefs = np.power(np.arange(1, self.num_fbs + 1), (-self.a)) + self.b

        num_targs = len(list_freqs)
        events, _, num_smpls = eeg.shape
        y_ref = self.cca_reference(list_freqs, fs, num_smpls)

        r = np.zeros((self.num_fbs, num_targs))
        r_mode = []
        r_corr_avg = []
        rhos = []

        for event in range(eeg.shape[0]):
            test_tmp = np.squeeze(eeg[event, :, :])
            for fb_i in range(self.num_fbs):
                for class_i in range(num_targs):
                    testdata = self.filterbank(test_tmp, fs, fb_i)
                    refdata = np.squeeze(y_ref[class_i, :, :])
                    test_C, ref_C = self.cca.fit_transform(testdata.T, refdata.T)
                    r_tmp, _ = pearsonr(np.squeeze(test_C), np.squeeze(ref_C))
                    if np.isnan(r_tmp):
                        r_tmp = 0
                    r[fb_i, class_i] = r_tmp
            rho = np.dot(fb_coefs, r)
            rhos.append(rho)
            result = np.argmax(rho)
            r_mode.append(result)
            r_corr_avg.append(abs(rho[result]))

        return r_mode, rhos

    def filterbank(self, eeg, fs, idx_fb):
        if idx_fb is None:
            warnings.warn('stats:filterbank:MissingInput '
                          + 'Missing filter index. Default value (idx_fb = 0) will be used.')
            idx_fb = 0
        elif idx_fb < 0 or 9 < idx_fb:
            raise ValueError('stats:filterbank:InvalidInput '
                             + 'The number of sub-bands must be 0 <= idx_fb <= 9.')

        if len(eeg.shape) == 2:
            num_chans = eeg.shape[0]
            num_trials = 1
        else:
            _, num_chans, num_trials = eeg.shape

        Nq = fs / 2

        passband = [28, 58, 88]
        passband_h = [38, 74, 110]
        stopband = [24, 54, 84]
        stopband_h = [40, 76, 112]

        Wp = [passband[idx_fb] / Nq, 90 / Nq]
        Ws = [stopband[idx_fb] / Nq, 100 / Nq]

        N, Wn = scipy.signal.cheb1ord(Wp, Ws, 3, 40)
        B, A = scipy.signal.cheby1(N, 0.5, Wn, 'bandpass')

        y = np.zeros(eeg.shape)

        if num_trials == 1:
            for ch_i in range(num_chans):
                y[ch_i, :] = scipy.signal.filtfilt(B, A, eeg[ch_i, :])
        else:
            for trial_i in range(num_trials):
                for ch_i in range(num_chans):
                    y[:, ch_i, trial_i] = scipy.signal.filtfilt(B, A, eeg[:, ch_i, trial_i])
        return y