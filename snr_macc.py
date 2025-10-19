import numpy as np

#Note: These functions can take in a singular signal (T,) for pred and gt each or (B,T) a batch


def compute_snr(pred, gt, domain='time', Fs=30, N=1024, pulse_band=[45/60., 250/60.]):
    """
        domain: 'time' or 'frequency'
        Fs: Sampling frequency in Hz (used for frequency-domain SNR)
        N: FFT length
        pulse_band: List or tuple with [min_freq, max_freq] in Hz
    """
    eps = 1e-8
    #make 2d so that this function can be used for both individual signals and a batch of comparisons 
    pred = np.atleast_2d(pred)
    gt = np.atleast_2d(gt)
    
    if pred.shape != gt.shape:
        raise ValueError("Shape mismatch between pred and gt")

    if domain == 'time':
        signal_power = np.sum(gt**2, axis=1)
        noise_power = np.sum((gt - pred)**2, axis=1)
        snr = 10 * np.log10(signal_power / (noise_power + eps))
        return snr.mean()

    elif domain == 'frequency':
        f = np.linspace(0, Fs/2, N//2 + 1)
        min_idx = np.argmin(np.abs(f - pulse_band[0]))
        max_idx = np.argmin(np.abs(f - pulse_band[1]))
        wind_sz = N // 256

        def batch_fft_power(x):
            X = np.fft.rfft(x, n=N, axis=1, norm='forward')
            return np.abs(X)**2

        P_pred = batch_fft_power(pred)
        P_gt = batch_fft_power(gt)

        HRixs = np.argmax(P_gt[:, min_idx:max_idx], axis=1) + min_idx

        snr_values = []
        for i, ref_idx in enumerate(HRixs):
            lower = max(0, ref_idx - wind_sz)
            upper = ref_idx + wind_sz

            signal_power = np.sum(P_pred[i, lower:upper])
            noise_power = (
                np.sum(P_pred[i, min_idx:lower]) + 
                np.sum(P_pred[i, upper:max_idx])
            )
            snr_i = 10 * np.log10((signal_power + eps) / (noise_power + eps))
            snr_values.append(snr_i)

        return np.mean(snr_values)

    else:
        raise ValueError("domain must be either 'time' or 'frequency'")
    
    
def compute_macc(pred, gt):
    """
    MACC: Maximum Amplitude of Normalized Cross-Correlation between pred and gt.
    """
    
    pred = np.atleast_2d(pred)
    gt = np.atleast_2d(gt)

    def _single_macc(x, y):
        x = x - np.mean(x)
        y = y - np.mean(y)
        corr = np.correlate(y, x, mode='full')
        norm = np.linalg.norm(x) * np.linalg.norm(y)
        return np.max(corr) / (norm + 1e-8)

    maccs = [_single_macc(p, g) for p, g in zip(pred, gt)] #iterate through batches
    return np.mean(maccs)


