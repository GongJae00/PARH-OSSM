import numpy as np
from scipy import signal


def Welch_rpm(resp, fps, winsize, minHz=0.1, maxHz=0.4, fRes=0.1):
    """
    This method computes the spectrum of a respiratory signal

    Parameters
    ----------
        resp: the respiratory signal
        fps: the fps of the video from which signal is estimated
        winsize: the window size used to compute spectrum
        minHz: the lower bound for accepted frequencies
        maxHz: the upper bound for accepted frequencies

    Returns
    -------
        the array of frequencies and the corrisponding PSD
    """
    step = 1
    nperseg=fps*winsize
    noverlap=fps*(winsize-step)

    nyquistF = fps/2
    nfft = max(2048, (60*2*nyquistF) / fRes)

    # -- periodogram by Welch
    F, P = signal.welch(resp, nperseg=nperseg, noverlap=noverlap, fs=fps, nfft=nfft)
    F = F.astype(np.float32)
    P = P.astype(np.float32)
    # -- freq subband
    band = np.argwhere((F > minHz) & (F < maxHz)).flatten()

    Pfreqs = 60*F[band]
    Power = P[:, band]

    return Pfreqs, Power


def sig_to_RPM(sig, fps, winsize, minHz=0.1, maxHz=0.4):
    sig = [s for s in sig if np.asarray(s).size > 0]
    if len(sig) == 0:
        return np.array([np.nan])
    sig = np.vstack(sig)

    Pfreqs, Power = Welch_rpm(sig, fps, winsize, minHz, maxHz)
    Pmax = np.argmax(Power, axis=1)  # power max
    rpm = Pfreqs[Pmax.squeeze()]

    if (rpm.size == 1):
        return rpm.reshape(1, -1)

    return rpm


BDF_BELT_FILL = 0.015624  # physical value of the digital fill (16) on the MAHNOB-HCI belt channel


def recorded_belt(belt):
    """The belt up to its last recorded sample.

    A MAHNOB-HCI BDF file holds whole 1-s data records and fills the rest of the last record with one
    digital value, read as BDF_BELT_FILL, far from the recorded belt (near -6e4). Band-passing the step
    into that fill distorts the reference far from the end, so the fill is not part of the belt. A belt
    that does not end in the fill, such as every COHFACE belt, is returned unchanged.
    """
    belt = np.asarray(belt, dtype=float).reshape(-1)
    fill = int(np.argmin(np.isclose(belt[::-1], BDF_BELT_FILL, rtol=0, atol=1e-6)))
    return belt[:belt.size - fill]


def filter_RW(sig, fps, lo=0.08, hi=0.5):
    """
    This method performs posptprocessing steps of fiedler methods; the postprocessing process performs on the signal a normalization, computes the gradient of the signal and applies a band-pass filter

    Parameters
    ----------
        sig: the considered signal
        fps : the fps of the considered video

    Returns
    -------
        the postprocessed signal
    """
    #sig = np.diff(np.asarray(sig), axis=0)
    #sig = np.squeeze(sig)
    if (sig.ndim == 1):
        sig = sig[np.newaxis,:]

    b, a = signal.butter(N=2, Wn=[lo, hi], fs=fps, btype='bandpass')
    padlen = 3 * max(len(a), len(b))
    if sig.shape[-1] <= padlen:
        # Signal too short for filtfilt; return zeros
        return np.zeros_like(sig)
    filtered_sig = signal.filtfilt(b, a, sig)

    return filtered_sig


def sig_windowing(sig, fps, wsize, stride=1):
    """ Performs signal windowing

    Args:
      sig (list/array): full signal
      fps       (float): frames per seconds      
      wsize     (float): size of the window (in seconds)
      stride    (float): stride (in seconds)

    Returns:
      win_sig (list): windowed signal
      timesES (list): times of (centers) windows 
    """
    sig = np.array(sig).squeeze()
    block_idx, timesES = sliding_straded_win_idx(sig.shape[0], wsize, stride, fps)
    sig_win  = []
    for e in block_idx:
        st_frame = int(e[0])
        end_frame = int(e[-1])
        wind_signal = np.copy(sig[st_frame: end_frame+1])
        sig_win.append(wind_signal[np.newaxis, :])

    return sig_win, timesES


def sliding_straded_win_idx(N, wsize, stride, fps):
    """
    This method is used to compute the indices for creating an overlapping windows signal.

    Args:
        N (int): length of the signal.
        wsize (float): window size in seconds.
        stride (float): stride between overlapping windows in seconds.
        fps (float): frames per seconds.

    Returns:
        List of ranges, each one contains the indices of a window, and a 1D ndarray of times in seconds, where each one is the center of a window.
    """
    wsize_fr = wsize*fps
    stride_fr = stride*fps
    idx = []
    timesES = []
    num_win = int((N-wsize_fr)/stride_fr)+1
    s = 0
    for i in range(num_win):
        idx.append(np.arange(s, s+wsize_fr))
        s += stride_fr
        timesES.append(wsize/2+stride*i)
    return idx, np.array(timesES, dtype=np.float32)
