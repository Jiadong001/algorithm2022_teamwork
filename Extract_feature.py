import numpy as np
import librosa
import soundfile as sf
from scipy import signal
from scipy.fftpack import dct

# extract logmag-spectrogram
def extract_stft(wav_file_path, window_duration=0.04, window_shift=0.02, n_remain=400, option="cutoff"):
    # read
    wav_data, wav_fs = sf.read(wav_file_path)   # 10s, single channel
    # spectrogram
    window_length = int(window_duration*wav_fs)
    window_overlap = int((window_duration-window_shift)*wav_fs)
    _,_,X = signal.spectral.spectrogram(
                wav_data, window='hamming',
                nperseg=window_length, noverlap=window_overlap, nfft=window_length,
                detrend=False, return_onesided=True, mode='magnitude')
    stftX = np.swapaxes(X,0,1)                  # [f, frames] -> [frames,f]
    if option=="cutoff":
        return np.log10(stftX[:,:n_remain])     # [frames, 400]
    else:
        return np.log10(stftX)                  # [frames, f]

# extract fbanks or mfcc(cutoffed dcted fbanks)
def extract_mfcc(wav_file_path, window_duration=0.04, window_shift=0.02, n_mels=26, n_remain=13, option="mfcc"):
    # read
    wav_data, wav_fs = sf.read(wav_file_path)       # 10s, single channel
    # spectrogram
    window_length = int(window_duration*wav_fs)
    window_overlap = int((window_duration-window_shift)*wav_fs)
    _,_,X = signal.spectral.spectrogram(
                wav_data, window='hamming',
                nperseg=window_length, noverlap=window_overlap, nfft=window_length,
                detrend=False, return_onesided=True, mode='magnitude')
    # mel filter banks
    melW = librosa.filters.mel(
            sr=wav_fs, n_fft=window_length,
            n_mels=n_mels, fmin=0., fmax=wav_fs/2)  # melW:[n_mels, f.dim]      
    melW /= np.max(melW,axis=-1)[:,None]            # normalization, [:,None]: 1dim->2dim
    # filtering
    melX = np.dot(melW, X)                          # melX:[n_mels, t.dim(frames)]
    fbanks = np.swapaxes(melX,0,1)                  # [n_mels, frames] -> [frames,n_mels]
    fbanks = np.log10(fbanks)
    if option=="fbanks":
        return fbanks
    else:
        mfcc = dct(fbanks, type=2, axis=1, norm='ortho')[:, :n_remain]  # DCT
        return mfcc                                 # [frames, 13]
