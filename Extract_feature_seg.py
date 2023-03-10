import numpy as np
import librosa
import soundfile as sf
from scipy import signal
from scipy.fftpack import dct

# extract logmag-spectrogram
def extract_stft(wav_file_path, window_duration=0.04, window_shift=0.02, n_remain=400, option="cutoff_stft", n_seg=10):
    # read
    wav_data, wav_fs = sf.read(wav_file_path)   # 10s, single channel
    # window
    window_length = int(window_duration*wav_fs)
    window_overlap = int((window_duration-window_shift)*wav_fs)
    # spectrogram
    _,_,X = signal.spectral.spectrogram(
                wav_data, window='hamming',
                nperseg=window_length, noverlap=window_overlap, nfft=window_length,
                detrend=False, return_onesided=True, mode='magnitude')
    stftX = np.swapaxes(X,0,1)                  # [f, frames] -> [frames,f]
    # segmentation
    seg_frames = int((len(stftX)-n_seg+1)/n_seg)   # n_seg=10: 49
    start_frame, end_frame = 0, seg_frames
    features = []
    if option=="cutoff_stft":
        for _ in range(n_seg):
            seg_feature = np.log10(stftX[start_frame:end_frame,:n_remain])
            features.append(np.expand_dims(seg_feature, axis=0))                # [1, seg_frames, 400]
            start_frame = end_frame + 1
            end_frame = start_frame + seg_frames
    else:
        for _ in range(n_seg):
            seg_feature = np.log10(stftX[start_frame:end_frame])
            features.append(np.expand_dims(seg_feature, axis=0))                # [1, seg_frames, f]
            start_frame = end_frame + 1
            end_frame = start_frame + seg_frames
    features = np.concatenate(features, axis=0)
    return features

# extract fbanks or mfcc(cutoffed dcted fbanks)
def extract_mfcc(wav_file_path, window_duration=0.04, window_shift=0.02, n_mels=26, n_remain=13, option="mfcc", n_seg=10):
    # read
    wav_data, wav_fs = sf.read(wav_file_path)       # 10s, single channel
    # window
    window_length = int(window_duration*wav_fs)
    window_overlap = int((window_duration-window_shift)*wav_fs)
    # spectrogram
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
    fbanks = np.swapaxes(melX,0,1)                  # [n_mels, frames] -> [frames, n_mels]
    # segmentation
    seg_frames = int((len(fbanks)-n_seg+1)/n_seg)   # n_seg=10: 49
    start_frame, end_frame = 0, seg_frames
    features = []
    if option=="fbanks":
        for _ in range(n_seg):
            seg_feature = fbanks[start_frame:end_frame]
            features.append(np.expand_dims(seg_feature, axis=0))            # [1, seg_frames, n_mels]
            start_frame = end_frame + 1
            end_frame = start_frame + seg_frames
    elif option=="logfbanks":
        fbanks = np.log10(fbanks)
        for _ in range(n_seg):
            seg_feature = fbanks[start_frame:end_frame]
            features.append(np.expand_dims(seg_feature, axis=0))            # [1, seg_frames, n_mels]
            start_frame = end_frame + 1
            end_frame = start_frame + seg_frames
    else:
        fbanks = np.log10(fbanks)
        for _ in range(n_seg):
            mfcc = dct(fbanks, type=2, axis=1, norm='ortho')[:, :n_remain]  # DCT [frames, 13]
            seg_feature = mfcc[start_frame:end_frame]
            features.append(np.expand_dims(seg_feature, axis=0))            # [1, seg_frames, 13]
            start_frame = end_frame + 1
            end_frame = start_frame + seg_frames
    features = np.concatenate(features, axis=0)
    return features
