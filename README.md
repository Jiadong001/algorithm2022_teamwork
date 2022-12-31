Steps:
- [x] 频域特征提取
- [ ] 特征增强,归一化?
- [x] 独立特征MFCC做 Kmeans (flatten all frames' mfcc)
- [x] 独立特征MFCC做 GMM , diag/full covariance
- [x] 做 DenseNN, 先MFCC，再底层特征fbanks
- [x] 做 CNN, 底层特征fbanks

Settings:
- window: hamming
  - duration: 40ms
  - shift: 20ms
- mel banks: 26/40
- mfcc: 13
- GMM n_component: 4/8
- metric:
  - accurary(recall)
  - confusion matrix

Features of one wav file:
- one frame, one feature -> flatten frames' features -> one feature
- one wav -> one feature
  - signal: 44100Hz*10s=441000
  - stft: 499frames\*400=200000(ignore high-freq), 250frames\*883=220000(no overlap, consider high-freq)
  - fbanks: 499frames\*26=13000, 499frames\*40=20000
  - mfcc: 499frames*13=6500
- one wav -> 10 segments(for example) -> 10 features
  - one feature -> model -> one score
  - 10 features -> model -> 10 scores -> vote
