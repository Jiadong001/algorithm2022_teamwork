- [x] 频域特征提取
- [ ] 特征增强,归一化?
- [ ] 独立特征MFCC做Kmeans，GMM (flatten)
- [ ] 底层特征fbanks/stft做DNN

settings:
- window: hamming
  - duration: 40ms
  - shift: 20ms
- mel banks: 26
- mfcc: 13

one wav file:
- signal: 44100Hz*10s=441000
- stft: 499frames*400=200000(ignore high-freq)
- fbanks: 499frames*26=13000
- mfcc: 499frames*13=6500