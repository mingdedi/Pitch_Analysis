import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import random
#抽取训练集中的文件，使用DFT进行分析，查看数据集是否正确标注
#需要注意的是DFT分析只是看一个大概，真实基频数据以标注为准，要不为啥不用开窗傅里叶进行基频检测呢？
# 不过貌似频域上的分析又是另一种深度学习模型了
dirlist=os.listdir("./Data/MIR-1K/Train/")
random.shuffle(dirlist)
j=0

for filename in dirlist[1:11]:
    file_path = os.path.join("./Data/MIR-1K/Train/", filename)
    waveform, _ = librosa.load(file_path, sr=None)
    
    Fre_waveform = np.fft.fft(waveform)
    Fre_magnitudes = np.abs(Fre_waveform)[:int(len(Fre_waveform)/2)]
    frequency = float(filename.split('_')[-1].replace('Hz.wav', ''))
    # 会自动过滤掉height小于5的峰值
    peaks_indices,_= find_peaks(Fre_magnitudes, height=5)  
    print(frequency)
    print(peaks_indices/0.064)
    plt.plot(Fre_magnitudes)
    plt.scatter(peaks_indices,Fre_magnitudes[peaks_indices])
    plt.show()

print(j)

