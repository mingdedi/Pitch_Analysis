import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from voiceDataLoader import PitchDataset
import os
import torch
from torch.utils.data import Dataset, DataLoader
import librosa
from tqdm import tqdm
#这个代码是是使用傅里叶变换对音频进行基频检测的代码
#进行训练前的最终测试
if __name__ == "__main__":
    #测试数据集
    data_dir = '../Data/'
    dataset = PitchDataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
 
    #数据加载测试，通过傅里叶寻找基频，然后进行对比
    batch = iter(dataloader)
    waveforms, y_i = next(batch)
    y_i=np.array(y_i)
    Fre_y=np.fft.fft(np.array(waveforms[0]))
    # 只取正频率部分
    positive_magnitudes = np.abs(Fre_y)[:len(Fre_y)//2]
    # 寻找峰值
    peaks, _ = find_peaks(positive_magnitudes, height=10)#会自动过滤掉height的峰值
    print(peaks*1/0.064)#打印各个峰值
    c_true=(y_i[0]*np.arange(2000,9200,20)).sum()/y_i[0].sum()
    print(2**(c_true/1200)*10)
