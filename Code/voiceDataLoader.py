import os
import torch
from torch.utils.data import Dataset, DataLoader
import librosa
from tqdm import tqdm   
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import find_peaks

class PitchDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.file_list = os.listdir(data_dir)
        self.c_i=np.arange(2000,9200,20)
        # 这个其实是模型输出360个节点，每个节点代表的音分输出，
        # 然后每个节点的输出代表为这个音分的可能性（置信度）

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_name = self.file_list[idx]
        file_path = os.path.join(self.data_dir, file_name)
        
        #加载音频文件
        waveform, sample_rate = librosa.load(file_path, sr=None)
        waveform=torch.from_numpy(waveform.reshape(1,1024))
        #通过文件名确定标注的基频
        frequency = float(file_name.split('_')[-1].replace('Hz.wav', ''))
        #计算目标值
        c_true=1200 * np.log2(frequency / 10)
        y_i=np.exp(-(self.c_i-c_true)**2/1250)
        return waveform, y_i

class MIR_1K_Dataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.file_list = os.listdir(data_dir)
        self.c_i=np.arange(2000,9200,20)
        # 这个其实是模型输出360个节点，每个节点代表的音分输出，
        # 然后每个节点的输出代表为这个音分的可能性（置信度）

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_name = self.file_list[idx]
        file_path = os.path.join(self.data_dir, file_name)
        
        #加载音频文件
        waveform, sample_rate = librosa.load(file_path, sr=None)
        waveform=torch.from_numpy(waveform.reshape(1,1024))
        #通过文件名确定标注的基频
        frequency = float(file_name.split('_')[-1].replace('Hz.wav', ''))
        #计算目标值
        c_true=1200 * np.log2(frequency / 10)
        y_i=np.exp(-(self.c_i-c_true)**2/1250)
        return waveform, y_i


if __name__ == "__main__":
    #测试数据集
    data_dir = './Data/MIR-1K/Train'
    dataset = MIR_1K_Dataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    #一batch数据加载测试
    batch = iter(dataloader)
    waveforms, y_i = next(batch)
    print(waveforms.shape)#查看数据形状
    print(y_i.shape)

    #傅里叶测试，只是当作参考，省着去人眼数波形数量了
    Fre_y=np.fft.fft(waveforms[0][0])
    # 只取正频率部分
    positive_magnitudes = np.abs(Fre_y)[:len(Fre_y)//2]
    #寻找波峰
    peaks, _ = find_peaks(positive_magnitudes, height=10)#会自动过滤掉height的峰值
    #打印各个峰值的索引，其实可以理解为打印强度为峰值的频率分量
    print(peaks*1/0.064)

    plt.subplot(2,1,1)
    #画出一个波形图
    plt.plot(waveforms[0][0])
    plt.subplot(2,1,2)
    plt.plot(y_i[0])
    plt.show()
    #完整加载测试
    # for batch in tqdm(dataloader):
    #     waveforms, f0 = batch
