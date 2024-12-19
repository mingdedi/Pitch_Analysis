import torch
import torch.nn as nn
import torch.optim as optim
from model import CREPEModel
from voiceDataLoader import MIR_1K_Dataset
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt
import librosa
#这个可以用来测试单个wav文件，注意wav文件波形

#data_inputs,_=librosa.load(f"../Data/MIR-1K/Test/MIR_1K_196_369Hz.wav",sr=None)

#data_inputs=librosa.load(f"./Test_wave/DR8_MMPM0_SA1.wav",sr=None)
data_inputs,_=librosa.load(f"./Test_wave/Zhao_Read.wav",sr=16000)

#查看波形变化，方便确定切片开始索引
# plt.plot(data_inputs)
# plt.show()
start_index=40882
data_inputs=data_inputs[start_index:start_index+1024]
data_inputs=torch.from_numpy(data_inputs).reshape(1,1,1024)

# Initialize model, loss function, and optimizer
model = CREPEModel()
model.load_state_dict(torch.load('CREPE.pth'))
model.eval()

with torch.no_grad():
    y_i=model(data_inputs)
    #计算预测音分和真实的音分
    c_hat=(y_i[0]*torch.arange(2000,9200,20)).sum()/y_i[0].sum()
    print(f"{start_index/16000}s到{(start_index+1024)/16000}s的基频是{2**(c_hat/1200)*10}Hz")
    plt.subplot(3,1,1)
    plt.plot(data_inputs[0][0])
    plt.subplot(3,1,2)
    #查看测试片段的傅里叶频谱
    plt.plot(np.abs(np.fft.fft(np.array(data_inputs[0][0])))[:len(np.array(data_inputs[0][0]))//2])
    plt.subplot(3,1,3)
    plt.plot(y_i[0])
    plt.show()

