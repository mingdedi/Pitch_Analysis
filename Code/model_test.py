import torch
import torch.nn as nn
import torch.optim as optim
from model import CREPEModel
from voiceDataLoader import MIR_1K_Dataset
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt

if __name__ == "__main__":
    Test_dataset = MIR_1K_Dataset('../Data/MIR-1K/Test')
    Test_dataloader = DataLoader(Test_dataset, batch_size=1, shuffle=True)
    # Initialize model, loss function, and optimizer
    model = CREPEModel()

    criterion = nn.CrossEntropyLoss()  # 修改为交叉熵损失函数

    model.load_state_dict(torch.load('CREPe.pth'))
    model.eval()
    i=0
    for Test_inputs, Test_targets in Test_dataloader:
        with torch.no_grad():
            #为了方便后续处理这里将其转换为numpy数组
            #同时注意360个节点代表的音分
            #节点输出代表的是为这个音分范围的”可能性”。
            y_i=model(Test_inputs)
            #计算预测音分和真实的音分
            c_true=(Test_targets[0]*torch.arange(2000,9200,20)).sum()/Test_targets[0].sum()
            c_hat=(y_i[0]*torch.arange(2000,9200,20)).sum()/y_i[0].sum()
            #print(2**(c_true/1200)*10,2**(c_hat/1200)*10)
            plt.subplot(2,2,1)
            plt.plot(Test_inputs[0][0])
            plt.subplot(2,2,2)
            plt.plot(np.abs(np.fft.fft(np.array(Test_inputs[0][0])))[:len(np.array(Test_inputs[0][0]))//2])
            plt.scatter(2**(c_true/1200)*10*0.064, np.abs(np.fft.fft(np.array(Test_inputs[0][0]))).max(),color='r',)
            plt.subplot(2,2,3)
            plt.plot(Test_targets[0])
            plt.plot(y_i[0])
            plt.show()
            
    print(i)

