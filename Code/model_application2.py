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
import soundfile as sf
#这个用来画出wav文件的pitch变化

data_inputs,_=librosa.load(f"C:\\Users\\13767\Desktop\Test_wave\\test2_Liu_Sing.wav",sr=16000)

# plt.plot(data_inputs[538:6881])
# plt.show()
# sf.write("C:\\Users\\13767\Desktop\Test_wave\\test2_Liu_Sing.wav",data_inputs[538:6881],16000)

model = CREPEModel()
model.load_state_dict(torch.load('./Code/CREPE.pth'))
model.eval()
f0_list=[]
for i in range(0,4864,128):
    with torch.no_grad():
        y_i=model(torch.from_numpy(data_inputs[i:i+1024]).reshape(1,1,-1))
        #计算预测音分，打印预测基频
        c_hat=(y_i[0]*torch.arange(2000,9200,20)).sum()/y_i[0].sum()
        f0_list.append(2**(c_hat/1200)*10)

praat_data=[
297.883190,
298.180135,
298.576830,
298.330718,
297.231945,
296.756736,
296.809241,
297.290587,
297.752686,
297.950754,
297.952736,
298.116441,
297.559440,
295.044332,
292.965836,
292.333128,
293.088478,
298.674825,
303.241253,
306.693332,
310.210623,
315.896903,
318.453209,
312.862125,
311.469217,
311.597866,
312.088771,
312.526614,
310.819562,
306.761121,
306.908501,
307.206837,
308.263301,
310.926489,
319.691063,
316.024543,
311.742704,
310.366349,
]        

plt.plot(f0_list,label='model f0', color='red')
plt.plot(praat_data,label='Praat f0', color='blue')
plt.ylim(75,500)
plt.legend()
plt.show()


