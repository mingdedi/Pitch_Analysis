import numpy as np
import scipy.signal
import soundfile as sf
import librosa
from tqdm import tqdm
#这个代码用来生成数据集的，这个信号十分简单就是简单的几个正弦波相加
#这个数据集是用来测试用的，几乎没有实际意义
def generate_synthetic_voice(f0,f1,f2,f3,sample_num=1024, sr=16000,noise_generate=False):
    t = np.linspace(0, sample_num/sr,sample_num, endpoint=False)
    
    # 基频信号
    f0_signal = 0.5 * np.sin(2 * np.pi * f0 * t)
    #共振峰信号
    f1_signal = 0.16 * np.sin(2 * np.pi * f1 * t)
    f2_signal = 0.16 * np.sin(2 * np.pi * f2 * t)
    f3_signal = 0.16 * np.sin(2 * np.pi * f3 * t)
    #噪声信号
    if(noise_generate==True):
        noise = 0.02 * np.random.uniform(-1, 1, sample_num)
    else:
        noise=0.02 *np.sin(2 * np.pi * f0 * t)

    # 合成信号
    signal = f0_signal + f1_signal + f2_signal + f3_signal+noise

    return signal

def save_wav_file(signal, sr, filename):
    sf.write(filename, signal, sr)

#开始生成训练集
print("开始生成训练集")
for i in tqdm(range(9600)):
    #随机挑选基频，共振峰范围
    f0 = np.random.uniform(80, 255)  # 人声基频范围
    f1 = np.random.uniform(300, 800)  # 第一共振峰范围eeeee
    f2 = np.random.uniform(800, 1500)  # 第二共振峰范围
    f3 = np.random.uniform(1500, 2500)  # 第三共振峰范围
    signal = generate_synthetic_voice(f0,f1,f2,f3,noise_generate=True)
    filename = f"./Train/synthetic_voice_{i}_{int(f0)}Hz.wav"
    sf.write(filename, signal, sr=16000)
    #用来查看生成的文件名，注意大量生成的时候记得关，要不进度条会乱码
    #print(f"Generated {filename}")

#开始生成训练集
print("开始生成验证集")
for i in tqdm(range(3200)):
    #随机挑选基频，共振峰范围
    f0 = np.random.uniform(80, 255)  # 人声基频范围
    f1 = np.random.uniform(300, 800)  # 第一共振峰范围eeeee
    f2 = np.random.uniform(800, 1500)  # 第二共振峰范围
    f3 = np.random.uniform(1500, 2500)  # 第三共振峰范围
    signal = generate_synthetic_voice(f0,f1,f2,f3,noise_generate=True)
    filename = f"./Validation/synthetic_voice_{i}_{int(f0)}Hz.wav"
    sf.write(filename, signal, sr=16000)
    #用来查看生成的文件名，注意大量生成的时候记得关，要不进度条会乱码
    #print(f"Generated {filename}")

#开始生成训练集
print("开始生成测试集")
for i in tqdm(range(3200)):
    #随机挑选基频，共振峰范围
    f0 = np.random.uniform(80, 255)  # 人声基频范围
    f1 = np.random.uniform(300, 800)  # 第一共振峰范围eeeee
    f2 = np.random.uniform(800, 1500)  # 第二共振峰范围
    f3 = np.random.uniform(1500, 2500)  # 第三共振峰范围
    signal = generate_synthetic_voice(f0,f1,f2,f3,noise_generate=True)
    filename = f"./Test/synthetic_voice_{i}_{int(f0)}Hz.wav"
    sf.write(filename, signal, sr=16000)
    #用来查看生成的文件名，注意大量生成的时候记得关，要不进度条会乱码
    #print(f"Generated {filename}")