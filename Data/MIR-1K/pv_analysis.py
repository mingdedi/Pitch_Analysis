import numpy as np
import matplotlib.pyplot as plt
import librosa
from scipy.interpolate import interp1d

#这个是用来测试高音值划分的
def midi_to_frequency(midi_number):
    # A4 is MIDI number 69 and is set to 440 Hz
    a4_frequency = 440.0
    a4_midi_number = 69
    frequency = a4_frequency * (2 ** ((midi_number - a4_midi_number) / 12.0))
    return frequency

"""
    读取 .pv 文件，假设每行只有一个音高值，时间戳通过固定的时间间隔计算。
    
    :param file_path: .pv 文件的路径
    :param time_interval: 每个音高值之间的时间间隔，默认为 0.01 秒
    :return: 时间戳数组和音高值数组
    """
def read_pv_file(file_path, time_interval=0.02):
    
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # 初始化时间戳和音高值列表
    timestamps = []
    pitch_values = []
    
    # 计算时间戳并读取音高值
    current_time = 0.02
    for line in lines:
        if line.strip():
            pitch_value = float(line.strip())
            timestamps.append(current_time)
            pitch_values.append(pitch_value)
            current_time += time_interval  # 更新时间戳
    
    return np.array(timestamps), np.array(pitch_values)


# 示例：读取 .pv 文件
timestamps, pitch_values = read_pv_file(f'./Data/PitchLabel/amy_1_01.pv')
waveform,_=librosa.load(f'./Data/Wavfile/abjones_2_02.wav',sr=None,mono=False)
#注意这里一定要提取正确的声道，一个声道是唱歌配乐，另一个是人声
waveform=waveform[1]
fre_values=midi_to_frequency(pitch_values)
# # 打印前10个时间点和音高值
print("Timestamps:", timestamps[-1])
print("Wav Times:",len(waveform))
# 创建时间轴
waveform_time = np.linspace(0, 1, len(waveform))
fre_time = np.linspace(0, 1, len(fre_values))

# 插值pitch_values以匹配waveform的时间轴
interp_func = interp1d(fre_time, fre_values, kind='linear', fill_value='extrapolate')
aligned_pitch_values = interp_func(waveform_time)
plt.plot(waveform_time, waveform, label='Waveform')
plt.plot(waveform_time, aligned_pitch_values/840.,label="aligned_pitch_values")
plt.show()