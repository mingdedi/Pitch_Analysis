import numpy as np
import matplotlib.pyplot as plt
import librosa
from scipy.interpolate import interp1d
import soundfile as sf
from tqdm import tqdm
import os
import random
#这个文件用来对MIR-1K中的数据进行切片，切成1024长度的片段，相当于是64ms

"""
    Convert a MIDI number to its corresponding frequency in Hz.

    Parameters:
    midi_number (float): The MIDI number to convert.

    Returns:
    float: The frequency in Hz corresponding to the given MIDI number.

    Notes:
    - The conversion is based on the standard MIDI tuning where A4 (MIDI number 69) is set to 440 Hz.
    - The formula used is: frequency = 440 * (2 ** ((midi_number - 69) / 12))
    """
def midi_to_frequency(midi_number):
    # A4 is MIDI number 69 and is set to 440 Hz
    a4_frequency = 440.0
    a4_midi_number = 69
    frequency = a4_frequency * (2 ** ((midi_number - a4_midi_number) / 12.0))
    # Return the calculated frequency value
    
    return frequency

def read_pv_file(file_path, time_interval=0.02):
    """
    Reads a pitch value (pv) file and extracts pitch values.

    Parameters:
    file_path (str): The path to the pv file to be read.
    time_interval (float): The time interval between consecutive pitch values in seconds. Default is 0.02 seconds.

    Returns:
    numpy.ndarray: An array of pitch values extracted from the file.

    Notes:
    - The pv file is expected to contain one pitch value per line.
    - The function assumes that the pitch values are spaced at regular time intervals.
    - The function skips any empty lines in the file.
    """
    # Open the file in read mode
    with open(file_path, 'r') as file:
        # Read all lines from the file
        lines = file.readlines()
    
    # Initialize empty lists to store timestamps and pitch values
    timestamps = []
    pitch_values = []

    # Start with the initial time interval
    current_time = 0.02
    # Iterate over each line in the file
    for line in lines:
        # Check if the line is not empty (ignoring whitespace)
        if line.strip():
            # Convert the stripped line to a float and append it to the pitch_values list
            pitch_value = float(line.strip())
            # Append the current time to the timestamps list
            timestamps.append(current_time)
            # Append the pitch value to the pitch_values list
            pitch_values.append(pitch_value)
            # Increment the current time by the specified time interval
            current_time += time_interval
    
    # Convert the pitch_values list to a numpy array and return it
    return np.array(pitch_values)

# 获取./Data/MIR-1K/Data/PitchLabel/目录下所有文件名，并去除文件格式名
pitch_label_files = [f.split('.')[0] for f in os.listdir(f"./Data/MIR-1K/Data/PitchLabel/") if os.path.isfile(os.path.join(f"./Data/MIR-1K/Data/PitchLabel/", f))]
j=0
#打乱顺序随机抽取文件名
random.shuffle(pitch_label_files)
# 使用tqdm显示进度条

#训练集样本数
TrainSet_number=9600
progress_bar = tqdm(total=TrainSet_number,desc="生成训练集")
for n in range(len(pitch_label_files)):
    file_name=pitch_label_files[n]
    # print(f"Split {file_name}.wav")#这个会和tqdm冲突的需要注意
    pitch_values = read_pv_file(f'./Data/MIR-1K/Data/PitchLabel/{file_name}.pv')
    waveform,_=librosa.load(f'./Data/MIR-1K/Data/Wavfile/{file_name}.wav',sr=None,mono=False)
    waveform=waveform[1]
    for i in range(0,len(waveform)-2048,1024):
        if(abs(pitch_values[int(i/320)]-pitch_values[int((1023+i)/320)])<20):
            f0=midi_to_frequency(pitch_values[int(i/320):int((1023+i)/320)+1].mean())
            if(f0>=80):
                filename = f"./Data/MIR-1K/Train/MIR_1K_{j}_{int(f0)}Hz.wav"
                sf.write(filename, waveform[i:i+1023+1], samplerate=16000)
                j+=1
                progress_bar.update(1)
                if(j>=TrainSet_number):
                    break
    if(j>=TrainSet_number):
        break

j=0
#验证集样本数
ValidationSet_number=3200
progress_bar = tqdm(total=ValidationSet_number,desc="生成验证集")
for n in range(n+1,len(pitch_label_files)):
    file_name=pitch_label_files[n]
    # print(f"Split {file_name}.wav")#这个会和tqdm冲突的需要注意
    pitch_values = read_pv_file(f'./Data/MIR-1K/Data/PitchLabel/{file_name}.pv')
    waveform,_=librosa.load(f'./Data/MIR-1K/Data/Wavfile/{file_name}.wav',sr=None,mono=False)
    waveform=waveform[1]
    for i in range(0,len(waveform)-2048,1024):
        if(abs(pitch_values[int(i/320)]-pitch_values[int((1023+i)/320)])<20):
            f0=midi_to_frequency(pitch_values[int(i/320):int((1023+i)/320)+1].mean())
            if(f0>=80):
                filename = f"./Data/MIR-1K/Validation/MIR_1K_{j}_{int(f0)}Hz.wav"
                sf.write(filename, waveform[i:i+1023+1], samplerate=16000)
                j+=1
                progress_bar.update(1)
                if(j>=ValidationSet_number):
                    break
    if(j>=ValidationSet_number):
        break

j=0
#测试集样本数
TestSet_number=3200
progress_bar = tqdm(total=TestSet_number,desc="生成测试集")
for n in range(n+1,len(pitch_label_files)):
    file_name=pitch_label_files[n]
    # print(f"Split {file_name}.wav")#这个会和tqdm冲突的需要注意
    pitch_values = read_pv_file(f'./Data/MIR-1K/Data/PitchLabel/{file_name}.pv')
    waveform,_=librosa.load(f'./Data/MIR-1K/Data/Wavfile/{file_name}.wav',sr=None,mono=False)
    waveform=waveform[1]
    for i in range(0,len(waveform)-2048,1024):
        if(abs(pitch_values[int(i/320)]-pitch_values[int((1023+i)/320)])<20):
            f0=midi_to_frequency(pitch_values[int(i/320):int((1023+i)/320)+1].mean())
            if(f0>=80):
                filename = f"./Data/MIR-1K/Test/MIR_1K_{j}_{int(f0)}Hz.wav"
                sf.write(filename, waveform[i:i+1023+1], samplerate=16000)
                j+=1
                progress_bar.update(1)
                if(j>=TestSet_number):
                    break
    if(j>=TestSet_number):
        break