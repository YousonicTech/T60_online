# -*- coding: utf-8 -*-
"""
@file      :  dataloader.py
@Time      :  2022/11/22 12:44
@Software  :  PyCharm
@summary   :
@Author    :  Bajian Xiang
"""
import numpy as np
import torch
import splweighting
import scipy
from torch.utils.data import DataLoader, Dataset
from wav2spec import All_Frequency_Spec, imageListTransform
from webrtc_vad import get_utter_time
from snr import wada_snr
from acoustics.signal import bandpass
from torchvision import transforms
from moviepy.editor import AudioFileClip
import torch
import torch.nn as nn
import torchaudio.functional as F
import librosa
from torchaudio.transforms import Spectrogram, InverseSpectrogram, Resample, AmplitudeToDB, TimeMasking
import acoustics
from acoustics.bands import (_check_band_type, octave_low, octave_high, third_low, third_high)
from acoustics.signal import bandpass,highpass
import matplotlib.pyplot as plt
from torchvision.transforms import Normalize
def imageTransform(temp_image):
    temp_image = temp_image.unsqueeze(0)
    return temp_image

def Filter_Downsample_Spec(waveform, fs):
    waveform = torch.from_numpy(waveform).unsqueeze(0)
    channel = len(waveform)
    nframes = len(waveform[0])
    raw_signal = waveform.T
    bands = acoustics.signal.OctaveBand(fstart=125, fstop=4000, fraction=1).nominal

    band_type = _check_band_type(bands)
    # print(band_type, end=', ')

    if band_type == 'octave':
        low = octave_low(bands[0], bands[-1])
        high = octave_high(bands[0], bands[-1])
    elif band_type == 'third':
        low = third_low(bands[0], bands[-1])
        high = third_high(bands[0], bands[-1])

    #for nch in range(channel):
    nch = 0
    filtered_signal = np.zeros((bands.size, nframes))
    for band in range(bands.size):
        # 信号，频率下限，频率上限， 采样率
        # print("low:",low[band],"high:",high[band])
        filtered_signal[band] = bandpass(raw_signal[:, nch], low[band], high[band], fs, order=bands.size)
        filtered_signal[band] = highpass(filtered_signal[band],low[band],fs,order=6)
        #plt.figure()
        #plt.plot(filtered_signal[band])
        #plt.clf()

    downsample_signal = []
    for i in range(len(bands)):
        temp_rate = 2 * high[i]
        temp_data = filtered_signal[i]
        number_of_samples = round(len(temp_data) * float(temp_rate) / fs)
        downsample_signal.append(scipy.signal.resample(temp_data, number_of_samples))
        #plt.figure()
        #plt.plot(downsample_signal[i])
        #plt.clf()

    spectrograms = []
    nfft = 256
    fs_each_band = []
    time = []

    for i in range(len(bands)):
        spec, freq, t, _ = plt.specgram(downsample_signal[i], NFFT=nfft, Fs=high[i] * 2, window=np.hanning(M=nfft),
                                        scale_by_freq=True)
        # print(spec.shape)
        plt.clf()
        spec = 10 * np.log10(spec)
        high_index = round((high[i] - low[i]) / (freq[1] - freq[0])) - 1
        spectrograms.append(spec)
        fs_each_band.append(freq)
        time.append(t)
    #del spec,freq,t
    #gc
    return spectrograms, fs_each_band, time

class PipeLineNew(nn.Module):
    """新的数据集，500, 1k, 2k, 4k分别占一个维度,直接存torch.complex"""
    def __init__(self):
        super().__init__()
        self.n_fft = 512
        self.n_hop = 256
        self.freq = 16000
        self.spec = Spectrogram(n_fft=self.n_fft, hop_length=self.n_hop, power=None)
        self.inverseSpec = InverseSpectrogram(n_fft=self.n_fft, hop_length=self.n_hop)

    def forward(self, chunk, fs):
        # 送进来的应该是一个4s的片段,类型为numpy array,先滤波,再downsample,最后画语谱图
        chunk = torch.from_numpy(chunk).squeeze()
        if fs != self.freq:
            chunk = F.resample(torch.from_numpy(chunk), orig_freq=fs, new_freq=self.freq)
        chunk = chunk.float()
        save_tensor = []
        for freq, fft in zip([500, 1000, 2000, 4000], [256, 512, 1024, 2048]):
            # 1. 滤波
            temp_chunk = bandpass(chunk, int(freq / 1.414), int(freq * 1.414), fs)
            # 2. downsample
            temp_chunk = F.resample(torch.from_numpy(temp_chunk), orig_freq=fs, new_freq=int(freq * 1.414 * 2))
            # 3. 语谱图
            spec = Spectrogram(n_fft=fft, hop_length=fft//2, power=None)
            temp_chunk = spec(temp_chunk).squeeze()
            save_tensor.append(temp_chunk)
        return save_tensor

def replace_nan(a):
    if torch.any(torch.isnan(a)):
        a = torch.where(torch.isnan(a), torch.full_like(a, 0), a)
    if torch.any(torch.isinf(a)):
        a = torch.where(torch.isinf(a), torch.full_like(a, 0), a)
    a = torch.clamp(a, 1e-10, 1e10)
    return a

def get_amplitude(aa):
    amp = replace_nan(aa.real ** 2 + aa.imag ** 2).sqrt().log10()
    return amp

class DatasetSlice(Dataset):

    def __init__(self, waves, fs, overlap, ifVad=True,which_freq=2):
        self.waves = waves
        self.which_freq = which_freq
        self.fs = fs
        self.overlap = overlap
        self.chunk_length = 4
        self.total_cut_num = int(((len(self.waves)/self.fs)-self.chunk_length) // self.overlap)
        # for vad
        self.ifVad = ifVad
        self.snr_threshold = 8
        self.cut_slice_portion = 0.75
        self.cut_slice_time = int(self.fs * 4 * self.cut_slice_portion) # 3s
        self.cat_head = False
        self.pip = PipeLineNew()
        self.transform = transforms.Compose([transforms.Resize([224, 224])])
        self.normalize = Normalize(mean=[0.5], std=[0.5])
        
    def __len__(self):
        return self.total_cut_num

    def __getitem__(self, idx):
        start = idx * self.fs
        end = start + self.chunk_length * self.fs
        audio_chunk = self.waves[start: end]

        if self.ifVad:
            snr = wada_snr(audio_chunk)
            if snr >= self.snr_threshold:
                utter_time_3_head = get_utter_time(audio_chunk[:self.cut_slice_time], self.fs)
                utter_time_1_tail = get_utter_time(audio_chunk[self.cut_slice_time:], self.fs)

                if utter_time_3_head + utter_time_1_tail < 2:
                    if utter_time_1_tail < self.chunk_length * (1 - self.cut_slice_portion) * 0.5:
                        return idx
                    else:
                        if isinstance(self.cat_head, bool):
                            return idx
                        else:
                            audio_chunk = np.append(self.cat_head, audio_chunk[self.cut_slice_time:])
                else:
                    if utter_time_1_tail < self.chunk_length * (1 - self.cut_slice_portion) * 0.5 and idx != 0:
                        return idx

        # chunk_result = self.pip(audio_chunk, self.fs)  # [tensor1, tensor2, tensor3, tensor4]:List
        # chunk_result = [self.transform(get_amplitude(x).unsqueeze(0)) for x in chunk_result]
        # image = torch.cat(chunk_result)  # shape=[257, 166]
        
        chunk_a_weighting = splweighting.weight_signal(audio_chunk, self.fs)
        chunk_result, _, _ = Filter_Downsample_Spec(chunk_a_weighting, self.fs)
        temp_image = chunk_result[self.which_freq][chunk_result[self.which_freq].shape[0] // 2:]
        temp_image = torch.from_numpy(temp_image)
        
        temp_image = torch.unsqueeze(temp_image, dim=0)
        
        temp_image = self.transform(temp_image)
        
                # 先归一化到[0, 1], 相当于Totensor()
        temp_norm = temp_image[0]
        temp_norm = (temp_norm - temp_norm.min()) / (temp_norm.max() - temp_norm.min())
        temp_image[0] = temp_norm
            # Normalize到[-1, 1]
        temp_image = self.normalize(temp_image)
        # print(temp_image.shape)
        # chunk_result = [self.transform(x) for x in chunk_result]
        
        # image = [torch.from_numpy(m.astype("float32")) for m in chunk_result]
        
        # chunk_result = [self.transform(get_amplitude(x).unsqueeze(0)) for x in chunk_result]
        #chunk_result = np.array(chunk_result)
        #chunk_result = torch.from_numpy(chunk_result)
        #image = torch.cat(chunk_result)  # shape=[257, 166]
        #image = torch.from_numpy(image)
        #image = imageTransform(image)           # shape=[1, 257, 166]
        return temp_image


def vad_collate_fn(batch):
    # batch -> list: [tensor(1, 257, 166), 1, tensor(1, 257, 166), tensor(1, 257, 166), 4, ...]
    nan_lst = []
    image_lst = []
    for item in batch:
        
        if isinstance(item, int):
            nan_lst.append(item)
        else:
            image_lst.append(item)
    if image_lst:
        #print(image_lst)
        image_lst = torch.cat(image_lst, dim=0)
        image_lst = imageListTransform(image_lst)
        return image_lst, nan_lst
    else:
        return False, nan_lst
   
class VideoProcessing(object):
    def __init__(self, path):
        self.path = path
        self.audio = None
        self.audio_time = None
        self.fs = None
        self.get_wav_from_video()

    def get_wav_from_video(self):
        # my_audio_clip = AudioFileClip(self.path)
        # audio_data = my_audio_clip.to_soundarray()
        audio_data, fs = librosa.load(self.path, sr=16000, mono=False)
        
        framerate = fs
        if framerate != 16000:
            audio_data = scipy.signal.resample(audio_data, int(len(audio_data) / framerate * 16000))
        nframes, nchannels = audio_data.shape,1
        if nchannels == 2:
            audio_data = audio_data.T[0]
        if isinstance(audio_data[0], np.float32):
            audio_data = np.array(audio_data * 32768.0, dtype=np.int16)
        elif isinstance(audio_data[0], np.int32):
            audio_data = (audio_data >> 16).astype(np.int16)
        audio_time = len(audio_data) / 16000
        self.audio, self.audio_time, self.fs = audio_data, audio_time, 16000

if __name__ == "__main__":
    
    video_path = '/root/Wzd/project_t60/test.mp4'
    video = VideoProcessing(video_path)
    
    # video = VideoProcessing(video_path)
    
    #x, fs = librosa.load('/root/Wzd/project_t60/YQH207_YQH207-ch1_上海话女声-1_a003-190-200_30dB.wav', mono=False, sr=None)
    path = "/root/Wzd/project_t60/test.wav"
    audio_data, fs = librosa.load(path, sr=16000, mono=False)
    
    print("123")
    # ds = DatasetSlice(video.audio, video.fs, 1, False)
    # batch_size = 1
    # videoloader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=vad_collate_fn, drop_last=False)
    # #waves, fs, overlap, ifVad=True
    # for i, datas in enumerate(videoloader):
    #     images, temp_vad_lst = datas
    #     print("i",images.shape,temp_vad_lst)
    