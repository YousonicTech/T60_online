# -*- coding: utf-8 -*-
"""
@file      :  wav2spec.py
@Time      :  2022/10/11 11:17
@Software  :  PyCharm
@summary   :
@Author    :  Bajian Xiang
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import splweighting
from torchvision import transforms
from torchvision.transforms import Normalize
from snr import wada_snr


chunk_length = 4
chunk_overlap = 1

skip_utter_portion = 0.5
cut_slice_portion = 0.75
transform = transforms.Compose([transforms.Resize([224, 224])])
normalize = Normalize(mean=[0.5], std=[0.5])

def All_Frequency_Spec(waveform, fs):
    waveform = torch.from_numpy(waveform).unsqueeze(0)
    raw_signal = waveform[0]
    nfft = 512

    spec, freq, t, _ = plt.specgram(raw_signal, NFFT=nfft, Fs=fs, window=np.hanning(M=nfft), scale_by_freq=True)
    plt.clf()

    spec = 10 * np.log10(spec)
    spec = np.clip(spec, -1e8, 1e8)

    return spec, freq, t


def imageTransform(temp_image, model):
    if 'attention' or 'Attention' in model:
        temp_image = temp_image.unsqueeze(0)
    else:
        temp_image = temp_image[temp_image.shape[0] // 2:].unsqueeze(0)
    return temp_image


def imageListTransform(images):
    images = transform(images).unsqueeze(1)  # [32, 1, 224, 224]
    for i in range(images.shape[0]):
        # 先归一化到[0, 1], 相当于Totensor()
        temp_norm = images[i][0]
        temp_norm = (temp_norm - temp_norm.min()) / (temp_norm.max() - temp_norm.min())
        images[i][0] = temp_norm
    # Normalize到[-1, 1]
    images = normalize(images)
    # print(images.shape)
    images = images.repeat(1, 3, 1, 1)
    return images


def wave2spec(wave_data, audio_time, framerate, model, ifLess4=False):
    # cut wave
    if audio_time < 4:
        raise ValueError('Audio time less than 4s, cannot infer its STI!')
    if audio_time < 4.5:
        cut_parameters = [0.0]
    else:
        available_part_num = (audio_time - chunk_length + 1) // chunk_overlap
        cut_parameters = [i * 0.5 for i in range(int(available_part_num))]

    image_list = []
    empty_index = []
    cut_slice_time = int(framerate * chunk_length * cut_slice_portion) # 3s
    cat_head = None # 备拼接的部分
    for index, t in enumerate(cut_parameters):
        # TODO 还没有对双声道进行处理

        start = int(t * framerate)
        end = int((t + chunk_length) * framerate)
        audio_chunk = wave_data[start:end]

        if ifLess4:
            chunk_a_weighting = splweighting.weight_signal(audio_chunk, framerate)  # shape=64000
            chunk_result, _, _ = All_Frequency_Spec(chunk_a_weighting, framerate)  # shape=166

            image = torch.from_numpy(chunk_result)  # shape=[257, 166]
            image = imageTransform(image, model)  # shape=[1, 129, 166] (same as server)
            image_list.append(image)
            continue

        # SNR >= 5 再考虑Nan的问题
        snr = wada_snr(audio_chunk)
        print('snr:', snr)
        # if snr >= 8:
        #     # 处理NaN
        #     utter_time_3_head = get_utter_time(audio_chunk[:cut_slice_time], framerate)
        #     utter_time_1_tail = get_utter_time(audio_chunk[cut_slice_time:], framerate)
        #
        #     if utter_time_3_head > 2:
        #         cat_head = audio_chunk[:cut_slice_time] # 备用片段
        #
        #     # 当前slice中发声片段小于2s
        #     if utter_time_3_head + utter_time_1_tail < 2:
        #         if utter_time_1_tail < chunk_length * (1-cut_slice_portion) * 0.5:
        #             # 最后1s内人声小于0.5s，输出NaN
        #             empty_index.append(index)
        #             continue
        #         else:
        #             # 最后1s内人声大于0.5s
        #             if isinstance(cat_head, bool):
        #                 # 没有备用片段，输出NaN
        #                 empty_index.append(index)
        #                 continue
        #             else:
        #                 # 有备用片段，拼接后进行STI计算
        #                 audio_chunk = np.append(cat_head, audio_chunk[cut_slice_time:])
        #     # 当前slice中发声片段>=2s
        #     else:
        #         if utter_time_1_tail < chunk_length * (1-cut_slice_portion) * 0.5 and len(image_list) != 0:
        #             # 最后1s内人声小于0.5 s，且非第一个片段，输出NaN
        #             empty_index.append(index)
        #             continue

        if None in audio_chunk:
            empty_index.append(index)
            continue

        chunk_a_weighting = splweighting.weight_signal(audio_chunk, framerate)  # shape=64000
        chunk_result, _, _ = All_Frequency_Spec(chunk_a_weighting, framerate)  # shape=166

        image = torch.from_numpy(chunk_result)  # shape=[257, 166]
        image = imageTransform(image, model)  # shape=[1, 129, 166] (same as server)
        image_list.append(image)
    # print('shape in image_lst:',[i.shape for i in image_list])

    if len(image_list) == 1:
        return imageListTransform(image_list[0]), empty_index

    image_list = torch.cat(image_list, dim=0)  # tensor -- shape of [32, 129, 166]
    image_list = imageListTransform(image_list)  # tensor -- shape of [32, 3, 224, 224]

    return image_list, empty_index


def wave2specPrintTime(wave_data, audio_time, framerate):
    # cut wave
    available_part_num = (audio_time - chunk_length) // chunk_overlap
    if available_part_num == 0:
        raise ValueError('Audio time less than 4s, cannot infer its STI!')
    else:
        cut_parameters = [i * 0.5 for i in range(int(available_part_num))]

    image_list = []
    empty_index = []
    head_3_lst = []
    tail_1_lst = []
    cut_slice_time = int(framerate * chunk_length * cut_slice_portion) # 3s
    cat_head = None # 备拼接的部分
    for index, t in enumerate(cut_parameters):

        start = int(t * framerate)
        end = int((t + chunk_length) * framerate)
        audio_chunk = wave_data[start:end]

        # 处理NaN
        utter_time_3_head = get_utter_time(audio_chunk[:cut_slice_time], framerate)
        utter_time_1_tail = get_utter_time(audio_chunk[cut_slice_time:], framerate)
        head_3_lst.append(utter_time_3_head)
        tail_1_lst.append(utter_time_1_tail)

        if utter_time_3_head > 2:
            cat_head = audio_chunk[:cut_slice_time] # 备用片段

        # 当前slice中发声片段小于2s
        if utter_time_3_head + utter_time_1_tail < 2:
            if utter_time_1_tail < chunk_length * (1-cut_slice_portion) * 0.5:
                # 最后1s内人声小于0.5s，输出NaN
                empty_index.append(index)
                continue
            else:
                # 最后1s内人声大于0.5s
                if isinstance(cat_head, bool):
                    # 没有备用片段，输出NaN
                    empty_index.append(index)
                    continue
                else:
                    # 有备用片段，拼接后进行STI计算
                    audio_chunk = np.append(cat_head, audio_chunk[cut_slice_time:])
        # 当前slice中发声片段>=2s
        else:
            if utter_time_1_tail < chunk_length * (1-cut_slice_portion) * 0.5 and len(image_list) != 0:
                # 最后1s内人声小于0.5s，且非第一个片段，输出NaN
                empty_index.append(index)
                continue

        if None in audio_chunk:
            empty_index.append(index)
            continue

        chunk_a_weighting = splweighting.weight_signal(audio_chunk, framerate)  # shape=64000
        chunk_result, _, _ = All_Frequency_Spec(chunk_a_weighting, framerate)  # shape=166

        image = torch.from_numpy(chunk_result)  # shape=[257, 166]
        image = imageTransform(image)  # shape=[1, 129, 166] (same as server)
        image_list.append(image)
    # print('shape in image_lst:',[i.shape for i in image_list])

    image_list = torch.cat(image_list, dim=0)  # tensor -- shape of [32, 129, 166]
    image_list = imageListTransform(image_list)  # tensor -- shape of [32, 3, 224, 224]

    return image_list, empty_index, head_3_lst, tail_1_lst


def wave2specWithMaskPrintTime(wave_data, mask_data, audio_time, framerate):
    # cut wave
    available_part_num = (audio_time - chunk_length) // chunk_overlap
    if available_part_num == 0:
        raise ValueError('Audio time less than 4s, cannot infer its STI!')
    else:
        cut_parameters = [i * 0.5 for i in range(int(available_part_num))]

    image_list = []
    empty_index = []
    head_3_lst = []
    tail_1_lst = []
    cut_slice_time = int(framerate * chunk_length * cut_slice_portion) # 3s
    cat_head = None # 备拼接的部分

    for index, t in enumerate(cut_parameters):

        start = int(t * framerate)
        end = int((t + chunk_length) * framerate)
        audio_chunk = wave_data[start:end]
        audio_chunk_mask = mask_data[start:end]

        # 处理NaN
        utter_time_3_head = sum(audio_chunk_mask[:cut_slice_time]) / 16000
        utter_time_1_tail = sum(audio_chunk_mask[cut_slice_time:]) / 16000
        head_3_lst.append(utter_time_3_head)
        tail_1_lst.append(utter_time_1_tail)

        if utter_time_3_head > 2:
            cat_head = audio_chunk[:cut_slice_time] # 备用片段

        # 当前slice中发声片段小于2s
        if utter_time_3_head + utter_time_1_tail < 2:
            if utter_time_1_tail < chunk_length * (1-cut_slice_portion) * 0.5:
                # 最后1s内人声小于0.5s，输出NaN
                empty_index.append(index)
                continue
            else:
                # 最后1s内人声大于0.5s
                if isinstance(cat_head, bool):
                    # 没有备用片段，输出NaN
                    empty_index.append(index)
                    continue
                else:
                    # 有备用片段，拼接后进行STI计算
                    audio_chunk = np.append(cat_head, audio_chunk[cut_slice_time:])
        # 当前slice中发声片段>=2s
        else:
            if utter_time_1_tail < chunk_length * (1-cut_slice_portion) * 0.5 and len(image_list) != 0:
                # 最后1s内人声小于0.5s，且非第一个片段，输出NaN
                empty_index.append(index)
                continue


        chunk_a_weighting = splweighting.weight_signal(audio_chunk, framerate)  # shape=64000
        chunk_result, _, _ = All_Frequency_Spec(chunk_a_weighting, framerate)  # shape=166

        image = torch.from_numpy(chunk_result)  # shape=[257, 166]
        image = imageTransform(image)  # shape=[1, 129, 166] (same as server)
        image_list.append(image)
    # print('shape in image_lst:',[i.shape for i in image_list])

    image_list = torch.cat(image_list, dim=0)  # tensor -- shape of [32, 129, 166]
    image_list = imageListTransform(image_list)  # tensor -- shape of [32, 3, 224, 224]

    return image_list, empty_index, head_3_lst, tail_1_lst