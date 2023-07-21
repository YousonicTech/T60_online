# -*- coding: utf-8 -*-
"""
@file      :  realTimeEstimation.py
@Time      :  2022/10/21 14:13
@Software  :  PyCharm
@summary   :  This is for GUI
@Author    :  Bajian Xiang
"""

import pandas as pd

import model.FpnAttentionEncoder
# from model.FPNEncoder import FPN
from model.FpnAttentionEncoder import AttentionFPN
from wav2spec import wave2spec, wave2specWithMaskPrintTime, wave2specPrintTime
import torch
import time
import wave
import numpy as np
from pathlib import Path

OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH = OUTPUT_PATH / Path("save_model")


def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)


model_path = relative_to_assets('19epoch_attention_STI_FPN_Encoder.pt')

cut_time = 120
Test = False


def load_checkpoint(checkpoint_path=None, trained_epoch=None, model=None, device=None):
    save_model = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(save_model)

    return model


def second_to_time(a):
    if '.5' in str(a):
        ms = 5000
        a = int(a - 0.5)
    else:
        ms = 0000
        a = int(a)
    h = a // 3600
    m = a // 60 % 60
    s = a % 60
    return str("{:0>2}:{:0>2}:{:0>2}.{:0>4}".format(h, m, s, ms))


class ResultPrinter(object):
    """一个无情的结果打印机器"""

    def __init__(self):
        self.out_dict = {'start': [], 'end': [], 'STI': []}

    def print_results(self, out):
        already_time = self.get_start_len()
        for i in range(len(out)):
            start = (already_time + i) * 0.5
            end = start + 4
            print('time: [%.1f]s ~ [%.1f]s      STI: [%.4f] ' % (start, end, float(out[i])))
            self.out_dict['start'].append(start)
            self.out_dict['end'].append(end)
            self.out_dict['STI'].append(float(out[i]))

    def save_csv_result(self, save_file_path):
        out_data = pd.DataFrame.from_dict(self.out_dict)
        out_data.to_csv(save_file_path)
        print('Save csv already! Path: ', save_file_path)

    def get_start_len(self):
        return len(self.out_dict['start'])


def get_offline_res(wave_data, audio_time, framerate, model):
    print('Load Model...')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = AttentionFPN(num_blocks=[2, 4, 23, 3], num_classes=3, back_bone="resnet50", pretrained=False)

    # check model
    start_time = time.time()
    net = load_checkpoint(model_path, 99, net, device)
    print('Successfully Loaded model: {}'.format(model_path))
    print('Finished Initialization in {:.3f}s.\n'.format(time.time() - start_time))
    resultPrinter = ResultPrinter()
    net.to(device)

    if audio_time <= cut_time:
        # 25s 以内的音频
        print('Translating wav file into spectrogram...')
        if Test:
            images, empty_index, headLst, tailLst = wave2specPrintTime(wave_data, audio_time,
                                                                       framerate)  # tensor:[chunk, 3, 224, 224]
        else:
            images, empty_index = wave2spec(wave_data, audio_time, framerate, model)  # tensor:[chunk, 3, 224, 224]
        print('Available chunks in total: [ ', images.shape[0], ' ] \n')

        with torch.no_grad():
            net.eval()
            images_reshape = images.to(torch.float32).to(device)
            output_pts = net(images_reshape, [1 for _ in range(images_reshape.shape[0])])
            output_pts = output_pts.squeeze().tolist()

        if empty_index:
            for item in empty_index:
                output_pts.insert(item, 'NaN')
        print('STI Results:')
        if not Test:
            resultPrinter.print_results(out=output_pts)
        else:
            resultPrinter.print_results(output_pts, headLst, tailLst)

    else:
        print('Translating wav file into spectrogram... \nWav time is long, please wait with patience')
        # 对wave进行切分，每cut_time送一次进去, default = 25
        # 10s --  CUDA 占用 2399 Mib
        # 25s --  CUDA 占用 3277 Mib
        print('STI Results:')
        flag = False
        for x in range(0, len(wave_data), cut_time * framerate):

            if x + (cut_time + 4) * framerate > len(wave_data):
                partial_wav = wave_data[x:]
                images, empty_index = wave2spec(wave_data, audio_time, framerate)  # tensor:[chunk, 3, 224, 224]
                flag = True
                if len(partial_wav) <= framerate:
                    break
            else:
                partial_wav = wave_data[x:x + (cut_time + 4) * framerate]
                # print('send:', x, x + (cut_time+4)*framerate)
                images, empty_index = wave2spec(partial_wav, (cut_time + 4), framerate)  # tensor:[chunk, 3, 224, 224]

            # print('partial_wav, len', len(partial_wav))
            with torch.no_grad():
                net.eval()
                images_reshape = images.to(torch.float32).to(device)
                output_pts = net(images_reshape).tolist()
            if empty_index:
                for item in empty_index:
                    output_pts.insert(item, 'NaN')
            resultPrinter.print_results(out=output_pts)
            if flag:
                break
        print('Saving csv file, please wait.')
    return resultPrinter
