# -*- coding: utf-8 -*-
"""
@file      :  Trim_model.py
@Time      :  2022/10/14 12:29
@Software  :  PyCharm
@summary   :  input the origin of STI_FPN MODEl, delete the decoder part
@Author    :  Bajian Xiang
"""
import torch

epoch = 68
origin_path = "/data2/xbj/0930_STI_catTIMIT_noise_Dataset_Detection/save_model" \
              "/0924_STI_FPN_lre5_alpha1_beta05_Step05per25/t60_predict_model_" + str(epoch) + "_fullT"\
              "60_rir_timit_noise.pt"
save_path = '/data2/xbj/Test_pipeline_STI/cut_model/' + str(epoch) + 'epoch_STI_FPN_Encoder.pt'

if __name__ == "__main__":
    x = torch.load(origin_path)
    y = x['model']
    deletes = []
    for key in y.keys():
        if "back_bone" not in key and "t60" not in str(key):
            deletes.append(key)
    for layer in deletes:
        if layer in y.keys():
            del y[layer]
    torch.save(y, save_path)
