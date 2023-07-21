import os
import sys
import time
import csv
import pandas as pd
import torch
import time
import wave
import numpy as np
import scipy
from torch.utils.data import DataLoader
from fastapi import FastAPI
from wav2spec import imageListTransform
# from starlette.requests import Request

onnx_500hz_path = "./Model/500hz_model_nonoise.onnx"
onnx_1khz_path = "Model/1khz_model.onnx"
onnx_2khz_path = "./Model/2khz_model.onnx"

import librosa
import onnxruntime as ort
import base64

ort_session_500hz = ort.InferenceSession(onnx_500hz_path)
ort_session_1khz = ort.InferenceSession(onnx_1khz_path)
ort_session_2khz = ort.InferenceSession(onnx_2khz_path)

from fastapi import FastAPI, File, UploadFile, Form
from scipy.io.wavfile import write
from dataloader_t60 import DatasetSlice, vad_collate_fn
# import librosa
from starlette.responses import FileResponse
from moviepy.editor import AudioFileClip
from tqdm import tqdm
# from Model.attentionFPNEncoder import FPN

number = 0

save_dir = "./Save_folder/"
output__dir = "./Pro_folder/"
zip_dir = './zip_folder/'
oss_zip_folder = './oss_zip_folder/'
log_dir = "/root/Wzd/project/log/"


app = FastAPI()
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import zipfile
import oss2
import glob
import argparse

overlap = 1
batch_size = 4
average_iterval = 5
ifVad = False #True:没有声音的片段进这个nanlist；False：没有声音的片段也进image_lst

from fastapi.responses import HTMLResponse

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--port', type=int, default = 60)
def upload_oss(filename, file):
    access_key_id = 'LTAI5tAzLJwWiQBHqX6ywLxa'
    access_key_secret = 'R60PTPKqrMgPwli9dZjwL1Mp3xmCxR'
    bucket_name = 'sti-user-file'
    endpoint = 'https://oss-cn-hangzhou.aliyuncs.com'
    auth = oss2.Auth(access_key_id, access_key_secret)
    bucket = oss2.Bucket(auth, endpoint, bucket_name, connect_timeout=40)
    final = bucket.put_object_from_file(filename, file)
    print('finished!!!!')


def gen_plot(d, file_path, save_path):
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111)
    avgList = []
    
    for index, x in enumerate(['500hz', '1khz','2khz']):

        result = d[x]
        if isinstance(result[0], float):
            filtered_res = [result[0]]
        else:
            filtered_res= []

        sti_total = 0
        sti_count = 0
        for item in result:
            if item not in ['nan', 'NaN']:
                sti_total += float(item)
                sti_count += 1
                filtered_res.append(float(item))
        if sti_count != 0:
            avarage = str(sti_total / sti_count)[:4]
        else:
            avarage = 0
        avgList.append(avarage)
    # 画图
    categories = ['500hz', '1khz','2khz']
    values = [float(i) for i in avgList]
    normalized_values =(np.array(values) - min(values)) / (max(values) - min(values))
    cm = plt.cm.get_cmap('coolwarm')
    bar_colors = cm(normalized_values)
    bars = plt.bar(categories, values,width=0.4, color='skyblue')
    for i, bar in enumerate(bars):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1, str(values[i]), ha='center',fontsize = 25)
    #plt.plot(['500hz','1khz'], [float(i) for i in avgList], linestyle='--', color='red', marker='.', markersize=18, linewidth=2)

    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams["axes.unicode_minus"] = False
    
    plt.xlabel('Frequency / Hz', fontsize=20)
    plt.ylabel('T60', fontsize=20)

    plt.ylim(0, 2.5)
    plt.xlim(-0.5, len(categories)-0.5)
    plt.grid(linestyle='-.')
    plt.yticks(size=15)
    plt.xticks(size=15)

    yMajorLocator = MultipleLocator(0.1)
    ax.yaxis.set_major_locator(yMajorLocator)

    title = "\n".join(['T60 Result', 'File:' + file_path.split('/')[-1]])
    plt.suptitle(title, fontsize=25)
    ss = 'Average T60:'
    for temp in avgList:
        ss += temp + ' '
    fig.text(s=ss, x=0.75, y=0.955, fontsize=25, ha='center', color='red')
    fig.savefig(save_path)
    return save_path


class ResultPrinter(object):
    """一个无情的结果打印机器"""

    def __init__(self):
        self.out_dict = {'start': [],
                         'end': [],
                         '500hz': [],
                         '1khz': [],
                         '2khz': []
                         }
        self.end_time_num = []

    def print_results(self, out):
        already_time = self.get_start_len()
        for i in range(len(out['1khz'])):
            start = (already_time + i) * overlap
            end = start + 4
            self.out_dict['start'].append(second_to_time(start))
            self.end_time_num.append(start)
            self.out_dict['end'].append(second_to_time(end))
            if out['500hz'][i] != 'NaN':
                self.out_dict['500hz'].append(out['500hz'][i])
            else:
                self.out_dict['500hz'].append('NaN')
                
            if out['1khz'][i] != 'NaN':
                self.out_dict['1khz'].append(out['1khz'][i])
            else:
                self.out_dict['1khz'].append('NaN')
                
            if out['2khz'][i] != 'NaN':
                self.out_dict['2khz'].append(out['2khz'][i])
            else:
                self.out_dict['2khz'].append('NaN')


        print(self.out_dict)
    def save_csv_result(self, save_file_path):
        for x in ['500hz', '1khz','2khz']:
            for i in range(len(self.out_dict[x])):
                self.out_dict[x][i] = str(self.out_dict[x][i])[:4]
        out_data = pd.DataFrame.from_dict(self.out_dict)
        out_data.to_csv(save_file_path)
        print('Save csv already! Path: ', save_file_path)

    def get_start_len(self):
        return len(self.out_dict['start'])


class VideoProcessing(object):
    def __init__(self, path):
        self.path = path
        self.audio = None
        self.audio_time = None
        self.fs = None
        self.get_wav_from_video()

    def get_wav_from_video(self):
        my_audio_clip = AudioFileClip(self.path)
        audio_data = my_audio_clip.to_soundarray()
        framerate = my_audio_clip.fps
        if framerate != 16000:
            audio_data = scipy.signal.resample(audio_data, int(len(audio_data) / framerate * 16000))
        nframes, nchannels = audio_data.shape
        if nchannels == 2:
            audio_data = audio_data.T[0]
        if isinstance(audio_data[0], np.float):
            audio_data = np.array(audio_data * 32768.0, dtype=np.int16)
        elif isinstance(audio_data[0], np.int32):
            audio_data = (audio_data >> 16).astype(np.int16)
        audio_time = len(audio_data) / 16000
        self.audio, self.audio_time, self.fs = audio_data, audio_time, 16000


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


# Load model
def load_model(path, output_dir):
    video_path = path
    
    audio, fs = librosa.load(video_path, mono=False, sr=16000)
    
    # video = VideoProcessing(video_path)
    resultPrinter = ResultPrinter()
    # if video.audio_time <= 4:
    #     raise ValueError('Audio time less than 4s, cannot infer its T60!')
    
    dataset_500hz = DatasetSlice(audio, fs, overlap, ifVad=ifVad, which_freq=2)
    videoloader_500hz = DataLoader(dataset_500hz, batch_size=batch_size, shuffle=False, collate_fn=vad_collate_fn, drop_last=True)
    
    dataset_1khz = DatasetSlice(audio, fs, overlap, ifVad=ifVad, which_freq=3)
    videoloader_1khz = DataLoader(dataset_1khz, batch_size=batch_size, shuffle=False, collate_fn=vad_collate_fn, drop_last=True)

    dataset_2khz = DatasetSlice(audio, fs, overlap, ifVad=ifVad, which_freq=4)
    videoloader_2khz = DataLoader(dataset_2khz, batch_size=batch_size, shuffle=False, collate_fn=vad_collate_fn, drop_last=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_collection = {'500hz':[],'1khz':[], '2khz':[]}
    
    empty_index = []

    # Model processing
    with torch.no_grad():
        
        """1khz"""
        progress_bar = tqdm(videoloader_1khz)
        for i, datas in enumerate(progress_bar):
            images_1khz, temp_vad_lst = datas
            
            if temp_vad_lst:
                empty_index.extend(temp_vad_lst)
            if isinstance(images_1khz, bool):
                continue
            if not images_1khz.shape[0] == 4:
               continue

            input_image_1khz = images_1khz.numpy().astype(np.float32)
            output_1khz_pts = ort_session_1khz.run(None, {'modelInput': input_image_1khz})
            output_1khz_pts = np.squeeze(output_1khz_pts).tolist()

            if isinstance(output_1khz_pts, float):
               output_collection['1khz'].append(output_1khz_pts)
            else:
               output_collection['1khz'].extend(output_1khz_pts)
  
        """500hz"""
        progress_bar = tqdm(videoloader_500hz)
        for i, datas in enumerate(progress_bar):
            images_500hz, temp_vad_lst = datas
            
            if temp_vad_lst:
                empty_index.extend(temp_vad_lst)
            if isinstance(images_500hz, bool):
                continue
            if not images_500hz.shape[0] == 4:
               continue

            input_image_500hz = images_500hz.numpy().astype(np.float32)
            output_500hz_pts = ort_session_500hz.run(None, {'modelInput': input_image_500hz})
            output_500hz_pts = np.squeeze(output_500hz_pts).tolist()
            if isinstance(output_500hz_pts, float):
               output_collection['500hz'].append(output_500hz_pts)
            else:
               output_collection['500hz'].extend(output_500hz_pts)
               
        """2khz"""
        progress_bar = tqdm(videoloader_2khz)
        for i, datas in enumerate(progress_bar):
            images_2khz, temp_vad_lst = datas
            
            if temp_vad_lst:
                empty_index.extend(temp_vad_lst)
            if isinstance(images_2khz, bool):
                continue
            if not images_2khz.shape[0] == 4:
               continue

            input_image_2khz = images_2khz.numpy().astype(np.float32)
            output_2khz_pts = ort_session_2khz.run(None, {'modelInput': input_image_2khz})
            output_2khz_pts = np.squeeze(output_2khz_pts).tolist()
            if isinstance(output_2khz_pts, float):
               output_collection['2khz'].append(output_2khz_pts)
            else:
               output_collection['2khz'].extend(output_2khz_pts)
               
               
    if empty_index:
        for item in empty_index:
            output_collection['1khz'].insert(item, 'NaN')
            output_collection['500hz'].insert(item, 'NaN')
            output_collection['2khz'].insert(item, 'NaN')
            
    save_name = os.path.join(output_dir, video_path.split('/')[-1].split('.')[0])
    # print("save_name",save_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    resultPrinter.print_results(out=output_collection)
    resultPrinter.save_csv_result(save_name + '.csv')
    print('- Csv results saved:', save_name + '.csv')

    img_save_path = gen_plot(resultPrinter.out_dict,video_path, save_name + '.png')

    print('- Plot results saved:', save_name + '.png')
    return img_save_path


@app.get("/", response_class=HTMLResponse)
async def main():
    return open('./frontend.html', 'r').read()


def getFileSize(filePath):
    fsize = os.path.getsize(filePath)
    if fsize < 1024:
        return str(str(round(fsize, 2)) + 'Byte')
    else:
        KBX = fsize / 1024
        if KBX < 1024:
            return str(str(round(KBX, 2)) + 'K')
        else:
            MBX = KBX / 1024
            if MBX < 1024:
                return str(str(round(MBX, 2)) + 'M')
            else:
                return str(str(round(MBX / 1024)) + 'G')


@app.post("/uploaded/")
async def create_files(files: UploadFile = File(...)):
    clean_lst = []

    # In order to obtain user file info of uploading
    name = files.filename
    start = time.time()
    current_time = time.strftime('%Y_%m_%d_%H:%M:%S', time.localtime())

    info = await files.read()
    with open(save_dir + files.filename, "wb") as f:
        f.write(info)
    f.close()

    client_file_path = os.path.join(save_dir, name)
    # user upload file  processed  by model then save file to 'output__dir'
    oss_load_file = oss_zip_folder + name.split(".")[0] + '.zip'
    oss_zip = zipfile.ZipFile(oss_load_file, 'w', zipfile.ZIP_DEFLATED)
    oss_zip.write(client_file_path, arcname=name)
    img_saved_path = load_model(client_file_path, output__dir)
    f = open(img_saved_path, 'rb')
    ls_f = base64.b64encode(f.read()) #读取文件内容，转换为base64编码
    f.close()
    # to client zip file
    zip_name = zip_dir + name.split(".")[0] + '.zip'
    zip = zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED)
    for item in os.listdir(output__dir):
        compare_name = str(name.split('.')[0])
        new_item = str(item)
        if compare_name in new_item:
            file = os.path.join(output__dir, item)
            clean_lst.append(file)
            print(file)
            zip.write(file, arcname=item)
            # oss bucket add csv file
            if new_item.split('.')[-1] == "csv":
                oss_zip.write(file, arcname=item)

    zip.write(client_file_path, arcname=name)
    zip.close()
    oss_zip.close()
    # zip file upload file to oss
    client_name = name.split(".")[0] + '.zip'
    oss_name = name.split(".")[0] + '_oss_' + '.zip'
    # oss zip file upload  *.mp4 or wav  | model process csv | log information

    upload_oss(oss_name, oss_load_file)
    # upload_users_info

    '''
    ## Function: clean rubbish ###
    cleaner: 
        ① client_upload_file mp4 or wav
        ② model_process *.csv and *.png
        ③ zip_folder for client   
    '''

    # client_upload_file
    client_upload_file = os.path.join(save_dir, name)
    file_size = getFileSize(client_file_path)

    clean_lst.append(client_upload_file)
    for del_file in clean_lst:
        os.remove(del_file)
    end = time.time()
    final_time = str(round((end - start) / 60, 3)) + "Min"

    print("~~Finished~~")
    visit_number = number + 1

    csv_head = [" ID:", "User_Info:", "Visit_Time:", "File_Size:", "Handle_Time:"]
    log_lst = []
    log_lst.append(str(visit_number))
    log_lst.append(str(name))
    log_lst.append(str(current_time))
    log_lst.append(file_size)
    log_lst.append(str(final_time))
    print("log_lst", log_lst)
    log_name = "log_info.csv"
    path = log_dir + log_name
    with open(path, 'a+', encoding="utf-8", newline='') as csvfile:
        if os.path.getsize(path) == 0:
            writer = csv.DictWriter(csvfile, csv_head)
            writer.writeheader()
        writer = csv.DictWriter(csvfile, log_lst)
        writer.writeheader()

    upload_oss(log_name, path)
    encoder_base64 = "data:image/jpeg;base64," + str(ls_f)[2:-1]
    html_content = open('./resultImage.html', 'r').read().replace('ResultImageFile', encoder_base64)
    
    #return FileResponse(str(zip_dir + client_name), filename=client_name)
    return HTMLResponse(html_content)

if __name__ == "__main__":
    import uvicorn
    args = parser.parse_args()
    uvicorn.run(app='main_t60:app', host='0.0.0.0', port=args.port, reload=True)
