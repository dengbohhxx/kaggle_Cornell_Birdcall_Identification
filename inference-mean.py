#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from pathlib import Path
import torch
from fastprogress import progress_bar
from dataset.mp3_dataset import INV_BIRD_CODE
import warnings
from contextlib import contextmanager
from typing import Optional
import logging
import time
import librosa
from models.backbones import  Cnn14_16k
from  config.config_Cnn14_16k import model_config
import torch.nn.functional as F


ROOT = Path.cwd()
print(ROOT)
RAW_DATA = ROOT / "birdsong-recognition"
test = pd.read_csv(RAW_DATA / "test.csv")
TEST_AUDIO_DIR = RAW_DATA / "test_audio"
weights_path=ROOT/"output"/"best-checkpoint-035epoch.tar"
SR = 32000

def get_model(config, weights_path):
    model = Cnn14_16k(**config)
    model_dict=model.state_dict()
    checkpoint = torch.load(weights_path,map_location="cpu")
    new_dict = {k: v for k, v in checkpoint["model_state_dict"].items() if k in model_dict.keys()}
    model_dict.update(new_dict)
    model.load_state_dict(model_dict)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return model

def split_clip(length,clip,period,stride,sr):
    clips=[]
    clip=clip[0:int(length*sr)].astype(np.float32)
    start=0
    while start<int(length*sr):
        y=clip[start:start+period*sr]
        if len(y)!=period*sr:
            y_pad = np.zeros(period * sr, dtype=np.float32)
            y_pad[0:len(y)] = y
            clips.append(y_pad)
            break
        start=start+stride*sr
        clips.append(y)
    return clips 
def rand_split_clip(rand_len,length,clip,period,stride,sr):
    clips=[]
    clip=clip[0:int(length*sr)].astype(np.float32)
    start = np.random.randint(length*sr - rand_len*sr)
    end=int(start+rand_len*sr)
    while start<end:
        y=clip[start:start+period*sr]
        if len(y)!=period*sr:
            y_pad = np.zeros(period * sr, dtype=np.float32)
            y_pad[0:len(y)] = y
            clips.append(y_pad)
            break
        start=start+stride*sr
        clips.append(y)
    return clips 

def prediction_for_clip(test_df, clip,model,threshold=0.5,period=5,stride=2,sr=32000,cut_length=30):
    prediction_dict = {}
    for i in range(test_df.shape[0]):
        temp=test_df.loc[i, :]
        row_id = temp.row_id
        seconds=temp.seconds
        print(seconds)
        if np.isnan(seconds) :
            seconds=int(len(clip)/sr)
            #print(f'nan:{seconds}')
        if seconds<=cut_length:
            splited=split_clip(seconds, clip, period, stride, sr)
        else:
            splited=rand_split_clip(cut_length,seconds, clip, period, stride, sr)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        all_events = set()
        audio=torch.tensor(splited).to(device).float()
        print(audio.size())
        #audio=torch.unsqueeze(audio, 0)
        with torch.no_grad():
            prediction = model(audio)["clipwise_output"]
            proba = prediction.detach().cpu().numpy()
            print(proba.sum())
            proba=np.mean(proba,axis=0)                   
        labels = proba >= threshold
        label=np.argwhere(labels).reshape(-1).tolist()
        for i in label:
            all_events.add(i)
        if len(all_events) == 0:
            prediction_dict[row_id] = "nocall"
        else:
            labels_str_list = list(map(lambda x: INV_BIRD_CODE[x], all_events))
            label_string = " ".join(labels_str_list)
            prediction_dict[row_id] = label_string
        print(prediction_dict[row_id])
    return prediction_dict         
        
# In[ ]:


def prediction(test_df,test_audio,model_config,target_sr,threshold,period,stride,weights_path,cut_length):
    model = get_model(model_config,weights_path)
    unique_audio_id = test_df.audio_id.unique()
    warnings.filterwarnings("ignore")
    prediction_dfs = []
    for audio_id in unique_audio_id:
        clip, _ = librosa.load(test_audio / (audio_id + ".mp3"),sr=target_sr,mono=True,res_type="kaiser_fast")      
        test_df_for_audio_id = test_df.query(f"audio_id == '{audio_id}'").reset_index(drop=True)
        prediction_dict = prediction_for_clip(test_df_for_audio_id,clip,model,threshold,period,stride,target_sr,cut_length)
        row_id = list(prediction_dict.keys())
        birds = list(prediction_dict.values())
        prediction_df = pd.DataFrame({"row_id": row_id,"birds": birds})
        prediction_dfs.append(prediction_df)
    prediction_df = pd.concat(prediction_dfs, axis=0, sort=False).reset_index(drop=True)
    return prediction_df


# ## Prediction

# In[ ]:


submission = prediction(test_df=test,
                        test_audio=TEST_AUDIO_DIR,
                        model_config=model_config,
                        target_sr=SR,
                        threshold=0.4,
                        period=5,
                        stride=5,
                        weights_path=weights_path,
                        cut_length=30
                        )
submission.to_csv("submission.csv", index=False)


# In[ ]:


submission