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
from config.config_Cnn14_16k import model_config
import torch.nn.functional as F


ROOT = Path.cwd()
print(ROOT)
RAW_DATA = ROOT / "birdsong-recognition"
test = pd.read_csv(RAW_DATA / "test.csv")
TEST_AUDIO_DIR = RAW_DATA / "test_audio"
weights_path=ROOT/"cnn14_16k_weight"/"best-checkpoint-033epoch.bin"
SR = 32000

def get_model(config, weights_path):
    model = Cnn14_16k(**config)
    checkpoint = torch.load(weights_path)
    model.load_state_dict(checkpoint["model_state_dict"])
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

def prediction_for_clip(test_df, clip,model,threshold=0.5,period=5,stride=2,sr=32000):
    prediction_dict = {}
    for i in range(test_df.shape[0]):
        temp=test_df.loc[i, :]
        row_id = temp.row_id
        seconds=temp.seconds
        print(seconds)
        print(type(seconds))
        if np.isnan(seconds) :
            seconds=int(len(clip)/sr)
            #print(f'nan:{seconds}')
        splited=split_clip(seconds, clip, period, stride, sr)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        all_events = set()
        for j in range(len(splited)):
            audio=torch.tensor(splited[j]).to(device).float()
            audio=torch.unsqueeze(audio, 0)
            with torch.no_grad():
                prediction = model(audio)["clipwise_output"]
                proba = prediction.detach().cpu().numpy()
                print(proba.sum())                   
            event = proba >= threshold
            labels = np.argwhere(event).reshape(-1).tolist()
            for label in labels:
                all_events.add(label)
        if len(labels) == 0:
            prediction_dict[row_id] = "nocall"
        else:
            labels_str_list = list(map(lambda x: INV_BIRD_CODE[x], labels))
            label_string = " ".join(labels_str_list)
            prediction_dict[row_id] = label_string
        print(prediction_dict[row_id])
    return prediction_dict         
        
# In[ ]:


def prediction(test_df,test_audio,model_config,target_sr,threshold,period,stride,weights_path):
    model = get_model(model_config,weights_path)
    unique_audio_id = test_df.audio_id.unique()
    warnings.filterwarnings("ignore")
    prediction_dfs = []
    for audio_id in unique_audio_id:
        clip, _ = librosa.load(test_audio / (audio_id + ".mp3"),sr=target_sr,mono=True,res_type="kaiser_fast")      
        test_df_for_audio_id = test_df.query(f"audio_id == '{audio_id}'").reset_index(drop=True)
        prediction_dict = prediction_for_clip(test_df_for_audio_id,clip,model,threshold,period,stride,target_sr)
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
                        stride=3,
                        weights_path=weights_path
                        )
submission.to_csv("submission.csv", index=False)


# In[ ]:


submission