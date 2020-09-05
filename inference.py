import sys
from pathlib import Path
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
from models.backbones import Cnn14_DecisionLevelAtt,AttBlock
from config.Cnn14_DecisionLevelAtt import model_config,TrainGlobalConfig
import torch.nn.functional as F

ROOT = Path.cwd()
print(ROOT)
RAW_DATA = ROOT / "birdsong-recognition"
TEST_AUDIO_DIR = RAW_DATA / "test_audio"
weights_path=ROOT/"output/best-checkpoint-025epoch.tar"
SR = 32000
test = pd.read_csv(RAW_DATA / "test.csv")
    
def get_model(config: dict, weights_path: str):
    model = Cnn14_DecisionLevelAtt(**config)
    checkpoint = torch.load(weights_path,map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    device = torch.device("cpu")
    model.to(device)
    model.eval()
    return model


# In[ ]:


def prediction_for_clip(test_df,
                        clip, 
                        model,
                        threshold=0.5):
    PERIOD = 30
    audios = []
    y = clip.astype(np.float32)
    len_y = len(y)
    start = 0
    end = PERIOD * SR
    while True:
        y_batch = y[start:end].astype(np.float32)
        if len(y_batch) != PERIOD * SR:
            y_pad = np.zeros(PERIOD * SR, dtype=np.float32)
            y_pad[:len(y_batch)] = y_batch
            audios.append(y_pad)
            break
        start = end
        end += PERIOD * SR
        audios.append(y_batch)
        
    array = np.asarray(audios)
    tensors = torch.from_numpy(array)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.eval()
    estimated_event_list = []
    global_time = 0.0
    site = test_df["site"].values[0]
    audio_id = test_df["audio_id"].values[0]
    for image in progress_bar(tensors):
        image = image.view(1, image.size(0))
        image = image.to(device)

        with torch.no_grad():
            prediction = model(image)
            framewise_outputs = prediction["framewise_output"].detach(
                ).cpu().numpy()[0]
                
        thresholded = framewise_outputs >= threshold

        for target_idx in range(thresholded.shape[1]):
            if thresholded[:, target_idx].mean() == 0:
                pass
            else:
                detected = np.argwhere(thresholded[:, target_idx]).reshape(-1)
                head_idx = 0
                tail_idx = 0
                while True:
                    if (tail_idx + 1 == len(detected)) or (
                            detected[tail_idx + 1] - 
                            detected[tail_idx] != 1):
                        onset = 0.01 * detected[
                            head_idx] + global_time
                        offset = 0.01 * detected[
                            tail_idx] + global_time
                        onset_idx = detected[head_idx]
                        offset_idx = detected[tail_idx]
                        max_confidence = framewise_outputs[
                            onset_idx:offset_idx, target_idx].max()
                        mean_confidence = framewise_outputs[
                            onset_idx:offset_idx, target_idx].mean()
                        estimated_event = {
                            "site": site,
                            "audio_id": audio_id,
                            "ebird_code": INV_BIRD_CODE[target_idx],
                            "onset": onset,
                            "offset": offset,
                            "max_confidence": max_confidence,
                            "mean_confidence": mean_confidence
                        }
                        estimated_event_list.append(estimated_event)
                        head_idx = tail_idx + 1
                        tail_idx = tail_idx + 1
                        if head_idx >= len(detected):
                            break
                    else:
                        tail_idx += 1
        global_time += PERIOD
        
    prediction_df = pd.DataFrame(estimated_event_list)
    return prediction_df


# In[ ]:


def prediction(test_df: pd.DataFrame,
               test_audio: Path,
               model_config: dict,
               weights_path: str,
               threshold=0.5):
    model = get_model(model_config, weights_path)
    unique_audio_id = test_df.audio_id.unique()

    warnings.filterwarnings("ignore")
    prediction_dfs = []
    for audio_id in unique_audio_id:
        clip, _ = librosa.load(test_audio / (audio_id + ".mp3"),
                                   sr=SR,
                                   mono=True,
                                   res_type="kaiser_fast")
        
        test_df_for_audio_id = test_df.query(f"audio_id == '{audio_id}'").reset_index(drop=True)
        prediction_df = prediction_for_clip(test_df_for_audio_id,
                                                clip=clip,
                                                model=model,
                                                threshold=threshold)

        prediction_dfs.append(prediction_df)
    
    prediction_df = pd.concat(prediction_dfs, axis=0, sort=False).reset_index(drop=True)
    return prediction_df


# In[ ]:


prediction_df = prediction(test_df=test,
                           test_audio=TEST_AUDIO_DIR,
                           model_config=model_config,
                           weights_path=weights_path,
                           threshold=0.5)

labels = {}

for audio_id, sub_df in prediction_df.groupby("audio_id"):
    events = sub_df[["ebird_code", "onset", "offset", "max_confidence", "site"]].values
    n_events = len(events)
    removed_event = []
    # Overlap deletion: this part may not be necessary
    # I deleted this part in other model and found there's no difference on the public LB score.
    for i in range(n_events):
        for j in range(n_events):
            if i == j:
                continue
            if i in removed_event:
                continue
            if j in removed_event:
                continue
            
            event_i = events[i]
            event_j = events[j]
            
            if (event_i[1] - event_j[2] >= 0) or (event_j[1] - event_i[2] >= 0):
                pass
            else:
                later_onset = max(event_i[1], event_j[1])
                sooner_onset = min(event_i[1], event_j[1])
                sooner_offset = min(event_i[2], event_j[2])
                later_offset = max(event_i[2], event_j[2])

                intersection = sooner_offset - later_onset
                union = later_offset - sooner_onset
                
                iou = intersection / union
                if iou > 0.4:
                    if event_i[3] > event_j[3]:
                        removed_event.append(j)
                    else:
                        removed_event.append(i)

    site = events[0][4]
    for i in range(n_events):
        if i in removed_event:
            continue
        event = events[i][0]
        onset = events[i][1]
        offset = events[i][2]
        if site in {"site_1", "site_2"}:
            start_section = int((onset // 5) * 5) + 5
            end_section = int((offset // 5) * 5) + 5
            cur_section = start_section

            row_id = f"{site}_{audio_id}_{start_section}"
            if labels.get(row_id) is not None:
                labels[row_id].add(event)
            else:
                labels[row_id] = set()
                labels[row_id].add(event)

            while cur_section != end_section:
                cur_section += 5
                row_id = f"{site}_{audio_id}_{cur_section}"
                if labels.get(row_id) is not None:
                    labels[row_id].add(event)
                else:
                    labels[row_id] = set()
                    labels[row_id].add(event)
        else:
            row_id = f"{site}_{audio_id}"
            if labels.get(row_id) is not None:
                labels[row_id].add(event)
            else:
                labels[row_id] = set()
                labels[row_id].add(event)


for key in labels:
    labels[key] = " ".join(sorted(list(labels[key])))
    
    
row_ids = list(labels.keys())
birds = list(labels.values())
post_processed = pd.DataFrame({
    "row_id": row_ids,
    "birds": birds
})
post_processed.head()


# In[ ]:


all_row_id = test[["row_id"]]
submission = all_row_id.merge(post_processed, on="row_id", how="left")
submission = submission.fillna("nocall")
submission.to_csv("submission.csv", index=False)
submission.head(20)    