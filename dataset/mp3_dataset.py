import torch.utils.data as data

import numpy as np
import pandas as pd
import librosa

data_csv = pd.read_csv("birdsong-recognition/train.csv")
bird_species=np.unique(np.array(list(data_csv["ebird_code"])))
BIRD_CODE={ bird:i for i,bird in enumerate(bird_species) }
INV_BIRD_CODE = {v: k for k, v in BIRD_CODE.items()}

class Ori_Mp3_Dataset(data.Dataset):
    def __init__(self,sounds_id,train_all,waveform_transforms=None,peroid=5):
        self.sounds_id = sounds_id  
        self.train_csv=train_all
        self.waveform_transforms = waveform_transforms
        self.peroid=peroid
    def __len__(self):
        return self.sounds_id.shape[0]
    def __getitem__(self, idx):
        mp3_path=self.train_csv.loc[[self.sounds_id[idx]],["file_path"]].values[0][0]
        ebird_code=self.train_csv.loc[[self.sounds_id[idx]],["ebird_code"]].values[0][0]        
        y, sr = librosa.load(mp3_path,sr=16000, mono=True, res_type="kaiser_fast")
        if self.waveform_transforms:
            y = self.waveform_transforms(y)
        else:
            len_y = len(y)
            effective_length = sr * self.peroid
            if len_y < effective_length:
                new_y = np.zeros(effective_length, dtype=y.dtype)
                start = np.random.randint(effective_length - len_y)
                new_y[start:start + len_y] = y
                y = new_y.astype(np.float32)
            elif len_y > effective_length:
                start = np.random.randint(len_y - effective_length)
                y = y[start:start + effective_length].astype(np.float32)
            else:
                y = y.astype(np.float32)
        labels = np.zeros(len(BIRD_CODE), dtype="f")
        labels[BIRD_CODE[ebird_code]] = 1

        return  y,labels

