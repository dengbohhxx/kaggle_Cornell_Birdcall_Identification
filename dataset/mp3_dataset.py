import torch.utils.data as data
import numpy as np
import pandas as pd
import librosa
from config.config_ResNet38_sigmoid_mixup import model_config

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
        y, sr = librosa.load(mp3_path,sr=model_config["sample_rate"], mono=True, res_type="kaiser_fast")
        while sr == 0:
            rand_id = np.random.randint(0, self.__len__())
            mp3_path=self.train_csv.loc[[self.sounds_id[rand_id]],["file_path"]].values[0][0]
            ebird_code=self.train_csv.loc[[self.sounds_id[rand_id]],["ebird_code"]].values[0][0]
            y, sr = librosa.load(mp3_path,sr=model_config["sample_rate"], mono=True, res_type="kaiser_fast")
        assert sr == model_config["sample_rate"]
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


