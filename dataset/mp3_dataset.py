import torch.utils.data as data
import numpy as np
import pandas as pd
import librosa
from config.resnest50d_1s4x24d import model_config 
import cv2

data_csv = pd.read_csv("birdsong-recognition/train.csv")
bird_species=np.unique(np.array(list(data_csv["ebird_code"])))
BIRD_CODE={ bird:i for i,bird in enumerate(bird_species) }
INV_BIRD_CODE = {v: k for k, v in BIRD_CODE.items()}

def mono_to_color(
    X: np.ndarray, mean=None, std=None,
    norm_max=None, norm_min=None, eps=1e-6
):
    # Stack X as [X,X,X]
    X = np.stack([X, X, X], axis=-1)

    # Standardize
    mean = mean or X.mean()
    X = X - mean
    std = std or X.std()
    Xstd = X / (std + eps)
    _min, _max = Xstd.min(), Xstd.max()
    norm_max = norm_max or _max
    norm_min = norm_min or _min
    if (_max - _min) > eps:
        # Normalize to [0, 255]
        V = Xstd
        V[V < norm_min] = norm_min
        V[V > norm_max] = norm_max
        V = 255 * (V - norm_min) / (norm_max - norm_min)
        V = V.astype(np.uint8)
    else:
        # Just zero
        V = np.zeros_like(Xstd, dtype=np.uint8)
    return V


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
    
class Mp3_LogmelDataset(data.Dataset):
    def __init__(self,sounds_id,train_all,waveform_transforms=None,peroid=5,spectrogram_transforms=None,img_size=224):
        self.sounds_id = sounds_id  
        self.train_csv=train_all
        self.waveform_transforms = waveform_transforms
        self.peroid=peroid
        self.img_size = img_size
        self.spectrogram_transforms = spectrogram_transforms
    def __len__(self):
        return self.sounds_id.shape[0]
    def __getitem__(self, idx):
        mp3_path=self.train_csv.loc[[self.sounds_id[idx]],["file_path"]].values[0][0]
        ebird_code=self.train_csv.loc[[self.sounds_id[idx]],["ebird_code"]].values[0][0]
        y, sr = librosa.load(mp3_path,sr=model_config["sample_rate"], mono=True, res_type="kaiser_fast")
        '''
        while sr == 0:
            rand_id = np.random.randint(0, self.__len__())
            mp3_path=self.train_csv.loc[[self.sounds_id[rand_id]],["file_path"]].values[0][0]
            ebird_code=self.train_csv.loc[[self.sounds_id[rand_id]],["ebird_code"]].values[0][0]
            y, sr = librosa.load(mp3_path,sr=model_config["sample_rate"], mono=True, res_type="kaiser_fast")
        assert sr == model_config["sample_rate"]
        '''
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
        melspec = librosa.feature.melspectrogram(y, sr=sr,n_fft=model_config["window_size"],
                                                 hop_length=model_config["hop_size"],n_mels=model_config["mel_bins"],
                                                 fmin=model_config["fmin"],fmax=model_config["fmax"])
        melspec = librosa.power_to_db(melspec).astype(np.float32)        
        if self.spectrogram_transforms:
            melspec = self.spectrogram_transforms(melspec)
        else:
            pass
        image = mono_to_color(melspec)
        height, width, _ = image.shape
        image = cv2.resize(image, (int(width * self.img_size / height), self.img_size))
        image = np.moveaxis(image, 2, 0)
        image = (image / 255.0).astype(np.float32)     
        labels = np.zeros(len(BIRD_CODE), dtype="f")
        labels[BIRD_CODE[ebird_code]] = 1

        return  image,labels

