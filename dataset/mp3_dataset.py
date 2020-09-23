import torch.utils.data as data
import numpy as np
import pandas as pd
import librosa
from config.efficientnet_b5 import model_config 
import cv2
import random
import torch

data_csv = pd.read_csv("./birdsong-recognition/train.csv")
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

def audio2melspec(idx,train_csv,sounds_id,waveform_transforms,peroid):
    mp3_path=train_csv.loc[[sounds_id[idx]],["file_path"]].values[0][0]
    ebird_code=train_csv.loc[[sounds_id[idx]],["ebird_code"]].values[0][0]
    y, sr = librosa.load(mp3_path,sr=model_config["sample_rate"], mono=True, res_type="kaiser_fast")
    labels = np.zeros(len(BIRD_CODE), dtype="f")
    labels[BIRD_CODE[ebird_code]] = 1
    len_y = len(y)
    effective_length = sr * peroid
    if len_y < effective_length:
        new_y = np.zeros(effective_length, dtype=y.dtype)
        start = np.random.randint(effective_length - len_y)
        new_y[start:start + len_y] = y
        y = new_y.astype(np.float32)
    elif len_y > effective_length:
        for i in range(5):
            start = np.random.randint(len_y - effective_length)
            audio=y
            y_1 = audio[start:start + effective_length].astype(np.float32)
            y_trimmed,index=librosa.effects.trim(y_1, top_db=20)
            if len(y_trimmed)/effective_length >=0.8:
                y=y_1
                break
            if i==4:
                y=y_1
    else:
        y = y.astype(np.float32)    
    melspec = librosa.feature.melspectrogram(y, sr=sr,n_fft=model_config["window_size"],
                                                 hop_length=model_config["hop_size"],n_mels=model_config["mel_bins"],
                                                 fmin=model_config["fmin"],fmax=model_config["fmax"])
    melspec = librosa.power_to_db(melspec).astype(np.float32)    
    return melspec,labels

def get_random_sample(dataset):
    rnd_idx = random.randint(0, len(dataset) - 1)
    rnd_image,rnd_target=audio2melspec(rnd_idx,dataset.train_csv,dataset.sounds_id,dataset.waveform_transforms,dataset.peroid)
    rnd_image=rnd_image.copy()
    rnd_target=rnd_target
    rnd_image = dataset.transform(rnd_image)
    return rnd_image, rnd_target


class AddMixer:
    def __init__(self, alpha_dist='uniform'):
        assert alpha_dist in ['uniform', 'beta']
        self.alpha_dist = alpha_dist

    def sample_alpha(self):
        if self.alpha_dist == 'uniform':
            return random.uniform(0, 0.5)
        elif self.alpha_dist == 'beta':
            return np.random.beta(0.4, 0.4)

    def __call__(self, dataset, image, target):
        rnd_image, rnd_target = get_random_sample(dataset)

        alpha = self.sample_alpha()
        image = (1 - alpha) * image + alpha * rnd_image
        target = (1 - alpha) * target + alpha * rnd_target
        return image, target


class SigmoidConcatMixer:
    def __init__(self, sigmoid_range=(3, 12)):
        self.sigmoid_range = sigmoid_range

    def sample_mask(self, size):
        x_radius = random.randint(*self.sigmoid_range)

        step = (x_radius * 2) / size[1]
        x = np.arange(-x_radius, x_radius, step=step)
        y = torch.sigmoid(torch.from_numpy(x)).numpy()
        mix_mask = np.tile(y, (size[0], 1))
        return torch.from_numpy(mix_mask.astype(np.float32))

    def __call__(self, dataset, image, target):
        rnd_image, rnd_target = get_random_sample(dataset)

        mix_mask = self.sample_mask(image.shape[-2:])
        rnd_mix_mask = 1 - mix_mask

        image = mix_mask * image + rnd_mix_mask * rnd_image
        target = target + rnd_target
        target = np.clip(target, 0.0, 1.0)
        return image, target


class RandomMixer:
    def __init__(self, mixers, p=None):
        self.mixers = mixers
        self.p = p

    def __call__(self, dataset, image, target):
        mixer = np.random.choice(self.mixers, p=self.p)
        image, target = mixer(dataset, image, target)
        return image, target


class UseMixerWithProb:
    def __init__(self, mixer, prob=.5):
        self.mixer = mixer
        self.prob = prob

    def __call__(self, dataset, image, target):
        if random.random() < self.prob:
            return self.mixer(dataset, image, target)
        return image, target


class Ori_Mp3_Dataset(data.Dataset):
    def __init__(self,sounds_id,train_all,waveform_transforms=None,peroid=model_config["duration"]):
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
    def __init__(self,sounds_id,train_all,waveform_transforms=None,peroid=model_config["duration"],spectrogram_transforms=None,mixer=None,img_size=224):
        self.sounds_id = sounds_id  
        self.train_csv=train_all
        self.waveform_transforms = waveform_transforms
        self.peroid=peroid
        self.img_size = img_size
        self.transform = spectrogram_transforms
        self.mixer = mixer
    def __len__(self):
        return self.sounds_id.shape[0]      
    def __getitem__(self, idx):
        melspec,labels=audio2melspec(idx,self.train_csv,self.sounds_id,self.waveform_transforms,self.peroid)      
        if self.transform:
            melspec = self.transform(melspec)
        if self.mixer is not None:
            image,labels= self.mixer(self, melspec, labels) 
        return  image,labels
    

class TestDataset(data.Dataset):
    def __init__(self, df: pd.DataFrame, clip: np.ndarray,
                 img_size=256,transforms=None,sr=32000,rand_len=30,period=5):
        self.df = df
        self.clip = clip
        self.img_size = img_size
        self.transforms=transforms
        self.sr=sr
        self.rand_len=rand_len
        self.period=period
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx: int):
        sample = self.df.loc[idx, :]
        site = sample.site
        row_id = sample.row_id
        
        if site == "site_3":
            y = self.clip.astype(np.float32)
            length = len(y)/self.sr
            images = []
            clip=self.clip.astype(np.float32)
            if length>=self.rand_len:
                start = np.random.randint(length*self.sr - self.rand_len*self.sr)
                end=int(start+self.rand_len*self.sr)
            else:
                start=0
                end=len(y)
            while start<end:
                y=clip[start:start+self.period*self.sr]
                if len(y)!=self.period*self.sr:
                    y_pad = np.zeros(self.period * self.sr, dtype=np.float32)
                    y_pad[0:len(y)] = y
                    melspec = librosa.feature.melspectrogram(y_pad, sr=self.sr,n_fft=model_config["window_size"],
                                                 hop_length=model_config["hop_size"],n_mels=model_config["mel_bins"],
                                                 fmin=model_config["fmin"],fmax=model_config["fmax"])
                    y = librosa.power_to_db(melspec).astype(np.float32)
                    if self.transforms:
                        y= self.transforms(y)
                    images.append(y)
                    break
                start=start+self.period*self.sr
                melspec = librosa.feature.melspectrogram(y, sr=self.sr,n_fft=model_config["window_size"],
                                                 hop_length=model_config["hop_size"],n_mels=model_config["mel_bins"],
                                                 fmin=model_config["fmin"],fmax=model_config["fmax"])
                y = librosa.power_to_db(melspec).astype(np.float32)
                if self.transforms:
                    y = self.transforms(y)
                images.append(y)                
            images=torch.stack(images,0).to(self.device).float()
            return images, row_id, site
        else:
            y = self.clip.astype(np.float32)
            length = len(y)
            for i in range(5):
                start_index = np.random.randint(length - self.period*self.sr)
                end_index=start_index+self.period*self.sr
                y_1 = self.clip[start_index:end_index].astype(np.float32)
                y_trimmed,index=librosa.effects.trim(y_1, top_db=15)
                if len(y_trimmed)/(self.period*self.sr) >=0.8:
                    y=y_1
                    break
                if i==4:
                    y=y_1
            melspec = librosa.feature.melspectrogram(y, sr=self.sr,n_fft=model_config["window_size"],
                                                 hop_length=model_config["hop_size"],n_mels=model_config["mel_bins"],
                                                 fmin=model_config["fmin"],fmax=model_config["fmax"])
            image = librosa.power_to_db(melspec).astype(np.float32)  
            if self.transforms:
                image = self.transforms(image)
            return image, row_id, site
