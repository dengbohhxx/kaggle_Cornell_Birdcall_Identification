#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import librosa
import soundfile as sf

def resample(ebird_code: str,filename: str, target_sr: int):    
    audio_dir = TRAIN_AUDIO_DIR
    resample_dir = TRAIN_RESAMPLED_DIR
    ebird_dir = resample_dir / ebird_code
    
    try:
        y, _ = librosa.load(
            audio_dir / ebird_code / filename,
            sr=target_sr, mono=True, res_type="kaiser_fast")

        filename = filename.replace(".mp3", ".wav")
        sf.write(ebird_dir / filename, y, samplerate=target_sr)
    except Exception as e:
        print(e)
        with open("skipped.txt", "a") as f:
            file_path = str(audio_dir / ebird_code / filename)
            f.write(file_path + "\n")


# train_org = train.copy()
# TRAIN_RESAMPLED_DIR = Path("/kaggle/processed_data/train_audio_resampled")
# TRAIN_RESAMPLED_DIR.mkdir(parents=True)

# for ebird_code in train.ebird_code.unique():
#     ebird_dir = TRAIN_RESAMPLED_DIR / ebird_code
#     ebird_dir.mkdir()

# warnings.simplefilter("ignore")
# train_audio_infos = train[["ebird_code", "filename"]].values.tolist()
# Parallel(n_jobs=NUM_THREAD, verbose=10)(
#     delayed(resample)(ebird_code, file_name, TARGET_SR) for ebird_code, file_name in train_audio_infos)

# train["resampled_sampling_rate"] = TARGET_SR
# train["resampled_filename"] = train["filename"].map(
#     lambda x: x.replace(".mp3", ".wav"))
# train["resampled_channels"] = "1 (mono)"