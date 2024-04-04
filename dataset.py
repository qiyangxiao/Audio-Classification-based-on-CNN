import os
import librosa
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from utils import preProcess

def readWavs(meta_file:str, wavs_folder:str, channel:int=1, sr:int=44100, n_mfcc:int=64, n_fft:int=1024, time_length:int=345, save:bool=False, save_path:str=''):
    '''
    preprocess data from wavs file 
    parameters
    ---
    meta_file: meta file path
    wavs_folder: folder to the audio waves
    channel: the value of audio channel
    sr: sampling rate
    n_mfcc: the window size of MFCC
    n_fft: the window size of FFT
    save: whether to save the X, Y
    save_path: if save, declare your save path
    ---
    returns
    ---
    numpy arrays of X and the corresponding labels
    '''
    df = pd.read_csv(meta_file, encoding='UTF-8')

    audio_files = []
    audio_labels = []
    for idx, row in df.iterrows():
        audio_meta = row
        folder = 'fold' + str(audio_meta['fold'])
        filepath = os.path.join(folder, audio_meta['slice_file_name'])
        fullpath = os.path.join(wavs_folder, filepath) # the full path to certain wav file
        if os.path.exists(fullpath):
            label = audio_meta['classID']
            audio_labels.append(label)
            audio_files.append(fullpath)

    X = []

    for idx, audio_file in enumerate(audio_files):
        aud = librosa.load(audio_file)
        mfcc_feature = preProcess(audio=aud, channel=channel, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, time_length=time_length)
        X.append(mfcc_feature)


    X = np.array(X)
    Y = np.array(audio_labels)

    # the files that save the data, as save_path is given
    X_filename = 'datax.npy'
    Y_filename = 'labely.npy'

    if save:
        saveData(os.path.join(save_path, X_filename), X)
        saveData(os.path.join(save_path, Y_filename), Y)
        print(f'Data and labels have been saved to {save_path}')

    print(f'X.shape={X.shape}, Y.shape={Y.shape}')
    return X, Y

def saveData(savepath:str, arr:np.array):
    np.save(savepath, arr)

def loadData(filepath:str):
    return np.load(filepath)

class AudioDataset(Dataset):
    def __init__(self, datax, labely):
        self.X = datax
        self.y = labely

    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, index):
        feature = self.X[index]
        label = self.y[index]

        feature = torch.tensor(feature, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)

        return feature, label
    
def load_torch_data(X, y, split_size:float=1):
    '''
    load numpy arrays to torch tensor iter
    ---
    parameters
    ---
    X: numpy arrays of data
    y: numpy arrays of corresponding labels
    split_size: the percentage of train size, 1 means train and test dataloaders are the same
    ---
    returns
    ---
    train loader and test loader in torch tensor
    '''
    dataset = AudioDataset(X, y)
    if split_size == 1:
        train_loader = DataLoader(dataset, batch_size=32 ,shuffle=True)
        test_loader = DataLoader(dataset, batch_size=32 ,shuffle=True)
    else:
        train_size = int(split_size * y.shape[0])
        test_size = y.shape[0] - train_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

        train_loader = DataLoader(train_dataset, batch_size=32 ,shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32 ,shuffle=True)

    return train_loader, test_loader
'''
metafile = '.\\UrbanSound8K\\data\\UrbanSound8K\\metadata\\UrbanSound8K.csv'
wavsfolder = '.\\UrbanSound8K\\data\\UrbanSound8K\\audio'
datafolder = '.\\npy_data'
X, y = readWavs(metafile, wavsfolder, save=True, save_path=datafolder)
'''
