import numpy as np
import librosa

def mfcc(audio:tuple, sr:int=44100, n_mfcc:int=64, n_fft:int=1024, time_length:int=345):
    '''
    calculate the MFCC feature of audio
    ---
    parameters
    ---
    audio: tuple that after librosa processing
    sr: sampling rate
    n_mfcc: the step size of MFCC
    n_fft: the windows size of FFT
    time_length: the time feature dimension size
    ---
    returns
    ---
    MFCCs feature, mfccs.shape=(n_mfcc, time_length) 
    '''
    data, sr = audio
    #mfccs = np.mean(librosa.feature.mfcc(y=data, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft).T, axis=0)
    mfccs = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft)
    
    max_length = time_length
    mfccs_length = mfccs.shape[1]

    padding = max_length - mfccs_length

    # use zero padding to adjust the time length
    if padding > 0:
        mfccs = np.pad(mfccs, ((0, 0), (0, padding)), mode='constant')
    elif mfccs_length > max_length:
        mfccs = mfccs[:, :max_length]
    
    return mfccs


def rechannel(audio:tuple, new_channel:int):
    '''
    rechannel the input audio to the given channel
    ---
    parameters
    ---
    audio: tuple after librosa processing
    new_channel: 1 or 2 for a single or double channel
    ---
    returns
    ---
    the new tuple of the audio
    '''
    data, sr = audio
    if(new_channel == 1):
        if len(data.shape) == 2:
            rechanneled_data = np.repeat(data, 2, axis=1) # repeat single channel to double
        else:
            return audio
    elif(new_channel == 2):
        if len(data.shape) == 1:
            rechanneled_data = librosa.to_mono(data)
        else:
            return audio

    return ((rechanneled_data, sr))

def resample(audio:tuple, new_sr:int):
    '''
    resample the audio to the given sampling rate
    ---
    parameters
    ---
    audio: tuple after librosa processing
    new_sr: the sampling rate
    ---
    returns
    ---
    the new tuple of the audio
    '''
    data, sr = audio

    if (sr == new_sr):
        return audio
    
    resampled_data = librosa.resample(data, orig_sr=sr, target_sr=new_sr)

    return ((resampled_data, new_sr))


def preProcess(audio:tuple, channel:int, sr:int, n_mfcc:int, n_fft:int, time_length:int):
    '''
    integrate rechannel, resample and MFCC feature extraction steps
    ---
    parameters
    ---
    audio: tuple after librosa processing
    channel: standard channel size
    sr: standard sampling rate
    n_mfcc, n_fft, time_length: parameters for MFCC
    ---
    returns
    ---
    the MFCC feature numpy array of given audio, mfccs.shape = (n_mfcc, time_length)
    '''
    aud = rechannel(audio, channel)
    aud = resample(aud, new_sr=sr)

    mfccs = mfcc(audio=aud, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, time_length=time_length)
    return mfccs