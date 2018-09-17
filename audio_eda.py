import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from python_speech_features import mfcc
from python_speech_features import logfbank

def plot_signals(signals):
    fig, axes = plt.subplots(nrows=2, ncols=5, sharex=False, 
                         sharey=True, figsize=(20, 5))
    fig.suptitle('Time Series', size=16, y=1.02)
    i = 0
    for x in range(2):
        for y in range(5):
            axes[x,y].set_title(list(signals.keys())[i])
            axes[x,y].plot(list(signals.values())[i])
            axes[x,y].get_xaxis().set_visible(False)
            axes[x,y].get_yaxis().set_visible(False)
            i+=1

def plot_fft(fft):
    fig, axes = plt.subplots(nrows=2, ncols=5, sharex=False, 
                         sharey=True, figsize=(20, 5))
    fig.suptitle('Fourier Transforms', size=16, y=1.02)
    i = 0
    for x in range(2):
        for y in range(5):
            data = list(fft.values())[i]
            Y, freq = data[0], data[1]
            axes[x,y].set_title(list(fft.keys())[i])
            axes[x,y].plot(freq, Y)
            axes[x,y].get_xaxis().set_visible(False)
            axes[x,y].get_yaxis().set_visible(False)
            i+=1

def plot_mfccs(mfccs):
    fig, axes = plt.subplots(nrows=2, ncols=5, sharex=False, 
                         sharey=True, figsize=(20, 5))
    fig.suptitle('Mel Cepstrum Coefficients', size=16, y=1.02)
    i = 0
    for x in range(2):
        for y in range(5):
            axes[x,y].set_title(list(mfccs.keys())[i])
            axes[x,y].imshow(list(mfccs.values())[i],
                        cmap='hot', interpolation='nearest')
            axes[x,y].get_xaxis().set_visible(False)
            axes[x,y].get_yaxis().set_visible(False)
            i+=1

def plot_fbank(fbanks):
    fig, axes = plt.subplots(nrows=2, ncols=5, sharex=False, 
                         sharey=True, figsize=(20, 5))
    fig.suptitle('Filter Bank Coefficients', size=16, y=1.02)
    i = 0
    for x in range(2):
        for y in range(5):
            axes[x,y].set_title(list(fbanks.keys())[i])
            axes[x,y].imshow(list(fbanks.values())[i],
                        cmap='hot', interpolation='nearest')
            axes[x,y].get_xaxis().set_visible(False)
            axes[x,y].get_yaxis().set_visible(False)
            i+=1

def calc_fft(y, rate):
    n = len(y)
    freq = np.fft.rfftfreq(n, d=1/rate)
    Y = abs(np.fft.rfft(y)/n)
    return (Y,freq)

def envelope(y, rate, threshold):
    mask = []
    y = pd.Series(y).apply(np.abs)
    y_mean = y.rolling(window=int(rate/10), min_periods=1, center=True).mean()
    for mean in y_mean:
        if mean > threshold:
            mask.append(True)
        else:
            mask.append(False)
    return mask

df = pd.read_csv('instruments.csv')
df.set_index('fname', inplace=True)

for f in df.index:
    rate, signal = wavfile.read('wavfiles/'+f)
    df.at[f, 'length'] = signal.shape[0]/rate
    
classes = list(np.unique(df.label))
class_dist = df.groupby(['label'])['length'].mean()

fig, ax = plt.subplots()
ax.set_title('Class Distribution', y=1.08)
ax.pie(class_dist, labels=class_dist.index, autopct='%1.1f%%',
       shadow=False, startangle=90)
ax.axis('equal')
plt.show()
df.reset_index(inplace=True)

signals = {}
fft = {}
mfccs = {}
fbank = {}

for c in classes:
    wav_file = df[df.label==c].iloc[0,0]
    rate, signal = wavfile.read('wavfiles/'+wav_file)
    mask = envelope(signal, rate, 20)
    signal = signal[mask]
    wavfile.write('samples/'+c+'.wav', rate, signal)
    
    signals[c] = signal
    fft[c] = calc_fft(signal, rate)
    signal = signal.astype(float)
    mel = mfcc(signal[:rate], samplerate=rate,
               numcep=64, nfilt=64, nfft=1103).T
    mfccs[c] = mel
    bank = logfbank(signal[:rate], rate,
                    nfilt=64, nfft=1103).T
    fbank[c] = bank

plot_signals(signals)
plt.show()

plot_fft(fft)
plt.show()

plot_fbank(fbank)
plt.show()

plot_mfccs(mfccs)
plt.show()

if len(os.listdir('clean')) == 0:
    for f in tqdm(df.fname):
        rate, signal = wavfile.read('wavfiles/'+f)
        mask = envelope(signal, rate, 20)
        wavfile.write('clean/'+f, rate, signal[mask])

