from scipy.io import wavfile
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Conv1D, MaxPool1D, GlobalMaxPooling1D, Dropout, Dense
from keras.layers import Conv2D, MaxPool2D, Flatten
from keras.models import Sequential
from keras.utils import to_categorical
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from python_speech_features import mfcc

def build_mfcc(config):
    c = config
    X = []
    y = []
    _min, _max = float('inf'), float('-inf')
    for f in tqdm(df.index):
        rate, wav = wavfile.read('clean/'+f)
        wav = wav.astype(float)
        label = df.at[f, 'label']
        step = c.step
        for i in range(0, len(wav), step):
            partition = i+step
            if step > wav.shape[0]:
                signal = np.zeros((step, 1))
                signal[:wav.shape[0], :] = wav.reshape(-1, 1)
                X_mfcc = mfcc(signal, rate,
                                   numcep=c.nfeat, nfilt=c.nfeat, nfft=c.nfft).T
            elif partition > len(wav):
                X_mfcc = mfcc(wav[-step:], rate,
                                   numcep=c.nfeat, nfilt=c.nfeat, nfft=c.nfft).T
            else:
                X_mfcc = mfcc(wav[i:i+step], rate,
                                   numcep=c.nfeat, nfilt=c.nfeat, nfft=c.nfft).T
            _min = min(np.amin(X_mfcc), _min)
            _max = max(np.amax(X_mfcc), _max)
            X.append(X_mfcc)
            y.append(classes.index(label))
    X, y = np.array(X), np.array(y)
    X = (X - _min) / (_max - _min)
    print(_min, _max)
    X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
    y = to_categorical(y, num_classes=10)
    return X, y

def build_1d(config):
    c = config
    X = []
    y = []
    for f in df.index:
        rate, wav = wavfile.read('clean/'+f)
        label = df.at[f, 'label']
        step = c.step
        for i in range(0, len(wav), step):
            partition = i+step
            if step > wav.shape[0]:
                signal = np.zeros((step, 1))
                signal[:wav.shape[0], :] = wav
            elif partition > len(wav):
                signal = wav[-step:]
            else:
                signal = wav[i:i+step]
            X.append(signal)
            y.append(classes.index(label))
    X, y = np.array(X), np.array(y)
    mms = MinMaxScaler()
    X = mms.fit_transform(X)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    y = to_categorical(y, num_classes=10)
    return X, y

def get_1d_model():
    model = Sequential()
    model.add(Conv1D(16, 9, activation='relu', padding='same',
                     input_shape=input_shape))
    model.add(Conv1D(16, 9, activation='relu', padding='same'))
    model.add(MaxPool1D(16))
    model.add(Dropout(rate=0.1))
    model.add(Conv1D(32, 3, activation='relu', padding='same'))
    model.add(Conv1D(32, 3, activation='relu', padding='same'))
    model.add(MaxPool1D(4))
    model.add(Dropout(rate=0.1))
    model.add(Conv1D(64, 3, activation='relu', padding='same'))
    model.add(Conv1D(64, 3, activation='relu', padding='same'))
    model.add(MaxPool1D(4))
    model.add(Dropout(rate=0.1))
    model.add(Conv1D(128, 3, activation='relu', padding='same'))
    model.add(Conv1D(128, 3, activation='relu', padding='same'))
    model.add(GlobalMaxPooling1D())
    model.add(Dropout(rate=0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['acc'])
    return model

def get_2d_model():
    model = Sequential()
    model.add(Conv2D(16, (5, 5), activation='relu', strides=(2, 2),
                     padding='same', input_shape=input_shape))
    model.add(Conv2D(32, (3, 3), activation='relu', strides=(1, 1),
                     padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', strides=(1, 1),
                     padding='same'))
    model.add(MaxPool2D((2,2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['acc'])
    return model

class Config:
    def __init__(self, mode='two', nfeat=64, nfft=1103, rate=44100):
        self.mode = mode
        self.nfeat = nfeat
        self.nfft = nfft
        self.rate = rate
        self.step = int(rate/10)

df = pd.read_csv('instruments.csv')
df.set_index('fname', inplace=True)

for f in df.index:
    rate, signal = wavfile.read('clean/'+f)
    df.at[f, 'length'] = signal.shape[0]/rate

classes = list(np.unique(df.label))
class_dist = df.groupby(['label'])['length'].mean()

fig, ax = plt.subplots()
ax.set_title('Class Distribution', y=1.08)
ax.pie(class_dist, labels=class_dist.index, autopct='%1.1f%%',
       shadow=False, startangle=90)
ax.axis('equal')
plt.show()

config = Config(mode='two', nfeat=64, nfft=1103, rate=44100)

if config.mode == 'one':
    X, y = build_1d(config)
    y_flat = np.argmax(y, axis=1)
    input_shape = (X.shape[1], 1)
    model = get_1d_model()
    
elif config.mode == 'two':
    X, y = build_mfcc(config)
    y_flat = np.argmax(y, axis=1)
    input_shape = (X.shape[1], X.shape[2], 1)
    model = get_2d_model()
    
print(input_shape)

class_weight = compute_class_weight('balanced',
                                     np.unique(y_flat),
                                     y_flat)

model.fit(X, y, epochs=10, batch_size=32,
          shuffle=True,
          class_weight=class_weight)


model_yaml = model.to_yaml()
with open('mfcc.yaml', 'w') as yaml_file:
    yaml_file.write(model_yaml)
model.save_weights('mfcc.h5')
