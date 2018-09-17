'''
Displays a tkinter window to visualize playback of audio.
Data is updated 10 times a second. Shows time domain and mfcc features.
Class predictions are distributed alphabetically (buttons and image).
'''

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib import style
import matplotlib.pyplot as plt
import wave
import pyaudio
import numpy as np
from python_speech_features import mfcc
from tkinter import ttk
import threading
import pandas as pd
import tkinter as tk
from keras.models import model_from_yaml

LARGE_FONT = ('Verdana', 12)
style.use('ggplot')
plt.ioff()

class ThreadWithReturn(threading.Thread):
    
    'overridden thread class used to return value after join'
    
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs={}, Verbose=None):
        threading.Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None
        
    def run(self):
        if self._target is not None:
            self._return = self._target(*self._args, **self._kwargs)
            
    def join(self, *args):
        threading.Thread.join(self, *args)
        return self._return

class Audio(object):
    
    'audio is played back as a separate thread. a lock must be acquired for \
    data to update to the original source managed by the tkinter window.'
    
    def __init__(self, fname, kwargs):
        
        lock = kwargs['lock']
        line = kwargs['line']
        mfcc_ax = kwargs['mfcc']
        class_ax = kwargs['class']
        model = kwargs['model']
        
        wf = wave.open('clean/' + fname, 'rb')
        
        n_steps = 4
        step = 4410
        rate = 44100
        p = pyaudio.PyAudio()
        stream = p.open(format=p.get_format_from_width(2),
                        channels=1,
                        rate=rate,
                        output=True)
        
        queue = []
        for i in range(n_steps-1):
            queue.append(np.array(np.random.rand(step)))
        
        wav_bytes = wf.readframes(step)
        prev_wav = np.zeros(step)
        
        while len(wav_bytes) > 0:

            lock.acquire()
            
            mfcc_ax.clear()
            mfcc_ax.set_title('MFCC (64 filters)')
            mfcc_ax.grid(False)
            mfcc_ax.set_xticks([])
            mfcc_ax.set_yticks([])
            X = np.zeros((9,64))
            mfcc_ax.imshow(X, cmap='hot', interpolation='nearest')
            
            class_ax.clear()
            class_ax.grid(False)
            class_ax.set_xticks([])
            class_ax.set_yticks([])
            X = np.zeros((10,1))
            class_ax.imshow(X, cmap='hot', interpolation='nearest')
            
            wav = np.frombuffer(wav_bytes, dtype=np.int16)
            X_mfcc = mfcc(wav, rate, numcep=64, nfilt=64, nfft=1103).T
            
            if X_mfcc.shape == (64,9):
                X = (X_mfcc + 137.4) / (112 + 137.4)
                mfcc_ax.imshow(X.T, cmap='hot', interpolation='nearest')
                X = X.reshape(1, X.shape[0], X.shape[1], 1)
                y_pred = model.predict(X).reshape(10)
                x = y_pred.reshape(10,1)
                class_ax.imshow(x, cmap='cool', interpolation='nearest')
            
            if wav.shape[0] < step:
                wav = np.concatenate((prev_wav, wav), axis=0)[-step:]
            
            queue.append(wav)
            if len(queue) > n_steps:
                queue.pop(0)
                
            line.set_ydata(np.concatenate(queue, axis=0))
            lock.release()
            stream.write(wav_bytes)
            wav_bytes = wf.readframes(step)
        
        stream.stop_stream()
        stream.close()
        p.terminate()
        self.__str__()
    
    def __str__(self):
        return 'done'

class App(tk.Tk):
    
    'main tkinter window. handles all figures and audio threads as nessesary'
    
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        self.wm_title('Instrument Classifier')
        self.bind('<Escape>',lambda e: self.destroy())
        self.geometry('1200x700')
        self.init_figures()
        self.init_model()
        self.lock = threading.Lock()
        self.build_kwargs()
        
        self.playing = False
        
        for i in range(26):
            self.grid_rowconfigure(i, weight=1)
        for y in range(6):
            self.grid_columnconfigure(y, weight=1)
        self.time_canvas = FigureCanvasTkAgg(self.time_fig, master=self)
        self.time_canvas.draw()
        self.time_canvas.get_tk_widget().grid(row=0, column=0,
                                         padx=60, pady=10, rowspan=14)
        
        self.mfcc_canvas = FigureCanvasTkAgg(self.mfcc_fig, master=self)
        self.mfcc_canvas.draw()
        self.mfcc_canvas.get_tk_widget().grid(row=14, column=0, padx=10, pady=10,
                                              rowspan=4)
        
        self.class_canvas = FigureCanvasTkAgg(self.class_fig, master=self)
        self.class_canvas.draw()
        self.class_canvas.get_tk_widget().grid(row=10, column=1, padx=10, pady=10,
                                            rowspan=10)
        
        self.after(0, self.update_canvas)
        
        ttk.Style().configure('TButton', height=10, width=14)
        buttons = []
        for c in classes:
            buttons.append(ttk.Button(self, text=c,
                                      command=lambda c=c: self.thread_audio(c)))
        
        i=0
        for x in range(5):
            for y in range(2):
                buttons[i].grid(row=x+3, column=y+1, padx=10)
                i+=1
    
    def update_canvas(self):
        'continually schedules updates to canvas based on audio state'
        self.lock.acquire()
        
        if not self.playing:
            self.clear_figures()
        
        self.time_canvas.draw()
        self.mfcc_canvas.draw()
        self.class_canvas.draw()
        self.lock.release()
        self.after(25, self.update_canvas)
    
    def init_model(self):
        yaml_file = open('mfcc.yaml', 'r')
        loaded_model_yaml = yaml_file.read()
        yaml_file.close()
        loaded_model = model_from_yaml(loaded_model_yaml)
        loaded_model.load_weights('mfcc.h5')
        loaded_model.compile(loss='categorical_crossentropy', optimizer='adam')
        self.model = loaded_model
        self.model._make_predict_function()
    
    def init_figures(self):
        #time figure
        n_steps = 4
        step = 4410
        rate = 44100
        
        self.time_fig, ax = plt.subplots(1, figsize=(10, 5))
        x = np.arange(0, n_steps * step, 1)
        self.line, = ax.plot(x, np.random.rand(len(x)), '-', lw=2)
        
        half = int(np.power(2,16)/2)
        ax.set_title('Time Series')
        ax.set_xlabel('Length')
        ax.set_ylabel('Amplitude')
        ax.set_ylim(-half, half-1)
        ax.set_xlim(0, n_steps * step)
        plt.setp(ax, yticks=[])
        length = n_steps*step/rate
        self.t = np.arange(0, length, 1/rate)
        
        #mfcc figure
        self.mfcc_fig, self.mfcc_ax = plt.subplots(1, figsize=(10, 2))
        self.mfcc_ax.set_title('MFCC (64 filters)')
        X = np.zeros((9,64))
        self.mfcc_ax.imshow(X, cmap='hot', interpolation='nearest')
        
        #class figure
        self.class_fig, self.class_ax = plt.subplots(1, figsize=(1, 4))
        X = np.zeros((10,1))
        self.class_ax.imshow(X, cmap='hot', interpolation='nearest')
        
    def build_kwargs(self):
        self.kwargs = {}
        self.kwargs['line'] = self.line
        self.kwargs['mfcc'] = self.mfcc_ax
        self.kwargs['class'] = self.class_ax
        self.kwargs['lock'] = self.lock
        self.kwargs['model'] = self.model
    
    def clear_figures(self):
        ff = 12
        y = 10*np.sin(2 * np.pi * ff * self.t)
        noise = 10* np.random.normal(0,2,self.t.shape[0])
        y = y + noise
        self.line.set_ydata(y)
        
        self.mfcc_ax.clear()
        self.mfcc_ax.grid(False)
        self.mfcc_ax.set_xticks([])
        self.mfcc_ax.set_yticks([])
        X = np.zeros((9,64))
        self.mfcc_ax.imshow(X, cmap='hot', interpolation='nearest')
        
        self.class_ax.clear()
        self.mfcc_ax.set_title('MFCC (64 filters)')
        self.class_ax.grid(False)
        self.class_ax.set_xticks([])
        self.class_ax.set_yticks([])
        X = np.zeros((10,1))
        self.class_ax.imshow(X, cmap='hot', interpolation='nearest')
    
    def thread_audio(self, c):
        'creates separate manager thread to keep track if audio is playing'
        if not self.playing:
            self.playing = True
            thread = threading.Thread(target=self.load_audio, args=(c,))
            thread.start()
    
    def load_audio(self, c):
        'plays audio back and returns upon terminatation'
        n = np.random.randint(30)
        index = get_index[c][n]
        fname = df.loc[index, 'fname']
        thread = ThreadWithReturn(target=Audio,
                                  args=(fname, self.kwargs))
        thread.start()
        result = str(thread.join())
        if result == 'done':
            self.playing = False

df = pd.read_csv('instruments.csv')
classes = list(np.unique(df.label))
df.sort_values(by='label', inplace=True)
df.reset_index(drop=True, inplace=True)
indices = [list(range(i,i+30)) for i in range(0,df.shape[0],30)]
get_index = dict(zip(classes, indices))

app = App()
app.mainloop()
