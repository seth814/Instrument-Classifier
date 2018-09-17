# Instrument-Classifier
Classifies instruments in real time. <br/>

Uses data from the freesound audio tagging competition: <br/>
https://www.kaggle.com/c/freesound-audio-tagging

## Install Instructions
1. Clone repository.
2. Download data from google drive link: https://drive.google.com/open?id=1_M-XuuhBKo2gjjAnaAe5Bq5GEFPLSh6U
3. Extract the 3 directories (clean, samples, wavfiles) and place them with all the other files.
4. Make sure some unique dependencies are installed.
  - pip install python_speech_features
  - pip install tqdm
  - pip install pyaudio
  
  A working version of tensorflow-gpu was used during testing. I'd imagine tensorflow-cpu could work just as well.
  
## File Descriptions (run order)
audio_eda.py - run this first to see overview of audio methods. this generates clean files if clean directory is empty.<br/>
model.py - builds the cnn to run on mfcc features.<br/>
app.py - runs the tkinter application to randomly pick instruments for realtime playback and classification.<br/>

It should be possible to only run app.py, but if you're having any problems it might be best to run those three files to make sure everything got built properly.
