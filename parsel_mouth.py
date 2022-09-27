import os,glob
import numpy as np 
import librosa
from sklearn.model_selection import train_test_split

from numpy.linalg import  det
import parselmouth

from parselmouth.praat import call
import soundfile as sf
from spafe.features.lpc import lpc, lpcc

def energy(frame):
    """Computes signal energy of frame"""
    print(frame)
    return np.sum(frame ** 2) / np.float64(len(frame))


def energy_entropy(frame, n_short_blocks=10):
    """Computes entropy of energy"""
    # total frame energy
    frame_energy = np.sum(frame ** 2)
    frame_length = len(frame)
    sub_win_len = int(np.floor(frame_length / n_short_blocks))
    if frame_length != sub_win_len * n_short_blocks:
        frame = frame[0:sub_win_len * n_short_blocks]

    # sub_wins is of size [n_short_blocks x L]
    sub_wins = frame.reshape(sub_win_len, n_short_blocks, order='F').copy()

    # Compute normalized sub-frame energies:
    s = np.sum(sub_wins ** 2, axis=0) / (frame_energy + eps)

    # Compute entropy of the normalized sub-frame energies:
    entropy = -np.sum(s * np.log2(s + eps))
    return entropy
def extractFeatures(fileName):
    y, sr = librosa.load(fileName)

    result=np.array([])
#The LIBROSA library includes MFCC, chroma, mel spectrogram, contrast, tonnetz, pitch, ZCR

    # extracting chroma features
    stft=np.abs(librosa.stft(y))
    chroma=np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T,axis=0)
    result=np.hstack((result, chroma))

    # extracting mfcc features
    mfccs=np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    result=np.hstack((result, mfccs))

    # extracting mel features
    mel=np.mean(librosa.feature.melspectrogram(y=y, sr=sr).T,axis=0)
    result=np.hstack((result, mel))
    
    # extracting contrast features
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sr).T,axis=0)
    result=np.hstack((result, contrast))
    
    # extracting tonnetz features
    
    y = librosa.effects.harmonic(y)
    tonnetz = np.mean(librosa.feature.tonnetz(y=y, sr=sr).T,axis=0)
    result=np.hstack((result, tonnetz))
    
    # extracting zcr features
    
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=y).T, axis=0)
    result=np.hstack((result, zcr)) # stacking horizontally
    
    # extracting pitch features
    pitch = np.mean(librosa.effects.pitch_shift(y, sr=sr, n_steps=4).T,axis=0)
    
    result=np.hstack((result, pitch)) # stacking horizontally



# The PARSELMOUTH library for praat software-based features includes duration, harmonic, jit-
# ter, shimmer, and formant

    sound = parselmouth.Sound(fileName) # read the sound
    
    
    # extracting harmonicity features
    harmonicity = call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
    hnr = call(harmonicity, "Get mean", 0, 0)
    result=np.hstack((result, hnr)) # stacking horizontally
    
    f0min=75
    f0max=500
    
     # extracting jitter features
    pointProcess = call(sound, "To PointProcess (periodic, cc)", f0min, f0max)
    localJitter = call(pointProcess, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
    result=np.hstack((result, localJitter)) # stacking horizontally
    
    # extracting shimmer features
    localShimmer =  call([sound, pointProcess], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    result=np.hstack((result, localShimmer)) # stacking horizontally
    
    # extracting formant features
    formant = call(sound, "To Formant (burg)", 0.0, 5, 5500, 0.025, 100)
    
    result=np.hstack((result, formant)) # stacking horizontally
    pitchh = parselmouth.praat.call(sound, "To Pitch (cc)", 0.0, 75.0, 15, True, 0.03, 0.45, 0.01, 0.35, 0.14, 600.0)
    # extracting duration features
    duration = pitchh.get_end_time() 
    result=np.hstack((result, duration)) # stacking horizontally
    
# The SPAFE library includes LPCC and PLP


#     LPCC –This method is an improvement over LPC. It has been derived to overcome the
#     channel effects. The Inverse Fourier Transform is performed to obtain the cepstrum and
#     classified into real cepstrum, complex cepstrum, power cepstrum, and phase cepstrum, and
#     speech analysis uses power ceptrum from it

# extracting LPCC features
    sig, fs = sf.read(fileName, dtype="float64")
    #sig- a mono audio signal (Nx1) from which to compute features.
    #fs- the sampling frequency of the signal we are working with. Default is 16000.
    #lpcc returns 2d array of LPCC features 
    
#     print("sig=",sig)
    try:
        feature=spafe.features.lpc.lpcc(sig=sig, fs=fs)

        result=np.hstack((result, feature)) # stacking horizontally
    except:
        pass
#         print("singular matrix")
    try:
        feature_plp=spafe.features.rplp.plp(sig=sig, fs=fs)

        result=np.hstack((result, feature_plp)) # stacking horizontally
    except:
        pass

#     engy=energy(sound)
#     etpy=energy_entropy(sound)

    return result
def loadData(testDataSizeRatio = 0.4):
    emotions = {
            '01':'neutral',
            '02':'calm',
            '03':'happy',
            '04':'sad',
            '05':'angry',
            '06':'fearful',
            '07':'disgust',
            '08':'surprised'
            }

    x,y = [] , []
    limit=60
    for folder in glob.glob('/Users/paras/Downloads/archive/audio_speech_actors_01-24/Actor_*'):
        for file in glob.glob(folder+'/*.wav'):
            if limit==0:
                break
            limit=limit-1
            fileName = os.path.basename(file)
            print(fileName)
            emotion = emotions[fileName.split('-')[2]]
            feature = extractFeatures(file)
            x.append(feature)
            y.append(emotion)

    return train_test_split(np.array(x), y, test_size = testDataSizeRatio, random_state = 9)
x_train,x_test,y_train,y_test=loadData()

Output exceeds the size limit. Open the full output data in a text editor
Output exceeds the size limit. Open the full output data in a text editor
03-01-05-01-02-01-16.wav
03-01-06-01-02-02-16.wav
03-01-06-02-01-02-16.wav
03-01-05-02-01-01-16.wav
03-01-07-01-01-01-16.wav
03-01-04-01-01-02-16.wav
03-01-04-02-02-02-16.wav
03-01-07-02-02-01-16.wav
03-01-08-02-02-01-16.wav
03-01-08-01-01-01-16.wav
03-01-03-02-02-02-16.wav
03-01-03-01-01-02-16.wav
03-01-02-02-01-01-16.wav
03-01-01-01-02-02-16.wav
03-01-02-01-02-01-16.wav
03-01-03-02-01-01-16.wav
03-01-03-01-02-01-16.wav
03-01-02-02-02-02-16.wav
03-01-02-01-01-02-16.wav
03-01-01-01-01-01-16.wav
03-01-06-01-01-01-16.wav
03-01-05-01-01-02-16.wav
03-01-05-02-02-02-16.wav
03-01-06-02-02-01-16.wav
03-01-04-01-02-01-16.wav
...
03-01-02-01-01-01-16.wav
03-01-02-02-02-01-16.wav
03-01-03-01-02-02-16.wav
03-01-03-02-01-02-16.wav
print((x_train.shape[0], x_test.shape[0]))
print(f'Features extracted: {x_train.shape[1]}')
