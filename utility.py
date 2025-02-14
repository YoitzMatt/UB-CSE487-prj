import librosa
import numpy as np
import soundfile as sf
import pandas as pd
import math
import matplotlib.pyplot as plt

FRAME_LENGTH = 1024
HOP_LENGTH = 512

# sliceAudio is provided in a separate file in an effort to reduce
# the amount of storage neeed to run the project and upload it to
# our GitHub repository

# enter duration in seconds
# input is wav file
# output is wav file sliced down to duration seconds

def sliceAudio(duration, input, output):
    audioArray, sampleRate = librosa.load(input)
    secondsPerSample = 1/sampleRate
    samplesForDuration = duration/secondsPerSample
    print("\nNumber of samples for duration: " + str(samplesForDuration))
    print("Desired duration: " + str(samplesForDuration/sampleRate) + "\n")
    slicedAudio = audioArray[0:int(samplesForDuration)]
    sf.write(output, slicedAudio, sampleRate)


def amplitudeEnvelope(signal, frameLength, hopLength):
    envelope = []
    for i in range(0, len(signal), hopLength):
        frameAE = max(signal[i:i+frameLength])
        envelope.append(frameAE)
    return np.array(envelope)

def removeAmbience(inputWav):
    audioArray, sampleRate = librosa.load(inputWav)
    
    # filter out the most minimal signals
    magSpec, phase = librosa.magphase(librosa.stft(audioArray)) # D = S*P
    specFilter = librosa.decompose.nn_filter(magSpec, aggregate=np.median,
                    metric='cosine', width=int(librosa.time_to_frames(2, sr=sampleRate)))
    specFilter = np.minimum(magSpec, specFilter)

    # apply masks over spectrogram magnitude
    ambientMargin = 2
    ambientMask = librosa.util.softmask(specFilter,
                        (ambientMargin*(magSpec-specFilter)), power=2)
    foregroundMargin = 10
    foregroundMask = librosa.util.softmask((magSpec-specFilter),
                        (foregroundMargin*specFilter), power=2)
    ambientSpec = ambientMask*magSpec
    foregroundSpec = foregroundMask*magSpec

    # reconstruct foreground signal
    complexSpec = foregroundSpec*phase

    #CLEANING/PREPROCESSING STEP 11: pad lost samples from reconstruction
    reconstructSignal = librosa.istft(complexSpec)
    padding = len(audioArray)-len(reconstructSignal)
    reconstructPadded = np.pad(reconstructSignal, (0, padding), 'constant', constant_values=(0, 0))
    
    return reconstructPadded, sampleRate

#plot the amplitude envelope
def drawAE(input, filterAmbience):
    if filterAmbience:
        audioArray, sampleRate = removeAmbience(input)
    else:
        audioArray, sampleRate = librosa.load(input)
    AE = amplitudeEnvelope(audioArray, FRAME_LENGTH, HOP_LENGTH)
    frames = range(0, math.ceil(len(audioArray)/512))
    framesToTime = librosa.frames_to_time(frames, hop_length=512)
    plt.figure(figsize=(8, 4))
    librosa.display.waveshow(audioArray, alpha=0.5)
    plt.title("Amplitude Envelope")
    plt.ylabel("Amplitude")
    plt.ylim((-1, 1))
    plt.plot(framesToTime, AE, color="r")
    plt.show()
    
    #print average
    print(np.average(AE))

#plot the root mean squared energy
def drawRMSE(input, filterAmbience=False):
    if filterAmbience:
        audioArray, sampleRate = removeAmbience(input)
    else:
        audioArray, sampleRate = librosa.load(input)
    rms = librosa.feature.rms(y=audioArray, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)[0]
    frames = range(0, math.ceil(len(audioArray)/512))
    framesToTime = librosa.frames_to_time(frames, hop_length=512)
    plt.figure(figsize=(8, 4))
    librosa.display.waveshow(audioArray, alpha=0.5)
    plt.title("Root Mean Square Energy")
    plt.ylabel("Amplitude")
    plt.ylim((-1, 1))
    plt.plot(framesToTime, rms, color="r")
    plt.show()
    #print average
    print(np.average(rms))

#plot the zero crossing rate
def drawZCR(input, filterAmbience=False):
    if filterAmbience:
        audioArray, sampleRate = removeAmbience(input)
    else:
        audioArray, sampleRate = librosa.load(input)
    zcr = librosa.feature.zero_crossing_rate(y=audioArray, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)[0]
    frames = range(0, math.ceil(len(audioArray)/512))
    framesToTime = librosa.frames_to_time(frames, hop_length=512)
    plt.figure(figsize=(8, 4))
    plt.title("Zero Crossing Rate of Joe Biden")
    plt.ylim((0, 400))
    plt.plot(framesToTime, zcr*FRAME_LENGTH, color="r") #multiply by FRAME_SIZE to get non-normalized value
    plt.show()
    #print average
    print(np.average(zcr*FRAME_LENGTH))

#plot MFCC

def drawMFCC(input, filterAmbience=False):
    if filterAmbience:
        audioArray, sampleRate = removeAmbience(input)
    else:
        audioArray, sampleRate = librosa.load(input)
    mfcc20 = librosa.feature.mfcc(y=audioArray, n_mfcc=20, sr=sampleRate)
    #print average
    for i in range(len(mfcc20)):
        print("MFCC"+str(i+1) + ": " + str(np.average(mfcc20[i])) + "\n")

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfcc20, x_axis="time", sr=sampleRate)
    plt.ylabel("MFCCs")
    plt.title("MFCC Values")
    clb = plt.colorbar(format="%+2.f")
    clb.ax.set_xlabel("shape")
    #higher shape = louder/bass
    #lower shape = queiter/soprano
    plt.show()
    
#plot spectral centroid
def drawSC(input, filterAmbience=False):
    if filterAmbience:
        audioArray, sampleRate = removeAmbience(input)
    else:
        audioArray, sampleRate = librosa.load(input)
    frames = range(0, math.ceil(len(audioArray)/512))
    framesToTime = librosa.frames_to_time(frames, hop_length=512)
    SC = librosa.feature.spectral_centroid(y=audioArray, sr=sampleRate, n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH)[0]
    plt.figure(figsize=(10, 4))
    plt.plot(framesToTime, SC)
    plt.xlabel("Time")
    plt.ylabel("Spectral Mass")
    plt.show()
    #print average
    print(np.average(SC))

#plot spectral bandwidth
def drawSB(input, filterAmbience=False):
    if filterAmbience:
        audioArray, sampleRate = removeAmbience(input)
    else:
        audioArray, sampleRate = librosa.load(input)
    frames = range(0, math.ceil(len(audioArray)/512))
    framesToTime = librosa.frames_to_time(frames, hop_length=512)
    SB = librosa.feature.spectral_bandwidth(y=audioArray, sr=sampleRate, n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH)[0]
    plt.figure(figsize=(10, 4))
    plt.plot(framesToTime, SB)
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    #dipicts range/variance present in the current spectral centriod's frequencies
    plt.show()
    #print average
    print(np.average(SB))

#plot spectral rolloff
def drawSR(input, filterAmbience=False):
    if filterAmbience:
        audioArray, sampleRate = removeAmbience(input)
    else:
        audioArray, sampleRate = librosa.load(input)
    frames = range(0, math.ceil(len(audioArray)/512))
    framesToTime = librosa.frames_to_time(frames, hop_length=512)
    SR = librosa.feature.spectral_rolloff(y=audioArray, sr=sampleRate, n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH)[0]
    plt.figure(figsize=(10, 4))
    plt.plot(framesToTime, SR)
    plt.xlabel("Time")
    plt.ylabel("Frequencies")
    plt.show()
    #print average
    print(np.average(SR))