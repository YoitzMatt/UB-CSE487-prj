import librosa
import numpy as np
import soundfile as sf

# This utility is provided in a separate file in an effort to reduce
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