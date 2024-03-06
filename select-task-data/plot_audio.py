import wave
import numpy as np
import matplotlib.pyplot as plt

# Open the audio file
audio = wave.open('data/1.wav', 'r')

# Read the audio data
signal = audio.readframes(-1)
signal = np.frombuffer(signal, dtype=np.int16)

# Get the parameters of the audio file
framerate = audio.getframerate()
nchannels = audio.getnchannels()
sampwidth = audio.getsampwidth()

audio.close()

# Plot the audio data
plt.figure(1)
plt.title('Audio Waveform')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.plot(signal)
plt.savefig('data/1.png')
