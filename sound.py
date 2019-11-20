# library to play and record sounds
import sounddevice as sd
import soundfile as sf
from math import pi
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft

# define sampling frequency (samples per second):
Fs = 20000
# duration of the test signal or recording in seconds:
duration = 2

#print(sd.default.device)

# create a one-dimensional array from 0 to duration (start to end) with spacing (time interval) of Ts=1/Fs:
t = np.linspace(0, duration, duration*Fs)

# create a test signal with frequency in Hz:
f = 500
# np.sin() function takes an ndarray as argument, where the sin is calculated for every point of the array. n is discrete time:
#x = np.sin(2*pi*f*t) + np.sin(2*pi*2.4*f*t) + np.sin(2*pi*6.75*f*t)

# the problem is that python (or the application) has not requested permission to access the internal microphone on
# the mac. In system preferences -> security and privacy -> Privacy we see the applications that have requested permissions,
# but we cannot add more. A possible solution was proposed: do the following (possibly with a sudo to get superuser access):
# rm -rf ~/Library/Application\ Support/com.apple.TCC
# (https://www.reddit.com/r/MacOS/comments/9lwyz0/mojave_not_privacy_settings_blocking_all_mic/)
# get full disk access to terminal by going to Full disk access and clicking on terminal.
# Problem solved! I need to run it from terminal, so it requests access to the microphone.

print("recording...")
# first parameter is the number of frames (samples) to record, ie. duration * Fs. Stores in numpy array
x = sd.rec(duration * Fs, samplerate=Fs, channels=1)
# wait for recording to finish, otherwise it records noise:
sd.wait()

# playback the recording:
sd.play(x, Fs)
# write it to file as .wav:
#sf.write("./out.wav", x, Fs)
print(np.shape(x))

# plot two plots in two rows and one column, this is the first one:
plt.subplot(2, 1, 1)
plt.xlabel("Time (s)")
plt.ylabel("Magnitude")
plt.plot(t, x)

# generate frequency axis:
n = np.size(t)  # calculate the number of elements of the array t (time samples) (the length of the samples)
# the FFT produces the same length as output, but we use the first half
# linspace creates an array from start to end with specific number of points, half of the time samples for frequency:
frequency = np.linspace(0, Fs/2, n/2)

# perform FFT:
X = fft(x)
# get only the first half of the samples:
X_m = (2/n) * X[0:np.size(frequency)]

# This is the second plot of the frequency below:
plt.subplot(2,1,2)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Spectrum Magnitude")
plt.plot(frequency, abs(X_m))
plt.tight_layout()

plt.show()