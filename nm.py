import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt

import matplotlib.pyplot as plt
import matplotlib.animation as animation

def banpassFilter(signal, lowcut=20, highcut=450, fs=2000, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal, axis=0)

# primer: raw_signal.shape = (uzorci, kanali)

'''
datasetovi su oblika 8x8senzora + labela (klasa e [0,3])
'''

data0 = pd.read_csv('emgSignaliKlasa0.CSV', header=None)
data1 = pd.read_csv('emgSignaliKlasa1.CSV', header=None)
data2 = pd.read_csv('emgSignaliKlasa2.CSV', header=None)
data3 = pd.read_csv('emgSignaliKlasa3.CSV', header=None)

'''
print(data0.shape)
print(data1.shape)
print(data2.shape)
print(data3.shape)
'''

izlaz0 = data0.iloc[:, -1].values
izlaz1 = data1.iloc[:, -1].values
izlaz2 = data2.iloc[:, -1].values
izlaz3 = data3.iloc[:, -1].values

svi_izlazi = np.concatenate([izlaz0, izlaz1, izlaz2, izlaz3])

'''
plt.figure()
plt.hist(svi_izlazi, bins=20, color='skyblue', edgecolor='black')
plt.xlabel("Vrednost izlaza")
plt.ylabel("Broj uzoraka")
plt.title("Histogram izlaza po svih 4 dataset-a")
plt.show()
'''
signal = data0.iloc[:500, :-1].values
labels = data0.iloc[:500, -1].values 

# signal.shape = (500, 64) → 500 uzoraka, 64 kanala
fs = 200  # Hz
dt = 1/fs  # 0.05
t = np.arange(0, signal.shape[0]*dt, dt)
plt.figure(figsize=(10,4))
plt.plot(t, signal[:,0])
plt.title("EMG kanal 0 - 20s signal")
plt.xlabel("Vreme [s]")
plt.ylabel("Amplituda")
plt.show()

filtered_signal = banpassFilter(signal)

# Koristiti sliding window??
# za CNN dimenzije(8, 8, 1)