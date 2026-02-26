import pandas as pd
import numpy as np
#from scipy.signal import butter, filtfilt

import matplotlib.pyplot as plt
import matplotlib.animation as animation

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

'''
def banpassFilter(signal, lowcut=20, highcut=450, fs=2000, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal, axis=0)
'''

# primer: raw_signal.shape = (uzorci, kanali)

'''
datasetovi su oblika 8x8senzora + labela (klasa e [0,3])
'''

data0 = pd.read_csv('emgSignaliKlasa0.CSV', header=None)
data1 = pd.read_csv('emgSignaliKlasa1.CSV', header=None)
data2 = pd.read_csv('emgSignaliKlasa2.CSV', header=None)
data3 = pd.read_csv('emgSignaliKlasa3.CSV', header=None)

data = np.concatenate([data0, data1, data2, data3], axis=0)

ind = np.random.permutation(data.shape[0])
data_mix = data[ind, :]

ulaz = data_mix[:, :-1]
izlaz = data_mix[:, -1]

ulaz_trening, ulaz_test, izlaz_trening, izlaz_test = train_test_split(ulaz, izlaz, test_size=0.2, shuffle=True, random_state=45, stratify=izlaz)



print(data.shape)

'''
print(data0.shape)
print(data1.shape)
print(data2.shape)
print(data3.shape)
'''

'''
izlaz0 = data0.iloc[:, -1].values
izlaz1 = data1.iloc[:, -1].values
izlaz2 = data2.iloc[:, -1].values
izlaz3 = data3.iloc[:, -1].values

svi_izlazi = np.concatenate([izlaz0, izlaz1, izlaz2, izlaz3])
'''

'''
plt.figure()
plt.hist(svi_izlazi, bins=20, color='skyblue', edgecolor='black')
plt.xlabel("Vrednost izlaza")
plt.ylabel("Broj uzoraka")
plt.title("Histogram izlaza po svih 4 dataset-a")
plt.show()
'''

#prikaz prvih 8 uzastopnih trenutaka, umesto K ubaciti klasu
'''
signal = dataK.iloc[0, :-1].values
#label = data1.iloc[0, -1]

signal_reshaped = signal.reshape(8, 8)  # (time, sensors)

fs = 200
dt = 1/fs  # 0.005 s

t = np.arange(8) * dt   # samo 8 vremenskih tačaka

plt.figure(figsize=(8,4))
plt.plot(t, signal_reshaped[:, 0])  # senzor 0 kroz 8 trenutaka
plt.title("EMG senzor 0 - 40ms segment")
plt.xlabel("Vreme [s]")
plt.ylabel("Amplituda")
plt.show()

plt.figure(figsize=(10,6))
for i in range(8):
    plt.plot(t, signal_reshaped[:, i], label=f"Senzor {i}")

plt.xlabel("Vreme [s]")
plt.ylabel("Amplituda")
plt.title("8 senzora kroz 8 uzastopnih trenutaka (40ms)")
plt.legend()
plt.show()
'''


# za CNN dimenzije(8, 8, 1)