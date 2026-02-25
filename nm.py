import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

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

plt.figure()
plt.hist(svi_izlazi, bins=20, color='skyblue', edgecolor='black')
plt.xlabel("Vrednost izlaza")
plt.ylabel("Broj uzoraka")
plt.title("Histogram izlaza po svih 4 dataset-a")
plt.show()