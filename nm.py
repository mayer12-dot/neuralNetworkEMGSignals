import pandas as pd

data0 = pd.read_csv('emgSignaliKlasa0.CSV', header=None)
data1 = pd.read_csv('emgSignaliKlasa1.CSV', header=None)
data2 = pd.read_csv('emgSignaliKlasa2.CSV', header=None)
data3 = pd.read_csv('emgSignaliKlasa3.CSV', header=None)

print(data0.shape)
print(data1.shape)
print(data2.shape)
print(data3.shape)