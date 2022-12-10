import matplotlib.pyplot as plt
import numpy as np
import csv

F="audio_samples_periodic_hf/unified_freqdomain_1670690467.0895236.csv"
datas = []
with open(F) as f:
    r = csv.reader(f)
    for row in r:
        a = np.array([float(x) for x in row[1:130]])
        b = np.array([float(x) for x in row[130:129*2+1]])
        c = np.array([float(x) for x in row[129*2+1:129*3+1]])
        d = np.array([float(x) for x in row[129*3+1:129*4+1]])
        datas.append((a,b,c,d))

fig, axs = plt.subplots(4,1)

axs[0].plot(np.arange(1,129,1), datas[20][0][1:])
axs[1].plot(np.arange(1,129,1), datas[20][1][1:])
axs[2].plot(np.arange(1,129,1), datas[20][2][1:])
axs[3].plot(np.arange(1,129,1), datas[20][3][1:])
plt.show()
print()


