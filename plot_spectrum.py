import librosa
import numpy as np
import matplotlib.pyplot as plt
import math

my_path = '/home/jerry/plot_example/'
file = '/media/iml/jerry/JapaneseBirdSound/1/10-Track-10.wav'

# def sigmoid(x):
#     return 1 / (1 + math.exp(-x))

y_list = list()
# load the file
y, sr = librosa.load(file, sr=None, offset=5, duration=10)

print(y)
# for dot in y:
#     x = sigmoid(dot)
#     y_list.append(x)

n_fft = 2048
S = librosa.stft(y, n_fft=n_fft, hop_length=n_fft//2)
# convert to db
# (for your CNN you might want to skip this and rather ensure zero mean and unit variance)
print(S)
S = abs(S)
print(S)
# D = librosa.amplitude_to_db(S, ref=np.max)
# average over file
# D = abs(S)
# print(D, D.shape)
D_AVG = np.mean(S, axis=1)

plt.figure(figsize=(25, 12))

plt.bar(np.arange(D_AVG.shape[0]), D_AVG)
x_ticks_positions = [n for n in range(0, n_fft // 2, n_fft // 16)]
x_ticks_labels = [str(sr / 2048 * n) + 'Hz' for n in x_ticks_positions]
plt.xticks(x_ticks_positions, x_ticks_labels)
plt.xlabel('Frequency')
plt.ylabel('dB')
plt.savefig(my_path + 'spectrum_1.png')
# plt.show()