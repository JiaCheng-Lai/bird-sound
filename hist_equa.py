import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import cv2
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import glob
import librosa
from librosa import display
import math

##fff

my_path = '/home/jerry/plot_example/'
file = '/media/iml/jerry/JapaneseBirdSound/1/01-Track-01.wav'

def plot_spectrum(y):
    plt.figure(2, figsize=(25, 12))
    n_fft = 2048
    S = librosa.stft(y, n_fft=n_fft, hop_length=n_fft//2)
    # convert to db
    # (for your CNN you might want to skip this and rather ensure zero mean and unit variance)
    # D = librosa.amplitude_to_db(np.abs(S), ref=np.max)
    # average over file
    D_AVG = np.mean(S, axis=1)

    plt.bar(np.arange(D_AVG.shape[0]), D_AVG)
    x_ticks_positions = [n for n in range(0, n_fft // 2, n_fft // 16)]
    x_ticks_labels = [str(sr / 2048 * n) + 'Hz' for n in x_ticks_positions]
    plt.xticks(x_ticks_positions, x_ticks_labels)
    plt.xlabel('Frequency')
    plt.ylabel('dB')
    plt.savefig(my_path + "test.png")

def plot_STFT(y):
    D = librosa.stft(y, n_fft=96 ,win_length=96 ,window='hann', center=True)
    fig = plt.figure(figsize=[0.72,0.72])
    ax = fig.add_subplot(111)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_frame_on(False)
    librosa.display.specshow(librosa.amplitude_to_db(D**2,ref=np.max))
    plt.savefig(my_path+f'STFT2.jpg', dpi=400, bbox_inches='tight',pad_inches=0)

# Read audio
y , sr = librosa.load(file, sr=None, offset=10, duration=5)
original_STFT = librosa.stft(y)
original_STFT = abs(original_STFT)
print(original_STFT.max())
print(original_STFT.min())
# D = abs(librosa.stft(y, n_fft=n_fft, hop_length=n_fft//2))
# y = abs(y)
D = np.reshape(y,(350,315))
# D = abs(D)
print("largest num in D :", np.amax(D))
print("smallest num in D :", np.amin(D))
[m,n] = D.shape

# D = np.around(D, decimals=4)
# D[D<1]=D[D<1]*100
D = D*sr
D = D.astype(int)
print("largest num in D :", np.amax(D))
print("smallest num in D :", np.amin(D))
print(D, D.shape,type(D))

# hist,bins = np.histogram(D.flatten(),200,[0,200])

# cdf = hist.cumsum()
# cdf_normalized = cdf * hist.max()/ cdf.max()


# origin_img = cv2.imread(f_name[0] ,cv2.IMREAD_GRAYSCALE)
# [m,n] = origin_img.shape
h_arr = [0]*22050
h_arr_1 = [0]*22050
new_img = np.zeros_like(D)

#Calculate Original Histogram
for i in range(m):
    for j in range(n):
        h_arr[D[i][j]]+=1



plt.subplot(311)
plt.bar(range(22050),h_arr)
plt.xlabel('Frequency_Scale')
plt.ylabel('Data Points')
plt.title('Original Histogram')

#Transfer function
h_arr = np.array(h_arr)*22050/(m*n)
CDF = [sum(h_arr[:i+1]) for i in range(len(h_arr))]
TF = np.float16(CDF)

#Replace pixels with transfer function
for i in range(m):
    for j in range(n):
        new_img[i][j] = TF[D[i][j]]

#Calculate Equalized Histogram
for i in range(m):
    for j in range(n):
        h_arr_1[new_img[i][j]]+=1


# TF = np.reshape(TF,(315, 70))

print(TF.shape)
print(TF.max())
print(TF.min())

plt.figure(1)
plt.subplot(312)
plt.bar(range(22050),h_arr_1)
plt.xlabel('Frequency_Scale')
plt.ylabel('Data Points')
plt.title('Equalized Histogram')
plt.subplot(313)
plt.plot(TF)
plt.xlabel('Frequency_Scale')
plt.ylabel('Y') 
plt.title('Transform function')
plt.savefig(my_path + "example.png")

TF /= 22050
STFT_matrix = librosa.stft(TF)
STFT_matrix = abs(STFT_matrix)
# STFT_matrix = math.log(STFT_matrix, 2)
print(STFT_matrix.max())  
print(STFT_matrix.min())
print(STFT_matrix.shape)
print(np.count_nonzero(STFT_matrix == 0))

# plot_spectrum(TF)
# plot_STFT(X)