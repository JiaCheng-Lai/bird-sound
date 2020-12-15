import random
from tqdm import tqdm
import time
import os
import struct
import matplotlib.pyplot as plt
import IPython.display as ipd
import pandas as pd
import numpy as np
import librosa
import librosa.display
my_path = "/home/jerry/japanesebirdsound_plot_train/"

path = '/media/iml/jerry/JapaneseBirdSound/'
data = pd.read_csv("JapaneseBirdSound.csv")

# y,sr = librosa.load(path, sr=None, offset=5, duration=10)

N_FFT = 1024       
HOP_SIZE = 1024       
N_MELS = 60            
WIN_SIZE = 1024      
WINDOW_TYPE = 'hann' 
FEATURE = 'mel'      
FMIN = 500

x_train = []
x_test = []

y_train = [*range(1,421)]
y_train = np.array(y_train)
y_test = [*range(1,421)]
y_test = np.array(y_test)

#training data
for i in tqdm(range(len(data))):
    fold_no=str(data.iloc[i]["fold"])
    file=data.iloc[i]["slice_file_name"]
    label=data.iloc[i]["classID"]
    filename=path+fold_no+"/"+file
    
    y,sr = librosa.load(filename, sr=22050, offset=5, duration=2)
    
    D = np.reshape(y,(210,210))
    [m,n] = D.shape

    # D = np.around(D, decimals=4)
    # D[D<1]=D[D<1]*100
    D = D*sr
    D = D.astype(int)

    h_arr = [0]*22050
    h_arr_1 = [0]*22050
    new_img = np.zeros_like(D)

    #Calculate Original Histogram
    for i in range(m):
        for j in range(n):
            h_arr[D[i][j]]+=1

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
    TF /= 22050
    STFT_matrix = librosa.stft(TF)
    STFT_matrix = abs(STFT_matrix)

    x_train.append(STFT_matrix)

#testing data
for i in tqdm(range(len(data))):
    fold_no=str(data.iloc[i]["fold"])
    file=data.iloc[i]["slice_file_name"]
    label=data.iloc[i]["classID"]
    filename=path+fold_no+"/"+file
    
    y,sr = librosa.load(filename, sr=22050, offset=15, duration=2)

    D = np.reshape(y,(210,210))
    [m,n] = D.shape


    # D = np.around(D, decimals=4)
    # D[D<1]=D[D<1]*100
    D = D*sr
    D = D.astype(int)

    h_arr = [0]*22050
    h_arr_1 = [0]*22050
    new_img = np.zeros_like(D)

    #Calculate Original Histogram
    for i in range(m):
        for j in range(n):
            h_arr[D[i][j]]+=1

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
    TF /= 22050
    STFT_matrix = librosa.stft(TF)
    STFT_matrix = abs(STFT_matrix)

    x_test.append(STFT_matrix)


x_train = np.array(x_train)
x_test = np.array(x_test)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

#reshaping into 2d to save in csv format
# x_train_2d=np.reshape(x_train,(x_train.shape[0],x_train.shape[1]*x_train.shape[2]))
# x_test_2d=np.reshape(x_test,(x_test.shape[0],x_test.shape[1]*x_test.shape[2]))
# print(x_train_2d.shape, x_test_2d.shape)

# np.save("x_train_japan_stft",x_train,allow_pickle=True)
# np.save("x_test_japan_stft",x_test,allow_pickle=True)
# np.save("y_train_japan_360",y_train,allow_pickle=True)
# np.save("y_test_japan_360",y_test,allow_pickle=True)

# train1 = pd.DataFrame(x_train)
# train2 = pd.DataFrame(x_test)
# train3 = pd.DataFrame(y_train)
# train4 = pd.DataFrame(y_test)

# train1.to_csv("x_train_japan.csv",index=False)
# train2.to_csv("x_test_japan.csv",index=False)
# train3.to_csv("y_train_japan_360.csv",index=False)
# train4.to_csv("y_test_japan_360.csv",index=False)