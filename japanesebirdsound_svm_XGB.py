import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV ,train_test_split
from sklearn.svm import SVC, NuSVC
import sys
path = "/home/jerry/libsvm-3.24/python"
sys.path.append(path)
from svmutil import *
import xgboost
from sklearn import ensemble, metrics
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sn
import matplotlib.pyplot as plt
import pandas as pd

# #load Training & Testing Data
x_train = np.load("x_train_japan_STFT.npy")
x_test = np.load("x_test_japan_STFT.npy")
y_train = np.load("y_train_japan_STFT.npy")
y_test = np.load("y_test_japan_STFT.npy")

print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

# libsvm
m = svm_train(y_train, x_train, '-s 0 -t 0 -m 3000')
p_label, p_acc, p_val = svm_predict(y_test, x_test, m)


#XGBoost
XGG = XGBClassifier(
        # # Number of trees
        # n_estimators=280,
        # # control overfitting 1st
        # max_depth=5, #normal 6-10
        # gamma=0, #0-infinite
        # min_child_weight=0,
        # # overfitting tunning 2nd
        # subsample=1,
        # colsample_bytree=1,
        # # # if increase eta ,need to decrease num_round
        # eta=0.36,
        # # num_boost_round=50,
        # # # 控制模型複雜度的權重值的L2正則化項引數，引數越大，模型越不容易過擬合。
        # reg_lambda=1,
        # # #最大增量步長，我們允許每個樹的權重估計。
        # max_delta_step=0,       
        # # # 生成樹時進行的列取樣
        # tree_method='auto',
        # scale_pos_weight=1,
)

#train
# XGG.fit(x_train, y_train)
# # 預測值
# y_pred=XGG.predict(x_test)
# # 真實值 賦值
# y_true = y_test

# # 計算精度
# print("train Accuracy :",XGG.score(x_train, y_train))
# print("test Accuracy : %.5g" % metrics.accuracy_score(y_true, y_pred))