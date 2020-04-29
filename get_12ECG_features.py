#!/usr/bin/env python

import numpy as np
from scipy.signal import butter, lfilter  # Butterworth滤波器
from scipy import stats   #  包含统计函数


 
def bandpass_filter(data, lowcut, highcut, signal_freq, filter_order):
        """
        Method responsible for creating and applying Butterworth filter. 创建和使用Butterworth滤波器
        :param deque data: raw data
        :param float lowcut: filter lowcut frequency value
        :param float highcut: filter highcut frequency value
        :param int signal_freq: signal frequency in samples per second (Hz)
        :param int filter_order: filter order
        :return array: filtered data
        """
        nyquist_freq = 0.5 * signal_freq
        low = lowcut / nyquist_freq
        high = highcut / nyquist_freq
        b, a = butter(filter_order, [low, high], btype="band")
        y = lfilter(b, a, data)
        return y


#def get_12ECG_features(data, header_data):

def get_12ECG_features(data):
    data_train=[]
    data_train_temp=[] 
    #bandpass
    filter_lowcut = 0.001
    filter_highcut = 15.0
    filter_order = 1
    sample_Fs = 500
    for i in range(12):
        data_temp = bandpass_filter(data[i], lowcut=filter_lowcut, highcut=filter_highcut, signal_freq=sample_Fs, filter_order=filter_order)#    
    #Normalized
        data_mean = np.mean(data_temp)
        data_std = np.std(data_temp,ddof=1)
        data_new = [(x-data_mean)/data_std for x in data_temp]
        data_train.append(data_new)
    for i in range(12):
        data_train_temp.append(data_train[i][0:3000])
    data_train = np.array(data_train_temp)
    
    return data_train


