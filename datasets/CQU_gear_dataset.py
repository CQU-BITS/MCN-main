"""
CQU (Chongqing University) Gearbox dataset form a two-stage Gearbox
Available at:
"""


import os
import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from datasets.sequence_aug import *
from tqdm import tqdm
from torch.utils.data import Dataset
import scipy.fftpack as fftpack
from scipy.fft import fft  # FFT分析需要的包
from scipy import signal


Fs = 20480  # 采样频率
lab = [0, 1, 2, 3, 4]  # The filename of vibration signals data was labeled to 0-4
data_load_path = 'R:/★My Researches/★公开数据集/CQU gearbox dataset/ConstantSpeed/'



def data_transforms(normlize_type=None):
    transforms = Compose([
        Normalize(normlize_type),
        Retype()])
    return transforms


class dataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.transforms = transform

        self.data = list(self.transforms(dataset['data']))
        self.labels = dataset['label'].tolist()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        seq = self.data[item]
        label = self.labels[item]
        return seq, label


def get_data(speed, snr, domain):
    """
    :param speed: The rotating speed in rpm
    :param snr: The signal-to-noise in dB
    :param domain: 'Time' or 'Frequency'
    """

    sample_len = 10240  # the length of the signal to calculate the spectrum
    if domain == 'Time':
        sample_len = 1024  # the length of time-domain samples
    else:
        pass

    data, labels = [], []
    for ii in range(len(lab)):
        data_group, labels_group = [], []
        filename = str(lab[ii]+1) + '_' + str(speed) + '_0.02.mat'
        data_root = os.path.join(data_load_path, filename)
        sig = loadmat(data_root)['Signal'][0][0]['y_values'][0][0]['values'][:, [1]]  # 通道2（传感器v2)的数据被使用
        sig = sig[0: min(int(len(sig) / 10240), 300) * 10240]  # 对信号截断

        if snr != None:
            sig = Add_noise(snr)(sig)

        if domain == 'Time':
            sig = sig[::2, :]
        else:
            pass

        start, end = 0, sample_len
        while end <= sig.shape[0]:
            data_group.append(sig[start: end])
            labels_group.append(lab[ii])
            start += sample_len
            end += sample_len

        data.append(data_group)
        labels.append(labels_group)

    data = np.array(data)
    data = np.transpose(data, (1, 0, 3, 2))  # data >> [num_for_each_classes, num_classes, 1, sample_len]
    labels = np.array(labels)  # label >> [num_for_each_classes, num_classes]
    labels = np.transpose(labels, (1, 0))

    ff = []
    if domain == 'Frequency':
        nfft = int(np.power(2, np.ceil(np.log2(sample_len))))
        data = signal.detrend(data, axis=-1)
        data = 2 * abs(fft(data, nfft, axis=-1)) / sample_len
        data = data[:, :, :, 0: 1024]
        ff = np.arange(0, int(nfft / 2)) / nfft * Fs
        ff = ff[0: 1024] / Fs  # 以 Fs 归一化
    else:
        pass
    print(data.shape, labels.shape)

    return data, labels, ff


class CQU_gear_dataset(object):
    in_channels = 1
    num_classes = len(lab)

    def __init__(self, speed, snr, normlizetype, domain='Frequency'):
        self.speed = speed
        self.snr = snr
        self.normlizetype = normlizetype

        if domain in ['Time', 'Frequency']:
            self.domain = domain
        else:
            raise NameError('This normalization is not included!')


    def data_preprare(self):
        data, labels, ff = get_data(speed=self.speed, snr=self.snr, domain=self.domain)
        # 顺序划分，生成训练集、验证集、测试集
        train_X, val_X, train_Y, val_Y = train_test_split(data, labels, train_size=1/3, shuffle=False)
        val_X, test_X, val_Y, test_Y = train_test_split(val_X, val_Y, train_size=1/2, shuffle=False)

        train_X, train_Y = train_X.reshape(-1, 1, 1024), train_Y.reshape(-1)
        val_X, val_Y = val_X.reshape(-1, 1, 1024), val_Y.reshape(-1)
        test_X, test_Y = test_X.reshape(-1, 1, 1024), test_Y.reshape(-1)

        train_set = {"data": train_X, "label": train_Y}
        val_set = {"data": val_X, "label": val_Y}
        test_set = {"data": test_X, "label": test_Y}

        train_set = dataset(train_set, transform=data_transforms(self.normlizetype))
        val_set = dataset(val_set, transform=data_transforms(self.normlizetype))
        test_set = dataset(test_set, transform=data_transforms(self.normlizetype))

        return train_set, val_set, test_set, ff


if __name__ == '__main__':
    train_set, val_set, test_set, ff = CQU_gear_dataset(speed=200, snr=None, normlizetype=None, domain='Frequency').data_preprare()
    data = train_set.data
    print(np.mean(data[1]), np.std(data[1]))
    print(np.mean(data), np.std(data))
    pass

