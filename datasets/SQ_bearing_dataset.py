"""
SQ bearing dataset
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


Fs = 25600  # 采样频率
lab = [x for x in range(7)]
data_load_path = 'R:\★My Researches\★公开数据集\SQ bearing dataset/'

files_9Hz = {'NC': ['REC3629_ch2.txt', 'REC3630_ch2.txt', 'REC3631_ch2.txt'],
             'Outer_1': ['REC3500_ch2.txt', 'REC3501_ch2.txt', 'REC3502_ch2.txt'],
             'Outer_2': ['REC3482_ch2.txt', 'REC3483_ch2.txt', 'REC3484_ch2.txt'],
             'Outer_3': ['REC3464_ch2.txt', 'REC3465_ch2.txt', 'REC3466_ch2.txt'],
             'Inner_1': ['REC3520_ch2.txt', 'REC3521_ch2.txt', 'REC3522_ch2.txt'],
             'Inner_2': ['REC3607_ch2.txt', 'REC3608_ch2.txt', 'REC3609_ch2.txt'],
             'Inner_3': ['REC3585_ch2.txt', 'REC3586_ch2.txt', 'REC3587_ch2.txt']}

files_19Hz = {'NC': ['REC3632_ch2', 'REC3633_ch2.txt', 'REC3634_ch2.txt'],
             'Outer_1': ['REC3503_ch2.txt', 'REC3504_ch2.txt', 'REC3505_ch2.txt'],
             'Outer_2': ['REC3485_ch2.txt', 'REC3486_ch2.txt', 'REC3487_ch2.txt'],
             'Outer_3': ['REC3467_ch2.txt', 'REC3468_ch2.txt', 'REC3469_ch2.txt'],
             'Inner_1': ['REC3523_ch2.txt', 'REC3524_ch2.txt', 'REC3525_ch2.txt'],
             'Inner_2': ['REC3610_ch2.txt', 'REC3611_ch2.txt', 'REC3612_ch2.txt'],
             'Inner_3': ['REC3588_ch2.txt', 'REC3589_ch2.txt', 'REC3590_ch3.txt']}

files_29Hz = {'NC': ['REC3635_ch2.txt', 'REC3636_ch2.txt', 'REC3637_ch2.txt'],
             'Outer_1': ['REC3506_ch2.txt', 'REC3507_ch2.txt', 'REC3508_ch2.txt'],
             'Outer_2': ['REC3488_ch2.txt', 'REC3489_ch2.txt', 'REC3490_ch2.txt'],
             'Outer_3': ['REC3470_ch2.txt', 'REC3471_ch2.txt', 'REC3472_ch2.txt'],
             'Inner_1': ['REC3526_ch2.txt', 'REC3527_ch2.txt', 'REC3528_ch2.txt'],
             'Inner_2': ['REC3613_ch2.txt', 'REC3614_ch2.txt', 'REC3615_ch2.txt'],
             'Inner_3': ['REC3591_ch2.txt', 'REC3592_ch2.txt', 'REC3593_ch2.txt']}

files_39Hz = {'NC': ['REC3638_ch2.txt', 'REC3639_ch2.txt', 'REC3640_ch2.txt'],
             'Outer_1': ['REC3510_ch2.txt', 'REC3511_ch2.txt', 'REC3512_ch2.txt'],
             'Outer_2': ['REC3491_ch2.txt', 'REC3492_ch2.txt', 'REC3493_ch2.txt'],
             'Outer_3': ['REC3473_ch2.txt', 'REC3474_ch2.txt', 'REC3475_ch2.txt'],
             'Inner_1': ['REC3529_ch2.txt', 'REC3530_ch2.txt', 'REC3531_ch2.txt'],
             'Inner_2': ['REC3616_ch2.txt', 'REC3617_ch2.txt', 'REC3618_ch2.txt'],
             'Inner_3': ['REC3594_ch2.txt', 'REC3595_ch2.txt', 'REC3596_ch2.txt']}



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
    :return:
    """

    if str(speed) in ['9', '19', '29', '39']:
        file_dict = eval('files_' + str(speed) + 'Hz')
    else:
        assert "Input speed is not involved"

    sample_len = 4096  # the length of the signal to calculate the spectrum
    if domain == 'Time':
        sample_len = 1024  # the length of time-domain samples
    else:
        pass

    data, labels = [], []
    for cond in file_dict:
        data_group, labels_group = [], []
        lab = list(file_dict).index(cond)  # 在字典中的索引顺序作为标签
        files = file_dict[cond]
        for filename in files:
            file_path = os.path.join(data_load_path, cond, filename)
            sig = pd.read_csv(file_path, sep='\\s+', header=15).values[:, [1]]  # 第二列为振动信号
            sig = sig[0: int(len(sig) / 4096) * 4096]  # 对信号截断

            if snr != None:
                sig = Add_noise(snr)(sig)

            start, end = 0, sample_len
            while end <= sig.shape[0]:
                data_group.append(sig[start: end])
                labels_group.append(lab)
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
        ff = np.arange(0, int(nfft/2)) / nfft * Fs
        ff = ff[0: 1024] / Fs  # 以 Fs 归一化
    else:
        pass
    print(data.shape, labels.shape)

    return data, labels, ff


class SQ_bearing_dataset(object):
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

        print('train_X.shape: {}, train_Y.shape: {}'.format(str(train_X.shape), str(train_Y.shape)))
        print('val_X.shape: {}, val_Y.shape: {}'.format(str(val_X.shape), str(val_Y.shape)))
        print('test_X.shape: {}, test_Y.shape: {}'.format(str(test_X.shape), str(test_Y.shape)))

        train_set = {"data": train_X, "label": train_Y}
        val_set = {"data": val_X, "label": val_Y}
        test_set = {"data": test_X, "label": test_Y}

        train_set = dataset(train_set, transform=data_transforms(self.normlizetype))
        val_set = dataset(val_set, transform=data_transforms(self.normlizetype))
        test_set = dataset(test_set, transform=data_transforms(self.normlizetype))

        return train_set, val_set, test_set, ff


if __name__ == '__main__':
    train_set, val_set, test_set, ff = SQ_bearing_dataset(speed=9, snr=None, normlizetype=None, domain='Frequency').data_preprare()
    data = train_set.data
    print(np.mean(data[1]), np.std(data[1]))
    print(np.mean(data), np.std(data))
    pass

