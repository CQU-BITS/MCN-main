
import numpy as np
import random
from scipy.signal import resample


class Add_noise(object):
    def __init__(self, snr_db):
        self.snr_db = snr_db

    def __call__(self, signal):
        signal_power = np.mean(signal ** 2)  # 计算信号功率
        noise_power = signal_power / (10 ** (self.snr_db / 10))  # 计算噪声功率
        noise = np.sqrt(noise_power) * np.random.randn(*signal.shape)
        # 添加噪声后的信号
        noisy_signal = signal + noise
        return noisy_signal


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, seq):
        for t in self.transforms:
            seq = t(seq)
        return seq


class Reshape(object):
    def __call__(self, seq):
        #print(seq.shape)
        return seq.transpose()


class Retype(object):
    def __call__(self, seq):
        return seq.astype(np.float32)


class AddGaussian(object):
    def __init__(self, sigma=0.01):
        self.sigma = sigma
    def __call__(self, seq):
        return seq + np.random.normal(loc=0, scale=self.sigma, size=seq.shape)


class RandomAddGaussian(object):
    def __init__(self, sigma=0.01):
        self.sigma = sigma
    def __call__(self, seq):
        if np.random.randint(2):
            return seq
        else:
            return seq + np.random.normal(loc=0, scale=self.sigma, size=seq.shape)


class Scale(object):
    def __init__(self, sigma=0.01):
        self.sigma = sigma
    def __call__(self, seq):
        scale_factor = np.random.normal(loc=1, scale=self.sigma, size=(seq.shape[0], 1))
        scale_matrix = np.matmul(scale_factor, np.ones((1, seq.shape[1])))
        return seq*scale_matrix


class RandomScale(object):
    def __init__(self, sigma=0.01):
        self.sigma = sigma
    def __call__(self, seq):
        if np.random.randint(2):
            return seq
        else:
            scale_factor = np.random.normal(loc=1, scale=self.sigma, size=(seq.shape[0], 1))
            scale_matrix = np.matmul(scale_factor, np.ones((1, seq.shape[1])))
            return seq*scale_matrix


class RandomStretch(object):
    def __init__(self, sigma=0.3):
        self.sigma = sigma
    def __call__(self, seq):
        if np.random.randint(2):
            return seq
        else:
            seq_aug = np.zeros(seq.shape)
            len = seq.shape[1]
            length = int(len * (1 + (random.random()-0.5)*self.sigma))
            for i in range(seq.shape[0]):
                y = resample(seq[i, :], length)
                if length < len:
                    if random.random() < 0.5:
                        seq_aug[i, :length] = y
                    else:
                        seq_aug[i, len-length:] = y
                else:
                    if random.random() < 0.5:
                        seq_aug[i, :] = y[:len]
                    else:
                        seq_aug[i, :] = y[length-len:]
            return seq_aug


class RandomCrop(object):
    def __init__(self, crop_len=20):
        self.crop_len = crop_len
    def __call__(self, seq):
        if np.random.randint(2):
            return seq
        else:
            max_index = seq.shape[1] - self.crop_len
            random_index = np.random.randint(max_index)
            seq[:, random_index:random_index+self.crop_len] = 0
            return seq


class Normalize(object):
    def __init__(self, type="mean~std"):  # "0-1","1-1","mean-std"
        self.type = type
    def __call__(self, seq):
        if self.type == "0~1":
            seq = (seq-seq.min())/(seq.max()-seq.min())
        elif self.type == "-1~1":
            seq = 2*(seq-seq.min())/(seq.max()-seq.min()) + -1
        elif self.type == "mean~std":
            # print(seq.shape, seq.mean().shape)
            seq = (seq-seq.mean())/seq.std()
        elif self.type == None:
            seq = seq
        else:
            raise NameError('This normalization is not included!')
        return seq