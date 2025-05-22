# _*_ coding:utf-8 _*_

import numpy as np
import torch


class StandardScaler:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit_transform(self, y):
        self.mean = torch.mean(y)
        self.std = torch.std(y) + 1e-4
        return (y - self.mean) / self.std, self.mean, self.std

    # def inverse_transform(self, y):
    #     return y * self.std + self.mean

    def inverse_transform(self, y, mean, std):
        self.mean = mean
        self.std = std
        return y * self.std + self.mean

    def transform(self, y, mean, std):
        self.mean = mean
        self.std = std
        return (y - self.mean) / self.std



class MaxScaler:
    def __init__(self):
        self.max = None

    def fit_transform(self, y):
        self.max = torch.max(y)
        return y / self.max, self.max

    def inverse_transform(self, y, max):
        self.max = max
        return y * self.max

    def transform(self, y, max):
        self.max = max
        return y / self.max


class MeanScaler:
    def __init__(self):
        self.mean = None

    def fit_transform(self, y):
        self.mean = torch.mean(y)
        return y / self.mean, self.mean

    def inverse_transform(self, y, mean):
        self.mean = mean
        return y * self.mean

    def transform(self, y, mean):
        self.mean = mean
        return y / self.mean


class LogScaler:

    def fit_transform(self, y):
        return torch.log1p(y)

    def inverse_transform(self, y):
        return torch.expm1(y)

    def transform(self, y):
        return torch.log1p(y)

def smooth(scalar, weight=0.5):
    smoothed = []
    last = scalar[0]
    for point in scalar:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed
