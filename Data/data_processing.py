import os
import sys
import math
import numpy as np
import pandas as pd
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd.function import Function
import torch.optim as optim
import scipy.io
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt



days_of_week = {
    "Monday": 1,
    "Tuesday": 2,
    "Wednesday": 3,
    "Thursday": 4,
    "Friday": 5,
    "Saturday": 6,
    "Sunday": 0
}


def data_preprocessing(input_len, window, t, day):
    # sample usage: data_preprocessing(15, 24, t, "Monday")
    #input_len is the length of time steps use to input the model
    #window is the time step of a day, (usually 24)
    # t is test step, definde which day is use for test. in our test, t is vary between 0 and 5, repeat 5 times
  
  index = days_of_week.get(day)
  
  data_path = 'Data/trafficFlow.csv'
  df = pd.read_csv(data_path)
  df = df.loc[8400:9264]
  df = df.reset_index(drop=True)

  X, y = [], []
  data = df['Vehicles']/10.0
  for i in range(len(df) - window):
    if i % 168 == 5 + 24*index:

      X.append(data[i: i + input_len])
      y.append(data[i: i + window])

  X = np.array(X)
  y = np.array(y)


  X_train = np.concatenate((X[:t], X[t+1:]), axis=0)
  y_train = np.concatenate((y[:t], y[t+1:]), axis=0)


  # X_train = X[0:4]
  # y_train = y[0:4]
  # X_train = X[0:total - 5]
  # y_train = y[0:total - 5]

  X_test = X[t].reshape(1,input_len)
  y_test = y[t].reshape(1,window)

  # X_test = X[total-5:]
  # y_test = y[total-5:]

  #---------

  dataset_train = {
    'samples': X_train,
    'labels': y_train
    }

  dataset_test = {
    'samples': X_test,
    'labels': y_test
  }
  return dataset_train, dataset_test, X_train, y_train, X_test, y_test


class TrafficDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return self.dataset['samples'].shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        qrs = self.dataset['samples'][idx, :].reshape(1, -1)
        target_label = self.dataset['labels'][idx].reshape(1, -1)  # [0]
        sample = [torch.from_numpy(qrs).float(), torch.from_numpy(target_label).float()]

        if self.transform:
            sample = self.transform(sample)

        return sample
    

def load_traffic_real(batch_size, dataset_train, dataset_test):
    """Args:
        batch_size (int): the dataset will be split into mini-batches of this size.
        data_path (string): Path to the mat file that contains both the QRS complexes and the target labels.
    """
    transform = None

    trainset = TrafficDataset(dataset_train, transform = transform)

    testset = TrafficDataset(dataset_test,transform=transform)

    trainloader = DataLoader(
        trainset, batch_size=batch_size, drop_last=False, shuffle=False, num_workers=0)
    testloader = DataLoader(testset, batch_size=batch_size,
                            drop_last=False, shuffle=False, num_workers=0)

    return trainloader, testloader
