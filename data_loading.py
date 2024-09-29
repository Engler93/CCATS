
import numpy as np
import os
import math
import torch
from torch.utils.data import Dataset


def load_ucr_dataset(data_name, ds_base_dir='data/UCRArchive_2018', normalize='min_max'):
  """Time-series datasets loading

  Args:
      dataset_name (str): Name of the dataset.
  """
  data_train = np.loadtxt(os.path.join(ds_base_dir, data_name, "{}_TRAIN.tsv".format(data_name)), delimiter='\t')
  data_test = np.loadtxt(os.path.join(ds_base_dir, data_name, "{}_TEST.tsv".format(data_name)), delimiter='\t')



  x_train = torch.tensor(data_train[:, 1:]).float().unsqueeze(1)
  y_train = torch.tensor(data_train[:, 0].astype('int16')).long()
  x_test = torch.tensor(data_test[:, 1:]).float().unsqueeze(1)
  y_test = torch.tensor(data_test[:, 0].astype('int16')).long()
  min = torch.tensor(np.nanmin(x_train))
  max = torch.tensor(np.nanmax(x_train))

  if normalize == 'mean_std':
    mean = x_train.mean()
    std = x_train.std()
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std
  elif normalize == 'min_max':
    #print(x_train)
    #min = x_train.min()
    #max = x_train.max()
    x_train = (x_train - min) / (max-min) * 2 - 1
    x_test = (x_test - min) / (max-min) * 2 - 1
    std = (max - min) / 2.0
    mean = (max+min) / 2.0
    #print(min, max)
    #print(x_train)
    #exit()
  else:
    raise NotImplementedError

  if y_train.min() > 0:
    y_train -= y_train.min()
    y_test -= y_test.min()

  if y_train.min() == -1:
    y_train[y_train == -1] = 0
    y_test[y_test == -1] = 0

  return x_train, y_train, x_test, y_test, mean, std