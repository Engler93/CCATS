"""
Perform a single run of Training a Diffusion Model.
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore")

import torch
from sacred import Experiment
from sacred.observers import file_storage
import math
import numpy as np
import random
from eval import re_evaluate
import trainer
import unet_1d
import unet_1d_blocks

from generate import diffusion
from data_loading import load_ucr_dataset

from torch.utils.data import DataLoader

# setup for sacred logging
EXP_FOLDER = './exp/'
exp = Experiment(os.path.splitext(os.path.basename(sys.argv[0]))[0])

log_location = os.path.join(EXP_FOLDER)
if len(exp.observers) == 0:
    print('Adding a file observer in %s' % log_location)
    exp.observers.append(file_storage.FileStorageObserver(log_location))


# define config parameters, used by sacred for logging and scheduling runs
# noinspection PyUnusedLocal
@exp.config
def config():
    # dataset
    data_name = 'TwoPatterns'
    # number of iterations to train over
    iteration = 40000
    # size of mini-batches, capped by train set size
    batch_size = 100
    # learning rate
    lr = 1e-4
    # type of loss, only 'mse' implemented
    loss = 'mse'
    # number of diffusion steps in forward (and backward) process
    diffusion_steps = 1000
    # way of conditioning the DDPM on classes: None for no conditioning, label for class-conditioning
    cond = "label"
    # fraction of labels to randomly drop during training for classifier-free guidance
    drop_labels = 0
    # guidance strength for classifier-free guidance, 0 for no additional guidance
    w = 0
    # load model with given exp_id (given as integer)
    load_model = None
    # normalization
    normalize = "min_max"
    # path to pretrained models for computing evaluation metrics
    eval_model_path = "exp/_pretrained/"
    # path to base directory where all the datasets are stored
    ds_base_dir = 'data/UCRArchive_2018'

def run_experiment(args, checkpoint_dir=None):
    """
    run a single experiment, checkpoint_dir can be used for loading a pre-trained checkpoint
    """
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    args['device'] = device
    if checkpoint_dir is None:
        ## Data loading
        data_name = args['data_name']

        x_train, y_train, x_test, y_test, mean, std = load_ucr_dataset(data_name, args['ds_base_dir'], normalize=args.get("normalize", "min_max"))
        # fix for logging of number of iterations
        args['iterations_original'] = args['iteration']

        train_len = len(x_train)
        if train_len < args['batch_size']:
            args['batch_size'] = train_len
            warnings.warn("Batch size larger than dataset size, reducing batch size to: " + str(len(x_train)))

        print(data_name + ' dataset is ready.')

        args['checkpoint_dir'] = None

        args['mean'] = mean
        args['std'] = std
        diffusion(x_train, y_train, args, device=device)
        res = re_evaluate(args, "exp/_models/" + args['exp_id'] + "_", metric_iteration=3, test_eval=True)
        print(res)
    else:
        res = re_evaluate(args, "exp/_models/"+str(checkpoint_dir)+"_", metric_iteration=3, test_eval=True)
        print(res)

    print("Exp:", args['exp_id'])

@exp.automain
def main(data_name,
         iteration,
         batch_size,
         lr,
         loss,
         diffusion_steps,
         cond,
         drop_labels,
         w,
         load_model,
         normalize,
         eval_model_path,
         ds_base_dir
         ):
    exp_id = str(os.path.split(exp.observers[0].dir)[-1])
    args = dict(data_name=data_name,
                iteration=iteration,
                batch_size=batch_size,
                lr=lr,
                loss=loss,
                diffusion_steps=diffusion_steps,
                cond=cond,
                drop_labels=drop_labels,
                w=w,
                load_model=load_model,
                normalize=normalize,
                eval_model_path=eval_model_path,
                ds_base_dir=ds_base_dir,
                exp_id=exp_id
                )
    return run_experiment(args, checkpoint_dir=load_model)
