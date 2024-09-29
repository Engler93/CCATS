"""
Evaluate diffusion model.
"""

import warnings

import numpy as np
import torch
from torch import nn

from data_loading import load_ucr_dataset
from metrics.evaluation import EvaluatorVQVAE

from metrics.run_CAS import train_synthetic_test_real_FCN
import math

def re_evaluate(config, checkpoint_dir, metric_iteration=5, test_eval=False):
    """
    Run evaluation based on diffusion model from checkpoint directory
    """
    from generate import diffusion
    config['iteration'] = 0

    ## Data loading
    data_name = config['data_name']
    x_train_min_max, y_train_min_max, x_test_min_max, y_test_min_max, mean_gen, std_gen = load_ucr_dataset(data_name, config['ds_base_dir'], normalize=config.get("normalize", "min_max"))
    x_train, y_train, x_test, y_test, mean, std = load_ucr_dataset(data_name, config['ds_base_dir'], normalize="mean_std")

    config['mean'] = mean
    config['std'] = std
    if config['batch_size'] == 'auto':
        config['batch_size'] = 1000
    if len(x_train) < config['batch_size']:
        config['batch_size'] = len(x_train)
        warnings.warn("Batch size larger than dataset size, reducing batch size to: " + str(len(x_train)))
    if test_eval:
        x_val = x_test
        y_val = y_test
    else:
        x_val = x_train
        y_val = y_train

    ## Performance metrics
    # Output initialization
    metric_results = dict()

    metrics = dict()
    config['checkpoint_dir'] = checkpoint_dir  # os.path.join(checkpoint_dir, "checkpoint")

    evaluator = EvaluatorVQVAE(data_name, mean=mean.detach().cpu().numpy(), std=std.detach().cpu().numpy())

    if test_eval:
        # compute number of samples
        num_samples_fid = max(x_val.shape[0], 256)
        num_samples_cas = max(x_train.shape[0], 1000)

        oversamplings = max(1, math.ceil(float(max(num_samples_cas,num_samples_fid))/float(x_train.shape[0])))
        x_train = x_train.repeat(oversamplings, 1, 1)
        print("SHAPE", x_train.shape)
        y_train = y_train.repeat(oversamplings)

    else:
        num_samples_fid = x_train.shape
        num_samples_cas = x_train.shape

    for tt in range(metric_iteration):
        idx = torch.randperm(x_train.shape[0])
        idx_fid = idx[:num_samples_fid]
        idx_cas = idx[:num_samples_cas]

        generated_data = diffusion(x_train, y_train, config)
        # rescale
        generated_data = generated_data * std_gen.detach().cpu().numpy() + mean_gen.detach().cpu().numpy()

        print(generated_data.mean(), generated_data.std())
        # FID, TRTS, ITS
        scores = evaluator.compute_metrics(generated_data[idx_fid], y_train[idx_fid], x_val, config['batch_size'])
        for key in scores:
            if key not in metrics.keys():
                metrics[key] = []
            metrics[key].append(scores[key])

        # TSTR
        # only in test_eval do more than one tstr run
        if tt > 0 and not test_eval:
            continue

        temp_tstr = train_synthetic_test_real_FCN(x_val, y_val, (generated_data[idx_cas] - mean.detach().cpu().numpy())/std.detach().cpu().numpy(), y_train[idx_cas], max_epochs=1000,
                                              batch_size=256,
                                              device=config.get('device', 'cuda'))

        if 'tstr' not in metrics.keys():
            metrics['tstr'] = []
        metrics['tstr'].append(temp_tstr)

    for key in metrics.keys():
        metric_results[key] = np.mean(metrics[key])
        metric_results[key + '_std'] = np.std(metrics[key])
    return metric_results

