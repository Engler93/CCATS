"""
CAS (= TSTR):
- Train on the Synthetic samples, and
- Test on the Real samples.

Smith, Kaleb E., and Anthony O. Smith. "Conditional GAN for timeseries generation." arXiv preprint arXiv:2006.16477 (2020).
Source: https://github.com/ML4ITS/TimeVQVAE
License: MIT License

Copyright (c) 2023 ML4ITS

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

# Modifications by Philipp Engler for evaluating CCATS model

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
#import pandas as pd
#import matplotlib.pyplot as plt

#from preprocessing.preprocess_ucr import DatasetImporterUCR
#from preprocessing.data_pipeline import build_data_pipeline
#from utils import load_yaml_param_settings, get_root_dir
#from evaluation.cas import SyntheticDataset
#from supervised_FCN.experiments.exp_train import ExpFCN as ExpFCN_original


from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import accuracy_score

from supervised_FCN.models.fcn import FCNBaseline
from supervised_FCN.utils import *

class ExpBase(pl.LightningModule):
    def __init__(self):
        super().__init__()

    def training_step(self, batch, batch_idx):
        raise NotImplemented

    def validation_step(self, batch, batch_idx):
        raise NotImplemented

    def on_training_epoch_end(self) -> None:
        return

    def training_step_end(self, out) -> None:
        return

    def on_validation_epoch_end(self) -> None:
        return

    def validation_step_end(self, out) -> None:
        return

    def configure_optimizers(self):
        raise NotImplemented

    def on_test_epoch_end(self) -> None:
        return

    def test_step_end(self, out) -> None:
        return

def detach_the_unnecessary(loss_hist: dict):
    """
    apply `.detach()` on Tensors that do not need back-prop computation.
    :return:
    """
    for k in loss_hist.keys():
        if k not in ['loss']:
            try:
                loss_hist[k] = loss_hist[k].detach()
            except AttributeError:
                pass


class ExpFCN_original(ExpBase):
    def __init__(self,
                 config: dict,
                 n_train_samples: int,
                 n_classes: int,
                 ):
        super().__init__()
        self.config = config
        self.T_max = config['trainer_params']['max_epochs'] * (np.ceil(n_train_samples / config['dataset']['batch_size']) + 1)
        in_channels = config['dataset']['in_channels']

        self.fcn = FCNBaseline(in_channels, n_classes)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.test_res = []
        self.val_res = []
        self.test_weights = []
        self.val_weights = []
        self.batch_idxs = []

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.float()
        y = y.squeeze()

        yhat = self.fcn(x)  # (b n_classes)
        loss = self.criterion(yhat, y)

        # lr scheduler
        sch = self.lr_schedulers()
        sch.step()

        # log
        acc = accuracy_score(y.flatten().detach().cpu().numpy(),
                             yhat.argmax(dim=-1).flatten().cpu().detach().numpy())
        loss_hist = {'loss': loss, 'acc': acc}

        detach_the_unnecessary(loss_hist)
        return loss_hist

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.float()
        y = y.squeeze()

        yhat = self.fcn(x)  # (b n_classes)
        loss = self.criterion(yhat, y)

        # log
        acc = accuracy_score(y.flatten().detach().cpu().numpy(),
                             yhat.argmax(dim=-1).flatten().cpu().detach().numpy())
        loss_hist = {'loss': loss, 'acc': acc}

        if batch_idx == 0:
            self.val_res.append([])
            self.val_weights = []
        self.val_res[-1].append(acc)
        self.val_weights.append(x.shape[0])
        self.batch_idxs.append(batch_idx)
        detach_the_unnecessary(loss_hist)
        return loss_hist

    def configure_optimizers(self):
        opt = torch.optim.AdamW([{'params': self.fcn.parameters(), 'lr': self.config['exp_params']['LR']},
                                 ], weight_decay=self.config['exp_params']['weight_decay'])
        return {'optimizer': opt, 'lr_scheduler': CosineAnnealingLR(opt, self.T_max)}

    def test_step(self, batch, batch_idx):
        x, y = batch
        x = x.float()
        y = y.squeeze()

        yhat = self.fcn(x)  # (b n_classes)
        loss = self.criterion(yhat, y)

        # log
        acc = accuracy_score(y.flatten().detach().cpu().numpy(),
                             yhat.argmax(dim=-1).flatten().cpu().detach().numpy())
        loss_hist = {'loss': loss, 'acc': acc}

        print("Test acc:", acc)
        self.test_res.append(acc)
        self.test_weights.append(x.shape[0])
        detach_the_unnecessary(loss_hist)

class ExpFCN(ExpFCN_original):
    def __init__(self,
                 config_fcn: dict,
                 n_train_samples: int,
                 n_classes: int,
                 ):
        super().__init__(config_fcn, n_train_samples, n_classes)
        self.config = config_fcn
        self.T_max = config_fcn['trainer_params']['max_epochs'] * (np.ceil(n_train_samples / config_fcn['dataset']['batch_size']) + 1)


class UCRDataset(Dataset):
    def __init__(self,
                 X, Y,
                 **kwargs):
        """
        :param kind: "train" / "test"
        :param dataset_importer: instance of the `DatasetImporter` class.
        """
        super().__init__()

        self.X, self.Y = X, Y.unsqueeze(1)

        self._len = self.X.shape[0]

    @staticmethod
    def _assign_float32(*xs):
        """
        assigns `dtype` of `float32`
        so that we wouldn't have to change `dtype` later before propagating data through a model.
        """
        new_xs = []
        for x in xs:
            new_xs.append(x.astype(np.float32))
        return new_xs[0] if (len(xs) == 1) else new_xs

    def getitem_default(self, idx):
        x, y = self.X[idx, :], self.Y[idx, :]
        #x = x[None, :]  # adds a channel dim
        return x, y

    def __getitem__(self, idx):
        return self.getitem_default(idx)

    def __len__(self):
        return self._len


def train_synthetic_test_real_FCN(x_test, y_test, x_gen, y_gen, batch_size=256, max_epochs=1000, device="cuda"):
    # load config
    #args = load_args()
    #config = load_yaml_param_settings(args.config)
    #config_cas = load_yaml_param_settings(args.config_cas)
    num_classes = int(y_test.max() + 1)
    # pre-settings
    config_cas = dict(
        dataset = {"in_channels": 1, "batch_size": batch_size, "num_workers": 0},
        exp_params = {"LR": 0.001, "weight_decay": 0.00001},
        trainer_params = {"gpus": [0], "max_epochs": 1000}
    )
    #config['trainer_params']['gpus'] = [args.gpu_device_idx]
    #config_cas['trainer_params']['gpus'] = [args.gpu_device_idx]
    #config_cas['dataset']['dataset_name'] = dataset_name

    #train_dataset = UCRDataset(x_train, y_train)
    gen_dataset = UCRDataset(x_gen, y_gen)
    test_dataset = UCRDataset(x_test, y_test)
    #real_train_data_loader = DataLoader(train_dataset, config_cas['dataset']['batch_size'], num_workers=config_cas['dataset']['num_workers'], shuffle=True, drop_last=False, pin_memory=True)
    train_data_loader = DataLoader(gen_dataset, config_cas['dataset']['batch_size'], num_workers=config_cas['dataset']['num_workers'], shuffle=True, drop_last=False, pin_memory=True)
    real_test_data_loader = DataLoader(test_dataset, config_cas['dataset']['batch_size'], num_workers=config_cas['dataset']['num_workers'], shuffle=True, drop_last=False, pin_memory=True)


    # fit
    train_exp = ExpFCN(config_cas, len(train_data_loader.dataset), num_classes)
    trainer = pl.Trainer(enable_checkpointing=False,
                         callbacks=[LearningRateMonitor(logging_interval='epoch')],
                         devices=config_cas['trainer_params']['gpus'],
                         accelerator='gpu',
                         max_epochs=config_cas['trainer_params']['max_epochs'])
    trainer.fit(train_exp,
                train_dataloaders=train_data_loader,
                val_dataloaders=real_test_data_loader, )

    # test
    trainer.test(train_exp, real_test_data_loader)
    res = np.array(train_exp.test_res)
    weights = np.array(train_exp.test_weights)
    res_last = (res * weights).sum() / weights.sum()
    print("Res:", res_last)
    res = np.array(train_exp.val_res[1:])
    weights = np.array(train_exp.val_weights)
    res = (res * weights).sum(axis=1) / weights.sum()
    res = res.max()
    print("Res max:", res)
    return res_last
