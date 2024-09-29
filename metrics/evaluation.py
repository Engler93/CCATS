"""
Source: https://github.com/ML4ITS/TimeVQVAE/
FID, IS, JS divergence.

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

import torch
import numpy as np


#from preprocessing.preprocess_ucr import DatasetImporterUCR
from supervised_FCN.example_pretrained_model_loading import load_pretrained_FCN
from supervised_FCN.example_compute_FID import calculate_fid
from supervised_FCN.example_compute_IS import calculate_inception_score
from einops import rearrange

def time_to_timefreq(x, n_fft: int, C: int):
    """
    x: (B, C, L)
    """
    x = rearrange(x, 'b c l -> (b c) l')
    x = torch.stft(x, n_fft, normalized=False, return_complex=True)
    x = torch.view_as_real(x)  # (B, N, T, 2); 2: (real, imag)
    x = rearrange(x, '(b c) n t z -> b (c z) n t ', c=C)  # z=2 (real, imag)
    return x  # (B, C, H, W)


def timefreq_to_time(x, n_fft: int, C: int):
    x = rearrange(x, 'b (c z) n t -> (b c) n t z', c=C).contiguous()
    x = torch.view_as_complex(x)
    x = torch.istft(x, n_fft, normalized=False, return_complex=False)
    x = rearrange(x, '(b c) l -> b c l', c=C)
    return x

class EvaluatorVQVAE(object):
    """
    - FID
    - IS
    - visual inspection
    - PCA
    - t-SNE
    """
    def __init__(self, data_name, mean, std, device='cuda', batch_size = 256):
        self.subset_dataset_name = data_name
        self.device = device
        self.batch_size = batch_size
        self.mean = mean
        self.std = std

        # load the pretrained FCN
        self.fcn = load_pretrained_FCN(data_name).to(self.device)
        self.fcn.eval()

        # load the numpy matrix of the test samples
        #dataset_importer = DatasetImporterUCR(subset_dataset_name, data_scaling=True)
        #n_fft = self.config['VQ-VAE']['n_fft']
        #self.X_test = timefreq_to_time(time_to_timefreq(torch.from_numpy(self.X_test), n_fft, 1), n_fft, 1)

    def compute_metrics(self, generated_data, generated_data_labels, real_data, batch_size):
        #generated_data = generated_data * self.std + self.mean
        #real_data = real_data * self.std + self.mean
        generated_data = (generated_data - self.mean) / self.std
        #real_data = (real_data - real_data.mean()) / real_data.std()
        print(generated_data.mean(), generated_data.std())
        print(real_data.mean(), real_data.std())

        self.batch_size = batch_size
        with torch.no_grad():
            generated_data = torch.tensor(generated_data).to(self.device).float()#.swapaxes(1, 2)
            real_data = torch.tensor(real_data).to(self.device).float()#.swapaxes(1, 2)

            #generated_data_labels = torch.tensor(generated_data_labels).to(self.device)
            #acc = float(torch.sum(y_pred.argmax(dim=1) == generated_data_labels.argmax(dim=1)).cpu().numpy()) / \
            #      y_pred.shape[0]

            score, std = self.inception_score(generated_data)
            z_test, z_gen = self.compute_z(generated_data, real_data)
            distance, _ = self.fid_score(z_test, z_gen)
        print("inception", score)
        print("fid", distance)
        return dict(inception=score, fid=distance, trts=0)

    def compute_z(self, X_gen: torch.Tensor, X_test: torch.Tensor) -> (np.ndarray, np.ndarray):
        """
        It computes representation z given input x
        :param X_gen: generated X
        :return: z_test (z on X_test), z_gen (z on X_generated)
        """
        n_samples = X_test.shape[0]
        n_iters = n_samples // self.batch_size
        if n_samples % self.batch_size > 0:
            n_iters += 1

        # get feature vectors from `X_test` and `X_gen`
        z_test, z_gen = [], []
        for i in range(n_iters):
            s = slice(i * self.batch_size, (i + 1) * self.batch_size)

            z_t = self.fcn(X_test[s].float().to(self.device),
                           return_feature_vector=True).cpu().detach().numpy()

            z_test.append(z_t)

        n_samples_gen = X_gen.shape[0]
        n_iters_gen = n_samples_gen // self.batch_size
        if n_samples % self.batch_size > 0:
            n_iters += 1
        for i in range(n_iters_gen):
            s = slice(i * self.batch_size, (i + 1) * self.batch_size)
            z_g = self.fcn(X_gen[s].float().to(self.device), return_feature_vector=True).cpu().detach().numpy()

            z_gen.append(z_g)
        z_test, z_gen = np.concatenate(z_test, axis=0), np.concatenate(z_gen, axis=0)
        return z_test, z_gen

    def fid_score(self, z_test: np.ndarray, z_gen: np.ndarray) -> (int, (np.ndarray, np.ndarray)):
        fid = calculate_fid(z_test, z_gen)
        return fid, (z_test, z_gen)

    def inception_score(self, X_gen: torch.Tensor):
        # assert self.X_test.shape[0] == X_gen.shape[0], "shape of `X_test` must be the same as that of `X_gen`."

        n_samples = X_gen.shape[0]
        n_iters = n_samples // self.batch_size
        if n_samples % self.batch_size > 0:
            n_iters += 1

        # get the softmax distribution from `X_gen`
        p_yx_gen = []
        for i in range(n_iters):
            s = slice(i * self.batch_size, (i + 1) * self.batch_size)

            p_yx_g = self.fcn(X_gen[s].float().to(self.device))  # p(y|x)
            p_yx_g = torch.softmax(p_yx_g, dim=-1).cpu().detach().numpy()

            p_yx_gen.append(p_yx_g)
        p_yx_gen = np.concatenate(p_yx_gen, axis=0)

        IS_mean, IS_std = calculate_inception_score(p_yx_gen)
        return IS_mean, IS_std
