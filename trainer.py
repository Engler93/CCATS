"""
Train diffusion model and / or generate data.
"""

import math
import os.path as pt

import torch
from torch import nn
import torch.nn.functional as F
from tqdm.auto import tqdm
import os


class Trainer():
    def __init__(self, model, noise_scheduler, opt, lr_scheduler, loss, config, device='cuda'):
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.opt = opt
        self.lr_scheduler = lr_scheduler
        self.total_steps = 0
        self.last_checkpoint_dir = None
        self.device = device
        self.losses = {'loss': []}
        self.loss = loss
        self.config = config
        self.best_model = self.create_checkpoint_dict()

    def generate_synthetic_data(self, condition, num_samples=None, batch_size=200):
        gen_data = []
        assert (num_samples is None and condition is not None) or (condition is None and num_samples is not None) or num_samples==condition.shape[0]
        if num_samples is None:
            num_samples = condition.shape[0]
        for iter_eval in range(math.ceil(num_samples / batch_size)):
            start_index = iter_eval * batch_size
            end_index = min((iter_eval + 1) * batch_size, num_samples)
            if self.config["cond"] in ["label"]:
                cond = condition[start_index:end_index]
            elif self.config["cond"] is None:
                cond = None
            else:
                raise NotImplementedError

            noise = torch.randn((end_index-start_index, self.ds_shape[1], self.ds_shape[2])).to(self.device)
            input = noise
            # noise_preds = []
            for t in self.noise_scheduler.timesteps:
                #print(t)
                with torch.no_grad():
                    noisy_residual = self.model(input, t, label=cond, return_dict=True)["sample"]  # .sample
                    if self.config.get("w", 0) != 0:
                        noisy_residual_unconditioned = self.model(input, t, label=None, return_dict=True)["sample"]
                        noisy_residual = (self.config["w"] + 1) * noisy_residual - self.config["w"] * noisy_residual_unconditioned

                previous_noisy_sample = self.noise_scheduler.step(noisy_residual, t, input).prev_sample  # prev_sample
                input = previous_noisy_sample
            gen_data.append(input)
        gen_data = torch.cat(gen_data, dim=0)

        return gen_data

    def diffuse_augment(self, samples, cond):
        noise = torch.randn_like(samples)

        timesteps = torch.randint(len(self.noise_scheduler.timesteps)//2, len(self.noise_scheduler.timesteps), (1,)).long()

        actual_step = self.noise_scheduler.timesteps[timesteps]
        actual_step = actual_step.expand([samples.shape[0]])

        input = self.noise_scheduler.add_noise(samples, noise, actual_step)
        # Sample a random timestep for each image

        for t in self.noise_scheduler.timesteps[timesteps:]:
            with torch.no_grad():
                noisy_residual = self.model(input, t, label=cond, return_dict=True)["sample"]  # .sample
                if self.config.get("w", 0) != 0:
                    noisy_residual_unconditioned = self.model(input, t, label=None, return_dict=True)["sample"]
                    noisy_residual = (self.config["w"] + 1) * noisy_residual - self.config["w"] * noisy_residual_unconditioned

            previous_noisy_sample = self.noise_scheduler.step(noisy_residual, t, input).prev_sample  # prev_sample
            input = previous_noisy_sample
        return input

    def train(self, x_train, y_train, iterations, batch_size, batch_size_eval=200):
        self.ds_shape = x_train.shape
        x_train = x_train.to(self.device)
        y_train = y_train.to(self.device)

        if self.config['cond'] == 'label':
            condition = y_train
        elif self.config['cond'] is None:
            condition = None
        else:
            raise NotImplementedError

        if self.total_steps < iterations:
            progress_bar = tqdm(total=iterations)
            progress_bar.set_description("Training")
        while self.total_steps < iterations:
            self.model.train()
            idx = torch.randperm(len(x_train))
            x_train_shuffled = x_train[idx]
            if condition is not None:
                #print(condition[:5])
                condition_shuffled = condition[idx]
            for i in range(len(x_train_shuffled) // batch_size):
                x_train_step = x_train_shuffled[i * batch_size:(i + 1) * batch_size]
                nan_mask = ~x_train_step.isnan().any(axis=1).unsqueeze(1)
                nan_mask = torch.broadcast_to(nan_mask, x_train_step.shape)
                x_train_step[~nan_mask] = 0
                if condition is not None:
                    cond_step = condition_shuffled[i * batch_size:(i + 1) * batch_size]
                else:
                    cond_step = None

                clean_samples = x_train_step
                # Sample noise to add to the images
                noise = torch.randn_like(clean_samples)

                bs = clean_samples.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0, self.noise_scheduler.config.num_train_timesteps, (bs,), device=self.device
                ).long()

                # Add noise to the clean images according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_samples = self.noise_scheduler.add_noise(clean_samples, noise, timesteps)

                output = self.model(noisy_samples, timesteps, label=cond_step, drop_labels=self.config.get("drop_labels", 0), return_dict=True)
                noise_pred = output["sample"]


                if self.loss == 'mse':
                    loss = F.mse_loss(noise_pred, noise)

                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.opt.step()
                self.lr_scheduler.step()
                self.opt.zero_grad()

                progress_bar.update(1)
                logs = {"loss": loss.detach().item(), "lr": self.lr_scheduler.get_last_lr()[0], "step": self.total_steps}
                progress_bar.set_postfix(**logs)

                self.total_steps += 1

                if self.total_steps >= iterations:
                    break

        with torch.no_grad():
            self.create_checkpoint()
            self.model.eval()
            self.model.eval()
            gen_data = self.generate_synthetic_data(condition, x_train.shape[0], batch_size_eval)

        return gen_data


    def create_checkpoint(self, dict=None):
        if dict is None:
            dict = self.create_checkpoint_dict()
        checkpoint_dir_ = 'exp/_models/' + self.config['exp_id'] + '_'
        torch.save(dict, checkpoint_dir_ + 'trainer.pth')

        self.last_checkpoint_dir = checkpoint_dir_

    def create_checkpoint_dict(self):
        return {
            'model': self.model.state_dict().copy(),
            'total_steps': self.total_steps,
            'config': self.config
        }

    def load_checkpoint(self, checkpoint_dir):
        state_dict = torch.load(checkpoint_dir+'trainer.pth')
        if self.config["cond"] is None:
            if state_dict['model']["cond_embedding.weight"].shape[0] != self.model.state_dict()['cond_embedding.weight'].shape[0]:
                state_dict['model']["cond_embedding.weight"] = self.model.state_dict()['cond_embedding.weight']
        self.model.load_state_dict(state_dict['model'], strict=False)
        self.total_steps = state_dict['total_steps']
        self.last_checkpoint_dir = checkpoint_dir
