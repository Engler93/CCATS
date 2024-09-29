"""
Train Diffusion model and / or generate data.
"""

import torch

from metrics.evaluation import EvaluatorVQVAE
from trainer import Trainer
from diffusers import DDPMScheduler, DDIMScheduler
from unet_1d import UNet1DModel
from diffusers.optimization import get_cosine_schedule_with_warmup


def diffusion(x_train, y_train, parameters, device='cuda'):
    """
    Train Diffusion model and generate samples. Generated samples follow classes from train_loader.
    """

    # Dataset Parameters
    num_classes = int(y_train.max()) + 1

    # Network / Training Parameters
    iterations = parameters['iteration']
    loss = parameters['loss']
    lr = parameters['lr']
    diffusion_steps = parameters.get("diffusion_steps", 1000)
    normalize = parameters.get("normalize", "min_max")
    assert normalize in ["min_max", "mean_std"]

    down_block_types = ["DownBlock1DNoSkip"]
    up_block_types = ["UpBlock1DNoSkip"]
    if x_train.shape[2] >= 1024:
        down_block_types.append("DownBlock1D")
        up_block_types.append("UpBlock1D")
    else:
        down_block_types.append("DownBlock1DNoSkip")
        up_block_types.append("UpBlock1DNoSkip")
    if x_train.shape[2] >= 512:
        down_block_types.append("DownBlock1D")
        up_block_types.append("UpBlock1D")
    else:
        down_block_types.append("DownBlock1DNoSkip")
        up_block_types.append("UpBlock1DNoSkip")
    if x_train.shape[2] >= 256:
        down_block_types.append("DownBlock1D")
        up_block_types.append("UpBlock1D")

    else:
        down_block_types.append("DownBlock1DNoSkip")
        up_block_types.append("UpBlock1DNoSkip")

    up_block_types.reverse()


    model = UNet1DModel(sample_size=x_train.shape[2], in_channels=x_train.shape[1], out_channels=x_train.shape[1], layers_per_block=1,
                        block_out_channels=(32, 64, 128, 256), time_embedding_type='positional', ts_step_embedding_type="positional", encode_embeddings=True, #positional
                        condition_embedding="embedding",
                        condition_dim = num_classes if parameters['cond'] == 'label' else 128,
                        down_block_types=down_block_types,
                        up_block_types=up_block_types
                        ).to(device)
    noise_scheduler = DDPMScheduler(num_train_timesteps=diffusion_steps, clip_sample=True if normalize=="min_max" else False, beta_schedule='squaredcos_cap_v2')

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=50,
        num_training_steps=iterations,
    )

    trainer = Trainer(model, noise_scheduler, optimizer, lr_scheduler, loss, parameters, device=device)

    if parameters['checkpoint_dir'] is not None:
        trainer.load_checkpoint(parameters['checkpoint_dir'])

    # only trains, if number of iterations not reached, otherwise only generates
    gen_data = trainer.train(x_train, y_train, iterations, parameters['batch_size']).detach().cpu().numpy()

    return gen_data
