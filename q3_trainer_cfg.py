
import torch
import torch.utils.data
import os
# import torchvision
# from torch import nn
# from typing import Tuple, Optional
# import torch.nn.functional as F
# from easydict import EasyDict
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.amp import GradScaler, autocast

import numpy as np 
import copy 

torch.manual_seed(42)

def one_param(m):
    "get model first parameter"
    return next(iter(m.parameters()))

class EMA:
    """Exponential Moving Average class to improve training"""
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.step = 0

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model, step_start_ema=2000):
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())


class Trainer:
    def __init__(self, args, eps_model, diffusion_model):

        self.eps_model = eps_model.to(args.device)

        self.diffusion = diffusion_model

        self.optimizer = torch.optim.Adam(
            self.eps_model.parameters(), lr=args.learning_rate
        )
        self.args = args
        self.current_epoch = 0

        self.ema = EMA(0.995)
        self.ema_model = copy.deepcopy(self.eps_model).eval().requires_grad_(False)


    def train_epoch(self, dataloader, scaler):
        current_lr = round(self.optimizer.param_groups[0]['lr'], 8)
        i = 0
        running_loss = 0.
        with tqdm(range(len(dataloader)), desc=f'Epoch : - lr: - Loss :') as progress:
            for x0, labels in dataloader:
                i += 1
                # Move data to device
                x0 = x0.to(self.args.device)
                # Use guidance
                labels = labels.to(self.args.device)
                if np.random.random() < 0.1:
                    labels = None

                # Calculate the loss
                with autocast(device_type=self.args.device, enabled=self.args.fp16_precision):
                    loss = self.diffusion.loss(x0, labels)
                    
                # Zero gradients
                self.optimizer.zero_grad()
                # Backward pass
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
                # loss.backward()
                # self.optimizer.step()
                self.ema.step_ema(self.ema_model, self.eps_model)

                running_loss += loss.item()

                self.loss_per_iter.append(running_loss / i)
                progress.update()
                progress.set_description(f'Epoch: {self.current_epoch}/{self.args.epochs} - lr: {current_lr} - Loss: {round(running_loss / i, 2)}')
            progress.set_description(f'Epoch: {self.current_epoch}/{self.args.epochs} - lr: {current_lr} - Loss: {round(running_loss / len(dataloader), 2)}')

            # Step the scheduler after each epoch
            self.scheduler.step()

    def train(self, dataloader):
            scaler = GradScaler(device=self.args.device, enabled=self.args.fp16_precision)
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.5)
            start_epoch = self.current_epoch
            self.loss_per_iter = []
            for current_epoch in range(start_epoch, self.args.epochs):
                self.current_epoch = current_epoch
                self.train_epoch(dataloader, scaler)
                if current_epoch % self.args.show_every_n_epochs == 0:
                    self.sample(cfg_scale=self.args.cfg_scale)

                if (current_epoch + 1) % self.args.save_every_n_epochs == 0:
                    self.save_model()

    def sample(self, labels=None, cfg_scale=3., n_steps=None, set_seed=False):
        if set_seed:
            torch.manual_seed(42)
        if n_steps is None:
            n_steps = self.args.n_steps

        self.eps_model.eval()
        with torch.no_grad():
            z_t = torch.randn(
                    [
                        self.args.n_samples,
                        self.args.image_channels,
                        self.args.image_size,
                        self.args.image_size,
                    ],
                    device=self.args.device,
                )

            if labels == None:
                labels = torch.randint(0, 9, (self.args.n_samples,), device=self.args.device)

            if self.args.nb_save is not None:
                saving_steps = [n_steps - 1]

            # Remove noise for $T$ steps
            for t_ in tqdm(range(n_steps)):
                t_val = n_steps - 1 - t_
                t = torch.full((self.args.n_samples,), t_val, device=z_t.device, dtype=torch.long)
                # t_prim = torch.full((self.args.n_samples,), t_val+1, device=z_t.device, dtype=torch.long)

                lambda_t = self.diffusion.get_lambda(t)
                lambda_t_prim = self.diffusion.get_lambda(t - 1)

                eps_cond = self.eps_model(z_t, labels)
                eps_uncond = self.eps_model(z_t, None)
                eps_theta = (1 + cfg_scale) * eps_cond - cfg_scale * eps_uncond

                alpha_lambda = self.diffusion.alpha_lambda(lambda_t)
                sigma_lambda = self.diffusion.sigma_lambda(lambda_t)
                x_t = (z_t - sigma_lambda * eps_theta) / alpha_lambda

                z_t = self.diffusion.p_sample(z_t, lambda_t, lambda_t_prim, x_t)

                if self.args.nb_save is not None and t_ in saving_steps:
                    print(f"Showing/saving samples from epoch {self.current_epoch} with labels: {labels.tolist()}")
                    self.show_save(
                        img_tensor=x_t,
                        labels=labels,
                        show=True,
                        save=True,
                        file_name=f"CFG_epoch_{self.current_epoch}_sample_{t_}.png"
                    )

            self.eps_model.train()
        return x_t

    def save_model(self):
        torch.save({
                'epoch': self.current_epoch,
                'model_state_dict': self.eps_model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                }, self.args.MODEL_PATH)
    
    def show_save(self, img_tensor, labels=None, show=True, save=True, file_name="sample.png"):
        fig, axs = plt.subplots(3, 3, figsize=(10, 10))  # Create a 4x4 grid of subplots
        assert img_tensor.shape[0] >= 9, "Number of images should be at least 9"
        img_tensor = img_tensor[:9]
        for i, ax in enumerate(axs.flat):
            # Remove the channel dimension and convert to numpy
            img = img_tensor[i].squeeze().cpu().numpy()
            label = labels[i].item()
            ax.imshow(img, cmap="gray")  # Display the image in grayscale
            ax.set_title(f'Digit:{label}')
            ax.axis("off")  # Hide the axis

        plt.tight_layout()
        if save:
            os.makedirs("images", exist_ok=True)
            plt.savefig(os.path.join("images", file_name))
        if show:
            plt.show()
        plt.close(fig)