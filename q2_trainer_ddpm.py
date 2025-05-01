import torch
from matplotlib import pyplot as plt 
from tqdm import tqdm
from torch.amp import GradScaler, autocast
import copy
import os
import numpy as np

from ddpm_utils.args import * 

torch.manual_seed(42)

def one_param(m):
    "get model first parameter"
    return next(iter(m.parameters()))

class EMA:
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
        current_lr = round(self.optimizer.param_groups[0]['lr'], 5)
        i = 0
        running_loss = 0.
        with tqdm(range(len(dataloader)), desc=f'Epoch : - lr: - Loss :') as progress:
            for x0 in dataloader:
                i += 1
                # Move data to device
                x0 = x0.to(self.args.device)
                # Calculate the loss
                with autocast(device_type=args.device, enabled=self.args.fp16_precision):
                    loss = self.diffusion.loss(x0)
                
                # Zero gradients
                self.optimizer.zero_grad()
                # Backward pass
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
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
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)
            start_epoch = self.current_epoch
            self.loss_per_iter = []
            for current_epoch in range(start_epoch, self.args.epochs):
                self.current_epoch = current_epoch
                self.train_epoch(dataloader, scaler)

                if current_epoch % self.args.show_every_n_epochs == 0:
                    self.sample()

                if (current_epoch + 1) % self.args.save_every_n_epochs == 0:
                    self.save_model()


    def sample(self, n_steps=None, set_seed=False):
        if set_seed:
            torch.manual_seed(42)
        if n_steps is None:
            n_steps = self.args.n_steps

        with torch.no_grad():
            # $x_T \sim p(x_T) = \mathcal{N}(x_T; \mathbf{0}, \mathbf{I})$
            x = torch.randn(
                [
                    self.args.n_samples,
                    self.args.image_channels,
                    self.args.image_size,
                    self.args.image_size,
                ],
                device=self.args.device,
            )

            if self.args.nb_save is not None:
                saving_steps = [n_steps - 1]

            # Remove noise for $T$ steps
            for step in tqdm(range(n_steps)):
                t_val = n_steps - 1 - step
                t = torch.full((x.size(0),), t_val, device=self.args.device, dtype=torch.long)

                x = self.diffusion.p_sample(x, t)

                if self.args.nb_save is not None and t_val in saving_steps:
                    print(f"Showing/saving samples from epoch {self.current_epoch}")
                    self.show_save(
                        x,
                        show=True,
                        save=True,
                        file_name=f"DDPM_epoch_{self.current_epoch}_sample_{step}.png",
                    )

        return x


    def save_model(self):
        torch.save({
                'epoch': self.current_epoch,
                'model_state_dict': self.eps_model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                }, self.args.MODEL_PATH)

    def show_save(self, img_tensor, show=True, save=True, file_name="sample.png"):
        fig, axs = plt.subplots(3, 3, figsize=(10, 10))  # Create a 4x4 grid of subplots
        assert img_tensor.shape[0] >= 9, "Number of images should be at least 9"
        img_tensor = img_tensor[:9]
        for i, ax in enumerate(axs.flat):
            # Remove the channel dimension and convert to numpy
            img = img_tensor[i].squeeze().cpu().numpy()

            ax.imshow(img, cmap="gray")  # Display the image in grayscale
            ax.axis("off")  # Hide the axis

        plt.tight_layout()
        if save:
            os.makedirs("images", exist_ok=True)
            plt.savefig(os.path.join("images", file_name))
        if show:
            plt.show()
        plt.close(fig)
        
        
    def generate_intermediate_samples(self, n_samples=4, img_size=32, steps_to_show=[0,999], n_steps=None, set_seed=False):
        """
        Generate multiple images and return intermediate steps of the diffusion process
        Args:
            n_samples: Number of images to generate
            img_size: Size of the images (assumes square images)
            every_n_steps: Capture intermediate result every n steps
        Returns:
            List of tensors representing the images at different steps
        """
        
        if set_seed:
            torch.manual_seed(42)
        
        if n_steps is None:
            n_steps = self.args.n_steps
            
        # Start from random noise
        x = torch.randn(n_samples, 1, img_size, img_size, device=self.args.device, requires_grad=False)

        # Store images at each step we want to show
        images = []
        if 0 in steps_to_show:
            images.append(x.detach().cpu())  # Initial noise

        for step in tqdm(reversed(range(1, n_steps+1, 1))):  # from T-1 to 0
            t = torch.full((n_samples,), step, device=self.args.device, dtype=torch.long)
            with torch.no_grad():
                x = self.diffusion.p_sample(x, t, set_seed=set_seed)
            if step in steps_to_show:
                images.append(x.detach().cpu())

        return torch.stack(images)