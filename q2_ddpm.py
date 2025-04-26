
import torch
from torch import nn 
from typing import Optional, Tuple


class DenoiseDiffusion():
    def __init__(self, eps_model: nn.Module, n_steps: int, device: torch.device):
        super().__init__()
        self.eps_model = eps_model
        self.beta = torch.linspace(0.0001, 0.02, n_steps).to(device)
        self.alpha = 1.0 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        self.n_steps = n_steps
        self.sigma2 = self.beta


    ### UTILS
    def gather(self, c: torch.Tensor, t: torch.Tensor):
        c_ = c.gather(-1, t)
        return c_.reshape(-1, 1, 1, 1)

    ### FORWARD SAMPLING
    def q_xt_x0(self, x0: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = x0.shape[0]
        sqrt_alpha_bar = torch.sqrt(self.alpha_bar[t]).view(batch_size, 1, 1, 1)
        one_minus_alpha_bar = 1.0 - self.alpha_bar[t]

        var = one_minus_alpha_bar.view(batch_size, 1, 1, 1)
        mean = sqrt_alpha_bar * x0

        return mean, var

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, eps: Optional[torch.Tensor] = None):
        if eps is None:
            eps = torch.randn_like(x0)

        sqrt_alpha_bar = torch.sqrt(self.alpha_bar[t]).view(-1, 1, 1, 1)
        one_minus_alpha_bar = 1.0 - self.alpha_bar[t]
        sqrt_one_minus_alpha_bar = torch.sqrt(one_minus_alpha_bar).view(-1, 1, 1, 1)

        sample = sqrt_alpha_bar * x0 + sqrt_one_minus_alpha_bar * eps
        return sample

    ### REVERSE SAMPLING
    def p_xt_prev_xt(self, xt: torch.Tensor, t: torch.Tensor):
        batch_size = xt.shape[0]

        alpha_t = self.alpha[t].view(batch_size, 1, 1, 1)
        beta_t = self.beta[t].view(batch_size, 1, 1, 1)
        alpha_bar_t = self.alpha_bar[t].view(batch_size, 1, 1, 1)

        coef1 = 1 / torch.sqrt(alpha_t)
        coef2 = (1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)
        eps_theta = self.eps_model(xt, t)

        mu_theta = coef1 * (xt - coef2 * eps_theta)
        var = beta_t

        return mu_theta, var


    def p_sample(self, xt: torch.Tensor, t: torch.Tensor, set_seed=False):
        if set_seed:
            torch.manual_seed(42)

        # Predict noise using the model
        eps_theta = self.eps_model(xt, t)

        batch_size = xt.shape[0]
        beta_t = self.beta[t].view(batch_size, 1, 1, 1)
        alpha_t = self.alpha[t].view(batch_size, 1, 1, 1)
        alpha_bar_t = self.alpha_bar[t].view(batch_size, 1, 1, 1)

        # Compute mu_theta
        coef1 = 1 / torch.sqrt(alpha_t)
        coef2 = (1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)
        mu_theta = coef1 * (xt - coef2 * eps_theta)

        # Sample noise
        noise = torch.randn_like(xt)

        # If t == 0, just return the mean (no noise at step 0)
        is_zero = (t == 0).float().view(-1, 1, 1, 1)
        sample = mu_theta + (1 - is_zero) * torch.sqrt(beta_t) * noise

        return sample

    ### LOSS
    def loss(self, x0: torch.Tensor, noise: Optional[torch.Tensor] = None, set_seed=False):
        if set_seed:
            torch.manual_seed(42)

        batch_size = x0.shape[0]
        t = torch.randint(0, self.n_steps, (batch_size,), device=x0.device, dtype=torch.long)

        if noise is None:
            noise = torch.randn_like(x0)

        # Get relevant schedule values
        alpha_bar_t = self.alpha_bar[t].view(batch_size, 1, 1, 1)
        sqrt_alpha_bar = torch.sqrt(alpha_bar_t)
        sqrt_one_minus_alpha_bar = torch.sqrt(1 - alpha_bar_t)

        # Sample x_t from q(x_t | x_0)
        xt = sqrt_alpha_bar * x0 + sqrt_one_minus_alpha_bar * noise

        # Predict noise using model
        pred_noise = self.eps_model(xt, t)

        # MSE between predicted and actual noise
        loss = torch.mean((pred_noise - noise) ** 2)

        return loss