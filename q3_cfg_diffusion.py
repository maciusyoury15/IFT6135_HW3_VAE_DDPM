# %%
# import torch
# import torchvision
import torch.utils.data
from torch import nn

from cfg_utils.args import *


class CFGDiffusion():
    def __init__(self, eps_model: nn.Module, n_steps: int, device: torch.device):
        super().__init__()
        self.eps_model = eps_model
        self.n_steps = n_steps
        self.device = device

        self.lambda_min = -20
        self.lambda_max = 20

    ### UTILS
    def get_exp_ratio(self, l: torch.Tensor, l_prim: torch.Tensor):
        return torch.exp(l - l_prim)

    def get_lambda(self, t: torch.Tensor):
        u = t / self.n_steps

        lambda_min = torch.tensor(self.lambda_min, device=t.device)
        lambda_max = torch.tensor(self.lambda_max, device=t.device)

        b = torch.atan(torch.exp(-lambda_max / 2))
        a = torch.atan(torch.exp(-lambda_min / 2)) - b

        lambda_t = -2 * torch.log(torch.tan(a * u + b))
        return lambda_t.view(-1, 1, 1, 1)

    def alpha_lambda(self, lambda_t: torch.Tensor):
        # Use get_exp_ratio(0, lambda_t) instead of torch.exp(-lambda_t)
        zero_tensor = torch.zeros_like(lambda_t)
        alpha_squared = 1 / (1 + self.get_exp_ratio(zero_tensor, lambda_t))
        return torch.sqrt(alpha_squared)

    def sigma_lambda(self, lambda_t: torch.Tensor):
        alpha = self.alpha_lambda(lambda_t)
        sigma_squared = 1 - alpha ** 2
        return torch.sqrt(sigma_squared)

    ## Forward sampling
    def q_sample(self, x: torch.Tensor, lambda_t: torch.Tensor, noise: torch.Tensor):
        alpha = self.alpha_lambda(lambda_t)
        sigma = self.sigma_lambda(lambda_t)
        return alpha * x + sigma * noise

    def sigma_q(self, lambda_t: torch.Tensor, lambda_t_prim: torch.Tensor):
        sigma_lambda_sq = self.sigma_lambda(lambda_t) ** 2
        exp_ratio = self.get_exp_ratio(lambda_t, lambda_t_prim)
        var = (1 - exp_ratio) * sigma_lambda_sq
        return torch.sqrt(torch.clamp(var, min=1e-10))

    def sigma_q_x(self, lambda_t: torch.Tensor, lambda_t_prim: torch.Tensor):
        sigma_lambda_sq = self.sigma_lambda(lambda_t_prim) ** 2
        ratio = 1 - torch.exp(lambda_t - lambda_t_prim)
        var = ratio * sigma_lambda_sq
        return torch.sqrt(torch.clamp(var, min=1e-10))

    ### REVERSE SAMPLING
    def mu_p_theta(self, z_lambda_t: torch.Tensor, x: torch.Tensor, lambda_t: torch.Tensor,
                   lambda_t_prim: torch.Tensor):
        exp_ratio = self.get_exp_ratio(lambda_t, lambda_t_prim)
        alpha_t = self.alpha_lambda(lambda_t)
        alpha_t_prim = self.alpha_lambda(lambda_t_prim)

        term1 = exp_ratio * (alpha_t_prim / alpha_t) * z_lambda_t
        term2 = (1 - exp_ratio) * alpha_t_prim * x
        return term1 + term2

    def var_p_theta(self, lambda_t: torch.Tensor, lambda_t_prim: torch.Tensor, v: float = 0.3):
        sigma_lambda_sq = self.sigma_lambda(lambda_t) ** 2
        sigma_lambda_prim_sq = self.sigma_lambda(lambda_t_prim) ** 2
        sigma_ratio = 1 - torch.exp(lambda_t - lambda_t_prim)

        base_var = sigma_ratio * sigma_lambda_sq
        base_var_prim = sigma_ratio * sigma_lambda_prim_sq

        # According to Eq (4): interpolate variance
        return torch.clamp((base_var_prim ** (1 - v)) * (base_var ** v), min=1e-10)

    def p_sample(self, z_lambda_t: torch.Tensor, lambda_t: torch.Tensor, lambda_t_prim: torch.Tensor, x_t: torch.Tensor,
                 set_seed=False):
        if set_seed:
            torch.manual_seed(42)

        mu = self.mu_p_theta(z_lambda_t, x_t, lambda_t, lambda_t_prim)
        var = self.var_p_theta(lambda_t, lambda_t_prim)

        noise = torch.randn_like(z_lambda_t)
        return mu + torch.sqrt(var) * noise

    ### LOSS
    def loss(self, x0: torch.Tensor, labels: torch.Tensor, noise=None, set_seed=False):
        if set_seed:
            torch.manual_seed(42)

        batch_size = x0.shape[0]
        dim = tuple(range(1, x0.ndim))

        # Step 1: Sample t ~ Uniform({0, ..., T-1})
        t = torch.randint(0, self.n_steps, (batch_size,), device=x0.device, dtype=torch.long)

        # Step 2: Get λ(t)
        lambda_t = self.get_lambda(t)

        # Step 3: Sample noise ε ~ N(0, I)
        if noise is None:
            noise = torch.randn_like(x0)

        # Step 4: Compute z_λ using the existing q_sample function
        z_lambda = self.q_sample(x0, lambda_t, noise)

        # Step 5: Predict noise using ε_θ(z_λ, labels)
        eps_pred = self.eps_model(z_lambda, labels)

        loss = (eps_pred - noise) ** 2
        return loss.sum(dim=dim).mean()  # sum over pixels, mean over batch