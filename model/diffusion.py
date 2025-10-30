# Diffusion Model
import abc

import torch
import torch.nn.functional as F
import torch.nn as nn


class AbstractDiffusion(nn.Module, metaclass=abc.ABCMeta):
    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def sample(self, x0):
        ## Given x0, sample points using DDPM forward
        pass


def extract(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    out = torch.gather(v, index=t, dim=0).float()
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))


class Diffusion(nn.Module):
    def __init__(self, beta_1, beta_T, T=1000, t_end=None):
        super().__init__()
        # Here T means the total diffusion steps
        self.T = T
        self.t_end = t_end

        self.register_buffer("betas", torch.linspace(beta_1, beta_T, T).float())
        alphas = 1.0 - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer("sqrt_alphas_bar", torch.sqrt(alphas_bar))  # [T]
        self.register_buffer("sqrt_one_minus_alphas_bar", torch.sqrt(1.0 - alphas_bar))

    def sample(self, x_0):
        """
        Run DDPM forward and get all the intermediate points
        Input:
            x_0: [B, D]
        Returns:
            [B, t_end+1, D] which denotes the sequence of forward diffusions (it also includes x_0)
        """
        if self.t_end is None:
            t = self.T
        else:
            t = self.t_end
        bb, dd = x_0.shape
        eps = torch.randn(bb, t, dd, dtype=x_0.dtype, device=x_0.device)  # [B, t, D]
        x_0 = x_0.unsqueeze(1)  # [B, 1, D]
        # [t] → [1, t, 1]
        sqrt_alpha_bar = self.sqrt_alphas_bar[:t].view(1, t, 1)
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alphas_bar[:t].view(1, t, 1)
        res = sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * eps  # [B, t, D]
        res = torch.cat([x_0, res], dim=1)
        print(res.shape)
        return res


if __name__ == "__main__":
    model = Diffusion(beta_1=0.02, beta_T=1.0e-4, t_end=3)
    x_0 = torch.ones(3, 2, dtype=torch.float)
    out = model.sample(x_0)
    print(out.shape)
    print(out)
    pass
