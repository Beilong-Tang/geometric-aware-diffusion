import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def cal_diff_vector(x: torch.Tensor) -> torch.Tensor:
    # x: [B, T, D]
    # Return: d [B, T, D]
    # For each i, d[:,i] = x_{i+1} - x_{i-1}
    # Special case when i = 0: x1 - x0. when i = T, xT - xT-1

    diff = x[:, 2:] - x[:, :-2]  # [x2-x0, x3-x1,...,xT-xT-2] # [B, T-2, D]
    first = x[:, 1:2] - x[:, 0:1]  # [B, 1, D]
    last = x[:, -1:] - x[:, -2:-1]  # [B, 1, D]
    diff = torch.cat([first, diff, last], dim=1)  # [B, T, D]
    return diff


class GeometricEmbedding(nn.Module):
    def __init__(self, start_from_T=True, emb_dim=128, arc_emb_dim=32):
        # in_features: the input feature dim of the latent image vector
        # emb_dim: the input feature output for Wx * x_t
        # arc_emb_dim: The embedding for the arc length feature
        super().__init__()
        self.start_from_T = start_from_T
        self.emb_dim = emb_dim
        self.arc_emb_dim = arc_emb_dim
        self.arc = ArcLengthEmbed(start_from_T, arc_emb_dim)
        self.curv = CurvatureEmbedding(start_from_T, False)

    def setup(self, in_features):
        self.encoder = MLP(in_features, self.emb_dim)
        self.in_features = in_features
        pass

    def get_token_emb_dim(self):
        if self.curv.use_curvature is False:
            return self.emb_dim + self.arc_emb_dim + self.in_features
        else:
            return self.emb_dim + self.arc_emb_dim + self.in_features * 2

    def forward(self, x):
        # x [B, T, D], which is a sequence from x_0 to x_T
        # assert x.size(1) == self.T
        # Result shape: [B, T, emb_dim + arc_emb_dim + 2*in_feature (or feature)]
        latent = self.encoder(x)  # [B, T, emb_dim]
        arc_emb = self.arc(x)  # [B, T, arc_emb_dim]
        curvature = self.curv(x)  # [B, T, 2*D] or [B, T, D]
        return torch.cat([latent, arc_emb, curvature], dim=-1)


class MLP(nn.Module):  # w_x
    def __init__(self, in_features, emb_dim):
        # This serves as W_t
        super().__init__()
        self.model = nn.Linear(in_features, emb_dim)

    def forward(self, x):
        return self.model(x)


class ArcLengthEmbed(nn.Module):  # phi_s
    def __init__(self, start_from_T=True, emb_dim=32):
        # If the arc length sum from the reversed direction
        super().__init__()
        self.start_from_T = start_from_T
        self.emb_dim = emb_dim
        self.positional = PositionalEmbedding(emb_dim)

    def _get_norm_sum(self, x):
        # Get the cumulative arc length
        ## x: [B, T, N]
        ## Return: [B, T]
        diff = x[:, :-1] - x[:, 1:]  # [B, (T-1), N]
        norm = torch.norm(diff, p=2, dim=-1)  # [B, (T-1)]
        if self.start_from_T:
            norm = torch.flip(norm, dims=(1,))
        norm_sum = torch.cumsum(norm, dim=1)
        norm_sum = F.pad(norm_sum, (1, 0), "constant", 0)
        if self.start_from_T:
            norm_sum = torch.flip(norm_sum, dims=(1,))  # [B, T]
        return norm_sum

    def forward(self, x):
        # [B,T,N]
        # Return [B,T,D]
        norm_sum = self._get_norm_sum(x)  # [B, T]
        emb = self.positional(norm_sum)
        return emb


class CurvatureEmbedding(nn.Module):  # phi_g

    def __init__(self, start_from_T=True, use_curvature=False):
        super().__init__()
        self.start_from_T = start_from_T
        self.use_curvature = use_curvature

    def _get_diff_vector(self, x):
        # x: [B, T, D]
        # Return: T [B,T,D] The diff vector at each position
        diff = cal_diff_vector(x)  # [B, T, D]
        if self.start_from_T:
            return -1 * diff
        else:
            return diff

    def _get_tangent(self, x, eps=1.0e-6):
        tangent = self._get_diff_vector(x)  # [B, T, D]
        norm = torch.norm(tangent, p=2, dim=-1, keepdim=True)
        non_zero_mask = 1 - (norm < eps).float()
        tangent = F.normalize(tangent, dim=-1, eps=eps)
        tangent = tangent * non_zero_mask
        return tangent

    def _get_curvature(self, tangent, eps=1.0e-6):
        delta_s = 1.0 / (tangent.size(1) - 1)
        curvature = self._get_diff_vector(tangent)  # [B, T, D]
        curvature[:, 0] = curvature[:, 0] / delta_s
        curvature[:, 1:-1] = curvature[:, 1:-1] / (2 * delta_s)
        curvature[:, -1] = curvature[:, -1] / delta_s
        norm = torch.norm(curvature, p=2, dim=-1, keepdim=True)
        non_zero_mask = 1 - (norm < eps).float()
        curvature = F.normalize(curvature, dim=-1, eps=eps)
        curvature = curvature * non_zero_mask
        return curvature

    def forward(self, x):
        # x: [B, T, D]
        # Returns: [B, T, 2D] phi_g(x_i) which contatenates Tangent and Curvature
        # If use_curvarture is False, return only tangent [B, T, D]
        tangent = self._get_tangent(x)  # [B, T, D]
        if self.use_curvature:
            curvature = self._get_curvature(tangent)  # [B, T, D]
            emb = torch.cat([tangent, curvature], dim=-1)
        else:
            emb = tangent
        return emb


class PositionalEmbedding(nn.Module):

    def __init__(self, d_model=32, add_sinusoidal=False):
        super().__init__()
        ## add_positonal: whether to add the sinusoidal to the embedding
        T = 1001
        emb = torch.arange(0, d_model, step=2) / d_model * math.log(10000)
        emb = torch.exp(-emb)
        pos = torch.arange(T).float()
        emb = pos[:, None] * emb[None, :]
        assert list(emb.shape) == [T, d_model // 2]
        emb = torch.stack([torch.sin(emb), torch.cos(emb)], dim=-1)
        assert list(emb.shape) == [T, d_model // 2, 2]
        emb = emb.view(T, d_model)  # [T, D]

        self.add_sinusoidal = add_sinusoidal
        self.emb = emb
        self.d_model = d_model
        self.mlp = nn.Linear(1, d_model // 2)

    def forward(self, x):
        # x: [B, T]
        # return [B, T, D]
        bb, tt = x.shape
        x = x.unsqueeze(-1)  # [B, T, 1]
        x = self.mlp(x)  # [B, T, d_model//2]
        x = torch.stack([torch.sin(x), torch.cos(x)], dim=-1)  # [B, T, d_model//2, 2]
        x = x.view(bb, tt, self.d_model)
        if self.add_sinusoidal:
            x = x + self.emb.unsqueeze(0)[:, : x.size(1)]
        return x


if __name__ == "__main__":
    ## Test 1
    # x = torch.tensor([1.0, 2, 4])  # [B]
    # x = x.unsqueeze(-1)
    # x = x.repeat(1, 2)
    # x = x.unsqueeze(0).repeat(3, 1, 1)
    # print("x shape: ", x.shape)
    # print("x: ", x)
    # model = ArcLengthEmbed(True)
    # out = model._get_norm_sum(x)
    # print("out: ", out.shape)
    # emb = model(x)
    # print("emb: ", emb.shape)

    ## Test 2
    # x = torch.tensor([[[1.0, 4], [5, 2], [7, 8]]])
    # model = CurvatureEmbedding(start_from_T=True, use_curvature=True)
    # out = model(x)
    # print(out)
    # print(out.shape)

    ## Test 3
    x = torch.randn(4, 101, 256)
    model = GeometricEmbedding()
    model.setup(256)
    out = model(x)
    print(out.shape)
