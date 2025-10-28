# Decoder-only style transformer
import torch.nn as nn
import torch

import torch.nn.functional as F


def generate_square_subsequent_mask(sz: int) -> torch.Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on the diag."""
    return torch.triu(torch.full((sz, sz), float("-inf")), diagonal=1)


class DecoderOnlyTransformer(nn.Module):
    def __init__(
        self,
        nhead: int,
        num_encoder_layers: int,
        dim_feedforward: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout

    def setup(self, d_model):
        self.d_model = d_model
        # Token and positional embeddings

        # The core of the model: A standard TransformerEncoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, self.nhead, self.dim_feedforward, self.dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, self.num_encoder_layers
        )
        if self.final_linear:
            self.linear = nn.Linear(d_model, d_model)
        pass

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """
        Args:
            src: Tensor, shape [batch_size, seq_len, D]

        Returns:
            output Tensor of shape [batch_size, seq_len, vocab_size]
        """
        seq_len = src.size(1)
        # 1. Generate the causal mask
        causal_mask = generate_square_subsequent_mask(seq_len).to(src.device)

        # 2. Pass data and masks to the TransformerEncoder
        output = self.transformer_encoder(
            src,
            mask=causal_mask,
            is_causal=False,  # Important: We provide the mask manually
        )  # [B, T, D]
        return output


class GeometricDecoderOnly(nn.Module):
    def __init__(self, decoder_only: DecoderOnlyTransformer):
        super().__init__()
        self.decoder_only = decoder_only

    def forward(self, x):
        """
        Input:
            x shape: [B, T, D]
        """
        result = {}
        input = x[:, :-1]  # [B, T-1, D]
        output = self.decoder_only(input)  # [B, T-1, D]
        target = x[:, 1:]
        loss = F.mse_loss(output, target)

        result["output"] = output
        return loss, result

    @torch.no_grad()
    def inference(self, xT, T=1000):
        # xT: [B, D] or [B, 1, D]
        # T: inference time step
        # Return [B,D]
        if len(xT.shape) == 2:
            xT = xT.unsqueeze(1)  # [B, 1, D]
        res = [xT]
        for _ in range(0, T):
            input = torch.cat(res, dim=1)  # [B, T, D]
            output = self.forward(input)  # [B, T, D]
            res.append(output[:, -1:])  # [B, 1, D]
        return res[-1].squeeze(1)  # [B. D]


if __name__ == "__main__":

    x = torch.randn(3, 100, 128)
    model = DecoderOnlyTransformer(128, 128, 8, 8, 512)
    print(f"num parameters {sum(i.numel() for i in model.parameters())}")
    out = model(x)
    print(out.shape)

    x = torch.randn(3, 1, 128)
    # out = model.inference(x, T=100)
    # print(out.shape)
    pass
