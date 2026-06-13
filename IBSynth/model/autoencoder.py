from typing import Sequence
import torch
from torch import nn, Tensor
from einops import rearrange
from torch import cos, sin, tensor, pi


class NeRFEmbedding(nn.Module):

    def __init__(self, L: int) -> None:
        super().__init__()
        self.L = L
        self.register_buffer(
            "_emb_vec",
            tensor([pi * 2**i for i in range(L)], requires_grad=False),
        )

    def forward(self, x: Tensor) -> Tensor:
        x_emb = x.unsqueeze(-1) * self._emb_vec

        sin_emb = sin(x_emb)
        cos_emb = cos(x_emb)

        embedded = torch.cat([sin_emb, cos_emb], dim=-1)
        return rearrange(embedded, "b d l -> b (d l)")


class UnitaryAutoencoder(nn.Module):

    def __init__(
        self,
        num_qubits: int,
        latent_dim: int,
        nerf_dim: int,
        hidden_dims: Sequence[int],
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.num_qubits = num_qubits
        self.matrix_dim = 2**self.num_qubits

        original_input_dim = 2 * (self.matrix_dim**2)

        if nerf_dim > 0:
            self.nerf = NeRFEmbedding(nerf_dim)
            encoder_input_dim = original_input_dim * 2 * nerf_dim
        else:
            self.nerf = None
            encoder_input_dim = original_input_dim

        encoder_layers = []
        in_dim = encoder_input_dim
        for h_dim in hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, h_dim))
            encoder_layers.append(nn.GELU())
            encoder_layers.append(nn.Dropout(p=dropout))
            in_dim = h_dim
        encoder_layers.append(nn.Linear(in_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        in_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(in_dim, h_dim))
            decoder_layers.append(nn.GELU())
            decoder_layers.append(nn.Dropout(p=dropout))
            in_dim = h_dim
        decoder_layers.append(nn.Linear(in_dim, original_input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: Tensor) -> Tensor:
        x_flat = rearrange(x, "b c h w -> b (c h w)")

        if self.nerf:
            x_embedded = self.nerf(x_flat)
        else:
            x_embedded = x_flat

        latent_vector = self.encoder(x_embedded)

        reconstructed_flat = self.decoder(latent_vector)

        reconstructed_matrix = rearrange(
            reconstructed_flat,
            "b (c h w) -> b c h w",
            c=2,
            h=self.matrix_dim,
            w=self.matrix_dim,
        )

        return reconstructed_matrix
