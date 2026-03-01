import numpy as np
import torch
import torch.nn as nn


class IRNetwork(nn.Module):
    ACTION_DIM = 15

    def __init__(self, n_layers: int, hidden_size: int, embedding_size: int):
        super().__init__()
        self.card_embeddings = nn.Embedding(3 + 1, embedding_size)
        self.location_embeddings = nn.Embedding(5 + 1, embedding_size)
        self.mlp = nn.Sequential(*sum([
            [
                nn.Linear(
                    16 * embedding_size if i == 0 else hidden_size,
                    self.ACTION_DIM if i == n_layers else hidden_size,
                    bias=(i == n_layers)
                ),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
            ]
            for i in range(n_layers + 1)
        ], [])[:-2], nn.Tanh()) # Drop the last ReLU
    
    def forward(self, inputs: torch.LongTensor):
        """
        Args:
            inputs: LongTensor of shape (Batch, Cards (6) + Locations (10))
        """
        card_inputs, location_inputs = torch.split(inputs, (6, 10), -1)
        card_latent = self.card_embeddings(card_inputs)
        location_latent = self.location_embeddings(location_inputs)
        latents = torch.cat((card_latent, location_latent), -2)
        return self.mlp(latents.flatten(-2)) * 23


# class IRNetwork(nn.Module):
#     ACTION_DIM = 15

#     def __init__(self, n_layers: int, hidden_size: int, embedding_size: int):
#         super().__init__()
#         self.card_embeddings = nn.Embedding(3 + 1, embedding_size)
#         self.location_embeddings = nn.Embedding(5 + 1, embedding_size)
#         self.first_linear = nn.Linear(16 * embedding_size, hidden_size)
#         self.last_linear = nn.Linear(hidden_size, self.ACTION_DIM)
#         self.residual_mlp = nn.ModuleList([
#             nn.Linear(hidden_size, hidden_size)
#             for _ in range(n_layers - 1)
#         ])
    
#     def forward(self, inputs: torch.LongTensor):
#         """
#         Args:
#             inputs: LongTensor of shape (Batch, Cards (6) + Locations (10))
#         """
#         card_inputs, location_inputs = torch.split(inputs, (6, 10), -1)
#         card_latent = self.card_embeddings(card_inputs)
#         location_latent = self.location_embeddings(location_inputs)
#         latents = torch.cat((card_latent, location_latent), -2)
#         x = self.first_linear(latents.flatten(-2))
#         for ff in self.residual_mlp:
#             x = nn.functional.relu(x + ff(x))
#         return torch.tanh(self.last_linear(x)) * 23

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def np2torch(x: np.ndarray, cast_double_to_float: bool = True) -> torch.Tensor:
    """
    Utility function that accepts a numpy array and does the following:
        1. Convert to torch tensor
        2. Move it to the GPU (if CUDA is available)
        3. Optionally casts float64 to float32 (torch is picky about types)
    """
    assert isinstance(x, np.ndarray), f"np2torch expected 'np.ndarray' but received '{type(x).__name__}'"
    x = torch.from_numpy(x).to(device)
    if cast_double_to_float and x.dtype is torch.float64:
        x = x.float()
    return x
