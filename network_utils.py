import numpy as np
import torch
import torch.nn as nn


# def build_mlp(input_size, output_size, n_layers, size):
#     """
#     Args:
#         input_size: int, the dimension of inputs to be given to the network
#         output_size: int, the dimension of the output
#         n_layers: int, the number of hidden layers of the network
#         size: int, the size of each hidden layer
#     Returns:
#         An instance of (a subclass of) nn.Module representing the network.

#     TODO:
#     Build a feed-forward network (multi-layer perceptron, or mlp) that maps
#     input_size-dimensional vectors to output_size-dimensional vectors.
#     It should have 'n_layers' layers, each of 'size' units and followed
#     by a ReLU nonlinearity. Additionally, the final layer should be linear (no ReLU).

#     That is, the network architecture should be the following:
#     [LINEAR LAYER]_1 -> [RELU] -> [LINEAR LAYER]_2 -> ... -> [LINEAR LAYER]_n -> [RELU] -> [LINEAR LAYER]

#     "nn.Linear" and "nn.Sequential" may be helpful.
#     """
#     #######################################################
#     #########   YOUR CODE HERE - 7-15 lines.   ############
#     return nn.Sequential(*sum([
#         [
#             nn.Linear(input_size if i == 0 else size, output_size if i == n_layers else size),
#             nn.ReLU(),
#         ]
#         for i in range(n_layers + 1)
#     ], [])[:-1])
#     #######################################################
#     #########          END YOUR CODE.          ############


class IRNetwork(nn.Module):
    def __init__(self, n_layers: int, hidden_size: int, embedding_size: int):
        super().__init__()
        self.card_embeddings = nn.Embedding(3 + 1, embedding_size)
        self.location_embeddings = nn.Embedding(5 + 1, embedding_size)
        self.mlp = nn.Sequential(*sum([
            [
                nn.Linear(16 * embedding_size if i == 0 else hidden_size, 15 if i == n_layers else hidden_size),
                nn.ReLU(),
            ]
            for i in range(n_layers + 1)
        ], [])[:-1]) # Drop the last ReLU
    
    def forward(self, inputs: torch.LongTensor):
        """
        Args:
            inputs: LongTensor of shape (Batch, Cards (6) + Locations (10))
        """
        card_inputs, location_inputs = torch.split(inputs, (6, 10), -1)
        card_latent = self.card_embeddings(card_inputs)
        location_latent = self.location_embeddings(location_inputs)
        latents = torch.cat((card_latent, location_latent), -2)
        return self.mlp(latents.flatten(-2))
    

class IRPolicyValueNetwork(nn.Module):
    """Returns a tuple of action logits and scalar value."""

    def __init__(self, n_layers: int, hidden_size: int, embedding_size: int, max_value: float = 23):
        super().__init__()
        self.max_value = max_value
        self.card_embeddings = nn.Embedding(3 + 1, embedding_size)
        self.location_embeddings = nn.Embedding(5 + 1, embedding_size)
        self.mlp = nn.Sequential(*sum([
            [
                nn.Linear(16 * embedding_size if i == 0 else hidden_size, hidden_size),
                nn.ReLU(),
            ]
            for i in range(n_layers)
        ], []))
        self.policy_head = nn.Linear(hidden_size, 3 * 5)
        self.value_head = nn.Sequential(*[
            nn.Linear(hidden_size, 1),
            nn.Tanh(),
        ])
        nn.init.zeros_(self.value_head[0].weight)
        nn.init.zeros_(self.value_head[0].bias)
    
    def forward(self, inputs: torch.LongTensor, return_value: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            inputs: LongTensor of shape (Batch, Cards (6) + Locations (10))
        Returns:
            logits: Logits of policy probabilities (Batch, Action Space)
            value: Value of current observations (Batch, 1)
        """
        card_inputs, location_inputs = torch.split(inputs, (6, 10), -1)
        card_embedding = self.card_embeddings(card_inputs)
        location_embedding = self.location_embeddings(location_inputs)
        embeddings = torch.cat((card_embedding, location_embedding), -2)
        latents = self.mlp(embeddings.flatten(-2))
        if return_value:
            return self.policy_head(latents), self.value_head(latents) * self.max_value
        return self.policy_head(latents)


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
