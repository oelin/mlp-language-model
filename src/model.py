import torch
import torch.nn as nn
import torch.nn.functional as F

from src.configuration import MLPLMConfiguration


class MLPLM(nn.Module):
    """MLP language model."""

    def __init__(self, configuration: MLPLMConfiguration) -> None:
        super().__init__()

        self.configuration = configuration

        self.embedding = nn.Embedding(
            num_embeddings=configuration.vocabulary_size,
            embedding_dim=configuration.embedding_dimension,
            padding_idx=configuration.padding_index,
        )

        self.mlp = nn.Sequential(
            nn.Linear(configuration.embedding_dimension * configuration.context_size, configuration.hidden_dimension),
            nn.ReLU(),
            nn.Dropout(p=configuration.dropout_probability),
            nn.LayerNorm(configuration.hidden_dimension),

            nn.Linear(configuration.hidden_dimension, configuration.hidden_dimension // 2),
            nn.ReLU(),
            nn.Dropout(p=configuration.dropout_probability),
            nn.LayerNorm(configuration.hidden_dimension // 2),

            nn.Linear(configuration.hidden_dimension // 2, configuration.vocabulary_size),
            nn.LogSoftmax(dim=-1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Computes the forward pass."""

        batch_size, context_size = x.size()

        assert context_size == self.configuration.context_size

        x = self.embedding(x)
        x = x.view(-1, context_size * self.configuration.embedding_dimension) # Concatenate.
        x = self.mlp(x)

        return x
