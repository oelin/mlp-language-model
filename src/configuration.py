from dataclasses import dataclass


@dataclass
class MLPLMConfiguration:

    vocabulary_size: int
    context_size: int
    padding_index: int
    embedding_dimension: int
    hidden_dimension: int
    dropout_probability: float = 0.5
