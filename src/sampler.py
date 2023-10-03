from dataclasses import dataclass

import torch
import torch.nn.functional as F

from tokenizers import Tokenizer

from src.configuration import MLPLMConfiguration
from src.preprocessor import MLPLMPreprocessor


@dataclass
class MLPLMSampler:
    """Samples from an MLP language model."""
  
    configuration: MLPLMConfiguration
    preprocessor: MLPLMPreprocessor
    tokenizer: Tokenizer

    def sample_top_k(self, context: str, k: int = 5, sample_size: int = 10) -> torch.Tensor:
        """Sample from the model using top-k."""

        for _ in range(sample_size):

          inputs, labels = self.preprocessor.preprocess(context)
          logits = model(inputs).detach().cpu()[-1]

          top_k_logits = torch.topk(logits, k=k)
          top_k_distribution = F.softmax(top_k_logits.values, dim=-1)

          index = torch.multinomial(top_k_distribution, num_samples=1, replacement=True)
          token_index = top_k_logits.indices[index].item()
          token = self.tokenizer.id_to_token(token_index)

          context += token
        
        return context.replace('▁', ' ').strip()
