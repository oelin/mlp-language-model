from nltk.util import ngrams


@dataclass
class MLPLMPreprocessor:
    """MLP language model preprocessor."""

    configuration: MLPLMConfiguration

    def preprocess(self, string: str) -> torch.Tensor:
        """Preprocesses a string."""

        # Convert string to padded sequence of tokens.

        padding = '<pad>' * self.configuration.context_size
        string = padding + f'<bos>{string}<eos>'
        tokens = tokenizer.encode(string, add_special_tokens=False).ids

        # Convert tokens to a batch of inputs.

        inputs = list(ngrams(tokens, n=self.configuration.context_size))
        inputs = inputs[: -1]  # Omit the final context.
        inputs = torch.tensor(inputs).to('cuda')

        # Get labels (all but the first `context_size` tokens).

        labels = tokens[self.configuration.context_size :]
        labels = torch.tensor(labels).to('cuda')

        return inputs, labels
