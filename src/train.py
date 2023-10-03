"""MLP language model training script."""

from typing import Dict

import torch
from normality import ascii_text
from datasets import load_dataset
from tokenizers import SentencePieceBPETokenizer

from src.configuration import MLPLMConfiguration
from src.preprocessor import MLPLMPreprocessor
from src.model import MLPLM

dataset = load_dataset('roneneldan/TinyStories')


# Create a training dataset.

print('Creating training dataset (this may take up to 15 minutes)...')

def convert_to_ascii(example: Dict) -> Dict:
    """Converts an example to ASCII."""

    text = ascii_text(example['text'])

    return {'text': text}


train_dataset = dataset['train'].map(convert_to_ascii, num_proc=10)
validation_dataset = dataset['validation'].map(convert_to_ascii, num_proc=10)


# Train a tokenizer.

print('Training tokenizer (this may take up to 2 minutes)...')

texts = (example['text'] for example in train_dataset.shard(100, index=0))

tokenizer = SentencePieceBPETokenizer()
tokenizer.train_from_iterator(texts, vocab_size=10_000)
tokenizer.add_special_tokens(['<bos>', '<eos>', '<pad>'])

tokenizer.save('./tokenizer.json')


# Initialize the model.

print('Initializing model...')

configuration = MLPLMConfiguration(
    vocabulary_size=tokenizer.get_vocab_size(),
    context_size=64,
    padding_index=tokenizer.token_to_id('<pad>'),
    embedding_dimension=256,
    hidden_dimension=256,
    dropout_probability=0.2,
)

model = MLPLM(configuration).to('cuda')

print(f'Using configuration: {configuration}.') 


# Initialize the preprocessor.

print('Initializing preprocessor...')

preprocessor = MLPLMPreprocessor(configuration)


# Initialize the optimizer.

print('Initializing optimizer...'))

optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=0.01)

print(f'Using optimizer: {optimizer}')


number_of_batches = 30_000
number_of_batches_per_validation = 10
#number_of_processes = 10

criterion = nn.NLLLoss()

def train(model: MLPLM, process_index: int, shard) -> None:
    """Trains an MLP language model."""

    losses = []

    for i, example in enumerate(shard):

        optimizer.zero_grad()
        inputs, labels = preprocessor.preprocess(example['text'])
        logits = model(inputs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if (i + 1) > number_of_batches:
            break

        if (i + 1) % number_of_batches_per_validation == 0:

            mean_loss = torch.mean(torch.tensor(losses))
            losses.clear()

            print(f'(process {process_index:02d}) Batch {i+1}/{number_of_batches}, loss: {mean_loss}')

print(f'Training for 1 epoch on {number_of_batches} batches...')

train(model, 0, train_dataset.shuffle())
