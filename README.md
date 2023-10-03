# MLP Language Model

A 2M parameter neural language model trained on the TinyStories corpus.


## Completions

Prompt: `<bos>`

> Once upon a time there was a little girl named Lily. She was very happy and she had a lot of fun with her friends. One day, Lily went to visit the kitchen. She was so happy that she didn't know what to do.

Prompt: `<bos>`

> Once there was a little boy named Timmy. Timmy loved to play outside with his friends. One day, he went to the park to play. Jack was playing in the park with his friends. He saw a big red car and his toy car. Timmy was happy to have the toy car.

Prompt: `<bos>When Alice saw Eve she said`

> When Alice saw Eve she said it was time to go home. She said goodbye to her mom and dad.

Prompt: `<bos>One day, the sun`

> One day, the sun was in the sky. It was a big, beautiful sky. The sun had many colors of the animals in the garden. The animals was so happy and thanked the rabbit for his own new place.


## Architecture

```
MLPLM(
  (embedding): Embedding(10003, 256, padding_idx=10002)
  (mlp): Sequential(
    (0): Linear(in_features=16384, out_features=128, bias=True)
    (1): ReLU()
    (2): Dropout(p=0.05, inplace=False)
    (3): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
    (4): Linear(in_features=128, out_features=64, bias=True)
    (5): ReLU()
    (6): Dropout(p=0.05, inplace=False)
    (7): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
    (8): Linear(in_features=64, out_features=10003, bias=True)
    (9): LogSoftmax(dim=-1)
  )
)
```


## Performance

MLPLM-V3.

| Batch # | NLL Loss (Train) | NLL Loss (Validation) |
|---------|------------------|-----------------------|
| 0       | 8.94             | 8.93                  |
| 1000    | 5.71             | 5.66                  |
| 10000   | 4.29             | 4.28                  |
| 20000   | 3.91             | 3.89                  |
| 30000   | 3.88             | 3.86                  |
