import torch
import math
import random
from tokenizer import tokenize, untokenize, vocab_size, tokens_to_vecs

# https://www.youtube.com/watch?v=kCc8FmEb1nY
class TayTayDatasetNaive(torch.utils.data.Dataset):
  def __init__(self, filename, chunk_size, device='cpu', randomize=False):
    with open(filename, 'r') as f:
      self.text = f.read()
      total_length = len(self.text)

      pieces = []
      if randomize:
        target_length = math.ceil(2 * total_length / chunk_size)
        self.length = target_length
        for i in range(target_length):
          offset = random.randint(0, total_length - chunk_size - 1)
          x = torch.tensor([tokenize(self.text[offset:offset + chunk_size])]).to(device)
          y = tokens_to_vecs([tokenize(self.text[offset + 1:offset + chunk_size + 1])]).to(device)
          pieces.append((x, y))
      else:
        self.length = total_length // chunk_size - 1
        for offset in range(0, total_length, chunk_size):
          x = torch.tensor([tokenize(self.text[offset:offset + chunk_size])])
          y = tokens_to_vecs([tokenize(self.text[offset + 1:offset + chunk_size + 1])])
          pieces.append((x, y))

      self.tokensArray = pieces

  def __len__(self):
    return self.length

  def __getitem__(self, idx):
    return self.tokensArray[idx]

def collate_fn(batch):
  x = torch.stack([x[0][0] for x in batch])
  y = torch.stack([y[1][0] for y in batch])
  return (x, y)