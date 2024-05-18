import torch
import time
import math
import random
from gen import gen

# https://www.youtube.com/watch?v=kCc8FmEb1nY
class TayTayDatasetNaive(torch.utils.data.Dataset):
  def __init__(self, filename, chunk_size, tokenizer, device, randomize=False):
    encode = tokenizer.encode
    self.vocab_size = tokenizer.vocab_size
    self.chunk_size = chunk_size

    def token_to_vec(token):
      out = torch.zeros(self.vocab_size, dtype=int)
      out[token] = 1
      return out

    def tokens_to_vecs(tokensArrays):
      return torch.stack([
        torch.stack([token_to_vec(token) for token in tokens]) for tokens in tokensArrays
      ]).float()

    with open(filename, 'r') as f:
      self.text = f.read()
      total_length = len(self.text)

      pieces = []
      if randomize:
        target_length = math.ceil(2 * total_length / chunk_size)
        for i in range(target_length):
          offset = random.randint(0, total_length - chunk_size - 1)
          x = encode(self.text[offset:offset + chunk_size * 6], return_tensors='pt', max_length=chunk_size, truncation=True).to(device)
          y = encode(self.text[offset:offset + (chunk_size+1) * 6], return_tensors='pt', max_length=chunk_size+1, truncation=True)[:, 1:].to(device)
          if len(x[0]) != chunk_size or len(y[0]) != chunk_size:
            continue
          pieces.append((x, tokens_to_vecs(y)))
      else:
        for offset in range(0, total_length - 1, chunk_size):
          x = encode(self.text[offset:offset + chunk_size * 6], return_tensors='pt', max_length=chunk_size, truncation=True).to(device)
          y = encode(self.text[offset:offset + (chunk_size+1) * 6], return_tensors='pt', max_length=chunk_size+1, truncation=True)[:, 1:].to(device)
          if len(x[0]) != chunk_size or len(y[0]) != chunk_size:
            continue
          pieces.append((x, tokens_to_vecs(y)))
      self.length = len(pieces)
      self.tokensArray = pieces

  def __len__(self):
    return self.length

  def __getitem__(self, idx):
    return self.tokensArray[idx]

def train(
  model,
  tokenizer,
  device,
  epochs = 3,
  checkpoint_name="taytay-autosave",
):

  chunk_size = 128
  vocab_size = tokenizer.vocab_size
  def collate_fn(batch):
    x = torch.stack([pair[0].reshape(-1) for pair in batch]).transpose(0, 1).to(device)
    y = torch.stack([pair[1].reshape(chunk_size, vocab_size) for pair in batch]).transpose(0, 1).to(device)
    return (x, y)
  min_v_loss = float('inf')
  all_v = []
  all_t = []

  optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

  loss = torch.nn.CrossEntropyLoss()
  print("Loading dataset...")
  dataset = TayTayDatasetNaive('songs.txt', chunk_size, tokenizer, device, randomize=True)
  print("Loaded {} entries".format(len(dataset)))
  train_size = int(0.9 * len(dataset))
  val_size = len(dataset) - train_size
  train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
  train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, collate_fn=collate_fn)
  val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=128, collate_fn=collate_fn)

  print("Starting training...")
  for epoch in range(epochs):
    start = time.time()
    model.train()
    tl = 0
    for batch in train_loader:
      (x, y) = batch
      outputs = model(x)
      l = loss(outputs, y)
      l.backward()
      optimizer.step()
      optimizer.zero_grad()
      tl += l.detach().cpu().numpy()

    model.eval()
    vl = 0
    for batch in val_loader:
      (x, y) = batch
      outputs = model(x)
      l = loss(outputs, y)
      vl += l.detach().cpu().numpy()

    avg_t_loss = tl / 9
    avg_v_loss = vl

    if avg_v_loss < min_v_loss:
      min_v_loss = avg_v_loss
      torch.save(model.state_dict(), checkpoint_name + '.pt')

    end = time.time()
    deltaT = end - start
    ETA = deltaT * (epochs - epoch - 1)

    (text, probability) = gen(model, 'a', tokenizer, device)

    print(
        '\ntrain loss: {}, val loss: {}, took {:.2f}s, ETA: {:.2f}s\nprediction {}: {}\n'
          .format(avg_t_loss, avg_v_loss, end - start, ETA, probability, text)
    )
  return all_t, all_v
