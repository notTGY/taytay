import torch

M = ['~', '$', ' ', '\n', '\'', ',']
for i in range(26):
    M.append(chr(ord('a') + i))
vocab_size = len(M)

def tokenize(str):
  tokens = []
  for char in str:
      index = M.index(char) if char in M else 1
      tokens.append(index)
  return tokens

def untokenize(ids):
  chars = []
  id_list = ids.tolist()
  for idx in id_list:
      char = M[idx]
      chars.append(char)
  return ''.join(chars)

def token_to_vec(token):
  out = torch.zeros(vocab_size, dtype=int)
  out[token] = 1
  return out

def tokens_to_vecs(tokensArrays):
  return torch.stack([torch.stack([token_to_vec(token) for token in tokens]) for tokens in tokensArrays]).float()