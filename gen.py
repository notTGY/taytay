import torch

def gen(model, text, tokenizer, device):
  probability = 1
  for i in range(10):
    test_tokens = tokenizer.encode(text, return_tensors='pt').to(device)
    model.eval()
    out = model(test_tokens)
    token = torch.argmax(out[0, -1, :])
    probability *= out[0, -1, token]
    text += tokenizer.decode(token)
  return (text, probability)
