from transformers import GemmaModel, GemmaConfig
import torch

from tokenizer import tokenize, untokenize, vocab_size


conf = {
  "vocab_size": vocab_size,
  "hidden_size": 32, #embedding size????
  "intermediate_size": 32,
  "num_hidden_layers": 3,
  "num_attention_heads": 4,
  "num_key_value_heads": 4,
  "head_dim": 32,
  "hidden_activation": "relu",
  "max_position_embeddings": 128,
  "eos_token_id": 0,
  "pad_token_id": 0,
  "bos_token_id": 0,
  "attention_dropout": 0.2,
}

model = GemmaModel(GemmaConfig(**conf))

def gen(model, input_text, new_tokens=10, device='cpu'):
  for i in range(new_tokens):
    tokens = tokenize(input_text)
    input_ids = torch.tensor([tokens]).transpose(0, 1).to(device)
    preds = torch.argmax(model.forward(input_ids).last_hidden_state, dim=2).transpose(0, 1)
    new_token = untokenize(torch.tensor([preds[0][-1]]))
    input_text = input_text + '' + new_token
  return input_text

def process_batch(model, batch, optimizer, loss):
  (x, y) = batch
  output = model(x)
  l = loss(output.last_hidden_state, y)
  l.backward()
  optimizer.step()
  optimizer.zero_grad(set_to_none=True)
  return l
def eval_batch(model, batch, loss):
  (x, y) = batch
  output = model(x)
  l = loss(output.last_hidden_state, y)
  return l