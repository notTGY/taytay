import torch
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
#print('device', device)

from tokenizers import Tokenizer
from transformers import GPT2TokenizerFast
tokenizer = GPT2TokenizerFast(
  tokenizer_object=Tokenizer.from_file("tokenizer.json")
)
#print(tokenizer.decode(tokenizer.encode("i love you baby.")))

from taytay import TayTay
model = TayTay(
  vocab_size=tokenizer.vocab_size,
  embedding_dim=128,
  n_heads=16,
  scaling_factor=4,
  ffn_layers=1,
  decoder_layers=12
)
model.load_state_dict(torch.load("taytay-hobbs.pt"))
model.to(device)
print("Loaded model with {} params".format(model.count_params()))

#from gen import gen
#print(gen(model, "a", tokenizer, device))
from train import train
train(model, tokenizer, device, epochs=100)

