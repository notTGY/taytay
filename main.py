import torch
import time
#import matplotlib.pyplot as plt
#from IPython.display import clear_output

from taytay import model, gen, process_batch, eval_batch
#from gemma import model, gen, process_batch, eval_batch
from tokenizer import untokenize
from dataset import TayTayDatasetNaive, collate_fn
from train import train

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Using: {}'.format(device))

print('Loading model...')

model = model.to(device)
model.load_state_dict(torch.load('taytay-autosave.pt'))

# see all parameters
n_params = 0
for name, param in model.named_parameters():
    a = 1
    for i in param.shape:
        a *= i
    n_params += a
    #print(name, param.shape, a)
    #print(param.tolist())

print('Parameters: ', n_params)

print(gen(model, 'i love', 128 - 6, device))
exit(0)

print('Loading dataset...')
dataset = TayTayDatasetNaive('songs.txt', chunk_size=128, device=device, randomize=True)
print(
    'Data length: ',
    len(dataset),
    'Example: ',
    untokenize(dataset[0][0][0])
)

train_split, val_split = torch.utils.data.random_split(dataset, [0.9, 0.1])
train_loader = torch.utils.data.DataLoader(train_split, batch_size=128, collate_fn=collate_fn, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_split, batch_size=128, collate_fn=collate_fn, shuffle=True)
print('Training set has {} instances'.format(len(train_split)))
print('Validation set has {} instances'.format(len(val_split)))

loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

all_t, all_v = train(
  model,
  process_batch,
  eval_batch,
  gen,
  train_loader,
  val_loader,
  optimizer,
  loss,
  1000,
  'taytay-autosave',
  device,
)

