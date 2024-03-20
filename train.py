import torch
import time
#import matplotlib.pyplot as plt
#from IPython.display import clear_output

def train(
  model,
  process_batch,
  eval_batch,
  gen,
  train_loader,
  val_loader,
  optimizer,
  loss,
  epochs,
  checkpoint_name,
  device='cpu',
  plot=False,
):
  min_v_loss = float('inf')
  all_v = []
  all_t = []

  for epoch in range(epochs):
    start = time.time()
    model.train()
    #train_loss = 0
    tl = 0
    for batch in train_loader:
      tl = process_batch(model, batch, optimizer, loss)

    #val_loss = 0
    model.eval()
    vl = 0
    for batch in val_loader:
      vl = eval_batch(model, batch, loss)
      #val_loss += l

    avg_t_loss = tl.detach().cpu().numpy()
    avg_v_loss = vl.detach().cpu().numpy()
    if plot:
      all_v.append(avg_v_loss)
      all_t.append(avg_t_loss)

    if avg_v_loss < min_v_loss:
      min_v_loss = avg_v_loss
      torch.save(model.state_dict(), checkpoint_name + '.pt')

    end = time.time()
    deltaT = end - start
    ETA = deltaT * (epochs - epoch - 1)
    if plot:
      clear_output()
      plt.plot([i for i in range(len(all_t))], all_t, label='train')
      plt.plot([i for i in range(len(all_v))], all_v, label='val')
      plt.legend()
      plt.show()
    print(
        '\ntrain loss: {}, val loss: {}, took {:.2f}s, ETA: {:.2f}s,\ntest generation: {}'
          .format(avg_t_loss, avg_v_loss, end - start, ETA, gen(model, 'like any r', 20, device))
    )
  return all_t, all_v
