import torch
import torch.nn as nn

from tokenizer import tokenize, untokenize, vocab_size

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, context_length):
        super().__init__()

        base_exp = 10000
        position = torch.arange(context_length).unsqueeze(1)
        div_term = torch.arange(0, d_model, 2) / 2 + 1
        pe = torch.zeros(context_length, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position / (base_exp ** (2*div_term/d_model)) )
        pe[:, 0, 1::2] = torch.cos(position / (base_exp ** (2*div_term/d_model)) )
        self.register_buffer('pe', pe)

    @torch.no_grad()
    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return x

class MaskedSelfAttention(nn.Module):

  def __init__(self, key_dim, embedding_dim, value_dim, dropout):
    super().__init__()
    self.key_dim = key_dim
    query_dim = key_dim
    self.K = nn.Linear(embedding_dim, key_dim, bias=False)
    self.Q = nn.Linear(embedding_dim, query_dim, bias=False)
    self.V = nn.Linear(embedding_dim, value_dim, bias=False)
    self.softmax = nn.Softmax(dim=2)
    self.dropout = nn.Dropout(dropout)


  def forward(self, x):
    """
    Arguments:
        x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
    """
    key = self.K(x)
    query = self.Q(x)
    value = self.V(x)

    scores = torch.matmul(query, key.transpose(1, 2)) * self.key_dim ** -0.5

    scores = self.softmax(scores.masked_fill(torch.tril(scores) == 0, float('-inf')))
    x = torch.matmul(scores, value)
    return self.dropout(x)

class FFN(nn.Module):

    def __init__(self, emebedding_dim, dim, n_layers, dropout):
        super().__init__()
        self.n_hidden_layers = n_layers
        self.input_fc = nn.Linear(emebedding_dim, dim)
        self.hidden_fc = nn.ModuleList([nn.Linear(dim, dim) for i in range(n_layers)])
        self.relu = nn.ReLU()
        self.proj = nn.Linear(dim, emebedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = self.input_fc(x)
        x = self.relu(x)
        for i in range(self.n_hidden_layers):
            x = self.hidden_fc[i](x)
            x = self.relu(x)
        x = self.proj(x)
        return self.dropout(x)

class MultiHeadAttention(nn.Module):

    def __init__(self, key_dim, embedding_dim, n_heads, dropout):
        super().__init__()
        value_dim = embedding_dim // n_heads
        self.msas = nn.ModuleList([MaskedSelfAttention(key_dim, embedding_dim, value_dim, dropout) for i in range(n_heads)])
        self.proj = nn.Linear(embedding_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = torch.cat([self.msas[i](x) for i in range(len(self.msas))], dim=-1)
        x = self.proj(x)
        return self.dropout(x)

class DecoderBlock(nn.Module):

    def __init__(self, embedding_dim, key_dim, n_heads, hidden_dim, ffn_layers, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(embedding_dim)
        self.msa = MultiHeadAttention(key_dim, embedding_dim, n_heads, dropout)

        self.ln2 = nn.LayerNorm(embedding_dim)
        self.ffn = FFN(embedding_dim, hidden_dim, ffn_layers, dropout)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.msa(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class TayTay(nn.Module):

    def __init__(
        self,
        embedding_dim=2,
        key_dim=2,
        n_heads=1,
        scaling_factor=1,
        ffn_layers=0,
        decoder_layers=1,
        dropout=0.2,
        context_length=128,
      ):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embedding_dim)
        self.positional_encoding = PositionalEncoding(embedding_dim, context_length)

        hidden_dim = embedding_dim * scaling_factor
        self.decoder = nn.ModuleList([
            DecoderBlock(embedding_dim, key_dim, n_heads, embedding_dim, ffn_layers, dropout)
            for i in range(decoder_layers)
        ])

        self.ln = nn.LayerNorm(embedding_dim)

        self.lm_head = nn.Linear(embedding_dim, vocab_size)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = self.emb(x)
        x = self.positional_encoding(x)
        for i in range(len(self.decoder)):
          x = self.decoder[i](x)
        x = self.lm_head(self.ln(x))
        x = self.softmax(x)
        return x


model = TayTay(embedding_dim=72, n_heads=8, scaling_factor=4, ffn_layers=1, decoder_layers=4)
def gen(model, input_text, new_tokens=10, device='cpu'):
  for i in range(new_tokens):
    tokens = tokenize(input_text)
    input_ids = torch.tensor([tokens]).transpose(0, 1).to(device)
    preds = torch.argmax(model.forward(input_ids), dim=2).transpose(0, 1)
    new_token = untokenize(torch.tensor([preds[0][-1]]))
    input_text = input_text + '' + new_token
  return input_text

def process_batch(model, batch, optimizer, loss):
  (x, y) = batch
  output = model(x)
  l = loss(output, y)
  l.backward()
  optimizer.step()
  optimizer.zero_grad(set_to_none=True)
  return l
def eval_batch(model, batch, loss):
  (x, y) = batch
  output = model(x)
  l = loss(output, y)
  return l