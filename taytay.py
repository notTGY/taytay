import torch
import torch.nn as nn

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
        pe = self.pe[:x.size(0)].reshape(x.shape[0], x.shape[2])
        x = torch.einsum('ijk,ik->ijk', x, pe)
        #print(x.shape)
        #x = x + pe
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
        vocab_size=128,
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

    def count_params(self):
      n_params = 0
      for name, param in self.named_parameters():
          a = 1
          for i in param.shape:
              a *= i
          n_params += a
      return n_params

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size]``
        """
        #print("input", x.shape)
        x = self.emb(x)
        #print("emb", x.shape)
        x = self.positional_encoding(x)
        for i in range(len(self.decoder)):
          x = self.decoder[i](x)
        x = self.lm_head(self.ln(x))
        #print("lm head", x.shape)
        x = self.softmax(x)
        #print("softmax", x.shape)
        return x


