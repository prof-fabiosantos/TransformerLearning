
import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)

        # perform linear operation and split into N heads
        k = self.k_linear(k).view(bs, -1, self.n_heads, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.n_heads, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.n_heads, self.d_k)

        # transpose to get dimensions bs * N * sl * d_model
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        # calculate attention using scaled dot product
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        scores = torch.softmax(scores, dim=-1)
        output = torch.matmul(scores, v)

        # concatenate heads and put through final linear layer
        concat = output.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        output = self.out(concat)
        return output

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.dropout(torch.relu(self.linear1(x)))
        x = self.linear2(x)
        return x

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ff = PositionwiseFeedForward(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        x2 = self.norm1(x)
        x = x + self.dropout1(self.attn(x2, x2, x2, mask))
        x2 = self.norm2(x)
        x = x + self.dropout2(self.ff(x2))
        return x

class Encoder(nn.Module):
    """
    Transformer Encoder
    
    Args:
        vocab_size (int): size of the vocabulary
        d_model (int): dimension of the model
        n_layers (int): number of encoder layers
        n_heads (int): number of attention heads
        dropout (float): dropout rate
        
    Example:
        # a small example
        src_vocab_size = 1000
        d_model = 512
        n_layers = 6
        n_heads = 8
        dropout = 0.1

        # create an instance of the Encoder
        encoder = Encoder(src_vocab_size, d_model, n_layers, n_heads, dropout)

        # create a dummy input tensor
        src = torch.randint(0, src_vocab_size, (32, 10)) # batch size 32, sequence length 10

        # create a dummy mask
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)

        # forward pass
        encoded_output = encoder(src, src_mask)

        print(encoded_output.shape) # expected output: torch.Size([32, 10, 512])
    """
    def __init__(self, vocab_size, d_model, n_layers, n_heads, dropout=0.1):
        super(Encoder, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([EncoderLayer(d_model, n_heads, dropout) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, src, mask):
        x = self.embedding(src) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

if __name__ == "__main__":
    src_vocab_size = 1000
    d_model = 512
    n_layers = 6
    n_heads = 8
    dropout = 0.1

    # create an instance of the Encoder
    encoder = Encoder(src_vocab_size, d_model, n_layers, n_heads, dropout)

    # create a dummy input tensor
    src = torch.randint(0, src_vocab_size, (32, 10)) # batch size 32, sequence length 10

    # create a dummy mask
    src_mask = (src != 0).unsqueeze(1).unsqueeze(2)

    # forward pass
    encoded_output = encoder(src, src_mask)

    print(encoded_output.shape) # expected output: torch.Size([32, 10, 512])
