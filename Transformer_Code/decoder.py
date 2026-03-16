
import torch
import torch.nn as nn
import math
from encoder import MultiHeadAttention, PositionwiseFeedForward, PositionalEncoding

class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        
        self.attn1 = MultiHeadAttention(d_model, n_heads)
        self.attn2 = MultiHeadAttention(d_model, n_heads)
        self.ff = PositionwiseFeedForward(d_model)

    def forward(self, x, e_outputs, src_mask, trg_mask):
        x2 = self.norm1(x)
        x = x + self.dropout1(self.attn1(x2, x2, x2, trg_mask))
        x2 = self.norm2(x)
        x = x + self.dropout2(self.attn2(x2, e_outputs, e_outputs, src_mask))
        x2 = self.norm3(x)
        x = x + self.dropout3(self.ff(x2))
        return x

class Decoder(nn.Module):
    """
    Transformer Decoder
    
    Args:
        vocab_size (int): size of the vocabulary
        d_model (int): dimension of the model
        n_layers (int): number of decoder layers
        n_heads (int): number of attention heads
        dropout (float): dropout rate
        
    Example:
        # a small example
        trg_vocab_size = 1200
        d_model = 512
        n_layers = 6
        n_heads = 8
        dropout = 0.1

        # create an instance of the Decoder
        decoder = Decoder(trg_vocab_size, d_model, n_layers, n_heads, dropout)

        # create dummy input tensors
        # e_outputs would come from the Encoder
        e_outputs = torch.randn(32, 10, d_model) # batch size 32, sequence length 10
        trg = torch.randint(0, trg_vocab_size, (32, 12)) # batch size 32, sequence length 12

        # create dummy masks
        src_mask = torch.ones(32, 1, 10).byte() # dummy source mask
        trg_mask = (trg != 0).unsqueeze(1).unsqueeze(2) # dummy target mask

        # forward pass
        decoded_output = decoder(trg, e_outputs, src_mask, trg_mask)

        print(decoded_output.shape) # expected output: torch.Size([32, 12, 512])
    """
    def __init__(self, vocab_size, d_model, n_layers, n_heads, dropout=0.1):
        super(Decoder, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([DecoderLayer(d_model, n_heads, dropout) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, trg, e_outputs, src_mask, trg_mask):
        x = self.embedding(trg) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        for layer in self.layers:
            x = layer(x, e_outputs, src_mask, trg_mask)
        return self.norm(x)

if __name__ == "__main__":
    # a small example
    trg_vocab_size = 1200
    d_model = 512
    n_layers = 6
    n_heads = 8
    dropout = 0.1

    # create an instance of the Decoder
    decoder = Decoder(trg_vocab_size, d_model, n_layers, n_heads, dropout)

    # create dummy input tensors
    # e_outputs would come from the Encoder
    e_outputs = torch.randn(32, 10, d_model) # batch size 32, sequence length 10
    trg = torch.randint(0, trg_vocab_size, (32, 12)) # batch size 32, sequence length 12

    # create dummy masks
    src_mask = torch.ones(32, 1, 1, 10).bool() # dummy source mask

    trg_pad_mask = (trg != 0).unsqueeze(1).unsqueeze(2) # (32, 1, 1, 12)
    trg_len = trg.shape[1]
    trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), dtype=torch.bool)) # (12, 12)
    trg_mask = trg_pad_mask & trg_sub_mask # (32, 1, 12, 12)

    # forward pass
    decoded_output = decoder(trg, e_outputs, src_mask, trg_mask)

    print(decoded_output.shape) # expected output: torch.Size([32, 12, 512])
