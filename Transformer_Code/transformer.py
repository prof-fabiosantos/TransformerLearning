
import torch
import torch.nn as nn
from encoder import Encoder
from decoder import Decoder

class Transformer(nn.Module):
    def __init__(self, src_vocab, trg_vocab, d_model, n_layers, n_heads, dropout=0.1):
        super(Transformer, self).__init__()
        self.encoder = Encoder(src_vocab, d_model, n_layers, n_heads, dropout)
        self.decoder = Decoder(trg_vocab, d_model, n_layers, n_heads, dropout)
        self.out = nn.Linear(d_model, trg_vocab)

    def forward(self, src, trg, src_mask, trg_mask):
        e_outputs = self.encoder(src, src_mask)
        d_output = self.decoder(trg, e_outputs, src_mask, trg_mask)
        output = self.out(d_output)
        return output

if __name__ == "__main__":
    src_vocab = 1000
    trg_vocab = 1200
    d_model = 512
    n_layers = 6
    n_heads = 8
    dropout = 0.1

    # create an instance of the Transformer
    model = Transformer(src_vocab, trg_vocab, d_model, n_layers, n_heads, dropout)

    # create dummy input tensors
    src = torch.randint(0, src_vocab, (32, 10)) # batch size 32, sequence length 10
    trg = torch.randint(0, trg_vocab, (32, 12)) # batch size 32, sequence length 12

    # create dummy masks
    src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
    trg_mask = (trg != 0).unsqueeze(1).unsqueeze(2)

    # forward pass
    output = model(src, trg, src_mask, trg_mask)

    print(output.shape) # expected output: torch.Size([32, 12, 1200])
