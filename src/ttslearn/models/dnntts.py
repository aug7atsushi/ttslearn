from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from ttslearn.models.basemodel import BaseModel


class SimpleFNN(BaseModel):
    def __init__(
        self, in_dim: int, hidden_dim: int, out_dim: int, num_layers: int = 2
    ) -> None:
        super(SimpleFNN, self).__init__()
        model = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_layers):
            model.append(nn.Linear(hidden_dim, hidden_dim))
            model.append(nn.ReLU())
        model.append(nn.Linear(hidden_dim, out_dim))
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class LSTMRNN(BaseModel):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers=1,
        bidirectional=True,
        dropout=0.0,
    ) -> None:
        super(LSTMRNN, self).__init__()
        self.num_layers = num_layers
        num_direction = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            in_dim,
            hidden_dim,
            num_layers,
            bidirectional=bidirectional,
            batch_first=True,
            dropout=dropout,
        )
        self.hidden2out = nn.Linear(num_direction * hidden_dim, out_dim)

    def forward(self, seqs, lens):
        seqs = pack_padded_sequence(seqs, lens, batch_first=True)
        out, _ = self.lstm(seqs)
        out, _ = pad_packed_sequence(out, batch_first=True)
        out = self.hidden2out(out)
        return out
